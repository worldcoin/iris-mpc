#!/bin/bash
set -euo pipefail

# Run each genesis e2e test in sequence and assert the db-sanity-check binary
# passes for all 3 parties after each one.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SEED=42
PARTIES=(0 1 2)
TESTS=(100 101 102 103 104 105 106)
EXCLUSIONS_S3_URI="s3://wf-smpcv2-dev-sync-protocol/dev_deleted_serial_ids.json"

# Override the Docker-internal LocalStack hostname for host-side runs.
export AWS_ENDPOINT_URL=http://localhost:4566
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_REGION=us-east-1

log()  { echo "$(date +%Y-%m-%dT%H:%M:%S) [INFO] $*"; }
fail() { echo "$(date +%Y-%m-%dT%H:%M:%S) [FAIL] $*" >&2; exit 1; }

cd "$REPO_ROOT"

log "Building genesis e2e tests..."
(cd "$REPO_ROOT/iris-mpc-upgrade-hawk" && \
    cargo test --release --test e2e_genesis --no-run 2>&1 | grep -E "^error|Compiling|Finished")

log "Building db-sanity-check..."
cargo build --release -p iris-mpc-bins --bin db-sanity-check 2>&1 | grep -E "^error|Finished"
SANITY_BIN="$REPO_ROOT/target/release/db-sanity-check"

for test_id in "${TESTS[@]}"; do
    log "=== Genesis e2e test $test_id ==="

    (
        cd "$REPO_ROOT/iris-mpc-upgrade-hawk"
        cargo test --release --test e2e_genesis "test_hnsw_genesis_${test_id}" \
            -- --include-ignored 2>&1 | grep -E "test.*ok|test.*FAILED|^error"
    )

    # Allow DB persistence pipelines to flush before checking.
    sleep 5

    for party_id in "${PARTIES[@]}"; do
        log "Sanity check: genesis_${test_id} party ${party_id}"
        output_dir="$REPO_ROOT/sanity-check/genesis_${test_id}/party${party_id}"
        mkdir -p "$output_dir"
        log_file="$output_dir/run.log"

        if ! "$SANITY_BIN" \
            --hnsw-db-url "postgres://postgres:postgres@localhost:5432/SMPC_dev_${party_id}" \
            --gpu-db-url "postgres://postgres:postgres@localhost:5432/SMPC_dev_${party_id}" \
            --hnsw-schema "SMPC_hnsw_dev_${party_id}" \
            --gpu-schema "SMPC_dev_${party_id}" \
            --seed "$SEED" \
            --exclusions-s3-uri "$EXCLUSIONS_S3_URI" \
            --checkpoint-s3-bucket "wf-smpcv2-dev-hnsw-checkpoint" \
            --force-path-style \
            --output-dir "$output_dir" \
            > "$log_file" 2>&1
        then
            grep -E "^\[FAIL\]|^===" "$log_file" || true
            fail "Sanity check failed: genesis_${test_id} party ${party_id} (see $log_file)"
        fi

        grep "^=== Summary" "$log_file"
    done

    # Cross-party checkpoint consistency: blake3 hashes and metadata must match.
    base_dir="$REPO_ROOT/sanity-check/genesis_${test_id}"
    hashes=()
    iris_ids=()
    mod_ids=()
    for party_id in "${PARTIES[@]}"; do
        stats_file="$base_dir/party${party_id}/stats.json"
        h=$(python3 -c "import json; d=json.load(open('$stats_file')); print(d.get('checkpoint_blake3_hash',''))")
        i=$(python3 -c "import json; d=json.load(open('$stats_file')); print(d.get('checkpoint_last_indexed_iris_id',''))")
        m=$(python3 -c "import json; d=json.load(open('$stats_file')); print(d.get('checkpoint_last_indexed_modification_id',''))")
        hashes+=("$h")
        iris_ids+=("$i")
        mod_ids+=("$m")
        log "  Party ${party_id} checkpoint: blake3=${h} iris_id=${i} mod_id=${m}"
    done
    if [[ "${hashes[0]}" != "" ]]; then
        if [[ "${hashes[0]}" == "${hashes[1]}" && "${hashes[1]}" == "${hashes[2]}" ]]; then
            log "  Cross-party checkpoint blake3 hashes match: ${hashes[0]}"
        else
            fail "Cross-party checkpoint blake3 mismatch: ${hashes[*]}"
        fi
        if [[ "${iris_ids[0]}" == "${iris_ids[1]}" && "${iris_ids[1]}" == "${iris_ids[2]}" ]]; then
            log "  Cross-party checkpoint last_indexed_iris_id match: ${iris_ids[0]}"
        else
            fail "Cross-party checkpoint last_indexed_iris_id mismatch: ${iris_ids[*]}"
        fi
        if [[ "${mod_ids[0]}" == "${mod_ids[1]}" && "${mod_ids[1]}" == "${mod_ids[2]}" ]]; then
            log "  Cross-party checkpoint last_indexed_modification_id match: ${mod_ids[0]}"
        else
            fail "Cross-party checkpoint last_indexed_modification_id mismatch: ${mod_ids[*]}"
        fi
    fi

    log "Genesis ${test_id}: all parties passed"
done

log "All genesis e2e tests passed sanity checks."
