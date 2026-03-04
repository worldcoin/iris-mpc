#!/bin/bash
set -euo pipefail

# Run each genesis e2e test in sequence and assert the db-sanity-check binary
# passes for all 3 parties after each one.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SEED=42
PARTIES=(0 1 2)
TESTS=(100 101 102 103 104 105 106)

# Override the Docker-internal LocalStack hostname for host-side runs.
export AWS_ENDPOINT_URL=http://localhost:4566

log()  { echo "$(date +%Y-%m-%dT%H:%M:%S) [INFO] $*"; }
fail() { echo "$(date +%Y-%m-%dT%H:%M:%S) [FAIL] $*" >&2; exit 1; }

cd "$REPO_ROOT"

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
            --output-dir "$output_dir" \
            > "$log_file" 2>&1
        then
            grep -E "^\[FAIL\]|^===" "$log_file" || true
            fail "Sanity check failed: genesis_${test_id} party ${party_id} (see $log_file)"
        fi

        grep "^=== Summary" "$log_file"
    done

    log "Genesis ${test_id}: all parties passed"
done

log "All genesis e2e tests passed sanity checks."
