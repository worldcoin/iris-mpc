#!/usr/bin/env bash
set -euo pipefail

# Node-side lifecycle helper for the distributed real-server benchmark. The
# controller copies the same committed tree and TLS bundle to all three hosts,
# then invokes this script over SSH/SSM with the environment documented below.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
RUN_DIR=${LINEAR_SCAN_BENCH_RUN_DIR:-/var/tmp/iris-mpc-real-server-bench}
POSTGRES_PORT=${LINEAR_SCAN_BENCH_POSTGRES_PORT:-55432}
DATABASE_SIZE=${LINEAR_SCAN_BENCH_DATABASE_SIZE:-1048576}
REQUEST_COUNT=${LINEAR_SCAN_BENCH_REQUEST_COUNT:-6}
REQUEST_PARALLELISM=${LINEAR_SCAN_BENCH_REQUEST_PARALLELISM:-48}
CONNECTION_PARALLELISM=${LINEAR_SCAN_BENCH_CONNECTION_PARALLELISM:-16}
TOKIO_CORES=${LINEAR_SCAN_BENCH_TOKIO_CORES:-11}
CLIENT_RNG_SEED=${LINEAR_SCAN_BENCH_CLIENT_RNG_SEED:-8675309}
PIPELINED_REQUESTS=${LINEAR_SCAN_BENCH_PIPELINED_REQUESTS:-0}
NTT_SCAN=${LINEAR_SCAN_BENCH_NTT:-0}
[[ ${NTT_SCAN} =~ ^[01]$ ]] || {
    echo "LINEAR_SCAN_BENCH_NTT must be 0 or 1" >&2
    exit 2
}
AUX_CPU_LIST="0-$((TOKIO_CORES - 1))"

usage() {
    echo "usage: $0 <prepare-db|start-server|stop-server|status> [party-id]" >&2
    echo "       $0 <start-moto|stop-moto|init-moto|rotate-keys|run-client>" >&2
    exit 2
}

start_moto() {
    : "${LINEAR_SCAN_BENCH_AWS_ENDPOINT:?set the externally reachable Moto endpoint}"
    local pid_file="${RUN_DIR}/moto.pid"
    local python=${LINEAR_SCAN_BENCH_MOTO_PYTHON:-${RUN_DIR}/moto-venv/bin/python}
    mkdir -p "$RUN_DIR"
    if [[ -f ${pid_file} ]] && pid_running "$(<"$pid_file")"; then
        echo "Moto is already running: pid=$(<"$pid_file")" >&2
        exit 1
    fi
    if [[ ! -x ${python} ]]; then
        python3 -m venv "${RUN_DIR}/moto-venv"
        python="${RUN_DIR}/moto-venv/bin/python"
    fi
    if ! "$python" -c 'import boto3, moto' >/dev/null 2>&1; then
        "$python" -m pip install -q 'moto[server]==5.1.22' boto3
    fi
    nohup taskset -c "$AUX_CPU_LIST" env \
        MOTO_ACCOUNT_ID=000000000000 S3_IGNORE_SUBDOMAIN_BUCKETNAME=true \
        "$python" "${PROJECT_ROOT}/scripts/moto-server-with-sns-sequence.py" \
        -H 0.0.0.0 -p 4566 >"${RUN_DIR}/moto.log" 2>&1 &
    echo "$!" >"$pid_file"
    for _ in $(seq 1 120); do
        curl -fsS "http://127.0.0.1:4566/moto-api/" >/dev/null 2>&1 && {
            echo "NODE_BENCH_MOTO_STARTED pid=$! endpoint=${LINEAR_SCAN_BENCH_AWS_ENDPOINT}"
            return
        }
        pid_running "$!" || {
            tail -100 "${RUN_DIR}/moto.log" >&2
            exit 1
        }
        sleep 1
    done
    echo "Moto did not become ready" >&2
    exit 1
}

stop_moto() {
    local pid_file="${RUN_DIR}/moto.pid"
    if [[ -f ${pid_file} ]]; then
        local pid
        pid=$(<"$pid_file")
        pid_running "$pid" && kill "$pid"
        rm -f "$pid_file"
    fi
    echo "NODE_BENCH_MOTO_STOPPED"
}

init_moto() {
    : "${LINEAR_SCAN_BENCH_AWS_ENDPOINT:?set the externally reachable Moto endpoint}"
    local python=${LINEAR_SCAN_BENCH_MOTO_PYTHON:-${RUN_DIR}/moto-venv/bin/python}
    [[ -x ${python} ]] || {
        echo "missing Moto Python environment; run start-moto first" >&2
        exit 1
    }
    "$python" "${PROJECT_ROOT}/scripts/init-moto-linear-scan-benchmark.py" \
        --endpoint "$LINEAR_SCAN_BENCH_AWS_ENDPOINT"
}

rotate_keys() {
    : "${LINEAR_SCAN_BENCH_AWS_ENDPOINT:?set the externally reachable Moto endpoint}"
    local binary=${LINEAR_SCAN_BENCH_KEY_MANAGER_BINARY:-${PROJECT_ROOT}/target/release/key-manager}
    [[ -x ${binary} ]] || {
        echo "missing key-manager binary: ${binary}" >&2
        exit 1
    }
    for party in 0 1 2; do
        for _ in 1 2; do
            AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_REGION=us-east-1 \
                AWS_DEFAULT_REGION=us-east-1 AWS_ENDPOINT_URL="$LINEAR_SCAN_BENCH_AWS_ENDPOINT" \
                "$binary" --region us-east-1 \
                --endpoint-url "$LINEAR_SCAN_BENCH_AWS_ENDPOINT" \
                --node-id "$party" --env dev rotate \
                --public-key-bucket-name wf-dev-public-keys
        done
    done
    echo "NODE_BENCH_KEYS_READY"
}

run_client() {
    : "${LINEAR_SCAN_BENCH_AWS_ENDPOINT:?set the externally reachable Moto endpoint}"
    local binary=${LINEAR_SCAN_BENCH_CLIENT_BINARY:-${PROJECT_ROOT}/target/release/service-client}
    local config="${RUN_DIR}/client.toml"
    local aws_config="${RUN_DIR}/aws.toml"
    local output="${RUN_DIR}/results.json"
    [[ -x ${binary} ]] || {
        echo "missing service-client binary: ${binary}" >&2
        exit 1
    }
    mkdir -p "$RUN_DIR"
    local batch_count=$REQUEST_COUNT
    local batch_size=1
    if [[ ${PIPELINED_REQUESTS} == 1 ]]; then
        # Publish independent requests together so the production server has a
        # sustained queue. SMPC__MAX_BATCH_SIZE=1 still makes the server scan
        # them serially; this only removes client-side S3/response idle gaps.
        batch_count=1
        batch_size=$REQUEST_COUNT
    fi
    cat >"$config" <<EOF
results_output_path = "$output"
record_timings = true
cleanup_on_exit = false

[request_batch.Simple]
batch_count = $batch_count
batch_size = $batch_size
batch_kind = "uniqueness"

[shares_generator.FromCompute]
rng_seed = $CLIENT_RNG_SEED
EOF
    cat >"$aws_config" <<EOF
environment = "dev"
public_key_base_url = "$LINEAR_SCAN_BENCH_AWS_ENDPOINT/wf-dev-public-keys"
s3_request_bucket_name = "wf-smpcv2-dev-sns-requests"
sns_request_topic_arn = "arn:aws:sns:us-east-1:000000000000:iris-mpc-input.fifo"
sqs_long_poll_wait_time = 2
sqs_response_queue_urls = ["$LINEAR_SCAN_BENCH_AWS_ENDPOINT/000000000000/iris-mpc-results-us-east-1.fifo"]
sqs_wait_time_seconds = 1
EOF
    taskset -c "$AUX_CPU_LIST" env \
        AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_REGION=us-east-1 \
        AWS_DEFAULT_REGION=us-east-1 AWS_ENDPOINT_URL="$LINEAR_SCAN_BENCH_AWS_ENDPOINT" \
        "$binary" --path-to-opts "$config" --path-to-opts-aws "$aws_config" \
        >"${RUN_DIR}/client.log" 2>&1
    python3 - "$output" "$REQUEST_COUNT" <<'PY'
import json
import sys

path, expected = sys.argv[1], int(sys.argv[2])
with open(path, encoding="utf-8") as result_file:
    records = json.load(result_file).get("records", [])
if len(records) != expected:
    raise SystemExit(f"expected {expected} result records, got {len(records)}")
for index, record in enumerate(records):
    responses = record.get("responses", [])
    if len(responses) != 3:
        raise SystemExit(f"record {index} has {len(responses)} party responses")
print(f"NODE_BENCH_CLIENT_COMPLETE records={len(records)}")
PY
}

find_postgres_command() {
    local name=$1 candidate
    if command -v "$name" >/dev/null 2>&1; then
        command -v "$name"
        return
    fi
    for candidate in /usr/lib/postgresql/*/bin/"$name" /usr/pgsql-*/bin/"$name"; do
        if [[ -x ${candidate} ]]; then
            echo "$candidate"
            return
        fi
    done
    echo "missing PostgreSQL command: $name" >&2
    exit 1
}

pid_running() {
    local pid=$1 state
    kill -0 "$pid" >/dev/null 2>&1 || return 1
    state=$(ps -o stat= -p "$pid" 2>/dev/null || true)
    [[ -n ${state} && ${state} != Z* ]]
}

prepare_db() {
    local party=$1
    local initdb pg_ctl createdb dropdb pg_data db_name
    initdb=$(find_postgres_command initdb)
    pg_ctl=$(find_postgres_command pg_ctl)
    createdb=$(find_postgres_command createdb)
    dropdb=$(find_postgres_command dropdb)
    pg_data="${RUN_DIR}/postgres"
    db_name="SMPC_bench_${party}"
    mkdir -p "$RUN_DIR"

    if [[ ! -f ${pg_data}/PG_VERSION ]]; then
        mkdir -p "$pg_data"
        "$initdb" -D "$pg_data" -A trust -U postgres --no-locale >/dev/null
    fi
    if ! "$pg_ctl" -D "$pg_data" status >/dev/null 2>&1; then
        taskset -c "$AUX_CPU_LIST" "$pg_ctl" -D "$pg_data" -l "${pg_data}/postgres.log" \
            -o "-h 127.0.0.1 -k ${pg_data} -p ${POSTGRES_PORT}" start >/dev/null
    fi

    # Postgres is local only for the self-contained benchmark. Keep it off the
    # production-equivalent dot cores, including a server reused from an older
    # run and all of its currently live children. Future children inherit the
    # postmaster's affinity.
    local postgres_pid
    postgres_pid=$(head -n 1 "${pg_data}/postmaster.pid")
    taskset -pc "$AUX_CPU_LIST" "$postgres_pid" >/dev/null
    while read -r child_pid; do
        [[ -z ${child_pid} ]] || taskset -pc "$AUX_CPU_LIST" "$child_pid" >/dev/null
    done < <(pgrep -P "$postgres_pid" || true)

    if [[ ${LINEAR_SCAN_BENCH_RESET_DB:-1} == 1 ]]; then
        "$dropdb" -h 127.0.0.1 -p "$POSTGRES_PORT" -U postgres \
            --if-exists "$db_name"
        "$createdb" -h 127.0.0.1 -p "$POSTGRES_PORT" -U postgres "$db_name"
    else
        "$createdb" -h 127.0.0.1 -p "$POSTGRES_PORT" -U postgres "$db_name" \
            2>/dev/null || true
    fi
    echo "NODE_BENCH_DB_READY party=${party} database=${db_name} port=${POSTGRES_PORT}"
}

start_server() {
    local party=$1
    : "${LINEAR_SCAN_BENCH_NODE_HOSTNAMES:?set JSON array of the three private host addresses}"
    : "${LINEAR_SCAN_BENCH_AWS_ENDPOINT:?set the shared Moto endpoint}"

    local binary=${LINEAR_SCAN_BENCH_SERVER_BINARY:-${PROJECT_ROOT}/target/release/iris-mpc-linear-scan}
    local tls_dir=${LINEAR_SCAN_BENCH_TLS_DIR:-${RUN_DIR}/tls}
    local db_name="SMPC_bench_${party}"
    local pid_file="${RUN_DIR}/server.pid"
    local log_file="${RUN_DIR}/server-${party}.log"
    [[ -x ${binary} ]] || {
        echo "missing server binary: ${binary}" >&2
        exit 1
    }
    for file in "${tls_dir}/tls.key" "${tls_dir}/tls.crt" "${tls_dir}/ca.crt"; do
        [[ -r ${file} ]] || {
            echo "missing TLS file: ${file}" >&2
            exit 1
        }
    done
    if [[ -f ${pid_file} ]] && pid_running "$(<"$pid_file")"; then
        echo "server is already running: pid=$(<"$pid_file")" >&2
        exit 1
    fi

    local endpoint=${LINEAR_SCAN_BENCH_AWS_ENDPOINT%/}
    local max_db_size=$((DATABASE_SIZE + REQUEST_COUNT + 1024))
    local root_certs="[\"${tls_dir}/ca.crt\",\"${tls_dir}/ca.crt\",\"${tls_dir}/ca.crt\"]"
    local image_name=${LINEAR_SCAN_BENCH_IMAGE_NAME:-real-server-benchmark}
    mkdir -p "$RUN_DIR"
    : >"$log_file"

    nohup env \
        AWS_ACCESS_KEY_ID=test \
        AWS_SECRET_ACCESS_KEY=test \
        AWS_REGION=us-east-1 \
        AWS_DEFAULT_REGION=us-east-1 \
        AWS_ENDPOINT_URL="$endpoint" \
        AWS_EC2_METADATA_DISABLED=true \
        RUST_LOG=${RUST_LOG:-info} \
        RUST_BACKTRACE=1 \
        RUST_MIN_STACK=104857600 \
        SMPC__ENVIRONMENT=dev \
        IRIS_MPC_CPU_NTT="$NTT_SCAN" \
        SMPC__PARTY_ID="$party" \
        SMPC__DATABASE__URL="postgres://postgres@127.0.0.1:${POSTGRES_PORT}/${db_name}" \
        SMPC__DATABASE__MIGRATE=true \
        SMPC__DATABASE__CREATE=true \
        SMPC__DATABASE__LOAD_PARALLELISM=8 \
        SMPC__AWS__REGION=us-east-1 \
        SMPC__AWS__ENDPOINT="$endpoint" \
        SMPC__PUBLIC_KEY_BASE_URL="${endpoint}/wf-dev-public-keys" \
        SMPC__REQUESTS_QUEUE_URL="${endpoint}/000000000000/smpcv2-${party}-dev.fifo" \
        SMPC__RESULTS_TOPIC_ARN="arn:aws:sns:us-east-1:000000000000:iris-mpc-results.fifo" \
        SMPC__SHARES_BUCKET_NAME=wf-smpcv2-dev-sns-requests \
        SMPC__GRAPH_CHECKPOINT_BUCKET_NAME=wf-smpcv2-dev-hnsw-checkpoint \
        SMPC__KMS_KEY_ARNS='["unused-0","unused-1","unused-2"]' \
        SMPC__FIXED_SHARED_SECRETS=true \
        SMPC__MAX_BATCH_SIZE=1 \
        SMPC__MAX_DB_SIZE="$max_db_size" \
        SMPC__INIT_DB_SIZE="$DATABASE_SIZE" \
        SMPC__CLEAR_DB_BEFORE_INIT=true \
        SMPC__FAKE_DB_SIZE=0 \
        SMPC__DISABLE_PERSISTENCE=false \
        SMPC__RETURN_PARTIAL_RESULTS=true \
        SMPC__ENABLE_REAUTH=true \
        SMPC__ENABLE_DELETION=true \
        SMPC__ENABLE_RESET=true \
        SMPC__ENABLE_RECOVERY=true \
        SMPC__LUC_ENABLED=true \
        SMPC__LUC_LOOKBACK_RECORDS=500 \
        SMPC__COLD_EYE_LFU_CACHE_RECORDS=12288 \
        SMPC__LUC_SERIAL_IDS_FROM_SMPC_REQUEST=true \
        SMPC__FULL_SCAN_SIDE=Left \
        SMPC__FULL_SCAN_SIDE_SWITCHING_ENABLED=false \
        SMPC__HAWK_REQUEST_PARALLELISM="$REQUEST_PARALLELISM" \
        SMPC__HAWK_CONNECTION_PARALLELISM="$CONNECTION_PARALLELISM" \
        SMPC__SEPARATE_TOKIO_CORES_PER_NODE="$TOKIO_CORES" \
        SMPC__SERVICE_PORTS='["4000","4001","4002"]' \
        SMPC__NODE_HOSTNAMES="$LINEAR_SCAN_BENCH_NODE_HOSTNAMES" \
        SMPC__SERVER_COORDINATION__NODE_HOSTNAMES="$LINEAR_SCAN_BENCH_NODE_HOSTNAMES" \
        SMPC__SERVER_COORDINATION__PARTY_ID="$party" \
        SMPC__SERVER_COORDINATION__HEALTHCHECK_PORTS='["13000","13000","13000"]' \
        SMPC__SERVER_COORDINATION__IMAGE_NAME="$image_name" \
        SMPC__SERVER_COORDINATION__HTTP_QUERY_RETRY_DELAY_MS=250 \
        SMPC__SERVER_COORDINATION__HEARTBEAT_INTERVAL_SECS=1 \
        SMPC__SERVER_COORDINATION__HEARTBEAT_INITIAL_RETRIES=3600 \
        SMPC__TLS__PRIVATE_KEY="${tls_dir}/tls.key" \
        SMPC__TLS__LEAF_CERT="${tls_dir}/tls.crt" \
        SMPC__TLS__ROOT_CERTS="$root_certs" \
        SMPC__SERVICE__SERVICE_NAME="iris-mpc-linear-scan-benchmark-${party}" \
        SMPC__SERVICE__METRICS__HOST=127.0.0.1 \
        SMPC__SERVICE__METRICS__PORT=8125 \
        SMPC__SERVICE__METRICS__QUEUE_SIZE=5000 \
        SMPC__SERVICE__METRICS__BUFFER_SIZE=1024 \
        SMPC__SERVICE__METRICS__PREFIX="linear-scan-benchmark-${party}" \
        "$binary" >>"$log_file" 2>&1 &
    echo "$!" >"$pid_file"
    echo "NODE_BENCH_SERVER_STARTED party=${party} pid=$! log=${log_file}"
}

stop_server() {
    local pid_file="${RUN_DIR}/server.pid"
    if [[ ! -f ${pid_file} ]]; then
        echo "NODE_BENCH_SERVER_STOPPED already=true"
        return
    fi
    local pid
    pid=$(<"$pid_file")
    if pid_running "$pid"; then
        kill "$pid"
        for _ in $(seq 1 30); do
            pid_running "$pid" || break
            sleep 1
        done
        pid_running "$pid" && kill -KILL "$pid"
    fi
    rm -f "$pid_file"
    echo "NODE_BENCH_SERVER_STOPPED pid=${pid}"
}

status() {
    local party=$1 pid_file="${RUN_DIR}/server.pid"
    if [[ -f ${pid_file} ]] && pid_running "$(<"$pid_file")"; then
        curl -fsS "http://127.0.0.1:13000/ready" >/dev/null
        echo "NODE_BENCH_SERVER_READY party=${party} pid=$(<"$pid_file")"
    else
        echo "NODE_BENCH_SERVER_NOT_RUNNING party=${party}" >&2
        return 2
    fi
}

action=${1:-}
party=${2:-}
case "$action" in
prepare-db | start-server | status)
    [[ ${party} =~ ^[0-2]$ ]] || usage
    "${action//-/_}" "$party"
    ;;
stop-server)
    stop_server
    ;;
start-moto | stop-moto | init-moto | rotate-keys | run-client)
    "${action//-/_}"
    ;;
*) usage ;;
esac
