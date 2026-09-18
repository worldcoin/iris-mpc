#!/usr/bin/env bash
set -euo pipefail

# Run the real three-party iris-mpc service on one host without Docker. Moto
# provides S3/SNS/SQS/Secrets Manager, and an ephemeral native PostgreSQL
# cluster provides the three party databases. Every request is sent and
# processed in a batch of one. Results are grouped by logical request, ordered
# by request position and then by party id, and written as canonical JSON.
#
# Usage:
#   scripts/run-native-ground-truth.sh gpu OUTPUT.json [REQUEST_COUNT]
#   scripts/run-native-ground-truth.sh linear-scan OUTPUT.json [REQUEST_COUNT]
#   scripts/run-native-ground-truth.sh compare GPU.json CPU.json
#
# Set GROUND_TRUTH_INITIAL_DB_SIZE to seed each party database with the same
# deterministic MPC shares before starting the request sequence. For example,
# 40000 exceeds both the CPU (4096) and GPU (32768) scan chunk sizes.
#
# Required commands: cargo, curl, Python 3.10+, initdb, pg_ctl, createdb.
# Set MOTO_PYTHON_BIN to a Python interpreter containing moto and boto3.
# Otherwise a temporary venv is populated with moto[server]==5.2.2 and boto3.
# Set MOTO_BASE_PYTHON to select the interpreter used to create that venv.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

usage() {
    sed -n '4,13p' "$0" >&2
    exit 2
}

compare_results() {
    [[ $# -eq 2 ]] || usage
    python3 - "$1" "$2" <<'PY'
import json
import sys

left_path, right_path = sys.argv[1:]
with open(left_path, encoding="utf-8") as f:
    left = json.load(f)
with open(right_path, encoding="utf-8") as f:
    right = json.load(f)

if left == right:
    print(f"GROUND_TRUTH_MATCH records={len(left['records'])}")
    raise SystemExit(0)

left_records = left.get("records", [])
right_records = right.get("records", [])
for index, (left_record, right_record) in enumerate(zip(left_records, right_records)):
    if left_record != right_record:
        print(f"GROUND_TRUTH_MISMATCH first_record={index}", file=sys.stderr)
        print("GPU/left:", json.dumps(left_record, sort_keys=True), file=sys.stderr)
        print("CPU/right:", json.dumps(right_record, sort_keys=True), file=sys.stderr)
        break
else:
    print(
        f"GROUND_TRUTH_MISMATCH record_counts={len(left_records)}/{len(right_records)}",
        file=sys.stderr,
    )
raise SystemExit(1)
PY
}

if [[ ${1:-} == compare ]]; then
    shift
    compare_results "$@"
    exit
fi

[[ $# -ge 2 && $# -le 3 ]] || usage
MODE=$1
OUTPUT_PATH=$2
REQUEST_COUNT=${3:-100}
INITIAL_DB_SIZE=${GROUND_TRUTH_INITIAL_DB_SIZE:-0}
case "$MODE" in
    gpu) SERVER_BINARY=iris-mpc-gpu ;;
    linear-scan) SERVER_BINARY=iris-mpc-linear-scan ;;
    *) usage ;;
esac
[[ "$REQUEST_COUNT" =~ ^[1-9][0-9]*$ ]] || {
    echo "REQUEST_COUNT must be a positive integer" >&2
    exit 2
}
[[ "$INITIAL_DB_SIZE" =~ ^[0-9]+$ ]] || {
    echo "GROUND_TRUTH_INITIAL_DB_SIZE must be a non-negative integer" >&2
    exit 2
}
if (( REQUEST_COUNT < 16 )); then
    echo "REQUEST_COUNT must be at least 16" >&2
    exit 2
fi

OUTPUT_PATH=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTPUT_PATH")
RUN_ROOT=${GROUND_TRUTH_WORK_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/iris-mpc-ground-truth.XXXXXX")}
mkdir -p "$RUN_ROOT" "$(dirname "$OUTPUT_PATH")"
if [[ $(id -u) -eq 0 ]]; then
    # The unprivileged postgres user must be able to traverse the temporary
    # root directory to reach its data directory.
    chmod 755 "$RUN_ROOT"
fi

MOTO_PORT=${MOTO_PORT:-4566}
POSTGRES_PORT=${POSTGRES_PORT:-55432}
AWS_ENDPOINT="http://127.0.0.1:${MOTO_PORT}"
ACCOUNT_ID=000000000000
REGION=us-east-1
RESULT_QUEUE="${AWS_ENDPOINT}/${ACCOUNT_ID}/iris-mpc-results-us-east-1.fifo"

PIDS=()
POSTGRES_STARTED=false
cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    for pid in "${PIDS[@]}"; do
        kill "$pid" >/dev/null 2>&1 || true
    done
    # The production servers perform coordinated graceful shutdown and can
    # otherwise wait indefinitely after one peer has already failed.
    for _ in $(seq 1 10); do
        local any_alive=false
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" >/dev/null 2>&1; then
                any_alive=true
                break
            fi
        done
        [[ "$any_alive" == false ]] && break
        sleep 1
    done
    for pid in "${PIDS[@]}"; do
        kill -KILL "$pid" >/dev/null 2>&1 || true
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    if [[ "$POSTGRES_STARTED" == true ]]; then
        pg_as_owner "$PG_CTL" -D "$PG_DATA" -m fast stop >/dev/null 2>&1 || true
    fi
    echo "Ground-truth logs: $RUN_ROOT"
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

find_postgres_command() {
    local name=$1 candidate
    if command -v "$name" >/dev/null 2>&1; then
        command -v "$name"
        return
    fi
    for candidate in \
        /usr/lib/postgresql/*/bin/"$name" \
        /usr/pgsql-*/bin/"$name"; do
        [[ -x "$candidate" ]] && {
            echo "$candidate"
            return
        }
    done
    echo "Missing PostgreSQL command: $name" >&2
    exit 1
}

INITDB=$(find_postgres_command initdb)
PG_CTL=$(find_postgres_command pg_ctl)
CREATEDB=$(find_postgres_command createdb)

PG_OWNER=$(id -un)
if [[ $(id -u) -eq 0 ]]; then
    if ! id postgres >/dev/null 2>&1; then
        echo "PostgreSQL is installed but no postgres user exists" >&2
        exit 1
    fi
    PG_OWNER=postgres
fi
pg_as_owner() {
    if [[ $(id -u) -eq 0 ]]; then
        runuser -u "$PG_OWNER" -- "$@"
    else
        "$@"
    fi
}

pid_is_running() {
    local pid=$1 state
    kill -0 "$pid" >/dev/null 2>&1 || return 1
    state=$(ps -o stat= -p "$pid" 2>/dev/null || true)
    [[ -n "$state" && "$state" != Z* ]]
}

PG_DATA="$RUN_ROOT/postgres"
mkdir -p "$PG_DATA"
if [[ $(id -u) -eq 0 ]]; then
    chown -R "$PG_OWNER" "$PG_DATA"
fi
pg_as_owner "$INITDB" -D "$PG_DATA" -A trust -U postgres --no-locale >/dev/null
pg_as_owner "$PG_CTL" -D "$PG_DATA" \
    -l "$PG_DATA/postgres.log" \
    -o "-h 127.0.0.1 -k $PG_DATA -p ${POSTGRES_PORT}" start >/dev/null
POSTGRES_STARTED=true
for party in 0 1 2; do
    "$CREATEDB" -h 127.0.0.1 -p "$POSTGRES_PORT" -U postgres "SMPC_dev_${party}"
done

MOTO_PYTHON=""
if [[ -n ${MOTO_PYTHON_BIN:-} ]]; then
    MOTO_PYTHON=$MOTO_PYTHON_BIN
elif [[ -z ${MOTO_BASE_PYTHON:-} ]] && python3 -c 'import boto3, moto' >/dev/null 2>&1; then
    MOTO_PYTHON=python3
else
    MOTO_BASE_PYTHON=${MOTO_BASE_PYTHON:-python3}
    "$MOTO_BASE_PYTHON" -c 'import sys; assert sys.version_info >= (3, 10)' || {
        echo "Moto 5.2.2 requires Python 3.10 or newer; set MOTO_BASE_PYTHON" >&2
        exit 1
    }
    "$MOTO_BASE_PYTHON" -m venv "$RUN_ROOT/moto-venv"
    "$RUN_ROOT/moto-venv/bin/pip" install -q 'moto[server]==5.2.2' boto3
    MOTO_PYTHON="$RUN_ROOT/moto-venv/bin/python"
fi

MOTO_ACCOUNT_ID=$ACCOUNT_ID \
S3_IGNORE_SUBDOMAIN_BUCKETNAME=true \
"$MOTO_PYTHON" "$SCRIPT_DIR/moto-server-with-sns-sequence.py" \
    -H 127.0.0.1 -p "$MOTO_PORT" >"$RUN_ROOT/moto.log" 2>&1 &
PIDS+=("$!")
for _ in $(seq 1 60); do
    curl -fsS "$AWS_ENDPOINT/moto-api/" >/dev/null 2>&1 && break
    sleep 1
done
curl -fsS "$AWS_ENDPOINT/moto-api/" >/dev/null

AWS_ACCESS_KEY_ID=test \
AWS_SECRET_ACCESS_KEY=test \
AWS_DEFAULT_REGION=$REGION \
MOTO_ACCOUNT_ID=$ACCOUNT_ID \
"$MOTO_PYTHON" - "$AWS_ENDPOINT" <<'PY'
import json
import sys

import boto3

endpoint = sys.argv[1]
common = dict(endpoint_url=endpoint, region_name="us-east-1")
s3 = boto3.client("s3", **common)
sns = boto3.client("sns", **common)
sqs = boto3.client("sqs", **common)
secrets = boto3.client("secretsmanager", **common)

for bucket in [
    "wf-dev-public-keys",
    "wf-smpcv2-dev-sns-requests",
    "wf-smpcv2-dev-sync-protocol",
    "wf-smpcv2-dev-hnsw-performance-reports",
    "wf-smpcv2-dev-hnsw-checkpoint",
]:
    s3.create_bucket(Bucket=bucket)
s3.put_bucket_policy(
    Bucket="wf-dev-public-keys",
    Policy=json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": "s3:GetObject",
                    "Resource": "arn:aws:s3:::wf-dev-public-keys/*",
                }
            ],
        }
    ),
)
s3.put_object(
    Bucket="wf-smpcv2-dev-sync-protocol",
    Key="dev_deleted_serial_ids.json",
    Body=json.dumps({"deleted_serial_ids": []}).encode(),
)

input_topic = sns.create_topic(
    Name="iris-mpc-input.fifo",
    Attributes={"FifoTopic": "true", "ContentBasedDeduplication": "true"},
)["TopicArn"]
result_topic = sns.create_topic(
    Name="iris-mpc-results.fifo",
    Attributes={"FifoTopic": "true", "ContentBasedDeduplication": "true"},
)["TopicArn"]

queues = []
for name in [
    "smpcv2-0-dev.fifo",
    "smpcv2-1-dev.fifo",
    "smpcv2-2-dev.fifo",
    "iris-mpc-results-us-east-1.fifo",
]:
    url = sqs.create_queue(
        QueueName=name,
        Attributes={
            "FifoQueue": "true",
            "ContentBasedDeduplication": "true",
            "VisibilityTimeout": "30",
        },
    )["QueueUrl"]
    arn = sqs.get_queue_attributes(
        QueueUrl=url, AttributeNames=["QueueArn"]
    )["Attributes"]["QueueArn"]
    queues.append((url, arn))

for _, arn in queues[:3]:
    sns.subscribe(TopicArn=input_topic, Protocol="sqs", Endpoint=arn)
sns.subscribe(TopicArn=result_topic, Protocol="sqs", Endpoint=queues[3][1])

for party in range(3):
    secrets.create_secret(
        Name=f"dev/iris-mpc/ecdh-private-key-{party}",
        SecretString='{"private-key":""}',
    )
PY

cd "$PROJECT_ROOT"
if [[ ${GROUND_TRUTH_SKIP_BUILD:-0} != 1 ]]; then
    if [[ "$MODE" == gpu ]]; then
        cargo build --release -p iris-mpc-bins \
            --bin iris-mpc-gpu --bin key-manager --bin service-client \
            --features gpu-reference
    else
        cargo build --release -p iris-mpc-bins \
            --bin iris-mpc-linear-scan --bin key-manager --bin service-client
    fi
fi

export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_REGION=$REGION
export AWS_DEFAULT_REGION=$REGION
export AWS_ENDPOINT_URL=$AWS_ENDPOINT
export AWS_EC2_METADATA_DISABLED=true
for party in 0 1 2; do
    for _ in 1 2; do
        "$PROJECT_ROOT/target/release/key-manager" \
            --region "$REGION" \
            --endpoint-url "$AWS_ENDPOINT" \
            --node-id "$party" \
            --env dev rotate \
            --public-key-bucket-name wf-dev-public-keys \
            >>"$RUN_ROOT/key-manager-${party}.log" 2>&1
    done
done

NODE_HOSTNAMES='["127.0.0.1","127.0.0.1","127.0.0.1"]'
# RunPod reserves several low ports for its own proxy, so keep the native test
# topology in a separate high-port range.
SERVICE_PORTS='["14000","14001","14002"]'
HEALTH_PORTS='["13000","13001","13002"]'
KMS_ARNS='["unused-0","unused-1","unused-2"]'
if [[ "$MODE" == gpu ]]; then
    # The GPU actor preallocates its phase-2 buffers in groups of 64. The
    # client still submits exactly one request and waits for its response
    # before submitting the next, so every observed processing batch is one.
    SERVER_MAX_BATCH_SIZE=64
    NCCL_HOST=${GROUND_TRUTH_NCCL_HOST:-$(hostname -I | awk '{print $1}')}
    [[ -n "$NCCL_HOST" ]] || {
        echo "Could not determine a non-loopback NCCL bootstrap address" >&2
        exit 1
    }
else
    SERVER_MAX_BATCH_SIZE=1
fi

for party in 0 1 2; do
    (
        export RUST_LOG=${RUST_LOG:-info}
        export RUST_BACKTRACE=1
        export RUST_MIN_STACK=104857600
        export SMPC__ENVIRONMENT=dev
        export SMPC__PARTY_ID=$party
        export SMPC__DATABASE__URL="postgres://postgres@127.0.0.1:${POSTGRES_PORT}/SMPC_dev_${party}"
        export SMPC__DATABASE__MIGRATE=true
        export SMPC__DATABASE__CREATE=true
        export SMPC__DATABASE__LOAD_PARALLELISM=8
        export SMPC__CPU_DATABASE__URL="$SMPC__DATABASE__URL"
        export SMPC__CPU_DATABASE__MIGRATE=true
        export SMPC__CPU_DATABASE__CREATE=true
        export SMPC__CPU_DATABASE__LOAD_PARALLELISM=8
        export SMPC__AWS__REGION=$REGION
        export SMPC__AWS__ENDPOINT=$AWS_ENDPOINT
        export SMPC__PUBLIC_KEY_BASE_URL="$AWS_ENDPOINT/wf-dev-public-keys"
        export SMPC__REQUESTS_QUEUE_URL="${AWS_ENDPOINT}/${ACCOUNT_ID}/smpcv2-${party}-dev.fifo"
        export SMPC__RESULTS_TOPIC_ARN="arn:aws:sns:${REGION}:${ACCOUNT_ID}:iris-mpc-results.fifo"
        export SMPC__SHARES_BUCKET_NAME=wf-smpcv2-dev-sns-requests
        export SMPC__GRAPH_CHECKPOINT_BUCKET_NAME=wf-smpcv2-dev-hnsw-checkpoint
        export SMPC__KMS_KEY_ARNS=$KMS_ARNS
        export SMPC__FIXED_SHARED_SECRETS=true
        export SMPC__MAX_BATCH_SIZE=$SERVER_MAX_BATCH_SIZE
        export SMPC__MAX_DB_SIZE=$((INITIAL_DB_SIZE + REQUEST_COUNT + 64))
        export SMPC__INIT_DB_SIZE=$INITIAL_DB_SIZE
        export SMPC__CLEAR_DB_BEFORE_INIT=true
        export SMPC__DISABLE_PERSISTENCE=false
        export SMPC__RETURN_PARTIAL_RESULTS=true
        export SMPC__ENABLE_REAUTH=true
        export SMPC__ENABLE_DELETION=true
        export SMPC__ENABLE_RESET=true
        export SMPC__ENABLE_RECOVERY=true
        export SMPC__LUC_ENABLED=true
        export SMPC__LUC_LOOKBACK_RECORDS=0
        export SMPC__LUC_SERIAL_IDS_FROM_SMPC_REQUEST=true
        export SMPC__FULL_SCAN_SIDE=Left
        export SMPC__FULL_SCAN_SIDE_SWITCHING_ENABLED=false
        export SMPC__HAWK_REQUEST_PARALLELISM=1
        export SMPC__HAWK_CONNECTION_PARALLELISM=1
        export SMPC__SERVICE_PORTS=$SERVICE_PORTS
        export SMPC__NODE_HOSTNAMES=$NODE_HOSTNAMES
        export SMPC__SERVER_COORDINATION__NODE_HOSTNAMES=$NODE_HOSTNAMES
        export SMPC__SERVER_COORDINATION__PARTY_ID=$party
        export SMPC__SERVER_COORDINATION__HEALTHCHECK_PORTS=$HEALTH_PORTS
        export SMPC__SERVER_COORDINATION__IMAGE_NAME=ground-truth
        export SMPC__SERVER_COORDINATION__HTTP_QUERY_RETRY_DELAY_MS=250
        export SMPC__SERVER_COORDINATION__HEARTBEAT_INTERVAL_SECS=1
        export SMPC__SERVER_COORDINATION__HEARTBEAT_INITIAL_RETRIES=120
        export SMPC__HAWK_SERVER_HEALTHCHECK_PORT="1300${party}"
        if [[ "$MODE" == gpu ]]; then
            export CUDA_VISIBLE_DEVICES=$party
            # All three production GPU server processes are colocated for this
            # test. NCCL still spans the three parties, so give every process
            # the same loopback bootstrap address while keeping it separate
            # from the application MPC ports.
            # NCCL does not reliably establish a multi-process communicator
            # through loopback; use the host's routable interface even though
            # all three ranks are colocated.
            export NCCL_COMM_ID="${NCCL_HOST}:15000"
            # A colocated process sees only its assigned GPU. Force NCCL onto
            # the socket path used by the multi-host production deployment;
            # CUDA IPC/P2P discovery across those isolated visibility masks can
            # otherwise initialize successfully and then hang on a collective.
            export NCCL_NET=Socket
            export NCCL_SOCKET_IFNAME=${GROUND_TRUTH_NCCL_INTERFACE:-eth0}
            export NCCL_P2P_DISABLE=1
            export NCCL_SHM_DISABLE=1
        fi
        exec "$PROJECT_ROOT/target/release/$SERVER_BINARY"
    ) >"$RUN_ROOT/server-${party}.log" 2>&1 &
    PIDS+=("$!")
done

for party in 0 1 2; do
    ready=false
    for _ in $(seq 1 240); do
        if curl -fsS "http://127.0.0.1:1300${party}/ready" >/dev/null 2>&1; then
            ready=true
            break
        fi
        for pid in "${PIDS[@]}"; do
            pid_is_running "$pid" || {
                echo "A background process exited while waiting for party $party" >&2
                tail -100 "$RUN_ROOT"/*.log >&2 || true
                exit 1
            }
        done
        sleep 1
    done
    [[ "$ready" == true ]] || {
        echo "Party $party did not become healthy" >&2
        tail -100 "$RUN_ROOT/server-${party}.log" >&2 || true
        exit 1
    }
done

CLIENT_CONFIG="$RUN_ROOT/client.toml"
AWS_CONFIG="$RUN_ROOT/aws.toml"
cat >"$CLIENT_CONFIG" <<EOF
results_output_path = "$OUTPUT_PATH"
cleanup_on_exit = false

[shares_generator.FromFile]
path_to_ndjson_file = "$PROJECT_ROOT/iris-mpc-utils/assets/iris-codes-plaintext/20250710-1k.ndjson"
rng_seed = 424242
selection_strategy = "All"

[request_batch.GroundTruth]
request_count = $REQUEST_COUNT
rng_seed = 20260814
initial_uniqueness_count = 16
EOF
cat >"$AWS_CONFIG" <<EOF
environment = "dev"
public_key_base_url = "$AWS_ENDPOINT/wf-dev-public-keys"
s3_request_bucket_name = "wf-smpcv2-dev-sns-requests"
sns_request_topic_arn = "arn:aws:sns:$REGION:$ACCOUNT_ID:iris-mpc-input.fifo"
sqs_long_poll_wait_time = 2
sqs_response_queue_urls = ["$RESULT_QUEUE"]
sqs_wait_time_seconds = 1
EOF

"$PROJECT_ROOT/target/release/service-client" \
    --path-to-opts "$CLIENT_CONFIG" \
    --path-to-opts-aws "$AWS_CONFIG" \
    >"$RUN_ROOT/client.log" 2>&1

python3 - "$OUTPUT_PATH" "$REQUEST_COUNT" "$MODE" "$INITIAL_DB_SIZE" <<'PY'
import json
import sys

path, expected, mode, initial_db_size = (
    sys.argv[1],
    int(sys.argv[2]),
    sys.argv[3],
    int(sys.argv[4]),
)
with open(path, encoding="utf-8") as f:
    result = json.load(f)
records = result.get("records", [])
if len(records) != expected:
    raise SystemExit(f"expected {expected} result records, got {len(records)}")
for index, record in enumerate(records):
    responses = record.get("responses", [])
    if len(responses) != 3:
        raise SystemExit(f"record {index} has {len(responses)} party responses")
print(
    f"GROUND_TRUTH_COMPLETE mode={mode} records={len(records)} "
    f"initial_db_size={initial_db_size} path={path}"
)
PY
