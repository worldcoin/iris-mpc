#!/usr/bin/env bash
set -euo pipefail

# Run the actual iris-mpc-linear-scan service across three remote hosts. Moto is
# only the AWS control plane; every request still traverses the production
# client, S3/SNS/SQS ingestion, server scheduler, TLS MPC network, persistence,
# and result publication paths.
#
# Usage:
#   LINEAR_SCAN_BENCH_SSH_KEY=/path/to/key.pem \
#     scripts/run-distributed-linear-scan-benchmark.sh \
#     ubuntu@host0 ubuntu@host1 ubuntu@host2 [output-directory]
#
# Important overrides:
#   LINEAR_SCAN_BENCH_DATABASE_SIZE=1048576  # >= 256 production 4K chunks
#   LINEAR_SCAN_BENCH_REQUEST_COUNT=6        # first request is warm-up
#   LINEAR_SCAN_BENCH_REQUEST_PARALLELISM=48  # tuned for r8g.24xlarge
#   LINEAR_SCAN_BENCH_CONNECTION_PARALLELISM=16
#   LINEAR_SCAN_BENCH_PIPELINED_REQUESTS=1    # queue independent requests together
#   LINEAR_SCAN_BENCH_REUSE_DB=1             # reuse the expensive seeded DB
#   LINEAR_SCAN_BENCH_SKIP_BUILD=1           # reuse binaries already copied
#   LINEAR_SCAN_BENCH_KEEP_RUNNING=1          # leave servers and Moto running
#   LINEAR_SCAN_BENCH_NODE_ADDRESSES=a,b,c    # override detected private IPs

[[ $# -ge 3 && $# -le 4 ]] || {
    sed -n '5,20p' "$0" >&2
    exit 2
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
HOSTS=("$1" "$2" "$3")
OUTPUT_DIR=${4:-${PROJECT_ROOT}/target/linear-scan-real-server-benchmark}
DATABASE_SIZE=${LINEAR_SCAN_BENCH_DATABASE_SIZE:-1048576}
REQUEST_COUNT=${LINEAR_SCAN_BENCH_REQUEST_COUNT:-6}
WARMUP_REQUESTS=${LINEAR_SCAN_BENCH_WARMUP_REQUESTS:-1}
REQUEST_PARALLELISM=${LINEAR_SCAN_BENCH_REQUEST_PARALLELISM:-48}
CONNECTION_PARALLELISM=${LINEAR_SCAN_BENCH_CONNECTION_PARALLELISM:-16}
TOKIO_CORES=${LINEAR_SCAN_BENCH_TOKIO_CORES:-11}
CLIENT_RNG_SEED=${LINEAR_SCAN_BENCH_CLIENT_RNG_SEED:-8675309}
PIPELINED_REQUESTS=${LINEAR_SCAN_BENCH_PIPELINED_REQUESTS:-0}
REMOTE_RUN_DIR=${LINEAR_SCAN_BENCH_REMOTE_RUN_DIR:-/var/tmp/iris-mpc-real-server-bench}
COMMIT=$(git -C "$PROJECT_ROOT" rev-parse HEAD)
REMOTE_SOURCE=${LINEAR_SCAN_BENCH_REMOTE_SOURCE:-/var/tmp/iris-mpc-source-${COMMIT}}

[[ ${DATABASE_SIZE} =~ ^[1-9][0-9]*$ ]] || {
    echo "LINEAR_SCAN_BENCH_DATABASE_SIZE must be positive" >&2
    exit 2
}
[[ ${REQUEST_COUNT} =~ ^[1-9][0-9]*$ ]] || {
    echo "LINEAR_SCAN_BENCH_REQUEST_COUNT must be positive" >&2
    exit 2
}
[[ ${CLIENT_RNG_SEED} =~ ^[0-9]+$ ]] || {
    echo "LINEAR_SCAN_BENCH_CLIENT_RNG_SEED must be a non-negative integer" >&2
    exit 2
}
[[ ${CONNECTION_PARALLELISM} =~ ^[1-9][0-9]*$ ]] || {
    echo "LINEAR_SCAN_BENCH_CONNECTION_PARALLELISM must be positive" >&2
    exit 2
}
[[ ${TOKIO_CORES} =~ ^[1-9][0-9]*$ ]] || {
    echo "LINEAR_SCAN_BENCH_TOKIO_CORES must be positive" >&2
    exit 2
}
[[ ${PIPELINED_REQUESTS} =~ ^[01]$ ]] || {
    echo "LINEAR_SCAN_BENCH_PIPELINED_REQUESTS must be 0 or 1" >&2
    exit 2
}
[[ ${WARMUP_REQUESTS} =~ ^[0-9]+$ && ${WARMUP_REQUESTS} -lt ${REQUEST_COUNT} ]] || {
    echo "warm-up count must be non-negative and smaller than request count" >&2
    exit 2
}
if [[ ${LINEAR_SCAN_BENCH_SKIP_SYNC:-0} != 1 ]] && \
    [[ -n $(git -C "$PROJECT_ROOT" status --short) ]]; then
    echo "distributed benchmark requires a clean committed tree" >&2
    exit 1
fi

SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30)
SCP_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=10)
if [[ -n ${LINEAR_SCAN_BENCH_SSH_KEY:-} ]]; then
    SSH_OPTIONS+=(-i "$LINEAR_SCAN_BENCH_SSH_KEY" -o IdentitiesOnly=yes)
    SCP_OPTIONS+=(-i "$LINEAR_SCAN_BENCH_SSH_KEY" -o IdentitiesOnly=yes)
fi

remote() {
    local host=$1 quoted
    shift
    printf -v quoted '%q ' "$@"
    ssh "${SSH_OPTIONS[@]}" "$host" "${quoted% }"
}

remote_env() {
    local host=$1
    shift
    remote "$host" env \
        "LINEAR_SCAN_BENCH_RUN_DIR=${REMOTE_RUN_DIR}" \
        "LINEAR_SCAN_BENCH_DATABASE_SIZE=${DATABASE_SIZE}" \
        "LINEAR_SCAN_BENCH_REQUEST_COUNT=${REQUEST_COUNT}" \
        "LINEAR_SCAN_BENCH_REQUEST_PARALLELISM=${REQUEST_PARALLELISM}" \
        "LINEAR_SCAN_BENCH_CONNECTION_PARALLELISM=${CONNECTION_PARALLELISM}" \
        "LINEAR_SCAN_BENCH_TOKIO_CORES=${TOKIO_CORES}" \
        "LINEAR_SCAN_BENCH_CLIENT_RNG_SEED=${CLIENT_RNG_SEED}" \
        "LINEAR_SCAN_BENCH_PIPELINED_REQUESTS=${PIPELINED_REQUESTS}" \
        "LINEAR_SCAN_BENCH_NODE_HOSTNAMES=${NODE_HOSTNAMES_JSON}" \
        "LINEAR_SCAN_BENCH_AWS_ENDPOINT=${AWS_ENDPOINT}" \
        "LINEAR_SCAN_BENCH_SERVER_BINARY=${REMOTE_SOURCE}/target/release/iris-mpc-linear-scan" \
        "LINEAR_SCAN_BENCH_KEY_MANAGER_BINARY=${REMOTE_SOURCE}/target/release/key-manager" \
        "LINEAR_SCAN_BENCH_CLIENT_BINARY=${REMOTE_SOURCE}/target/release/service-client" \
        "LINEAR_SCAN_BENCH_IMAGE_NAME=real-server-benchmark-${COMMIT}" \
        "$@"
}

mkdir -p "$OUTPUT_DIR"
LOCAL_TMP=$(mktemp -d "${TMPDIR:-/tmp}/iris-mpc-real-server-bench.XXXXXX")
SERVERS_STARTED=false
MOTO_STARTED=false
cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [[ ${LINEAR_SCAN_BENCH_KEEP_RUNNING:-0} != 1 ]]; then
        if [[ ${SERVERS_STARTED} == true ]]; then
            for host in "${HOSTS[@]}"; do
                remote_env "$host" "${REMOTE_SOURCE}/scripts/run-distributed-linear-scan-node.sh" \
                    stop-server >/dev/null 2>&1 &
            done
            wait || true
        fi
        if [[ ${MOTO_STARTED} == true ]]; then
            remote_env "${HOSTS[0]}" \
                "${REMOTE_SOURCE}/scripts/run-distributed-linear-scan-node.sh" \
                stop-moto >/dev/null 2>&1 || true
        fi
    fi
    rm -rf "$LOCAL_TMP"
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

for host in "${HOSTS[@]}"; do
    remote "$host" true
done

if [[ -n ${LINEAR_SCAN_BENCH_NODE_ADDRESSES:-} ]]; then
    IFS=',' read -r -a NODE_ADDRESSES <<<"$LINEAR_SCAN_BENCH_NODE_ADDRESSES"
else
    NODE_ADDRESSES=()
    for host in "${HOSTS[@]}"; do
        NODE_ADDRESSES+=("$(remote "$host" sh -c "hostname -I | cut -d' ' -f1")")
    done
fi
[[ ${#NODE_ADDRESSES[@]} -eq 3 ]] || {
    echo "expected exactly three node addresses" >&2
    exit 1
}
NODE_HOSTNAMES_JSON=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' \
    "${NODE_ADDRESSES[@]}")
AWS_ENDPOINT="http://${NODE_ADDRESSES[0]}:4566"
echo "REAL_SERVER_BENCH_TOPOLOGY commit=${COMMIT} nodes=${NODE_ADDRESSES[*]} moto=${AWS_ENDPOINT} request_parallelism=${REQUEST_PARALLELISM} connection_parallelism=${CONNECTION_PARALLELISM} tokio_cores=${TOKIO_CORES} client_rng_seed=${CLIENT_RNG_SEED} pipelined_requests=${PIPELINED_REQUESTS}"

if [[ ${LINEAR_SCAN_BENCH_SKIP_SYNC:-0} != 1 ]]; then
    git -C "$PROJECT_ROOT" archive --format=tar "$COMMIT" -o "${LOCAL_TMP}/source.tar"
    for host in "${HOSTS[@]}"; do
        remote "$host" mkdir -p "$REMOTE_SOURCE"
        scp "${SCP_OPTIONS[@]}" "${LOCAL_TMP}/source.tar" "${host}:${REMOTE_SOURCE}/source.tar"
        remote "$host" tar -xf "${REMOTE_SOURCE}/source.tar" -C "$REMOTE_SOURCE"
        remote "$host" chmod +x \
            "${REMOTE_SOURCE}/scripts/run-distributed-linear-scan-node.sh" \
            "${REMOTE_SOURCE}/scripts/moto-server-with-sns-sequence.py" \
            "${REMOTE_SOURCE}/scripts/init-moto-linear-scan-benchmark.py"
    done
fi

if [[ ${LINEAR_SCAN_BENCH_SKIP_BUILD:-0} != 1 ]]; then
    remote "${HOSTS[0]}" bash -lc \
        "cd '$REMOTE_SOURCE' && RUSTFLAGS='--cfg aes_armv8 -C force-frame-pointers=yes -Ctarget-cpu=neoverse-v2 -Ctarget-feature=+lse' cargo build --release -p iris-mpc-bins --features aes_rng_prf --bin iris-mpc-linear-scan --bin key-manager --bin service-client"
    for binary in iris-mpc-linear-scan key-manager service-client; do
        scp "${SCP_OPTIONS[@]}" \
            "${HOSTS[0]}:${REMOTE_SOURCE}/target/release/${binary}" \
            "${LOCAL_TMP}/${binary}"
    done
    for host in "${HOSTS[@]}"; do
        remote "$host" mkdir -p "${REMOTE_SOURCE}/target/release"
        scp "${SCP_OPTIONS[@]}" "${LOCAL_TMP}/iris-mpc-linear-scan" \
            "${LOCAL_TMP}/key-manager" "${LOCAL_TMP}/service-client" \
            "${host}:${REMOTE_SOURCE}/target/release/"
        remote "$host" chmod +x \
            "${REMOTE_SOURCE}/target/release/iris-mpc-linear-scan" \
            "${REMOTE_SOURCE}/target/release/key-manager" \
            "${REMOTE_SOURCE}/target/release/service-client"
    done
fi

TLS_DIR="${LOCAL_TMP}/tls"
mkdir -p "$TLS_DIR"
SAN=$(printf 'IP:%s,' "${NODE_ADDRESSES[@]}")
SAN=${SAN%,}
openssl req -x509 -newkey rsa:2048 -nodes -days 2 \
    -subj '/CN=iris-mpc-real-server-benchmark-ca' \
    -addext 'basicConstraints=critical,CA:TRUE' \
    -addext 'keyUsage=critical,keyCertSign,cRLSign' \
    -keyout "${TLS_DIR}/ca.key" -out "${TLS_DIR}/ca.crt" >/dev/null 2>&1
openssl req -newkey rsa:2048 -nodes \
    -subj '/CN=iris-mpc-real-server-benchmark' \
    -addext "subjectAltName=${SAN}" \
    -keyout "${TLS_DIR}/tls.key" -out "${TLS_DIR}/tls.csr" >/dev/null 2>&1
printf 'subjectAltName=%s\nextendedKeyUsage=serverAuth,clientAuth\n' "$SAN" \
    >"${TLS_DIR}/tls.ext"
openssl x509 -req -days 2 -in "${TLS_DIR}/tls.csr" \
    -CA "${TLS_DIR}/ca.crt" -CAkey "${TLS_DIR}/ca.key" -CAcreateserial \
    -extfile "${TLS_DIR}/tls.ext" -out "${TLS_DIR}/tls.crt" >/dev/null 2>&1
for host in "${HOSTS[@]}"; do
    remote "$host" mkdir -p "${REMOTE_RUN_DIR}/tls"
    scp "${SCP_OPTIONS[@]}" "${TLS_DIR}/ca.crt" "${TLS_DIR}/tls.crt" \
        "${TLS_DIR}/tls.key" "${host}:${REMOTE_RUN_DIR}/tls/"
done

remote_env "${HOSTS[0]}" "${REMOTE_SOURCE}/scripts/run-distributed-linear-scan-node.sh" \
    start-moto
MOTO_STARTED=true
remote_env "${HOSTS[0]}" "${REMOTE_SOURCE}/scripts/run-distributed-linear-scan-node.sh" \
    init-moto
remote_env "${HOSTS[0]}" "${REMOTE_SOURCE}/scripts/run-distributed-linear-scan-node.sh" \
    rotate-keys

RESET_DB=1
[[ ${LINEAR_SCAN_BENCH_REUSE_DB:-0} == 1 ]] && RESET_DB=0
for party in 0 1 2; do
    remote_env "${HOSTS[$party]}" "LINEAR_SCAN_BENCH_RESET_DB=${RESET_DB}" \
        "${REMOTE_SOURCE}/scripts/run-distributed-linear-scan-node.sh" prepare-db "$party" &
done
wait

for party in 0 1 2; do
    remote_env "${HOSTS[$party]}" \
        "${REMOTE_SOURCE}/scripts/run-distributed-linear-scan-node.sh" start-server "$party" &
done
wait
SERVERS_STARTED=true

for _ in $(seq 1 7200); do
    all_ready=true
    for party in 0 1 2; do
        remote_env "${HOSTS[$party]}" \
            "${REMOTE_SOURCE}/scripts/run-distributed-linear-scan-node.sh" status "$party" \
            >/dev/null 2>&1 || all_ready=false
    done
    [[ ${all_ready} == true ]] && break
    sleep 1
done
[[ ${all_ready} == true ]] || {
    echo "servers did not become ready" >&2
    exit 1
}
echo "REAL_SERVER_BENCH_SERVERS_READY database_size=${DATABASE_SIZE}"

remote_env "${HOSTS[0]}" "${REMOTE_SOURCE}/scripts/run-distributed-linear-scan-node.sh" \
    run-client

for party in 0 1 2; do
    scp "${SCP_OPTIONS[@]}" \
        "${HOSTS[$party]}:${REMOTE_RUN_DIR}/server-${party}.log" \
        "${OUTPUT_DIR}/server-${party}.log"
done
scp "${SCP_OPTIONS[@]}" "${HOSTS[0]}:${REMOTE_RUN_DIR}/client.log" \
    "${OUTPUT_DIR}/client.log"

ANALYZER_ARGS=(
    "${SCRIPT_DIR}/analyze-linear-scan-server-benchmark.py"
    --warmup-requests "$WARMUP_REQUESTS"
    --json "${OUTPUT_DIR}/summary.json"
)
if [[ -n ${LINEAR_SCAN_BENCH_MINIMUM_CPS:-} ]]; then
    ANALYZER_ARGS+=(--minimum-cps "$LINEAR_SCAN_BENCH_MINIMUM_CPS")
fi
python3 "${ANALYZER_ARGS[@]}" \
    "${OUTPUT_DIR}/server-0.log" \
    "${OUTPUT_DIR}/server-1.log" \
    "${OUTPUT_DIR}/server-2.log"
echo "REAL_SERVER_BENCH_COMPLETE output=${OUTPUT_DIR}"
