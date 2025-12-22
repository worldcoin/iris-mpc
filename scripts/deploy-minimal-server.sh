#!/usr/bin/env bash
set -euo pipefail

# Build the minimal server on each remote host instead of cross-compiling locally.
# Steps:
#   1. Rsync the repo (excluding target/.git) to ~/minimal-hawk-server-src on each host.
#   2. Run cargo build --release on the host to produce ~/minimal-hawk-server.
#   3. Upload graph snapshot + run script.

BIN_NAME="minimal-hawk-server"
GRAPH_SNAPSHOT="graph.bin"
RUN_SCRIPT="scripts/run-minimal-server.sh"
REMOTE_USER="ec2-user"
REMOTE_SRC_DIR="/home/${REMOTE_USER}/minimal-hawk-server-src"
REMOTE_BIN_PATH="/home/${REMOTE_USER}/${BIN_NAME}"
HOSTS=("aws0" "aws1" "aws2")

echo "[1/2] Syncing source tree to remote hosts..."
for host in "${HOSTS[@]}"; do
  echo "  -> ${host}"
  rsync -az \
    --delete \
    --exclude target \
    --exclude .git \
    --exclude .cursor \
    ./ \
    "${REMOTE_USER}@${host}:${REMOTE_SRC_DIR}/"
done

echo "[2/2] Building ${BIN_NAME} on remote hosts..."
pids=()
for host in "${HOSTS[@]}"; do
  (
    echo "  -> Building on ${host}"
    ssh "${REMOTE_USER}@${host}" "
      set -euo pipefail
      cd ${REMOTE_SRC_DIR}
      if command -v yum >/dev/null 2>&1; then
        sudo yum -y install openssl-devel pkgconfig protobuf-compiler >/dev/null
      elif command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -y >/dev/null
        sudo apt-get install -y libssl-dev pkg-config protobuf-compiler >/dev/null
      fi
      cargo build --release -p iris-mpc-bins --bin ${BIN_NAME}
      cp target/release/${BIN_NAME} ${REMOTE_BIN_PATH}
      chmod +x ${REMOTE_BIN_PATH}
    "
    if [[ -f "${GRAPH_SNAPSHOT}" ]]; then
      echo "      [${host}] Uploading ${GRAPH_SNAPSHOT}"
      scp "${GRAPH_SNAPSHOT}" "${REMOTE_USER}@${host}:/home/${REMOTE_USER}/${GRAPH_SNAPSHOT}"
    fi
    echo "      [${host}] Uploading run script"
    scp "${RUN_SCRIPT}" "${REMOTE_USER}@${host}:/home/${REMOTE_USER}/run-minimal-server.sh"
    echo "  -> ${host} done"
  ) &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "Done. Binaries deployed to ${REMOTE_BIN_PATH} on ${HOSTS[*]}."
