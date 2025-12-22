#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <party-index 0|1|2> [db-size] [force]"
  exit 1
fi

PARTY_INDEX="$1"
DB_SIZE="${2:-1000}"
FORCE="${3:-}"

if [[ "${PARTY_INDEX}" != "0" && "${PARTY_INDEX}" != "1" && "${PARTY_INDEX}" != "2" ]]; then
  echo "party-index must be 0, 1, or 2"
  exit 1
fi

INITIATOR_FLAG=""
if [[ "${PARTY_INDEX}" == "0" ]]; then
  INITIATOR_FLAG="--initiator"
fi

BIN_PATH="/home/ec2-user/minimal-hawk-server"
GRAPH_PATH="/home/ec2-user/graph.bin"
ADDRS="172.31.21.251:16000,172.31.22.229:16000,172.31.24.35:16000"

# Remove cached graph if force flag is set
if [[ "${FORCE}" == "force" ]]; then
  echo "Removing cached graph at ${GRAPH_PATH}"
  rm -f "${GRAPH_PATH}"
fi

exec "${BIN_PATH}" \
  --party-index "${PARTY_INDEX}" \
  --addresses "${ADDRS}" \
  --outbound-addrs "${ADDRS}" \
  --hnsw-param-m 256 \
  --hnsw-param-ef-constr 320 \
  --hnsw-param-ef-search 320 \
  --graph-cache-mode auto \
  --graph-cache-path "${GRAPH_PATH}" \
  --db-size "${DB_SIZE}" \
  --single-request \
  ${INITIATOR_FLAG}

