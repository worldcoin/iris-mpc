#!/usr/bin/env bash
set -euo pipefail

# Usage: snapshot_node.sh <container_name> [out_dir=profiles]
# Captures NUMA and TCP snapshots to aid profiling runs.

CONTAINER=${1:?container name is required}
OUTDIR=${2:-profiles}

mkdir -p "${OUTDIR}"
STAMP=$(date +%Y%m%d_%H%M%S)
OUTBASE="${OUTDIR}/${CONTAINER}_${STAMP}"

PID=$(docker exec -i "${CONTAINER}" sh -lc 'pidof iris-mpc-hawk || true')
if [[ -z "${PID}" ]]; then
  echo "[snap] ERROR: iris-mpc-hawk process not found in ${CONTAINER}" >&2
  exit 1
fi

echo "[snap] collecting numastat"
docker exec -i "${CONTAINER}" sh -lc "numastat -p ${PID} || true" > "${OUTBASE}.numastat.txt" || true

echo "[snap] collecting numa_maps"
docker exec -i "${CONTAINER}" sh -lc "cat /proc/${PID}/numa_maps || true" > "${OUTBASE}.numa_maps.txt" || true

echo "[snap] collecting ss -ti"
docker exec -i "${CONTAINER}" sh -lc "ss -ti || true" > "${OUTBASE}.ss.txt" || true

echo "[snap] done. Artifacts under ${OUTDIR} (prefix ${OUTBASE})."

