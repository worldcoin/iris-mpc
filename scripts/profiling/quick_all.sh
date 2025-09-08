#!/usr/bin/env bash
set -euo pipefail

# Quick multi-node collector (pprof flame + snapshots).
# Defaults match docker-compose.test.yaml and profiling override.
#
# Usage:
#   ./scripts/profiling/quick_all.sh [seconds=30]
#
# Artifacts will be placed under profiles/ with timestamped filenames.

SECONDS=${1:-30}

declare -A HOSTS=(
  "127.0.0.1:3000"
  "127.0.0.1:3001"
  "127.0.0.1:3002"
)

OUTDIR="profiles"
mkdir -p "${OUTDIR}"
STAMP=$(date +%Y%m%d_%H%M%S)

for NAME in "${!HOSTS[@]}"; do
  PORT=${HOSTS[$NAME]}
  PREFIX="${OUTDIR}/${NAME}_${STAMP}"
  echo "=== Node ${NAME} (port ${PORT}) ==="
  ./scripts/profiling/collect_pprof.sh localhost ${PORT} ${SECONDS} 99 "${PREFIX}" || true
  ./scripts/profiling/snapshot_node.sh ${NAME} "${OUTDIR}" || true
done

echo "[quick_all] done. Profiles in ${OUTDIR}/"

