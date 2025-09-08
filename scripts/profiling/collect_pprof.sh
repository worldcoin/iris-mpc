#!/usr/bin/env bash
set -euo pipefail

# Usage: collect_pprof.sh <host> <port> [seconds=30] [frequency=99] [out_prefix=profile]
# Example: ./scripts/profiling/collect_pprof.sh localhost 3000 30 99 node0

HOST=${1:-localhost}
PORT=${2:-3000}
SECONDS=${3:-30}
FREQ=${4:-99}
OUT=${5:-profile}

BASE_URL="http://${HOST}:${PORT}/pprof"

echo "[pprof] collecting flamegraph SVG (${SECONDS}s @ ${FREQ}Hz) from ${BASE_URL}/flame"
curl -sS "${BASE_URL}/flame?seconds=${SECONDS}&frequency=${FREQ}" -o "${OUT}.flame.svg"
echo "[pprof] saved ${OUT}.flame.svg"

echo "[pprof] collecting protobuf profile (${SECONDS}s @ ${FREQ}Hz) from ${BASE_URL}/profile"
curl -sS "${BASE_URL}/profile?seconds=${SECONDS}&frequency=${FREQ}" -o "${OUT}.pprof"
echo "[pprof] saved ${OUT}.pprof"

echo "[pprof] done"

