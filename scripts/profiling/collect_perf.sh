#!/usr/bin/env bash
set -euo pipefail

# Usage: collect_perf.sh <container_name> [seconds=60] [frequency=99] [out_dir=profiles]
# Example: ./scripts/profiling/collect_perf.sh hawk_participant_0 60 99 profiles

CONTAINER=${1:?container name is required}
SECONDS=${2:-60}
FREQ=${3:-99}
OUTDIR=${4:-profiles}

mkdir -p "${OUTDIR}"

echo "[perf] targeting container ${CONTAINER} for ${SECONDS}s @ ${FREQ}Hz"

PID=$(docker exec -i "${CONTAINER}" sh -lc 'pidof iris-mpc-hawk || true')
if [[ -z "${PID}" ]]; then
  echo "[perf] ERROR: iris-mpc-hawk process not found in ${CONTAINER}" >&2
  exit 1
fi

HAS_PERF=$(docker exec -i "${CONTAINER}" sh -lc 'command -v perf >/dev/null 2>&1 && echo yes || echo no')
if [[ "${HAS_PERF}" != "yes" ]]; then
  echo "[perf] WARNING: perf is not installed in container ${CONTAINER}. Skipping perf collection."
  echo "        You can still use pprof-based flamegraphs (collect_pprof.sh)." 
  echo "        To use perf, install linux-tools matching the host kernel on the host, or into the image,"
  echo "        and ensure the container has caps: SYS_ADMIN, SYS_PTRACE, PERFMON and seccomp=unconfined."
  exit 0
fi

STAMP=$(date +%Y%m%d_%H%M%S)
OUTBASE="${OUTDIR}/${CONTAINER}_${STAMP}"

echo "[perf] recording... (PID ${PID})"
docker exec -i "${CONTAINER}" sh -lc \
  "perf record -F ${FREQ} --call-graph fp -p ${PID} --output=/tmp/perf.data -- sleep ${SECONDS}"

echo "[perf] extracting perf.script"
docker exec -i "${CONTAINER}" sh -lc "perf script > /tmp/perf.script"

echo "[perf] copying artifacts"
docker cp "${CONTAINER}:/tmp/perf.data" "${OUTBASE}.perf.data" || true
docker cp "${CONTAINER}:/tmp/perf.script" "${OUTBASE}.perf.script" || true

echo "[perf] done. Artifacts: ${OUTBASE}.perf.data, ${OUTBASE}.perf.script"

