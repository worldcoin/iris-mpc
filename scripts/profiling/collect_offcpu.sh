#!/usr/bin/env bash
set -euo pipefail

# Usage: collect_offcpu.sh <container_name> [seconds=60] [out_dir=profiles]
# Example: ./scripts/profiling/collect_offcpu.sh hawk_participant_0 60 profiles

CONTAINER=${1:?container name is required}
SECONDS=${2:-60}
OUTDIR=${3:-profiles}

mkdir -p "${OUTDIR}"

PID=$(docker exec -i "${CONTAINER}" sh -lc 'pidof iris-mpc-hawk || true')
if [[ -z "${PID}" ]]; then
  echo "[offcpu] ERROR: iris-mpc-hawk process not found in ${CONTAINER}" >&2
  exit 1
fi

HAS_OFFCPU=$(docker exec -i "${CONTAINER}" sh -lc 'command -v offcputime-bpfcc >/dev/null 2>&1 && echo yes || echo no')
if [[ "${HAS_OFFCPU}" != "yes" ]]; then
  echo "[offcpu] WARNING: offcputime-bpfcc is not installed in container ${CONTAINER}. Skipping off-CPU collection."
  echo "          Consider running on the host or installing bcc-tools inside the image."
  exit 0
fi

STAMP=$(date +%Y%m%d_%H%M%S)
OUTBASE="${OUTDIR}/${CONTAINER}_${STAMP}"

echo "[offcpu] collecting folded stacks for ${SECONDS}s (PID ${PID})"
docker exec -i "${CONTAINER}" sh -lc \
  "offcputime-bpfcc -p ${PID} -f 99 -d ${SECONDS} > /tmp/offcpu.folded"

docker cp "${CONTAINER}:/tmp/offcpu.folded" "${OUTBASE}.offcpu.folded" || true
echo "[offcpu] done. Artifact: ${OUTBASE}.offcpu.folded (render with FlameGraph or inferno)"

