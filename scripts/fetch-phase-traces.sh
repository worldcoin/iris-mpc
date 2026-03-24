#!/usr/bin/env bash
# Fetch phase trace JSON files from dev cluster pods to local machine.
# Usage: ./scripts/fetch-phase-traces.sh [output_dir]
#
# Copies /tmp/hawk_phase_trace_*.json from each party pod into:
#   <output_dir>/party<N>/hawk_phase_trace_*.json
#
# Default output_dir: ./flamegraphs/phase_traces

set -euo pipefail

# Unset localstack env vars (direnv sets these and they break real AWS auth)
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_ENDPOINT_URL

OUTPUT_DIR="${1:-./flamegraphs/phase_traces}"
NAMESPACE="ampc-hnsw"

# Get pod name for a given party index
get_pod() {
  kubectl --context "ampc-hnsw-${1}-dev" -n "$NAMESPACE" \
    get pods -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
}

# List trace files in a pod
list_traces() {
  kubectl --context "ampc-hnsw-${1}-dev" -n "$NAMESPACE" \
    exec "$2" -- sh -c 'ls /tmp/hawk_phase_trace_*.json 2>/dev/null' || true
}

for i in 0 1 2; do
  echo "=== Party $i ==="
  pod=$(get_pod "$i")
  if [ -z "$pod" ]; then
    echo "  No pod found, skipping"
    continue
  fi
  echo "  Pod: $pod"

  # List available trace files
  files=$(list_traces "$i" "$pod")
  if [ -z "$files" ]; then
    echo "  No trace files found"
    continue
  fi

  # Create output directory
  dest="${OUTPUT_DIR}/party${i}"
  mkdir -p "$dest"

  # Copy each file
  for f in $files; do
    basename=$(basename "$f")
    echo "  Copying $basename..."
    kubectl --context "ampc-hnsw-${i}-dev" -n "$NAMESPACE" \
      cp "${pod}:${f}" "${dest}/${basename}"
  done

  echo "  Done. Files in ${dest}/"
done

echo ""
echo "All traces saved to ${OUTPUT_DIR}/"
echo "Open in https://ui.perfetto.dev/"
