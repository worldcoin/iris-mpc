#!/usr/bin/env bash
#
# Port-forward all 3 dev hawk parties to local ports.
#
# Usage:
#   ./scripts/port-forward-dev.sh [base_port]
#
# Forwards:
#   party 0 :3000 -> localhost:<base_port>
#   party 1 :3000 -> localhost:<base_port+1>
#   party 2 :3000 -> localhost:<base_port+2>
#
# Default base_port is 3100, giving localhost:3100, :3101, :3102.
# Ctrl+C stops all forwards.

set -euo pipefail

BASE_PORT="${1:-3100}"

# Unset localstack env vars that direnv may have set
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_ENDPOINT_URL 2>/dev/null || true

# Ensure EKS contexts exist
for i in 0 1 2; do
  AWS_PROFILE="worldcoin-smpcv-io-${i}-dev" aws eks update-kubeconfig \
    --name "ampc-hnsw-${i}-dev" --region eu-central-1 --alias "ampc-hnsw-${i}-dev" 2>/dev/null
done

# Get pod names
PODS=()
for i in 0 1 2; do
  POD=$(kubectl --context "ampc-hnsw-${i}-dev" -n ampc-hnsw \
    get pods -l app.kubernetes.io/managed-by=Helm \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  if [ -z "$POD" ]; then
    echo "ERROR: Could not find pod for party $i" >&2
    exit 1
  fi
  PODS+=("$POD")
  echo "Party $i: pod=${POD}"
done

# Start port-forwards in background
PIDS=()
cleanup() {
  echo ""
  echo "Stopping port-forwards..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

for i in 0 1 2; do
  LOCAL_PORT=$((BASE_PORT + i))
  echo "Forwarding localhost:${LOCAL_PORT} -> ${PODS[$i]}:3000 (party $i)"
  kubectl --context "ampc-hnsw-${i}-dev" -n ampc-hnsw \
    port-forward "${PODS[$i]}" "${LOCAL_PORT}:3000" &
  PIDS+=($!)
done

echo ""
echo "Port forwards active:"
for i in 0 1 2; do
  echo "  Party $i: http://localhost:$((BASE_PORT + i))"
done
echo ""
echo "Press Ctrl+C to stop."

# Wait for any child to exit (or Ctrl+C)
wait
