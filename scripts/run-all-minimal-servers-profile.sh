#!/usr/bin/env bash
set -euo pipefail

# Run all minimal hawk servers, with samply profiling on party 0 (aws0) only.
# Parties 1 and 2 run normally without profiling overhead.

DB_SIZE="${1:-1000}"
PROFILE_DELAY="${2:-5}"
HOSTS=("aws0" "aws1" "aws2")
PROFILE_HOST="aws0"

echo "=== Running All Minimal Servers (Profiling on ${PROFILE_HOST} only) ==="
echo "DB Size: ${DB_SIZE}"
echo "Profile Delay: ${PROFILE_DELAY}s"
echo ""

prefix_logs() {
  local host="$1"
  sed -u "s/^/[${host}] /"
}

cleanup() {
  echo ""
  echo "Stopping remote servers..."
  for host in "${HOSTS[@]}"; do
    ssh "ec2-user@${host}" "pkill -f minimal-hawk-server" 2>/dev/null &
  done
  wait
  echo "Done."
  exit 0
}

trap cleanup SIGINT SIGTERM

# Check that graph caches exist on all hosts
echo "Checking for graph caches on all hosts..."
MISSING_CACHE=false
for host in "${HOSTS[@]}"; do
  if ! ssh "ec2-user@${host}" "test -f /home/ec2-user/graph.bin"; then
    echo "  [${host}] Graph cache missing!"
    MISSING_CACHE=true
  else
    echo "  [${host}] Graph cache found."
  fi
done

if [[ "${MISSING_CACHE}" == "true" ]]; then
  echo ""
  echo "Error: Graph cache missing on one or more hosts."
  echo "Please run servers once without profiling to build caches:"
  echo "  ./run-all-minimal-servers.sh ${DB_SIZE}"
  exit 1
fi

# Kill any existing servers first
echo ""
echo "Killing any existing servers..."
for host in "${HOSTS[@]}"; do
  ssh "ec2-user@${host}" "pkill -f minimal-hawk-server" >/dev/null 2>&1 || true
done

echo ""
echo "Starting servers (profiling on ${PROFILE_HOST} only)..."

for idx in "${!HOSTS[@]}"; do
  host="${HOSTS[$idx]}"
  if [[ "${host}" == "${PROFILE_HOST}" ]]; then
    echo "Starting party ${idx} on ${host} WITH PROFILING (delay=${PROFILE_DELAY}s)..."
    ssh "ec2-user@${host}" "bash ~/run-minimal-server-profile.sh ${idx} ${DB_SIZE} ${PROFILE_DELAY}" 2>&1 | prefix_logs "${host}" &
  else
    echo "Starting party ${idx} on ${host} (no profiling)..."
    ssh "ec2-user@${host}" "bash ~/run-minimal-server.sh ${idx} ${DB_SIZE}" 2>&1 | prefix_logs "${host}" &
  fi
done

echo ""
echo "All parties started. Waiting for request to complete..."

# Wait for all background jobs (servers will exit after single request)
wait

echo ""
echo "=== All parties finished ==="
echo ""
echo "Profile saved on ${PROFILE_HOST} at /home/ec2-user/profiles/"
echo ""
echo "Samply server should now be running on ${PROFILE_HOST}:3000"
echo "If you have a tunnel running (start-profile-tunnel.sh), view at: http://localhost:3000"

