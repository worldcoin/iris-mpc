#!/usr/bin/env bash

DB_SIZE="${1:-1000}"
FORCE="${2:-}"
PERF="${3:-}"
HOSTS=("aws0" "aws1" "aws2")

prefix_logs() {
  local host="$1"
  sed -u "s/^/[${host}] /"
}

cleanup() {
  echo ""
  echo "Stopping remote servers..."
  # Kill remote servers
  for host in "${HOSTS[@]}"; do
    ssh "ec2-user@${host}" "pkill -f minimal-hawk-server" 2>/dev/null &
  done
  wait
  echo "Done."
  exit 0
}

trap cleanup SIGINT SIGTERM

# Kill any existing servers first
for host in "${HOSTS[@]}"; do
  ssh "ec2-user@${host}" "pkill -f minimal-hawk-server" >/dev/null 2>&1 || true
done

for idx in "${!HOSTS[@]}"; do
  host="${HOSTS[$idx]}"
  echo "Starting party ${idx} on ${host} (db_size=${DB_SIZE}, force=${FORCE:-no}, perf=${PERF:-no})..."
  ssh "ec2-user@${host}" "bash ~/run-minimal-server.sh ${idx} ${DB_SIZE} ${FORCE} ${PERF}" 2>&1 | prefix_logs "${host}" &
done

echo "All parties started. Waiting for request to complete..."

# Wait for all background jobs (servers will exit after single request)
wait

echo "All parties finished."

