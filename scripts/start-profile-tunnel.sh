#!/usr/bin/env bash
set -euo pipefail

# Start a persistent SSH tunnel for viewing samply profiles.
# Run this once, then use run-all-minimal-servers-profile.sh which will
# automatically start samply load after profiling completes.
#
# Usage: ./start-profile-tunnel.sh [server-host] [local-port] [remote-port]

SERVER_HOST="${1:-aws0}"
LOCAL_PORT="${2:-3000}"
REMOTE_PORT="${3:-3000}"

echo "=== Starting Persistent Profile Tunnel ==="
echo "Server: ${SERVER_HOST}"
echo "Local port: ${LOCAL_PORT} -> Remote port: ${REMOTE_PORT}"
echo ""

# Check if the local port is already in use
if lsof -i ":${LOCAL_PORT}" &>/dev/null; then
  echo "Port ${LOCAL_PORT} is already in use."
  echo "If it's an existing tunnel, you're good to go!"
  echo "Otherwise, kill it with: lsof -ti :${LOCAL_PORT} | xargs kill -9"
  exit 0
fi

echo "Tunnel will stay open until you press Ctrl+C."
echo "View profiles at: http://localhost:${LOCAL_PORT}"
echo ""

cleanup() {
  echo ""
  echo "Tunnel closed."
  exit 0
}

trap cleanup SIGINT SIGTERM

# Start SSH tunnel in foreground (just port forwarding, no remote command)
# -N = no remote command
# -L = local port forwarding
ssh -N -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" "ec2-user@${SERVER_HOST}"

