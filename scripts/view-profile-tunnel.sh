#!/usr/bin/env bash
set -euo pipefail

# View samply profiles from a remote server by tunneling the port locally.
# 
# Usage:
#   ./view-profile-tunnel.sh <server-host> [profile-file] [local-port] [remote-port]
#
# This script:
#   1. SSHs into the server and starts samply load on the profile file
#   2. Sets up port forwarding so you can view the profile at http://localhost:<local-port>
#   3. Opens your browser to the profile viewer

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <server-host> [profile-file] [local-port] [remote-port]"
  echo ""
  echo "Arguments:"
  echo "  server-host   SSH host (e.g., ec2-user@your-server.com or use ~/.ssh/config alias)"
  echo "  profile-file  Path to profile on the server (default: latest in /home/ec2-user/profiles/)"
  echo "  local-port    Local port to bind (default: 3000)"
  echo "  remote-port   Remote port samply uses (default: 3000)"
  echo ""
  echo "Examples:"
  echo "  $0 mpc-server-0"
  echo "  $0 mpc-server-0 /home/ec2-user/profiles/profile-party-0-20250122-120000.json"
  echo "  $0 mpc-server-0 latest 8080"
  echo ""
  echo "The script will start samply on the server and tunnel the port to your local machine."
  echo "Open http://localhost:<local-port> in your browser to view the profile."
  exit 1
fi

SERVER_HOST="$1"
PROFILE_FILE="${2:-latest}"
LOCAL_PORT="${3:-3000}"
REMOTE_PORT="${4:-3000}"

PROFILE_DIR="/home/ec2-user/profiles"

echo "=== Samply Profile Viewer Tunnel ==="
echo "Server: ${SERVER_HOST}"
echo "Local port: ${LOCAL_PORT}"
echo "Remote port: ${REMOTE_PORT}"
echo ""

# Build the command to find and load the profile
if [[ "${PROFILE_FILE}" == "latest" ]]; then
  FIND_CMD="ls -t ${PROFILE_DIR}/*.json 2>/dev/null | head -1"
  PROFILE_CMD="PROFILE=\$(${FIND_CMD}); if [[ -z \"\${PROFILE}\" ]]; then echo 'No profiles found in ${PROFILE_DIR}'; exit 1; fi; echo \"Loading: \${PROFILE}\"; samply load --port ${REMOTE_PORT} \"\${PROFILE}\""
else
  PROFILE_CMD="echo \"Loading: ${PROFILE_FILE}\"; samply load --port ${REMOTE_PORT} \"${PROFILE_FILE}\""
fi

echo "Starting samply on remote server and setting up tunnel..."
echo "Press Ctrl+C to stop the tunnel and exit."
echo ""
echo "Once connected, open: http://localhost:${LOCAL_PORT}"
echo ""

# Check if the local port is already in use
if lsof -i ":${LOCAL_PORT}" &>/dev/null; then
  echo "Warning: Port ${LOCAL_PORT} is already in use locally."
  echo "Either stop the process using that port or specify a different local port."
  exit 1
fi

# Track child processes for cleanup
SSH_PID=""
BROWSER_PID=""

cleanup() {
  echo ""
  echo "Cleaning up..."
  
  # Kill the browser opener if still running
  if [[ -n "${BROWSER_PID}" ]] && kill -0 "${BROWSER_PID}" 2>/dev/null; then
    kill "${BROWSER_PID}" 2>/dev/null || true
  fi
  
  # Kill samply on the remote server
  ssh "${SERVER_HOST}" "pkill -f 'samply load'" 2>/dev/null || true
  
  # Kill SSH if still running
  if [[ -n "${SSH_PID}" ]] && kill -0 "${SSH_PID}" 2>/dev/null; then
    kill "${SSH_PID}" 2>/dev/null || true
    wait "${SSH_PID}" 2>/dev/null || true
  fi
  
  echo "Done."
  exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Function to open browser (cross-platform)
open_browser() {
  local url="http://localhost:${LOCAL_PORT}"
  sleep 3  # Wait for server to start
  
  if command -v open &>/dev/null; then
    # macOS
    open "${url}"
  elif command -v xdg-open &>/dev/null; then
    # Linux
    xdg-open "${url}"
  else
    echo "Browser not auto-opened. Please navigate to: ${url}"
  fi
}

# Start browser opener in background
google-chrome &
BROWSER_PID=$!

# SSH with port forwarding and run samply
# -L forwards local port to remote port
# -tt forces pseudo-terminal allocation (better signal handling)
ssh -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" -tt "${SERVER_HOST}" "${PROFILE_CMD}" &
SSH_PID=$!

# Wait for SSH to finish
wait "${SSH_PID}" 2>/dev/null || true
