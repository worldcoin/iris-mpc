#!/usr/bin/env bash
set -euo pipefail

# Profile the minimal hawk server with samply.
# This script ensures the graph cache exists (to minimize startup overhead)
# then profiles the actual batch processing with samply.
#
# Supports delayed profiling: start the server, wait for startup, then attach samply.

usage() {
  echo "Usage: $0 <party-index 0|1|2> [db-size] [profile-delay-secs] [profile-output-dir]"
  echo ""
  echo "This script profiles the minimal server using samply."
  echo ""
  echo "Arguments:"
  echo "  party-index         Party index (0, 1, or 2). Party 0 is the initiator."
  echo "  db-size             Database size (default: 1000)"
  echo "  profile-delay-secs  Seconds to wait before attaching samply (default: 0)"
  echo "                      Use this to skip profiling startup/connection establishment."
  echo "                      Recommended: 5-10 seconds for larger DBs."
  echo "  profile-output-dir  Directory to save profiles (default: /home/ec2-user/profiles)"
  echo ""
  echo "After profiling, use view-profile-tunnel.sh to view the profile locally."
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

PARTY_INDEX="$1"
DB_SIZE="${2:-1000}"
PROFILE_DELAY="${3:-0}"
PROFILE_DIR="${4:-/home/ec2-user/profiles}"

if [[ "${PARTY_INDEX}" != "0" && "${PARTY_INDEX}" != "1" && "${PARTY_INDEX}" != "2" ]]; then
  echo "party-index must be 0, 1, or 2"
  exit 1
fi

INITIATOR_FLAG=""
if [[ "${PARTY_INDEX}" == "0" ]]; then
  INITIATOR_FLAG="--initiator"
fi

BIN_PATH="/home/ec2-user/minimal-hawk-server"
GRAPH_PATH="/home/ec2-user/graph.bin"
ADDRS="172.31.21.251:16000,172.31.22.229:16000,172.31.24.35:16000"
PROFILE_OUTPUT="${PROFILE_DIR}/profile-party-${PARTY_INDEX}-$(date +%Y%m%d-%H%M%S).json"

# Ensure profile directory exists
mkdir -p "${PROFILE_DIR}"

# Check if samply is installed
if ! command -v samply &> /dev/null; then
  echo "Error: samply is not installed. Install it with: cargo install samply"
  exit 1
fi

# Step 1: Ensure graph cache exists (skip profiling startup/graph building)
if [[ ! -f "${GRAPH_PATH}" ]]; then
  echo "=== Graph cache not found at ${GRAPH_PATH} ==="
  echo "Please run the servers once without profiling to build the graph cache."
  echo "Use: ./run-all-minimal-servers.sh ${DB_SIZE}"
  exit 1
fi

echo ""
echo "=== Starting profiled run ==="
echo "Profile output: ${PROFILE_OUTPUT}"
echo "Profile delay: ${PROFILE_DELAY}s"
echo ""

# Build the server command
SERVER_CMD="${BIN_PATH} \
  --party-index ${PARTY_INDEX} \
  --addresses ${ADDRS} \
  --outbound-addrs ${ADDRS} \
  --hnsw-param-m 256 \
  --hnsw-param-ef-constr 320 \
  --hnsw-param-ef-search 320 \
  --graph-cache-mode load \
  --graph-cache-path ${GRAPH_PATH} \
  --db-size ${DB_SIZE} \
  --single-request \
  ${INITIATOR_FLAG}"

if [[ "${PROFILE_DELAY}" -gt 0 ]]; then
  # Delayed profiling mode: start server, wait, then attach samply
  echo "Starting server in background..."
  ${SERVER_CMD} &
  SERVER_PID=$!
  
  echo "Server PID: ${SERVER_PID}"
  echo "Waiting ${PROFILE_DELAY}s for startup before attaching profiler..."
  sleep "${PROFILE_DELAY}"
  
  # Check if server is still running
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Error: Server exited before profiler could attach."
    echo "Try reducing the delay or check server logs."
    exit 1
  fi
  
  echo "Attaching samply profiler to PID ${SERVER_PID}..."
  
  # Attach samply to the running process
  # samply will profile until the process exits
  # Use high sampling rate (10kHz) to capture short bursts of work
  samply record \
    --save-only \
    --output "${PROFILE_OUTPUT}" \
    --rate 10000 \
    --pid "${SERVER_PID}"
  
  # Wait for server to finish (it should exit after single request)
  wait "${SERVER_PID}" 2>/dev/null || true
  
else
  # Immediate profiling mode: profile entire run (including startup)
  # Use high sampling rate (10kHz) to capture short bursts of work
  echo "Profiling entire run (use profile-delay-secs to skip startup)..."
  samply record \
    --save-only \
    --output "${PROFILE_OUTPUT}" \
    --rate 10000 \
    -- \
    ${SERVER_CMD}
fi

echo ""
echo "=== Profiling complete ==="
echo "Profile saved to: ${PROFILE_OUTPUT}"

# Kill any existing samply server
pkill -f "samply load" 2>/dev/null || true

# # Start samply load in background
# echo ""
# echo "Starting samply server on port 3000..."
# samply load --port 3000 "${PROFILE_OUTPUT}" &
# SAMPLY_PID=$!

# # Cleanup function to kill samply when script exits
# cleanup() {
#   echo ""
#   echo "Stopping samply server..."
#   kill "${SAMPLY_PID}" 2>/dev/null || true
#   wait "${SAMPLY_PID}" 2>/dev/null || true
#   echo "Done."
#   exit 0
# }

# trap cleanup SIGINT SIGTERM EXIT

# # Give it a moment to start
# sleep 1

# if kill -0 "${SAMPLY_PID}" 2>/dev/null; then
#   echo "Samply server running (PID: ${SAMPLY_PID})"
#   echo "If you have a tunnel running (start-profile-tunnel.sh), view at: http://localhost:3000"
#   echo ""
#   echo "Press Ctrl+C to stop."
#   echo ""
  
#   # Wait forever until killed
#   wait "${SAMPLY_PID}"
# else
#   echo "Warning: samply failed to start."
#   exit 1
# fi
