#!/usr/bin/env bash
set -e

# Arg :: Node ID :: MPC node ordinal identifier.
NODE_ID="$1"
if [ -z "$NODE_ID" ]; then
  echo "Usage: run-anon-stats-server.sh <node_id>"
  exit 1
fi
export RUST_LOG=info
export SMPC__DB_URL="postgres://postgres:postgres@localhost:5432/SMPC_dev_${NODE_ID}"
export SMPC__POLL_INTERVAL_SECS="10"
export SMPC__PARTY_ID="${NODE_ID}"
export SMPC__AWS__ENDPOINT="http://127.0.0.1:4566"
export SMPC__RESULTS_TOPIC_ARN="arn:aws:sns:us-east-1:000000000000:iris-mpc-results.fifo"
export SMPC__ADDRESSES='["127.0.0.1:7001","127.0.0.1:7002","127.0.0.1:7003"]'
export SMPC__HEALTHCHECK_PORT="500${NODE_ID}"
export AWS_ENDPOINT_URL="http://127.0.0.1:4566"

# Set the stack size to 100MB to receive large messages.
export RUST_MIN_STACK=104857600

cargo run --release -p iris-mpc-bins --bin iris-mpc-anon-stats-server -- --party-id "${NODE_ID}"
