#!/usr/bin/env bash
set -e

# Arg :: Node ID :: MPC node ordinal identifier.
NODE_ID="$1"
if [ -z "$NODE_ID" ]; then
  echo "Usage: run-server.sh <node_id> <binary> [--init-servers]"
  exit 1
fi

# Arg :: Binary :: Binary to run [standard | genesis].
BINARY="$2"
if [ -z "$BINARY" ]; then
  echo "Usage: run-server.sh <node_id> <binary> [--init-servers]"
  exit 1
fi

INIT_SERVERS=false
if [ "$3" == "--init-servers" ]; then
  INIT_SERVERS=true
fi

export RUST_LOG=info
export SMPC__DATABASE__URL="postgres://postgres:postgres@localhost:5432/SMPC_dev_${NODE_ID}"
export SMPC__CPU_DATABASE__URL="postgres://postgres:postgres@localhost:5432/SMPC_dev_${NODE_ID}"
export SMPC__CPU_DATABASE__MIGRATE=true
export SMPC__PARTY_ID="${NODE_ID}"
export SMPC__AWS__ENDPOINT="http://127.0.0.1:4566"
export SMPC__REQUESTS_QUEUE_URL="http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/smpcv2-${NODE_ID}-dev.fifo"
export SMPC__NODE_HOSTNAMES='["127.0.0.1","127.0.0.1","127.0.0.1"]'
export SMPC__SERVICE_PORTS='["4000","4001","4002"]'
export SMPC__HAWK_SERVER_HEALTHCHECK_PORT="300${NODE_ID}"
export AWS_ENDPOINT_URL="http://127.0.0.1:4566"

if [ "$INIT_SERVERS" = true ]; then
  ./scripts/tools/init-servers.sh
fi

# Set the stack size to 100MB to receive large messages.
export RUST_MIN_STACK=104857600


if [ "$BINARY" == "genesis" ]; then
    cargo run --release -p iris-mpc-bins --bin iris-mpc-hawk-genesis -- --max-height "${GENESIS_MAX_HEIGHT}" --perform-snapshot=false
else
    cargo run --release -p iris-mpc-bins --bin iris-mpc-hawk
fi
