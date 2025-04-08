#!/usr/bin/env bash
set -e

NODE_ID="$1"
if [ -z "$NODE_ID" ]; then
  echo "Usage: run-shadowisolation-server.sh <node_id>"
  exit 1
fi

INIT_SERVERS=false
if [ "$2" == "--init-servers" ]; then
  INIT_SERVERS=true
fi

export RUST_LOG=info
export SMPC__CPU_DATABASE__URL="postgres://postgres:postgres@localhost:5433/SMPC_dev_${NODE_ID}"
export SMPC__CPU_DATABASE__MIGRATE="true"
export SMPC__CPU_DATABASE__CREATE="true"
export SMPC__PARTY_ID="${NODE_ID}"
export SMPC__AWS__ENDPOINT="http://127.0.0.1:4566"
export SMPC__REQUESTS_QUEUE_URL="http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/smpcv2-${NODE_ID}-dev.fifo"
export SMPC__NODE_HOSTNAMES='["127.0.0.1","127.0.0.1","127.0.0.1"]'
export SMPC__HAWK_SERVER_HEALTHCHECK_PORT="300${NODE_ID}"
export SMPC__MODE_OF_COMPUTE="CPU"
export SMPC__MODE_OF_DEPLOYMENT="SHADOWISOLATION"
export AWS_ENDPOINT_URL="http://127.0.0.1:4566"

if [ "$INIT_SERVERS" = true ]; then
  ./scripts/tools/init-servers.sh
fi

# Set the stack size to 100MB to receive large messages.
export RUST_MIN_STACK=104857600

cargo run --bin iris-mpc-hawk
