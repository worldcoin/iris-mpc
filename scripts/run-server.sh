#!/usr/bin/env bash
set -e

NODE_ID="$1"
if [ -z "$NODE_ID" ]; then
  echo "Usage: run-server.sh <node_id>"
  exit 1
fi

INIT_SERVERS=false
if [ "$2" == "--init-servers" ]; then
  INIT_SERVERS=true
fi

export SMPC__DATABASE__URL="postgres://postgres:postgres@localhost:5432/SMPC_dev_${NODE_ID}"
export SMPC__PARTY_ID="${NODE_ID}"
export SMPC__AWS__ENDPOINT="http://127.0.0.1:4566"
export SMPC__REQUESTS_QUEUE_URL="http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/smpcv2-${NODE_ID}-dev.fifo"
export SMPC__NODE_HOSTNAMES='["127.0.0.1","127.0.0.1","127.0.0.1"]'
export SMPC__HAWK_SERVER_HEALTHCHECK_PORT="300${NODE_ID}"
export AWS_ENDPOINT_URL="http://127.0.0.1:4566"

if [ "$INIT_SERVERS" = true ]; then
  ./scripts/init-servers.sh
fi


cargo run --bin server-hawk
