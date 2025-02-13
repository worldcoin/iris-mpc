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
#
export RUST_LOG=info
export RUST_BACKTRACE=full
export SMPC__ENVIRONMENT=dev
export SMPC__SERVICE__SERVICE_NAME=smpcv2-server-dev
export SMPC__DATABASE__URL="postgres://postgres:postgres@localhost:5432/SMPC_dev_${NODE_ID}"
export SMPC__DATABASE__MIGRATE=true
export SMPC__DATABASE__CREATE=true
export SMPC__DATABASE__LOAD_PARALLELISM=8
export SMPC__PARTY_ID="${NODE_ID}"
export SMPC__MAX_BATCH_SIZE=64
export SMPC__INIT_DB_SIZE=1000
export SMPC__MAX_DB_SIZE=10000
export SMPC__HAWK_REQUEST_PARALLELISM=10
export SMPC__AWS__REGION=us-east-1
export SMPC__AWS__ENDPOINT="http://127.0.0.1:4566"
export SMPC__KMS_KEY_ARNS='["arn:aws:kms:us-east-1:000000000000:key/00000000-0000-0000-0000-000000000000","arn:aws:kms:us-east-1:000000000000:key/00000000-0000-0000-0000-000000000001","arn:aws:kms:us-east-1:000000000000:key/00000000-0000-0000-0000-000000000002"]'
export SMPC__REQUESTS_QUEUE_URL="http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/smpcv2-${NODE_ID}-dev.fifo"
export SMPC__NODE_HOSTNAMES='["127.0.0.1","127.0.0.1","127.0.0.1"]'
export SMPC__SERVICE_PORTS='["4000","4001","4002"]'
export SMPC__HEALTHCHECK_PORTS='["3000","3001","3002"]'
export SMPC__HAWK_SERVER_HEALTHCHECK_PORT="300${NODE_ID}"
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_ENDPOINT_URL="http://127.0.0.1:4566"
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_REGION=us-east-1

if [ "$INIT_SERVERS" = true ]; then
  ./scripts/init-servers.sh
fi


cargo run --bin server-hawk
