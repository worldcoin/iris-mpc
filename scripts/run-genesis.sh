#!/usr/bin/env bash

# this script is meant to be called by Dockerfile.genesis.hawk

set -e

export RUST_LOG=info
export SMPC__DATABASE__URL="postgres://postgres:postgres@localhost:5432/SMPC_dev_${NODE_ID}"
export SMPC__CPU_DATABASE__URL="postgres://postgres:postgres@localhost:5432/SMPC_dev_${NODE_ID}"
export SMPC__CPU_DATABASE__MIGRATE=true
export SMPC__ANON_STATS_DATABASE__MIGRATE=true
export SMPC__ANON_STATS_DATABASE__CREATE=true
export SMPC__ANON_STATS_DATABASE__LOAD_PARALLELISM=8
export SMPC__ANON_STATS_DATABASE__URL="postgres://postgres:postgres@localhost:5432/SMPC_dev_${NODE_ID}"
export SMPC__PARTY_ID="${NODE_ID}"
export SMPC__AWS__ENDPOINT="http://127.0.0.1:4566"
export SMPC__REQUESTS_QUEUE_URL="http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/smpcv2-${NODE_ID}-dev.fifo"
export SMPC__HAWK_SERVER_HEALTHCHECK_PORT="300${NODE_ID}"
export AWS_ENDPOINT_URL="http://127.0.0.1:4566"

# exec replaces the shell with the binary
# $@ allows arguments to be forwarded from kubernetes
exec /bin/iris-mpc-hawk-genesis $@

