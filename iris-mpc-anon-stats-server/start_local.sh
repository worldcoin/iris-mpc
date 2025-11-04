#!/bin/bash
set -euo pipefail

docker compose -f docker-compose.dev.yaml up -d

SMPC__DB_URL="postgres://postgres:postgres@localhost:5432/SMPC_dev_0" \
SMPC__RESULTS_TOPIC_ARN="arn:aws:sns:local:000000000000:anon-stats" \
SMPC__AWS__REGION="us-east-1" \
 cargo run --bin iris-mpc-anon-stats-server -p iris-mpc-bins -- --addresses 127.0.0.1:7000 --addresses 127.0.0.1:7001 --addresses 127.0.0.1:7002 --party-id 0 --healthcheck-port 8080 &
pid0=$!
SMPC__DB_URL="postgres://postgres:postgres@localhost:5432/SMPC_dev_1" \
SMPC__RESULTS_TOPIC_ARN="arn:aws:sns:local:000000000000:anon-stats" \
SMPC__AWS__REGION="us-east-1" \
 cargo run --bin iris-mpc-anon-stats-server -p iris-mpc-bins -- --addresses 127.0.0.1:7000 --addresses 127.0.0.1:7001 --addresses 127.0.0.1:7002 --party-id 1 --healthcheck-port 8081 &
pid1=$!
SMPC__DB_URL="postgres://postgres:postgres@localhost:5432/SMPC_dev_2" \
SMPC__RESULTS_TOPIC_ARN="arn:aws:sns:local:000000000000:anon-stats" \
SMPC__AWS__REGION="us-east-1" \
 cargo run --bin iris-mpc-anon-stats-server -p iris-mpc-bins -- --addresses 127.0.0.1:7000 --addresses 127.0.0.1:7001 --addresses 127.0.0.1:7002 --party-id 2 --healthcheck-port 8082 &
pid2=$!
trap "kill $pid0 $pid1 $pid2" SIGINT SIGTERM EXIT 
wait $pid0 $pid1 $pid2
