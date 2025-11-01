#!/bin/bash
set -euo pipefail


cargo run --bin iris-mpc-anon-stats-server -p iris-mpc-bins -- --addresses 127.0.0.1:7000 --addresses 127.0.0.1:7001 --addresses 127.0.0.1:7002 --party-id 0 --healthcheck-port 8080 &
pid0=$!
cargo run --bin iris-mpc-anon-stats-server -p iris-mpc-bins -- --addresses 127.0.0.1:7000 --addresses 127.0.0.1:7001 --addresses 127.0.0.1:7002 --party-id 1 --healthcheck-port 8081 &
pid1=$!
cargo run --bin iris-mpc-anon-stats-server -p iris-mpc-bins -- --addresses 127.0.0.1:7000 --addresses 127.0.0.1:7001 --addresses 127.0.0.1:7002 --party-id 2 --healthcheck-port 8082 &
pid2=$!
trap "kill $pid0 $pid1 $pid2" SIGINT SIGTERM EXIT 
wait $pid0 $pid1 $pid2
