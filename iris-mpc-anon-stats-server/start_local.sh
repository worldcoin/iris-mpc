#!/bin/bash
set -euo pipefail


cargo run --bin iris-mpc-anon-stats-server -- --addresses 127.0.0.1:7000 --addresses 127.0.0.1:7001 --addresses 127.0.0.1:7002 --party-id 0 --healthcheck-port 8080 &
cargo run --bin iris-mpc-anon-stats-server -- --addresses 127.0.0.1:7000 --addresses 127.0.0.1:7001 --addresses 127.0.0.1:7002 --party-id 1 --healthcheck-port 8081 &
cargo run --bin iris-mpc-anon-stats-server -- --addresses 127.0.0.1:7000 --addresses 127.0.0.1:7001 --addresses 127.0.0.1:7002 --party-id 2 --healthcheck-port 8082 &
