#!/usr/bin/env bash

rm -rf "*.log"

docker-compose down --remove-orphans
docker-compose up -d

sleep 5

cargo run --release --bin seed-v2-dbs -- --db-url-party1 postgres://postgres:postgres@localhost:6200 --db-url-party2 postgres://postgres:postgres@localhost:6201 --db-url-party3 postgres://postgres:postgres@localhost:6202 --schema-party1 SMPC_testing_0 --schema-party2 SMPC_testing_1 --schema-party3 SMPC_testing_2 --fill-to 10000
