#!/usr/bin/env bash

rm -rf "*.log"

docker-compose down --remove-orphans
docker-compose up -d

sleep 5

cargo build --release --bin seed-v2-dbs --bin reshare-server --bin reshare-client

TARGET_DIR=$(cargo metadata --format-version 1 | jq ".target_directory" -r)

$TARGET_DIR/release/seed-v2-dbs --db-url-party1 postgres://postgres:postgres@localhost:6200 --db-url-party2 postgres://postgres:postgres@localhost:6201 --db-url-party3 postgres://postgres:postgres@localhost:6202 --schema-name-party1 SMPC_testing_0 --schema-name-party2 SMPC_testing_1 --schema-name-party3 SMPC_testing_2 --fill-to 10000 --batch-size 100

$TARGET_DIR/release/reshare-server --party-id 2 --sender1-party-id 0 --sender2-party-id 1 --bind-addr 0.0.0.0:7000 --environment testing --db-url postgres://postgres:postgres@localhost:6203 --db-start 1 --db-end 10001 --batch-size 100 & > reshare-server.log

sleep 5

$TARGET_DIR/release/reshare-client --party-id 0 --other-party-id 1 --target-party-id 2 --server-url http://localhost:7000 --environment testing --db-url postgres://postgres:postgres@localhost:6200 --db-start 1 --db-end 10001 --batch-size 100 & > reshare-client-0.log

$TARGET_DIR/release/reshare-client --party-id 1 --other-party-id 0 --target-party-id 2 --server-url http://localhost:7000 --environment testing --db-url postgres://postgres:postgres@localhost:6201 --db-start 1 --db-end 10001 --batch-size 100 & > reshare-client-1.log
