#!/usr/bin/env bash

rm -rf "*.log"

docker-compose down --remove-orphans
docker-compose up -d

sleep 5

cargo run --release --bin seed-v1-dbs -- --side left --shares-db-urls postgres://postgres:postgres@localhost:6100 --shares-db-urls postgres://postgres:postgres@localhost:6101 --masks-db-url postgres://postgres:postgres@localhost:6111 --fill-to 20000 --create --migrate
#cargo run --release --bin seed-v1-dbs -- --side right --shares-db-urls postgres://postgres:postgres@localhost:6200 --shares-db-urls postgres://postgres:postgres@localhost:6201 --masks-db-url postgres://postgres:postgres@localhost:6211 --fill-to 20000 --create --migrate

cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8000 --db-url postgres://postgres:postgres@localhost:6200 --party-id 0 --eye left --environment dev --threads 1 # &> upgrade-server-8000.log &
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8001 --db-url postgres://postgres:postgres@localhost:6201 --party-id 1 --eye left --environment dev --threads 1 # &> upgrade-server-8001.log &
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8002 --db-url postgres://postgres:postgres@localhost:6202 --party-id 2 --eye left --environment dev --threads 1 # &> upgrade-server-8002.log &

#cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8003 --db-url postgres://postgres:postgres@localhost:6200 --party-id 0 --eye right --environment dev --threads 1 &> upgrade-server-8003.log &
#cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8004 --db-url postgres://postgres:postgres@localhost:6201 --party-id 1 --eye right --environment dev --threads 1 &> upgrade-server-8004.log &
#cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8005 --db-url postgres://postgres:postgres@localhost:6202 --party-id 2 --eye right --environment dev --threads 1 &> upgrade-server-8005.log &



cargo run --release --bin upgrade-client -- --server1 127.0.0.1:8000 --server2 127.0.0.1:8001 --server3 127.0.0.1:8002 --db-start 1 --db-end 20000 --party-id 0 --eye left --shares-db-url postgres://postgres:postgres@localhost:6100 --masks-db-url postgres://postgres:postgres@localhost:6111 &> upgrade-client-left-0.log &
cargo run --release --bin upgrade-client -- --server1 127.0.0.1:8000 --server2 127.0.0.1:8001 --server3 127.0.0.1:8002 --db-start 1 --db-end 20000 --party-id 1 --eye left --shares-db-url postgres://postgres:postgres@localhost:6101 --masks-db-url postgres://postgres:postgres@localhost:6111 &> upgrade-client-left-1.log &

#cargo run --release --bin upgrade-client -- --server1 127.0.0.1:8003 --server2 127.0.0.1:8004 --server3 127.0.0.1:8005 --db-start 1 --db-end 20000 --party-id 0 --eye right --shares-db-url postgres://postgres:postgres@localhost:6100 --masks-db-url postgres://postgres:postgres@localhost:6111 &> upgrade-client-right-0.log &
#cargo run --release --bin upgrade-client -- --server1 127.0.0.1:8003 --server2 127.0.0.1:8004 --server3 127.0.0.1:8005 --db-start 1 --db-end 20000 --party-id 1 --eye right --shares-db-url postgres://postgres:postgres@localhost:6101 --masks-db-url postgres://postgres:postgres@localhost:6111 &> upgrade-client-right-1.log &
