# Upgrade Protocol

Quick local test setup of upgrade protocol:

## Start some DBs

```bash
docker compose up -d
```

will bring up two old dbs on ports 6200 and 6201, and 3 new dbs on ports 6100,6101,6102.

## Fill DBs with test data

```bash
cargo run --release --bin seed-v1-dbs -- --side right --shares-db-urls postgres://postgres:postgres@localhost:6100 --shares-db-urls postgres://postgres:postgres@localhost:6101 --masks-db-url postgres://postgres:postgres@localhost:6111 --num-elements 10000
cargo run --release --bin seed-v1-dbs -- --side left --shares-db-urls postgres://postgres:postgres@localhost:6100 --shares-db-urls postgres://postgres:postgres@localhost:6101 --masks-db-url postgres://postgres:postgres@localhost:6111 --num-elements 10000
```

## Upgrade for left eye

### Run the 3 upgrade servers 

Concurrently run:

```bash
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8000 --db-url postgres://postgres:postgres@localhost:6200 --party-id 0 --eye left --environment dev
```

```bash
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8001 --db-url postgres://postgres:postgres@localhost:6201 --party-id 1 --eye left --environment dev
```

```bash
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8002 --db-url postgres://postgres:postgres@localhost:6202 --party-id 2 --eye left --environment dev
```

### Run the 2 upgrade clients

Concurrently run:

```bash
cargo run --release --bin upgrade-client -- --server1 127.0.0.1:8000 --server2 127.0.0.1:8001 --server3 127.0.0.1:8002 --db-start 0 --db-end 10000 --party-id 0 --eye left --shares-db-url postgres://postgres:postgres@localhost:6100 --masks-db-url postgres://postgres:postgres@localhost:6111
cargo run --release --bin upgrade-client -- --server1 127.0.0.1:8000 --server2 127.0.0.1:8001 --server3 127.0.0.1:8002 --db-start 0 --db-end 10000 --party-id 1 --eye left --shares-db-url postgres://postgres:postgres@localhost:6101 --masks-db-url postgres://postgres:postgres@localhost:6111
```

## Upgrade for right eye

### Run the 3 upgrade servers

Concurrently run:

```bash
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8000 --db-url postgres://postgres:postgres@localhost:6200 --party-id 0 --eye right --environment dev
```

```bash
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8001 --db-url postgres://postgres:postgres@localhost:6201 --party-id 1 --eye right --environment dev
```

```bash
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8002 --db-url postgres://postgres:postgres@localhost:6202 --party-id 2 --eye right --environment dev
```

### Run the 2 upgrade clients

(In practice these DBs would point to different old DBs, we just use the same old DBs for left and right in this example )

Concurrently run:

```bash
cargo run --release --bin upgrade-client -- --server1 127.0.0.1:8000 --server2 127.0.0.1:8001 --server3 127.0.0.1:8002 --db-start 0 --db-end 10000 --party-id 0 --eye right --shares-db-url postgres://postgres:postgres@localhost:6100 --masks-db-url postgres://postgres:postgres@localhost:6111
```
```bash
cargo run --release --bin upgrade-client -- --server1 127.0.0.1:8000 --server2 127.0.0.1:8001 --server3 127.0.0.1:8002 --db-start 0 --db-end 10000 --party-id 1 --eye right --shares-db-url postgres://postgres:postgres@localhost:6101 --masks-db-url postgres://postgres:postgres@localhost:6111

```
## Check the upgrade was successful

```bash
cargo run --release --bin upgrade-checker -- --environment dev --num-elements 10000 --db-urls postgres://postgres:postgres@localhost:6100 --db-urls postgres://postgres:postgres@localhost:6101 --db-urls postgres://postgres:postgres@localhost:6111 --db-urls postgres://postgres:postgres@localhost:6200 --db-urls postgres://postgres:postgres@localhost:6201 --db-urls postgres://postgres:postgres@localhost:6202
```
