# Upgrade Protocol

Quick local test setup of upgrade protocol:

## Start some DBs

```bash
docker compose up -d
```

will bring up two old dbs on ports 6200 and 6201, and 3 new dbs on ports 6100,6101,6102.

## Fill DBs with test data

```bash
cargo run --release --bin seed-v1-dbs -- --db-urls postgres://postgres:postgres@localhost:6100/postgres --db-urls postgres://postgres:postgres@localhost:6101/postgres --num-elements 10000
```

## Upgrade for left eye

### Run the 3 upgrade servers

Concurrently run:

```bash
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8000 --db-url postgres://postgres:postgres@localhost:6200/postgres --party-id 0 --eye left
```

```bash
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8001 --db-url postgres://postgres:postgres@localhost:6201/postgres --party-id 1 --eye left
```

```bash
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8002 --db-url postgres://postgres:postgres@localhost:6202/postgres --party-id 2 --eye left
```

### Run the 2 upgrade clients

Concurrently run:

```bash
cargo run --release --bin upgrade-client -- --server1 127.0.0.1:8000 --server2 127.0.0.1:8001 --server3 127.0.0.1:8002 --db-start 0 --db-end 10000 --party-id 0 --eye left --db-url postgres://postgres:postgres@localhost:6100/postgres
```

```bash
cargo run --release --bin upgrade-client -- --server1 127.0.0.1:8000 --server2 127.0.0.1:8001 --server3 127.0.0.1:8002 --db-start 0 --db-end 10000 --party-id 1 --eye left --db-url postgres://postgres:postgres@localhost:6101/postgres
```

## Upgrade for right eye

### Run the 3 upgrade servers

Concurrently run:

```bash
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8000 --db-url postgres://postgres:postgres@localhost:6200/postgres --party-id 0 --eye right
```

```bash
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8001 --db-url postgres://postgres:postgres@localhost:6201/postgres --party-id 1 --eye right
```

```bash
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8002 --db-url postgres://postgres:postgres@localhost:6202/postgres --party-id 2 --eye right
```

### Run the 2 upgrade clients

(In practice these DBs would point to different old DBs, we just use the same old DBs for left and right in this example )

Concurrently run:

```bash
cargo run --release --bin upgrade-client -- --server1 127.0.0.1:8000 --server2 127.0.0.1:8001 --server3 127.0.0.1:8002 --db-start 0 --db-end 10000 --party-id 0 --eye right --db-url postgres://postgres:postgres@localhost:6100/postgres
```

```bash
cargo run --release --bin upgrade-client -- --server1 127.0.0.1:8000 --server2 127.0.0.1:8001 --server3 127.0.0.1:8002 --db-start 0 --db-end 10000 --party-id 1 --eye right --db-url postgres://postgres:postgres@localhost:6101/postgres
```

## Check the upgrade was successful

```bash
cargo run --release --bin upgrade-checker -- --num-elements 10000 --db-urls postgres://postgres:postgres@localhost:6100/postgres --db-urls postgres://postgres:postgres@localhost:6101/postgres --db-urls postgres://postgres:postgres@localhost:6100/postgres --db-urls postgres://postgres:postgres@localhost:6101/postgres --db-urls postgres://postgres:postgres@localhost:6200/postgres --db-urls postgres://postgres:postgres@localhost:6201/postgres --db-urls postgres://postgres:postgres@localhost:6202/postgres
```
