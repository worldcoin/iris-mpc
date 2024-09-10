# Upgrade Protocol

Quick local test setup of upgrade protocol:

## Start some DBs

```bash
docker compose up -d
```

will bring up two old dbs on ports 6200 and 6201, and 3 new dbs on ports 6100,6101,6102.

## Fill DBs with test data

```bash
cargo run --release --bin seed-v1-dbs -- --shares-db-urls postgres://postgres:postgres@localhost:6100/shares --shares-db-urls postgres://postgres:postgres@localhost:6101/shares --masks-db-url postgres://postgres:postgres@localhost:6100/masks --num-elements 10000
```

## Generate Self-Signed certificates for Client-Server communication

```bash
cargo run --release --bin gen_certs -- --sans localhost --sans party0 --key-path key0.pem --cert-path cert0.pem
cargo run --release --bin gen_certs -- --sans localhost --sans party1 --key-path key1.pem --cert-path cert1.pem
cargo run --release --bin gen_certs -- --sans localhost --sans party2 --key-path key2.pem --cert-path cert2.pem
```

## Upgrade for left eye

### Run the 3 upgrade servers

Concurrently run:

```bash
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8000 --key key0.pem --cert-chain cert0.pem --db-url postgres://postgres:postgres@localhost:6200/postgres --party-id 0 --eye left
```

```bash
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8001 --key key1.pem --cert-chain cert1.pem --db-url postgres://postgres:postgres@localhost:6201/postgres --party-id 1 --eye left
```

```bash
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8002 --key key2.pem --cert-chain cert2.pem --db-url postgres://postgres:postgres@localhost:6202/postgres --party-id 2 --eye left
```

### Run the 2 upgrade clients

Concurrently run:

```bash
cargo run --release --bin upgrade-client -- --server1 localhost:8000 --server2 localhost:8001 --server3 localhost:8002 --trusted-cert cert0.pem --trusted-cert cert1.pem --trusted-cert cert2.pem --db-start 1 --db-end 10001 --party-id 0 --eye left --shares-db-url postgres://postgres:postgres@localhost:6100/shares --masks-db-url postgres://postgres:postgres@localhost:6100/masks
```

```bash
cargo run --release --bin upgrade-client -- --server1 localhost:8000 --server2 localhost:8001 --server3 localhost:8002 --trusted-cert cert0.pem --trusted-cert cert1.pem --trusted-cert cert2.pem --db-start 1 --db-end 10001 --party-id 1 --eye left --shares-db-url postgres://postgres:postgres@localhost:6101/shares --masks-db-url postgres://postgres:postgres@localhost:6100/masks
```

## Upgrade for right eye

### Run the 3 upgrade servers

Concurrently run:

```bash
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8000 --key key0.pem --cert-chain cert0.pem --db-url postgres://postgres:postgres@localhost:6200/postgres --party-id 0 --eye right
```

```bash
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8001 --key key1.pem --cert-chain cert1.pem --db-url postgres://postgres:postgres@localhost:6201/postgres --party-id 1 --eye right
```

```bash
cargo run --release --bin upgrade-server -- --bind-addr 127.0.0.1:8002 --key key2.pem --cert-chain cert2.pem --db-url postgres://postgres:postgres@localhost:6202/postgres --party-id 2 --eye right
```

### Run the 2 upgrade clients

(In practice these DBs would point to different old DBs, we just use the same old DBs for left and right in this example )

Concurrently run:

```bash
cargo run --release --bin upgrade-client -- --server1 localhost:8000 --server2 localhost:8001 --server3 localhost:8002 --trusted-cert cert0.pem --trusted-cert cert1.pem --trusted-cert cert2.pem --db-start 1 --db-end 10001 --party-id 0 --eye right --shares-db-url postgres://postgres:postgres@localhost:6100/shares --masks-db-url postgres://postgres:postgres@localhost:6100/masks
```

```bash
cargo run --release --bin upgrade-client -- --server1 localhost:8000 --server2 localhost:8001 --server3 localhost:8002 --trusted-cert cert0.pem --trusted-cert cert1.pem --trusted-cert cert2.pem --db-start 1 --db-end 10001 --party-id 1 --eye right --shares-db-url postgres://postgres:postgres@localhost:6101/shares --masks-db-url postgres://postgres:postgres@localhost:6100/masks
```

## Check the upgrade was successful

```bash
cargo run --release --bin upgrade-checker -- --num-elements 10000 --db-urls postgres://postgres:postgres@localhost:6100/shares --db-urls postgres://postgres:postgres@localhost:6101/shares --db-urls postgres://postgres:postgres@localhost:6100/masks --db-urls postgres://postgres:postgres@localhost:6100/shares --db-urls postgres://postgres:postgres@localhost:6101/shares --db-urls postgres://postgres:postgres@localhost:6100/masks --db-urls postgres://postgres:postgres@localhost:6200/postgres --db-urls postgres://postgres:postgres@localhost:6201/postgres --db-urls postgres://postgres:postgres@localhost:6202/postgres
```
