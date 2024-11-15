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

# Reshare Protocol

The aim of the reshare protocol is to allow 2 existing parties in SMPCv2 to work together to recover the share of another party using a simple MPC functionality.

## Example Protocol run

In this example we start a reshare process where parties 0 and 1 are the senders (i.e., clients) and party 2 is the receiver (i.e., server).

### Bring up some DBs and seed them

Here, the seed-v2-dbs binary just creates fully replicated DB for 3 parties, in DBs with ports 6200,6201,6202. Additionally, there is also another DB at 6204, which we will use as a target for the reshare protocol to fill into.

```bash
docker-compose up -d
cargo run --release --bin seed-v2-dbs -- --db-url-party1 postgres://postgres:postgres@localhost:6200 --db-url-party2 postgres://postgres:postgres@localhost:6201 --db-url-party3 postgres://postgres:postgres@localhost:6202 --schema-name-party1 SMPC_testing_0 --schema-name-party2 SMPC_testing_1 --schema-name-party3 SMPC_testing_2 --fill-to 10000 --batch-size 100
```

### Start a server for the receiving party

```bash
cargo run --release --bin reshare-server -- --party-id 2 --sender1-party-id 0 --sender2-party-id 1 --bind-addr 0.0.0.0:7000 --environment testing --db-url postgres://postgres:postgres@localhost:6203 --db-start 1 --db-end 10001 --batch-size 100
```

Short rundown of the parameters:

* `party-id`: the 0-indexed party id of the receiving party. This corresponds to the (i+1)-th point on the exceptional sequence for Shamir poly evaluation
* `sender1-party-id`: The party id of the first sender, just for sanity checks against received packets. (Order between sender1 and sender2 does not matter here)
* `sender2-party-id`: The party id of the second sender, just for sanity checks against received packets.
* `bind-addr`: Socket addr to bind to for gGRPC server.
* `environment`: Which environment are we running in, used for DB schema name
* `db-url`: Postgres connection string. We save the results in this DB
* `db-start`: Expected range of DB entries to receive, just used for sanity checks. Start is inclusive.
* `db-end`: Expected range of DB entries to receive, just used for sanity checks. End is exclusive.
* `batch-size`: maximum size of received reshare batches

### Start clients for the sending parties

```bash
cargo run --release --bin reshare-client -- --party-id 0 --other-party-id 1 --target-party-id 2 --server-url http://localhost:7000 --environment testing --db-url postgres://postgres:postgres@localhost:6200 --db-start 1 --db-end 10001 --batch-size 100
```

```bash
cargo run --release --bin reshare-client -- --party-id 1 --other-party-id 0 --target-party-id 2 --server-url http://localhost:7000 --environment testing --db-url postgres://postgres:postgres@localhost:6201 --db-start 1 --db-end 10001 --batch-size 100
```

Short rundown of the parameters:

* `party-id`: the 0-indexed party id of our own client party. This corresponds to the (i+1)-th point on the exceptional sequence for Shamir poly evaluation
* `other-party-id`: the 0-indexed party id of the other client party. This needs to be passed for the correct calculation of lagrange interpolation polynomials.
* `target-party-id`: the 0-indexed party id of the receiving party. This needs to be passed for the correct calculation of lagrange interpolation polynomials.
* `server-url`: Url where to reach the GRPC server (can also be https, client supports both).
* `environment`: Which environment are we running in, used for DB schema name
* `db-url`: Postgres connection string. We load our shares from this DB
* `db-start`: Range of DB entries to send. Start is inclusive.
* `db-end`: Range of DB entries to send. End is exclusive.
* `batch-size`: maximum size of sent reshare batches

### Checking results

Since the shares on a given shamir poly are deterministic given the party ids, the above upgrade process can be checked by comparing the databases at port 6202 and 6203 for equality.
