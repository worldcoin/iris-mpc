# Reshare Protocol

The aim of the reshare protocol is to allow 2 existing parties in SMPCv2 to work together to recover the share of another party using a simple MPC functionality.

## Internal Server structure

The current internal structure of this service works as follows:

* The receiving party hosts a GRPC server to receive reshare batches from the two sending parties.
* The two sending parties send reshare batches via GRPC.
* The GPRC server collects reshare request batches from the two clients and stores it internally.
* Once matching requests from both parties are collected, the server processes the requests and stores them to the DB.

Currently, the matching is not very robust and requires that both clients send batches for the exact ranges (i.e., client 1 and 2 send batch for ids 1-100, it cannot handle client 1 sending 1-100 and client 2 sending 1-50 and 51-100).

## Example Protocol run

In this example we start a reshare process where parties 0 and 1 are the senders (i.e., clients) and party 2 is the receiver (i.e., server).
The server is TLS aware and uses NGINX with hardcoded certificates for the server and clients.

Clients load certificates and connect directly through NGINX, which then pass-throughs the request to the server

### Create SSH keys and self-signed certificates

To generate the SSH keys and self-signed certificates for the clients, use:

```shell
./ssh_chain.sh
```

This will generate server private keys, self-signed CA roots, and public keys for SSL.
You will also need to edit the `/etc/hosts` file to include the following line:

```shell
127.0.0.1 localhost reshare-server.1.stage.smpcv2.worldcoin.dev reshare-server.2.stage.smpcv2.worldcoin.dev reshare-server.3.stage.smpcv2.worldcoin.dev
```

These will be used by NGNIX, as well as the clients

### Bring up the DBs, localstack, and reshare sever

```bash
docker-compose up
```

### Generate KMS keys

We also need generate some KMS keys to be used by the clients to derive the common seed. Use:

```shell
./aws_local.sh
```

This will output 2 keys ARNs. You will use them in a later step.

### Seed the databases

Here, the seed-v2-dbs binary just creates fully replicated DB for 3 parties, in DBs with ports 6200,6201,6202. Additionally, there is also another DB at 6203, which we will use as a target for the reshare protocol to fill into.

```shell
cargo run --release -p iris-mpc-bins --bin seed-v2-dbs -- --db-url-party1 postgres://postgres:postgres@localhost:6200 --db-url-party2 postgres://postgres:postgres@localhost:6201 --db-url-party3 postgres://postgres:postgres@localhost:6202 --schema-name-party1 SMPC_testing_0 --schema-name-party2 SMPC_testing_1 --schema-name-party3 SMPC_testing_2 --fill-to 10000 --batch-size 100
```

Short rundown of the parameters:

* `db-url-party1`: Postgres connection string for the first party
* `db-url-party2`: Postgres connection string for the second party
* `db-url-party3`: Postgres connection string for the third party
* `schema-name-party1`: Schema name for the first party
* `schema-name-party2`: Schema name for the second party
* `schema-name-party3`: Schema name for the third party
* `fill-to`: Number of entries to fill in the DB
* `batch-size`: Batch size for the inserts

### Start clients for the sending parties

```bash
cd iris-mpc-upgrade/src/bin
cargo run --release -p iris-mpc-bins --bin reshare-client -- --party-id 0 --other-party-id 1 --target-party-id 2 --server-url https://upgrade-left.1.smpcv2.stage.worldcoin.dev:6443 --environment testing --db-url postgres://postgres:postgres@localhost:6200 --db-start 1 --db-end 10001 --batch-size 100 --my-kms-key-arn <kms_key_arn-1> --other-kms-key-arn <kms_key_arn-2> --reshare-run-session-id test --ca-root-file-path nginx/cert/ca.txt
```

```bash
cd iris-mpc-upgrade/src/bin
cargo run --release -p iris-mpc-bins --bin reshare-client -- --party-id 1 --other-party-id 0 --target-party-id 2 --server-url https://upgrade-left.2.smpcv2.stage.worldcoin.dev:6443 --environment testing --db-url postgres://postgres:postgres@localhost:6200 --db-start 1 --db-end 10001 --batch-size 100 --my-kms-key-arn <kms_key_arn-1> --other-kms-key-arn <kms_key_arn-2> --reshare-run-session-id test --ca-root-file-path nginx/cert/ca.txt
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
* `my-kms-key-arn`: ARN of the KMS key to use for the common seed derivation
* `other-kms-key-arn`: ARN of the KMS key to use for the common seed derivation
* `ca-root-file-path`: Path to the CA Root TLS certificate
* `reshare-run-sessin-id`: a random string to identify the current reshare run

### Checking results

Since the shares on a given shamir poly are deterministic given the party ids, the above upgrade process can be checked by comparing the databases at port 6202 and 6203 for equality.
