# Scripts

This directory contains runtime scripts for local development and testing.

`tools` subdirectory has been introduced to isolate the fundamental scripts used for development and testing from some tooling / debug scripts.

## Cleanup of environment

Between running the server in different modes, it is recommended to clean up the environment. This can be done by running:

```bash
docker compose -f docker-compose.dev.yaml down
docker compose -f docker-compose.dev.yaml up -d
```

## Running server in different deployment modes

### Standard

This is used currently for local only. It is the default mode of operation, in which the server reads/writes data to the CPU database (both irises and graph data)

### Genesis Local Testing

First, you need to generate some test data for the Genesis mode. This is done by running the following command:

```bash
cargo run --release -p iris-mpc-bins --bin generate-benchmark-data
```

```bash
docker compose -f docker-compose.test.genesis.yaml up init_db
```

In another terminal, run:

```shell
SMPC__HNSW_SCHEMA_NAME_SUFFIX=_hnsw GENESIS_MAX_HEIGHT=100 ./scripts/run-server.sh 0 genesis
SMPC__HNSW_SCHEMA_NAME_SUFFIX=_hnsw GENESIS_MAX_HEIGHT=100 ./scripts/run-server.sh 1 genesis
SMPC__HNSW_SCHEMA_NAME_SUFFIX=_hnsw GENESIS_MAX_HEIGHT=100 ./scripts/run-server.sh 2 genesis
```
