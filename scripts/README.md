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

### Exact CPU linear scan with LocalStack

Build and run the separate linear-scan integration target, then execute the
existing encrypted-request client against all three participants:

```bash
docker build -f Dockerfile.dev.linear-scan -t linear-scan-server-local-build:latest .
docker compose -f docker-compose.test.linear-scan.yaml up -d --wait
docker compose -f docker-compose.test.linear-scan.yaml exec iris_mpc_linear_scan_client ./run-client-docker.sh
docker compose -f docker-compose.test.linear-scan.yaml down -v
```

This target uses the GPU-compatible `SMPC__DATABASE` contract expected by the
exact CPU scanner. It is additive and does not change the Hawk Compose target.

### Standard

This is used currently for local only. It is the default mode of operation, in which the server reads/writes data to the CPU database (both irises and graph data)

### Genesis Local Testing

First, you need to generate some test data for the Genesis mode. This is done by running the following command:

```bash
cd iris-mpc-bins && cargo run --release --bin construct-graph-ptxt -- --job-spec resources/iris-mpc-cpu/construct_graph_ptxt_benchmark.toml
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
