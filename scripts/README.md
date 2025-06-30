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