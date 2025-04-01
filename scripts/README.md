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

### ShadowReadOnly

Script: `./scripts/tools/run-shadowreadonly-server.sh`

This mode represents setup in which the irises are read from the `iris-mpc` production database and graph is written to a shadow / CPU database.

The difference in configuration is the `MODE_OF_DEPLOYMENT=SHADOWREADONLY` and `DISABLE_PERSISTANCE=true`. This allows us to run the system with confidence that data in the `iris-mpc` database will not be altered. 

Also, in this mode the `iris-mpc` database is being accessed via a read-only user. To see details of how it is set up, please refer to `init-db-pgres.sql` script.
### ShadowIsolation

Script: `./scripts/tools/run-shadowisolaton-server.sh`

This mode represents setup in which all the data is read/written to the shadow / CPU database. 
