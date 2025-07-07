## Genesis: local docker

To run dockerised genesis nodes against test data:

```
# Build MPC node docker images.
mpctl-dkr-build-images

# Spin up services (localstack + postgres).
mpctl-dkr-services-up

# Initialise dB in readiness for genesis.
mpctl-dkr-init-db-for-genesis

# Spin up MPC genesis nodes.
mpctl-dkr-net-up-genesis

# View MPC node logs.
mpctl-dkr-node-view-logs-genesis node=0
mpctl-dkr-node-view-logs-genesis node=1
mpctl-dkr-node-view-logs-genesis node=2
```

To re-run dockerised genesis nodes:

```
# Tear down MPC genesis nodes.
mpctl-dkr-net-down-genesis

# Truncate CPU tables.
mpctl-job-pgres-truncate-cpu-tables

# Rebuild MPC node docker images.
mpctl-dkr-build-images

# Spin up MPC genesis nodes.
mpctl-dkr-net-up-genesis

# View MPC node logs.
mpctl-dkr-node-view-logs-genesis node=0
mpctl-dkr-node-view-logs-genesis node=1
mpctl-dkr-node-view-logs-genesis node=2
```

## Genesis: local baremetal

To run baremetal genesis nodes against test data:

```
# Spin up services (localstack + postgres).
mpctl-dkr-services-up

# Initialise dB in readiness for genesis.
mpctl-job-pgres-init-from-plain-text-iris-file

# Setup local MPC network assets.
mpctl-local-net-setup

# Start MPC genesis network.
mpctl-local-net-start-genesis

# View MPC node logs.
mpctl-local-node-view-logs node=0 filter="HNSW-GENESIS"
mpctl-local-node-view-logs node=1 filter="HNSW-GENESIS"
mpctl-local-node-view-logs node=2 filter="HNSW-GENESIS"
```

To re-run baremetal genesis nodes:

```
# Reset MPC genesis nodes.
mpctl-local-net-reset

# Truncate CPU tables.
mpctl-job-pgres-truncate-cpu-tables

# Start MPC genesis network.
mpctl-local-net-start-genesis

# View MPC node logs.
mpctl-local-node-view-logs node=0 filter="HNSW-GENESIS"
mpctl-local-node-view-logs node=1 filter="HNSW-GENESIS"
mpctl-local-node-view-logs node=2 filter="HNSW-GENESIS"
```
