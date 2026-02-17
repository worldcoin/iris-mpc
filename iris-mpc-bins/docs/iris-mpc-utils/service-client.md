# Running Service Client Utility

The `service-client` tool supports end-to-end system testing by dispatching various traffic patterns into the system. Patterns differ in terms of time (short vs extended bursts) and/or content (different request types).

### Environments

- **dev-dkr**: Local dockerised development environment (LocalStack)
- **dev-stg**: Remote AWS staged development environment

## Prerequisites

### AWS Profile

For **dev-dkr** (LocalStack), add the following to `~/.aws/config` and `~/.aws/credentials`:

```ini
# ~/.aws/config
[profile worldcoin-dev-dkr]
region = us-east-1
...
[profile worldcoin-smpcv-io-vpc-dev-dkr]
source_profile = worldcoin-dev-dkr
role_arn = arn:aws:iam::238407200320:role/ampc-hnsw-developer-role


# ~/.aws/credentials
[worldcoin-dev-dkr]
aws_access_key_id = test
aws_secret_access_key = test
```

For **dev-stg** (real AWS), the client only needs the VPC profile (`worldcoin-smpcv-io-vpc-dev`). See [deploy/dev/README.md](../../deploy/dev/README.md) for full AWS account setup.

## Usage

```
scripts/run-service-client.sh [-e ENV] [-i IRIS_SHARES] [TOML_FILE]
```

### Examples

```bash
# Default: dev-dkr environment, simple-1.toml request batch
./scripts/run-service-client.sh

# Staging environment
./scripts/run-service-client.sh -e dev-stg

# Custom request batch
./scripts/run-service-client.sh -i 20250710-1k.ndjson complex-1.toml

# Staging with custom request batch
./scripts/run-service-client.sh -e dev-stg -i 20250710-1k.ndjson complex-1.toml

# Help
./scripts/run-service-client.sh -h
```

## Request Batch Examples

Pre-built request batches are in `iris-mpc-utils/assets/service-client/`:

| File | Description |
|------|-------------|
| `simple-1.toml` | Uniqueness requests |
| `simple-2.toml` | Reauth requests |
| `simple-3.toml` | Reset update requests |
| `simple-4.toml` | Identity deletion requests |
| `simple-5.toml` | Reset check requests |
| `complex-1.toml` | Multi-operation batch with dependencies |

### Custom request batch: computed shares

```toml
[request_batch.Simple]
batch_count = 10
batch_size = 10
batch_kind = "uniqueness"

[shares_generator.FromCompute]
rng_seed = 42
```

### Custom request batch: shares from file

```toml
[request_batch.Simple]
batch_count = 10
batch_size = 10
batch_kind = "uniqueness"

[shares_generator.FromFile]
# this path can be overridden by providing a .ndjson file to run-service-client.sh.
path_to_ndjson_file = "YOUR-WORKING-DIR/iris-mpc/iris-mpc-utils/assets/iris-codes-plaintext/20250710-1k.ndjson"
rng_seed = 42
selection_strategy = "All"
```
