# Running Service Client Utility

The `service-client` tool supports end-to-end system testing by dispatching various traffic patterns into the system. Patterns differ in terms of time (short vs extended bursts) and/or content (different request types).

### Environments

- **dev-dkr**: Local dockerised development environment (LocalStack)
- **dev-stg**: Remote AWS staged development environment

## Prerequisites

### AWS Profile

Add the relevant profile to your `~/.aws/config`. Reference templates are in `scripts/iris-mpc-utils/service-client/profiles/`.

For **dev-dkr** (LocalStack), add the `worldcoin-dev-dkr` profile and set credentials to `test`/`test` in `~/.aws/credentials`.

For **dev-stg** (real AWS), add the `worldcoin-smpcv-io-vpc-dev` profile with your real credentials.

### DatadogHQ Account

- Obtain credentials from TFH
- Verify credentials @ [Datadog login page](https://app.datadoghq.com/account/login?redirect=f)

### AWS Management Console Account

- Obtain credentials from TFH
- Login to [AWS Management Console](https://aws.amazon.com/console/)
- Create Access Key via IAM > Security credentials > Create access key > CLI

## Usage

```
scripts/iris-mpc-utils/service-client/run.sh [-e ENV] [EXEC_OPTS_TOML]
```

### Examples

```bash
# Default: dev-dkr environment, simple-1.toml request batch
./run.sh

# Staging environment
./run.sh -e dev-stg

# Custom request batch
./run.sh requests/complex-1.toml

# Staging with custom request batch
./run.sh -e dev-stg path/to/custom.toml

# Help
./run.sh -h
```

## Request Batch Examples

Pre-built request batches are in `scripts/iris-mpc-utils/service-client/requests/`:

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
path_to_ndjson_file = "YOUR-WORKING-DIR/iris-mpc/iris-mpc-utils/assets/iris-codes-plaintext/20250710-1k.ndjson"
rng_seed = 42
selection_strategy = "All"
```
