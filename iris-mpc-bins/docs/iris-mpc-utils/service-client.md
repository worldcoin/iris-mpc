# Running Service Client Utility

A new HNSW tool, named `service-client` has been developed to support end-to-end system testing.  The tool's design intent is to support the dispatch of various traffic patterns into the system.  Such patterns will differ in terms of time (short vs extended bursts) and/or content (different request types).  

### Environments

The tool is designed to be used in the following environments:

- *dev-dkr*: Local dockerised development environment

- *dev-stg*: Remove AWS staged development environment

## Step 1: Setup Datadog & AWS accounts

### DatadogHQ Account

- Obtain credentials from TFH
- Verify credentials @ Datadog login page
    - [https://app.datadoghq.com/account/login](https://app.datadoghq.com/account/login?redirect=f)

### AWS Management Console Account

- Obtain credentials from TFH
- Login to AWS Management Console
    - https://aws.amazon.com/console/
- Create Access Key
    - Open “I AM” page
    - Click tab: `Security credentials`
    - Click button: `Create access key`
    - Click option: `Command Line Interface (CLI)`
    - Follow instructions to create access key and:
        - Save: `Access Key`
        - Save: `Secret access key`

## Step 2: Activate

Activates service client CLI commands.  

```
source YOUR-WORKING-DIR/iris-mpc/iris-mpc-bins/scripts/service-client/activate
```

NOTE: To be executed once *per terminal session*.

HINT: you may wish to add activation to your local `~/.bashrc` file.

## Step 3: Initialise

Initialises local resources for each environment.  One time execution.

```
hnsw-service-client-init
```

NOTE: for `dev-stg` environment you will be instructed to edit the following file:

```
YOUR-WORKING-DIR/iris-mpc/iris-mpc-bins/scripts/service-client/envs/dev-stg/aws-credentials
```

## Step 4: Start session

Start a service client session against a supported environment.  To be executed once per terminal session.

```
hnsw-service-client-set-env env="dev-dkr"
```

NOTE: To be executed once *per terminal session*.  Repeat if you switch between supported environments within a session.

## Step 5: Execute Client

```
hnsw-service-client-exec env="dev-dkr" config=PATH-TO-A-SERVICE-CLIENT-CONFIG-FILE
```

## Example service client execution option files.

It is good practice to create a local folder, e.g. `~/.hnsw/service-client/exec-config`, into which to store execution option files.  Below is a non-exhaustive set of example execution option files. 

### Example 1.

- Simple uniqueness requests
- Iris shares generated on the fly

Copy following to `~/.hnsw/service-client/exec-config/example-1.toml`

```
[request_batch.Simple]
batch_count = 10
batch_size = 10
batch_kind = "uniqueness"

[shares_generator.FromCompute]
rng_seed = 42
```

### Example 2.

- Simple uniqueness requests
- Iris shares generated from an Iris codes NDJSON file

Copy following to `~/.hnsw/service-client/exec-config/example-2.toml`

```
[request_batch.Simple]
batch_count = 10
batch_size = 10
batch_kind = "uniqueness"

[shares_generator.FromFile]
path_to_ndjson_file = "YOUR-WORKING-DIR/iris-mpc/iris-mpc-utils/assets/iris-codes-plaintext/20250710-1k.ndjson"
rng_seed = 42
selection_strategy = "All"
```

Copy following to `~/.hnsw/service-client/exec-config/example-3.toml`

### Example 3.

- Simple reauth requests
- Iris shares generated on the fly

```
[request_batch.Simple]
batch_count = 10
batch_size = 10
batch_kind = "reauth"

[shares_generator.FromCompute]
rng_seed = 42
```
