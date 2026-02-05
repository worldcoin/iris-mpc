# Running Service Client Utility

A new HNSW tool, named `service-client` has been developed to support end-to-end system testing.  The tool's design intent is to support the dispatch of various traffic patterns into the system.  Such patterns will differ in terms of time (short vs extended bursts) and/or content (different request types).  

### Environments

The tool is designed to be used in the following environments:

- *dev-dkr*: Local dockerised development environment

- *dev-stg*: Remote AWS staged development environment

## Prelude: Setup Datadog & AWS accounts

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

## Step 1: Activate

```
source YOUR-WORKING-DIR/iris-mpc/iris-mpc-bins/scripts/iris-mpc-utils/service-client/activate
```

*HINT*: you may wish to add activation to your local `~/.bashrc` file.

## Step 2: View Help

```
hnsw-service-client-init help
hnsw-service-client-set-env help
hnsw-service-client-exec help
```

## Step 3: Initialise

Initialises local resources for each environment.  One time execution.

```
hnsw-service-client-init
```

*NOTE*: you will be instructed to review/edit the following files:

```
${HOME}/.hnsw/service-client/aws-opts/dev-dkr/aws-credentials
${HOME}/.hnsw/service-client/aws-opts/dev-stg/aws-credentials
```

## Step 4: Start Session

Starts service client session against a supported environment.  One time execution per terminal session.

```
hnsw-service-client-set-env dev-dkr
```

Options: 
- env = dev-dkr | dev-stg

Defaults: 
- env = dev-dkr

*NOTE*: Repeat if you switch between supported environments within a terminal session.

## Step 5: Execute

```
hnsw-service-client-exec PATH-TO-A-SERVICE-CLIENT-EXEC-OPTIONS-FILE
```

Defaults: 
- filepath = ${HOME}/.hnsw/service-client/exec-opts/examples/example-1.toml

## Example Service Client Execution Options.

It is good practice to create a local folder, e.g. `~/.hnsw/service-client/my-exec-opts`, into which to store execution option files.  Below is a non-exhaustive set of example execution option files. 

### Example 1.

- Simple uniqueness requests
- Iris shares generated on the fly

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
