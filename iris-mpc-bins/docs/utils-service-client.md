# Running Service Client Utility

A new tool, named `service-client` has been developed to support end-to-end system testing.  The tool's design intent is to support the dispatch of various traffic patterns into the system.  Such patterns will differ in terms of time (short vs extended bursts) and/or content (different request types).  It is important that the tool is easy to use in an array of environments (local and/or staged).

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

## Step 2: Setup Local AWS credentials

Copy following to `~/.aws/credentials` & **edit accordingly**.

```
[default]
aws_access_key_id = YOUR-AWS-ACCESS-KEY-ID
aws_secret_access_key = YOUR-SECRET-AWS-ACCESS-KEY

[worldcoin-smpcv-io-vpc-dev]
aws_access_key_id = YOUR-AWS-ACCESS-KEY-ID
aws_secret_access_key = YOUR-SECRET-AWS-ACCESS-KEY
```

## Step 3: Setup Local AWS Config

Copy following to `~/.aws/config`

```
[default]
region = eu-central-1
output = table

[profile worldcoin-smpcv-io-0-dev]
source_profile=default
region = eu-central-1
role_arn=arn:aws:iam::004304088310:role/ampc-hnsw-developer-role

[profile worldcoin-smpcv-io-1-dev]
source_profile=default
region = eu-central-1
role_arn=arn:aws:iam::284038850594:role/ampc-hnsw-developer-role

[profile worldcoin-smpcv-io-2-dev]
source_profile=default
region = eu-central-1
role_arn=arn:aws:iam::882222437714:role/ampc-hnsw-developer-role

[profile worldcoin-smpcv-io-vpc-dev]
source_profile=default
region = eu-central-1
role_arn=arn:aws:iam::238407200320:role/ampc-hnsw-developer-role
output = None
```

## Step 4: Setup Local Environment Variables

Copy following to `~/.hnsw/envs/dev-staging`.

```
# AWS: Force usage of ~/.aws/config & ~/.aws/credentials.
unset AWS_ACCESS_KEY_ID
unset AWS_ENDPOINT_URL
unset AWS_REGION
unset AWS_SECRET_ACCESS_KEY
export AWS_PROFILE="worldcoin-smpcv-io-vpc-dev"
```

## Step 4: Setup Local Configuration Files

Copy following to `~/.hnsw/service-client/config/aws/dev-local.toml`.

```
environment = "dev"
public_key_base_url = "http://localhost:4566/wf-dev-public-keys"
s3_request_bucket_name = "wf-smpcv2-dev-sns-requests"
sns_request_topic_arn = "arn:aws:sns:us-east-1:000000000000:iris-mpc-input.fifo"
sqs_long_poll_wait_time = 10
sqs_response_queue_url = "http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/iris-mpc-results-us-east-1.fifo"
sqs_wait_time_seconds = 5
```

Copy following to `~/.hnsw/service-client/config/aws/dev-staging.toml`.

```
environment = "dev"
public_key_base_url = "https://pki-smpcv2-dev.worldcoin.org"
s3_request_bucket_name = "wf-smpcv2-dev-sns-requests-v2"
sns_request_topic_arn = "arn:aws:sns:eu-central-1:238407200320:iris-mpc-input-dev.fifo"
sqs_long_poll_wait_time = 10
sqs_response_queue_url = "https://sqs.eu-central-1.amazonaws.com/238407200320/hnsw-smpc-results.fifo"
sqs_wait_time_seconds = 5
```

Copy following to `~/.hnsw/service-client/config/requests/simple-rng-0.toml`.

```
[request_batch.SimpleBatchKind]
batch_count = 1
batch_size = 1
batch_kind = "uniqueness"

[shares_generator.FromRng]
rng_seed = Some(42)
```

Copy following to `~/.hnsw/service-client/config/requests/simple-ndjson-0.toml`.

```
[request_batch.SimpleBatchKind]
batch_count = 1
batch_size = 1
batch_kind = "uniqueness"
known_iris_serial_id = 1

[shares_generator.FromFile]
path_to_ndjson = <PATH_TO_AN_NDJSON_FILE>
```
## Step 5: Setup Local Execution Script

Copy following to `~/.hnsw/exec/exec_service_client.sh` & **edit accordingly**.

```
pushd "YOUR-WORKING-DIRECTORY/iris-mpc/iris-mpc-bins"

cargo run --release --bin service-client -- \
    --path-to-config \
        "${HOME}/.hnsw/config/service-client-dev-staging-0.toml"
    --path-to-config-aws \
        "${HOME}/.hnsw/service-client/config/requests/simple-rng-0.toml"

popd
```

## Step 6: Execute Client

```
# Execute client.
. ~/.hnsw/exec/exec_service_client.sh
```
