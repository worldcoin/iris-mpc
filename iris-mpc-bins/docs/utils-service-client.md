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

### Step 3: Setup Local AWS Config

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

# AMPC: CLI args.
export AMPC_PUBLIC_KEY_BASE_URL="https://pki-smpcv2-dev.worldcoin.org"
export AMPC_S3_REQUEST_BUCKET="wf-smpcv2-dev-sns-requests-v2"
export AMPC_SQS_LONG_POLL_WAIT_TIME=10
export AMPC_SQS_RESPONSE_QUEUE_URL="https://sqs.eu-central-1.amazonaws.com/238407200320/hnsw-smpc-results.fifo"
export AMPC_SNS_REQUEST_TOPIC_ARN=arn:aws:sns:eu-central-1:238407200320:iris-mpc-input-dev.fifo
```

## Step 5: Setup Local Execution Script

Copy following to `~/.hnsw/exec/exec_service_client.sh` & **edit accordingly**.

```
pushd "YOUR-WORKING-DIRECTORY/iris-mpc/iris-mpc-bins"

cargo run --release --bin service-client -- \
    --aws-public-key-base-url "${AMPC_PUBLIC_KEY_BASE_URL}" \
    --aws-s3-request-bucket-name "${AMPC_S3_REQUEST_BUCKET}" \
    --aws-sns-request-topic-arn "${AMPC_SNS_REQUEST_TOPIC_ARN}" \
    --aws-sqs-long-poll-wait-time "${AMPC_SQS_LONG_POLL_WAIT_TIME}" \
    --aws-sqs-response-queue-url "${AMPC_SQS_RESPONSE_QUEUE_URL}" \
    --environment "dev" \
    --request-batch-count 5 \
    --request-batch-size 10 \
    --rng-seed 42

popd
```

## Step 6: Execute Client

```
# Activate environment variables.
. ~/.hnsw/envs/dev-staging

# Execute client.
. ~/.hnsw/exec/exec_service_client.sh
```
