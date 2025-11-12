# Running Service Client Utility

TODO: overview

## Prerequisites

TODO

## Step 1: Setup AWS credentials

Copy following to `~/.aws/credentials` & edit accordingly.

```
[default]
aws_access_key_id = YOUR-AWS-ACCESS-KEY-ID
aws_secret_access_key = YOUR-SECRET-AWS-ACCESS-KEY

[worldcoin-smpcv-io-vpc-dev]
aws_access_key_id = YOUR-AWS-ACCESS-KEY-ID
aws_secret_access_key = YOUR-SECRET-AWS-ACCESS-KEY
```

### Step 2: Setup AWS Config

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

## Step 3: Setup Environment Variables

Copy following to `~/.hnsw/envs/dev-staging`

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

## Step 4: Setup Execution Script

Copy following to `~/.hnsw/exec/exec_service_client.sh`

```
pushd "${IRIS_MPC_HOME}/iris-mpc-bins"
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

## Step 5: Execute Client

```
# Activate environment variables.
. ~/.hnsw/envs/dev-staging

# Execute client.
. ~/.hnsw/exec/exec_service_client.sh
```
