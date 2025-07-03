#!/bin/bash
set -e

AWS_REGION=us-east-1

create_fifo_queue() {
  local QUEUE_NAME_TO_CREATE=$1
  awslocal sqs create-queue --queue-name "${QUEUE_NAME_TO_CREATE}" --region $AWS_REGION --attributes "{\"FifoQueue\":\"true\", \"ContentBasedDeduplication\":\"true\", \"VisibilityTimeout\":\"30\"}"
}

create_sns(){
  local TOPIC_NAME_TO_CREATE=$1
  awslocal sns create-topic --name "${TOPIC_NAME_TO_CREATE}" --region $AWS_REGION --attributes "{\"FifoTopic\":\"true\", \"ContentBasedDeduplication\":\"true\"}"
}

create_bucket() {
  local BUCKET_NAME_TO_CREATE=$1
  awslocal s3 mb s3://"${BUCKET_NAME_TO_CREATE}"
}

upload_file_to_bucket() {
  local BUCKET_NAME=$1
  local FILE_PATH=$2
  local S3_KEY=$3
  awslocal s3 cp "$FILE_PATH" "s3://$BUCKET_NAME/$S3_KEY"
}

print_deleted_serial_id_file() {
  local BUCKET="wf-smpcv2-dev-sync-protocol"
  local KEY="dev_deleted_serial_ids.json"
  echo "Reading deleted_serial_ids from s3://${BUCKET}/${KEY}:"
  awslocal s3 cp "s3://${BUCKET}/${KEY}" -
}

echo "Creating S3 bucket..."
create_bucket "wf-dev-public-keys"
create_bucket "wf-smpcv2-dev-sns-requests"
create_bucket "wf-smpcv2-dev-sync-protocol"

# Ensure dev_deleted_serial_ids.json exists before uploading
echo "Creating static dev_deleted_serial_ids.json"
cat > ./dev_deleted_serial_ids.json <<EOF
{
    "deleted_serial_ids": [50, 99, 100, 200, 1000]
}
EOF


# add data to the genesis-deletion bucket
echo "Sample data for wf-smpcv2-dev-sync-protocol"
upload_file_to_bucket "wf-smpcv2-dev-sync-protocol" "./dev_deleted_serial_ids.json" "dev_deleted_serial_ids.json"

print_deleted_serial_id_file

# mpcv2 queues and topics
SNS_IRIS_MPC_INPUTS_TOPIC_NAME=iris-mpc-input.fifo
SNS_IRIS_MPC_INPUTS_TOPIC_ARN=arn:aws:sns:us-east-1:000000000000:$SNS_IRIS_MPC_INPUTS_TOPIC_NAME

SNS_IRIS_MPC_RESULTS_TOPIC_NAME=iris-mpc-results.fifo
SNS_IRIS_MPC_RESULTS_TOPIC_ARN=arn:aws:sns:us-east-1:000000000000:$SNS_IRIS_MPC_RESULTS_TOPIC_NAME

SQS_IRIS_MPC_INPUTS_PARTICIPANT_0_QUEUE_NAME=smpcv2-0-dev.fifo
SQS_IRIS_MPC_INPUTS_PARTICIPANT_0_QUEUE_ARN=arn:aws:sqs:us-east-1:000000000000:$SQS_IRIS_MPC_INPUTS_PARTICIPANT_0_QUEUE_NAME
SQS_IRIS_MPC_INPUTS_PARTICIPANT_1_QUEUE_NAME=smpcv2-1-dev.fifo
SQS_IRIS_MPC_INPUTS_PARTICIPANT_1_QUEUE_ARN=arn:aws:sqs:us-east-1:000000000000:$SQS_IRIS_MPC_INPUTS_PARTICIPANT_1_QUEUE_NAME
SQS_IRIS_MPC_INPUTS_PARTICIPANT_2_QUEUE_NAME=smpcv2-2-dev.fifo
SQS_IRIS_MPC_INPUTS_PARTICIPANT_2_QUEUE_ARN=arn:aws:sqs:us-east-1:000000000000:$SQS_IRIS_MPC_INPUTS_PARTICIPANT_2_QUEUE_NAME

SQS_IRIS_MPC_RESULTS_QUEUE_NAME=iris-mpc-results-us-east-1.fifo
SQS_IRIS_MPC_RESULTS_QUEUE_ARN=arn:aws:sqs:us-east-1:000000000000:$SQS_IRIS_MPC_RESULTS_QUEUE_NAME

#SQS_IRIS_MPC_DELETION_RESULTS_QUEUE_NAME=iris-mpc-identity-deletion-results-us-east-1.fifo
#SQS_IRIS_MPC_DELETION_QUEUE_ARN=arn:aws:sqs:us-east-1:000000000000:$SQS_IRIS_MPC_DELETION_RESULTS_QUEUE_NAME

awslocal secretsmanager create-secret --name dev/iris-mpc/ecdh-private-key-0 --description "Secret for 0" --secret-string "{\"private-key\":\"\"}" --region us-east-1
awslocal secretsmanager create-secret --name dev/iris-mpc/ecdh-private-key-1 --description "Secret for 1" --secret-string "{\"private-key\":\"\"}" --region us-east-1
awslocal secretsmanager create-secret --name dev/iris-mpc/ecdh-private-key-2 --description "Secret for 2" --secret-string "{\"private-key\":\"\"}" --region us-east-1

awslocal kms create-key --region us-east-1 --key-spec ECC_NIST_P256 --key-usage KEY_AGREEMENT --tags "[{\"TagKey\":\"_custom_id_\",\"TagValue\":\"00000000-0000-0000-0000-000000000000\"}]"
awslocal kms create-key --region us-east-1 --key-spec ECC_NIST_P256 --key-usage KEY_AGREEMENT --tags "[{\"TagKey\":\"_custom_id_\",\"TagValue\":\"00000000-0000-0000-0000-000000000001\"}]"
awslocal kms create-key --region us-east-1 --key-spec ECC_NIST_P256 --key-usage KEY_AGREEMENT --tags "[{\"TagKey\":\"_custom_id_\",\"TagValue\":\"00000000-0000-0000-0000-000000000002\"}]"

create_sns $SNS_IRIS_MPC_INPUTS_TOPIC_NAME
create_sns $SNS_IRIS_MPC_RESULTS_TOPIC_NAME

create_fifo_queue $SQS_IRIS_MPC_INPUTS_PARTICIPANT_0_QUEUE_NAME
create_fifo_queue $SQS_IRIS_MPC_INPUTS_PARTICIPANT_1_QUEUE_NAME
create_fifo_queue $SQS_IRIS_MPC_INPUTS_PARTICIPANT_2_QUEUE_NAME
create_fifo_queue $SQS_IRIS_MPC_RESULTS_QUEUE_NAME


awslocal sns subscribe --topic-arn "$SNS_IRIS_MPC_INPUTS_TOPIC_ARN" --protocol sqs --notification-endpoint $SQS_IRIS_MPC_INPUTS_PARTICIPANT_0_QUEUE_ARN --region $AWS_REGION
awslocal sns subscribe --topic-arn "$SNS_IRIS_MPC_INPUTS_TOPIC_ARN" --protocol sqs --notification-endpoint $SQS_IRIS_MPC_INPUTS_PARTICIPANT_1_QUEUE_ARN --region $AWS_REGION
awslocal sns subscribe --topic-arn "$SNS_IRIS_MPC_INPUTS_TOPIC_ARN" --protocol sqs --notification-endpoint $SQS_IRIS_MPC_INPUTS_PARTICIPANT_2_QUEUE_ARN --region $AWS_REGION
awslocal sns subscribe --topic-arn "$SNS_IRIS_MPC_RESULTS_TOPIC_ARN" --protocol sqs --notification-endpoint $SQS_IRIS_MPC_RESULTS_QUEUE_ARN --region $AWS_REGION

echo "LocalStack initialization complete"
echo "LocalStack initialization complete"
