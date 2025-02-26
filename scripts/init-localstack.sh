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


echo "Creating S3 bucket..."
create_bucket "wf-dev-public-keys"
create_bucket "wf-smpcv2-dev-sns-requests"

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

#SQS_IRIS_MPC_RESULTS_QUEUE_NAME=iris-mpc-results-us-east-1.fifo
#SQS_IRIS_MPC_RESULTS_QUEUE_ARN=arn:aws:sqs:us-east-1:000000000000:$SQS_IRIS_MPC_RESULTS_QUEUE_NAME
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

awslocal sns subscribe --topic-arn "$SNS_IRIS_MPC_INPUTS_TOPIC_ARN" --protocol sqs --notification-endpoint $SQS_IRIS_MPC_INPUTS_PARTICIPANT_0_QUEUE_ARN --region $AWS_REGION
awslocal sns subscribe --topic-arn "$SNS_IRIS_MPC_INPUTS_TOPIC_ARN" --protocol sqs --notification-endpoint $SQS_IRIS_MPC_INPUTS_PARTICIPANT_1_QUEUE_ARN --region $AWS_REGION
awslocal sns subscribe --topic-arn "$SNS_IRIS_MPC_INPUTS_TOPIC_ARN" --protocol sqs --notification-endpoint $SQS_IRIS_MPC_INPUTS_PARTICIPANT_2_QUEUE_ARN --region $AWS_REGION

echo "LocalStack initialization complete"
