#!/usr/bin/env bash
set -e

export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_ENDPOINT_URL="http://localhost:4566"
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_REGION=us-east-1


for i in 0 1 2
do
  echo "Checking hawk\_participant\_${i} at <url>:300${i}/health..."
  curl -f hawk_participant_${i}:300${i}/health
done

echo "All endpoints are healthy."


# Prepare the results queue.
SQS_IRIS_MPC_RESULTS_QUEUE_NAME=iris-mpc-results-us-east-1.fifo
SQS_IRIS_MPC_RESULTS_QUEUE_URL=$(aws sqs get-queue-url --queue-name "$SQS_IRIS_MPC_RESULTS_QUEUE_NAME" --query 'QueueUrl' --output text)

echo "Clearing the results queue..."
aws sqs purge-queue --queue-url "$SQS_IRIS_MPC_RESULTS_QUEUE_URL"


echo "Sending requests..."
/bin/client \
    --request-topic-arn arn:aws:sns:$AWS_REGION:000000000000:iris-mpc-input.fifo \
    --requests-bucket-name wf-smpcv2-dev-sns-requests \
    --public-key-base-url "http://localhost:4566/wf-dev-public-keys" \
    --region $AWS_REGION \
    --n-repeat 1 \
    --random true

# TODO: re-add these once ready to consume results
# --response-queue-region $AWS_REGION \
# --response-queue-url https://sqs.eu-north-1.amazonaws.com/654654380399/temporal-results.fifo \


echo "Waiting for a result to show up..."
WAIT_TIME_SECONDS=10
MAX_RETRIES=12
COUNTER=0

# Polling loop
while true; do
  # Increment the retry counter
  COUNTER=$((COUNTER + 1))

  # Receive a message from the SQS queue
  MESSAGE=$(aws sqs receive-message --queue-url "$SQS_IRIS_MPC_RESULTS_QUEUE_URL" --max-number-of-messages 1 --wait-time-seconds "$WAIT_TIME_SECONDS" --query 'Messages[0].Body' --output text)

  # Check if a message was received
  if [ "$MESSAGE" != "None" ]; then
    echo "Received result: $MESSAGE"
    # Exit the loop once a message is processed
    break
  else
    echo "No results received, attempt $COUNTER of $MAX_RETRIES."
  fi

  # Check if maximum retries have been reached
  if [ "$COUNTER" -ge "$MAX_RETRIES" ]; then
    echo "Error: No results were produced."
    exit 1
  fi
done
