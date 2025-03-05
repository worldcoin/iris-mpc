#!/usr/bin/env bash
set -e

export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_ENDPOINT_URL="http://localstack:4566"
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1


for i in 0 1 2
do
  echo "Checking hawk\_participant\_${i} at <url>:300${i}/health..."
  curl -f hawk_participant_${i}:300${i}/health
done

echo "All endpoints are healthy. Running now the client..."


/bin/client \
   --request-topic-arn arn:aws:sns:$AWS_REGION:000000000000:iris-mpc-input.fifo \
   --requests-bucket-name wf-smpcv2-dev-sns-requests \
   --public-key-base-url "http://localstack:4566/wf-dev-public-keys" \
   --response-queue-url http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/iris-mpc-results-us-east-1.fifo \
   --endpoint-url $AWS_ENDPOINT_URL \
   --region $AWS_REGION \
   --random true
