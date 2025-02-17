export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_ENDPOINT_URL="http://localstack:4566"
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_REGION=us-east-1


/bin/client \
    --request-topic-arn arn:aws:sns:$AWS_REGION:000000000000:iris-mpc-input.fifo \
    --request-topic-region $AWS_REGION \
    --requests-bucket-name wf-smpcv2-dev-sns-requests \
    --public-key-base-url "http://wf-dev-public-keys.s3.$AWS_REGION.localhost.localstack.cloud:4566" \
    --requests-bucket-region $AWS_REGION \
    --random true

# TODO: re-add these once ready to consume results
# --response-queue-region $AWS_REGION \
# --response-queue-url https://sqs.eu-north-1.amazonaws.com/654654380399/temporal-results.fifo \
