export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_ENDPOINT_URL="http://127.0.0.1:4566"
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_REGION=us-east-1


cargo run --release --bin client -- \
    --request-topic-arn arn:aws:sns:us-east-1:000000000000:iris-mpc-input.fifo \
    --request-topic-region us-east-1 \
    --requests-bucket-name wf-smpcv2-dev-sns-requests \
    --public-key-base-url "http://wf-dev-public-keys.s3.us-east-1.localhost.localstack.cloud:4566" \
    --requests-bucket-region us-east-1 \
    --random true

# TODO: re-add these once ready to consume results
# --response-queue-region us-east-1 \
# --response-queue-url https://sqs.eu-north-1.amazonaws.com/654654380399/temporal-results.fifo \
