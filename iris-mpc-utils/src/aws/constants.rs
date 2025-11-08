/// Default AWS region.
pub const AWS_REGION: &str = "eu-central-1";

/// Base URL for downloading node encryption public keys.
pub const AWS_PUBLIC_KEY_BASE_URL: &str = "http://localhost:4566/wf-dev-public-keys";

/// AWS: system request ingress queue URL.
pub const AWS_REQUEST_BUCKET_NAME: &str = "wf-smpcv2-dev-sns-requests";

/// AWS: system request ingress queue topic.
pub const AWS_REQUEST_TOPIC_ARN: &str = "arn:aws:sns:us-east-1:000000000000:iris-mpc-input.fifo";

/// AWS: system response egress queue URL.
pub const AWS_RESPONSE_QUEUE_URL: &str = "http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/iris-mpc-results-us-east-1.fifo";
