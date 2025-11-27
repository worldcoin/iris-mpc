/// Base URL for downloading node encryption public keys.
pub const AWS_PUBLIC_KEY_BASE_URL: &str = "http://localhost:4566/wf-dev-public-keys";

/// AWS: system request ingress queue URL.
pub const AWS_S3_REQUEST_BUCKET_NAME: &str = "wf-smpcv2-dev-sns-requests";

/// AWS: system request ingress queue topic.
pub const AWS_SNS_REQUEST_TOPIC_ARN: &str =
    "arn:aws:sns:us-east-1:000000000000:iris-mpc-input.fifo";

/// AWS: system response egress queue URL.
pub const AWS_SQS_RESPONSE_QUEUE_URL: &str = "http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/iris-mpc-results-us-east-1.fifo";

/// Default AWS region.
pub const AWS_SQS_LONG_POLL_WAIT_TIME: usize = 10;

/// Test graph sizes.
pub const GRAPH_SIZE_RANGE: [usize; 8] = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 2_000_000];

/// Node config kinds.
pub const NODE_CONFIG_KIND: [&str; 2] = [NODE_CONFIG_KIND_MAIN, NODE_CONFIG_KIND_GENESIS];
pub const NODE_CONFIG_KIND_GENESIS: &str = "genesis";
pub const NODE_CONFIG_KIND_MAIN: &str = "main";

/// MPC parties.
pub const N_PARTIES: usize = PARTY_INDICES.len();
pub const PARTY_INDICES: [usize; 3] = [PARTY_IDX_0, PARTY_IDX_1, PARTY_IDX_2];
pub const PARTY_IDX_0: usize = 0;
pub const PARTY_IDX_1: usize = 1;
pub const PARTY_IDX_2: usize = 2;

/// Default application environment.
pub const DEFAULT_ENV: &str = "dev";
