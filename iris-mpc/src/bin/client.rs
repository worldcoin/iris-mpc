use iris_mpc::client::{run_client, Opt};

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let opts = Opt {
        request_topic_arn:      "arn:aws:sns:us-east-1:000000000000:iris-mpc-input.fifo"
            .to_string(),
        request_topic_region:   "us-east-1".to_string(),
        requests_bucket_name:   "wf-smpcv2-dev-sns-requests".to_string(),
        public_key_base_url:
            "http://wf-dev-public-keys.s3.us-east-1.localhost.localstack.cloud:4566".to_string(),
        requests_bucket_region: "us-east-1".to_string(),
        db_index:               None,
        rng_seed:               None,
        n_repeat:               Some(1), /* use a lower number for benchmarking iterations if
                                          * appropriate */
        random:                 Some(true),
    };
    run_client(opts).await
}
