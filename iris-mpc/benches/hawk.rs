use criterion::{criterion_group, criterion_main, Criterion, SamplingMode};
use iris_mpc::client::{run_client, Opt};

// Derive Clone on Opt (if not already) so we can reuse it in each iteration.
fn bench_hawk_server(c: &mut Criterion) {
    let mut group = c.benchmark_group("hsnw_e2e_performance_tests".to_string());
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    tracing_subscriber::fmt::init();

    let opts = Opt {
        request_topic_arn:    "arn:aws:sns:us-east-1:000000000000:iris-mpc-input.fifo".to_string(),
        region:               "us-east-1".to_string(),
        requests_bucket_name: "wf-smpcv2-dev-sns-requests".to_string(),
        public_key_base_url:
            "http://wf-dev-public-keys.s3.us-east-1.localhost.localstack.cloud:4566".to_string(),
        db_index:             None,
        rng_seed:             None,
        n_repeat:             Some(1), /* use a lower number for benchmarking iterations if
                                        * appropriate */
        random:               Some(true),
        endpoint_url:         Some("http://localhost:4566".to_string()),
    };

    // Create a single Tokio runtime outside the iteration to avoid re-creating it
    // each time.
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function("e2e_server_batch_processing", move |b| {
        b.iter(|| {
            let result = rt.block_on(run_client(opts.clone()));
            assert!(result.is_ok());
        })
    });
    group.finish();
}

criterion_group! {
    hawk,
    bench_hawk_server,
}

criterion_main!(hawk);
