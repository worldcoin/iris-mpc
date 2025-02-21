#![feature(test)]
extern crate test;
use criterion::{criterion_group, criterion_main, Criterion};
use iris_mpc::client::{run_client, Opt};

// Derive Clone on Opt (if not already) so we can reuse it in each iteration.
fn client_performance_tests(c: &mut Criterion) {
    let mut group = c.benchmark_group("Client performance tests".to_string());
    // Build the options exactly as you would pass them on the CLI:
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

    // Create a single Tokio runtime outside the iteration to avoid re-creating it
    // each time.
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function("with copy to host", move |b| {
        b.iter(|| {
            let result = rt.block_on(run_client(opts.clone()));
            assert!(result.is_ok());
        })
    });
}

criterion_group!(
    name = client_perf;
    config = Criterion::default();
    targets = client_performance_tests
);

criterion_main!(client_perf);
