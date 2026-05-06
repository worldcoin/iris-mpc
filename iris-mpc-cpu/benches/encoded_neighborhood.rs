use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use iris_mpc_cpu::hnsw::graph::encoded_neighborhood::EncodedNeighborhood;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeSet;

fn sample_sorted_ids(seed: u64, k: usize, universe: u64) -> Vec<u32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut set = BTreeSet::new();
    while set.len() < k {
        set.insert(rng.gen_range(0..universe) as u32);
    }
    set.into_iter().collect()
}

fn bench_encode_at_k450(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode/k=450");
    group.throughput(Throughput::Elements(450));
    let universes: &[(&str, u64)] = &[
        ("n=1M", 1_000_000),
        ("n=10M", 10_000_000),
        ("n=100M", 100_000_000),
        ("n=2^32-1", u32::MAX as u64),
    ];
    for &(label, universe) in universes {
        let ids = sample_sorted_ids(0xA5A5_0000 ^ universe, 450, universe);
        group.bench_with_input(BenchmarkId::from_parameter(label), &ids, |b, ids| {
            b.iter(|| {
                let encoded = EncodedNeighborhood::encode(black_box(ids)).expect("encode");
                black_box(encoded);
            });
        });
    }
    group.finish();
}

fn bench_decode_at_k450(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode/k=450");
    group.throughput(Throughput::Elements(450));
    let universes: &[(&str, u64)] = &[
        ("n=1M", 1_000_000),
        ("n=10M", 10_000_000),
        ("n=100M", 100_000_000),
        ("n=2^32-1", u32::MAX as u64),
    ];
    for &(label, universe) in universes {
        let ids = sample_sorted_ids(0xA5A5_0000 ^ universe, 450, universe);
        let encoded = EncodedNeighborhood::encode(&ids).expect("encode");
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &encoded,
            |b, encoded| {
                b.iter(|| {
                    let decoded = encoded.decode().expect("decode");
                    black_box(decoded);
                });
            },
        );
    }
    group.finish();
}

fn bench_scaling_at_n10m(c: &mut Criterion) {
    let universe: u64 = 10_000_000;
    let ks: &[usize] = &[10, 100, 450, 2000];

    let mut enc_group = c.benchmark_group("scaling/n=10M/encode");
    for &k in ks {
        enc_group.throughput(Throughput::Elements(k as u64));
        let ids = sample_sorted_ids(0xC0DE_0000 ^ k as u64, k, universe);
        enc_group.bench_with_input(
            BenchmarkId::from_parameter(format!("k={k}")),
            &ids,
            |b, ids| {
                b.iter(|| {
                    let encoded = EncodedNeighborhood::encode(black_box(ids)).expect("encode");
                    black_box(encoded);
                });
            },
        );
    }
    enc_group.finish();

    let mut dec_group = c.benchmark_group("scaling/n=10M/decode");
    for &k in ks {
        dec_group.throughput(Throughput::Elements(k as u64));
        let ids = sample_sorted_ids(0xC0DE_0000 ^ k as u64, k, universe);
        let encoded = EncodedNeighborhood::encode(&ids).expect("encode");
        dec_group.bench_with_input(
            BenchmarkId::from_parameter(format!("k={k}")),
            &encoded,
            |b, e| {
                b.iter(|| {
                    let decoded = e.decode().expect("decode");
                    black_box(decoded);
                });
            },
        );
    }
    dec_group.finish();
}

criterion_group!(
    benches,
    bench_encode_at_k450,
    bench_decode_at_k450,
    bench_scaling_at_n10m
);
criterion_main!(benches);
