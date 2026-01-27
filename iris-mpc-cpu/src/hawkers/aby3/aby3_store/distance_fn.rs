use crate::execution::hawk_main::HAWK_MINFHD_ROTATIONS;

use super::{
    Aby3DistanceRef, Aby3Query, Aby3Store, Aby3VectorRef, ArcIris, DistanceShare, VectorId,
};
use clap::ValueEnum;
use eyre::Result;
use iris_mpc_common::iris_db::iris::IrisCode;
use itertools::Itertools;
use std::time::Instant;

/// Rounds a batch size to the nearest multiple of 100.
fn round_batch_size(size: usize) -> usize {
    ((size + 50) / 100) * 100
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "snake_case")]
pub enum DistanceFn {
    Fhd,
    MinFhd,
}

use serde::{Deserialize, Serialize};
use DistanceFn::*;

impl DistanceFn {
    pub fn plaintext_distance(self, a: &IrisCode, b: &IrisCode) -> (u16, u16) {
        match self {
            Fhd => a.get_distance_fraction(b),
            MinFhd => a.get_min_distance_fraction_rotation_aware::<HAWK_MINFHD_ROTATIONS>(b),
        }
    }

    pub async fn eval_pairwise_distances(
        self,
        store: &mut Aby3Store,
        pairs: Vec<Option<(ArcIris, ArcIris)>>,
    ) -> Result<Vec<DistanceShare<u32>>> {
        match self {
            Fhd => DistanceSimple::eval_pairwise_distances(store, pairs).await,
            MinFhd => DistanceMinimalRotation::eval_pairwise_distances(store, pairs).await,
        }
    }

    pub async fn eval_distance_pairs(
        self,
        store: &mut Aby3Store,
        pairs: &[(Aby3Query, Aby3VectorRef)],
    ) -> Result<Vec<Aby3DistanceRef>> {
        match self {
            Fhd => DistanceSimple::eval_distance_pairs(store, pairs).await,
            MinFhd => DistanceMinimalRotation::eval_distance_pairs(store, pairs).await,
        }
    }

    pub async fn eval_distance_batch(
        self,
        store: &mut Aby3Store,
        query: &Aby3Query,
        vectors: &[VectorId],
    ) -> Result<Vec<DistanceShare<u32>>> {
        match self {
            Fhd => DistanceSimple::eval_distance_batch(store, query, vectors).await,
            MinFhd => DistanceMinimalRotation::eval_distance_batch(store, query, vectors).await,
        }
    }

    /// Evaluates distances for multiple (query, vectors) batches.
    ///
    /// For MinFhd, this enables prerotation buffer reuse within each batch.
    pub async fn eval_distance_multibatch(
        self,
        store: &mut Aby3Store,
        batches: Vec<(Aby3Query, Vec<VectorId>)>,
    ) -> Result<Vec<Vec<Aby3DistanceRef>>> {
        match self {
            Fhd => {
                // Fallback: process batch-by-batch using existing method
                let mut results = Vec::with_capacity(batches.len());
                for (query, vectors) in batches {
                    let distances =
                        DistanceSimple::eval_distance_batch(store, &query, &vectors).await?;
                    results.push(distances);
                }
                Ok(results)
            }
            MinFhd => DistanceMinimalRotation::eval_distance_multibatch(store, batches).await,
        }
    }
}

struct DistanceSimple;

impl DistanceSimple {
    async fn eval_pairwise_distances(
        store: &mut Aby3Store,
        pairs: Vec<Option<(ArcIris, ArcIris)>>,
    ) -> Result<Vec<DistanceShare<u32>>> {
        let ds_and_ts = store.workers.galois_ring_pairwise_distances(pairs).await?;
        store.gr_to_lifted_distances(ds_and_ts).await
    }

    async fn eval_distance_pairs(
        store: &mut Aby3Store,
        pairs: &[(Aby3Query, Aby3VectorRef)],
    ) -> Result<Vec<Aby3DistanceRef>> {
        let pairs = pairs
            .iter()
            .map(|(q, v)| (q.iris_proc.clone(), *v))
            .collect_vec();
        let ds_and_ts = store.workers.dot_product_pairs(pairs).await?;
        store.gr_to_lifted_distances(ds_and_ts).await
    }

    async fn eval_distance_batch(
        store: &mut Aby3Store,
        query: &Aby3Query,
        vectors: &[VectorId],
    ) -> Result<Vec<DistanceShare<u32>>> {
        let total_start = Instant::now();
        let batch_size = vectors.len();
        let rounded_batch_size = round_batch_size(batch_size);
        let batch_size_label = rounded_batch_size.to_string();

        // Track batch size occurrence
        metrics::counter!(
            "eval_distance_batch_count",
            "batch_size" => batch_size_label.clone(),
            "distance_fn" => "fhd"
        )
        .increment(1);

        let dot_product_start = Instant::now();
        let ds_and_ts = store
            .workers
            .dot_product_batch(query.iris_proc.clone(), vectors.to_vec())
            .await?;
        metrics::histogram!(
            "eval_distance_dot_product_duration_ms",
            "batch_size" => batch_size_label.clone(),
            "distance_fn" => "fhd"
        )
        .record(dot_product_start.elapsed().as_secs_f64() * 1000.0);

        let gr_to_lifted_start = Instant::now();
        let result = store.gr_to_lifted_distances(ds_and_ts).await;
        metrics::histogram!(
            "eval_distance_gr_to_lifted_duration_ms",
            "batch_size" => batch_size_label.clone(),
            "distance_fn" => "fhd"
        )
        .record(gr_to_lifted_start.elapsed().as_secs_f64() * 1000.0);

        // Record total duration
        metrics::histogram!(
            "eval_distance_batch_total_duration_ms",
            "batch_size" => batch_size_label,
            "distance_fn" => "fhd"
        )
        .record(total_start.elapsed().as_secs_f64() * 1000.0);

        result
    }
}

struct DistanceMinimalRotation;

impl DistanceMinimalRotation {
    async fn eval_pairwise_distances(
        store: &mut Aby3Store,
        pairs: Vec<Option<(ArcIris, ArcIris)>>,
    ) -> Result<Vec<DistanceShare<u32>>> {
        let ds_and_ts = store
            .workers
            .rotation_aware_pairwise_distances(pairs)
            .await?;
        let distances = store.gr_to_lifted_distances(ds_and_ts).await?;
        store
            .oblivious_min_distance_batch(transpose_from_flat(&distances))
            .await
    }

    async fn eval_distance_pairs(
        store: &mut Aby3Store,
        pairs: &[(Aby3Query, Aby3VectorRef)],
    ) -> Result<Vec<Aby3DistanceRef>> {
        let ds_and_ts = store
            .workers
            .rotation_aware_dot_product_pairs(
                pairs
                    .iter()
                    .map(|(q, v)| (q.iris_proc.clone(), *v))
                    .collect(),
            )
            .await?;
        let distances = store.gr_to_lifted_distances(ds_and_ts).await?;
        store
            .oblivious_min_distance_batch(transpose_from_flat(&distances))
            .await
    }

    async fn eval_distance_batch(
        store: &mut Aby3Store,
        query: &Aby3Query,
        vectors: &[VectorId],
    ) -> Result<Vec<DistanceShare<u32>>> {
        let total_start = Instant::now();
        let batch_size = vectors.len();
        let rounded_batch_size = round_batch_size(batch_size);
        let batch_size_label = rounded_batch_size.to_string();

        // Track batch size occurrence
        metrics::counter!(
            "eval_distance_batch_count",
            "batch_size" => batch_size_label.clone(),
            "distance_fn" => "min_fhd"
        )
        .increment(1);

        let dot_product_start = Instant::now();
        let ds_and_ts = store
            .workers
            .rotation_aware_dot_product_batch(query.iris_proc.clone(), vectors.to_vec())
            .await?;
        metrics::histogram!(
            "eval_distance_dot_product_duration_ms",
            "batch_size" => batch_size_label.clone(),
            "distance_fn" => "min_fhd"
        )
        .record(dot_product_start.elapsed().as_secs_f64() * 1000.0);

        let gr_to_lifted_start = Instant::now();
        let distances = store.gr_to_lifted_distances(ds_and_ts).await?;
        metrics::histogram!(
            "eval_distance_gr_to_lifted_duration_ms",
            "batch_size" => batch_size_label.clone(),
            "distance_fn" => "min_fhd"
        )
        .record(gr_to_lifted_start.elapsed().as_secs_f64() * 1000.0);

        let oblivious_min_start = Instant::now();
        let result = store
            .oblivious_min_distance_batch(transpose_from_flat(&distances))
            .await;
        metrics::histogram!(
            "eval_distance_oblivious_min_duration_ms",
            "batch_size" => batch_size_label.clone(),
            "distance_fn" => "min_fhd"
        )
        .record(oblivious_min_start.elapsed().as_secs_f64() * 1000.0);

        // Record total duration
        metrics::histogram!(
            "eval_distance_batch_total_duration_ms",
            "batch_size" => batch_size_label,
            "distance_fn" => "min_fhd"
        )
        .record(total_start.elapsed().as_secs_f64() * 1000.0);

        result
    }

    /// Evaluates distances for multiple (query, vectors) batches efficiently.
    ///
    /// Each query's prerotation is reused across all its target vectors.
    async fn eval_distance_multibatch(
        store: &mut Aby3Store,
        batches: Vec<(Aby3Query, Vec<VectorId>)>,
    ) -> Result<Vec<Vec<DistanceShare<u32>>>> {
        if batches.is_empty() {
            return Ok(vec![]);
        }

        let batch_sizes: Vec<usize> = batches.iter().map(|(_, vids)| vids.len()).collect();

        // Convert to worker format (use preprocessed iris)
        let worker_batches: Vec<(ArcIris, Vec<VectorId>)> = batches
            .into_iter()
            .map(|(q, vids)| (q.iris_proc, vids))
            .collect();

        // Get raw dot products grouped by batch
        let ds_and_ts_batches = store
            .workers
            .rotation_aware_dot_product_multibatch(worker_batches)
            .await?;

        // Flatten all batches to allow single calls to post-processing functions
        let flattened_ds_and_ts: Vec<_> = ds_and_ts_batches.into_iter().flatten().collect();

        if flattened_ds_and_ts.is_empty() {
            // All batches were empty
            return Ok(batch_sizes.iter().map(|_| vec![]).collect());
        }

        // Process all items in single batched calls
        let distances = store.gr_to_lifted_distances(flattened_ds_and_ts).await?;
        let all_mins = store
            .oblivious_min_distance_batch(transpose_from_flat(&distances))
            .await?;

        // Split results back into per-batch vectors
        let mut results = Vec::with_capacity(batch_sizes.len());
        let mut offset = 0;
        for batch_size in batch_sizes {
            results.push(all_mins[offset..offset + batch_size].to_vec());
            offset += batch_size;
        }

        Ok(results)
    }
}

/// Convert the results of rotation-parallel evaluations into the format convenient for minimum finding.
///
/// The input `distances` is a flat array where the rotations of each item of the batch are concatenated.
/// The output is a 2D matrix with dimensions: (rotations, batch).
///
/// With rotation r and batch item i:
///     `input[r + i * ROTATIONS] == output[r][i]`
fn transpose_from_flat(distances: &[Aby3DistanceRef]) -> Vec<Vec<Aby3DistanceRef>> {
    (0..HAWK_MINFHD_ROTATIONS)
        .map(|i| {
            distances
                .iter()
                .skip(i)
                .step_by(HAWK_MINFHD_ROTATIONS)
                .cloned()
                .collect()
        })
        .collect()
}
