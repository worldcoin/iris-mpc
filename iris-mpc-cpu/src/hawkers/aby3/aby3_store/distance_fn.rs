use crate::execution::hawk_main::HAWK_MIN_ROTATIONS;

use super::{Aby3Query, Aby3Store, ArcIris, DistanceOps, DistanceShare, VectorId};
use ampc_secret_sharing::shares::int_ring::IntRing2k;
use clap::ValueEnum;
use eyre::Result;
use itertools::Itertools;
use rand_distr::{Distribution, Standard};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "snake_case")]
pub enum DistanceFn {
    Simple,
    MinRotation,
}

use serde::{Deserialize, Serialize};
use DistanceFn::{MinRotation, Simple};

impl DistanceFn {
    pub async fn eval_pairwise_distances<D: DistanceOps>(
        self,
        store: &mut Aby3Store<D>,
        pairs: Vec<Option<(ArcIris, ArcIris)>>,
    ) -> Result<Vec<DistanceShare<D::Ring>>>
    where
        Standard: Distribution<D::Ring>,
    {
        match self {
            Simple => DistanceSimple::eval_pairwise_distances(store, pairs).await,
            MinRotation => DistanceMinimalRotation::eval_pairwise_distances(store, pairs).await,
        }
    }

    pub async fn eval_distance_pairs<D: DistanceOps>(
        self,
        store: &mut Aby3Store<D>,
        pairs: &[(Aby3Query, VectorId)],
    ) -> Result<Vec<DistanceShare<D::Ring>>>
    where
        Standard: Distribution<D::Ring>,
    {
        match self {
            Simple => DistanceSimple::eval_distance_pairs(store, pairs).await,
            MinRotation => DistanceMinimalRotation::eval_distance_pairs(store, pairs).await,
        }
    }

    pub async fn eval_distance_batch<D: DistanceOps>(
        self,
        store: &mut Aby3Store<D>,
        query: &Aby3Query,
        vectors: &[VectorId],
    ) -> Result<Vec<DistanceShare<D::Ring>>>
    where
        Standard: Distribution<D::Ring>,
    {
        match self {
            Simple => DistanceSimple::eval_distance_batch(store, query, vectors).await,
            MinRotation => {
                DistanceMinimalRotation::eval_distance_batch(store, query, vectors).await
            }
        }
    }

    /// Evaluates distances for multiple (query, vectors) batches.
    ///
    /// For MinRotation, this enables prerotation buffer reuse within each batch.
    pub async fn eval_distance_multibatch<D: DistanceOps>(
        self,
        store: &mut Aby3Store<D>,
        batches: Vec<(Aby3Query, Vec<VectorId>)>,
    ) -> Result<Vec<Vec<DistanceShare<D::Ring>>>>
    where
        Standard: Distribution<D::Ring>,
    {
        match self {
            Simple => {
                // Fallback: process batch-by-batch using existing method
                let mut results = Vec::with_capacity(batches.len());
                for (query, vectors) in batches {
                    let distances =
                        DistanceSimple::eval_distance_batch(store, &query, &vectors).await?;
                    results.push(distances);
                }
                Ok(results)
            }
            MinRotation => DistanceMinimalRotation::eval_distance_multibatch(store, batches).await,
        }
    }
}

struct DistanceSimple;

impl DistanceSimple {
    async fn eval_pairwise_distances<D: DistanceOps>(
        store: &mut Aby3Store<D>,
        pairs: Vec<Option<(ArcIris, ArcIris)>>,
    ) -> Result<Vec<DistanceShare<D::Ring>>>
    where
        Standard: Distribution<D::Ring>,
    {
        let ds_and_ts = store.workers.galois_ring_pairwise_distances(pairs).await?;
        store.gr_to_lifted_distances(ds_and_ts).await
    }

    async fn eval_distance_pairs<D: DistanceOps>(
        store: &mut Aby3Store<D>,
        pairs: &[(Aby3Query, VectorId)],
    ) -> Result<Vec<DistanceShare<D::Ring>>>
    where
        Standard: Distribution<D::Ring>,
    {
        let pairs = pairs
            .iter()
            .map(|(q, v)| (q.iris_proc.clone(), *v))
            .collect_vec();
        let ds_and_ts = store.workers.dot_product_pairs(pairs).await?;
        store.gr_to_lifted_distances(ds_and_ts).await
    }

    async fn eval_distance_batch<D: DistanceOps>(
        store: &mut Aby3Store<D>,
        query: &Aby3Query,
        vectors: &[VectorId],
    ) -> Result<Vec<DistanceShare<D::Ring>>>
    where
        Standard: Distribution<D::Ring>,
    {
        let ds_and_ts = store
            .workers
            .dot_product_batch(query.iris_proc.clone(), vectors.to_vec())
            .await?;
        store.gr_to_lifted_distances(ds_and_ts).await
    }
}

struct DistanceMinimalRotation;

impl DistanceMinimalRotation {
    async fn eval_pairwise_distances<D: DistanceOps>(
        store: &mut Aby3Store<D>,
        pairs: Vec<Option<(ArcIris, ArcIris)>>,
    ) -> Result<Vec<DistanceShare<D::Ring>>>
    where
        Standard: Distribution<D::Ring>,
    {
        let ds_and_ts = store
            .workers
            .rotation_aware_pairwise_distances(pairs)
            .await?;
        let distances = store.gr_to_lifted_distances(ds_and_ts).await?;
        store
            .oblivious_min_distance_batch(transpose_from_flat(&distances))
            .await
    }

    async fn eval_distance_pairs<D: DistanceOps>(
        store: &mut Aby3Store<D>,
        pairs: &[(Aby3Query, VectorId)],
    ) -> Result<Vec<DistanceShare<D::Ring>>>
    where
        Standard: Distribution<D::Ring>,
    {
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

    async fn eval_distance_batch<D: DistanceOps>(
        store: &mut Aby3Store<D>,
        query: &Aby3Query,
        vectors: &[VectorId],
    ) -> Result<Vec<DistanceShare<D::Ring>>>
    where
        Standard: Distribution<D::Ring>,
    {
        let dot_start = std::time::Instant::now();
        let ds_and_ts = store
            .workers
            .rotation_aware_dot_product_batch(query.iris_proc.clone(), vectors.to_vec())
            .await?;
        metrics::histogram!("eval_distance_dot_product_duration")
            .record(dot_start.elapsed().as_secs_f64());

        let lift_start = std::time::Instant::now();
        let distances = store.gr_to_lifted_distances(ds_and_ts).await?;
        metrics::histogram!("eval_distance_lift_duration")
            .record(lift_start.elapsed().as_secs_f64());

        let omin_start = std::time::Instant::now();
        let result = store
            .oblivious_min_distance_batch(transpose_from_flat(&distances))
            .await;
        metrics::histogram!("eval_distance_oblivious_min_duration")
            .record(omin_start.elapsed().as_secs_f64());

        result
    }

    /// Evaluates distances for multiple (query, vectors) batches efficiently.
    ///
    /// Each query's prerotation is reused across all its target vectors.
    async fn eval_distance_multibatch<D: DistanceOps>(
        store: &mut Aby3Store<D>,
        batches: Vec<(Aby3Query, Vec<VectorId>)>,
    ) -> Result<Vec<Vec<DistanceShare<D::Ring>>>>
    where
        Standard: Distribution<D::Ring>,
    {
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
fn transpose_from_flat<T: IntRing2k>(distances: &[DistanceShare<T>]) -> Vec<Vec<DistanceShare<T>>> {
    (0..HAWK_MIN_ROTATIONS)
        .map(|i| {
            distances
                .iter()
                .skip(i)
                .step_by(HAWK_MIN_ROTATIONS)
                .cloned()
                .collect()
        })
        .collect()
}
