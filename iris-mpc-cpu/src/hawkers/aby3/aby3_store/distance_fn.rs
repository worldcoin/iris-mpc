use crate::execution::hawk_main::{
    iris_worker::{QueryId, QuerySpec},
    HAWK_MIN_DIST_ROTATIONS,
};
use crate::phase_trace;

use super::{Aby3Query, Aby3Store, DistanceOps, DistanceShare, VectorId};
use ampc_secret_sharing::shares::{
    int_ring::IntRing2k, vecshare_bittranspose::Transpose64, VecShare,
};
use clap::ValueEnum;
use eyre::Result;
use rand_distr::{Distribution, Standard};
use serde::{Deserialize, Serialize};

/// The two distance computation strategies.
///
/// This is the lightweight source-of-truth enum shared by every component
/// that cares about the simple-vs-min-rotation distinction.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "snake_case")]
pub enum DistanceMode {
    Simple,
    MinRotation,
}

/// MPC distance orchestration: calls the worker pool for dot products,
/// lifts to ring distances, and (for `MinRotation`) applies oblivious-min.
#[derive(Debug, Clone, Copy)]
pub struct DistanceFn {
    pub mode: DistanceMode,
}

impl DistanceFn {
    pub const fn new(mode: DistanceMode) -> Self {
        Self { mode }
    }
}

use DistanceMode::{MinRotation, Simple};

impl DistanceFn {
    /// Compute pairwise distances between pairs of cached queries using the
    /// worker pool's `compute_pairwise_distances` method.
    pub async fn eval_pairwise_distances<D: DistanceOps>(
        self,
        store: &mut Aby3Store<D>,
        pairs: Vec<Option<(QuerySpec, QueryId)>>,
    ) -> Result<Vec<DistanceShare<D::Ring>>>
    where
        Standard: Distribution<D::Ring>,
        VecShare<D::Ring>: Transpose64,
    {
        let ds_and_ts = store.workers.compute_pairwise_distances(pairs).await?;
        let distances = store.gr_to_lifted_distances(ds_and_ts).await?;
        match self.mode {
            Simple => Ok(distances),
            MinRotation => {
                store
                    .oblivious_min_distance_batch(transpose_from_flat(&distances))
                    .await
            }
        }
    }

    pub async fn eval_distance_pairs<D: DistanceOps>(
        self,
        store: &mut Aby3Store<D>,
        pairs: &[(Aby3Query, VectorId)],
    ) -> Result<Vec<DistanceShare<D::Ring>>>
    where
        Standard: Distribution<D::Ring>,
        VecShare<D::Ring>: Transpose64,
    {
        let batches = pairs.iter().map(|(q, v)| (*q, vec![*v])).collect();
        let ds_and_ts_batches = store.workers.compute_dot_products(batches).await?;
        let ds_and_ts: Vec<_> = ds_and_ts_batches.into_iter().flatten().collect();
        let distances = store.gr_to_lifted_distances(ds_and_ts).await?;
        match self.mode {
            Simple => Ok(distances),
            MinRotation => {
                store
                    .oblivious_min_distance_batch(transpose_from_flat(&distances))
                    .await
            }
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
        VecShare<D::Ring>: Transpose64,
    {
        let dot_start = std::time::Instant::now();
        phase_trace!("dot_product", "n_vectors" => vectors.len());
        let ds_and_ts_batches = store
            .workers
            .compute_dot_products(vec![(*query, vectors.to_vec())])
            .await?;
        let ds_and_ts = ds_and_ts_batches.into_iter().next().unwrap_or_default();
        metrics::histogram!("eval_distance_dot_product_duration")
            .record(dot_start.elapsed().as_secs_f64());

        let lift_start = std::time::Instant::now();
        phase_trace!("mpc_lift", "n_vectors" => vectors.len());
        let distances = store.gr_to_lifted_distances(ds_and_ts).await?;
        metrics::histogram!("eval_distance_lift_duration")
            .record(lift_start.elapsed().as_secs_f64());

        match self.mode {
            Simple => Ok(distances),
            MinRotation => {
                let omin_start = std::time::Instant::now();
                phase_trace!("oblivious_min", "n_vectors" => vectors.len());
                let result = store
                    .oblivious_min_distance_batch(transpose_from_flat(&distances))
                    .await;
                metrics::histogram!("eval_distance_oblivious_min_duration")
                    .record(omin_start.elapsed().as_secs_f64());
                result
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
        VecShare<D::Ring>: Transpose64,
    {
        if batches.is_empty() {
            return Ok(vec![]);
        }

        let batch_sizes: Vec<usize> = batches.iter().map(|(_, vids)| vids.len()).collect();

        let trait_batches: Vec<_> = batches;

        let ds_and_ts_batches = store.workers.compute_dot_products(trait_batches).await?;

        // Flatten all batches into a single lift call to minimize MPC round-trips.
        let flattened_ds_and_ts: Vec<_> = ds_and_ts_batches.into_iter().flatten().collect();

        if flattened_ds_and_ts.is_empty() {
            return Ok(batch_sizes.iter().map(|_| vec![]).collect());
        }

        let distances = store.gr_to_lifted_distances(flattened_ds_and_ts).await?;

        match self.mode {
            Simple => {
                // Split results back into per-batch vectors.
                let mut results = Vec::with_capacity(batch_sizes.len());
                let mut offset = 0;
                for batch_size in batch_sizes {
                    results.push(distances[offset..offset + batch_size].to_vec());
                    offset += batch_size;
                }
                Ok(results)
            }
            MinRotation => {
                let all_mins = store
                    .oblivious_min_distance_batch(transpose_from_flat(&distances))
                    .await?;

                // Split results back into per-batch vectors.
                let mut results = Vec::with_capacity(batch_sizes.len());
                let mut offset = 0;
                for batch_size in batch_sizes {
                    results.push(all_mins[offset..offset + batch_size].to_vec());
                    offset += batch_size;
                }
                Ok(results)
            }
        }
    }
}

/// Convert the results of rotation-parallel evaluations into the format convenient for minimum finding.
///
/// The input `distances` is a flat array where the rotations of each item of the batch are concatenated.
/// The output is a 2D matrix with dimensions: (rotations, batch).
///
/// With rotation r and batch item i:
///     `input[r + i * ROTATIONS] == output[r][i]`
pub(super) fn transpose_from_flat<T: IntRing2k>(
    distances: &[DistanceShare<T>],
) -> Vec<Vec<DistanceShare<T>>> {
    (0..HAWK_MIN_DIST_ROTATIONS)
        .map(|i| {
            distances
                .iter()
                .skip(i)
                .step_by(HAWK_MIN_DIST_ROTATIONS)
                .cloned()
                .collect()
        })
        .collect()
}
