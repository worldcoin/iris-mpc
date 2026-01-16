use crate::execution::hawk_main::HAWK_MINFHD_ROTATIONS;

use super::{
    Aby3DistanceRef, Aby3Query, Aby3Store, Aby3VectorRef, ArcIris, DistanceShare, VectorId,
};
use clap::ValueEnum;
use eyre::Result;
use iris_mpc_common::iris_db::iris::IrisCode;
use itertools::Itertools;

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
        let ds_and_ts = store
            .workers
            .dot_product_batch(query.iris_proc.clone(), vectors.to_vec())
            .await?;
        store.gr_to_lifted_distances(ds_and_ts).await
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
        let total_start = std::time::Instant::now();
        let ds_and_ts = store
            .workers
            .rotation_aware_dot_product_batch(query.iris_proc.clone(), vectors.to_vec())
            .await?;
        let distances = store.gr_to_lifted_distances(ds_and_ts).await?;
        let oblivious_min_start = std::time::Instant::now();
        let result = store
            .oblivious_min_distance_batch(transpose_from_flat(&distances))
            .await;
        let oblivious_min_duration = oblivious_min_start.elapsed();
        let total_duration = total_start.elapsed();
        if !total_duration.is_zero() {
            let oblivious_min_percent =
                (oblivious_min_duration.as_secs_f64() / total_duration.as_secs_f64()) * 100.0;
            metrics::histogram!("eval_distance_oblivious_min_percent")
                .record(oblivious_min_percent);
        }
        result
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
