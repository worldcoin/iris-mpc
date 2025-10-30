use super::{
    Aby3DistanceRef, Aby3Query, Aby3Store, Aby3VectorRef, ArcIris, DistanceShare, VectorId,
};
use eyre::Result;
use iris_mpc_common::ROTATIONS;
use itertools::Itertools;

pub struct DistanceSimple;

impl DistanceSimple {
    pub async fn eval_pairwise_distances(
        store: &mut Aby3Store,
        pairs: Vec<Option<(ArcIris, ArcIris)>>,
    ) -> Result<Vec<DistanceShare<u32>>> {
        let ds_and_ts = store.workers.galois_ring_pairwise_distances(pairs).await?;

        store.gr_to_lifted_distances(ds_and_ts).await
    }

    pub async fn eval_distance_pairs(
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

    pub async fn eval_distance_batch(
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

pub struct DistanceMinimalRotation;

impl DistanceMinimalRotation {
    pub async fn eval_pairwise_distances(
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

    pub async fn eval_distance_pairs(
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

    pub async fn eval_distance_batch(
        store: &mut Aby3Store,
        query: &Aby3Query,
        vectors: &[VectorId],
    ) -> Result<Vec<DistanceShare<u32>>> {
        let ds_and_ts = store
            .workers
            .rotation_aware_dot_product_batch(query.iris_proc.clone(), vectors.to_vec())
            .await?;

        let distances = store.gr_to_lifted_distances(ds_and_ts).await?;

        store
            .oblivious_min_distance_batch(transpose_from_flat(&distances))
            .await
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
    (0..ROTATIONS)
        .map(|i| {
            distances
                .iter()
                .skip(i)
                .step_by(ROTATIONS)
                .cloned()
                .collect()
        })
        .collect()
}
