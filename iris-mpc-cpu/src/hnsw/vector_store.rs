use eyre::{OptionExt, Result};
use serde::Serialize;
use std::{
    fmt::{Debug, Display},
    hash::Hash,
    str::FromStr,
};

pub trait TransientRef: Clone + Debug + PartialEq + Eq + Hash + Sync {}

impl<T> TransientRef for T where T: Clone + Debug + PartialEq + Eq + Hash + Sync {}

pub trait Ref:
    Send
    + Sync
    + Clone
    + Debug
    + PartialEq
    + Eq
    + Hash
    + Sync
    + Serialize
    + for<'de> serde::Deserialize<'de>
{
}

impl<T> Ref for T where
    T: Send
        + Sync
        + Clone
        + Debug
        + PartialEq
        + Eq
        + Hash
        + Sync
        + Serialize
        + for<'de> serde::Deserialize<'de>
{
}

/// The operations exposed by a vector store, sufficient for a search algorithm.
#[allow(async_fn_in_trait)]
pub trait VectorStore: Debug {
    /// Opaque reference to a query.
    ///
    /// Example: a preprocessed representation optimized for distance
    /// evaluations.
    type QueryRef: TransientRef;

    /// Opaque reference to a stored vector.
    ///
    /// Example: a vector ID.
    type VectorRef: Ref + Display + FromStr + Ord;

    /// Opaque reference to a distance metric.
    ///
    /// Example: an encrypted distance.
    type DistanceRef: Ref;

    /// Evaluate the distance between a query and a vector.
    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Result<Self::DistanceRef>;

    /// Check whether a distance is a match, meaning the query is considered
    /// equivalent to a previously inserted vector.
    async fn is_match(&mut self, distance: &Self::DistanceRef) -> Result<bool>;

    /// Compare two distances.
    async fn less_than(
        &mut self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> Result<bool>;

    // Batch variants

    /// Prepare queries from vectors. The query form may include some precomputation
    /// to help comparison to other vectors.
    async fn vectors_as_queries(&mut self, vectors: Vec<Self::VectorRef>) -> Vec<Self::QueryRef>;

    /// Retain only vectors that are valid and currently usable by other methods.
    async fn only_valid_vectors(&mut self, vectors: Vec<Self::VectorRef>) -> Vec<Self::VectorRef> {
        vectors
    }

    /// Evaluate the distance between pairs of (query, vector), in batch.
    /// The default implementation is a loop over `eval_distance`.
    /// Override for more efficient batch distance evaluations.
    async fn eval_distance_pairs(
        &mut self,
        pairs: &[(Self::QueryRef, Self::VectorRef)],
    ) -> Result<Vec<Self::DistanceRef>> {
        let mut results = Vec::with_capacity(pairs.len());
        for (query, vector) in pairs {
            results.push(self.eval_distance(query, vector).await?);
        }
        Ok(results)
    }

    /// Evaluate the distances between the query and a batch of vectors.
    /// The default implementation is a loop over `eval_distance`.
    /// Override for more efficient batch distance evaluations.
    async fn eval_distance_batch(
        &mut self,
        query: &Self::QueryRef,
        vectors: &[Self::VectorRef],
    ) -> Result<Vec<Self::DistanceRef>> {
        let mut results = Vec::with_capacity(vectors.len());
        for vector in vectors {
            results.push(self.eval_distance(query, vector).await?);
        }
        Ok(results)
    }

    /// Check whether a batch of distances are matches.
    /// The default implementation is a loop over `is_match`.
    /// Override for more efficient batch match checks.
    async fn is_match_batch(&mut self, distances: &[Self::DistanceRef]) -> Result<Vec<bool>> {
        let mut results = Vec::with_capacity(distances.len());
        for distance in distances {
            results.push(self.is_match(distance).await?);
        }
        Ok(results)
    }

    /// Compare pairs of distances in batch. For each pair (a, b),
    /// return the boolean `a < b`.
    /// The default implementation is a loop over `less_than`.
    /// Override for more efficient batch comparisons.
    async fn less_than_batch(
        &mut self,
        distances: &[(Self::DistanceRef, Self::DistanceRef)],
    ) -> Result<Vec<bool>> {
        let mut results: Vec<bool> = Vec::with_capacity(distances.len());
        for (d1, d2) in distances {
            results.push(self.less_than(d1, d2).await?);
        }
        Ok(results)
    }

    async fn get_argmin_distance(
        &mut self,
        distances: &[(Self::VectorRef, Self::DistanceRef)],
    ) -> Result<(Self::VectorRef, Self::DistanceRef)> {
        let mut min_dist = distances
            .first()
            .ok_or_eyre("Cannot get min of empty list")
            .cloned()?;

        for (id, dist) in distances.iter().skip(1) {
            if self.less_than(dist, &min_dist.1).await? {
                min_dist = (id.clone(), dist.clone());
            }
        }
        Ok(min_dist)
    }
}

/// The operations exposed by a vector store, including mutations.
#[allow(async_fn_in_trait)]
pub trait VectorStoreMut: VectorStore {
    /// Persist a query as a new vector in the store, and return a reference to it.
    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef;

    /// Persist a query as a vector in the store with specified vector reference.
    ///
    /// Returns an Err output when a specified insertion is not supported.
    async fn insert_at(
        &mut self,
        vector_ref: &Self::VectorRef,
        query: &Self::QueryRef,
    ) -> Result<Self::VectorRef>;
}
