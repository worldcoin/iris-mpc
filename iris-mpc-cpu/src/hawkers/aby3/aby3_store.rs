use crate::{
    database_generators::GaloisRingSharedIris,
    execution::{
        player::Identity,
        session::{Session, SessionHandles},
    },
    hawkers::plaintext_store::PointId,
    hnsw::VectorStore,
    protocol::ops::{
        batch_signed_lift_vec, compare_threshold_and_open, cross_compare,
        galois_ring_pairwise_distance, galois_ring_to_rep3,
    },
    shares::{
        ring_impl::RingElement,
        share::{DistanceShare, Share},
    },
};
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display},
    num::ParseIntError,
    str::FromStr,
    sync::Arc,
    vec,
};
use tokio::sync::{RwLock, RwLockWriteGuard};
use tracing::instrument;

#[derive(Copy, Default, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VectorId {
    pub(crate) id: PointId,
}

impl Display for VectorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.id, f)
    }
}

impl FromStr for VectorId {
    type Err = ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(VectorId {
            id: FromStr::from_str(s)?,
        })
    }
}

impl From<PointId> for VectorId {
    fn from(id: PointId) -> Self {
        VectorId { id }
    }
}

impl From<&PointId> for VectorId {
    fn from(id: &PointId) -> Self {
        VectorId { id: *id }
    }
}

impl From<usize> for VectorId {
    fn from(id: usize) -> Self {
        VectorId { id: id.into() }
    }
}

impl VectorId {
    pub fn to_serial_id(&self) -> u32 {
        self.id.0
    }
}

#[derive(Clone, Serialize, Deserialize, Hash, Eq, PartialEq, Debug)]
pub struct Query {
    pub query: GaloisRingSharedIris,
    pub processed_query: GaloisRingSharedIris,
}

type QueryRef = Arc<Query>;

/// Creates a new query from a shared iris.
/// The input iris is preprocessed for faster evaluation of distances, see [Aby3Store::eval_distance].
pub fn prepare_query(raw_query: GaloisRingSharedIris) -> QueryRef {
    let mut preprocessed_query = raw_query.clone();
    preprocessed_query.code.preprocess_iris_code_query_share();
    preprocessed_query.mask.preprocess_mask_code_query_share();

    Arc::new(Query {
        query: raw_query,
        processed_query: preprocessed_query,
    })
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct SharedIrises {
    pub points: Vec<GaloisRingSharedIris>,
}

#[derive(Clone)]
pub struct SharedIrisesRef {
    body: Arc<RwLock<SharedIrises>>,
}

pub type SharedIrisesMut<'a> = RwLockWriteGuard<'a, SharedIrises>;

impl std::fmt::Debug for SharedIrisesRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt("SharedIrisesRef", f)
    }
}

impl Default for SharedIrisesRef {
    fn default() -> Self {
        let body = SharedIrises { points: vec![] };
        SharedIrisesRef {
            body: Arc::new(RwLock::new(body)),
        }
    }
}

impl SharedIrisesRef {
    pub fn new(data: Vec<GaloisRingSharedIris>) -> Self {
        let body = SharedIrises { points: data };
        SharedIrisesRef {
            body: Arc::new(RwLock::new(body)),
        }
    }
}

impl SharedIrisesRef {
    pub async fn write(&self) -> SharedIrisesMut {
        self.body.write().await
    }

    pub async fn get_vector(&self, vector: &VectorId) -> GaloisRingSharedIris {
        let body = self.body.read().await;
        body.points[vector.id].clone()
    }

    pub async fn iter_vectors<'a>(
        &'a self,
        vector_ids: &'a [VectorId],
    ) -> impl Iterator<Item = GaloisRingSharedIris> + 'a {
        let body = self.body.read().await;
        vector_ids.iter().map(move |v| body.points[v.id].clone())
    }

    async fn insert(&mut self, query: &QueryRef) -> VectorId {
        let mut body = self.body.write().await;
        body.points.push(query.query.clone());

        let new_id = body.points.len() - 1;
        VectorId { id: new_id.into() }
    }
}

/// Implementation of VectorStore based on the ABY3 framework (<https://eprint.iacr.org/2018/403.pdf>).
///
/// Note that all SMPC operations are performed in a single session.
#[derive(Debug, Clone)]
pub struct Aby3Store {
    /// Identity of the party performing computations in this store
    pub owner: Identity,
    /// Reference to the shared irises
    pub storage: SharedIrisesRef,
    /// Session for the SMPC operations
    pub session: Session,
}

impl Aby3Store {
    /// Returns the index of the party in the session, which is used to propagate messages to the correct party.
    /// The index must be in the range [0, 2] and unique per party.
    pub fn get_owner_index(&self) -> usize {
        self.session.boot_session.own_role().unwrap().index()
    }
}

impl Aby3Store {
    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    pub async fn lift_distances(
        &mut self,
        distances: Vec<Share<u16>>,
    ) -> eyre::Result<Vec<DistanceShare<u32>>> {
        if distances.is_empty() {
            return Ok(vec![]);
        }
        let distances = batch_signed_lift_vec(&mut self.session, distances).await?;
        Ok(distances
            .chunks(2)
            .map(|dot_products| {
                DistanceShare::new(dot_products[0].clone(), dot_products[1].clone())
            })
            .collect::<Vec<_>>())
    }

    /// Assumes that the first iris of each pair is preprocessed.
    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn eval_pairwise_distances(
        &mut self,
        pairs: Vec<(GaloisRingSharedIris, GaloisRingSharedIris)>,
    ) -> Vec<Share<u16>> {
        if pairs.is_empty() {
            return vec![];
        }
        let ds_and_ts = galois_ring_pairwise_distance(&mut self.session, &pairs)
            .await
            .unwrap();
        galois_ring_to_rep3(&mut self.session, ds_and_ts)
            .await
            .unwrap()
    }
}

impl VectorStore for Aby3Store {
    type QueryRef = QueryRef; // Arc ref to a query.
    type VectorRef = VectorId; // Point ID of an inserted iris.
    type DistanceRef = DistanceShare<u32>; // Distance represented as shares.

    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        self.storage.insert(query).await
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Self::DistanceRef {
        let vector_point = self.storage.get_vector(vector).await;
        let pairs = vec![(query.processed_query.clone(), vector_point)];
        let dist = self.eval_pairwise_distances(pairs).await;
        self.lift_distances(dist).await.unwrap()[0].clone()
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all, fields(batch_size = vectors.len()))]
    async fn eval_distance_batch(
        &mut self,
        query: &Self::QueryRef,
        vectors: &[Self::VectorRef],
    ) -> Vec<Self::DistanceRef> {
        if vectors.is_empty() {
            return vec![];
        }
        let pairs = self
            .storage
            .iter_vectors(vectors)
            .await
            .map(|vector| (query.processed_query.clone(), vector))
            .collect::<Vec<_>>();

        let dist = self.eval_pairwise_distances(pairs).await;
        self.lift_distances(dist).await.unwrap()
    }

    async fn is_match(&mut self, distance: &Self::DistanceRef) -> bool {
        compare_threshold_and_open(&mut self.session, distance.clone())
            .await
            .unwrap()
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn less_than(
        &mut self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> bool {
        let code_dot1 = distance1.code_dot.clone();
        let mask_dot1 = distance1.mask_dot.clone();
        let code_dot2 = distance2.code_dot.clone();
        let mask_dot2 = distance2.mask_dot.clone();
        cross_compare(
            &mut self.session,
            code_dot1,
            mask_dot1,
            code_dot2,
            mask_dot2,
        )
        .await
        .unwrap()
    }
}

impl Aby3Store {
    pub fn get_trivial_share(&self, distance: u16) -> Share<u32> {
        let player = self.get_owner_index();
        let distance_elem = RingElement(distance as u32);
        let zero_elem = RingElement(0_u32);

        match player {
            0 => Share::new(distance_elem, zero_elem),
            1 => Share::new(zero_elem, distance_elem),
            2 => Share::new(zero_elem, zero_elem),
            _ => panic!("Invalid player index"),
        }
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    pub(crate) async fn eval_distance_vectors(
        &mut self,
        vector1: &<Aby3Store as VectorStore>::VectorRef,
        vector2: &<Aby3Store as VectorStore>::VectorRef,
    ) -> <Aby3Store as VectorStore>::DistanceRef {
        let point1 = self.storage.get_vector(vector1).await;
        let mut point2 = self.storage.get_vector(vector2).await;
        point2.code.preprocess_iris_code_query_share();
        point2.mask.preprocess_mask_code_query_share();
        let pairs = vec![(point1.clone(), point2)];
        let dist = self.eval_pairwise_distances(pairs).await;
        self.lift_distances(dist).await.unwrap()[0].clone()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{
        database_generators::generate_galois_iris_shares,
        hawkers::{
            aby3::test_utils::{
                lazy_random_setup, setup_local_store_aby3_players, shared_random_setup,
            },
            plaintext_store::PlaintextStore,
        },
        hnsw::{GraphMem, HnswSearcher},
        network::NetworkType,
    };
    use aes_prng::AesRng;
    use iris_mpc_common::iris_db::db::IrisDB;
    use itertools::Itertools;
    use rand::SeedableRng;
    use tokio::task::JoinSet;
    use tracing_test::traced_test;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_gr_hnsw() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 10;
        let cleartext_database = IrisDB::new_random_rng(database_size, &mut rng).db;
        let shared_irises: Vec<_> = cleartext_database
            .iter()
            .map(|iris| generate_galois_iris_shares(&mut rng, iris.clone()))
            .collect();

        let mut stores = setup_local_store_aby3_players(NetworkType::LocalChannel)
            .await
            .unwrap();

        let mut jobs = JoinSet::new();
        for store in stores.iter_mut() {
            let player_index = store.get_owner_index();
            let queries = (0..database_size)
                .map(|id| prepare_query(shared_irises[id][player_index].clone()))
                .collect::<Vec<_>>();
            let mut store = store.clone();
            let mut rng = rng.clone();
            jobs.spawn(async move {
                let mut aby3_graph = GraphMem::new();
                let db = HnswSearcher::default();

                let mut inserted = vec![];
                // insert queries
                for query in queries.iter() {
                    let inserted_vector = db
                        .insert(&mut store, &mut aby3_graph, query, &mut rng)
                        .await;
                    inserted.push(inserted_vector)
                }
                tracing::debug!("FINISHED INSERTING");
                // Search for the same codes and find matches.
                let mut matching_results = vec![];
                for v in inserted.into_iter() {
                    let query = prepare_query(store.storage.get_vector(&v).await);
                    let neighbors = db.search(&mut store, &mut aby3_graph, &query, 1).await;
                    tracing::debug!("Finished checking query");
                    matching_results.push(db.is_match(&mut store, &[neighbors]).await)
                }
                matching_results
            });
        }
        let matching_results = jobs.join_all().await;
        for (party_id, party_results) in matching_results.iter().enumerate() {
            for (index, result) in party_results.iter().enumerate() {
                assert!(
                    *result,
                    "Failed at index {:?} for party {:?}",
                    index, party_id
                );
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_gr_premade_hnsw() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 10;
        let network_t = NetworkType::LocalChannel;
        let (mut cleartext_data, secret_data) =
            lazy_random_setup(&mut rng, database_size, network_t.clone(), true)
                .await
                .unwrap();

        let mut rng = AesRng::seed_from_u64(0_u64);
        let vector_graph_stores = shared_random_setup(&mut rng, database_size, network_t)
            .await
            .unwrap();

        for ((v_from_scratch, _), (premade_v, _)) in
            vector_graph_stores.iter().zip(secret_data.iter())
        {
            assert_eq!(
                v_from_scratch.storage.body.read().await.points,
                premade_v.storage.body.read().await.points
            );
        }
        let hawk_searcher = HnswSearcher::default();

        for i in 0..database_size {
            let cleartext_neighbors = hawk_searcher
                .search(&mut cleartext_data.0, &mut cleartext_data.1, &i.into(), 1)
                .await;
            assert!(
                hawk_searcher
                    .is_match(&mut cleartext_data.0, &[cleartext_neighbors])
                    .await,
            );

            let mut jobs = JoinSet::new();
            for (v, g) in vector_graph_stores.iter() {
                let hawk_searcher = hawk_searcher.clone();
                let mut v = v.clone();
                let mut g = g.clone();
                let q = prepare_query(v.storage.get_vector(&i.into()).await);
                jobs.spawn(async move {
                    let secret_neighbors = hawk_searcher.search(&mut v, &mut g, &q, 1).await;

                    hawk_searcher.is_match(&mut v, &[secret_neighbors]).await
                });
            }
            let scratch_results = jobs.join_all().await;

            let mut jobs = JoinSet::new();
            for (v, g) in secret_data.iter() {
                let hawk_searcher = hawk_searcher.clone();
                let mut v = v.clone();
                let mut g = g.clone();
                jobs.spawn(async move {
                    let query = prepare_query(v.storage.get_vector(&i.into()).await);
                    let secret_neighbors = hawk_searcher.search(&mut v, &mut g, &query, 1).await;

                    hawk_searcher.is_match(&mut v, &[secret_neighbors]).await
                });
            }
            let premade_results = jobs.join_all().await;

            for (premade_res, scratch_res) in scratch_results.iter().zip(premade_results.iter()) {
                assert!(*premade_res && *scratch_res);
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_gr_aby3_store_plaintext() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let db_dim = 4;
        let cleartext_database = IrisDB::new_random_rng(db_dim, &mut rng).db;
        let shared_irises: Vec<_> = cleartext_database
            .iter()
            .map(|iris| generate_galois_iris_shares(&mut rng, iris.clone()))
            .collect();
        let mut local_stores = setup_local_store_aby3_players(NetworkType::LocalChannel)
            .await
            .unwrap();
        // Now do the work for the plaintext store
        let mut plaintext_store = PlaintextStore::default();
        let plaintext_preps: Vec<_> = (0..db_dim)
            .map(|id| plaintext_store.prepare_query(cleartext_database[id].clone()))
            .collect();
        let mut plaintext_inserts = Vec::new();
        for p in plaintext_preps.iter() {
            plaintext_inserts.push(plaintext_store.insert(p).await);
        }

        // pairs of indices to compare
        let it1 = (0..db_dim).combinations(2);
        let it2 = (0..db_dim).combinations(2);

        let mut plain_results = HashMap::new();
        for comb1 in it1.clone() {
            for comb2 in it2.clone() {
                // compute distances in plaintext
                let dist1_plain = plaintext_store
                    .eval_distance(&plaintext_inserts[comb1[0]], &plaintext_inserts[comb1[1]])
                    .await;
                let dist2_plain = plaintext_store
                    .eval_distance(&plaintext_inserts[comb2[0]], &plaintext_inserts[comb2[1]])
                    .await;
                let bit = plaintext_store.less_than(&dist1_plain, &dist2_plain).await;
                plain_results.insert((comb1.clone(), comb2.clone()), bit);
            }
        }

        let mut aby3_inserts = vec![];
        for store in local_stores.iter_mut() {
            let player_index = store.get_owner_index();
            let player_preps: Vec<_> = (0..db_dim)
                .map(|id| prepare_query(shared_irises[id][player_index].clone()))
                .collect();
            let mut player_inserts = vec![];
            for p in player_preps.iter() {
                player_inserts.push(store.insert(p).await);
            }
            aby3_inserts.push(player_inserts);
        }

        for comb1 in it1 {
            for comb2 in it2.clone() {
                let mut jobs = JoinSet::new();
                for store in local_stores.iter() {
                    let player_index = store.get_owner_index();
                    let player_inserts = aby3_inserts[player_index].clone();
                    let mut store = store.clone();
                    let index10 = comb1[0];
                    let index11 = comb1[1];
                    let index20 = comb2[0];
                    let index21 = comb2[1];
                    jobs.spawn(async move {
                        let dist1_aby3 = store
                            .eval_distance_vectors(
                                &player_inserts[index10],
                                &player_inserts[index11],
                            )
                            .await;
                        let dist2_aby3 = store
                            .eval_distance_vectors(
                                &player_inserts[index20],
                                &player_inserts[index21],
                            )
                            .await;
                        store.less_than(&dist1_aby3, &dist2_aby3).await
                    });
                }
                let res = jobs.join_all().await;
                for bit in res {
                    assert_eq!(
                        bit,
                        plain_results[&(comb1.clone(), comb2.clone())],
                        "Failed at combo: {:?}, {:?}",
                        comb1,
                        comb2
                    )
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_gr_scratch_hnsw() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 2;
        let searcher = HnswSearcher::default();
        let mut vectors_and_graphs =
            shared_random_setup(&mut rng, database_size, NetworkType::LocalChannel)
                .await
                .unwrap();

        for i in 0..database_size {
            let mut jobs = JoinSet::new();
            for (store, graph) in vectors_and_graphs.iter_mut() {
                let mut store = store.clone();
                let mut graph = graph.clone();
                let searcher = searcher.clone();
                let q = prepare_query(store.storage.get_vector(&i.into()).await);
                jobs.spawn(async move {
                    let secret_neighbors = searcher.search(&mut store, &mut graph, &q, 1).await;
                    searcher.is_match(&mut store, &[secret_neighbors]).await
                });
            }
            let res = jobs.join_all().await;
            for (party_index, r) in res.iter().enumerate() {
                assert!(r, "Failed at index {:?} by party {:?}", i, party_index);
            }
        }
    }
}
