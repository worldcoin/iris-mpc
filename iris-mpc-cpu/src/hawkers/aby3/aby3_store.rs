use crate::{
    execution::{player::Identity, session::Session},
    hawkers::plaintext_store::PointId,
    hnsw::{vector_store::VectorStoreMut, VectorStore},
    protocol::{
        ops::{
            batch_signed_lift_vec, compare_threshold_and_open, cross_compare,
            galois_ring_pairwise_distance, galois_ring_to_rep3,
        },
        shared_iris::GaloisRingSharedIris,
    },
    shares::share::{DistanceShare, Share},
};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    num::ParseIntError,
    str::FromStr,
    sync::Arc,
    vec,
};
use tokio::sync::{RwLock, RwLockWriteGuard};
use tracing::instrument;

/// Reference to an iris in the Shamir secret shared form over a Galois ring.
pub type IrisRef = Arc<GaloisRingSharedIris>;

/// Unique identifier for an iris inserted into the store.
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
    pub fn from_serial_id(id: u32) -> Self {
        VectorId { id: id.into() }
    }

    /// Returns the ID of a vector as a number.
    pub fn to_serial_id(&self) -> u32 {
        self.id.0
    }
}

/// Iris to be searcher or inserted into the store.
#[derive(Clone, Serialize, Deserialize, Hash, Eq, PartialEq, Debug)]
pub struct Query {
    /// Iris in the Shamir secret shared form over a Galois ring.
    pub query: GaloisRingSharedIris,
    /// Preprocessed iris for faster evaluation of distances, see [Aby3Store::eval_distance].
    pub processed_query: GaloisRingSharedIris,
}

/// Reference to a query.
pub type QueryRef = Arc<Query>;

impl Query {
    pub fn from_processed(
        query: GaloisRingSharedIris,
        processed_query: GaloisRingSharedIris,
    ) -> QueryRef {
        Arc::new(Query {
            query,
            processed_query,
        })
    }
}

/// Creates a new query from a secret shared iris.
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

/// Storage of inserted irises.
#[derive(Clone, Serialize, Deserialize)]
pub struct SharedIrises {
    points: HashMap<VectorId, IrisRef>,
    next_id: u32,
    empty_iris: IrisRef,
}

impl SharedIrises {
    pub fn insert(&mut self, vector_id: VectorId, iris: IrisRef) {
        self.points.insert(vector_id, iris);
        self.next_id = self.next_id.max(vector_id.to_serial_id() + 1);
    }

    fn next_id(&mut self) -> VectorId {
        let new_id = VectorId {
            id: PointId(self.next_id),
        };
        self.next_id += 1;
        new_id
    }

    pub fn reserve(&mut self, additional: usize) {
        self.points.reserve(additional);
    }

    fn get_vector(&self, vector: &VectorId) -> IrisRef {
        // TODO: Handle missing vectors.
        Arc::clone(self.points.get(vector).unwrap_or(&self.empty_iris))
    }
}

/// Reference to inserted irises.
#[derive(Clone)]
pub struct SharedIrisesRef {
    body: Arc<RwLock<SharedIrises>>,
}

/// Mutable reference to inserted irises.
pub type SharedIrisesMut<'a> = RwLockWriteGuard<'a, SharedIrises>;

impl std::fmt::Debug for SharedIrisesRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt("SharedIrisesRef", f)
    }
}

impl Default for SharedIrisesRef {
    fn default() -> Self {
        SharedIrisesRef::new(HashMap::new())
    }
}

// Constructor.
impl SharedIrisesRef {
    pub fn new(points: HashMap<VectorId, IrisRef>) -> Self {
        let next_id = points
            .keys()
            .map(|v| v.to_serial_id() + 1)
            .max()
            .unwrap_or(0);
        let body = SharedIrises {
            points,
            next_id,
            empty_iris: Arc::new(GaloisRingSharedIris::default_for_party(0)),
        };
        SharedIrisesRef {
            body: Arc::new(RwLock::new(body)),
        }
    }
}

// Getters, iterators and mutators of the iris storage.
impl SharedIrisesRef {
    pub async fn write(&self) -> SharedIrisesMut {
        self.body.write().await
    }

    pub async fn get_vector(&self, vector: &VectorId) -> IrisRef {
        self.body.read().await.get_vector(vector)
    }

    pub async fn iter_vectors<'a>(&'a self, vector_ids: &'a [VectorId]) -> Vec<IrisRef> {
        let body = self.body.read().await;
        vector_ids.iter().map(|v| body.get_vector(v)).collect_vec()
    }

    pub async fn insert(&mut self, query: &QueryRef) -> VectorId {
        let new_vector = Arc::new(query.query.clone());
        let mut body = self.body.write().await;
        let new_id = body.next_id();
        body.insert(new_id, new_vector);
        new_id
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
    /// Converts distances from u16 secret shares to u32 shares.
    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    pub(crate) async fn lift_distances(
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

    /// Computes the dot product of the iris codes and masks of the given pairs of irises.
    /// The input irises are given in the Shamir secret sharing scheme, while the output distances are additive replicated secret shares used in the ABY3 framework.
    ///
    /// Assumes that the first iris of each pair is preprocessed.
    /// This first iris is usually preprocessed when a related query is created, see [prepare_query] for more details.
    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    pub(crate) async fn eval_pairwise_distances(
        &mut self,
        pairs: Vec<(&GaloisRingSharedIris, &GaloisRingSharedIris)>,
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
    /// Arc ref to a query.
    type QueryRef = QueryRef;
    /// Point ID of an inserted iris.
    type VectorRef = VectorId;
    /// Distance represented as a pair of u32 shares.
    type DistanceRef = DistanceShare<u32>;

    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Self::DistanceRef {
        let vector_point = self.storage.get_vector(vector).await;
        let pairs = vec![(&query.processed_query, &*vector_point)];
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
        let vectors = self.storage.iter_vectors(vectors).await;
        let pairs = vectors
            .iter()
            .map(|vector| (&query.processed_query, &**vector))
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
        cross_compare(&mut self.session, &[(distance1.clone(), distance2.clone())])
            .await
            .unwrap()[0]
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all, fields(batch_size = distances.len()))]
    async fn less_than_batch(
        &mut self,
        distances: &[(Self::DistanceRef, Self::DistanceRef)],
    ) -> Vec<bool> {
        if distances.is_empty() {
            return vec![];
        }
        cross_compare(&mut self.session, distances).await.unwrap()
    }
}

impl VectorStoreMut for Aby3Store {
    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        self.storage.insert(query).await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{
        hawkers::{
            aby3::test_utils::{
                eval_vector_distance, get_owner_index, lazy_random_setup,
                setup_local_store_aby3_players, shared_random_setup,
            },
            plaintext_store::PlaintextStore,
        },
        hnsw::{GraphMem, HnswSearcher},
        network::NetworkType,
        protocol::shared_iris::GaloisRingSharedIris,
    };
    use aes_prng::AesRng;
    use iris_mpc_common::iris_db::db::IrisDB;
    use itertools::Itertools;
    use rand::SeedableRng;
    use tokio::task::JoinSet;
    use tracing_test::traced_test;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_gr_hnsw() -> eyre::Result<()> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 10;
        let cleartext_database = IrisDB::new_random_rng(database_size, &mut rng).db;
        let shared_irises: Vec<_> = cleartext_database
            .iter()
            .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.clone()))
            .collect();

        let mut stores = setup_local_store_aby3_players(NetworkType::LocalChannel).await?;

        let mut jobs = JoinSet::new();
        for store in stores.iter_mut() {
            let player_index = get_owner_index(store)?;
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
                    let query = store.storage.get_vector(&v).await;
                    let query = prepare_query((*query).clone());
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
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_gr_premade_hnsw() -> eyre::Result<()> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 10;
        let network_t = NetworkType::LocalChannel;
        let (mut cleartext_data, secret_data) =
            lazy_random_setup(&mut rng, database_size, network_t.clone(), true).await?;

        let mut rng = AesRng::seed_from_u64(0_u64);
        let vector_graph_stores = shared_random_setup(&mut rng, database_size, network_t).await?;

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
                let q = v.storage.get_vector(&i.into()).await;
                let q = prepare_query((*q).clone());
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
                    let query = v.storage.get_vector(&i.into()).await;
                    let query = prepare_query((*query).clone());
                    let secret_neighbors = hawk_searcher.search(&mut v, &mut g, &query, 1).await;

                    hawk_searcher.is_match(&mut v, &[secret_neighbors]).await
                });
            }
            let premade_results = jobs.join_all().await;

            for (premade_res, scratch_res) in scratch_results.iter().zip(premade_results.iter()) {
                assert!(*premade_res && *scratch_res);
            }
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_gr_aby3_store_plaintext() -> eyre::Result<()> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let db_dim = 4;
        let cleartext_database = IrisDB::new_random_rng(db_dim, &mut rng).db;
        let shared_irises: Vec<_> = cleartext_database
            .iter()
            .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.clone()))
            .collect();
        let mut local_stores = setup_local_store_aby3_players(NetworkType::LocalChannel).await?;
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
            let player_index = get_owner_index(store)?;
            let player_preps: Vec<_> = (0..db_dim)
                .map(|id| prepare_query(shared_irises[id][player_index].clone()))
                .collect();
            let mut player_inserts = vec![];
            for p in player_preps.iter() {
                player_inserts.push(store.storage.insert(p).await);
            }
            aby3_inserts.push(player_inserts);
        }

        for comb1 in it1 {
            for comb2 in it2.clone() {
                let mut jobs = JoinSet::new();
                for store in local_stores.iter() {
                    let player_index = get_owner_index(store)?;
                    let player_inserts = aby3_inserts[player_index].clone();
                    let mut store = store.clone();
                    let index10 = comb1[0];
                    let index11 = comb1[1];
                    let index20 = comb2[0];
                    let index21 = comb2[1];
                    jobs.spawn(async move {
                        let dist1_aby3 = eval_vector_distance(
                            &mut store,
                            &player_inserts[index10],
                            &player_inserts[index11],
                        )
                        .await;
                        let dist2_aby3 = eval_vector_distance(
                            &mut store,
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
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_gr_aby3_store_plaintext_batch() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let db_size = 10;
        let cleartext_database = IrisDB::new_random_rng(db_size, &mut rng).db;
        let shared_irises: Vec<_> = cleartext_database
            .iter()
            .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.clone()))
            .collect();
        let mut local_stores = setup_local_store_aby3_players(NetworkType::LocalChannel)
            .await
            .unwrap();
        // Now do the work for the plaintext store
        let mut plaintext_store = PlaintextStore::default();
        let plaintext_preps: Vec<_> = (0..db_size)
            .map(|id| plaintext_store.prepare_query(cleartext_database[id].clone()))
            .collect();
        let mut plaintext_inserts = Vec::with_capacity(db_size);
        for p in plaintext_preps.iter() {
            plaintext_inserts.push(plaintext_store.insert(p).await);
        }

        // compute distances in plaintext
        let dist1_plain = plaintext_store
            .eval_distance_batch(&plaintext_inserts[0], &plaintext_inserts)
            .await;
        let dist2_plain = plaintext_store
            .eval_distance_batch(&plaintext_inserts[1], &plaintext_inserts)
            .await;
        let dist_plain = dist1_plain
            .into_iter()
            .zip(dist2_plain.into_iter())
            .collect::<Vec<_>>();
        let bits_plain = plaintext_store.less_than_batch(&dist_plain).await;

        let mut aby3_inserts = vec![];
        let mut queries = vec![];
        for store in local_stores.iter_mut() {
            let player_index = get_owner_index(store).unwrap();
            let player_preps: Vec<_> = (0..db_size)
                .map(|id| prepare_query(shared_irises[id][player_index].clone()))
                .collect();
            queries.push(player_preps.clone());
            let mut player_inserts = vec![];
            for p in player_preps.iter() {
                player_inserts.push(store.insert(p).await);
            }
            aby3_inserts.push(player_inserts);
        }

        let mut jobs = JoinSet::new();
        for store in local_stores.iter() {
            let player_index = get_owner_index(store).unwrap();
            let player_inserts = aby3_inserts[player_index].clone();
            let player_preps = queries[player_index].clone();
            let mut store = store.clone();
            jobs.spawn(async move {
                let dist1_aby3 = store
                    .eval_distance_batch(&player_preps[0], &player_inserts)
                    .await;
                let dist2_aby3 = store
                    .eval_distance_batch(&player_preps[1], &player_inserts)
                    .await;
                let dist_aby3 = dist1_aby3
                    .into_iter()
                    .zip(dist2_aby3.into_iter())
                    .collect::<Vec<_>>();
                store.less_than_batch(&dist_aby3).await
            });
        }
        let bits_aby3 = jobs.join_all().await;

        assert_eq!(bits_aby3[0], bits_aby3[1]);
        assert_eq!(bits_aby3[0], bits_aby3[2]);
        assert_eq!(bits_aby3[0], bits_plain);
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
                let q = store.storage.get_vector(&i.into()).await;
                let q = prepare_query((*q).clone());
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
