use crate::{
    execution::session::Session,
    hawkers::shared_irises::{SharedIrises, SharedIrisesRef},
    hnsw::{vector_store::VectorStoreMut, VectorStore},
    protocol::{
        ops::{
            batch_signed_lift_vec, cross_compare, galois_ring_pairwise_distance,
            galois_ring_to_rep3, lte_threshold_and_open,
        },
        shared_iris::GaloisRingSharedIris,
    },
    shares::share::{DistanceShare, Share},
};
use eyre::Result;
use itertools::{izip, Itertools};
use std::{collections::HashMap, fmt::Debug, sync::Arc, vec};
use tracing::instrument;

use iris_mpc_common::vector_id::VectorId;

/// Iris to be searcher or inserted into the store.
///
/// This is an iris reference along with cached preprocessed version, used for
/// efficient Galois ring MPC comparison.
#[derive(Clone, Hash, Eq, PartialEq, Debug)]
pub struct Aby3Query {
    /// Iris in the Shamir secret shared form over a Galois ring.
    pub iris: Arc<GaloisRingSharedIris>,

    /// Preprocessed iris for faster evaluation of distances; see [Aby3Store::eval_distance].
    pub iris_proc: Arc<GaloisRingSharedIris>,
}

impl Aby3Query {
    /// Creates a new query from a secret shared iris. The input iris is preprocessed for
    /// faster evaluation of distances; see [Aby3Store::eval_distance].
    pub fn new(iris_ref: &Arc<GaloisRingSharedIris>) -> Self {
        let iris = iris_ref.clone();

        let mut preprocessed = (**iris_ref).clone();
        preprocessed.code.preprocess_iris_code_query_share();
        preprocessed.mask.preprocess_mask_code_query_share();
        let iris_proc = Arc::new(preprocessed);

        Self { iris, iris_proc }
    }

    pub fn new_from_raw(iris: GaloisRingSharedIris) -> Self {
        let iris = Arc::new(iris);
        Self::new(&iris)
    }

    pub fn from_processed(iris: GaloisRingSharedIris, iris_proc: GaloisRingSharedIris) -> Self {
        Self {
            iris: Arc::new(iris),
            iris_proc: Arc::new(iris_proc),
        }
    }
}

pub type Aby3StoredIris = Arc<GaloisRingSharedIris>;

pub type Aby3SharedIrises = SharedIrises<Aby3StoredIris>;
pub type Aby3SharedIrisesRef = SharedIrisesRef<Aby3StoredIris>;

/// Implementation of VectorStore based on the ABY3 framework (<https://eprint.iacr.org/2018/403.pdf>).
///
/// Note that all SMPC operations are performed in a single session.
#[derive(Debug)]
pub struct Aby3Store {
    /// Reference to the shared irises
    pub storage: Aby3SharedIrisesRef,

    /// Session for the SMPC operations
    pub session: Session,
}

impl Aby3Store {
    /// Converts distances from u16 secret shares to u32 shares.
    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    pub(crate) async fn lift_distances(
        &mut self,
        distances: Vec<Share<u16>>,
    ) -> Result<Vec<DistanceShare<u32>>> {
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
        pairs: &[Option<(&GaloisRingSharedIris, &GaloisRingSharedIris)>],
    ) -> Result<Vec<Share<u16>>> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }
        let ds_and_ts = galois_ring_pairwise_distance(pairs).await;
        galois_ring_to_rep3(&mut self.session, ds_and_ts).await
    }

    /// Create a new `Aby3SharedIrises` storage using the specified points mapping.
    pub fn new_storage(points: Option<HashMap<VectorId, Aby3StoredIris>>) -> Aby3SharedIrises {
        SharedIrises::new(
            points.unwrap_or_default(),
            Arc::new(GaloisRingSharedIris::default_for_party(0)),
        )
    }
}

impl VectorStore for Aby3Store {
    /// Arc ref to a query.
    type QueryRef = Aby3Query;
    /// Point ID of an inserted iris.
    type VectorRef = VectorId;
    /// Distance represented as a pair of u32 shares.
    type DistanceRef = DistanceShare<u32>;

    async fn vectors_as_queries(&mut self, vectors: Vec<Self::VectorRef>) -> Vec<Self::QueryRef> {
        self.storage
            .get_vectors_or_empty(&vectors)
            .await
            .iter()
            .map(Aby3Query::new)
            .collect_vec()
    }

    async fn only_valid_vectors(
        &mut self,
        mut vectors: Vec<Self::VectorRef>,
    ) -> Vec<Self::VectorRef> {
        let storage = self.storage.read().await;
        vectors.retain(|v| storage.contains(v));
        vectors
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Result<Self::DistanceRef> {
        let vector_point = self.storage.get_vector(vector).await;
        let pairs = if let Some(v) = &vector_point {
            &[Some((&*query.iris_proc, &**v))]
        } else {
            &[None]
        };
        let dist = self.eval_pairwise_distances(pairs).await?;
        Ok(self.lift_distances(dist).await?[0].clone())
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all, fields(queries = pairs.len(), batch_size = pairs.len()))]
    async fn eval_distance_pairs(
        &mut self,
        pairs: &[(Self::QueryRef, Self::VectorRef)],
    ) -> Result<Vec<Self::DistanceRef>> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }
        let vectors = self.storage.get_vectors(pairs.iter().map(|(_, v)| v)).await;

        let pairs = izip!(pairs, &vectors)
            .map(|((q, _), vector)| vector.as_ref().map(|v| (&*q.iris_proc, &**v)))
            .collect_vec();

        let dist = self.eval_pairwise_distances(&pairs).await?;
        self.lift_distances(dist).await
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all, fields(queries = queries.len(), batch_size = vectors.len()))]
    async fn eval_distance_batch(
        &mut self,
        queries: &[Self::QueryRef],
        vectors: &[Self::VectorRef],
    ) -> Result<Vec<Self::DistanceRef>> {
        if vectors.is_empty() {
            return Ok(vec![]);
        }
        let vectors = self.storage.get_vectors(vectors).await;
        let pairs = queries
            .iter()
            .flat_map(|q| {
                vectors
                    .iter()
                    .map(|vector| vector.as_ref().map(|v| (&*q.iris_proc, &**v)))
            })
            .collect::<Vec<_>>();

        let dist = self.eval_pairwise_distances(&pairs).await?;
        self.lift_distances(dist).await
    }

    async fn is_match(&mut self, distance: &Self::DistanceRef) -> Result<bool> {
        Ok(lte_threshold_and_open(&mut self.session, &[distance.clone()]).await?[0])
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn less_than(
        &mut self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> Result<bool> {
        Ok(cross_compare(&mut self.session, &[(distance1.clone(), distance2.clone())]).await?[0])
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all, fields(batch_size = distances.len()))]
    async fn less_than_batch(
        &mut self,
        distances: &[(Self::DistanceRef, Self::DistanceRef)],
    ) -> Result<Vec<bool>> {
        if distances.is_empty() {
            return Ok(vec![]);
        }
        cross_compare(&mut self.session, distances).await
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all, fields(batch_size = distances.len()))]
    async fn is_match_batch(&mut self, distances: &[Self::DistanceRef]) -> Result<Vec<bool>> {
        if distances.is_empty() {
            return Ok(vec![]);
        }
        lte_threshold_and_open(&mut self.session, distances).await
    }
}

impl VectorStoreMut for Aby3Store {
    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        self.storage.append(&query.iris).await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{
        execution::hawk_main::scheduler::parallelize,
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
    async fn test_gr_hnsw() -> Result<()> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 10;
        let cleartext_database = IrisDB::new_random_rng(database_size, &mut rng).db;
        let shared_irises: Vec<_> = cleartext_database
            .iter()
            .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.clone()))
            .collect();

        let stores = setup_local_store_aby3_players(NetworkType::Local).await?;

        let mut jobs = JoinSet::new();
        for store in stores.iter() {
            let player_index = get_owner_index(store).await?;
            let queries = (0..database_size)
                .map(|id| Aby3Query::new_from_raw(shared_irises[id][player_index].clone()))
                .collect::<Vec<_>>();
            let mut rng = rng.clone();
            let store = store.clone();
            jobs.spawn(async move {
                let mut store = store.lock().await;
                let mut aby3_graph = GraphMem::new();
                let db = HnswSearcher::new_with_test_parameters();

                let mut inserted = vec![];
                // insert queries
                for query in queries.iter() {
                    let insertion_layer = db.select_layer_rng(&mut rng).unwrap();
                    let inserted_vector = db
                        .insert(&mut *store, &mut aby3_graph, query, insertion_layer)
                        .await
                        .unwrap();
                    inserted.push(inserted_vector)
                }
                tracing::debug!("FINISHED INSERTING");
                // Search for the same codes and find matches.
                let mut matching_results = vec![];
                for v in inserted.into_iter() {
                    let iris = store.storage.get_vector_or_empty(&v).await;
                    let query = Aby3Query::new(&iris);
                    let neighbors = db
                        .search(&mut *store, &aby3_graph, &query, 1)
                        .await
                        .unwrap();
                    tracing::debug!("Finished checking query");
                    matching_results.push(db.is_match(&mut *store, &[neighbors]).await.unwrap())
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
    async fn test_gr_premade_hnsw() -> Result<()> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 10;
        let network_t = NetworkType::Local;
        let (mut cleartext_data, secret_data) =
            lazy_random_setup(&mut rng, database_size, network_t.clone()).await?;

        let mut rng = AesRng::seed_from_u64(0_u64);
        let mut vector_graph_stores =
            shared_random_setup(&mut rng, database_size, network_t).await?;

        for ((v_from_scratch, _), (premade_v, _)) in
            vector_graph_stores.iter().zip(secret_data.iter())
        {
            let v_from_scratch = v_from_scratch.lock().await;
            let premade_v = premade_v.lock().await;
            assert_eq!(
                v_from_scratch.storage.data.read().await.points,
                premade_v.storage.data.read().await.points
            );
        }
        let hawk_searcher = HnswSearcher::new_with_test_parameters();

        for i in 0..database_size {
            let vector_id = VectorId::from_0_index(i as u32);
            let query = cleartext_data
                .0
                .storage
                .get_vector(&vector_id)
                .unwrap()
                .clone();
            let cleartext_neighbors = hawk_searcher
                .search(&mut cleartext_data.0, &cleartext_data.1, &query, 1)
                .await?;
            assert!(
                hawk_searcher
                    .is_match(&mut cleartext_data.0, &[cleartext_neighbors])
                    .await?,
            );

            let mut jobs = JoinSet::new();
            for (v, g) in vector_graph_stores.iter_mut() {
                let hawk_searcher = hawk_searcher.clone();
                let v_lock = v.lock().await;
                let g = g.clone();
                let q = v_lock.storage.get_vector_or_empty(&vector_id).await;
                let q = Aby3Query::new(&q);
                let v = v.clone();
                jobs.spawn(async move {
                    let mut v_lock = v.lock().await;
                    let secret_neighbors =
                        hawk_searcher.search(&mut *v_lock, &g, &q, 1).await.unwrap();

                    hawk_searcher
                        .is_match(&mut *v_lock, &[secret_neighbors])
                        .await
                });
            }
            let scratch_results = jobs.join_all().await;

            let mut jobs = JoinSet::new();
            for (v, g) in secret_data.iter() {
                let hawk_searcher = hawk_searcher.clone();
                let v = v.clone();
                let g = g.clone();
                jobs.spawn(async move {
                    let mut v_lock = v.lock().await;
                    let iris = v_lock.storage.get_vector_or_empty(&vector_id).await;
                    let query = Aby3Query::new(&iris);
                    let secret_neighbors = hawk_searcher
                        .search(&mut *v_lock, &g, &query, 1)
                        .await
                        .unwrap();

                    hawk_searcher
                        .is_match(&mut *v_lock, &[secret_neighbors])
                        .await
                });
            }
            let premade_results = jobs.join_all().await;

            for (premade_res, scratch_res) in izip!(scratch_results, premade_results) {
                assert!(premade_res? && scratch_res?);
            }
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_gr_aby3_store_plaintext() -> Result<()> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let db_dim = 4;
        let plaintext_database = IrisDB::new_random_rng(db_dim, &mut rng).db;
        let shared_irises: Vec<_> = plaintext_database
            .iter()
            .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.clone()))
            .collect();
        let mut local_stores = setup_local_store_aby3_players(NetworkType::Local).await?;
        // Now do the work for the plaintext store
        let mut plaintext_store = PlaintextStore::new();
        let plaintext_preps: Vec<_> = (0..db_dim)
            .map(|id| Arc::new(plaintext_database[id].clone()))
            .collect();
        let mut plaintext_inserts = Vec::new();
        for p in plaintext_preps.iter() {
            plaintext_inserts.push(plaintext_store.insert(p).await);
        }

        // pairs of indices to compare
        let it1 = (0..db_dim).combinations(2);
        let it2 = (0..db_dim).combinations(2);

        let plaintext_queries: Vec<_> = plaintext_database.into_iter().map(Arc::new).collect();

        let mut plain_results = HashMap::new();
        for comb1 in it1.clone() {
            for comb2 in it2.clone() {
                // compute distances in plaintext
                let dist1_plain = plaintext_store
                    .eval_distance(&plaintext_queries[comb1[0]], &plaintext_inserts[comb1[1]])
                    .await?;
                let dist2_plain = plaintext_store
                    .eval_distance(&plaintext_queries[comb2[0]], &plaintext_inserts[comb2[1]])
                    .await?;
                let bit = plaintext_store
                    .less_than(&dist1_plain, &dist2_plain)
                    .await?;
                plain_results.insert((comb1.clone(), comb2.clone()), bit);
            }
        }

        let mut aby3_inserts = vec![];
        for store in local_stores.iter_mut() {
            let player_index = get_owner_index(store).await?;
            let player_preps: Vec<_> = (0..db_dim)
                .map(|id| Aby3Query::new_from_raw(shared_irises[id][player_index].clone()))
                .collect();
            let mut player_inserts = vec![];
            let mut store_lock = store.lock().await;
            for p in player_preps.iter() {
                player_inserts.push(store_lock.storage.append(&p.iris).await);
            }
            aby3_inserts.push(player_inserts);
        }

        for comb1 in it1 {
            for comb2 in it2.clone() {
                let mut jobs = JoinSet::new();
                for store in local_stores.iter() {
                    let player_index = get_owner_index(store).await?;
                    let player_inserts = aby3_inserts[player_index].clone();
                    let store = store.clone();
                    let index10 = comb1[0];
                    let index11 = comb1[1];
                    let index20 = comb2[0];
                    let index21 = comb2[1];
                    jobs.spawn(async move {
                        let mut store = store.lock().await;
                        let dist1_aby3 = eval_vector_distance(
                            &mut store,
                            &player_inserts[index10],
                            &player_inserts[index11],
                        )
                        .await?;
                        let dist2_aby3 = eval_vector_distance(
                            &mut store,
                            &player_inserts[index20],
                            &player_inserts[index21],
                        )
                        .await?;
                        store.less_than(&dist1_aby3, &dist2_aby3).await
                    });
                }
                let res = jobs.join_all().await;
                for bit in res {
                    assert_eq!(
                        bit?,
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
    async fn test_gr_aby3_store_plaintext_batch() -> Result<()> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let db_size = 10;
        let plaintext_database = IrisDB::new_random_rng(db_size, &mut rng).db;
        let shared_irises: Vec<_> = plaintext_database
            .iter()
            .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.clone()))
            .collect();
        let mut local_stores = setup_local_store_aby3_players(NetworkType::Local).await?;
        // Now do the work for the plaintext store
        let mut plaintext_store = PlaintextStore::new();
        let plaintext_preps: Vec<_> = (0..db_size)
            .map(|id| Arc::new(plaintext_database[id].clone()))
            .collect();
        let mut plaintext_inserts = Vec::with_capacity(db_size);
        for p in plaintext_preps.iter() {
            plaintext_inserts.push(plaintext_store.insert(p).await);
        }

        // compute distances in plaintext
        let dist1_plain = plaintext_store
            .eval_distance_batch(
                &[Arc::new(plaintext_database[0].clone())],
                &plaintext_inserts,
            )
            .await?;
        let dist2_plain = plaintext_store
            .eval_distance_batch(
                &[Arc::new(plaintext_database[1].clone())],
                &plaintext_inserts,
            )
            .await?;
        let dist_plain = dist1_plain
            .into_iter()
            .zip(dist2_plain.into_iter())
            .collect::<Vec<_>>();
        let bits_plain = plaintext_store.less_than_batch(&dist_plain).await?;

        let mut aby3_inserts = vec![];
        let mut queries = vec![];
        for store in local_stores.iter_mut() {
            let player_index = get_owner_index(store).await?;
            let player_preps: Vec<_> = (0..db_size)
                .map(|id| Aby3Query::new_from_raw(shared_irises[id][player_index].clone()))
                .collect();
            queries.push(player_preps.clone());
            let mut player_inserts = vec![];
            let mut store_lock = store.lock().await;
            for p in player_preps.iter() {
                player_inserts.push(store_lock.insert(p).await);
            }
            aby3_inserts.push(player_inserts);
        }

        let mut jobs = JoinSet::new();
        for store in local_stores.iter() {
            let player_index = get_owner_index(store).await?;
            let player_inserts = aby3_inserts[player_index].clone();
            let player_preps = queries[player_index].clone();
            let store = store.clone();
            jobs.spawn(async move {
                let mut store_lock = store.lock().await;
                let dist1_aby3 = store_lock
                    .eval_distance_batch(&[player_preps[0].clone()], &player_inserts)
                    .await?;
                let dist2_aby3 = store_lock
                    .eval_distance_batch(&[player_preps[1].clone()], &player_inserts)
                    .await?;
                let dist_aby3 = dist1_aby3
                    .into_iter()
                    .zip(dist2_aby3.into_iter())
                    .collect::<Vec<_>>();
                store_lock.less_than_batch(&dist_aby3).await
            });
        }
        let bits_aby3 = jobs
            .join_all()
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()?;

        assert_eq!(bits_aby3[0], bits_aby3[1]);
        assert_eq!(bits_aby3[0], bits_aby3[2]);
        assert_eq!(bits_aby3[0], bits_plain);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_gr_scratch_hnsw() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 2;
        let searcher = HnswSearcher::new_with_test_parameters();
        let mut vectors_and_graphs =
            shared_random_setup(&mut rng, database_size, NetworkType::Local)
                .await
                .unwrap();

        for i in 0..database_size {
            let vector_id = VectorId::from_0_index(i as u32);
            let mut jobs = JoinSet::new();
            for (store, graph) in vectors_and_graphs.iter_mut() {
                let graph = graph.clone();
                let searcher = searcher.clone();
                let q = store
                    .lock()
                    .await
                    .storage
                    .get_vector_or_empty(&vector_id)
                    .await;
                let q = Aby3Query::new(&q);
                let store = store.clone();
                jobs.spawn(async move {
                    let mut store = store.lock().await;
                    let secret_neighbors =
                        searcher.search(&mut *store, &graph, &q, 1).await.unwrap();
                    searcher
                        .is_match(&mut *store, &[secret_neighbors])
                        .await
                        .unwrap()
                });
            }
            let res = jobs.join_all().await;
            for (party_index, r) in res.into_iter().enumerate() {
                assert!(r, "Failed at index {:?} by party {:?}", i, party_index);
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_gr_non_existent_vectors() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 2;
        let vectors_and_graphs = shared_random_setup(&mut rng, database_size, NetworkType::Local)
            .await
            .unwrap();

        let mut tasks = vec![];
        for (store, _graph) in vectors_and_graphs {
            let mut store = store.lock_owned().await;
            tasks.push(async move {
                let none = VectorId::from_0_index(999);
                let a = VectorId::from_0_index(0);
                let b = VectorId::from_0_index(1);

                let queries = store.vectors_as_queries(vec![a, b]).await;
                let vectors = vec![a, b, none];
                let n_vecs = vectors.len();

                let distances = store.eval_distance_batch(&queries, &vectors).await.unwrap();

                let is_match = store.is_match_batch(&distances).await.unwrap();
                assert_eq!(
                    is_match,
                    [vec![true, false, false], vec![false, true, false]].concat(),
                    "Vectors should match with themselves and not with the others"
                );

                let distances_to_none = vec![distances[2].clone(), distances[2 + n_vecs].clone()];
                let pairs = distances_to_none
                    .into_iter()
                    .cartesian_product(distances)
                    .collect_vec();
                let less_than = store.less_than_batch(&pairs).await.unwrap();

                assert_eq!(
                    less_than,
                    vec![false; pairs.len()],
                    "Nothing is less than a distance to a non-existent vector"
                );

                Ok(())
            });
        }
        parallelize(tasks.into_iter()).await.unwrap();
    }
}
