use super::{galois_store::GaloisRingPoint, plaintext_store::PlaintextStore};
use crate::{
    database_generators::{generate_galois_iris_shares, GaloisRingSharedIris},
    execution::session::Session,
    hawkers::plaintext_store::PointId,
    protocol::ops::{
        cross_compare, galois_ring_is_match, galois_ring_pairwise_distance, galois_ring_to_rep3,
    },
};
use aes_prng::AesRng;
use hawk_pack::{graph_store::GraphMem, hnsw_db::HawkSearcher, VectorStore};
use iris_mpc_common::iris_db::{db::IrisDB, iris::IrisCode};
use rand::{CryptoRng, RngCore, SeedableRng};

#[derive(Debug, Default, Clone)]
pub struct PlayerStorage {
    points: Vec<GaloisRingPoint>,
}

impl PlayerStorage {
    fn get_point(&self, index: usize) -> GaloisRingPoint {
        self.points[index].clone()
    }
}

type PlayerStorageEnsemble = (PlayerStorage, PlayerStorage, PlayerStorage);

impl PlayerStorage {
    fn insert(&mut self, query: &PointId) -> PointId {
        // The query is now accepted in the store. It keeps the same ID.
        *query
    }
}

impl PlayerStorage {
    pub fn new_with_shared_db(data: Vec<GaloisRingSharedIris>) -> Self {
        let points: Vec<GaloisRingPoint> = data
            .into_iter()
            .map(|d| GaloisRingPoint { data: d })
            .collect();
        PlayerStorage { points }
    }

    pub fn prepare_query(&mut self, raw_query: GaloisRingSharedIris) -> PointId {
        self.points.push(GaloisRingPoint { data: raw_query });

        let point_id = self.points.len() - 1;
        PointId(point_id)
    }
}

#[derive(Clone)]
pub struct SessionBasedStorage {
    pub player_storage: PlayerStorage,
    session:            Session,
}

impl std::fmt::Debug for SessionBasedStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.player_storage.fmt(f)
    }
}

pub fn storage_setup_preloaded_db<R: RngCore + CryptoRng>(
    rng: &mut R,
    database: Vec<IrisCode>,
) -> eyre::Result<PlayerStorageEnsemble> {
    let mut p0 = Vec::new();
    let mut p1 = Vec::new();
    let mut p2 = Vec::new();

    for iris in database {
        let all_shares = generate_galois_iris_shares(rng, iris);
        p0.push(all_shares[0].clone());
        p1.push(all_shares[1].clone());
        p2.push(all_shares[2].clone());
    }

    let player_0 = PlayerStorage::new_with_shared_db(p0);
    let player_1 = PlayerStorage::new_with_shared_db(p1);
    let player_2 = PlayerStorage::new_with_shared_db(p2);
    Ok((player_0, player_1, player_2))
}

impl SessionBasedStorage {
    pub fn prepare_query(&mut self, code: GaloisRingSharedIris) -> PointId {
        self.player_storage.prepare_query(code)
    }
    pub fn session_as_mut(&mut self) -> &mut Session {
        &mut self.session
    }

    pub fn get_storage(&self) -> &PlayerStorage {
        &self.player_storage
    }
}

impl VectorStore for SessionBasedStorage {
    type QueryRef = PointId; // Vector ID, pending insertion.
    type VectorRef = PointId; // Vector ID, inserted.
    type DistanceRef = (PointId, PointId); // Lazy distance representation.

    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        self.player_storage.insert(query);
        *query
    }

    async fn eval_distance(
        &self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Self::DistanceRef {
        // Do not compute the distance yet, just forward the IDs.
        (*query, *vector)
    }

    async fn is_match(&mut self, distance: &Self::DistanceRef) -> bool {
        let storage = self.get_storage();
        let x = storage.get_point(distance.0.val());
        let mut y = storage.get_point(distance.1.val());
        let session = self.session_as_mut();
        y.data.code.preprocess_iris_code_query_share();
        y.data.mask.preprocess_mask_code_query_share();
        galois_ring_is_match(session, &[(x.data, y.data)])
            .await
            .unwrap()
    }

    async fn less_than(
        &mut self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> bool {
        let d1 = *distance1;
        let d2 = *distance2;

        let player_storage = self.get_storage();
        let (x1, y1) = (
            player_storage.get_point(d1.0.val()),
            player_storage.get_point(d1.1.val()),
        );
        let (x2, y2) = (
            player_storage.get_point(d2.0.val()),
            player_storage.get_point(d2.1.val()),
        );

        let session = self.session_as_mut();
        let mut pairs = [(x1.data, y1.data), (x2.data, y2.data)];
        pairs.iter_mut().for_each(|(_x, y)| {
            y.code.preprocess_iris_code_query_share();
            y.mask.preprocess_mask_code_query_share();
        });
        let ds_and_ts = galois_ring_pairwise_distance(session, &pairs)
            .await
            .unwrap();
        let ds_and_ts = galois_ring_to_rep3(session, ds_and_ts).await.unwrap();
        cross_compare(
            session,
            ds_and_ts[0].clone(),
            ds_and_ts[1].clone(),
            ds_and_ts[2].clone(),
            ds_and_ts[3].clone(),
        )
        .await
        .unwrap()
    }
}

pub async fn session_based_insert(
    searcher: &HawkSearcher,
    vector_store: &mut SessionBasedStorage,
    graph_store: &mut GraphMem<SessionBasedStorage>,
    query: &PointId,
    rng: &mut impl RngCore,
) -> eyre::Result<()> {
    let neighbors = searcher
        .search_to_insert(vector_store, graph_store, query)
        .await;
    // Insert the new vector into the store.
    let inserted = vector_store.insert(query).await;
    searcher
        .insert_from_search_results(vector_store, graph_store, rng, inserted, neighbors)
        .await;
    Ok(())
}
pub async fn session_based_match(
    searcher: &HawkSearcher,
    vector_store: &mut SessionBasedStorage,
    graph_store: &mut GraphMem<SessionBasedStorage>,
    query: &PointId,
    _rng: &mut impl RngCore,
) -> eyre::Result<bool> {
    let neighbors = searcher
        .search_to_insert(vector_store, graph_store, query)
        .await;
    Ok(searcher.is_match(vector_store, &neighbors).await)
}

pub async fn session_based_ready_made_hawk_searcher<R: RngCore + Clone + CryptoRng>(
    rng: &mut R,
    database_size: usize,
) -> eyre::Result<(
    (PlaintextStore, GraphMem<PlaintextStore>),
    (
        (PlayerStorage, PlayerStorage, PlayerStorage),
        GraphMem<SessionBasedStorage>,
    ),
)> {
    // makes sure the searcher produces same graph structure by having the same rng
    let mut rng_searcher1 = AesRng::from_rng(rng.clone())?;
    let cleartext_database = IrisDB::new_random_rng(database_size, rng).db;

    let mut plaintext_vector_store = PlaintextStore::default();
    let mut plaintext_graph_store = GraphMem::new();
    let searcher = HawkSearcher::default();

    for raw_query in cleartext_database.iter() {
        let query = plaintext_vector_store.prepare_query(raw_query.clone());
        let neighbors = searcher
            .search_to_insert(
                &mut plaintext_vector_store,
                &mut plaintext_graph_store,
                &query,
            )
            .await;
        let inserted = plaintext_vector_store.insert(&query).await;
        searcher
            .insert_from_search_results(
                &mut plaintext_vector_store,
                &mut plaintext_graph_store,
                &mut rng_searcher1,
                inserted,
                neighbors,
            )
            .await;
    }
    let (p0_store, p1_store, p2_store) = storage_setup_preloaded_db(rng, cleartext_database)?;
    let session_graph = GraphMem::<SessionBasedStorage>::from_another(
        plaintext_graph_store.clone(),
        |vx| vx,
        |dx| dx,
    );
    let plaintext = (plaintext_vector_store, plaintext_graph_store);
    Ok((plaintext, ((p0_store, p1_store, p2_store), session_graph)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::local::LocalRuntime;
    use aes_prng::AesRng;
    use hawk_pack::{graph_store::GraphMem, hnsw_db::HawkSearcher};
    use iris_mpc_common::iris_db::db::IrisDB;
    use itertools::Itertools;
    use rand::SeedableRng;
    use tokio::task::JoinSet;
    use tracing_test::traced_test;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_based_hnsw() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 10;
        let cleartext_database = IrisDB::new_random_rng(database_size, &mut rng).db;

        let (p0_store, p1_store, p2_store) =
            storage_setup_preloaded_db(&mut rng, cleartext_database).unwrap();

        let runtime = LocalRuntime::replicated_test_config();
        let ready_sessions = runtime.create_player_sessions().await.unwrap();

        let mut set = JoinSet::new();

        for (player_no, player_identity) in runtime.identities.iter().enumerate() {
            let session = ready_sessions.get(player_identity).unwrap().clone();
            let store = match player_no {
                0 => p0_store.clone(),
                1 => p1_store.clone(),
                2 => p2_store.clone(),
                _ => unimplemented!(),
            };
            let mut session_store = SessionBasedStorage {
                player_storage: store,
                session,
            };
            let mut session_graph = GraphMem::new();
            let searcher = HawkSearcher::default();
            let queries: Vec<_> = (0..database_size).map(PointId).collect();

            set.spawn(async move {
                // insert queries
                let mut rng = AesRng::seed_from_u64(0_u64);
                for query in queries.iter() {
                    let _ = session_based_insert(
                        &searcher,
                        &mut session_store,
                        &mut session_graph,
                        query,
                        &mut rng,
                    )
                    .await;
                }
                for query in queries.iter() {
                    let match_result = session_based_match(
                        &searcher,
                        &mut session_store,
                        &mut session_graph,
                        query,
                        &mut rng,
                    )
                    .await;
                    assert!(match_result.unwrap())
                }
            });
        }
        let _ = set.join_all().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_session_based_premade_hnsw() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 10;
        let (_, secret_data) = session_based_ready_made_hawk_searcher(&mut rng, database_size)
            .await
            .unwrap();
        let ((p0_store, p1_store, p2_store), graph) = secret_data;

        let runtime = LocalRuntime::replicated_test_config();
        let ready_sessions = runtime.create_player_sessions().await.unwrap();
        let mut set = JoinSet::new();
        for (player_no, player_identity) in runtime.identities.iter().enumerate() {
            let session = ready_sessions.get(player_identity).unwrap().clone();
            let store = match player_no {
                0 => p0_store.clone(),
                1 => p1_store.clone(),
                2 => p2_store.clone(),
                _ => unimplemented!(),
            };
            let mut session_store = SessionBasedStorage {
                player_storage: store,
                session,
            };
            let mut session_graph = graph.clone();
            let searcher = HawkSearcher::default();
            let queries: Vec<_> = (0..database_size).map(PointId).collect();

            set.spawn(async move {
                // insert queries
                let mut rng = AesRng::seed_from_u64(0_u64);
                for query in queries.iter() {
                    let match_result = session_based_match(
                        &searcher,
                        &mut session_store,
                        &mut session_graph,
                        query,
                        &mut rng,
                    )
                    .await;
                    assert!(match_result.unwrap())
                }
            });
        }
        let _ = set.join_all().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_session_based_less_than() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 4;
        let cleartext_database = IrisDB::new_random_rng(database_size, &mut rng).db;
        // Now do the work for the plaintext store
        let mut plaintext_store = PlaintextStore::default();
        let plaintext_preps: Vec<_> = (0..database_size)
            .map(|id| plaintext_store.prepare_query(cleartext_database[id].clone()))
            .collect();
        let mut plaintext_inserts = Vec::new();
        for p in plaintext_preps.iter() {
            plaintext_inserts.push(plaintext_store.insert(p).await);
        }

        let (p0_store, p1_store, p2_store) =
            storage_setup_preloaded_db(&mut rng, cleartext_database).unwrap();

        let runtime = LocalRuntime::replicated_test_config();
        let ready_sessions = runtime.create_player_sessions().await.unwrap();
        let mut set = JoinSet::new();

        for (player_no, player_identity) in runtime.identities.iter().enumerate() {
            let session = ready_sessions.get(player_identity).unwrap().clone();
            let store = match player_no {
                0 => p0_store.clone(),
                1 => p1_store.clone(),
                2 => p2_store.clone(),
                _ => unimplemented!(),
            };
            let mut session_store = SessionBasedStorage {
                player_storage: store,
                session,
            };
            let queries: Vec<_> = (0..database_size).map(PointId).collect();

            set.spawn(async move {
                let it1 = (0..database_size).combinations(2);
                let it2 = (0..database_size).combinations(2);
                let mut results = Vec::new();
                for comb1 in it1 {
                    for comb2 in it2.clone() {
                        results.push(
                            session_store
                                .less_than(
                                    &(queries[comb1[0]], queries[comb1[1]]),
                                    &(queries[comb2[0]], queries[comb2[1]]),
                                )
                                .await,
                        );
                    }
                }
                results
            });
        }
        let it1 = (0..database_size).combinations(2);
        let it2 = (0..database_size).combinations(2);
        let mut plain_results = Vec::new();
        for comb1 in it1 {
            for comb2 in it2.clone() {
                plain_results.push(
                    plaintext_store
                        .less_than(
                            &(plaintext_inserts[comb1[0]], plaintext_inserts[comb1[1]]),
                            &(plaintext_inserts[comb2[0]], plaintext_inserts[comb2[1]]),
                        )
                        .await,
                );
            }
        }
        let parties_outputs = set.join_all().await;
        assert_eq!(parties_outputs[0], plain_results);
        assert_eq!(parties_outputs[1], plain_results);
        assert_eq!(parties_outputs[2], plain_results);
    }
}
