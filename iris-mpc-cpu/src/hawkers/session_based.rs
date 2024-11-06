use std::collections::HashMap;

use super::{galois_store::{eval_pairwise_distances, DistanceShare, GaloisRingPoint}, plaintext_store::{self, PlaintextStore}};
use crate::{
    database_generators::{generate_galois_iris_shares, GaloisRingSharedIris},
    execution::session::{Session, SessionHandles},
    hawkers::plaintext_store::PointId,
    protocol::ops::{
        cross_compare, galois_ring_is_match, galois_ring_pairwise_distance, galois_ring_to_rep3, is_dot_zero,
    },
};
use aes_prng::AesRng;
use hawk_pack::{graph_store::{graph_mem::Layer, GraphMem}, hnsw_db::{FurthestQueue, HawkSearcher}, GraphStore, VectorStore};
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

pub struct PlayerStorageEnsemble {
    storages: Vec<PlayerStorage>,
}

impl PlayerStorageEnsemble {
    pub fn new(p0: PlayerStorage, p1: PlayerStorage, p2: PlayerStorage) -> Self {
        PlayerStorageEnsemble {
            storages: vec![p0, p1, p2],
        }
    }
    pub fn get(&self, player_id: usize) -> eyre::Result<&PlayerStorage> {
        if player_id > 2 {
            return Err(eyre::eyre!("Invalid player number"));
        }
        Ok(&self.storages[player_id])
    }
}

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

impl SessionBasedStorage {
    pub fn new(session: Session, storage: PlayerStorage) -> Self {
        SessionBasedStorage {
            player_storage: storage,
            session,
        }
    }
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
    Ok(PlayerStorageEnsemble::new(player_0, player_1, player_2))
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
    type DistanceRef = DistanceShare<u16>; // Distance represented as shares.

    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        self.player_storage.insert(query);
        *query
    }

    // TODO: Here we need to adapt to the `eval_distance` implemented in the
    // galois_store. Once that is done, we can implement the
    // "eval_distance_batch" in here as well.
    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Self::DistanceRef {
        let storage = self.get_storage();
        let query_point = storage.points[query.val()].clone();
        let vector_point = storage.points[vector.val()].clone();
        let pairs = vec![(query_point.data, vector_point.data)];
        let mut session = self.session_as_mut();
        let ds_and_ts = eval_pairwise_distances(pairs, &mut session).await;
        DistanceShare::new (
            ds_and_ts[0].clone(),
            ds_and_ts[1].clone(),
            //TODO: this might be unnecessary
            self.session.own_identity(),
        )
    }

    async fn is_match(&mut self, distance: &Self::DistanceRef) -> bool {
        let mut session = self.session_as_mut();

        is_dot_zero(&mut session, distance.code_dot(), distance.mask_dot())
                    .await
                    .unwrap()
    }

    async fn less_than(
        &mut self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> bool {
        let session = self.session_as_mut();
        cross_compare(
            session,
            distance1.code_dot(),
            distance1.mask_dot(),
            distance2.code_dot(),
            distance2.mask_dot(),
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

pub struct SessionBasedSetup {
    pub plaintext_store: PlaintextStore,
    pub plaintext_graph: GraphMem<PlaintextStore>,
    pub session_graphs:   Vec<GraphMem<SessionBasedStorage>>, //each player has its own graph
    pub player_stores:   PlayerStorageEnsemble,
}

async fn session_graph_from_plain(
    plaintext_store: PlaintextStore,
    graph_store: GraphMem<PlaintextStore>,
    player_id: usize
) -> GraphMem<SessionBasedStorage> {
    let ep = graph_store.get_entry_point().await;

    let layers = graph_store.get_layers();

    let mut shared_layers = vec![];
    for layer in layers {
        let links = layer.get_links_map();
        let mut shared_links = HashMap::new();
        for (source_id, queue) in links {
            let mut shared_queue = vec![];
            let source_v = plaintext_store.points[source_id.val()].clone();
            for (target_id, _) in queue.as_vec_ref() {
                let target_v = plaintext_store.points[target_id.val()].clone();
                let shared_distance = source_v.compute_distance(&target_v);
                shared_queue.push((*target_v, shared_distance));
            }
            shared_links.insert(*source_v, FurthestQueue::from_ascending_vec(shared_queue));
        }
        shared_layers.push(Layer::from_links(shared_links));
    }

    GraphMem::from_precomputed(ep, shared_layers)
}

pub async fn session_based_ready_made_hawk_searcher<R: RngCore + Clone + CryptoRng>(
    rng: &mut R,
    database_size: usize,
) -> eyre::Result<SessionBasedSetup> {
    // makes sure the searcher produces same graph structure by having the same rng
    let mut rng_searcher1 = AesRng::from_rng(rng.clone())?;
    let cleartext_database = IrisDB::new_random_par(database_size, rng).db;

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
    let player_storage_ensemble = storage_setup_preloaded_db(rng, cleartext_database)?;

    let session_graphs = {
        for player_id in 0..3 {
            let player_storage = player_storage_ensemble.get(player_id).unwrap();
            let distance_map = |lazy_distance: (PointId, PointId)| -> DistanceShare<u16> {
                let (query, vector) = lazy_distance;
                let distance = player_storage.eval_distance(&query, &vector).await;
                DistanceShare::new(distance, distance, player_storage_ensemble.storages[0].points[0].data.0)
            };
        }
        
    };

    
    let session_graph = GraphMem::<SessionBasedStorage>::from_another(
        plaintext_graph_store.clone(),
        |vx| vx,
        |dx| dx,
    );
    Ok(SessionBasedSetup {
        plaintext_store: plaintext_vector_store,
        plaintext_graph: plaintext_graph_store,
        player_stores: player_storage_ensemble,
        session_graphs,
    })
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

    #[test]
    fn test_session_based_hnsw() {
        // TODO: Increasing the stack thread is required only when running
        // cargo test This works without any limits when running in
        // benchmark mode but there are some slight issues with the thread
        // stack size in cargo test
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_stack_size(10000000)
            .build()
            .unwrap()
            .block_on(async {
                let mut rng = AesRng::seed_from_u64(0_u64);
                let database_size = 10;
                let cleartext_database = IrisDB::new_random_rng(database_size, &mut rng).db;

                let players_storage =
                    storage_setup_preloaded_db(&mut rng, cleartext_database).unwrap();

                let runtime = LocalRuntime::replicated_test_config();
                let ready_sessions = runtime.create_player_sessions().await.unwrap();

                let mut set = JoinSet::new();

                for (player_no, player_identity) in runtime.identities.iter().enumerate() {
                    let session = ready_sessions.get(player_identity).unwrap().clone();
                    let store = players_storage.get(player_no).unwrap();
                    let mut session_store = SessionBasedStorage {
                        player_storage: store.clone(),
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
            })
    }

    #[test]
    #[traced_test]
    fn test_session_based_premade_hnsw() {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_stack_size(10000000)
            .build()
            .unwrap()
            .block_on(async {
                let mut rng = AesRng::seed_from_u64(0_u64);
                let database_size = 10;
                let setup = session_based_ready_made_hawk_searcher(&mut rng, database_size)
                    .await
                    .unwrap();
                let players_storage = setup.player_stores;
                let graph = setup.session_graph;

                let runtime = LocalRuntime::replicated_test_config();
                let ready_sessions = runtime.create_player_sessions().await.unwrap();
                let mut set = JoinSet::new();
                for (player_no, player_identity) in runtime.identities.iter().enumerate() {
                    let session = ready_sessions.get(player_identity).unwrap().clone();
                    let store = players_storage.get(player_no).unwrap();
                    let mut session_store = SessionBasedStorage {
                        player_storage: store.clone(),
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
            });
    }

    #[test]
    #[traced_test]
    fn test_session_based_less_than() {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_stack_size(10000000)
            .build()
            .unwrap()
            .block_on(async {
                let mut rng = AesRng::seed_from_u64(0_u64);
                let database_size = 4;
                let cleartext_database = IrisDB::new_random_rng(database_size, &mut rng).db; // Now do the work for the plaintext store
                let mut plaintext_store = PlaintextStore::default();
                let plaintext_preps: Vec<_> = (0..database_size)
                    .map(|id| plaintext_store.prepare_query(cleartext_database[id].clone()))
                    .collect();
                let mut plaintext_inserts = Vec::new();
                for p in plaintext_preps.iter() {
                    plaintext_inserts.push(plaintext_store.insert(p).await);
                }

                let players_storage =
                    storage_setup_preloaded_db(&mut rng, cleartext_database).unwrap();

                let runtime = LocalRuntime::replicated_test_config();
                let ready_sessions = runtime.create_player_sessions().await.unwrap();
                let mut set = JoinSet::new();

                for (player_no, player_identity) in runtime.identities.iter().enumerate() {
                    let session = ready_sessions.get(player_identity).unwrap().clone();
                    let store = players_storage.get(player_no).unwrap();
                    let mut session_store = SessionBasedStorage {
                        player_storage: store.clone(),
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
            });
    }
}
