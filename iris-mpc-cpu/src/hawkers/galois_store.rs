use super::plaintext_store::PlaintextStore;
use crate::{
    database_generators::{generate_galois_iris_shares, GaloisRingSharedIris},
    execution::{local::LocalRuntime, player::Identity, session::Session},
    hawkers::plaintext_store::PointId,
    protocol::ops::{
        cross_compare, galois_ring_pairwise_distance, galois_ring_to_rep3, is_dot_zero,
    },
    shares::{int_ring::IntRing2k, share::Share},
};
use aes_prng::AesRng;
use hawk_pack::{
    graph_store::{graph_mem::Layer, GraphMem},
    hnsw_db::{FurthestQueue, HawkSearcher},
    GraphStore, VectorStore,
};
use iris_mpc_common::iris_db::{db::IrisDB, iris::IrisCode};
use rand::{CryptoRng, RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::task::JoinSet;

#[derive(Default, Clone)]
pub struct Aby3NgStorePlayer {
    points: Vec<GaloisRingPoint>,
}

impl std::fmt::Debug for Aby3NgStorePlayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.points.fmt(f)
    }
}

#[derive(Eq, PartialEq, Clone, Debug)]
struct GaloisRingPoint {
    /// Whatever encoding of a vector.
    data: GaloisRingSharedIris,
}

impl Aby3NgStorePlayer {
    pub fn new_with_shared_db(data: Vec<GaloisRingSharedIris>) -> Self {
        let points: Vec<GaloisRingPoint> = data
            .into_iter()
            .map(|d| GaloisRingPoint { data: d })
            .collect();
        Aby3NgStorePlayer { points }
    }

    pub fn prepare_query(&mut self, raw_query: GaloisRingSharedIris) -> PointId {
        self.points.push(GaloisRingPoint { data: raw_query });

        let point_id = self.points.len() - 1;
        PointId(point_id)
    }
}

impl Aby3NgStorePlayer {
    fn insert(&mut self, query: &PointId) -> PointId {
        // The query is now accepted in the store. It keeps the same ID.
        *query
    }
}

pub fn setup_local_player_preloaded_db(
    database: Vec<GaloisRingSharedIris>,
) -> eyre::Result<Aby3NgStorePlayer> {
    let aby3_store = Aby3NgStorePlayer::new_with_shared_db(database);
    Ok(aby3_store)
}

pub fn setup_local_aby3_players_with_preloaded_db<R: RngCore + CryptoRng>(
    rng: &mut R,
    database: Vec<IrisCode>,
) -> eyre::Result<LocalNetAby3NgStoreProtocol> {
    let mut p0 = Vec::new();
    let mut p1 = Vec::new();
    let mut p2 = Vec::new();

    for iris in database {
        let all_shares = generate_galois_iris_shares(rng, iris);
        p0.push(all_shares[0].clone());
        p1.push(all_shares[1].clone());
        p2.push(all_shares[2].clone());
    }

    let player_0 = setup_local_player_preloaded_db(p0)?;
    let player_1 = setup_local_player_preloaded_db(p1)?;
    let player_2 = setup_local_player_preloaded_db(p2)?;
    let players = HashMap::from([
        (Identity::from("alice"), player_0),
        (Identity::from("bob"), player_1),
        (Identity::from("charlie"), player_2),
    ]);
    let runtime = LocalRuntime::replicated_test_config();
    Ok(LocalNetAby3NgStoreProtocol { runtime, players })
}

#[derive(Debug, Clone)]
pub struct LocalNetAby3NgStoreProtocol {
    pub players: HashMap<Identity, Aby3NgStorePlayer>,
    pub runtime: LocalRuntime,
}

pub fn setup_local_store_aby3_players() -> eyre::Result<LocalNetAby3NgStoreProtocol> {
    let player_0 = Aby3NgStorePlayer::default();
    let player_1 = Aby3NgStorePlayer::default();
    let player_2 = Aby3NgStorePlayer::default();
    let runtime = LocalRuntime::replicated_test_config();
    let players = HashMap::from([
        (Identity::from("alice"), player_0),
        (Identity::from("bob"), player_1),
        (Identity::from("charlie"), player_2),
    ]);
    Ok(LocalNetAby3NgStoreProtocol { runtime, players })
}

impl LocalNetAby3NgStoreProtocol {
    pub fn prepare_query(&mut self, code: Vec<GaloisRingSharedIris>) -> PointId {
        assert_eq!(code.len(), 3);
        assert_eq!(self.players.len(), 3);
        let pid0 = self
            .players
            .get_mut(&Identity::from("alice"))
            .unwrap()
            .prepare_query(code[0].clone());
        let pid1 = self
            .players
            .get_mut(&Identity::from("bob"))
            .unwrap()
            .prepare_query(code[1].clone());
        let pid2 = self
            .players
            .get_mut(&Identity::from("charlie"))
            .unwrap()
            .prepare_query(code[2].clone());
        assert_eq!(pid0, pid1);
        assert_eq!(pid1, pid2);
        pid0
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct DistanceShare<T: IntRing2k> {
    code_dot: Share<T>,
    mask_dot: Share<T>,
    player:   Identity,
}

async fn eval_pairwise_distances(
    mut pairs: Vec<(GaloisRingSharedIris, GaloisRingSharedIris)>,
    player_session: &mut Session,
) -> Vec<Share<u16>> {
    pairs.iter_mut().for_each(|(_x, y)| {
        y.code.preprocess_iris_code_query_share();
        y.mask.preprocess_mask_code_query_share();
    });
    let ds_and_ts = galois_ring_pairwise_distance(player_session, &pairs)
        .await
        .unwrap();
    galois_ring_to_rep3(player_session, ds_and_ts)
        .await
        .unwrap()
}

impl VectorStore for LocalNetAby3NgStoreProtocol {
    type QueryRef = PointId; // Vector ID, pending insertion.
    type VectorRef = PointId; // Vector ID, inserted.
    type DistanceRef = Vec<DistanceShare<u16>>; // Distance represented as shares.

    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        // The query is now accepted in the store. It keeps the same ID.
        for (_id, storage) in self.players.iter_mut() {
            storage.insert(query);
        }
        *query
    }

    async fn eval_distance(
        &self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Self::DistanceRef {
        let ready_sessions = self.runtime.create_player_sessions().await.unwrap();
        let mut jobs = JoinSet::new();
        for player in self.runtime.identities.clone() {
            let mut player_session = ready_sessions.get(&player).unwrap().clone();
            let storage = self.players.get(&player).unwrap();
            let query_point = storage.points[query.val()].clone();
            let vector_point = storage.points[vector.val()].clone();
            let pairs = vec![(query_point.data, vector_point.data)];
            jobs.spawn(async move {
                let ds_and_ts = eval_pairwise_distances(pairs, &mut player_session).await;
                DistanceShare {
                    code_dot: ds_and_ts[0].clone(),
                    mask_dot: ds_and_ts[1].clone(),
                    player:   player.clone(),
                }
            });
        }
        jobs.join_all().await
    }

    async fn eval_distance_batch(
        &self,
        query: &Self::QueryRef,
        vectors: &[Self::VectorRef],
    ) -> Vec<Self::DistanceRef> {
        let ready_sessions = self.runtime.create_player_sessions().await.unwrap();
        let mut jobs = JoinSet::new();
        for player in self.runtime.identities.clone() {
            let mut player_session = ready_sessions.get(&player).unwrap().clone();
            let storage = self.players.get(&player).unwrap();
            let query_point = storage.points[query.val()].clone();
            let pairs = vectors
                .iter()
                .map(|vector_id| {
                    let vector_point = storage.points[vector_id.val()].clone();
                    (query_point.data.clone(), vector_point.data)
                })
                .collect::<Vec<_>>();
            jobs.spawn(async move {
                let ds_and_ts = eval_pairwise_distances(pairs, &mut player_session).await;
                ds_and_ts
                    .chunks(2)
                    .map(|dot_products| DistanceShare {
                        code_dot: dot_products[0].clone(),
                        mask_dot: dot_products[1].clone(),
                        player:   player.clone(),
                    })
                    .collect::<Vec<_>>()
            });
        }
        // Now we have a vector of 3 vectors of DistanceShares, we need to transpose it
        // to a vector of DistanceRef
        let mut all_shares = jobs
            .join_all()
            .await
            .into_iter()
            .map(|player_shares| player_shares.into_iter())
            .collect::<Vec<_>>();
        (0..vectors.len())
            .map(|_| {
                all_shares
                    .iter_mut()
                    .map(|player_shares| player_shares.next().unwrap())
                    .collect::<Self::DistanceRef>()
            })
            .collect::<Vec<Self::DistanceRef>>()
    }

    async fn is_match(&self, distance: &Self::DistanceRef) -> bool {
        let ready_sessions = self.runtime.create_player_sessions().await.unwrap();
        let mut jobs = JoinSet::new();
        for distance_share in distance.iter() {
            let mut player_session = ready_sessions.get(&distance_share.player).unwrap().clone();
            let code_dot = distance_share.code_dot.clone();
            let mask_dot = distance_share.mask_dot.clone();
            jobs.spawn(async move {
                is_dot_zero(&mut player_session, code_dot, mask_dot)
                    .await
                    .unwrap()
            });
        }
        let r0 = jobs.join_next().await.unwrap().unwrap();
        let r1 = jobs.join_next().await.unwrap().unwrap();
        let r2 = jobs.join_next().await.unwrap().unwrap();
        assert_eq!(r0, r1);
        assert_eq!(r1, r2);
        r0
    }

    async fn less_than(
        &self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> bool {
        let ready_sessions = self.runtime.create_player_sessions().await.unwrap();
        let mut jobs = JoinSet::new();
        for share1 in distance1.iter() {
            for share2 in distance2.iter() {
                if share1.player == share2.player {
                    let mut player_session = ready_sessions.get(&share1.player).unwrap().clone();
                    let code_dot1 = share1.code_dot.clone();
                    let mask_dot1 = share1.mask_dot.clone();
                    let code_dot2 = share2.code_dot.clone();
                    let mask_dot2 = share2.mask_dot.clone();
                    jobs.spawn(async move {
                        cross_compare(
                            &mut player_session,
                            code_dot1,
                            mask_dot1,
                            code_dot2,
                            mask_dot2,
                        )
                        .await
                        .unwrap()
                    });
                }
            }
        }
        let res = jobs.join_all().await;
        assert_eq!(res[0], res[1]);
        assert_eq!(res[0], res[2]);
        res[0]
    }
}

impl LocalNetAby3NgStoreProtocol {
    async fn graph_from_plain(
        &self,
        graph_store: GraphMem<PlaintextStore>,
    ) -> GraphMem<LocalNetAby3NgStoreProtocol> {
        let ep = graph_store.get_entry_point().await;

        let layers = graph_store.get_layers();

        let mut shared_layers = vec![];
        for layer in layers {
            let links = layer.get_links_map();
            let mut shared_links = HashMap::new();
            for (source_v, queue) in links {
                let mut shared_queue = vec![];
                for (target_v, _) in queue.as_vec_ref() {
                    let shared_distance = self.eval_distance(source_v, target_v).await;
                    shared_queue.push((*target_v, shared_distance));
                }
                shared_links.insert(*source_v, FurthestQueue::from_ascending_vec(shared_queue));
            }
            shared_layers.push(Layer::from_links(shared_links));
        }

        GraphMem::from_precomputed(ep, shared_layers)
    }
}

pub async fn gr_create_ready_made_hawk_searcher<R: RngCore + Clone + CryptoRng>(
    rng: &mut R,
    database_size: usize,
) -> eyre::Result<(
    (PlaintextStore, GraphMem<PlaintextStore>),
    (
        LocalNetAby3NgStoreProtocol,
        GraphMem<LocalNetAby3NgStoreProtocol>,
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

    let protocol_store = setup_local_aby3_players_with_preloaded_db(rng, cleartext_database)?;
    let protocol_graph = protocol_store
        .graph_from_plain(plaintext_graph_store.clone())
        .await;

    let plaintext = (plaintext_vector_store, plaintext_graph_store);
    let secret = (protocol_store, protocol_graph);
    Ok((plaintext, secret))
}

pub async fn ng_create_from_scratch_hawk_searcher<R: RngCore + Clone + CryptoRng>(
    rng: &mut R,
    database_size: usize,
) -> eyre::Result<(
    LocalNetAby3NgStoreProtocol,
    GraphMem<LocalNetAby3NgStoreProtocol>,
)> {
    let mut rng_searcher = AesRng::from_rng(rng.clone())?;
    let cleartext_database = IrisDB::new_random_rng(database_size, rng).db;
    let shared_irises: Vec<_> = (0..database_size)
        .map(|id| generate_galois_iris_shares(rng, cleartext_database[id].clone()))
        .collect();

    let searcher = HawkSearcher::default();
    let mut aby3_store_protocol = setup_local_store_aby3_players().unwrap();
    let mut graph_store = GraphMem::new();

    let queries = (0..database_size)
        .map(|id| aby3_store_protocol.prepare_query(shared_irises[id].clone()))
        .collect::<Vec<_>>();

    // insert queries
    for query in queries.iter() {
        let neighbors = searcher
            .search_to_insert(&mut aby3_store_protocol, &mut graph_store, query)
            .await;
        searcher
            .insert_from_search_results(
                &mut aby3_store_protocol,
                &mut graph_store,
                &mut rng_searcher,
                *query,
                neighbors,
            )
            .await;
    }

    Ok((aby3_store_protocol, graph_store))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database_generators::generate_galois_iris_shares;
    use aes_prng::AesRng;
    use hawk_pack::{graph_store::GraphMem, hnsw_db::HawkSearcher};
    use iris_mpc_common::iris_db::db::IrisDB;
    use itertools::Itertools;
    use rand::SeedableRng;
    use tracing_test::traced_test;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_gr_hnsw() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 10;
        let cleartext_database = IrisDB::new_random_rng(database_size, &mut rng).db;

        let mut aby3_store = setup_local_store_aby3_players().unwrap();
        let mut aby3_graph = GraphMem::new();
        let db = HawkSearcher::default();

        let queries = (0..database_size)
            .map(|id| {
                aby3_store.prepare_query(generate_galois_iris_shares(
                    &mut rng,
                    cleartext_database[id].clone(),
                ))
            })
            .collect::<Vec<_>>();

        // insert queries
        for query in queries.iter() {
            let neighbors = db
                .search_to_insert(&mut aby3_store, &mut aby3_graph, query)
                .await;
            db.insert_from_search_results(
                &mut aby3_store,
                &mut aby3_graph,
                &mut rng,
                *query,
                neighbors,
            )
            .await;
        }
        println!("FINISHED INSERTING");
        // Search for the same codes and find matches.
        for (index, query) in queries.iter().enumerate() {
            let neighbors = db
                .search_to_insert(&mut aby3_store, &mut aby3_graph, query)
                .await;
            // assert_eq!(false, true);
            tracing::debug!("Finished query");
            assert!(
                db.is_match(&aby3_store, &neighbors).await,
                "failed at index {:?}",
                index
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_gr_premade_hnsw() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 10;
        let (mut cleartext_data, mut secret_data) =
            gr_create_ready_made_hawk_searcher(&mut rng, database_size)
                .await
                .unwrap();

        let mut rng = AesRng::seed_from_u64(0_u64);
        let (mut vector_store, mut graph_store) =
            ng_create_from_scratch_hawk_searcher(&mut rng, database_size)
                .await
                .unwrap();

        assert_eq!(
            vector_store
                .players
                .get(&Identity::from("alice"))
                .unwrap()
                .points,
            secret_data
                .0
                .players
                .get(&Identity::from("alice"))
                .unwrap()
                .points
        );
        assert_eq!(
            vector_store
                .players
                .get(&Identity::from("bob"))
                .unwrap()
                .points,
            secret_data
                .0
                .players
                .get(&Identity::from("bob"))
                .unwrap()
                .points
        );
        assert_eq!(
            vector_store
                .players
                .get(&Identity::from("charlie"))
                .unwrap()
                .points,
            secret_data
                .0
                .players
                .get(&Identity::from("charlie"))
                .unwrap()
                .points
        );
        let hawk_searcher = HawkSearcher::default();

        for i in 0..database_size {
            let cleartext_neighbors = hawk_searcher
                .search_to_insert(&mut cleartext_data.0, &mut cleartext_data.1, &PointId(i))
                .await;
            assert!(
                hawk_searcher
                    .is_match(&cleartext_data.0, &cleartext_neighbors)
                    .await,
            );

            let secret_neighbors = hawk_searcher
                .search_to_insert(&mut secret_data.0, &mut secret_data.1, &PointId(i))
                .await;
            assert!(
                hawk_searcher
                    .is_match(&secret_data.0, &secret_neighbors)
                    .await
            );

            let scratch_secret_neighbors = hawk_searcher
                .search_to_insert(&mut vector_store, &mut graph_store, &PointId(i))
                .await;
            assert!(
                hawk_searcher
                    .is_match(&vector_store, &scratch_secret_neighbors)
                    .await,
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_gr_aby3_store_plaintext() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let db_dim = 4;
        let cleartext_database = IrisDB::new_random_rng(db_dim, &mut rng).db;

        let mut aby3_store_protocol = setup_local_store_aby3_players().unwrap();

        let aby3_preps: Vec<_> = (0..db_dim)
            .map(|id| {
                aby3_store_protocol.prepare_query(generate_galois_iris_shares(
                    &mut rng,
                    cleartext_database[id].clone(),
                ))
            })
            .collect();
        let mut aby3_inserts = Vec::new();
        for p in aby3_preps.iter() {
            aby3_inserts.push(aby3_store_protocol.insert(p).await);
        }
        // Now do the work for the plaintext store
        let mut plaintext_store = PlaintextStore::default();
        let plaintext_preps: Vec<_> = (0..db_dim)
            .map(|id| plaintext_store.prepare_query(cleartext_database[id].clone()))
            .collect();
        let mut plaintext_inserts = Vec::new();
        for p in plaintext_preps.iter() {
            plaintext_inserts.push(plaintext_store.insert(p).await);
        }
        let it1 = (0..db_dim).combinations(2);
        let it2 = (0..db_dim).combinations(2);
        for comb1 in it1 {
            for comb2 in it2.clone() {
                let distance1 = aby3_store_protocol
                    .eval_distance(&aby3_inserts[comb1[0]], &aby3_inserts[comb1[1]])
                    .await;
                let distance2 = aby3_store_protocol
                    .eval_distance(&aby3_inserts[comb2[0]], &aby3_inserts[comb2[1]])
                    .await;
                assert_eq!(
                    aby3_store_protocol.less_than(&distance1, &distance2).await,
                    plaintext_store
                        .less_than(
                            &(plaintext_inserts[comb1[0]], plaintext_inserts[comb1[1]]),
                            &(plaintext_inserts[comb2[0]], plaintext_inserts[comb2[1]])
                        )
                        .await,
                    "Failed at combo: {:?}, {:?}",
                    comb1,
                    comb2
                )
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_gr_scratch_hnsw() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 2;
        let searcher = HawkSearcher::default();
        let (mut vector, mut graph) = ng_create_from_scratch_hawk_searcher(&mut rng, database_size)
            .await
            .unwrap();

        for i in 0..database_size {
            let secret_neighbors = searcher
                .search_to_insert(&mut vector, &mut graph, &PointId(i))
                .await;
            assert!(
                searcher.is_match(&vector, &secret_neighbors).await,
                "Failed at index {:?}",
                i
            );
        }
    }
}
