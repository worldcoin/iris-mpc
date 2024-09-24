use crate::{
    database_generators::{create_shared_database_raw, generate_iris_shares, SharedIris},
    execution::player::Identity,
    hawkers::plaintext_store::{PlaintextStore, PointId},
    next_gen_protocol::ng_worker::{
        rep3_single_iris_match_public_output, replicated_lift_and_cross_mul,
        replicated_pairwise_distance, LocalRuntime,
    },
};
use aes_prng::AesRng;
use hawk_pack::{graph_store::graph_mem::GraphMem, hnsw_db::HawkSearcher, VectorStore};
use iris_mpc_common::iris_db::{db::IrisDB, iris::IrisCode};
use rand::{RngCore, SeedableRng};
use std::collections::HashMap;
use tokio::task::JoinSet;

#[derive(Default, Clone)]
pub struct Aby3StorePlayer {
    points: Vec<Point>,
}

impl std::fmt::Debug for Aby3StorePlayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.points.fmt(f)
    }
}

#[derive(Eq, PartialEq, Clone, Debug)]
struct Point {
    /// Whatever encoding of a vector.
    data:          SharedIris,
    /// Distinguish between queries that are pending, and those that were
    /// ultimately accepted into the vector store.
    is_persistent: bool,
}

impl Aby3StorePlayer {
    pub fn new_with_shared_db(data: Vec<SharedIris>) -> Self {
        let points: Vec<Point> = data
            .into_iter()
            .map(|d| Point {
                data:          d,
                is_persistent: false,
            })
            .collect();
        Aby3StorePlayer { points }
    }

    pub fn prepare_query(&mut self, raw_query: SharedIris) -> PointId {
        self.points.push(Point {
            data:          raw_query,
            is_persistent: false,
        });

        let point_id = self.points.len() - 1;
        PointId(point_id)
    }
}

impl Aby3StorePlayer {
    fn insert(&mut self, query: &PointId) -> PointId {
        // The query is now accepted in the store. It keeps the same ID.
        self.points[query.0].is_persistent = true;
        *query
    }
}

#[derive(Debug, Clone)]
pub struct LocalNetAby3StoreProtocol {
    pub players: HashMap<Identity, Aby3StorePlayer>,
    pub runtime: LocalRuntime,
}

fn setup_local_player_preloaded_db(database: Vec<SharedIris>) -> eyre::Result<Aby3StorePlayer> {
    let aby3_store = Aby3StorePlayer::new_with_shared_db(database);
    Ok(aby3_store)
}

pub fn setup_local_store_aby3_players() -> eyre::Result<LocalNetAby3StoreProtocol> {
    let player_0 = Aby3StorePlayer::default();
    let player_1 = Aby3StorePlayer::default();
    let player_2 = Aby3StorePlayer::default();
    let runtime = LocalRuntime::replicated_test_config();
    let players = HashMap::from([
        (Identity::from("alice"), player_0),
        (Identity::from("bob"), player_1),
        (Identity::from("charlie"), player_2),
    ]);
    Ok(LocalNetAby3StoreProtocol { runtime, players })
}

pub fn setup_local_aby3_players_with_preloaded_db<R: RngCore>(
    rng: &mut R,
    database: Vec<IrisCode>,
) -> eyre::Result<LocalNetAby3StoreProtocol> {
    let shared_db = create_shared_database_raw(rng, &database)?;

    let player_0 = setup_local_player_preloaded_db(shared_db.player0_shares)?;
    let player_1 = setup_local_player_preloaded_db(shared_db.player1_shares)?;
    let player_2 = setup_local_player_preloaded_db(shared_db.player2_shares)?;
    let players = HashMap::from([
        (Identity::from("alice"), player_0),
        (Identity::from("bob"), player_1),
        (Identity::from("charlie"), player_2),
    ]);
    let runtime = LocalRuntime::replicated_test_config();
    Ok(LocalNetAby3StoreProtocol { runtime, players })
}

impl LocalNetAby3StoreProtocol {
    pub fn prepare_query(&mut self, code: Vec<SharedIris>) -> PointId {
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

impl VectorStore for LocalNetAby3StoreProtocol {
    type QueryRef = PointId; // Vector ID, pending insertion.
    type VectorRef = PointId; // Vector ID, inserted.
    type DistanceRef = (PointId, PointId); // Lazy distance representation.

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
        // Do not compute the distance yet, just forward the IDs.
        (*query, *vector)
    }

    async fn is_match(&self, distance: &Self::DistanceRef) -> bool {
        // TODO(Dragos) Need to feed in different session
        let ready_sessions = self.runtime.create_player_sessions().await.unwrap();

        let mut jobs = JoinSet::new();
        for player in self.runtime.identities.clone() {
            let mut player_session = ready_sessions.get(&player).unwrap().clone();
            let storage = self.players.get(&player).unwrap();
            let x = storage.points[distance.0 .0].clone();
            let y = storage.points[distance.1 .0].clone();
            jobs.spawn(async move {
                let (iris_to_match, mask_iris) = (&x.data.shares, &x.data.mask);
                let (ground_truth, mask_ground_truth) = (&y.data.shares, &y.data.mask);

                rep3_single_iris_match_public_output(
                    &mut player_session,
                    iris_to_match.as_slice(),
                    ground_truth.clone(),
                    mask_iris,
                    *mask_ground_truth,
                )
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
        let d1 = *distance1;
        let d2 = *distance2;
        let ready_sessions = self.runtime.create_player_sessions().await.unwrap();

        let mut jobs = JoinSet::new();
        for player in self.runtime.identities.clone() {
            let mut player_session = ready_sessions.get(&player).unwrap().clone();
            let storage = self.players.get(&player).unwrap();
            let (x1, y1) = (
                storage.points[d1.0.val()].clone(),
                storage.points[d1.1.val()].clone(),
            );
            let (x2, y2) = (
                storage.points[d2.0.val()].clone(),
                storage.points[d2.1.val()].clone(),
            );

            jobs.spawn(async move {
                let (d1, t1) = replicated_pairwise_distance(
                    &mut player_session,
                    &x1.data.shares.shares,
                    &y1.data.shares.shares,
                    &x1.data.mask,
                    &y1.data.mask,
                )
                .await
                .unwrap();
                let (d2, t2) = replicated_pairwise_distance(
                    &mut player_session,
                    &x2.data.shares.shares,
                    &y2.data.shares.shares,
                    &x2.data.mask,
                    &y2.data.mask,
                )
                .await
                .unwrap();
                replicated_lift_and_cross_mul(&mut player_session, d1, t1 as u32, d2, t2 as u32)
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
}

pub async fn create_ready_made_hawk_searcher<R: RngCore + Clone>(
    rng: &mut R,
    database_size: usize,
) -> eyre::Result<(
    HawkSearcher<PlaintextStore, GraphMem<PlaintextStore>>,
    HawkSearcher<LocalNetAby3StoreProtocol, GraphMem<LocalNetAby3StoreProtocol>>,
)> {
    // makes sure the searcher produces same graph structure by having the same rng
    let mut rng_searcher1 = AesRng::from_rng(rng.clone())?;
    let mut rng_searcher2 = rng_searcher1.clone();

    let cleartext_database = IrisDB::new_random_rng(database_size, rng).db;

    let vector_store = PlaintextStore::default();
    let graph_store = GraphMem::new();
    let mut cleartext_searcher = HawkSearcher::new(vector_store, graph_store, &mut rng_searcher1);

    for raw_query in cleartext_database.iter() {
        let query = cleartext_searcher
            .vector_store
            .prepare_query(raw_query.clone());
        let neighbors = cleartext_searcher.search_to_insert(&query).await;
        let inserted = cleartext_searcher.vector_store.insert(&query).await;
        cleartext_searcher
            .insert_from_search_results(inserted, neighbors)
            .await;
    }

    let protocol_store = setup_local_aby3_players_with_preloaded_db(rng, cleartext_database)?;
    let protocol_graph =
        GraphMem::<LocalNetAby3StoreProtocol>::from_another(cleartext_searcher.graph_store.clone());
    let secret_searcher = HawkSearcher::new(protocol_store, protocol_graph, &mut rng_searcher2);

    Ok((cleartext_searcher, secret_searcher))
}

pub async fn create_from_scratch_hawk_searcher<R: RngCore + Clone>(
    rng: &mut R,
    database_size: usize,
) -> eyre::Result<HawkSearcher<LocalNetAby3StoreProtocol, GraphMem<LocalNetAby3StoreProtocol>>> {
    let mut rng_searcher = AesRng::from_rng(rng.clone())?;
    let cleartext_database = IrisDB::new_random_rng(database_size, rng).db;
    let shared_irises: Vec<_> = (0..database_size)
        .map(|id| generate_iris_shares(rng, cleartext_database[id].clone()))
        .collect();
    let aby3_store_protocol = setup_local_store_aby3_players().unwrap();

    let graph_store = GraphMem::new();

    let mut searcher = HawkSearcher::new(aby3_store_protocol, graph_store, &mut rng_searcher);
    let queries = (0..database_size)
        .map(|id| {
            searcher
                .vector_store
                .prepare_query(shared_irises[id].clone())
        })
        .collect::<Vec<_>>();

    // insert queries
    for query in queries.iter() {
        let neighbors = searcher.search_to_insert(query).await;
        searcher.insert_from_search_results(*query, neighbors).await;
    }

    Ok(searcher)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        database_generators::generate_iris_shares, hawkers::plaintext_store::PlaintextStore,
    };
    use aes_prng::AesRng;
    use hawk_pack::{graph_store::graph_mem::GraphMem, hnsw_db::HawkSearcher};
    use itertools::Itertools;
    use rand::SeedableRng;
    use tracing_test::traced_test;

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_aby3_store_protocol() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let cleartext_database = IrisDB::new_random_rng(10, &mut rng).db;
        let mut aby3_store_protocol = setup_local_store_aby3_players().unwrap();

        let pid0 = aby3_store_protocol.prepare_query(generate_iris_shares(
            &mut rng,
            cleartext_database[0].clone(),
        ));
        let pid1 = aby3_store_protocol.prepare_query(generate_iris_shares(
            &mut rng,
            cleartext_database[1].clone(),
        ));
        let q0 = aby3_store_protocol.insert(&pid0).await;
        let q1 = aby3_store_protocol.insert(&pid1).await;
        let _ = aby3_store_protocol.less_than(&(q0, q1), &(q0, q1)).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_aby3_store_plaintext() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let cleartext_database = IrisDB::new_random_rng(10, &mut rng).db;

        let mut aby3_store_protocol = setup_local_store_aby3_players().unwrap();
        let db_dim = 4;

        let aby3_preps: Vec<_> = (0..db_dim)
            .map(|id| {
                aby3_store_protocol.prepare_query(generate_iris_shares(
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
                assert_eq!(
                    aby3_store_protocol
                        .less_than(
                            &(aby3_inserts[comb1[0]], aby3_inserts[comb1[1]]),
                            &(aby3_inserts[comb2[0]], aby3_inserts[comb2[1]])
                        )
                        .await,
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
    async fn test_hnsw() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 10;
        let cleartext_database = IrisDB::new_random_rng(database_size, &mut rng).db;

        let aby3_store_protocol = setup_local_store_aby3_players().unwrap();
        let graph_store = GraphMem::new();
        let mut db = HawkSearcher::new(aby3_store_protocol, graph_store, &mut rng);

        let queries = (0..database_size)
            .map(|id| {
                db.vector_store.prepare_query(generate_iris_shares(
                    &mut rng,
                    cleartext_database[id].clone(),
                ))
            })
            .collect::<Vec<_>>();

        // insert queries
        for query in queries.iter() {
            let neighbors = db.search_to_insert(query).await;
            db.insert_from_search_results(*query, neighbors).await;
        }
        println!("FINISHED INSERTING");
        // Search for the same codes and find matches.
        for query in queries.iter() {
            let neighbors = db.search_to_insert(query).await;
            // assert_eq!(false, true);
            tracing::debug!("Finished query");
            assert!(db.is_match(&neighbors).await);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_premade_hnsw() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 10;
        let (cleartext_searcher, secret_searcher) =
            create_ready_made_hawk_searcher(&mut rng, database_size)
                .await
                .unwrap();

        let mut rng = AesRng::seed_from_u64(0_u64);
        let scratch_secret_searcher = create_from_scratch_hawk_searcher(&mut rng, database_size)
            .await
            .unwrap();

        assert_eq!(
            scratch_secret_searcher
                .vector_store
                .players
                .get(&Identity::from("alice"))
                .unwrap()
                .points,
            secret_searcher
                .vector_store
                .players
                .get(&Identity::from("alice"))
                .unwrap()
                .points
        );
        assert_eq!(
            scratch_secret_searcher
                .vector_store
                .players
                .get(&Identity::from("bob"))
                .unwrap()
                .points,
            secret_searcher
                .vector_store
                .players
                .get(&Identity::from("bob"))
                .unwrap()
                .points
        );
        assert_eq!(
            scratch_secret_searcher
                .vector_store
                .players
                .get(&Identity::from("charlie"))
                .unwrap()
                .points,
            secret_searcher
                .vector_store
                .players
                .get(&Identity::from("charlie"))
                .unwrap()
                .points
        );

        for i in 0..database_size {
            let cleartext_neighbors = cleartext_searcher.search_to_insert(&PointId(i)).await;
            assert!(cleartext_searcher.is_match(&cleartext_neighbors).await,);

            let secret_neighbors = secret_searcher.search_to_insert(&PointId(i)).await;
            assert!(secret_searcher.is_match(&secret_neighbors).await);

            let scratch_secret_neighbors =
                scratch_secret_searcher.search_to_insert(&PointId(i)).await;
            assert!(
                scratch_secret_searcher
                    .is_match(&scratch_secret_neighbors)
                    .await,
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_scratch_hnsw() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 2;
        let secret_searcher = create_from_scratch_hawk_searcher(&mut rng, database_size)
            .await
            .unwrap();

        for i in 0..database_size {
            let secret_neighbors = secret_searcher.search_to_insert(&PointId(i)).await;
            assert!(
                secret_searcher.is_match(&secret_neighbors).await,
                "Failed at index {:?}",
                i
            );
        }
    }
}
