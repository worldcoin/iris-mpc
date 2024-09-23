use crate::{
    database_generators::{create_shared_database_raw, generate_iris_shares, SharedIris},
    execution::{player::Identity, session::SessionId},
    hawkers::plaintext_store::{PlaintextStore, PointId},
    next_gen_protocol::ng_worker::{
        rep3_single_iris_match_public_output, replicated_lift_and_cross_mul,
        replicated_pairwise_distance, LocalRuntime,
    },
    prelude::{
        IrisWorker, NetworkEstablisher, PartyTestNetwork, TestNetwork3p, TestNetworkEstablisher,
    },
};
use aes_prng::AesRng;
use hawk_pack::{graph_store::graph_mem::GraphMem, hnsw_db::HawkSearcher, VectorStore};
use iris_mpc_common::iris_db::{db::IrisDB, iris::IrisCode};
use rand::{RngCore, SeedableRng};
use std::{collections::VecDeque, sync::Arc};
use tokio::{sync::Mutex, task::JoinSet};

#[derive(Clone)]
pub struct Aby3StorePlayer {
    state:  Arc<Mutex<IrisWorker<PartyTestNetwork>>>,
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
    pub fn new(proto: IrisWorker<PartyTestNetwork>) -> Self {
        Aby3StorePlayer {
            state:  Arc::new(Mutex::new(proto)),
            points: vec![],
        }
    }
    pub fn new_with_shared_db(proto: IrisWorker<PartyTestNetwork>, data: Vec<SharedIris>) -> Self {
        let points: Vec<Point> = data
            .into_iter()
            .map(|d| Point {
                data:          d,
                is_persistent: false,
            })
            .collect();
        Aby3StorePlayer {
            state: Arc::new(Mutex::new(proto)),
            points,
        }
    }

    pub fn prepare_query(&mut self, raw_query: SharedIris) -> <Self as VectorStore>::QueryRef {
        self.points.push(Point {
            data:          raw_query,
            is_persistent: false,
        });

        let point_id = self.points.len() - 1;
        PointId(point_id)
    }

    pub async fn compute_distance(&self, x: &PointId, y: &PointId) -> u16 {
        let x = &self.points[x.0];
        let y = &self.points[y.0];

        let state = Arc::clone(&self.state);
        let mut lock = state.lock().await;

        let d = (*lock)
            .rep3_dot(&x.data.shares, &y.data.shares)
            .await
            .unwrap();
        let open = (*lock).open_async(d).await.unwrap();
        open.0
    }
}

impl VectorStore for Aby3StorePlayer {
    type QueryRef = PointId; // Vector ID, pending insertion.
    type VectorRef = PointId; // Vector ID, inserted.
    type DistanceRef = (PointId, PointId); // Lazy distance representation.

    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        // The query is now accepted in the store. It keeps the same ID.
        self.points[query.0].is_persistent = true;
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
        let x = &self.points[distance.0 .0];
        let y = &self.points[distance.1 .0];

        let (iris_to_match, mask_iris) = (&x.data.shares, &x.data.mask);
        let (ground_truth, mask_ground_truth) = (&y.data.shares, &y.data.mask);
        let state = Arc::clone(&self.state);
        let mut lock = state.lock().await;
        let res = tokio::task::block_in_place(move || {
            let res = (*lock)
                .rep3_single_iris_match_public_output(
                    iris_to_match.as_slice(),
                    ground_truth.clone(),
                    mask_iris,
                    *mask_ground_truth,
                )
                .unwrap();
            res
        });
        res
    }

    async fn less_than(
        &self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> bool {
        let (x1, y1) = (
            &self.points[distance1.0.val()],
            &self.points[distance1.1.val()],
        );
        let (x2, y2) = (
            &self.points[distance2.0.val()],
            &self.points[distance2.1.val()],
        );

        let state = Arc::clone(&self.state);
        let mut lock = state.lock().await;
        let (d1, t1) = (*lock)
            .rep3_pairwise_distance(
                &x1.data.shares,
                &y1.data.shares,
                &x1.data.mask,
                &y1.data.mask,
            )
            .await
            .unwrap();
        let (d2, t2) = (*lock)
            .rep3_pairwise_distance(
                &x2.data.shares,
                &y2.data.shares,
                &x2.data.mask,
                &y2.data.mask,
            )
            .await
            .unwrap();

        // Now need to compute d2*t1 - d1*t2
        tokio::task::block_in_place(move || {
            (*lock)
                .rep3_lift_and_cross_mul(d1, t1 as u32, d2, t2 as u32)
                .unwrap()
        })
    }
}

#[derive(Clone)]
pub struct LocalNetAby3StoreProtocol {
    pub player_0: Aby3StorePlayer,
    pub player_1: Aby3StorePlayer,
    pub player_2: Aby3StorePlayer,
    pub runtime:  LocalRuntime,
}

async fn setup_local_player(mut net: TestNetworkEstablisher) -> eyre::Result<Aby3StorePlayer> {
    tracing::info!("setup network..");
    let net_channel = net.open_channel().await?;
    let protocol = IrisWorker::new(net_channel);
    let aby3_store = Aby3StorePlayer::new(protocol);
    Ok(aby3_store)
}

async fn setup_local_player_preloaded_db(
    mut net: TestNetworkEstablisher,
    database: Vec<SharedIris>,
) -> eyre::Result<Aby3StorePlayer> {
    tracing::info!("setup network..");
    let net_channel = net.open_channel().await?;
    let protocol = IrisWorker::new(net_channel);
    let aby3_store = Aby3StorePlayer::new_with_shared_db(protocol, database);
    Ok(aby3_store)
}

pub async fn setup_local_store_aby3_players() -> eyre::Result<LocalNetAby3StoreProtocol> {
    let amount_workers = 1;
    let mut n1 = VecDeque::with_capacity(amount_workers);
    let mut n2 = VecDeque::with_capacity(amount_workers);
    let mut n3 = VecDeque::with_capacity(amount_workers);

    for _ in 0..amount_workers {
        let network = TestNetwork3p::new();
        let [net1, net2, net3] = network.get_party_networks();
        n1.push_back(net1);
        n2.push_back(net2);
        n3.push_back(net3);
    }

    let n0 = TestNetworkEstablisher::from(n1);
    let n1 = TestNetworkEstablisher::from(n2);
    let n2 = TestNetworkEstablisher::from(n3);

    let player_0 = setup_local_player(n0).await?;
    let player_1 = setup_local_player(n1).await?;
    let player_2 = setup_local_player(n2).await?;
    let runtime = LocalRuntime::replicated_test_config();
    Ok(LocalNetAby3StoreProtocol {
        player_0,
        player_1,
        player_2,
        runtime,
    })
}

pub async fn setup_local_aby3_players_with_preloaded_db<R: RngCore>(
    rng: &mut R,
    database: Vec<IrisCode>,
) -> eyre::Result<LocalNetAby3StoreProtocol> {
    let amount_workers = 1;
    let mut n1 = VecDeque::with_capacity(amount_workers);
    let mut n2 = VecDeque::with_capacity(amount_workers);
    let mut n3 = VecDeque::with_capacity(amount_workers);

    for _ in 0..amount_workers {
        let network = TestNetwork3p::new();
        let [net1, net2, net3] = network.get_party_networks();
        n1.push_back(net1);
        n2.push_back(net2);
        n3.push_back(net3);
    }

    let n0 = TestNetworkEstablisher::from(n1);
    let n1 = TestNetworkEstablisher::from(n2);
    let n2 = TestNetworkEstablisher::from(n3);

    let shared_db = create_shared_database_raw(rng, &database)?;

    let player_0 = setup_local_player_preloaded_db(n0, shared_db.player0_shares).await?;
    let player_1 = setup_local_player_preloaded_db(n1, shared_db.player1_shares).await?;
    let player_2 = setup_local_player_preloaded_db(n2, shared_db.player2_shares).await?;
    let runtime = LocalRuntime::replicated_test_config();
    Ok(LocalNetAby3StoreProtocol {
        player_0,
        player_1,
        player_2,
        runtime,
    })
}

fn local_call_async_setup(
    protocol: Arc<Mutex<IrisWorker<PartyTestNetwork>>>,
) -> tokio::task::JoinHandle<()> {
    let my_protocol = Arc::clone(&protocol);
    tracing::info!("Calling async setup...");
    tokio::spawn(async move {
        let mut lock = my_protocol.lock().await;
        (*lock).setup_prf().await.unwrap();
    })
}

impl LocalNetAby3StoreProtocol {
    pub async fn prf_key_setup(self) -> eyre::Result<LocalNetAby3StoreProtocol> {
        let aby3_protocol = self;
        let h0 = local_call_async_setup(aby3_protocol.player_0.state.clone());
        let h1 = local_call_async_setup(aby3_protocol.player_1.state.clone());
        let h2 = local_call_async_setup(aby3_protocol.player_2.state.clone());
        let (_p0, _p1, _p2) = tokio::join!(h0, h1, h2);
        Ok(aby3_protocol)
    }

    pub fn prepare_query(&mut self, code: Vec<SharedIris>) -> PointId {
        let pid0 = self.player_0.prepare_query(code[0].clone());
        let pid1 = self.player_1.prepare_query(code[1].clone());
        let pid2 = self.player_2.prepare_query(code[2].clone());
        assert_eq!(pid0, pid1);
        assert_eq!(pid1, pid2);
        pid0
    }

    #[cfg(test)]
    async fn compute_distance(&mut self, x: &PointId, y: &PointId) -> u16 {
        let h0 = self.player_0.compute_distance(x, y);
        let h1 = self.player_1.compute_distance(x, y);
        let h2 = self.player_2.compute_distance(x, y);
        let (r0, _r1, _r2) = tokio::join!(h0, h1, h2);
        r0
    }
}

impl VectorStore for LocalNetAby3StoreProtocol {
    type QueryRef = PointId; // Vector ID, pending insertion.
    type VectorRef = PointId; // Vector ID, inserted.
    type DistanceRef = (PointId, PointId); // Lazy distance representation.

    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        // The query is now accepted in the store. It keeps the same ID.
        let q0 = self.player_0.insert(query).await;
        let q1 = self.player_1.insert(query).await;
        let q2 = self.player_2.insert(query).await;
        assert_eq!(q0, q1);
        assert_eq!(q1, q2);
        q0
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
        let mut ready_sessions = self
            .runtime
            .create_player_sessions(SessionId(0))
            .await
            .unwrap();

        let mut jobs = JoinSet::new();
        for player in self.runtime.identities.clone() {
            let mut player_session = ready_sessions.get(&player).unwrap().clone();
            let storage = match player.0.as_str() {
                "alice" => self.player_0.clone(),
                "bob" => self.player_1.clone(),
                "charlie" => self.player_2.clone(),
                _ => panic!(),
            };
            let x = storage.points[distance.0 .0].clone();
            let y = storage.points[distance.1 .0].clone();
            jobs.spawn(async move {
                let (iris_to_match, mask_iris) = (&x.data.shares, &x.data.mask);
                let (ground_truth, mask_ground_truth) = (&y.data.shares, &y.data.mask);

                let r = rep3_single_iris_match_public_output(
                    &mut player_session,
                    iris_to_match.as_slice(),
                    ground_truth.clone(),
                    mask_iris,
                    *mask_ground_truth,
                )
                .await
                .unwrap();
                r
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
        let mut ready_sessions = self
            .runtime
            .create_player_sessions(SessionId(0))
            .await
            .unwrap();

        let mut jobs = JoinSet::new();
        for player in self.runtime.identities.clone() {
            let mut player_session = ready_sessions.get(&player).unwrap().clone();
            // TODO(Dragos) remove these hardcoded lines by having better fields in the
            // global struct.
            let storage = match player.0.as_str() {
                "alice" => self.player_0.clone(),
                "bob" => self.player_1.clone(),
                "charlie" => self.player_2.clone(),
                _ => panic!(),
            };
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
                let res = replicated_lift_and_cross_mul(
                    &mut player_session,
                    d1,
                    t1 as u32,
                    d2,
                    t2 as u32,
                )
                .await
                .unwrap();
                res
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

    let protocol_store =
        setup_local_aby3_players_with_preloaded_db(rng, cleartext_database).await?;
    let next_proto = protocol_store.prf_key_setup().await?;
    let protocol_graph =
        GraphMem::<LocalNetAby3StoreProtocol>::from_another(cleartext_searcher.graph_store.clone());
    let secret_searcher = HawkSearcher::new(next_proto, protocol_graph, &mut rng_searcher2);

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
    let aby3_store_protocol = setup_local_store_aby3_players().await.unwrap();
    let aby3_vector_store = aby3_store_protocol.prf_key_setup().await.unwrap();

    let graph_store = GraphMem::new();

    let mut searcher = HawkSearcher::new(aby3_vector_store, graph_store, &mut rng_searcher);
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
        let aby3_store_protocol = setup_local_store_aby3_players().await.unwrap();
        let mut aby3_store_protocol = aby3_store_protocol.prf_key_setup().await.unwrap();

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
    async fn test_aby3_dot() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let cleartext_database = IrisDB::new_random_rng(10, &mut rng).db;

        let aby3_store_protocol = setup_local_store_aby3_players().await.unwrap();
        let mut aby3_store_protocol = aby3_store_protocol.prf_key_setup().await.unwrap();
        let db_dim = 4;

        let raw_points: Vec<_> = (0..db_dim)
            .map(|id| generate_iris_shares(&mut rng, cleartext_database[id].clone()))
            .collect();

        let aby3_preps: Vec<_> = (0..db_dim)
            .map(|id| aby3_store_protocol.prepare_query(raw_points[id].clone()))
            .collect();
        let mut aby3_inserts = Vec::new();
        for p in aby3_preps.iter() {
            aby3_inserts.push(aby3_store_protocol.insert(p).await);
        }

        let _ = aby3_store_protocol
            .compute_distance(&aby3_inserts[0], &aby3_inserts[1])
            .await;
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_aby3_store_plaintext() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let cleartext_database = IrisDB::new_random_rng(10, &mut rng).db;

        let aby3_store_protocol = setup_local_store_aby3_players().await.unwrap();
        let mut aby3_store_protocol = aby3_store_protocol.prf_key_setup().await.unwrap();
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

        let aby3_store_protocol = setup_local_store_aby3_players().await.unwrap();
        let aby3_vector_store = aby3_store_protocol.prf_key_setup().await.unwrap();

        let graph_store = GraphMem::new();
        let mut db = HawkSearcher::new(aby3_vector_store, graph_store, &mut rng);

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
            scratch_secret_searcher.vector_store.player_0.points,
            secret_searcher.vector_store.player_0.points
        );
        assert_eq!(
            scratch_secret_searcher.vector_store.player_1.points,
            secret_searcher.vector_store.player_1.points
        );
        assert_eq!(
            scratch_secret_searcher.vector_store.player_2.points,
            secret_searcher.vector_store.player_2.points
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
