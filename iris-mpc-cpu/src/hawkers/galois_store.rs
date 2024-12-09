use super::plaintext_store::PlaintextStore;
use crate::{
    database_generators::{generate_galois_iris_shares, GaloisRingSharedIris},
    execution::{
        local::{generate_local_identities, LocalRuntime},
        player::Identity,
        session::Session,
    },
    hawkers::plaintext_store::PointId,
    network::NetworkType,
    protocol::ops::{
        compare_threshold_and_open, cross_compare, galois_ring_pairwise_distance,
        galois_ring_to_rep3,
    },
    py_bindings::{io::read_bin, plaintext_store::from_ndjson_file},
    shares::{
        ring_impl::RingElement,
        share::{DistanceShare, Share},
    },
};
use aes_prng::AesRng;
use hawk_pack::{
    graph_store::{graph_mem::Layer, EntryPoint, GraphMem},
    hnsw_db::{FurthestQueue, HawkSearcher},
    GraphStore, VectorStore,
};
use iris_mpc_common::iris_db::{db::IrisDB, iris::IrisCode};
use rand::{CryptoRng, RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt::Debug, sync::Arc, vec};
use tokio::task::JoinSet;

#[derive(Copy, Default, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VectorId {
    id: PointId,
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

type GaloisRingPoint = GaloisRingSharedIris;

#[derive(Clone, Serialize, Deserialize, Hash, Eq, PartialEq, Debug)]
pub struct Query {
    pub query:           GaloisRingPoint,
    pub processed_query: GaloisRingPoint,
}

type QueryRef = Arc<Query>;

#[derive(Default, Clone)]
pub struct Aby3NgStorePlayer {
    points: Vec<GaloisRingPoint>,
}

impl std::fmt::Debug for Aby3NgStorePlayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.points.fmt(f)
    }
}

impl Aby3NgStorePlayer {
    pub fn new_with_shared_db(data: Vec<GaloisRingSharedIris>) -> Self {
        Aby3NgStorePlayer { points: data }
    }

    pub fn prepare_query(&mut self, raw_query: GaloisRingSharedIris) -> QueryRef {
        let mut preprocessed_query = raw_query.clone();
        preprocessed_query.code.preprocess_iris_code_query_share();
        preprocessed_query.mask.preprocess_mask_code_query_share();

        Arc::new(Query {
            query:           raw_query,
            processed_query: preprocessed_query,
        })
    }

    pub fn get_vector(&self, vector: &VectorId) -> &GaloisRingPoint {
        &self.points[vector.id]
    }
}

impl Aby3NgStorePlayer {
    fn insert(&mut self, query: &QueryRef) -> VectorId {
        // The query is now accepted in the store.
        self.points.push(query.query.clone());

        let new_id = self.points.len() - 1;
        VectorId { id: new_id.into() }
    }
}

pub fn setup_local_player_preloaded_db(
    database: Vec<GaloisRingSharedIris>,
) -> eyre::Result<Aby3NgStorePlayer> {
    let aby3_store = Aby3NgStorePlayer::new_with_shared_db(database);
    Ok(aby3_store)
}

pub async fn setup_local_aby3_players_with_preloaded_db<R: RngCore + CryptoRng>(
    rng: &mut R,
    database: Vec<IrisCode>,
    network_t: NetworkType,
) -> eyre::Result<Vec<LocalNetAby3NgStoreProtocol>> {
    let identities = generate_local_identities();

    let mut shared_irises = vec![vec![]; identities.len()];

    for iris in database {
        let all_shares = generate_galois_iris_shares(rng, iris);
        for (i, shares) in all_shares.iter().enumerate() {
            shared_irises[i].push(shares.clone());
        }
    }

    let storages: Vec<Aby3NgStorePlayer> = shared_irises
        .into_iter()
        .map(|player_irises| setup_local_player_preloaded_db(player_irises).unwrap())
        .collect();
    let runtime = LocalRuntime::mock_setup(network_t).await?;

    let local_stores = identities
        .into_iter()
        .zip(storages.into_iter())
        .map(|(identity, storage)| LocalNetAby3NgStoreProtocol {
            runtime: runtime.clone(),
            storage,
            owner: identity,
        })
        .collect();

    Ok(local_stores)
}

#[derive(Debug, Clone)]
pub struct LocalNetAby3NgStoreProtocol {
    pub owner:   Identity,
    pub storage: Aby3NgStorePlayer,
    pub runtime: LocalRuntime,
}

impl LocalNetAby3NgStoreProtocol {
    pub fn get_owner_session(&self) -> Session {
        self.runtime.sessions.get(&self.owner).unwrap().clone()
    }

    pub fn get_owner_index(&self) -> usize {
        self.runtime
            .role_assignments
            .iter()
            .find_map(|(role, id)| {
                if id.clone() == self.owner {
                    Some(role.clone())
                } else {
                    None
                }
            })
            .unwrap()
            .index()
    }
}

pub async fn setup_local_store_aby3_players(
    network_t: NetworkType,
) -> eyre::Result<Vec<LocalNetAby3NgStoreProtocol>> {
    let runtime = LocalRuntime::mock_setup(network_t).await?;
    let players = generate_local_identities();
    let local_stores = players
        .into_iter()
        .map(|identity| LocalNetAby3NgStoreProtocol {
            runtime: runtime.clone(),
            storage: Aby3NgStorePlayer::default(),
            owner:   identity,
        })
        .collect();
    Ok(local_stores)
}

impl LocalNetAby3NgStoreProtocol {
    pub fn prepare_query(&mut self, code: GaloisRingSharedIris) -> QueryRef {
        self.storage.prepare_query(code)
    }
}

/// Assumes that the first iris of each pair is preprocessed.
async fn eval_pairwise_distances(
    pairs: Vec<(GaloisRingSharedIris, GaloisRingSharedIris)>,
    player_session: &mut Session,
) -> Vec<Share<u16>> {
    let ds_and_ts = galois_ring_pairwise_distance(player_session, &pairs)
        .await
        .unwrap();
    galois_ring_to_rep3(player_session, ds_and_ts)
        .await
        .unwrap()
}

impl VectorStore for LocalNetAby3NgStoreProtocol {
    type QueryRef = QueryRef; // Point ID, pending insertion.
    type VectorRef = VectorId; // Point ID, inserted.
    type DistanceRef = DistanceShare<u16>; // Distance represented as shares.

    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        self.storage.insert(query)
    }

    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Self::DistanceRef {
        let mut player_session = self.get_owner_session();
        let vector_point = self.storage.get_vector(vector);
        let pairs = vec![(query.processed_query.clone(), vector_point.clone())];
        let ds_and_ts = eval_pairwise_distances(pairs, &mut player_session).await;
        DistanceShare::new(ds_and_ts[0].clone(), ds_and_ts[1].clone())
    }

    async fn eval_distance_batch(
        &mut self,
        query: &Self::QueryRef,
        vectors: &[Self::VectorRef],
    ) -> Vec<Self::DistanceRef> {
        let mut player_session = self.get_owner_session();
        let pairs = vectors
            .iter()
            .map(|vector_id| {
                let vector_point = self.storage.get_vector(vector_id);
                (query.processed_query.clone(), vector_point.clone())
            })
            .collect::<Vec<_>>();
        let ds_and_ts = eval_pairwise_distances(pairs, &mut player_session).await;
        ds_and_ts
            .chunks(2)
            .map(|dot_products| {
                DistanceShare::new(dot_products[0].clone(), dot_products[1].clone())
            })
            .collect::<Vec<_>>()
    }

    async fn is_match(&mut self, distance: &Self::DistanceRef) -> bool {
        let mut player_session = self.get_owner_session();
        compare_threshold_and_open(&mut player_session, distance.clone())
            .await
            .unwrap()
    }

    async fn less_than(
        &mut self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> bool {
        let mut player_session = self.get_owner_session();
        let code_dot1 = distance1.code_dot.clone();
        let mask_dot1 = distance1.mask_dot.clone();
        let code_dot2 = distance2.code_dot.clone();
        let mask_dot2 = distance2.mask_dot.clone();
        cross_compare(
            &mut player_session,
            code_dot1,
            mask_dot1,
            code_dot2,
            mask_dot2,
        )
        .await
        .unwrap()
    }
}

impl LocalNetAby3NgStoreProtocol {
    pub fn get_trivial_share(&self, distance: u16) -> Share<u16> {
        let player = self.get_owner_index();
        let distance_elem = RingElement(distance);
        let zero_elem = RingElement(0_u16);

        match player {
            0 => Share::new(distance_elem, zero_elem),
            1 => Share::new(zero_elem, distance_elem),
            2 => Share::new(zero_elem, zero_elem),
            _ => panic!("Invalid player index"),
        }
    }

    async fn eval_distance_vectors(
        &mut self,
        vector1: &<LocalNetAby3NgStoreProtocol as VectorStore>::VectorRef,
        vector2: &<LocalNetAby3NgStoreProtocol as VectorStore>::VectorRef,
    ) -> <LocalNetAby3NgStoreProtocol as VectorStore>::DistanceRef {
        let mut player_session = self.get_owner_session();
        let point1 = self.storage.get_vector(vector1);
        let mut point2 = self.storage.get_vector(vector2).clone();
        point2.code.preprocess_iris_code_query_share();
        point2.mask.preprocess_mask_code_query_share();
        let pairs = vec![(point1.clone(), point2.clone())];
        let ds_and_ts = eval_pairwise_distances(pairs, &mut player_session).await;
        DistanceShare::new(ds_and_ts[0].clone(), ds_and_ts[1].clone())
    }

    async fn graph_from_plain(
        &mut self,
        graph_store: &GraphMem<PlaintextStore>,
        recompute_distances: bool,
    ) -> GraphMem<LocalNetAby3NgStoreProtocol> {
        let ep = graph_store.get_entry_point().await;
        let new_ep = ep.map(|ep| EntryPoint {
            vector_ref:  VectorId { id: ep.vector_ref },
            layer_count: ep.layer_count,
        });

        let layers = graph_store.get_layers();

        let mut shared_layers = vec![];
        for layer in layers {
            let links = layer.get_links_map();
            let mut shared_links = HashMap::new();
            for (source_v, queue) in links {
                let source_v = source_v.into();
                let mut shared_queue = vec![];
                for (target_v, dist) in queue.as_vec_ref() {
                    let target_v = target_v.into();
                    let distance = if recompute_distances {
                        // recompute distances of graph edges from scratch
                        self.eval_distance_vectors(&source_v, &target_v).await
                    } else {
                        DistanceShare::new(
                            self.get_trivial_share(dist.0),
                            self.get_trivial_share(dist.1),
                        )
                    };
                    shared_queue.push((target_v, distance.clone()));
                }
                shared_links.insert(
                    source_v,
                    FurthestQueue::from_ascending_vec(shared_queue.clone()),
                );
            }
            shared_layers.push(Layer::from_links(shared_links));
        }
        GraphMem::from_precomputed(new_ep, shared_layers)
    }
}

impl LocalNetAby3NgStoreProtocol {
    /// Generates 3 pairs of vector stores and graphs from a plaintext
    /// vector store and graph read from disk, which are returned as well.
    /// The network type is specified by the user.
    /// A recompute flag is used to determine whether to recompute the distances
    /// from stored shares. If recompute is set to false, the distances are
    /// naively converted from plaintext.
    pub async fn lazy_setup_from_files<R: RngCore + Clone + CryptoRng>(
        irises_file: &str,
        plainstore_file: &str,
        plaingraph_file: &str,
        rng: &mut R,
        database_size: usize,
        network_t: NetworkType,
        recompute_distances: bool,
    ) -> eyre::Result<(
        (PlaintextStore, GraphMem<PlaintextStore>),
        Vec<(Self, GraphMem<Self>)>,
    )> {
        if database_size > 100_000 {
            return Err(eyre::eyre!("Database size too large, max. 100,000"));
        }
        let cleartext_database: Vec<IrisCode> = read_bin::<Vec<IrisCode>>(irises_file)
            .map_err(|e| eyre::eyre!("Cannot find iris codes: {e}"))?
            .into_iter()
            .take(database_size)
            .collect();
        let plaintext_vector_store = from_ndjson_file(plainstore_file, Some(database_size))
            .map_err(|e| eyre::eyre!("Cannot find store: {e}"))?;
        let plaintext_graph_store: GraphMem<PlaintextStore> =
            read_bin(plaingraph_file).map_err(|e| eyre::eyre!("Cannot find graph: {e}"))?;

        let protocol_stores =
            setup_local_aby3_players_with_preloaded_db(rng, cleartext_database, network_t).await?;

        let mut jobs = JoinSet::new();
        for store in protocol_stores.iter() {
            let mut store = store.clone();
            let plaintext_graph_store = plaintext_graph_store.clone();
            jobs.spawn(async move {
                (
                    store.clone(),
                    store
                        .graph_from_plain(&plaintext_graph_store, recompute_distances)
                        .await,
                )
            });
        }
        let mut secret_shared_stores = jobs.join_all().await;
        secret_shared_stores.sort_by_key(|(store, _)| store.get_owner_index());
        let plaintext = (plaintext_vector_store, plaintext_graph_store);
        Ok((plaintext, secret_shared_stores))
    }

    /// Generates 3 pairs of vector stores and graphs from a plaintext
    /// vector store and graph read from disk, which are returned as well.
    /// Networking is based on gRPC.
    pub async fn lazy_setup_from_files_with_grpc<R: RngCore + Clone + CryptoRng>(
        irises_file: &str,
        plainstore_file: &str,
        plaingraph_file: &str,
        rng: &mut R,
        database_size: usize,
        recompute_distances: bool,
    ) -> eyre::Result<(
        (PlaintextStore, GraphMem<PlaintextStore>),
        Vec<(Self, GraphMem<Self>)>,
    )> {
        Self::lazy_setup_from_files(
            irises_file,
            plainstore_file,
            plaingraph_file,
            rng,
            database_size,
            NetworkType::GrpcChannel,
            recompute_distances,
        )
        .await
    }

    /// Generates 3 pairs of vector stores and graphs from a random plaintext
    /// vector store and graph, which are returned as well.
    /// The network type is specified by the user.
    /// A recompute flag is used to determine whether to recompute the distances
    /// from stored shares. If recompute is set to false, the distances are
    /// naively converted from plaintext.
    pub async fn lazy_random_setup<R: RngCore + Clone + CryptoRng>(
        rng: &mut R,
        database_size: usize,
        network_t: NetworkType,
        recompute_distances: bool,
    ) -> eyre::Result<(
        (PlaintextStore, GraphMem<PlaintextStore>),
        Vec<(Self, GraphMem<Self>)>,
    )> {
        let (cleartext_database, plaintext_vector_store, plaintext_graph_store) =
            PlaintextStore::create_random(rng, database_size).await?;

        let protocol_stores =
            setup_local_aby3_players_with_preloaded_db(rng, cleartext_database, network_t).await?;

        let mut jobs = JoinSet::new();
        for store in protocol_stores.iter() {
            let mut store = store.clone();
            let plaintext_graph_store = plaintext_graph_store.clone();
            jobs.spawn(async move {
                (
                    store.clone(),
                    store
                        .graph_from_plain(&plaintext_graph_store, recompute_distances)
                        .await,
                )
            });
        }
        let mut secret_shared_stores = jobs.join_all().await;
        secret_shared_stores.sort_by_key(|(store, _)| store.get_owner_index());
        let plaintext = (plaintext_vector_store, plaintext_graph_store);
        Ok((plaintext, secret_shared_stores))
    }

    /// Generates 3 pairs of vector stores and graphs from a random plaintext
    /// vector store and graph, which are returned as well. Networking is
    /// based on local async_channel.
    pub async fn lazy_random_setup_with_local_channel<R: RngCore + Clone + CryptoRng>(
        rng: &mut R,
        database_size: usize,
        recompute_distances: bool,
    ) -> eyre::Result<(
        (PlaintextStore, GraphMem<PlaintextStore>),
        Vec<(
            LocalNetAby3NgStoreProtocol,
            GraphMem<LocalNetAby3NgStoreProtocol>,
        )>,
    )> {
        Self::lazy_random_setup(
            rng,
            database_size,
            NetworkType::LocalChannel,
            recompute_distances,
        )
        .await
    }

    /// Generates 3 pairs of vector stores and graphs from a random plaintext
    /// vector store and graph, which are returned as well. Networking is
    /// based on gRPC.
    pub async fn lazy_random_setup_with_grpc<R: RngCore + Clone + CryptoRng>(
        rng: &mut R,
        database_size: usize,
        recompute_distances: bool,
    ) -> eyre::Result<(
        (PlaintextStore, GraphMem<PlaintextStore>),
        Vec<(
            LocalNetAby3NgStoreProtocol,
            GraphMem<LocalNetAby3NgStoreProtocol>,
        )>,
    )> {
        Self::lazy_random_setup(
            rng,
            database_size,
            NetworkType::GrpcChannel,
            recompute_distances,
        )
        .await
    }

    /// Generates 3 pairs of vector stores and graphs corresponding to each
    /// local player.
    pub async fn shared_random_setup<R: RngCore + Clone + CryptoRng>(
        rng: &mut R,
        database_size: usize,
        network_t: NetworkType,
    ) -> eyre::Result<Vec<(Self, GraphMem<Self>)>> {
        let rng_searcher = AesRng::from_rng(rng.clone())?;
        let cleartext_database = IrisDB::new_random_rng(database_size, rng).db;
        let shared_irises: Vec<_> = (0..database_size)
            .map(|id| generate_galois_iris_shares(rng, cleartext_database[id].clone()))
            .collect();

        let mut local_stores = setup_local_store_aby3_players(network_t).await?;

        let mut jobs = JoinSet::new();
        for store in local_stores.iter_mut() {
            let mut store = store.clone();
            let role = store.get_owner_index();
            let mut rng_searcher = rng_searcher.clone();
            let queries = (0..database_size)
                .map(|id| store.prepare_query(shared_irises[id][role].clone()))
                .collect::<Vec<_>>();
            jobs.spawn(async move {
                let mut graph_store = GraphMem::new();
                let searcher = HawkSearcher::default();
                // insert queries
                for query in queries.iter() {
                    let neighbors = searcher
                        .search_to_insert(&mut store, &mut graph_store, query)
                        .await;
                    let inserted_query = store.insert(query).await;
                    searcher
                        .insert_from_search_results(
                            &mut store,
                            &mut graph_store,
                            &mut rng_searcher,
                            inserted_query,
                            neighbors,
                        )
                        .await;
                }
                (store, graph_store)
            });
        }
        let mut result = jobs.join_all().await;
        // preserve order of players
        result.sort_by(|(store1, _), (store2, _)| {
            store1.get_owner_index().cmp(&store2.get_owner_index())
        });
        Ok(result)
    }

    /// Generates 3 pairs of vector stores and graphs corresponding to each
    /// local player. Networking is based on local async_channel.
    pub async fn shared_random_setup_with_local_channel<R: RngCore + Clone + CryptoRng>(
        rng: &mut R,
        database_size: usize,
    ) -> eyre::Result<Vec<(Self, GraphMem<Self>)>> {
        Self::shared_random_setup(rng, database_size, NetworkType::LocalChannel).await
    }

    /// Generates 3 pairs of vector stores and graphs corresponding to each
    /// local player. Networking is based on gRPC.
    pub async fn shared_random_setup_with_grpc<R: RngCore + Clone + CryptoRng>(
        rng: &mut R,
        database_size: usize,
    ) -> eyre::Result<Vec<(Self, GraphMem<Self>)>> {
        Self::shared_random_setup(rng, database_size, NetworkType::GrpcChannel).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database_generators::generate_galois_iris_shares;
    use aes_prng::AesRng;
    use hawk_pack::{graph_store::GraphMem, hnsw_db::HawkSearcher};
    use itertools::Itertools;
    use rand::SeedableRng;
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
                .map(|id| store.prepare_query(shared_irises[id][player_index].clone()))
                .collect::<Vec<_>>();
            let mut store = store.clone();
            let mut rng = rng.clone();
            jobs.spawn(async move {
                let mut aby3_graph = GraphMem::new();
                let db = HawkSearcher::default();

                let mut inserted = vec![];
                // insert queries
                for query in queries.iter() {
                    let neighbors = db
                        .search_to_insert(&mut store, &mut aby3_graph, query)
                        .await;
                    let inserted_query = store.insert(query).await;
                    inserted.push(inserted_query);
                    db.insert_from_search_results(
                        &mut store,
                        &mut aby3_graph,
                        &mut rng,
                        inserted_query,
                        neighbors,
                    )
                    .await;
                }
                tracing::debug!("FINISHED INSERTING");
                // Search for the same codes and find matches.
                let mut matching_results = vec![];
                for v in inserted.into_iter() {
                    let query = store.prepare_query(store.storage.get_vector(&v).clone());
                    let neighbors = db
                        .search_to_insert(&mut store, &mut aby3_graph, &query)
                        .await;
                    tracing::debug!("Finished checking query");
                    matching_results.push(db.is_match(&mut store, &neighbors).await)
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
        let (mut cleartext_data, secret_data) = LocalNetAby3NgStoreProtocol::lazy_random_setup(
            &mut rng,
            database_size,
            network_t.clone(),
            true,
        )
        .await
        .unwrap();

        let mut rng = AesRng::seed_from_u64(0_u64);
        let vector_graph_stores =
            LocalNetAby3NgStoreProtocol::shared_random_setup(&mut rng, database_size, network_t)
                .await
                .unwrap();

        for ((v_from_scratch, _), (premade_v, _)) in
            vector_graph_stores.iter().zip(secret_data.iter())
        {
            assert_eq!(v_from_scratch.storage.points, premade_v.storage.points);
        }
        let hawk_searcher = HawkSearcher::default();

        for i in 0..database_size {
            let cleartext_neighbors = hawk_searcher
                .search_to_insert(&mut cleartext_data.0, &mut cleartext_data.1, &i.into())
                .await;
            assert!(
                hawk_searcher
                    .is_match(&mut cleartext_data.0, &cleartext_neighbors)
                    .await,
            );

            let mut jobs = JoinSet::new();
            for (v, g) in vector_graph_stores.iter() {
                let hawk_searcher = hawk_searcher.clone();
                let mut v = v.clone();
                let mut g = g.clone();
                let q = v.prepare_query(v.storage.get_vector(&i.into()).clone());
                jobs.spawn(async move {
                    let secret_neighbors = hawk_searcher.search_to_insert(&mut v, &mut g, &q).await;

                    hawk_searcher.is_match(&mut v, &secret_neighbors).await
                });
            }
            let scratch_results = jobs.join_all().await;

            let mut jobs = JoinSet::new();
            for (v, g) in secret_data.iter() {
                let hawk_searcher = hawk_searcher.clone();
                let mut v = v.clone();
                let mut g = g.clone();
                jobs.spawn(async move {
                    let query = v.prepare_query(v.storage.get_vector(&i.into()).clone());
                    let secret_neighbors =
                        hawk_searcher.search_to_insert(&mut v, &mut g, &query).await;

                    hawk_searcher.is_match(&mut v, &secret_neighbors).await
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
                .map(|id| store.prepare_query(shared_irises[id][player_index].clone()))
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
        let searcher = HawkSearcher::default();
        let mut vectors_and_graphs = LocalNetAby3NgStoreProtocol::shared_random_setup(
            &mut rng,
            database_size,
            NetworkType::LocalChannel,
        )
        .await
        .unwrap();

        for i in 0..database_size {
            let mut jobs = JoinSet::new();
            for (store, graph) in vectors_and_graphs.iter_mut() {
                let mut store = store.clone();
                let mut graph = graph.clone();
                let searcher = searcher.clone();
                let q = store.prepare_query(store.storage.get_vector(&i.into()).clone());
                jobs.spawn(async move {
                    let secret_neighbors =
                        searcher.search_to_insert(&mut store, &mut graph, &q).await;
                    searcher.is_match(&mut store, &secret_neighbors).await
                });
            }
            let res = jobs.join_all().await;
            for (party_index, r) in res.iter().enumerate() {
                assert!(r, "Failed at index {:?} by party {:?}", i, party_index);
            }
        }
    }
}
