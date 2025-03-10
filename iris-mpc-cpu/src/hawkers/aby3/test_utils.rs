use std::{collections::HashMap, sync::Arc};

use aes_prng::AesRng;
use iris_mpc_common::iris_db::db::IrisDB;
use rand::{CryptoRng, RngCore, SeedableRng};
use tokio::task::JoinSet;

use crate::{
    execution::local::{generate_local_identities, LocalRuntime},
    hawkers::plaintext_store::PlaintextStore,
    hnsw::{graph::layered_graph::Layer, GraphMem, HnswSearcher, SortedNeighborhood},
    network::NetworkType,
    protocol::shared_iris::GaloisRingSharedIris,
    py_bindings::{io::read_bin, plaintext_store::from_ndjson_file},
    shares::share::DistanceShare,
};

use super::aby3_store::{Aby3Store, IrisRef, SharedIrisesRef, VectorId};

pub fn setup_local_player_preloaded_db(
    database: HashMap<VectorId, IrisRef>,
) -> eyre::Result<SharedIrisesRef> {
    let aby3_store = SharedIrisesRef::new(database);
    Ok(aby3_store)
}

pub async fn setup_local_aby3_players_with_preloaded_db<R: RngCore + CryptoRng>(
    rng: &mut R,
    plain_store: &PlaintextStore,
    network_t: NetworkType,
) -> eyre::Result<Vec<Aby3Store>> {
    let identities = generate_local_identities();

    let mut shared_irises = vec![HashMap::new(); identities.len()];

    for (vector_id, iris) in plain_store.points.iter().enumerate() {
        let vector_id = VectorId::from_serial_id(vector_id as u32);
        let all_shares = GaloisRingSharedIris::generate_shares_locally(rng, iris.data.0.clone());
        for (party_id, share) in all_shares.into_iter().enumerate() {
            shared_irises[party_id].insert(vector_id, Arc::new(share));
        }
    }

    let storages: Vec<SharedIrisesRef> = shared_irises
        .into_iter()
        .map(|player_irises| setup_local_player_preloaded_db(player_irises).unwrap())
        .collect();
    let runtime = LocalRuntime::mock_setup(network_t).await?;

    identities
        .into_iter()
        .zip(storages.into_iter())
        .map(|(identity, storage)| {
            let session = runtime.get_session(&identity)?;
            Ok(Aby3Store {
                session,
                storage,
                owner: identity,
            })
        })
        .collect()
}

pub async fn setup_local_store_aby3_players(
    network_t: NetworkType,
) -> eyre::Result<Vec<Aby3Store>> {
    let runtime = LocalRuntime::mock_setup(network_t).await?;
    generate_local_identities()
        .into_iter()
        .map(|identity| {
            let session = runtime.get_session(&identity)?;
            Ok(Aby3Store {
                session,
                storage: SharedIrisesRef::default(),
                owner: identity,
            })
        })
        .collect()
}

/// Converts a plaintext graph store to a secret-shared graph store.
///
/// If recompute_distance is true, distances are recomputed from scratch via
/// SMPC. Otherwise, distances are naively converted from plaintext ones
/// via trivial shares,
/// i.e., the sharing of a value x is a triple (x, 0, 0).
async fn graph_from_plain(
    vector_store: &mut Aby3Store,
    graph_store: &GraphMem<PlaintextStore>,
    recompute_distances: bool,
) -> GraphMem<Aby3Store> {
    let ep = graph_store.get_entry_point().await;
    let new_ep = ep.map(|(vector_ref, layer_count)| (VectorId { id: vector_ref }, layer_count));

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
                    vector_store
                        .eval_distance_vectors(&source_v, &target_v)
                        .await
                } else {
                    // convert plaintext distances to trivial shares, i.e., d -> (d, 0, 0)
                    DistanceShare::new(
                        vector_store.get_trivial_share(dist.0),
                        vector_store.get_trivial_share(dist.1),
                    )
                };
                shared_queue.push((target_v, distance.clone()));
            }
            shared_links.insert(
                source_v,
                SortedNeighborhood::from_ascending_vec(shared_queue.clone()),
            );
        }
        shared_layers.push(Layer::from_links(shared_links));
    }
    GraphMem::from_precomputed(new_ep, shared_layers)
}

/// Generates 3 pairs of vector stores and graphs from a plaintext
/// vector store and graph read from disk, which are returned as well.
/// The network type is specified by the user.
/// A recompute flag is used to determine whether to recompute the distances
/// from stored shares. If recompute is set to false, the distances are
/// naively converted from plaintext.
pub async fn lazy_setup_from_files<R: RngCore + Clone + CryptoRng>(
    plainstore_file: &str,
    plaingraph_file: &str,
    rng: &mut R,
    database_size: usize,
    network_t: NetworkType,
    recompute_distances: bool,
) -> eyre::Result<(
    (PlaintextStore, GraphMem<PlaintextStore>),
    Vec<(Aby3Store, GraphMem<Aby3Store>)>,
)> {
    if database_size > 100_000 {
        return Err(eyre::eyre!("Database size too large, max. 100,000"));
    }
    let generation_comment = "Please, generate benchmark data with cargo run --release --bin \
                                  generate_benchmark_data.";
    let plaintext_vector_store = from_ndjson_file(plainstore_file, Some(database_size))
        .map_err(|e| eyre::eyre!("Cannot find store: {e}. {generation_comment}"))?;
    let plaintext_graph_store: GraphMem<PlaintextStore> = read_bin(plaingraph_file)
        .map_err(|e| eyre::eyre!("Cannot find graph: {e}. {generation_comment}"))?;

    let protocol_stores =
        setup_local_aby3_players_with_preloaded_db(rng, &plaintext_vector_store, network_t).await?;

    let mut jobs = JoinSet::new();
    for store in protocol_stores.iter() {
        let mut store = store.clone();
        let plaintext_graph_store = plaintext_graph_store.clone();
        jobs.spawn(async move {
            (
                store.clone(),
                graph_from_plain(&mut store, &plaintext_graph_store, recompute_distances).await,
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
    plainstore_file: &str,
    plaingraph_file: &str,
    rng: &mut R,
    database_size: usize,
    recompute_distances: bool,
) -> eyre::Result<(
    (PlaintextStore, GraphMem<PlaintextStore>),
    Vec<(Aby3Store, GraphMem<Aby3Store>)>,
)> {
    lazy_setup_from_files(
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
    Vec<(Aby3Store, GraphMem<Aby3Store>)>,
)> {
    let searcher = HnswSearcher::default();
    let (plaintext_vector_store, plaintext_graph_store) =
        PlaintextStore::create_random(rng, database_size, &searcher).await?;

    let protocol_stores =
        setup_local_aby3_players_with_preloaded_db(rng, &plaintext_vector_store, network_t).await?;

    let mut jobs = JoinSet::new();
    for store in protocol_stores.iter() {
        let mut store = store.clone();
        let plaintext_graph_store = plaintext_graph_store.clone();
        jobs.spawn(async move {
            (
                store.clone(),
                graph_from_plain(&mut store, &plaintext_graph_store, recompute_distances).await,
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
    Vec<(Aby3Store, GraphMem<Aby3Store>)>,
)> {
    lazy_random_setup(
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
    Vec<(Aby3Store, GraphMem<Aby3Store>)>,
)> {
    lazy_random_setup(
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
) -> eyre::Result<Vec<(Aby3Store, GraphMem<Aby3Store>)>> {
    let rng_searcher = AesRng::from_rng(rng.clone())?;
    let cleartext_database = IrisDB::new_random_rng(database_size, rng).db;
    let shared_irises: Vec<_> = (0..database_size)
        .map(|id| {
            GaloisRingSharedIris::generate_shares_locally(rng, cleartext_database[id].clone())
        })
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
            let searcher = HnswSearcher::default();
            // insert queries
            for query in queries.iter() {
                searcher
                    .insert(&mut store, &mut graph_store, query, &mut rng_searcher)
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
) -> eyre::Result<Vec<(Aby3Store, GraphMem<Aby3Store>)>> {
    shared_random_setup(rng, database_size, NetworkType::LocalChannel).await
}

/// Generates 3 pairs of vector stores and graphs corresponding to each
/// local player. Networking is based on gRPC.
pub async fn shared_random_setup_with_grpc<R: RngCore + Clone + CryptoRng>(
    rng: &mut R,
    database_size: usize,
) -> eyre::Result<Vec<(Aby3Store, GraphMem<Aby3Store>)>> {
    shared_random_setup(rng, database_size, NetworkType::GrpcChannel).await
}
