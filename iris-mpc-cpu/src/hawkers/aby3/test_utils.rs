use std::{collections::HashMap, path::Path, sync::Arc};

use aes_prng::AesRng;
use eyre::{bail, Result};
use futures::future::join_all;
use iris_mpc_common::{iris_db::db::IrisDB, vector_id::VectorId};
use rand::{CryptoRng, RngCore, SeedableRng};
use tokio::{sync::Mutex, task::JoinHandle};

use crate::{
    execution::{
        hawk_main::iris_worker,
        local::{generate_local_identities, LocalRuntime},
        session::SessionHandles,
    },
    hawkers::{
        aby3::aby3_store::{Aby3Query, Aby3SharedIrisesRef, Aby3VectorRef},
        plaintext_store::{PlaintextStore, PlaintextVectorRef},
    },
    hnsw::{
        graph::{layered_graph::Layer, neighborhood::SortedEdgeIds},
        GraphMem, HnswSearcher, VectorStore,
    },
    network::NetworkType,
    protocol::shared_iris::GaloisRingSharedIris,
    shares::{RingElement, Share},
    utils::serialization::{
        graph::{read_graph_from_file, GraphFormat},
        iris_ndjson::{from_ndjson_file, IrisSelection},
    },
};

use super::aby3_store::Aby3Store;

type Aby3StoreRef = Arc<Mutex<Aby3Store>>;

pub fn setup_aby3_shared_iris_stores_with_preloaded_db<R: RngCore + CryptoRng>(
    rng: &mut R,
    plain_store: &PlaintextStore,
) -> Vec<Aby3SharedIrisesRef> {
    let identities = generate_local_identities();

    let mut shared_irises = vec![HashMap::new(); identities.len()];

    // sort the iris codes by their serial id
    // Collect and sort keys
    let sorted_serial_ids: Vec<_> = plain_store.storage.get_sorted_serial_ids();

    for serial_id in sorted_serial_ids {
        let iris = &plain_store
            .storage
            .get_vector_by_serial_id(serial_id)
            .expect("Key not found in plain store");

        let vector_id = VectorId::from_serial_id(serial_id);
        let all_shares =
            GaloisRingSharedIris::generate_shares_locally(rng, (**iris).as_ref().clone());
        for (party_id, share) in all_shares.into_iter().enumerate() {
            shared_irises[party_id].insert(vector_id, Arc::new(share));
        }
    }

    shared_irises
        .into_iter()
        .map(|db| Aby3Store::new_storage(Some(db)).to_arc())
        .collect()
}

pub async fn setup_local_aby3_players_with_preloaded_db<R: RngCore + CryptoRng>(
    rng: &mut R,
    plain_store: &PlaintextStore,
    network_t: NetworkType,
) -> Result<Vec<Aby3StoreRef>> {
    let storages = setup_aby3_shared_iris_stores_with_preloaded_db(rng, plain_store);
    let runtime = LocalRuntime::mock_setup(network_t).await?;

    runtime
        .sessions
        .into_iter()
        .zip(storages.into_iter())
        .map(|(session, storage)| {
            let workers = iris_worker::init_workers(0, storage.clone(), true);
            Ok(Arc::new(Mutex::new(Aby3Store {
                session,
                storage,
                workers,
            })))
        })
        .collect()
}

pub async fn setup_local_store_aby3_players(network_t: NetworkType) -> Result<Vec<Aby3StoreRef>> {
    let runtime = LocalRuntime::mock_setup(network_t).await?;
    runtime
        .sessions
        .into_iter()
        .map(|session| {
            let storage = Aby3Store::new_storage(None).to_arc();
            let workers = iris_worker::init_workers(0, storage.clone(), true);

            Ok(Arc::new(Mutex::new(Aby3Store {
                session,
                storage: storage.clone(),
                workers,
            })))
        })
        .collect()
}

/// Returns the index of the party in the session, which is used to propagate messages to the correct party.
/// The index must be in the range [0, 2] and unique per party.
pub async fn get_owner_index(store: &Aby3StoreRef) -> Result<usize> {
    let store = store.lock().await;
    Ok(store.session.network_session.own_role().index())
}

/// Returns a trivial share of a distance.
/// That is the additive sharing (distance, 0, 0)
pub fn get_trivial_share(distance: u16, player_index: usize) -> Result<Share<u32>> {
    let distance_elem = RingElement(distance as u32);
    let zero_elem = RingElement(0_u32);

    let res = match player_index {
        0 => Share::new(distance_elem, zero_elem),
        1 => Share::new(zero_elem, distance_elem),
        2 => Share::new(zero_elem, zero_elem),
        _ => {
            bail!("Invalid player index: {player_index}");
        }
    };
    Ok(res)
}

/// Returns the distance between two vectors inserted into Aby3Store.
pub async fn eval_vector_distance(
    store: &mut Aby3Store,
    vector1: &Aby3VectorRef,
    vector2: &Aby3VectorRef,
) -> Result<<Aby3Store as VectorStore>::DistanceRef> {
    let point1 = store.storage.get_vector_or_empty(vector1).await;
    let mut point2 = (*store.storage.get_vector_or_empty(vector2).await).clone();
    point2.code.preprocess_iris_code_query_share();
    point2.mask.preprocess_mask_code_query_share();
    let pairs = vec![Some((point1.clone(), Arc::new(point2)))];
    let dist = store.eval_pairwise_distances(pairs).await?;
    Ok(dist[0].clone())
}

// TODO Since GraphMem no longer caches distances, this function is now just a
// no-op.  Should refactor it as a call to clone.

/// Converts a plaintext graph store to a secret-shared graph store.
///
/// If recompute_distance is true, distances are recomputed from scratch via
/// SMPC. Otherwise, distances are naively converted from plaintext ones
/// via trivial shares,
/// i.e., the sharing of a value x is a triple (x, 0, 0).
async fn graph_from_plain(graph_store: &GraphMem<PlaintextVectorRef>) -> GraphMem<Aby3VectorRef> {
    let ep = graph_store.get_entry_point().await;
    let layers = graph_store.get_layers();

    let mut shared_layers = vec![];
    for layer in layers {
        let links = layer.get_links_map();
        let mut shared_layer = Layer::new();
        for (source_v, queue) in links {
            let mut shared_queue = vec![];
            for target_v in queue.iter() {
                shared_queue.push(*target_v);
            }
            shared_layer.set_links(*source_v, SortedEdgeIds::from_ascending_vec(shared_queue));
        }
        shared_layers.push(shared_layer);
    }
    GraphMem::from_precomputed(ep, shared_layers)
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
) -> Result<(
    (PlaintextStore, GraphMem<PlaintextVectorRef>),
    Vec<(Aby3StoreRef, GraphMem<Aby3VectorRef>)>,
)> {
    if database_size > 100_000 {
        return Err(eyre::eyre!("Database size too large, max. 100,000"));
    }
    let generation_comment =
        "Please, generate benchmark data with cargo run --release -p iris-mpc-bins --bin \
                                  generate-benchmark-data.";
    let plaintext_vector_store = from_ndjson_file(
        Path::new(plainstore_file),
        Some(database_size),
        IrisSelection::All,
    )
    .map_err(|e| eyre::eyre!("Cannot find store: {e}. {generation_comment}"))?;
    let plaintext_graph_store: GraphMem<PlaintextVectorRef> =
        read_graph_from_file(Path::new(plaingraph_file), GraphFormat::GraphMem)
            .map_err(|e| eyre::eyre!("Cannot find graph: {e}. {generation_comment}"))?;

    let protocol_stores =
        setup_local_aby3_players_with_preloaded_db(rng, &plaintext_vector_store, network_t).await?;

    let mut jobs = vec![];
    for store in protocol_stores.iter() {
        let store = store.clone();
        let plaintext_graph_store = plaintext_graph_store.clone();
        let task = tokio::spawn(async move {
            (
                store.clone(),
                graph_from_plain(&plaintext_graph_store).await,
            )
        });
        jobs.push(task);
    }
    let secret_shared_stores = join_all(jobs)
        .await
        .into_iter()
        .map(|res| res.map_err(eyre::Report::new))
        .collect::<Result<Vec<_>>>()?;
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
) -> Result<(
    (PlaintextStore, GraphMem<PlaintextVectorRef>),
    Vec<(Aby3StoreRef, GraphMem<Aby3VectorRef>)>,
)> {
    lazy_setup_from_files(
        plainstore_file,
        plaingraph_file,
        rng,
        database_size,
        NetworkType::default_grpc(),
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
) -> Result<(
    (PlaintextStore, GraphMem<PlaintextVectorRef>),
    Vec<(Aby3StoreRef, GraphMem<Aby3VectorRef>)>,
)> {
    let searcher = HnswSearcher::new_with_test_parameters();

    let mut plaintext_vector_store = PlaintextStore::new_random(rng, database_size);
    let plaintext_graph_store = plaintext_vector_store
        .generate_graph(rng, database_size, &searcher)
        .await?;

    let protocol_stores =
        setup_local_aby3_players_with_preloaded_db(rng, &plaintext_vector_store, network_t).await?;

    let mut jobs = vec![];
    for store in protocol_stores.iter() {
        let store = store.clone();
        let plaintext_graph_store = plaintext_graph_store.clone();
        let task = tokio::spawn(async move {
            (
                store.clone(),
                graph_from_plain(&plaintext_graph_store).await,
            )
        });
        jobs.push(task);
    }
    let secret_shared_stores = join_all(jobs)
        .await
        .into_iter()
        .map(|res| res.map_err(eyre::Report::new))
        .collect::<Result<Vec<_>>>()?;
    let plaintext = (plaintext_vector_store, plaintext_graph_store);
    Ok((plaintext, secret_shared_stores))
}

/// Generates 3 pairs of vector stores and graphs from a random plaintext
/// vector store and graph, which are returned as well. Networking is
/// based on local async_channel.
pub async fn lazy_random_setup_with_local_channel<R: RngCore + Clone + CryptoRng>(
    rng: &mut R,
    database_size: usize,
) -> Result<(
    (PlaintextStore, GraphMem<PlaintextVectorRef>),
    Vec<(Aby3StoreRef, GraphMem<Aby3VectorRef>)>,
)> {
    lazy_random_setup(rng, database_size, NetworkType::Local).await
}

/// Generates 3 pairs of vector stores and graphs from a random plaintext
/// vector store and graph, which are returned as well. Networking is
/// based on gRPC.
pub async fn lazy_random_setup_with_grpc<R: RngCore + Clone + CryptoRng>(
    rng: &mut R,
    database_size: usize,
) -> Result<(
    (PlaintextStore, GraphMem<PlaintextVectorRef>),
    Vec<(Aby3StoreRef, GraphMem<Aby3VectorRef>)>,
)> {
    lazy_random_setup(rng, database_size, NetworkType::default_grpc()).await
}

/// Generates 3 pairs of vector stores and graphs corresponding to each
/// local player.
pub async fn shared_random_setup<R: RngCore + Clone + CryptoRng>(
    rng: &mut R,
    database_size: usize,
    network_t: NetworkType,
) -> Result<Vec<(Aby3StoreRef, GraphMem<Aby3VectorRef>)>> {
    let rng_searcher = AesRng::from_rng(rng.clone())?;
    let cleartext_database = IrisDB::new_random_rng(database_size, rng).db;
    let shared_irises: Vec<_> = (0..database_size)
        .map(|id| {
            GaloisRingSharedIris::generate_shares_locally(rng, cleartext_database[id].clone())
        })
        .collect();

    let local_stores = setup_local_store_aby3_players(network_t).await?;

    let mut jobs = vec![];
    for store in local_stores.iter() {
        let role = get_owner_index(store).await?;
        let mut rng_searcher = rng_searcher.clone();
        let queries = (0..database_size)
            .map(|id| Aby3Query::new_from_raw(shared_irises[id][role].clone()))
            .collect::<Vec<_>>();
        let store = store.clone();
        let task: JoinHandle<Result<(Aby3StoreRef, GraphMem<Aby3VectorRef>)>> =
            tokio::spawn(async move {
                let mut store_lock = store.lock().await;
                let mut graph_store = GraphMem::new();
                let searcher = HnswSearcher::new_with_test_parameters();
                // insert queries
                for query in queries.iter() {
                    let insertion_layer = searcher.select_layer_rng(&mut rng_searcher)?;
                    searcher
                        .insert(&mut *store_lock, &mut graph_store, query, insertion_layer)
                        .await?;
                }
                Ok((store.clone(), graph_store))
            });
        jobs.push(task);
    }
    let res: Vec<_> = join_all(jobs)
        .await
        .into_iter()
        .map(|res| res.map_err(eyre::Report::new))
        .collect();

    let mut unwrapped = Vec::with_capacity(res.len());
    for r in res {
        unwrapped.push(r??);
    }

    Ok(unwrapped)
}

/// Generates 3 pairs of vector stores and graphs corresponding to each
/// local player. Networking is based on local async_channel.
pub async fn shared_random_setup_with_local_channel<R: RngCore + Clone + CryptoRng>(
    rng: &mut R,
    database_size: usize,
) -> Result<Vec<(Aby3StoreRef, GraphMem<Aby3VectorRef>)>> {
    shared_random_setup(rng, database_size, NetworkType::Local).await
}

/// Generates 3 pairs of vector stores and graphs corresponding to each
/// local player. Networking is based on gRPC.
pub async fn shared_random_setup_with_grpc<R: RngCore + Clone + CryptoRng>(
    rng: &mut R,
    database_size: usize,
) -> Result<Vec<(Aby3StoreRef, GraphMem<Aby3VectorRef>)>> {
    shared_random_setup(rng, database_size, NetworkType::default_grpc()).await
}
