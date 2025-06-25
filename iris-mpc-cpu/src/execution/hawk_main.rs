use super::player::Identity;
pub use crate::hawkers::aby3::aby3_store::VectorId;
use crate::{
    execution::{
        hawk_main::search::SearchIds,
        local::generate_local_identities,
        player::{Role, RoleAssignment},
        session::{NetworkSession, Session, SessionId},
    },
    hawkers::aby3::aby3_store::{
        prepare_query, Aby3Store, Query, SharedIrises, SharedIrisesMut, SharedIrisesRef,
    },
    hnsw::{
        graph::{graph_store, neighborhood::SortedNeighborhoodV},
        searcher::ConnectPlanV,
        GraphMem, HnswParams, HnswSearcher, VectorStore,
    },
    network::tcp::{
        handle::TcpNetworkHandle, networking::connection_builder::PeerConnectionBuilder, TcpConfig,
    },
    protocol::{
        ops::{setup_replicated_prf, setup_shared_seed},
        shared_iris::GaloisRingSharedIris,
    },
};
use clap::Parser;
use eyre::{eyre, Report, Result};
use futures::try_join;
use intra_batch::intra_batch_is_match;
use iris_mpc_common::helpers::{
    smpc_request::{REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE},
    statistics::BucketStatistics,
};
use iris_mpc_common::job::Eye;
use iris_mpc_common::{
    helpers::inmemory_store::InMemoryStore,
    job::{BatchQuery, JobSubmissionHandle},
    ROTATIONS,
};
use iris_mpc_common::{helpers::sync::ModificationKey, job::RequestIndex};
use itertools::{izip, Itertools};
use matching::{
    Decision, Filter, MatchId,
    OnlyOrBoth::{Both, Only},
    RequestType, UniquenessRequest,
};
use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use reset::{apply_deletions, search_to_reset, ResetPlan, ResetRequests};
use scheduler::parallelize;
use search::{SearchParams, SearchQueries};
use serde::{Deserialize, Serialize};
use siphasher::sip::SipHasher13;
use std::{
    collections::HashMap,
    future::Future,
    hash::{Hash, Hasher},
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs},
    ops::Not,
    sync::Arc,
    time::{Duration, Instant},
    vec,
};
use tokio::{
    join,
    sync::{mpsc, oneshot, RwLock, RwLockWriteGuard},
};

pub type GraphStore = graph_store::GraphPg<Aby3Store>;
pub type GraphTx<'a> = graph_store::GraphTx<'a, Aby3Store>;

pub(crate) mod insert;
mod intra_batch;
mod is_match_batch;
mod matching;
mod reset;
mod rot;
pub(crate) mod scheduler;
pub(crate) mod search;
pub mod state_check;
use crate::protocol::ops::{
    compare_threshold_buckets, open_ring, translate_threshold_a, MATCH_THRESHOLD_RATIO,
};
use crate::shares::share::DistanceShare;
use is_match_batch::calculate_missing_is_match;
use rot::VecRots;

#[derive(Clone, Parser)]
#[allow(non_snake_case)]
pub struct HawkArgs {
    #[clap(short, long)]
    pub party_index: usize,

    #[clap(short, long, value_delimiter = ',')]
    pub addresses: Vec<String>,

    #[clap(short, long, default_value_t = 2)]
    pub request_parallelism: usize,

    #[clap(long, default_value_t = 2)]
    pub connection_parallelism: usize,

    #[clap(long, default_value_t = 320)]
    pub hnsw_param_ef_constr: usize,

    #[clap(long, default_value_t = 256)]
    pub hnsw_param_M: usize,

    #[clap(long, default_value_t = 256)]
    pub hnsw_param_ef_search: usize,

    #[clap(long)]
    pub hnsw_prf_key: Option<u64>,

    #[clap(long, default_value_t = false)]
    pub disable_persistence: bool,

    #[clap(long, default_value_t = 64)]
    pub match_distances_buffer_size: usize,

    #[clap(long, default_value_t = 10)]
    pub n_buckets: usize,
}

/// HawkActor manages the state of the HNSW database and connections to other
/// MPC nodes.
pub struct HawkActor {
    args: HawkArgs,

    // ---- Shared setup ----
    searcher: Arc<HnswSearcher>,
    prf_key: Option<Arc<[u8; 16]>>,
    role_assignments: Arc<HashMap<Role, Identity>>,

    // ---- My state ----
    // TODO: Persistence.
    db_size: usize,
    iris_store: BothEyes<SharedIrisesRef>,
    graph_store: BothEyes<GraphRef>,
    anonymized_bucket_statistics: BothEyes<BucketStatistics>,
    distances_cache: BothEyes<Vec<DistanceShare<u32>>>,

    // ---- My network setup ----
    networking: TcpNetworkHandle,
    party_id: usize,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub enum StoreId {
    Left = 0,
    Right = 1,
}

pub const LEFT: usize = 0;
pub const RIGHT: usize = 1;
pub const STORE_IDS: BothEyes<StoreId> = [StoreId::Left, StoreId::Right];

impl TryFrom<usize> for StoreId {
    type Error = Report;

    fn try_from(value: usize) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(StoreId::Left),
            1 => Ok(StoreId::Right),
            _ => Err(eyre!(
                "Invalid usize representation of StoreId, valid inputs are 0 (left) and 1 (right)"
            )),
        }
    }
}

// TODO: Merge with the same in iris-mpc-gpu.
// Orientation enum to indicate the orientation of the iris code during the batch processing.
// Normal: Normal orientation of the iris code.
// Mirror: Mirrored orientation of the iris code: Used to detect full-face mirror attacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Normal = 0,
    Mirror = 1,
}

// TODO: Merge with the same in iris-mpc-gpu.
/// The index in `ServerJobResult::merged_results` which means "no matches and no insertions".
const NON_MATCH_ID: u32 = u32::MAX;

impl std::fmt::Display for StoreId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            StoreId::Left => {
                write!(f, "Left")
            }
            StoreId::Right => {
                write!(f, "Right")
            }
        }
    }
}

/// BothEyes is an alias for types that apply to both left and right eyes.
pub type BothEyes<T> = [T; 2];
/// BothOrient is an alias for types that apply to both orientations (normal and mirror).
pub type BothOrient<T> = [T; 2];
/// VecRequests are lists of things for each request of a batch.
pub(crate) type VecRequests<T> = Vec<T>;
type VecBuckets = Vec<u32>;
/// VecEdges are lists of things for each neighbor of a vector (graph edges).
type VecEdges<T> = Vec<T>;
/// MapEdges are maps from neighbor IDs to something.
type MapEdges<T> = HashMap<VectorId, T>;
/// If true, a match is `left OR right`, otherwise `left AND right`.
type UseOrRule = bool;

type GraphRef = Arc<RwLock<GraphMem<Aby3Store>>>;
pub type GraphMut<'a> = RwLockWriteGuard<'a, GraphMem<Aby3Store>>;

/// HawkSession is a unit of parallelism when operating on the HawkActor.
pub struct HawkSession {
    aby3_store: Aby3Store,
    graph_store: GraphRef,
    hnsw_prf_key: Arc<[u8; 16]>,
}

// Thread safe reference to a HakwSession instance.
pub type HawkSessionRef = Arc<RwLock<HawkSession>>;

pub type SearchResult = (
    <Aby3Store as VectorStore>::VectorRef,
    <Aby3Store as VectorStore>::DistanceRef,
);

/// InsertPlan specifies where a query may be inserted into the HNSW graph.
/// That is lists of neighbors for each layer.
pub type InsertPlan = InsertPlanV<Aby3Store>;

/// ConnectPlan specifies how to connect a new node to the HNSW graph.
/// This includes the updates to the neighbors' own neighbor lists, including
/// bilateral edges.
pub type ConnectPlan = ConnectPlanV<Aby3Store>;

#[derive(Debug)]
pub struct InsertPlanV<V: VectorStore> {
    query: V::QueryRef,
    links: Vec<SortedNeighborhoodV<V>>,
    match_count: usize,
    set_ep: bool,
}
// Manual implementation of Clone for InsertPlan, since derive(Clone) does not propagate the nested Clone bounds on V::QueryRef via TransientRef.
impl Clone for InsertPlan {
    fn clone(&self) -> Self {
        Self {
            query: self.query.clone(),
            links: self.links.clone(),
            match_count: self.match_count,
            set_ep: self.set_ep,
        }
    }
}

impl<V: VectorStore> InsertPlanV<V> {
    pub fn match_ids(&self) -> Vec<V::VectorRef> {
        self.links
            .iter()
            .take(1)
            .flat_map(|bottom_layer| bottom_layer.iter())
            .take(self.match_count)
            .map(|(id, _)| id.clone())
            .collect_vec()
    }
}

impl HawkActor {
    pub async fn from_cli(args: &HawkArgs) -> Result<Self> {
        Self::from_cli_with_graph_and_store(
            args,
            [(); 2].map(|_| GraphMem::<Aby3Store>::new()),
            [(); 2].map(|_| SharedIrises::default()),
        )
        .await
    }

    pub async fn from_cli_with_graph_and_store(
        args: &HawkArgs,
        graph: BothEyes<GraphMem<Aby3Store>>,
        iris_store: BothEyes<SharedIrises>,
    ) -> Result<Self> {
        let search_params = HnswParams::new(
            args.hnsw_param_ef_constr,
            args.hnsw_param_ef_search,
            args.hnsw_param_M,
        );
        let searcher = Arc::new(HnswSearcher {
            params: search_params,
        });

        let identities = generate_local_identities();

        let role_assignments: RoleAssignment = identities
            .iter()
            .enumerate()
            .map(|(index, id)| (Role::new(index), id.clone()))
            .collect();

        let my_index = args.party_index;
        let my_identity = identities[my_index].clone();
        let my_address = &args.addresses[my_index];

        let tcp_config = TcpConfig::new(
            Duration::from_secs(10),
            args.connection_parallelism,
            args.request_parallelism * 2, // x2 for both orientations.
        );
        tracing::debug!("{:?}", tcp_config);

        let connection_builder = PeerConnectionBuilder::new(
            my_identity,
            to_inaddr_any(my_address.parse::<SocketAddr>()?),
            tcp_config.clone(),
        )
        .await?;

        // Connect to other players.
        for (identity, address) in
            izip!(&identities, &args.addresses).filter(|(_, address)| address != &my_address)
        {
            let socket_addr = address
                .clone()
                .to_socket_addrs()?
                .next()
                .ok_or(eyre::eyre!("invalid peer address"))?;
            connection_builder
                .include_peer(identity.clone(), socket_addr)
                .await?;
        }

        let (reconnector, connections) = connection_builder.build().await?;
        let networking = TcpNetworkHandle::new(reconnector, connections, tcp_config);

        let graph_store = graph.map(GraphMem::to_arc);
        let iris_store = iris_store.map(SharedIrises::to_arc);

        let bucket_statistics_left = BucketStatistics::new(
            args.match_distances_buffer_size,
            args.n_buckets,
            my_index,
            Eye::Left,
        );
        let bucket_statistics_right = BucketStatistics::new(
            args.match_distances_buffer_size,
            args.n_buckets,
            my_index,
            Eye::Right,
        );

        Ok(HawkActor {
            args: args.clone(),
            searcher,
            prf_key: None,
            db_size: 0,
            iris_store,
            graph_store,
            anonymized_bucket_statistics: [bucket_statistics_left, bucket_statistics_right],
            distances_cache: [vec![], vec![]],
            role_assignments: Arc::new(role_assignments),
            networking,
            party_id: my_index,
        })
    }

    pub fn searcher(&self) -> Arc<HnswSearcher> {
        self.searcher.clone()
    }

    pub fn iris_store(&self, store_id: StoreId) -> SharedIrisesRef {
        self.iris_store[store_id as usize].clone()
    }

    pub fn graph_store(&self, store_id: StoreId) -> GraphRef {
        self.graph_store[store_id as usize].clone()
    }

    /// Initialize the shared PRF key for HNSW graph insertion layer selection.
    ///
    /// The PRF key is either statically injected via configuration in TEST environments or
    /// mutually derived with other MPC parties in PROD environments.
    ///
    /// This PRF key is used to determine insertion heights for new elements added to the
    /// HNSW graphs, so is configured to be equal across all sessions, and initialized once
    /// upon startup of the `HawkActor` instance.
    async fn get_or_init_prf_key(
        &mut self,
        network_session: &mut NetworkSession,
    ) -> Result<Arc<[u8; 16]>> {
        if self.prf_key.is_none() {
            let prf_key_ = if let Some(prf_key) = self.args.hnsw_prf_key {
                tracing::info!("Initializing HNSW shared PRF key to static value {prf_key:?}");
                (prf_key as u128).to_le_bytes()
            } else {
                tracing::info!("Initializing HNSW shared PRF key to mutually derived random value");
                let my_prf_key = thread_rng().gen();
                setup_shared_seed(network_session, my_prf_key)
                    .await
                    .unwrap_or_else(|err| {
                        tracing::warn!("Unable to initialize shared HNSW PRF key: {err}");
                        tracing::warn!("Using default PRF key value [0u8; 16]");
                        [0u8; 16]
                    })
            };
            let prf_key = Arc::new(prf_key_);

            self.prf_key = Some(prf_key);
        }

        Ok(self.prf_key.as_ref().unwrap().clone())
    }

    pub async fn new_sessions_orient(
        &mut self,
    ) -> Result<BothOrient<BothEyes<Vec<HawkSessionRef>>>> {
        let [mut left, mut right] = self.new_sessions().await?;

        let left_mirror = left.split_off(left.len() / 2);
        let right_mirror = right.split_off(right.len() / 2);
        Ok([[left, right], [left_mirror, right_mirror]])
    }

    pub async fn new_sessions(&mut self) -> Result<BothEyes<Vec<HawkSessionRef>>> {
        let mut network_sessions = vec![];
        for tcp_session in self.networking.make_sessions().await? {
            network_sessions.push(NetworkSession {
                session_id: tcp_session.id(),
                role_assignments: self.role_assignments.clone(),
                networking: Box::new(tcp_session),
                own_role: Role::new(self.party_id),
            });
        }
        let hnsw_prf_key = self.get_or_init_prf_key(&mut network_sessions[0]).await?;

        // todo: replace this with array_chunks::<2>() once that feature
        // is stabilized in Rust.
        let mut it = network_sessions.drain(..);
        let mut left = vec![];
        let mut right = vec![];
        while let (Some(l), Some(r)) = (it.next(), it.next()) {
            left.push(l);
            right.push(r);
        }

        // Futures to create sessions, ids interleaved by side: (Left, 0), (Right, 1), (Left, 2), (Right, 3), ...
        let (sessions_left, sessions_right): (Vec<_>, Vec<_>) = izip!(left, right)
            .map(|(left, right)| {
                (
                    self.create_session(StoreId::Left, left, &hnsw_prf_key),
                    self.create_session(StoreId::Right, right, &hnsw_prf_key),
                )
            })
            .unzip();

        let (l, r) = try_join!(
            parallelize(sessions_left.into_iter()),
            parallelize(sessions_right.into_iter()),
        )?;
        tracing::debug!("Created {} MPC sessions.", self.args.request_parallelism);
        Ok([l, r])
    }

    fn create_session(
        &self,
        store_id: StoreId,
        mut network_session: NetworkSession,
        hnsw_prf_key: &Arc<[u8; 16]>,
    ) -> impl Future<Output = Result<HawkSessionRef>> {
        let storage = self.iris_store(store_id);
        let graph_store = self.graph_store(store_id);
        let hnsw_prf_key = hnsw_prf_key.clone();

        async move {
            let my_session_seed = thread_rng().gen();
            let prf = setup_replicated_prf(&mut network_session, my_session_seed).await?;

            let hawk_session = HawkSession {
                aby3_store: Aby3Store {
                    session: Session {
                        network_session,
                        prf,
                    },
                    storage,
                },
                graph_store,
                hnsw_prf_key,
            };

            Ok(Arc::new(RwLock::new(hawk_session)))
        }
    }

    pub async fn insert(
        &mut self,
        sessions: &[HawkSessionRef],
        plans: VecRequests<Option<InsertPlan>>,
        update_ids: &VecRequests<Option<VectorId>>,
    ) -> Result<VecRequests<Option<ConnectPlan>>> {
        // Plans are to be inserted at the next version of non-None entries in `update_ids`
        let insertion_ids = update_ids
            .iter()
            .map(|id_option| id_option.map(|original_id| original_id.next_version()))
            .collect_vec();

        // Parallel insertions are not supported, so only one session is needed.
        let session = &sessions[0];

        insert::insert(session, &self.searcher, plans, &insertion_ids).await
    }

    async fn update_anon_stats(
        &mut self,
        sessions: &BothEyes<Vec<HawkSessionRef>>,
        search_results: &BothEyes<VecRequests<VecRots<InsertPlan>>>,
    ) -> Result<()> {
        for side in [LEFT, RIGHT] {
            self.cache_distances(side, &search_results[side]);
            let mut session = sessions[side][0].write().await;
            self.fill_anonymized_statistics_buckets(&mut session.aby3_store.session, side)
                .await?;
        }
        Ok(())
    }

    fn calculate_threshold_a(n_buckets: usize) -> Vec<u32> {
        (1..=n_buckets)
            .map(|x: usize| {
                translate_threshold_a(MATCH_THRESHOLD_RATIO / (n_buckets as f64) * (x as f64))
            })
            .collect_vec()
    }

    fn cache_distances(&mut self, side: usize, search_results: &[VecRots<InsertPlan>]) {
        let distances = search_results
            .iter() // All requests.
            .flat_map(|rots| rots.iter()) // All rotations.
            .flat_map(|plan| {
                plan.links.first().into_iter().flat_map(move |neighbors| {
                    neighbors
                        .iter()
                        .take(plan.match_count)
                        .map(|(_, distance)| distance.clone())
                })
            });
        tracing::info!(
            "Keeping {} distances for eye {side} out of {} search results. Cache size: {}/{}",
            distances.clone().count(),
            search_results.len(),
            self.distances_cache[side].len(),
            self.args.match_distances_buffer_size,
        );
        self.distances_cache[side].extend(distances);
    }

    async fn compute_buckets(&self, session: &mut Session, side: usize) -> Result<VecBuckets> {
        let translated_thresholds = Self::calculate_threshold_a(self.args.n_buckets);
        let bucket_result_shares = compare_threshold_buckets(
            session,
            translated_thresholds.as_slice(),
            self.distances_cache[side].as_slice(),
        )
        .await?;

        let buckets = open_ring(session, &bucket_result_shares).await?;
        Ok(buckets)
    }

    async fn fill_anonymized_statistics_buckets(
        &mut self,
        session: &mut Session,
        side: usize,
    ) -> Result<()> {
        if self.distances_cache[side].len() > self.args.match_distances_buffer_size {
            tracing::info!(
                "Gathered enough distances for eye {side}: {}, filling anonymized stats buckets",
                self.distances_cache[side].len()
            );
            let buckets = self.compute_buckets(session, side).await?;
            self.anonymized_bucket_statistics[side].fill_buckets(
                &buckets,
                MATCH_THRESHOLD_RATIO,
                self.anonymized_bucket_statistics[side].next_start_time_utc_timestamp,
            );
            self.distances_cache[side].clear();
        }
        Ok(())
    }

    /// Borrow the in-memory iris and graph stores to modify them.
    pub async fn as_iris_loader(&mut self) -> (IrisLoader, GraphLoader) {
        (
            IrisLoader {
                party_id: self.party_id,
                db_size: &mut self.db_size,
                irises: [
                    self.iris_store[0].write().await,
                    self.iris_store[1].write().await,
                ],
            },
            GraphLoader([
                self.graph_store[0].write().await,
                self.graph_store[1].write().await,
            ]),
        )
    }
}

pub fn session_seeded_rng(base_seed: u64, store_id: StoreId, session_id: SessionId) -> ChaCha8Rng {
    let mut hasher = SipHasher13::new();
    (base_seed, store_id, session_id).hash(&mut hasher);
    let seed = hasher.finish();
    ChaCha8Rng::seed_from_u64(seed)
}

pub struct IrisLoader<'a> {
    party_id: usize,
    db_size: &'a mut usize,
    irises: BothEyes<SharedIrisesMut<'a>>,
}

#[allow(clippy::needless_lifetimes)]
impl<'a> InMemoryStore for IrisLoader<'a> {
    fn load_single_record_from_db(
        &mut self,
        _index: usize, // TODO: Map.
        vector_id: VectorId,
        left_code: &[u16],
        left_mask: &[u16],
        right_code: &[u16],
        right_mask: &[u16],
    ) {
        for (side, code, mask) in izip!(
            &mut self.irises,
            [left_code, right_code],
            [left_mask, right_mask]
        ) {
            let iris = GaloisRingSharedIris::try_from_buffers(self.party_id, code, mask)
                .expect("Wrong code or mask size");
            side.insert(vector_id, iris);
        }
    }

    fn increment_db_size(&mut self, _index: usize) {
        *self.db_size += 1;
    }

    fn reserve(&mut self, additional: usize) {
        for side in &mut self.irises {
            side.reserve(additional);
        }
    }

    fn current_db_sizes(&self) -> impl std::fmt::Debug {
        *self.db_size
    }

    fn fake_db(&mut self, size: usize) {
        *self.db_size = size;
        let iris = Arc::new(GaloisRingSharedIris::default_for_party(self.party_id));
        for side in &mut self.irises {
            for i in 0..size {
                side.insert(VectorId::from_serial_id(i as u32), iris.clone());
            }
        }
    }
}

pub struct GraphLoader<'a>(BothEyes<GraphMut<'a>>);

#[allow(clippy::needless_lifetimes)]
impl<'a> GraphLoader<'a> {
    pub async fn load_graph_store(
        self,
        graph_store: &GraphStore,
        parallelism: usize,
    ) -> Result<()> {
        let now = Instant::now();

        // Spawn two independent transactions and load each graph in parallel.
        let (graph_left, graph_right) = join!(
            async {
                let mut graph_tx = graph_store.tx().await?;
                graph_tx
                    .with_graph(StoreId::Left)
                    .load_to_mem(graph_store.pool(), parallelism)
                    .await
            },
            async {
                let mut graph_tx = graph_store.tx().await?;
                graph_tx
                    .with_graph(StoreId::Right)
                    .load_to_mem(graph_store.pool(), parallelism)
                    .await
            }
        );
        let graph_left = graph_left.expect("Could not load left graph");
        let graph_right = graph_right.expect("Could not load right graph");

        let GraphLoader(mut graphs) = self;
        *graphs[LEFT] = graph_left;
        *graphs[RIGHT] = graph_right;
        tracing::info!(
            "GraphLoader: Loaded left and right graphs in {:?}",
            now.elapsed()
        );
        Ok(())
    }
}

struct HawkJob {
    request: HawkRequest,
    return_channel: oneshot::Sender<Result<HawkResult>>,
}

/// HawkRequest contains a batch of items to search.
#[derive(Clone, Debug)]
pub struct HawkRequest {
    batch: BatchQuery,
    queries: SearchQueries,
    queries_mirror: SearchQueries,
    ids: SearchIds,
}

// TODO: Unify `BatchQuery` and `HawkRequest`.
// TODO: Unify `BatchQueryEntries` and `Vec<GaloisRingSharedIris>`.
impl From<BatchQuery> for HawkRequest {
    fn from(batch: BatchQuery) -> Self {
        let n_queries = batch.request_ids.len();

        let extract_queries = |orient: Orientation| {
            let oriented = match orient {
                Orientation::Normal => [
                    // For left and right eyes.
                    (
                        &batch.left_iris_rotated_requests.code,
                        &batch.left_iris_rotated_requests.mask,
                        &batch.left_iris_interpolated_requests.code,
                        &batch.left_iris_interpolated_requests.mask,
                    ),
                    (
                        &batch.right_iris_rotated_requests.code,
                        &batch.right_iris_rotated_requests.mask,
                        &batch.right_iris_interpolated_requests.code,
                        &batch.right_iris_interpolated_requests.mask,
                    ),
                ],
                Orientation::Mirror => [
                    // Swap the left and right sides to match against the opposite side database:
                    // original left <-> mirrored interpolated right, and vice versa.
                    // The original not-swapped queries are kept for intra-batch matching.
                    (
                        &batch.left_iris_rotated_requests.code,
                        &batch.left_iris_rotated_requests.mask,
                        &batch.right_mirrored_iris_interpolated_requests.code,
                        &batch.right_mirrored_iris_interpolated_requests.mask,
                    ),
                    (
                        &batch.right_iris_rotated_requests.code,
                        &batch.right_iris_rotated_requests.mask,
                        &batch.left_mirrored_iris_interpolated_requests.code,
                        &batch.left_mirrored_iris_interpolated_requests.mask,
                    ),
                ],
            };

            let queries = oriented.map(|(codes, masks, codes_proc, masks_proc)| {
                // Associate the raw and processed versions of codes and masks.
                izip!(codes, masks, codes_proc, masks_proc)
                    // The batch is a concatenation of rotations.
                    .chunks(ROTATIONS)
                    .into_iter()
                    .map(|chunk| {
                        // Collect the rotations for one request.
                        chunk
                            .map(|(code, mask, code_proc, mask_proc)| {
                                // Convert to the type of Aby3Store and into Arc.
                                Query::from_processed(
                                    GaloisRingSharedIris {
                                        code: code.clone(),
                                        mask: mask.clone(),
                                    },
                                    GaloisRingSharedIris {
                                        code: code_proc.clone(),
                                        mask: mask_proc.clone(),
                                    },
                                )
                            })
                            .collect_vec()
                            .into()
                    })
                    .collect_vec()
            });

            assert_eq!(n_queries, queries[LEFT].len());
            assert_eq!(n_queries, queries[RIGHT].len());
            Arc::new(queries)
        };

        let ids = Arc::new(batch.request_ids.clone());

        assert!(n_queries <= batch.requests_order.len());
        assert_eq!(n_queries, batch.request_types.len());
        assert_eq!(n_queries, batch.or_rule_indices.len());
        Self {
            queries: extract_queries(Orientation::Normal),
            queries_mirror: extract_queries(Orientation::Mirror),
            batch,
            ids,
        }
    }
}

impl HawkRequest {
    fn request_types(
        &self,
        iris_store: &SharedIrises,
        orient: Orientation,
    ) -> VecRequests<RequestType> {
        use RequestType::*;

        self.batch
            .request_types
            .iter()
            .enumerate()
            .map(|(i, request_type)| match request_type.as_str() {
                UNIQUENESS_MESSAGE_TYPE => Uniqueness(UniquenessRequest {
                    skip_persistence: *self.batch.skip_persistence.get(i).unwrap(),
                }),
                REAUTH_MESSAGE_TYPE => Reauth(if orient == Orientation::Normal {
                    let request_id = &self.batch.request_ids[i];

                    let or_rule = *self
                        .batch
                        .reauth_use_or_rule
                        .get(request_id)
                        .unwrap_or(&false);

                    self.batch
                        .reauth_target_indices
                        .get(request_id)
                        .map(|&idx| {
                            let target_id = iris_store.from_0_indices(&[idx])[0];
                            (target_id, or_rule)
                        })
                } else {
                    None
                }),
                RESET_CHECK_MESSAGE_TYPE => ResetCheck,
                _ => Unsupported,
            })
            .collect_vec()
    }

    fn queries(&self, orient: Orientation) -> SearchQueries {
        match orient {
            Orientation::Normal => self.queries.clone(),
            Orientation::Mirror => self.queries_mirror.clone(),
        }
    }

    fn luc_ids(&self, iris_store: &SharedIrises) -> VecRequests<Vec<VectorId>> {
        let luc_lookback_ids = iris_store.last_vector_ids(self.batch.luc_lookback_records);

        izip!(&self.batch.or_rule_indices, &self.batch.request_types)
            .map(|(or_rule_idx, request_type)| {
                let mut or_rule_ids = iris_store.from_0_indices(or_rule_idx);

                let lookback =
                    request_type != REAUTH_MESSAGE_TYPE && request_type != RESET_CHECK_MESSAGE_TYPE;
                if lookback {
                    or_rule_ids.extend_from_slice(&luc_lookback_ids);
                };

                or_rule_ids
            })
            .collect_vec()
    }

    fn reset_updates(&self, iris_store: &SharedIrises) -> ResetRequests {
        let queries = [LEFT, RIGHT].map(|side| {
            self.batch
                .reset_update_shares
                .iter()
                .map(|iris| {
                    let iris = if side == LEFT {
                        GaloisRingSharedIris {
                            code: iris.code_left.clone(),
                            mask: iris.mask_left.clone(),
                        }
                    } else {
                        GaloisRingSharedIris {
                            code: iris.code_right.clone(),
                            mask: iris.mask_right.clone(),
                        }
                    };
                    let query = prepare_query(iris);
                    VecRots::new_center_only(query)
                })
                .collect_vec()
        });
        ResetRequests {
            vector_ids: iris_store.from_0_indices(&self.batch.reset_update_indices),
            request_ids: Arc::new(self.batch.reset_update_request_ids.clone()),
            queries: Arc::new(queries),
        }
    }

    fn deletion_ids(&self, iris_store: &SharedIrises) -> Vec<VectorId> {
        iris_store.from_0_indices(&self.batch.deletion_requests_indices)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HawkResult {
    batch: BatchQuery,
    match_results: matching::BatchStep3,
    connect_plans: HawkMutation,
    anonymized_bucket_statistics: BothEyes<BucketStatistics>,
}

impl HawkResult {
    fn new(
        batch: BatchQuery,
        match_results: matching::BatchStep3,
        connect_plans: HawkMutation,
        anonymized_bucket_statistics: BothEyes<BucketStatistics>,
    ) -> Self {
        HawkResult {
            batch,
            match_results,
            connect_plans,
            anonymized_bucket_statistics,
        }
    }

    /// For successful uniqueness insertions, return the inserted index.
    /// For successful reauths, return the index of the updated target.
    /// Otherwise, return the index of some match.
    /// In cases with neither insertions nor matches, return the special u32::MAX.
    fn merged_results(&self) -> Vec<u32> {
        let match_indices = self.select_indices(Filter {
            eyes: Both,
            orient: Both,
            intra_batch: true,
        });

        match_indices
            .into_iter()
            .enumerate()
            .map(|(request_i, match_indices)| {
                if let Some(inserted_id) = self.inserted_id(request_i) {
                    inserted_id.index()
                } else if let Some(&match_index) = match_indices.first() {
                    match_index
                } else {
                    NON_MATCH_ID
                }
            })
            .collect_vec()
    }

    fn inserted_id(&self, request_i: usize) -> Option<VectorId> {
        self.connect_plans
            .get_by_request_index(RequestIndex::UniqueReauthResetCheck(request_i))
            .and_then(|mutation| {
                mutation.plans[LEFT]
                    .as_ref()
                    .or(mutation.plans[RIGHT].as_ref())
                    .map(|plan| plan.inserted_vector)
            })
    }

    fn select(&self, filter: Filter) -> (VecRequests<Vec<u32>>, VecRequests<usize>) {
        let indices = self.select_indices(filter);
        let counts = indices.iter().map(|ids| ids.len()).collect_vec();
        (indices, counts)
    }

    fn select_indices(&self, filter: Filter) -> VecRequests<Vec<u32>> {
        use MatchId::*;

        self.match_results
            .select(filter)
            .iter()
            .map(|matches| {
                matches
                    .iter()
                    .filter_map(|&m| match m {
                        Search(id) | Luc(id) | Reauth(id) => Some(id),
                        IntraBatch(req_i) => self.inserted_id(req_i),
                    })
                    .map(|id| id.index())
                    .sorted()
                    .dedup()
                    .collect_vec()
            })
            .collect_vec()
    }

    fn matched_batch_request_ids(&self) -> Vec<Vec<String>> {
        let per_match = |id: &MatchId| match id {
            MatchId::IntraBatch(req_i) => Some(self.batch.request_ids[*req_i].clone()),
            _ => None,
        };

        self.match_results
            .select(Filter {
                eyes: Both,
                orient: Both,
                intra_batch: true,
            })
            .iter()
            .map(|matches| matches.iter().filter_map(per_match).collect_vec())
            .collect_vec()
    }

    const MATCH_IDS_FILTER: Filter = Filter {
        eyes: Both,
        orient: Only(Orientation::Normal),
        intra_batch: false,
    };

    fn job_result(self) -> ServerJobResult {
        use Decision::*;
        use Orientation::{Mirror, Normal};
        use StoreId::{Left, Right};

        let decisions = self.match_results.decisions();

        let matches = decisions
            .iter()
            .map(|&d| matches!(d, UniqueInsert).not())
            .collect_vec();

        let matches_with_skip_persistence = decisions
            .iter()
            .map(|&d| matches!(d, UniqueInsert | UniqueInsertSkipped).not())
            .collect_vec();

        let match_ids = self.select_indices(Self::MATCH_IDS_FILTER);

        let (partial_match_ids_left, partial_match_counters_left) = self.select(Filter {
            eyes: Only(Left),
            orient: Only(Normal),
            intra_batch: false,
        });

        let (partial_match_ids_right, partial_match_counters_right) = self.select(Filter {
            eyes: Only(Right),
            orient: Only(Normal),
            intra_batch: false,
        });

        let (full_face_mirror_match_ids, _) = self.select(Filter {
            eyes: Both,
            orient: Only(Mirror),
            intra_batch: false,
        });

        let (full_face_mirror_partial_match_ids_left, full_face_mirror_partial_match_counters_left) =
            self.select(Filter {
                eyes: Only(Left),
                orient: Only(Mirror),
                intra_batch: false,
            });

        let (
            full_face_mirror_partial_match_ids_right,
            full_face_mirror_partial_match_counters_right,
        ) = self.select(Filter {
            eyes: Only(Right),
            orient: Only(Mirror),
            intra_batch: false,
        });

        let full_face_mirror_attack_detected = izip!(&match_ids, &full_face_mirror_match_ids)
            .map(|(normal, mirror)| normal.is_empty() && !mirror.is_empty())
            .collect_vec();

        let merged_results = self.merged_results();
        let matched_batch_request_ids = self.matched_batch_request_ids();

        let anonymized_bucket_statistics_left = self.anonymized_bucket_statistics[LEFT].clone();
        let anonymized_bucket_statistics_right = self.anonymized_bucket_statistics[RIGHT].clone();

        let successful_reauths = decisions
            .iter()
            .map(|&d| matches!(d, ReauthUpdate(_)))
            .collect_vec();

        tracing::info!(
            "Reauths: {:?}, Matches: {:?}, Matches w/ skip persistence: {:?}",
            successful_reauths,
            matches,
            matches_with_skip_persistence
        );

        let batch = self.batch;
        let batch_size = batch.request_ids.len();

        ServerJobResult {
            merged_results,
            request_ids: batch.request_ids,
            request_types: batch.request_types,
            metadata: batch.metadata,
            matches_with_skip_persistence,
            matches,

            match_ids,

            partial_match_ids_left,
            partial_match_counters_left,
            partial_match_ids_right,
            partial_match_counters_right,
            partial_match_rotation_indices_left: vec![vec![]; batch_size],
            partial_match_rotation_indices_right: vec![vec![]; batch_size],

            full_face_mirror_match_ids,
            full_face_mirror_partial_match_ids_left,
            full_face_mirror_partial_match_counters_left,
            full_face_mirror_partial_match_ids_right,
            full_face_mirror_partial_match_counters_right,
            full_face_mirror_attack_detected,

            left_iris_requests: batch.left_iris_requests,
            right_iris_requests: batch.right_iris_requests,
            deleted_ids: batch.deletion_requests_indices,
            matched_batch_request_ids,
            anonymized_bucket_statistics_left,
            anonymized_bucket_statistics_right,
            anonymized_bucket_statistics_left_mirror: BucketStatistics::default(), // TODO.
            anonymized_bucket_statistics_right_mirror: BucketStatistics::default(), // TODO.

            successful_reauths,
            reauth_target_indices: batch.reauth_target_indices,
            reauth_or_rule_used: batch.reauth_use_or_rule,

            reset_update_indices: batch.reset_update_indices,
            reset_update_request_ids: batch.reset_update_request_ids,
            reset_update_shares: batch.reset_update_shares,

            modifications: batch.modifications,

            actor_data: self.connect_plans,
        }
    }
}

pub type ServerJobResult = iris_mpc_common::job::ServerJobResult<HawkMutation>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HawkMutation(pub Vec<SingleHawkMutation>);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SingleHawkMutation {
    pub plans: BothEyes<Option<ConnectPlan>>,

    #[serde(skip)]
    pub modification_key: Option<ModificationKey>,

    #[serde(skip)]
    pub request_index: Option<RequestIndex>,
}

impl SingleHawkMutation {
    pub fn serialize(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| eyre::eyre!("Serialization error: {}", e))
    }
}

impl HawkMutation {
    /// Get a serialized `SingleHawkMutation` by `ModificationKey`.
    ///
    /// Returns None if no mutation exists for the given key.
    pub fn get_serialized_mutation_by_key(&self, key: &ModificationKey) -> Option<Vec<u8>> {
        let mutation = self
            .0
            .iter()
            .find(|mutation| mutation.modification_key.as_ref() == Some(key))
            .cloned();
        mutation
            .as_ref()
            .map(|m| m.serialize().expect("failed to serialize graph mutation"))
    }

    pub fn get_by_request_index(&self, req_index: RequestIndex) -> Option<&SingleHawkMutation> {
        self.0
            .iter()
            .find(|mutation| mutation.request_index == Some(req_index))
    }

    pub async fn persist(self, graph_tx: &mut GraphTx<'_>) -> Result<()> {
        tracing::info!("Hawk Main :: Persisting Hawk mutations");
        for mutation in self.0 {
            for (side, plan_opt) in izip!(STORE_IDS, mutation.plans) {
                if let Some(plan) = plan_opt {
                    graph_tx.with_graph(side).insert_apply(plan).await?;
                }
            }
        }
        Ok(())
    }
}

/// HawkHandle is a handle to the HawkActor managing concurrency.
#[derive(Clone, Debug)]
pub struct HawkHandle {
    job_queue: mpsc::Sender<HawkJob>,
}

impl JobSubmissionHandle for HawkHandle {
    type A = HawkMutation;

    async fn submit_batch_query(
        &mut self,
        batch: BatchQuery,
    ) -> impl Future<Output = Result<ServerJobResult>> {
        let request = HawkRequest::from(batch);
        let (tx, rx) = oneshot::channel();
        let job = HawkJob {
            request,
            return_channel: tx,
        };

        // Wait for the job to be sent for backpressure.
        let sent = self.job_queue.send(job).await;

        async move {
            // In a second Future, wait for the result.
            sent?;
            let result = rx.await??;
            Ok(result.job_result())
        }
    }
}

impl HawkHandle {
    pub async fn new(mut hawk_actor: HawkActor) -> Result<Self> {
        let mut sessions = hawk_actor.new_sessions_orient().await?;

        // Validate the common state before starting.
        HawkSession::state_check([&sessions[0][LEFT][0], &sessions[0][RIGHT][0]]).await?;

        let (tx, mut rx) = mpsc::channel::<HawkJob>(1);

        // ---- Request Handler ----
        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                let job_result = Self::handle_job(&mut hawk_actor, &sessions, job.request).await;

                let health =
                    Self::health_check(&mut hawk_actor, &mut sessions, job_result.is_err()).await;

                let stop = health.is_err();
                let _ = job.return_channel.send(health.and(job_result));

                if stop {
                    tracing::error!("Stopping HawkActor in inconsistent state.");
                    break;
                }
            }

            rx.close();
            while let Some(job) = rx.recv().await {
                let _ = job.return_channel.send(Err(eyre::eyre!("stopping")));
            }
        });

        Ok(Self { job_queue: tx })
    }

    async fn handle_job(
        hawk_actor: &mut HawkActor,
        sessions_orient: &BothOrient<BothEyes<Vec<HawkSessionRef>>>,
        request: HawkRequest,
    ) -> Result<HawkResult> {
        tracing::info!("Processing an Hawk job…");
        let now = Instant::now();

        // Deletions.
        apply_deletions(hawk_actor, &request).await?;

        tracing::info!(
            "Processing an Hawk job with request types: {:?}, reauth targets: {:?}, skip persistence: {:?}, reauth use or rule: {:?}",
            request.batch.request_types,
            request.batch.reauth_target_indices,
            request.batch.skip_persistence,
            request.batch.reauth_use_or_rule,
        );

        let do_search = async |orient| -> Result<_> {
            let sessions = &sessions_orient[orient as usize];
            let search_queries = &request.queries(orient);
            let (luc_ids, request_types) = {
                // The store to find vector ids (same left or right).
                let store = hawk_actor.iris_store[LEFT].read().await;
                (
                    request.luc_ids(&store),
                    request.request_types(&store, orient),
                )
            };

            let intra_results = intra_batch_is_match(sessions, search_queries).await?;

            // Search for nearest neighbors.
            // For both eyes, all requests, and rotations.
            let search_ids = &request.ids;
            let search_params = SearchParams {
                hnsw: hawk_actor.searcher(),
                do_match: true,
            };
            let search_results =
                search::search(sessions, search_queries, search_ids, search_params).await?;

            let match_result = {
                let step1 = matching::BatchStep1::new(&search_results, &luc_ids, request_types);

                // Go fetch the missing vector IDs and calculate their is_match.
                let missing_is_match = calculate_missing_is_match(
                    search_queries,
                    step1.missing_vector_ids(),
                    sessions,
                )
                .await?;

                step1.step2(&missing_is_match, intra_results)
            };

            Ok((search_results, match_result))
        };

        let (search_results, match_result) = {
            let ((search_normal, matches_normal), (_, matches_mirror)) = try_join!(
                do_search(Orientation::Normal),
                do_search(Orientation::Mirror),
            )?;

            (search_normal, matches_normal.step3(matches_mirror))
        };
        let sessions = &sessions_orient[Orientation::Normal as usize];

        hawk_actor
            .update_anon_stats(sessions, &search_results)
            .await?;
        tracing::info!("Updated anonymized statistics.");

        // Reset Updates. Find how to insert the new irises into the graph.
        let resets = search_to_reset(hawk_actor, sessions, &request).await?;

        // Insert into the in memory stores.
        let mutations = Self::handle_mutations(
            hawk_actor,
            sessions,
            search_results,
            &match_result,
            resets,
            &request,
        )
        .await?;

        let results = HawkResult::new(
            request.batch,
            match_result,
            mutations,
            hawk_actor.anonymized_bucket_statistics.clone(),
        );

        metrics::histogram!("job_duration").record(now.elapsed().as_secs_f64());
        metrics::gauge!("db_size").set(hawk_actor.db_size as f64);
        let query_count = results.batch.request_ids.len();
        metrics::gauge!("search_queries_left").set(query_count as f64);
        metrics::gauge!("search_queries_right").set(query_count as f64);
        tracing::info!("Finished processing a Hawk job…");
        Ok(results)
    }

    async fn handle_mutations(
        hawk_actor: &mut HawkActor,
        sessions: &BothEyes<Vec<HawkSessionRef>>,
        search_results: BothEyes<VecRequests<VecRots<InsertPlan>>>,
        match_result: &matching::BatchStep3,
        resets: ResetPlan,
        request: &HawkRequest,
    ) -> Result<HawkMutation> {
        use Decision::*;
        let decisions = match_result.decisions();
        let requests_order = &request.batch.requests_order;

        // The vector IDs of reauths and resets, or None for uniqueness insertions.
        let update_ids = requests_order
            .iter()
            .map(|req_index| match req_index {
                RequestIndex::UniqueReauthResetCheck(i) => match decisions[*i] {
                    ReauthUpdate(update_id) => Some(update_id),
                    _ => None,
                },
                RequestIndex::ResetUpdate(i) => Some(resets.vector_ids[*i]),
                RequestIndex::Deletion(_) => None,
            })
            .collect_vec();

        tracing::info!("Updated decisions (reset + reauth): {:?}", update_ids);

        // Store plans for both sides using BothEyes structure
        let mut plans_both_sides: Vec<BothEyes<Option<ConnectPlan>>> =
            vec![[None, None]; requests_order.len()];

        // For both eyes.
        for (side, sessions, search_results, reset_results) in
            izip!(&STORE_IDS, sessions, search_results, resets.search_results)
        {
            let unique_insertions_persistence_skipped = decisions
                .iter()
                .map(|decision| matches!(decision, UniqueInsertSkipped))
                .collect_vec();

            let unique_insertions = decisions
                .iter()
                .map(|decision| matches!(decision, UniqueInsert))
                .collect_vec();

            // The accepted insertions for uniqueness, reauth, and resets.
            // Focus on the insertions and keep only the centered irises.
            tracing::info!(
                "Inserting {} new irises for eye {}",
                search_results.len(),
                side
            );

            tracing::info!(
                "Unique insertions: {}, persistence skipped: {}",
                unique_insertions.len(),
                unique_insertions_persistence_skipped.len()
            );

            let insert_plans = requests_order
                .iter()
                .map(|req_index| match req_index {
                    // If the decision is a mutation, return the insertion plan.
                    RequestIndex::UniqueReauthResetCheck(i) => decisions[*i]
                        .is_mutation()
                        .then(|| search_results[*i].center().clone()),
                    RequestIndex::ResetUpdate(i) => Some(reset_results[*i].center().clone()),
                    RequestIndex::Deletion(_) => None,
                })
                .collect_vec();

            // Insert in memory, and return the plans to update the persistent database.
            let plans = hawk_actor
                .insert(sessions, insert_plans, &update_ids)
                .await?;

            // Store plans for this side
            for (plan, both_sides) in izip!(plans, &mut plans_both_sides) {
                both_sides[*side as usize] = plan;
            }
        }

        // Combine ModificationKey and ConnectPlan into into SingleHawkMutation objects.
        let mut mutations = Vec::new();

        for (req_index, modif_plan) in izip!(requests_order, plans_both_sides) {
            let modification_key = match *req_index {
                RequestIndex::UniqueReauthResetCheck(i) => {
                    // This is a batch request mutation
                    match decisions[i] {
                        UniqueInsert => {
                            let request_id = &request.batch.request_ids[i];
                            Some(ModificationKey::RequestId(request_id.clone()))
                        }
                        ReauthUpdate(vector_id) => {
                            Some(ModificationKey::RequestSerialId(vector_id.serial_id()))
                        }
                        UniqueInsertSkipped | NoMutation => None,
                    }
                }
                RequestIndex::ResetUpdate(i) => {
                    // This is a reset update mutation.
                    if let Some(&vector_id) = resets.vector_ids.get(i) {
                        Some(ModificationKey::RequestSerialId(vector_id.serial_id()))
                    } else {
                        None
                    }
                }
                RequestIndex::Deletion(_) => None,
            };

            mutations.push(SingleHawkMutation {
                plans: modif_plan,
                modification_key,
                request_index: Some(*req_index),
            });
        }

        Ok(HawkMutation(mutations))
    }

    async fn health_check(
        hawk_actor: &mut HawkActor,
        sessions: &mut BothOrient<BothEyes<Vec<HawkSessionRef>>>,
        job_failed: bool,
    ) -> Result<()> {
        if job_failed {
            // There is some error so the sessions may be somehow invalid. Make new ones.
            *sessions = hawk_actor.new_sessions_orient().await?;
        }

        // Validate the common state after processing the requests.
        HawkSession::state_check([&sessions[0][LEFT][0], &sessions[0][RIGHT][0]]).await?;
        Ok(())
    }
}

fn to_inaddr_any(mut socket: SocketAddr) -> SocketAddr {
    if socket.is_ipv4() {
        socket.set_ip(IpAddr::V4(Ipv4Addr::UNSPECIFIED));
    } else {
        socket.set_ip(IpAddr::V6(Ipv6Addr::UNSPECIFIED));
    }
    socket
}

pub async fn hawk_main(args: HawkArgs) -> Result<HawkHandle> {
    println!("🦅 Starting Hawk node {}", args.party_index);
    let hawk_actor = HawkActor::from_cli(&args).await?;
    HawkHandle::new(hawk_actor).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        execution::local::get_free_local_addresses, protocol::shared_iris::GaloisRingSharedIris,
    };
    use aes_prng::AesRng;
    use futures::future::JoinAll;
    use iris_mpc_common::{
        galois_engine::degree4::preprocess_iris_message_shares,
        helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE,
        iris_db::db::IrisDB,
        job::{BatchMetadata, IrisQueryBatchEntries},
    };
    use rand::SeedableRng;
    use std::ops::Not;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_hawk_main() -> Result<()> {
        let go = |addresses: Vec<String>, index: usize| {
            async move {
                let args = HawkArgs::parse_from([
                    "hawk_main",
                    "--addresses",
                    &addresses.join(","),
                    "--party-index",
                    &index.to_string(),
                ]);

                // Make the test async.
                sleep(Duration::from_millis(100 * index as u64)).await;

                hawk_main(args).await.unwrap()
            }
        };

        let n_parties = 3;
        let addresses = get_free_local_addresses(n_parties).await?;

        let handles = (0..n_parties)
            .map(|i| go(addresses.clone(), i))
            .map(tokio::spawn)
            .collect::<JoinAll<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<HawkHandle>, _>>()?;

        // ---- Send requests ----

        let batch_size = 5;
        let iris_rng = &mut AesRng::seed_from_u64(1337);

        // Generate: iris_id -> party -> share
        let irises = IrisDB::new_random_rng(batch_size, iris_rng)
            .db
            .into_iter()
            .map(|iris| {
                (
                    GaloisRingSharedIris::generate_shares_locally(iris_rng, iris.clone()),
                    GaloisRingSharedIris::generate_mirrored_shares_locally(iris_rng, iris),
                )
            })
            .collect_vec();

        // Unzip: party -> iris_id -> (share, share_mirrored)
        let irises = (0..n_parties)
            .map(|party_index| {
                irises
                    .iter()
                    .map(|(iris, iris_mirrored)| {
                        (
                            iris[party_index].clone(),
                            iris_mirrored[party_index].clone(),
                        )
                    })
                    .collect_vec()
            })
            .collect_vec();

        let mut batch_0 = BatchQuery {
            luc_lookback_records: 2,
            ..BatchQuery::default()
        };
        for i in 0..batch_size {
            batch_0.push_matching_request(
                format!("sns_{i}"),
                format!("request_{i}"),
                UNIQUENESS_MESSAGE_TYPE,
                BatchMetadata::default(),
                vec![],
                false,
            );
        }

        let all_results =
            parallelize(izip!(&irises, handles.clone()).map(|(shares, mut handle)| {
                let batch = batch_of_party(&batch_0, shares);
                async move { handle.submit_batch_query(batch).await.await }
            }))
            .await?;

        let result = assert_all_equal(all_results);

        let inserted_indices = (0..batch_size as u32).collect_vec();

        assert_eq!(result.matches, vec![false; batch_size]);
        assert_eq!(result.merged_results, inserted_indices);
        assert_eq!(batch_size, result.request_ids.len());
        assert_eq!(batch_size, result.request_types.len());
        assert_eq!(batch_size, result.metadata.len());
        assert_eq!(batch_size, result.matches_with_skip_persistence.len());
        assert_eq!(result.match_ids, vec![Vec::<u32>::new(); batch_size]);
        assert_eq!(batch_size, result.partial_match_ids_left.len());
        assert_eq!(batch_size, result.partial_match_ids_right.len());
        assert_eq!(batch_size, result.partial_match_counters_left.len());
        assert_eq!(batch_size, result.partial_match_counters_right.len());
        assert_match_ids(&result);
        assert_eq!(batch_size, result.left_iris_requests.code.len());
        assert_eq!(batch_size, result.right_iris_requests.code.len());
        assert!(result.deleted_ids.is_empty());
        assert_eq!(batch_size, result.matched_batch_request_ids.len());
        assert!(result.anonymized_bucket_statistics_left.buckets.is_empty());
        assert!(result.anonymized_bucket_statistics_right.buckets.is_empty());
        assert_eq!(batch_size, result.successful_reauths.len());
        assert!(result.reauth_target_indices.is_empty());
        assert!(result.reauth_or_rule_used.is_empty());
        assert!(result.modifications.is_empty());
        assert_eq!(batch_size, result.actor_data.0.len());

        // --- Reauth ---

        let batch_1 = BatchQuery {
            request_types: vec![REAUTH_MESSAGE_TYPE.to_string(); batch_size],

            // Map the request ID to the inserted index.
            reauth_target_indices: izip!(&batch_0.request_ids, &inserted_indices)
                .map(|(req_id, inserted_index)| (req_id.clone(), *inserted_index))
                .collect(),
            reauth_use_or_rule: batch_0
                .request_ids
                .iter()
                .map(|req_id| (req_id.clone(), false))
                .collect(),

            ..batch_0.clone()
        };

        let failed_request_i = 1;
        let all_results = parallelize((0..n_parties).map(|party_i| {
            // Mess with the shares to make one request fail.
            let mut shares = irises[party_i].clone();
            shares[failed_request_i].0 = GaloisRingSharedIris::dummy_for_party(party_i);

            let batch = batch_of_party(&batch_1, &shares);
            let mut handle = handles[party_i].clone();
            async move { handle.submit_batch_query(batch).await.await }
        }))
        .await?;

        let result = assert_all_equal(all_results);
        assert_eq!(
            result.successful_reauths,
            (0..batch_size).map(|i| i != failed_request_i).collect_vec()
        );

        // --- Rejected Uniqueness ---

        let batch_2 = batch_0;

        let all_results = parallelize((0..n_parties).map(|party_i| {
            let batch = batch_of_party(&batch_2, &irises[party_i]);
            let mut handle = handles[party_i].clone();
            async move { handle.submit_batch_query(batch).await.await }
        }))
        .await?;
        let result = assert_all_equal(all_results);

        assert_eq!(
            result.match_ids.iter().map(|ids| ids[0]).collect_vec(),
            inserted_indices,
        );
        assert_eq!(result.merged_results, inserted_indices);
        assert_eq!(result.matches, vec![true; batch_size]);
        assert_match_ids(&result);

        Ok(())
    }

    /// Prepare shares in the same format as `receive_batch()`.
    fn receive_batch_shares(
        shares: Vec<GaloisRingSharedIris>,
        mirrored_shares: Vec<GaloisRingSharedIris>,
    ) -> [IrisQueryBatchEntries; 4] {
        let mut out = [(); 4].map(|_| IrisQueryBatchEntries::default());
        for (share, mirrored_share) in izip!(shares, mirrored_shares) {
            let one = preprocess_iris_message_shares(
                share.code,
                share.mask,
                mirrored_share.code,
                mirrored_share.mask,
            )
            .unwrap();
            out[0].code.push(one.code);
            out[0].mask.push(one.mask);
            out[1].code.extend(one.code_rotated);
            out[1].mask.extend(one.mask_rotated);
            out[2].code.extend(one.code_interpolated.clone());
            out[2].mask.extend(one.mask_interpolated.clone());
            out[3].code.extend(one.code_mirrored);
            out[3].mask.extend(one.mask_mirrored);
        }
        out
    }

    // Prepare a batch for a particular party, setting their shares.
    fn batch_of_party(
        batch: &BatchQuery,
        shares: &[(GaloisRingSharedIris, GaloisRingSharedIris)],
    ) -> BatchQuery {
        // TODO: different test irises for each eye.
        let shares_right_cloned = shares.to_vec();
        let shares_left_cloned = shares.to_vec();

        let shares_right = shares_right_cloned
            .clone()
            .into_iter()
            .map(|(share, _)| share)
            .collect();
        let shares_right_mirrored = shares_right_cloned
            .into_iter()
            .map(|(_, share)| share)
            .collect();

        let shares_left = shares_left_cloned
            .clone()
            .into_iter()
            .map(|(share, _)| share)
            .collect();
        let shares_left_mirrored = shares_left_cloned
            .into_iter()
            .map(|(_, share)| share)
            .collect();

        let [left_iris_requests, left_iris_rotated_requests, left_iris_interpolated_requests, left_mirrored_iris_interpolated_requests] =
            receive_batch_shares(shares_right, shares_right_mirrored);
        let [right_iris_requests, right_iris_rotated_requests, right_iris_interpolated_requests, right_mirrored_iris_interpolated_requests] =
            receive_batch_shares(shares_left, shares_left_mirrored);

        BatchQuery {
            // Iris shares.
            left_iris_requests,
            right_iris_requests,
            // All rotations.
            left_iris_rotated_requests,
            right_iris_rotated_requests,
            // All rotations, preprocessed.
            left_iris_interpolated_requests,
            right_iris_interpolated_requests,
            // All rotations, preprocessed, mirrored.
            left_mirrored_iris_interpolated_requests,
            right_mirrored_iris_interpolated_requests,
            // Details common to all parties.
            ..batch.clone()
        }
    }

    fn assert_all_equal(mut all_results: Vec<ServerJobResult>) -> ServerJobResult {
        // Ignore the actual secret shares because they are different for each party.
        for i in 1..all_results.len() {
            all_results[i].left_iris_requests = all_results[0].left_iris_requests.clone();
            all_results[i].right_iris_requests = all_results[0].right_iris_requests.clone();
            // Same for specific fields of the bucket statistics.
            // TODO: specific assertions for the bucket statistics results
            all_results[i].anonymized_bucket_statistics_left.party_id =
                all_results[0].anonymized_bucket_statistics_left.party_id;
            all_results[i].anonymized_bucket_statistics_right.party_id =
                all_results[0].anonymized_bucket_statistics_right.party_id;
            all_results[i]
                .anonymized_bucket_statistics_left
                .start_time_utc_timestamp = all_results[0]
                .anonymized_bucket_statistics_left
                .start_time_utc_timestamp;
            all_results[i]
                .anonymized_bucket_statistics_right
                .start_time_utc_timestamp = all_results[0]
                .anonymized_bucket_statistics_right
                .start_time_utc_timestamp;
        }

        assert!(
            all_results.iter().all_equal(),
            "All parties must agree on the results"
        );
        all_results[0].clone()
    }

    fn assert_match_ids(results: &ServerJobResult) {
        for (is_match, matches_both, matches_left, matches_right, count_left, count_right) in izip!(
            &results.matches,
            &results.match_ids,
            &results.partial_match_ids_left,
            &results.partial_match_ids_right,
            &results.partial_match_counters_left,
            &results.partial_match_counters_right,
        ) {
            assert_eq!(
                *is_match,
                matches_both.is_empty().not(),
                "Matches must have some matched IDs"
            );
            assert!(
                matches_both
                    .iter()
                    .all(|id| matches_left.contains(id) && matches_right.contains(id)),
                "Matched IDs must be repeated in left and rights lists"
            );
            assert!(
                matches_left.len() <= *count_left,
                "Partial counts must be consistent"
            );
            assert!(
                matches_right.len() <= *count_right,
                "Partial counts must be consistent"
            );
        }
    }
}

#[cfg(test)]
#[cfg(feature = "db_dependent")]
mod tests_db {
    use super::*;
    use crate::{
        hawkers::aby3::aby3_store::VectorId,
        hnsw::{
            graph::{graph_store::test_utils::TestGraphPg, neighborhood::SortedEdgeIds},
            searcher::ConnectPlanLayerV,
        },
    };
    type ConnectPlanLayer = ConnectPlanLayerV<Aby3Store>;

    #[tokio::test]
    async fn test_graph_load() -> Result<()> {
        // The test data is a sequence of mutations on the graph.
        let vectors = (0..5).map(VectorId::from_0_index).collect_vec();

        let make_plans = |side| {
            let side = side as usize; // Make some difference between sides.

            vectors
                .iter()
                .enumerate()
                .map(|(i, vector)| ConnectPlan {
                    inserted_vector: *vector,
                    layers: vec![ConnectPlanLayer {
                        neighbors: SortedEdgeIds::from_ascending_vec(vec![vectors[side]]),
                        nb_links: vec![SortedEdgeIds::from_ascending_vec(vec![*vector])],
                    }],
                    set_ep: i == side,
                })
                .map(Some)
                .collect_vec()
        };

        // Populate the SQL store with test data.
        let graph_store = TestGraphPg::<Aby3Store>::new().await.unwrap();
        {
            let plans_left = make_plans(StoreId::Left);
            let plans_right = make_plans(StoreId::Right);

            let mutations = plans_left
                .into_iter()
                .zip(plans_right.into_iter())
                .map(|(left_plan, right_plan)| SingleHawkMutation {
                    plans: [left_plan, right_plan],
                    modification_key: None,
                    request_index: None,
                })
                .collect();

            let mutation = HawkMutation(mutations);
            let mut graph_tx = graph_store.tx().await?;
            mutation.persist(&mut graph_tx).await?;
            graph_tx.tx.commit().await?;
        }

        // Start an actor and load the graph from SQL to memory.
        let args = HawkArgs {
            party_index: 0,
            addresses: vec!["0.0.0.0:1234".to_string()],
            request_parallelism: 4,
            connection_parallelism: 2,
            hnsw_param_ef_constr: 320,
            hnsw_param_M: 256,
            hnsw_param_ef_search: 256,
            hnsw_prf_key: None,
            match_distances_buffer_size: 64,
            n_buckets: 10,
            disable_persistence: false,
        };
        let mut hawk_actor = HawkActor::from_cli(&args).await?;
        let (_, graph_loader) = hawk_actor.as_iris_loader().await;
        graph_loader.load_graph_store(&graph_store, 2).await?;

        // Check the loaded graph.
        for (side, graph) in izip!(STORE_IDS, &hawk_actor.graph_store) {
            let side = side as usize; // Find some difference between sides.

            let ep = graph.read().await.get_entry_point().await;
            let expected_ep = vectors[side];
            assert_eq!(ep, Some((expected_ep, 0)), "Entry point is set");

            let links = graph.read().await.get_links(&vectors[2], 0).await;
            assert_eq!(
                links.0,
                vec![expected_ep],
                "vec_2 connects to the entry point"
            );
        }

        graph_store.cleanup().await.unwrap();
        Ok(())
    }
}

#[cfg(test)]
mod hawk_mutation_tests {
    use super::*;
    use crate::hawkers::aby3::aby3_store::VectorId;
    use crate::hnsw::{graph::neighborhood::SortedEdgeIds, searcher::ConnectPlanLayerV};
    use iris_mpc_common::helpers::sync::ModificationKey;

    type ConnectPlanLayer = ConnectPlanLayerV<Aby3Store>;

    fn create_test_connect_plan(vector_id: VectorId) -> ConnectPlan {
        ConnectPlan {
            inserted_vector: vector_id,
            layers: vec![ConnectPlanLayer {
                neighbors: SortedEdgeIds::from_ascending_vec(vec![vector_id]),
                nb_links: vec![SortedEdgeIds::from_ascending_vec(vec![vector_id])],
            }],
            set_ep: false,
        }
    }

    #[test]
    fn test_get_serialized_mutation_by_key() {
        let request_id = "test-request-456".to_string();
        let modification_key = ModificationKey::RequestId(request_id.clone());

        let mutation = SingleHawkMutation {
            plans: [
                Some(create_test_connect_plan(VectorId::from_serial_id(1))),
                None,
            ],
            modification_key: Some(modification_key.clone()),
            request_index: Some(RequestIndex::UniqueReauthResetCheck(0)),
        };

        let hawk_mutation = HawkMutation(vec![mutation.clone()]);

        // Test successful serialization
        let result = hawk_mutation.get_serialized_mutation_by_key(&modification_key);
        assert!(result.is_some());

        let serialized = result.unwrap();
        assert!(!serialized.is_empty());

        // Verify we can deserialize it back
        let deserialized: SingleHawkMutation = bincode::deserialize(&serialized).unwrap();
        // Note: modification_key is skipped during serialization, so we only compare plans
        assert_eq!(deserialized.plans, mutation.plans);

        // Test failed lookup
        let wrong_key = ModificationKey::RequestId("wrong-request".to_string());
        let result = hawk_mutation.get_serialized_mutation_by_key(&wrong_key);
        assert!(result.is_none());
    }

    #[test]
    fn test_multiple_mutations_serialized_lookup() {
        let request_id1 = "request-1".to_string();
        let request_id2 = "request-2".to_string();
        let serial_id = 100u32;

        let key1 = ModificationKey::RequestId(request_id1.clone());
        let key2 = ModificationKey::RequestId(request_id2.clone());
        let key3 = ModificationKey::RequestSerialId(serial_id);

        let index1 = RequestIndex::UniqueReauthResetCheck(0);
        let index2 = RequestIndex::UniqueReauthResetCheck(1);
        let index3 = RequestIndex::ResetUpdate(0);
        let index_wrong = RequestIndex::ResetUpdate(1);

        let mutation1 = SingleHawkMutation {
            plans: [
                Some(create_test_connect_plan(VectorId::from_serial_id(1))),
                None,
            ],
            modification_key: Some(key1.clone()),
            request_index: Some(index1),
        };

        let mutation2 = SingleHawkMutation {
            plans: [
                None,
                Some(create_test_connect_plan(VectorId::from_serial_id(2))),
            ],
            modification_key: Some(key2.clone()),
            request_index: Some(index2),
        };

        let mutation3 = SingleHawkMutation {
            plans: [
                Some(create_test_connect_plan(VectorId::from_serial_id(3))),
                Some(create_test_connect_plan(VectorId::from_serial_id(3))),
            ],
            modification_key: Some(key3.clone()),
            request_index: Some(index3),
        };

        let hawk_mutation = HawkMutation(vec![
            mutation1.clone(),
            mutation2.clone(),
            mutation3.clone(),
        ]);

        // Test all serialized lookups work correctly
        assert!(hawk_mutation
            .get_serialized_mutation_by_key(&key1)
            .is_some());
        assert!(hawk_mutation
            .get_serialized_mutation_by_key(&key2)
            .is_some());
        assert!(hawk_mutation
            .get_serialized_mutation_by_key(&key3)
            .is_some());

        assert_eq!(hawk_mutation.get_by_request_index(index1), Some(&mutation1));
        assert_eq!(hawk_mutation.get_by_request_index(index2), Some(&mutation2));
        assert_eq!(hawk_mutation.get_by_request_index(index3), Some(&mutation3));

        // Test non-existent key
        let wrong_key = ModificationKey::RequestId("non-existent".to_string());
        assert!(hawk_mutation
            .get_serialized_mutation_by_key(&wrong_key)
            .is_none());

        assert!(hawk_mutation.get_by_request_index(index_wrong).is_none());
    }

    #[test]
    fn test_mutation_without_modification_key() {
        let mutation_with_key = SingleHawkMutation {
            plans: [
                Some(create_test_connect_plan(VectorId::from_serial_id(1))),
                None,
            ],
            modification_key: Some(ModificationKey::RequestId("test".to_string())),
            request_index: Some(RequestIndex::UniqueReauthResetCheck(0)),
        };

        let mutation_without_key = SingleHawkMutation {
            plans: [
                None,
                Some(create_test_connect_plan(VectorId::from_serial_id(2))),
            ],
            modification_key: None,
            request_index: None,
        };

        let hawk_mutation = HawkMutation(vec![mutation_with_key.clone(), mutation_without_key]);

        // Should find the serialized mutation with key
        let key = ModificationKey::RequestId("test".to_string());
        assert!(hawk_mutation.get_serialized_mutation_by_key(&key).is_some());

        // Should not find mutations without keys
        let other_key = ModificationKey::RequestSerialId(123);
        assert!(hawk_mutation
            .get_serialized_mutation_by_key(&other_key)
            .is_none());
    }

    #[test]
    fn test_single_hawk_mutation_serialization() {
        let mutation = SingleHawkMutation {
            plans: [
                Some(create_test_connect_plan(VectorId::from_serial_id(1))),
                None,
            ],
            modification_key: Some(ModificationKey::RequestId("test".to_string())),
            request_index: Some(RequestIndex::UniqueReauthResetCheck(0)),
        };

        // Test serialization
        let serialized = mutation.serialize().unwrap();
        assert!(!serialized.is_empty());

        // Test deserialization
        let deserialized: SingleHawkMutation = bincode::deserialize(&serialized).unwrap();

        // modification_key is skipped during serialization, so it should be None
        assert_eq!(deserialized.plans, mutation.plans);
        assert_eq!(deserialized.modification_key, None);
    }
}
