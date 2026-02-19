//! Implements the main execution logic for the MPC protocol on the CPU.
//!
//! This module orchestrates the multi-party computation for iris matching using an HNSW
//! (Hierarchical Navigable Small World) graph. It is responsible for managing the state
//! of the MPC node, handling network communication, processing batches of queries, and
//! interacting with the underlying cryptographic and data storage layers.
//!
//! The central component is the [`HawkActor`], which acts as a state machine for the
//! entire process. It manages connections to other MPC parties, holds the in-memory
//! HNSW graphs, and processes incoming jobs.
//!
//! Key responsibilities of this module include:
//! - **Initialization**: Setting up network connections and initializing shared cryptographic state (e.g., PRF keys).
//! - **Session Management**: Creating [`HawkSession`]s to handle parallel MPC operations.
//!   Each session encapsulates the necessary cryptographic context for a
//!   single thread of work.
//! - **Request Processing**: Handling [`HawkRequest`]s, which represent batches of requests of the
//!   usual types: Uniqueness, ResetCheck, ResetUpdate, Reauth and Deletion.
//!   . This involves:
//!     - Searching the HNSW graph for nearest neighbors of a given iris, both for matching and graph insertion purposes.
//!     - Performing secret-shared distance evaluations and comparisons using the ABY3 protocol.
//!     - Deciding whether a query results in a match, an insertion, or a re-authentication.
//! - **State Mutation**: Applying changes to the HNSW graph (inserting new nodes) and
//!   persisting these changes to the database.
//! - **Anonymized Statistics**: Collecting distance data to generate privacy-preserving
//!   statistics on match distributions.
//!
//! # Configuration Master Switches
//!
//! This module uses several compile-time constants that act as "master switches" to control
//! fundamental trade-offs between performance, accuracy, and privacy in the protocol.
//! These switches are typically configured for a specific deployment and are not expected
//! to change at runtime.
//!
//! NOTE: As of Dec 9th 2025, the choice of optimal configuration for production deployment is still being researched.
//!
//! ### 1. HNSW Entry Point Strategy
//!
//! - **`Standard`**: The HNSW search starts from a single, pre-defined entry point.
//! - **`LinearScan`**: The HNSW search begins by evaluating a set of entry points
//!   candidates and choosing the one closest to the query vector.
//!
//! The entry point strategy is determined by the `LayerMode` in the `HnswSearcher`.

//! ### 2. Distance Function and base rotations
//!
//! There are two choices of distance function:
//!
//! - **`FHD` (Fractional Hamming Distance)**: Computes the standard fractional Hamming distance.
//! - **`MinFHDX` (Minimum Fractional Hamming Distance)**: Obliviously finds the minimum
//!   FHD distance across rotation amounts `-X, -(X-1).. 0 .. (X - 1), X`.
//!
//! The choice of distance function is set by the constant `HAWK_DISTANCE_FN`.
//! One must also set the `HAWK_MINFHD_ROTATIONS` constant, which refers to the total rotations considered by MinFHD.
//! Note that one should set it to `2 * X + 1` to work with `MinFHDX`.
//! Finally, the constant `HAWK_BASE_ROTATIONS_MASK` should be set to indicate the set of "base rotations" for which
//! the HawkActor will trigger independent HNSW searches. In practice, this set must be chosen so that the searches cover (at least)
//! rotations in the [-15, 15] interval. For example, an example of suitable mask for MinFhd5 encodes the set `{-10, 0, 10}`.
//!
//! ### 3. Neighborhood Strategy: `Sorted` vs. `Unsorted`
//!
//! This strategy governs how nearest neighbors candidate lists are managed during HNSW graph traversal.
//! - **`Sorted`**: Maintains a sorted list of candidates.
//! - **`Unsorted`**: Maintains an unsorted list of candidates.
//!
//! It is set by passing `NEIGHBORHOOD_MODE` constant to the search/insertion orchestrator methods.

use crate::{
    execution::{
        hawk_main::{
            insert::InsertPlanV,
            iris_worker::IrisPoolHandle,
            rot::{VecRotationSupport, ALL_ROTATIONS_MASK, CENTER_AND_10_MASK, CENTER_ONLY_MASK},
            search::SearchIds,
        },
        session::{NetworkSession, Session, SessionId},
    },
    hawkers::{
        aby3::aby3_store::{
            Aby3DistanceRef, Aby3Query, Aby3SharedIrises, Aby3SharedIrisesRef, Aby3Store,
            Aby3VectorRef, DistanceFn,
        },
        shared_irises::SharedIrises,
    },
    hnsw::{
        graph::graph_store,
        searcher::{ConnectPlanV, LayerDistribution, NeighborhoodMode, UpdateEntryPoint},
        GraphMem, HnswSearcher, VectorStore,
    },
    network::tcp::{build_network_handle, NetworkHandle, NetworkHandleArgs},
    protocol::{
        ops::{setup_replicated_prf, setup_shared_seed},
        shared_iris::GaloisRingSharedIris,
    },
};
use ampc_actor_utils::network::config::TlsConfig;
use ampc_anon_stats::types::Eye;
use ampc_anon_stats::{
    AnonStatsContext, AnonStatsOperation, AnonStatsOrientation, AnonStatsOrigin, AnonStatsStore,
};
use clap::Parser;
use eyre::{eyre, Report, Result};
use futures::{future::try_join_all, try_join};
use intra_batch::intra_batch_is_match;
use iris_mpc_common::{
    helpers::inmemory_store::InMemoryStore,
    job::{BatchQuery, JobSubmissionHandle},
    ROTATIONS,
};
use iris_mpc_common::{
    helpers::smpc_request::{
        REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
    },
    vector_id::VectorId,
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
use session_groups::SessionGroups;
use siphasher::sip::SipHasher13;
use std::{
    collections::{BTreeMap, HashMap},
    future::Future,
    hash::{Hash, Hasher},
    ops::Not,
    sync::Arc,
    time::Instant,
    vec,
};
use tokio::{
    join,
    sync::{mpsc, oneshot, RwLock, RwLockWriteGuard},
};
use tokio_util::sync::CancellationToken;

pub type GraphStore = graph_store::GraphPg<Aby3Store>;
pub type GraphTx<'a> = graph_store::GraphTx<'a, Aby3Store>;

pub mod insert;
mod intra_batch;
pub mod iris_worker;
mod is_match_batch;
mod matching;
mod reset;
mod rot;
pub(crate) mod scheduler;
pub(crate) mod search;
mod session_groups;
pub mod state_check;
use crate::shares::share::DistanceShare;
use is_match_batch::is_match_batch;

/// Distance function used by the HawkActor
pub const HAWK_DISTANCE_FN: DistanceFn = DistanceFn::MinFhd;
/// Number of rotations considered by the MinFhd distance.
/// Not used for non-MinFhd distance, but should be set to `1`.
pub const HAWK_MINFHD_ROTATIONS: usize = 11;
/// Bitmask of base rotations, i.e rotations of the query for which
/// the HawkActor will launch independent HNSW searches.
pub const HAWK_BASE_ROTATIONS_MASK: u32 = CENTER_AND_10_MASK;

// --- Compile-time checks for HAWK_DISTANCE_FN, HAWK_MINFHD_ROTATIONS, HAWK_BASE_ROTATIONS_MASK ---
const _: () = {
    match HAWK_DISTANCE_FN {
        DistanceFn::Fhd => {
            // For Fhd the base rotations should consist of all 31 rotations.
            // HAWK_MINFHD_ROTATIONS is not actually used in this case, but it must be set to 1.
            if HAWK_MINFHD_ROTATIONS != 1 || HAWK_BASE_ROTATIONS_MASK != ALL_ROTATIONS_MASK {
                panic!();
            }
        }
        _ => match HAWK_MINFHD_ROTATIONS {
            // Variants correspond to "full" minfhd, minfhd5 and minfhd6.
            // The former requires center-only as base, while the latter two
            // require -10, 0, 10 as base rotations for searches.
            31 => {
                if HAWK_BASE_ROTATIONS_MASK != CENTER_ONLY_MASK {
                    panic!();
                }
            }
            11 | 13 => {
                if HAWK_BASE_ROTATIONS_MASK != CENTER_AND_10_MASK {
                    panic!();
                }
            }
            _ => {
                panic!();
            }
        },
    }
};

/// Rotation support as configured by SearchRotations.
pub type VecRotations<T> = VecRotationSupport<T, HAWK_BASE_ROTATIONS_MASK>;

/// The choice of HNSW candidate list strategy
pub const NEIGHBORHOOD_MODE: NeighborhoodMode = NeighborhoodMode::Sorted;

const LINEAR_SCAN_MAX_GRAPH_LAYER: usize = 1;

#[derive(Clone, Parser)]
#[allow(non_snake_case)]
pub struct HawkArgs {
    #[clap(short, long)]
    pub party_index: usize,

    #[clap(short, long, value_delimiter = ',')]
    pub addresses: Vec<String>,

    // address to connect to. allows for inserting
    // a proxy between MPC parties for testing purposes.
    #[clap(short, long, value_delimiter = ',')]
    pub outbound_addrs: Vec<String>,

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
    pub hnsw_layer_density: Option<usize>,

    #[clap(long)]
    pub hnsw_prf_key: Option<u64>,

    #[clap(long, default_value_t = false)]
    pub disable_persistence: bool,

    #[clap(flatten)]
    pub tls: Option<TlsConfig>,

    /// Enables NUMA-aware optimizations.
    #[clap(long, default_value_t = false)]
    pub numa: bool,
}

/// Manages the state and execution of the HNSW-based MPC protocol.
///
/// The `HawkActor` is the central component for the CPU-based matching engine. It orchestrates
/// MPC sessions, manages the HNSW graphs and iris data stores for both eyes, and handles
/// incoming requests. It is designed to be a long-lived object that holds
/// the entire state of the MPC node.
///
/// # Responsibilities
/// - **State Management:** Holds in-memory HNSW graphs (`graph_store`) and the underlying
///   shared iris data (`iris_store`).
/// - **Session Management:** Creates and manages `HawkSession`s, which provide the
///   cryptographic context for MPC operations.
/// - **Request Handling:** Processes `HawkRequest` batches.
/// - **Persistence:** Coordinates with `GraphPg` to load the HNSW graph from and persist updates
///   to a Postgres database.
/// - **Anonymized Statistics:** Collects and aggregates distance data to generate anonymized
///   statistics about match distributions.
pub struct HawkActor {
    /// Command-line arguments and configuration for the actor.
    args: HawkArgs,
    // ---- Shared MPC & HNSW setup ----
    /// The HNSW searcher, containing parameters and logic for graph traversal.
    searcher: Arc<HnswSearcher>,
    /// An override for the shared HNSW PRF.
    /// If it is Some(`key``), `key` is used.
    /// If it is None, the parties mutually derive the PRF key.
    /// See `get_or_init_prf_key`.
    prf_key: Option<Arc<[u8; 16]>>,
    // ---- Core State ----
    /// A size used by the start-up loader.
    loader_db_size: usize,
    /// In-memory storage for the secret-shared iris codes for both left and right eyes.
    iris_store: BothEyes<Aby3SharedIrisesRef>,
    /// In-memory HNSW graphs for both left and right eyes.
    graph_store: BothEyes<GraphRef>,
    /// Handles to the iris worker pools for NUMA-aware data processing.
    workers_handle: BothEyes<IrisPoolHandle>,

    /// Store for persisting detailed anonymized statistics.
    anon_stats_store: Option<AnonStatsStore>,

    // ---- Networking ----
    /// Handle for managing network connections and creating MPC sessions with peers.
    networking: Box<dyn NetworkHandle>,
    /// A cancellation token to signal errors and gracefully shut down network activity.
    error_ct: CancellationToken,
    /// The index of this MPC party (0, 1, or 2).
    party_id: usize,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
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

/// Orientation enum to indicate the orientation of the iris code during the batch processing.
/// Normal: Normal orientation of the iris code.
/// Mirror: Mirrored orientation of the iris code: Used to detect full-face mirror attacks.
/// TODO: Merge with the same in iris-mpc-gpu.

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
/// VecEdges are lists of things for each neighbor of a vector (graph edges).
type VecEdges<T> = Vec<T>;
/// MapEdges are maps from neighbor IDs to something.
type MapEdges<T> = HashMap<VectorId, T>;
/// If true, a match is `left OR right`, otherwise `left AND right`.
type UseOrRule = bool;

type Aby3Ref = Arc<RwLock<Aby3Store>>;

type GraphRef = Arc<RwLock<GraphMem<Aby3VectorRef>>>;
pub type GraphMut<'a> = RwLockWriteGuard<'a, GraphMem<Aby3VectorRef>>;

/// A container for state required to perform parallel MPC operations.
///
/// A `HawkSession` encapsulates the necessary context for a single thread of execution
/// within the protocol. This includes a reference to the underlying ABY3 store
/// (which manages secret-shared data and cryptographic primitives) and the HNSW graph.
/// Multiple sessions can be created to parallelize operations across a batch of requests.
///
/// All sessions operate on the same shared `graph_store` and `iris_store` held by the `HawkActor`.
/// All sessions keep a copy of the HNSW PRF key (mostly used for generating insertion layers).
#[derive(Clone)]
pub struct HawkSession {
    pub aby3_store: Aby3Ref,
    pub graph_store: GraphRef,
    pub hnsw_prf_key: Arc<[u8; 16]>,
}

pub type SearchResult = (Aby3VectorRef, <Aby3Store as VectorStore>::DistanceRef);

/// A high-level plan for inserting a query into the HNSW graph after a search.
///
/// This struct is created as a result of the search phase that precedes an insertion.
/// It bundles the low-level `InsertPlanV` with the matches found during the search.
/// The `InsertPlanV` specifies the ideal connections for the new node, while the `matches`
/// are used for further processing, such as determining if the query is a full match or needs
/// to be inserted.
#[derive(Debug, Clone)]
pub struct HawkInsertPlan {
    pub plan: InsertPlanV<Aby3Store>,
    pub matches: Vec<(Aby3VectorRef, Aby3DistanceRef)>,
}

/// A concrete plan detailing the exact modifications to connect a new node into the HNSW graph.
///
/// A `ConnectPlan` is the final output of the insertion preparation phase (`HnswSearcher::insert_prepare_batch`).
/// Unlike `InsertPlanV`, which specifies the *desired* neighbors for the new node, `ConnectPlan`
/// represents the full set of atomic graph updates required. This includes not only the new
/// node's neighbors but also the reciprocal (bilateral) connections from existing nodes back to the
/// new one. It is the definitive set of changes that will be applied to the graph storage.
pub type ConnectPlan = ConnectPlanV<Aby3Store>;

impl HawkActor {
    pub async fn from_cli(args: &HawkArgs, shutdown_ct: CancellationToken) -> Result<Self> {
        Self::from_cli_with_graph_and_store(
            args,
            shutdown_ct,
            [(); 2].map(|_| GraphMem::new()),
            [(); 2].map(|_| Aby3Store::new_storage(None)),
        )
        .await
    }

    pub async fn from_cli_with_graph_and_store(
        args: &HawkArgs,
        shutdown_ct: CancellationToken,
        graph: BothEyes<GraphMem<Aby3VectorRef>>,
        iris_store: BothEyes<Aby3SharedIrises>,
    ) -> Result<Self> {
        let searcher = {
            let mut searcher_ = HnswSearcher::new_linear_scan(
                args.hnsw_param_ef_constr,
                args.hnsw_param_ef_search,
                args.hnsw_param_M,
                LINEAR_SCAN_MAX_GRAPH_LAYER,
            );

            if let Some(layer_density) = args.hnsw_layer_density {
                searcher_.layer_distribution =
                    LayerDistribution::new_geometric_from_M(layer_density);
            } else {
                // default geometric distribution uses layer_density value of `M`
            }

            Arc::new(searcher_)
        };

        let network_args = NetworkHandleArgs {
            party_index: args.party_index,
            addresses: args.addresses.clone(),
            outbound_addresses: args.outbound_addrs.clone(),
            connection_parallelism: args.connection_parallelism,
            request_parallelism: args.request_parallelism,
            sessions_per_request: SessionGroups::N_SESSIONS_PER_REQUEST,
            tls: args.tls.clone(),
        };
        let networking = build_network_handle(network_args, shutdown_ct).await?;
        let graph_store = graph.map(GraphMem::to_arc);
        let iris_store = iris_store.map(SharedIrises::to_arc);
        let workers_handle = [LEFT, RIGHT]
            .map(|side| iris_worker::init_workers(side, iris_store[side].clone(), args.numa));

        Ok(HawkActor {
            args: args.clone(),
            searcher,
            prf_key: None,
            loader_db_size: 0,
            iris_store,
            graph_store,
            anon_stats_store: None,
            networking,
            party_id: args.party_index,
            error_ct: CancellationToken::new(),
            workers_handle,
        })
    }

    pub fn set_anon_stats_store(&mut self, store: Option<AnonStatsStore>) {
        self.anon_stats_store = store;
    }

    pub fn searcher(&self) -> Arc<HnswSearcher> {
        self.searcher.clone()
    }

    pub fn iris_store(&self, store_id: StoreId) -> Aby3SharedIrisesRef {
        self.iris_store[store_id as usize].clone()
    }

    pub fn graph_store(&self, store_id: StoreId) -> GraphRef {
        self.graph_store[store_id as usize].clone()
    }

    pub fn workers_handle(&self, store_id: StoreId) -> IrisPoolHandle {
        self.workers_handle[store_id as usize].clone()
    }

    pub async fn db_size(&self) -> usize {
        self.iris_store[LEFT].read().await.db_size()
    }

    /// Initialize the shared PRF key for HNSW graph insertion layer selection.
    ///
    /// The PRF key is either statically injected via configuration in TEST environments or
    /// mutually derived with other MPC parties in PROD environments.
    ///
    /// This PRF key is used to determine insertion heights for new elements added to the
    /// HNSW graphs, so is configured to be equal across all sessions, and is initialized every time
    /// `new_sessions` is called.
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
                    .map_err(|err| eyre!("Unable to initialize shared HNSW PRF key: {err}"))?
            };
            let prf_key = Arc::new(prf_key_);

            self.prf_key = Some(prf_key);
        }

        Ok(self.prf_key.as_ref().unwrap().clone())
    }

    async fn new_session_groups(&mut self) -> Result<SessionGroups> {
        let sessions = self.new_sessions().await?;
        Ok(SessionGroups::new(sessions))
    }

    pub async fn new_sessions_orient(&mut self) -> Result<BothOrient<BothEyes<Vec<HawkSession>>>> {
        let [mut left, mut right] = self.new_sessions().await?;

        let left_mirror = left.split_off(left.len() / 2);
        let right_mirror = right.split_off(right.len() / 2);
        Ok([[left, right], [left_mirror, right_mirror]])
    }

    pub async fn new_sessions(&mut self) -> Result<BothEyes<Vec<HawkSession>>> {
        let (mut network_sessions, ct) = self.networking.make_network_sessions().await?;
        self.error_ct = ct;
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

    pub async fn sync_peers(&mut self) -> Result<()> {
        self.networking.sync_peers().await
    }

    fn create_session(
        &self,
        store_id: StoreId,
        mut network_session: NetworkSession,
        hnsw_prf_key: &Arc<[u8; 16]>,
    ) -> impl Future<Output = Result<HawkSession>> {
        let storage = self.iris_store(store_id);
        let graph_store = self.graph_store(store_id);
        let workers = self.workers_handle(store_id);
        let hnsw_prf_key = hnsw_prf_key.clone();

        async move {
            let my_session_seed = thread_rng().gen();
            let prf = setup_replicated_prf(&mut network_session, my_session_seed).await?;
            let aby3_store = Aby3Store::new(
                storage,
                Session {
                    network_session,
                    prf,
                },
                workers,
                HAWK_DISTANCE_FN,
            );

            let hawk_session = HawkSession {
                aby3_store: Arc::new(RwLock::new(aby3_store)),
                graph_store,
                hnsw_prf_key,
            };

            Ok(hawk_session)
        }
    }

    pub async fn insert(
        &mut self,
        sessions: &[HawkSession],
        plans: VecRequests<Option<HawkInsertPlan>>,
        update_ids: &VecRequests<Option<VectorId>>,
    ) -> Result<VecRequests<Option<ConnectPlan>>> {
        // Map insertion plans to inner InsertionPlanV
        let plans = plans.into_iter().map(|p| p.map(|p| p.plan)).collect_vec();

        // Plans are to be inserted at the next version of non-None entries in `update_ids`
        let insertion_ids = update_ids
            .iter()
            .map(|id_option| id_option.map(|original_id| original_id.next_version()))
            .collect_vec();

        // Parallel insertions are not supported, so only one session is needed.
        let session = &sessions[0];
        let mut store = session.aby3_store.write().await;
        let mut graph = session.graph_store.write().await;

        insert::insert(
            &mut *store,
            &mut *graph,
            &self.searcher,
            plans,
            &insertion_ids,
        )
        .await
    }

    async fn update_anon_stats(
        &mut self,
        search_results: &BothEyes<VecRequests<VecRotations<HawkInsertPlan>>>,
    ) -> Result<()> {
        for side in [LEFT, RIGHT] {
            let sided_search_results = &search_results[side];

            tracing::info!(
                "Keeping distances for eye {side} out of {} search results",
                sided_search_results.len(),
            );

            let partial_distances = self.get_partial_distances(sided_search_results);
            self.persist_cached_distances(side, partial_distances)
                .await?;
        }
        Ok(())
    }

    fn get_partial_distances(
        &mut self,
        search_results: &[VecRotations<HawkInsertPlan>],
    ) -> BTreeMap<i64, Vec<DistanceShare<u32>>> {
        // maps query_id and db_id to a vector of distances.
        let mut distances_with_ids: BTreeMap<i64, Vec<DistanceShare<u32>>> = BTreeMap::new();
        for (query_idx, vec_rots) in search_results.iter().enumerate() {
            for insert_plan in vec_rots.iter() {
                let matches = insert_plan.matches.clone();

                for (vector_id, distance) in matches {
                    let distance_share = distance;
                    let match_id = ((query_idx as i64) << 32) | vector_id.serial_id() as i64;
                    distances_with_ids
                        .entry(match_id)
                        .or_default()
                        .push(distance_share);
                }
            }
        }
        distances_with_ids
    }

    async fn persist_cached_distances(
        &self,
        side: usize,
        partial_distances: BTreeMap<i64, Vec<DistanceShare<u32>>>,
    ) -> Result<()> {
        let Some(store) = &self.anon_stats_store else {
            return Ok(());
        };

        let bundles = partial_distances
            .iter()
            .filter_map(|(match_id, shares)| {
                if shares.is_empty() {
                    None
                } else {
                    Some((*match_id, shares.clone()))
                }
            })
            .collect::<Vec<_>>();

        if bundles.is_empty() {
            return Ok(());
        }

        let eye = match side {
            LEFT => Eye::Left,
            RIGHT => Eye::Right,
            _ => return Err(eyre!("invalid side index {side}")),
        };

        let origin = AnonStatsOrigin {
            side: Some(eye),
            orientation: AnonStatsOrientation::Normal,
            context: AnonStatsContext::HNSW,
        };

        store
            .insert_anon_stats_batch_1d_lifted(&bundles, origin, AnonStatsOperation::Uniqueness)
            .await?;

        Ok(())
    }

    /// Borrow the in-memory iris and graph stores to modify them.
    pub async fn as_iris_loader(&mut self) -> (IrisLoader<'_>, GraphLoader<'_>) {
        (
            IrisLoader {
                party_id: self.party_id,
                db_size: &mut self.loader_db_size,
                iris_pools: self.workers_handle.clone(),
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

pub type Aby3SharedIrisesMut<'a> = RwLockWriteGuard<'a, Aby3SharedIrises>;

/// Extra space to reserve in the iris store to avoid reallocations during insertion.
const IRIS_STORE_RESERVE_EXTRA: f64 = 0.2;

pub struct IrisLoader<'a> {
    party_id: usize,
    db_size: &'a mut usize,
    iris_pools: BothEyes<IrisPoolHandle>,
}

impl IrisLoader<'_> {
    pub async fn wait_completion(self) -> Result<()> {
        try_join!(
            self.iris_pools[LEFT].wait_completion(),
            self.iris_pools[RIGHT].wait_completion(),
        )?;
        Ok(())
    }
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
        for (pool, code, mask) in izip!(
            &self.iris_pools,
            [left_code, right_code],
            [left_mask, right_mask]
        ) {
            let iris = GaloisRingSharedIris::try_from_buffers(self.party_id, code, mask)
                .expect("Wrong code or mask size");
            pool.insert(vector_id, iris).unwrap();
        }
    }

    fn increment_db_size(&mut self, _index: usize) {
        *self.db_size += 1;
    }

    fn reserve(&mut self, additional: usize) {
        let additional = additional + (additional as f64 * IRIS_STORE_RESERVE_EXTRA) as usize;
        for side in &self.iris_pools {
            side.reserve(additional).unwrap();
        }
    }

    fn current_db_sizes(&self) -> impl std::fmt::Debug {
        *self.db_size
    }

    fn fake_db(&mut self, size: usize) {
        *self.db_size = size;
        let iris = Arc::new(GaloisRingSharedIris::default_for_party(self.party_id));
        for side in &self.iris_pools {
            for i in 0..size {
                side.insert(VectorId::from_serial_id(i as u32), iris.clone())
                    .unwrap();
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

/// Represents a batch of queries to be processed by the `HawkActor`.
///
/// This struct encapsulates all the data required for a single batch operation, including
/// uniqueness checks, re-authentications, and potential insertions. It is constructed
/// from a `BatchQuery` and contains the iris data for both normal and mirrored orientations.
///
/// The `queries` and `queries_mirror` fields hold secret-shared iris codes that have been
/// prepared for MPC. The `Normal` orientation is used for standard matching against the
/// database, while the `Mirror` orientation is used to detect full-face mirror attacks by
/// matching the left query iris against the right iris database and vice-versa.
#[derive(Clone, Debug)]
pub struct HawkRequest {
    /// The original `BatchQuery` containing request metadata and raw iris data.
    batch: BatchQuery,
    /// Secret-shared iris queries for normal matching (left vs. left, right vs. right).
    queries: SearchQueries<HAWK_BASE_ROTATIONS_MASK>,
    /// Secret-shared iris queries for mirror-attack detection (left vs. right, right vs. left).
    queries_mirror: SearchQueries<HAWK_BASE_ROTATIONS_MASK>,
    /// The identifiers for each request in the batch.
    ids: SearchIds,
}

// TODO: Unify `BatchQuery` and `HawkRequest`.
// TODO: Unify `BatchQueryEntries` and `Vec<GaloisRingSharedIris>`.
impl From<BatchQuery> for HawkRequest {
    #[allow(clippy::iter_skip_zero)]
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
                            .enumerate()
                            .filter(|(rot_index, _)| {
                                ((HAWK_BASE_ROTATIONS_MASK >> rot_index) & 1) > 0
                            })
                            .map(|(_, (code, mask, code_proc, mask_proc))| {
                                Aby3Query::from_processed(code, mask, code_proc, mask_proc)
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
    /// Reallocates iris data to be local to the NUMA node of the worker threads.
    ///
    /// On NUMA (Non-Uniform Memory Access) architectures, memory is divided into nodes,
    /// and accessing memory on a remote node is slower than accessing local memory.
    /// This function optimizes performance by moving the secret-shared iris data for an
    /// incoming request to the same NUMA node where the cryptographic computations will
    /// be performed.
    ///
    /// It dispatches reallocation tasks to the `IrisPoolHandle` worker pools.
    /// The workers, which are pinned to specific CPU
    /// cores, handle the memory copy, ensuring data locality for subsequent processing.
    async fn numa_realloc(self, workers: BothEyes<IrisPoolHandle>) -> Self {
        // TODO: Result<Self>
        let start = Instant::now();

        let (queries, queries_mirror) = join!(
            Self::numa_realloc_orient(self.queries, &workers),
            Self::numa_realloc_orient(self.queries_mirror, &workers)
        );

        metrics::histogram!("numa_realloc_duration").record(start.elapsed().as_secs_f64());
        Self {
            batch: self.batch,
            queries,
            queries_mirror,
            ids: self.ids,
        }
    }

    async fn numa_realloc_orient(
        queries: SearchQueries<HAWK_BASE_ROTATIONS_MASK>,
        workers: &BothEyes<IrisPoolHandle>,
    ) -> SearchQueries<HAWK_BASE_ROTATIONS_MASK> {
        let (left, right) = join!(
            Self::numa_realloc_side(&queries[LEFT], &workers[LEFT]),
            Self::numa_realloc_side(&queries[RIGHT], &workers[RIGHT])
        );
        Arc::new([left, right])
    }

    async fn numa_realloc_side(
        requests: &VecRequests<VecRotations<Aby3Query>>,
        worker: &IrisPoolHandle,
    ) -> VecRequests<VecRotations<Aby3Query>> {
        // Iterate over all the irises.
        let all_irises_iter = requests.iter().flat_map(|rots| {
            rots.iter()
                .flat_map(|query| [&query.iris, &query.iris_proc])
        });

        // Go realloc the irises in parallel.
        let tasks = all_irises_iter.map(|iris| worker.numa_realloc(iris.clone()).unwrap());

        // Iterate over the results in the same order.
        let mut new_irises_iter = try_join_all(tasks).await.unwrap().into_iter();

        // Rebuild the same structure with the new irises.
        let new_requests = requests
            .iter()
            .map(|rots| {
                rots.iter()
                    .map(|_old_query| {
                        let iris = new_irises_iter.next().unwrap();
                        let iris_proc = new_irises_iter.next().unwrap();
                        Aby3Query { iris, iris_proc }
                    })
                    .collect_vec()
                    .into()
            })
            .collect_vec();

        assert!(new_irises_iter.next().is_none());
        new_requests
    }

    fn request_types(
        &self,
        iris_store: &Aby3SharedIrises,
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

    fn queries(&self, orient: Orientation) -> SearchQueries<HAWK_BASE_ROTATIONS_MASK> {
        match orient {
            Orientation::Normal => self.queries.clone(),
            Orientation::Mirror => self.queries_mirror.clone(),
        }
    }

    fn luc_ids(&self, iris_store: &Aby3SharedIrises) -> VecRequests<Vec<VectorId>> {
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

    fn reset_updates(&self, iris_store: &Aby3SharedIrises) -> ResetRequests {
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
                    let query = Aby3Query::new_from_raw(iris);
                    VecRotationSupport::new_center_only(query)
                })
                .collect_vec()
        });
        ResetRequests {
            vector_ids: iris_store.from_0_indices(&self.batch.reset_update_indices),
            request_ids: Arc::new(self.batch.reset_update_request_ids.clone()),
            queries: Arc::new(queries),
        }
    }

    fn deletion_ids(&self, iris_store: &Aby3SharedIrises) -> Vec<VectorId> {
        iris_store.from_0_indices(&self.batch.deletion_requests_indices)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HawkResult {
    batch: BatchQuery,
    match_results: matching::BatchStep3,
    connect_plans: HawkMutation,
}

impl HawkResult {
    fn new(
        batch: BatchQuery,
        match_results: matching::BatchStep3,
        connect_plans: HawkMutation,
    ) -> Self {
        HawkResult {
            batch,
            match_results,
            connect_plans,
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

        // Anonymized bucket statistics are no longer produced by the online pipeline.

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
            skip_persistence: batch.skip_persistence,
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
        // Group updates by side: side -> (key -> neighbors)
        // Key: (serial_id, version_id, layer)
        let mut updates_by_side: BothEyes<BTreeMap<(i64, i16, i16), Vec<_>>> = Default::default();

        for mutation in self.0 {
            for (side, updates_map, plan_opt) in
                izip!(STORE_IDS, updates_by_side.iter_mut(), mutation.plans)
            {
                if let Some(plan) = plan_opt {
                    let mut graph = graph_tx.with_graph(side);
                    // Updating entry points sequentially is fine in practice
                    match plan.update_ep {
                        UpdateEntryPoint::False => {}
                        UpdateEntryPoint::SetUnique { layer } => {
                            graph.set_entry_point(plan.inserted_vector, layer).await?;
                        }
                        UpdateEntryPoint::Append { layer } => {
                            graph.add_entry_point(plan.inserted_vector, layer).await?;
                        }
                    }

                    // Buffer link updates by side
                    for ((inserted_vector, lc), neighbors) in plan.updates {
                        let key = (
                            inserted_vector.serial_id() as i64,
                            inserted_vector.version_id(),
                            lc as i16,
                        );
                        // Deduplicate: If multiple updates for the same node exist, the last one wins
                        updates_map.insert(key, neighbors);
                    }
                }
            }
        }

        // Execute one batch per side
        for (side, batch_updates) in izip!(STORE_IDS, updates_by_side) {
            if !batch_updates.is_empty() {
                graph_tx
                    .with_graph(side)
                    .batch_set_links(batch_updates)
                    .await?;
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
        let mut sessions = hawk_actor.new_session_groups().await?;

        // Validate the common state before starting.
        HawkSession::state_check(sessions.for_state_check()).await?;

        let (tx, mut rx) = mpsc::channel::<HawkJob>(1);

        // ---- Request Handler ----
        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                // check if there was a networking error
                let error_ct = hawk_actor.error_ct.clone();
                let job_result = tokio::select! {
                    r = Self::handle_job(&mut hawk_actor, &mut sessions, job.request) => r,
                    _ = error_ct.cancelled() => Err(eyre!("networking error")),
                };

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
        sessions: &mut SessionGroups,
        request: HawkRequest,
    ) -> Result<HawkResult> {
        tracing::info!("Processing an Hawk job…");
        let now = Instant::now();

        let request = request
            .numa_realloc(hawk_actor.workers_handle.clone())
            .await;

        // All deletions in a batch are applied at the beginning of batch processing
        // This is consistent with the GPU code's handling of deletions
        apply_deletions(hawk_actor, &request).await?;

        tracing::info!(
            "Processing an Hawk job with request types: {:?}, reauth targets: {:?}, skip persistence: {:?}, reauth use or rule: {:?}",
            request.batch.request_types,
            request.batch.reauth_target_indices,
            request.batch.skip_persistence,
            request.batch.reauth_use_or_rule,
        );

        // Compute search results for a given orientation and compute matching information
        let do_search = async |orient| -> Result<_> {
            let search_queries = &request.queries(orient);
            let (luc_ids, request_types) = {
                // Choice of LEFT store here is arbitrary, because it's only used for VectorId bookkeeping.
                // The two sides are in sync w.r.t stored vector ids.
                let store = hawk_actor.iris_store[LEFT].read().await;
                (
                    request.luc_ids(&store),
                    request.request_types(&store, orient),
                )
            };

            // Job that computes intra-batch matches. Note that it is awaited later, allowing it
            // to run in parallel with the HNSW searches.
            let intra_results = {
                let sessions_intra = sessions.for_intra_batch(orient);
                let search_queries = search_queries.clone();
                tokio::spawn(
                    async move { intra_batch_is_match(&sessions_intra, &search_queries).await },
                )
            };

            // Search for nearest neighbors for all requests, all rotations (if applicable) and both eyes.
            let sessions_search = &sessions.for_search(orient);
            let search_ids = &request.ids;
            let search_params = SearchParams {
                hnsw: hawk_actor.searcher(),
                do_match: true,
            };

            let search_results = search::search::<HAWK_BASE_ROTATIONS_MASK>(
                sessions_search,
                search_queries,
                search_ids,
                search_params,
                NEIGHBORHOOD_MODE,
            )
            .await?;

            // Organize results per orientation. Consult the matching module for details on organizing steps.
            let match_result = {
                let step1 = matching::BatchStep1::new(&search_results, &luc_ids, request_types);

                // Fetch the missing vector IDs for each side and calculate their is_match.
                let missing_is_match =
                    is_match_batch(search_queries, step1.missing_vector_ids(), sessions_search)
                        .await?;

                step1.step2(&missing_is_match, intra_results.await??)
            };

            Ok((search_results, match_result))
        };

        // Search for both orientations
        let (search_results, match_result) = {
            let start = Instant::now();
            let ((search_normal, matches_normal), (_, matches_mirror)) = try_join!(
                do_search(Orientation::Normal),
                do_search(Orientation::Mirror),
            )?;
            metrics::histogram!("all_search_duration").record(start.elapsed().as_secs_f64());

            // Apply final organization + decision step, using results for both orientations
            (search_normal, matches_normal.step3(matches_mirror))
        };
        let sessions_mutations = &sessions.for_mutations(Orientation::Normal);

        hawk_actor.update_anon_stats(&search_results).await?;
        tracing::info!("Updated anonymized statistics.");

        // Reset Updates. Find how to insert the new irises into the graph.
        // TODO: Parallelize with the other searches
        let resets = search_to_reset(hawk_actor, sessions_mutations, &request).await?;

        // Insert into the in memory stores.
        let mutations = Self::handle_mutations(
            hawk_actor,
            sessions_mutations,
            search_results,
            &match_result,
            resets,
            &request,
        )
        .await?;

        let results = HawkResult::new(request.batch, match_result, mutations);

        metrics::histogram!("job_duration").record(now.elapsed().as_secs_f64());
        metrics::gauge!("db_size").set(hawk_actor.db_size().await as f64);
        let query_count = results.batch.request_ids.len();
        metrics::gauge!("search_queries_left").set(query_count as f64);
        metrics::gauge!("search_queries_right").set(query_count as f64);
        tracing::info!("Finished processing a Hawk job…");
        Ok(results)
    }

    async fn handle_mutations(
        hawk_actor: &mut HawkActor,
        sessions: &BothEyes<Vec<HawkSession>>,
        search_results: BothEyes<VecRequests<VecRotations<HawkInsertPlan>>>,
        match_result: &matching::BatchStep3,
        resets: ResetPlan,
        request: &HawkRequest,
    ) -> Result<HawkMutation> {
        use Decision::*;
        let start = Instant::now();
        let decisions = match_result.decisions();
        let requests_order = &request.batch.requests_order;

        // Fetch targeted vector IDs of reauths and resets (None for uniqueness insertions).
        let update_ids = requests_order
            .iter()
            .map(|req_index| match req_index {
                RequestIndex::UniqueReauthResetCheck(i) => match decisions[*i] {
                    ReauthUpdate(update_id) => {
                        if request
                            .batch
                            .skip_persistence
                            .get(*i)
                            .copied()
                            .unwrap_or(false)
                        {
                            None
                        } else {
                            Some(update_id)
                        }
                    }
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

            // Collect the HNSW insertion plans for all mutating decisions
            let insert_plans = requests_order
                .iter()
                .map(|req_index| match req_index {
                    RequestIndex::UniqueReauthResetCheck(i) => match decisions[*i] {
                        ReauthUpdate(_)
                            if request
                                .batch
                                .skip_persistence
                                .get(*i)
                                .copied()
                                .unwrap_or(false) =>
                        {
                            None
                        }
                        _ => decisions[*i]
                            .is_mutation()
                            .then(|| search_results[*i].center().clone()),
                    },
                    RequestIndex::ResetUpdate(i) => Some(reset_results[*i].center().clone()),
                    // Deletions were handled earlier in handle_job
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

        metrics::histogram!("handle_mutations_duration").record(start.elapsed().as_secs_f64());
        Ok(HawkMutation(mutations))
    }

    async fn health_check(
        hawk_actor: &mut HawkActor,
        sessions: &mut SessionGroups,
        job_failed: bool,
    ) -> Result<()> {
        if job_failed {
            tracing::error!("job failed. recreating sessions");
            // There is some error so the sessions may be somehow invalid. Make new ones.
            *sessions = hawk_actor.new_session_groups().await?;
        }

        // Validate the common state after processing the requests.
        HawkSession::state_check(sessions.for_state_check()).await?;

        // validate that the RNGs have not diverged
        // TODO: debug serialization issues encountered with this function and then re-enable
        // HawkSession::prf_check(&sessions.for_search).await?;
        Ok(())
    }
}

pub async fn hawk_main(args: HawkArgs) -> Result<HawkHandle> {
    println!("🦅 Starting Hawk node {}", args.party_index);
    let hawk_actor = HawkActor::from_cli(&args, CancellationToken::new()).await?;
    HawkHandle::new(hawk_actor).await
}

#[cfg(test)]
pub mod test_utils;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        execution::local::get_free_local_addresses, protocol::shared_iris::GaloisRingSharedIris,
        utils::constants::N_PARTIES,
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
    use std::{ops::Not, time::Duration};
    use tokio::time::sleep;
    use tracing_test::traced_test;

    #[tokio::test]
    #[traced_test]
    async fn test_hawk_main() -> Result<()> {
        let go = |addresses: Vec<String>, index: usize| {
            async move {
                let args = HawkArgs::parse_from([
                    "hawk_main",
                    "--addresses",
                    &addresses.join(","),
                    "--outbound-addrs",
                    &addresses.join(","),
                    "--party-index",
                    &index.to_string(),
                    "--hnsw-param-ef-constr",
                    &320.to_string(),
                    "--hnsw-param-m",
                    &256.to_string(),
                ]);

                // Make the test async.
                sleep(Duration::from_millis(100 * index as u64)).await;

                hawk_main(args).await.unwrap()
            }
        };

        let addresses = get_free_local_addresses(N_PARTIES).await?;

        let handles = (0..N_PARTIES)
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
        let irises = (0..N_PARTIES)
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
        let all_results = parallelize((0..N_PARTIES).map(|party_i| {
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

        let all_results = parallelize((0..N_PARTIES).map(|party_i| {
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

        tokio::time::sleep(Duration::from_millis(1100)).await;
        Ok(())
    }

    /// Prepare shares in the same format as `receive_batch()`.
    fn receive_batch_shares(
        shares_with_mirror: &[(GaloisRingSharedIris, GaloisRingSharedIris)],
    ) -> [IrisQueryBatchEntries; 4] {
        let mut out = [(); 4].map(|_| IrisQueryBatchEntries::default());
        for (share, mirrored_share) in shares_with_mirror.iter().cloned() {
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
    pub fn batch_of_party(
        batch: &BatchQuery,
        shares_with_mirror: &[(GaloisRingSharedIris, GaloisRingSharedIris)],
    ) -> BatchQuery {
        // TODO: different test irises for each eye.

        let [left_iris_requests, left_iris_rotated_requests, left_iris_interpolated_requests, left_mirrored_iris_interpolated_requests] =
            receive_batch_shares(shares_with_mirror);
        let [right_iris_requests, right_iris_rotated_requests, right_iris_interpolated_requests, right_mirrored_iris_interpolated_requests] =
            receive_batch_shares(shares_with_mirror);

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

            assert_eq!(
                all_results[i].reset_update_shares.len(),
                all_results[0].reset_update_shares.len(),
                "All parties must agree on the reset update shares"
            );
            all_results[i].reset_update_shares = all_results[0].reset_update_shares.clone();
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
    use crate::hnsw::{
        graph::graph_store::test_utils::TestGraphPg,
        searcher::{build_layer_updates, UpdateEntryPoint},
    };

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
                    updates: build_layer_updates(
                        *vector,
                        vec![vectors[side]],
                        vec![vec![*vector]],
                        0,
                    ),
                    update_ep: if i == side {
                        UpdateEntryPoint::SetUnique { layer: 0 }
                    } else {
                        UpdateEntryPoint::False
                    },
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

        let addresses = vec![
            "0.0.0.0:1234".to_string(),
            "0.0.0.0:1235".to_string(),
            "0.0.0.0:1236".to_string(),
        ];
        // Start an actor and load the graph from SQL to memory.
        let args = HawkArgs {
            party_index: 0,
            addresses: addresses.clone(),
            outbound_addrs: addresses,
            request_parallelism: 4,
            connection_parallelism: 2,
            hnsw_param_ef_constr: 320,
            hnsw_param_M: 256,
            hnsw_param_ef_search: 256,
            hnsw_layer_density: None,
            hnsw_prf_key: None,
            numa: true,
            disable_persistence: false,
            tls: None,
        };
        let mut hawk_actor = HawkActor::from_cli(&args, CancellationToken::new()).await?;
        let (_, graph_loader) = hawk_actor.as_iris_loader().await;
        graph_loader.load_graph_store(&graph_store, 2).await?;

        // Check the loaded graph.
        for (side, graph) in izip!(STORE_IDS, &hawk_actor.graph_store) {
            let side = side as usize; // Find some difference between sides.

            let ep = graph.read().await.get_first_entry_point().await;
            let expected_ep = vectors[side];
            assert_eq!(ep, Some((expected_ep, 0)), "Entry point is set");

            let links = graph.read().await.get_links(&vectors[2], 0).await;
            assert_eq!(
                links,
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
    use crate::hnsw::searcher::{build_layer_updates, UpdateEntryPoint};
    use iris_mpc_common::helpers::sync::ModificationKey;

    fn create_test_connect_plan(vector_id: VectorId) -> ConnectPlan {
        ConnectPlan {
            inserted_vector: vector_id,
            updates: build_layer_updates(vector_id, vec![vector_id], vec![vec![vector_id]], 0),
            update_ep: UpdateEntryPoint::False,
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
