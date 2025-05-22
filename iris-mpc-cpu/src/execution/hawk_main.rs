use super::player::Identity;
pub use crate::hawkers::aby3::aby3_store::VectorId;
use crate::{
    execution::{
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
    network::grpc::{GrpcConfig, GrpcHandle, GrpcNetworking},
    proto_generated::party_node::party_node_server::PartyNodeServer,
    protocol::{
        ops::{setup_replicated_prf, setup_shared_rng},
        shared_iris::GaloisRingSharedIris,
    },
};
use clap::Parser;
use eyre::Result;
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
use itertools::{izip, Itertools};
use matching::{
    Decision, Filter, MatchId,
    OnlyOrBoth::{Both, Only},
    RequestType, UniquenessRequest,
};
use rand::{thread_rng, Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use reset::{search_to_reset, ResetPlan, ResetRequests};
use scheduler::parallelize;
use search::{SearchParams, SearchQueries};
use siphasher::sip::SipHasher13;
use std::{
    collections::HashMap,
    future::Future,
    hash::{Hash, Hasher},
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    ops::Not,
    sync::Arc,
    time::{Duration, Instant},
    vec,
};
use tokio::{
    sync::{mpsc, oneshot, RwLock, RwLockWriteGuard},
    task::JoinSet,
};
use tonic::transport::Server;

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

    #[clap(short, long, default_value_t = 1)]
    pub stream_parallelism: usize,

    #[clap(long, default_value_t = 2)]
    pub connection_parallelism: usize,

    #[clap(long, default_value_t = 320)]
    pub hnsw_param_ef_constr: usize,

    #[clap(long, default_value_t = 256)]
    pub hnsw_param_M: usize,

    #[clap(long, default_value_t = 256)]
    pub hnsw_param_ef_search: usize,

    #[clap(long)]
    pub hnsw_prng_seed: Option<u64>,

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
    role_assignments: Arc<HashMap<Role, Identity>>,
    consensus: Consensus,

    // ---- My state ----
    // TODO: Persistence.
    db_size: usize,
    iris_store: BothEyes<SharedIrisesRef>,
    graph_store: BothEyes<GraphRef>,
    anonymized_bucket_statistics: BothEyes<BucketStatistics>,
    distances_cache: BothEyes<Vec<DistanceShare<u32>>>,

    // ---- My network setup ----
    networking: GrpcHandle,
    party_id: usize,
}

#[derive(Clone, Copy, Debug, Hash)]
pub enum StoreId {
    Left = 0,
    Right = 1,
}
pub const LEFT: usize = 0;
pub const RIGHT: usize = 1;
pub const STORE_IDS: BothEyes<StoreId> = [StoreId::Left, StoreId::Right];

// TODO: Merge with the same in iris-mpc-gpu.
// Orientation enum to indicate the orientation of the iris code during the batch processing.
// Normal: Normal orientation of the iris code.
// Mirror: Mirrored orientation of the iris code: Used to detect full-face mirror attacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Normal,
    Mirror,
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
    shared_rng: Box<dyn RngCore + Send + Sync>,
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

        let grpc_config = GrpcConfig {
            timeout_duration: Duration::from_secs(10),
            connection_parallelism: args.connection_parallelism,
            stream_parallelism: args.stream_parallelism,
        };

        let networking = GrpcNetworking::new(my_identity.clone(), grpc_config);
        let networking = GrpcHandle::new(networking).await?;

        // Start server.
        {
            let player = networking.clone();
            let socket = to_inaddr_any(my_address.parse().unwrap());
            tracing::info!("Starting Hawk server on {}", socket);
            tokio::spawn(async move {
                Server::builder()
                    .add_service(PartyNodeServer::new(player))
                    .serve(socket)
                    .await
                    .unwrap();
            });
        }

        // Connect to other players.
        izip!(&identities, &args.addresses)
            .filter(|(_, address)| address != &my_address)
            .map(|(identity, address)| {
                let player = networking.clone();
                let identity = identity.clone();
                let url = format!("http://{}", address);
                async move {
                    tracing::info!("Connecting to {}…", url);
                    player.connect_to_party(identity, &url).await?;
                    tracing::info!("_connected to {}!", url);
                    Ok(())
                }
            })
            .collect::<JoinSet<_>>()
            .join_all()
            .await
            .into_iter()
            .collect::<Result<()>>()?;

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
            db_size: 0,
            iris_store,
            graph_store,
            anonymized_bucket_statistics: [bucket_statistics_left, bucket_statistics_right],
            distances_cache: [vec![], vec![]],
            role_assignments: Arc::new(role_assignments),
            consensus: Consensus::default(),
            networking,
            party_id: my_index,
        })
    }

    pub fn searcher(&self) -> Arc<HnswSearcher> {
        self.searcher.clone()
    }

    fn iris_store(&self, store_id: StoreId) -> SharedIrisesRef {
        self.iris_store[store_id as usize].clone()
    }

    fn graph_store(&self, store_id: StoreId) -> GraphRef {
        self.graph_store[store_id as usize].clone()
    }

    pub async fn new_sessions(&mut self) -> Result<BothEyes<Vec<HawkSessionRef>>> {
        let (l, r) = try_join!(
            self.new_sessions_side(self.args.request_parallelism, StoreId::Left),
            self.new_sessions_side(self.args.request_parallelism, StoreId::Right),
        )?;
        tracing::debug!("Created {} MPC sessions.", self.args.request_parallelism);
        Ok([l, r])
    }

    fn new_sessions_side(
        &mut self,
        count: usize,
        store_id: StoreId,
    ) -> impl Future<Output = Result<Vec<HawkSessionRef>>> {
        let tasks = (0..count)
            .map(|_| {
                let session_id = self.consensus.next_session_id();
                self.create_session(store_id, session_id)
            })
            .collect_vec();
        parallelize(tasks.into_iter())
    }

    fn create_session(
        &self,
        store_id: StoreId,
        session_id: SessionId,
    ) -> impl Future<Output = Result<HawkSessionRef>> {
        let networking = self.networking.clone();
        let role_assignments = self.role_assignments.clone();
        let storage = self.iris_store(store_id);
        let graph_store = self.graph_store(store_id);
        let party_id = self.party_id;
        let hnsw_prng_seed = self.args.hnsw_prng_seed;

        async move {
            let grpc_session = networking.create_session(session_id).await?;

            let mut network_session = NetworkSession {
                session_id,
                role_assignments,
                networking: Box::new(grpc_session),
                own_role: Role::new(party_id),
            };

            let my_session_seed = thread_rng().gen();
            let prf = setup_replicated_prf(&mut network_session, my_session_seed).await?;

            // PRNG seed is either statically injected via configuration in TEST environments or
            // mutually derived with other MPC parties in PROD environments.
            let shared_rng: Box<dyn RngCore + Send + Sync> = if let Some(base_seed) = hnsw_prng_seed
            {
                let rng = session_seeded_rng(base_seed, store_id, session_id);
                Box::new(rng)
            } else {
                let my_rng_seed = thread_rng().gen();
                let rng = setup_shared_rng(&mut network_session, my_rng_seed).await?;
                Box::new(rng)
            };

            let hawk_session = HawkSession {
                aby3_store: Aby3Store {
                    session: Session {
                        network_session,
                        prf,
                    },
                    storage,
                },
                graph_store,
                shared_rng,
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
        session: &HawkSessionRef,
        search_results: &BothEyes<VecRequests<VecRots<InsertPlan>>>,
    ) -> Result<()> {
        let mut session = session.write().await;

        for side in [LEFT, RIGHT] {
            self.cache_distances(side, &search_results[side]);

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
            .collect::<Vec<_>>()
    }

    fn cache_distances(&mut self, side: usize, search_results: &[VecRots<InsertPlan>]) {
        let distances = search_results
            .iter() // All requests.
            .flat_map(|rots| rots.iter()) // All rotations.
            .flat_map(|plan| plan.links.first()) // Bottom layer.
            .flat_map(|neighbors| neighbors.iter()) // Nearest neighbors.
            .map(|(_, distance)| distance.clone());

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
    pub async fn load_graph_store(self, graph_store: &GraphStore) -> Result<()> {
        let mut graph_tx = graph_store.tx().await?;
        for (side, mut graph) in izip!(STORE_IDS, self.0) {
            *graph = graph_tx.with_graph(side).load_to_mem().await?;
        }
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

        assert_eq!(n_queries, batch.request_types.len());
        assert_eq!(n_queries, batch.or_rule_indices.len());
        Self {
            queries: extract_queries(Orientation::Normal),
            queries_mirror: extract_queries(Orientation::Mirror),
            batch,
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
                    skip_persistence: self.batch.skip_persistence[i],
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
            queries: Arc::new(queries),
        }
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
        self.connect_plans.0[LEFT][request_i]
            .as_ref()
            .map(|plan| plan.inserted_vector)
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
                    .unique()
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

        let match_ids = self.select_indices(Filter {
            eyes: Both,
            orient: Only(Normal),
            intra_batch: false,
        });

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

        let batch = self.batch;

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

            full_face_mirror_match_ids,
            full_face_mirror_partial_match_ids_left,
            full_face_mirror_partial_match_counters_left,
            full_face_mirror_partial_match_ids_right,
            full_face_mirror_partial_match_counters_right,
            full_face_mirror_attack_detected,

            left_iris_requests: batch.left_iris_requests,
            right_iris_requests: batch.right_iris_requests,
            deleted_ids: vec![], // TODO.
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
pub struct HawkMutation(pub BothEyes<Vec<Option<ConnectPlan>>>);

impl HawkMutation {
    pub async fn persist(self, graph_tx: &mut GraphTx<'_>) -> Result<()> {
        for (side, plans) in izip!(STORE_IDS, self.0) {
            for plan in plans.into_iter().flatten() {
                graph_tx.with_graph(side).insert_apply(plan).await?;
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
        let mut sessions = hawk_actor.new_sessions().await?;

        // Validate the common state before starting.
        try_join!(
            HawkSession::state_check(&sessions[LEFT][0]),
            HawkSession::state_check(&sessions[RIGHT][0]),
        )?;

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
        sessions: &BothEyes<Vec<HawkSessionRef>>,
        request: HawkRequest,
    ) -> Result<HawkResult> {
        tracing::info!("Processing an Hawk job…");
        let now = Instant::now();

        let do_search = async |orient| -> Result<_> {
            let search_queries = &request.queries(orient);
            let (luc_ids, request_types) = {
                let store = hawk_actor.iris_store[LEFT].read().await;
                (
                    request.luc_ids(&store),
                    request.request_types(&store, orient),
                )
            };

            let intra_results = intra_batch_is_match(sessions, search_queries).await?;

            // Search for nearest neighbors.
            // For both eyes, all requests, and rotations.
            let search_params = SearchParams {
                hnsw: hawk_actor.searcher(),
                do_match: true,
            };
            let search_results = search::search(sessions, search_queries, search_params).await?;

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
            let (search_normal, matches_normal) = do_search(Orientation::Normal).await?;
            let (_, matches_mirror) = do_search(Orientation::Mirror).await?;

            (search_normal, matches_normal.step3(matches_mirror))
        };

        hawk_actor
            .update_anon_stats(&sessions[0][0], &search_results)
            .await?;

        // Reset Updates. Find how to insert the new irises into the graph.
        let resets = search_to_reset(hawk_actor, sessions, &request).await?;

        // Insert into the in memory stores.
        let mutations =
            Self::handle_mutations(hawk_actor, sessions, search_results, &match_result, resets)
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

        Ok(results)
    }

    async fn handle_mutations(
        hawk_actor: &mut HawkActor,
        sessions: &BothEyes<Vec<HawkSessionRef>>,
        search_results: BothEyes<VecRequests<VecRots<InsertPlan>>>,
        match_result: &matching::BatchStep3,
        resets: ResetPlan,
    ) -> Result<HawkMutation> {
        use Decision::*;
        let decisions = match_result.decisions();

        // The vector IDs of reauths and resets, or None for uniqueness insertions.
        let update_ids = decisions
            .iter()
            .map(|decision| match decision {
                ReauthUpdate(update_id) => Some(*update_id),
                _ => None,
            })
            .chain(resets.vector_ids.into_iter().map(Some))
            .collect_vec();

        let mut connect_plans = HawkMutation([vec![], vec![]]);

        // For both eyes.
        for (side, sessions, search_results, reset_results) in
            izip!(&STORE_IDS, sessions, search_results, resets.search_results)
        {
            // The accepted insertions for uniqueness, reauth, and resets.
            // Focus on the insertions and keep only the centered irises.
            let insert_plans = izip!(search_results, &decisions)
                .map(|(search_result, &decision)| {
                    decision.is_mutation().then(|| search_result.into_center())
                })
                .chain(reset_results.into_iter().map(|res| Some(res.into_center())))
                .collect_vec();

            // Insert in memory, and return the plans to update the persistent database.
            connect_plans.0[*side as usize] = hawk_actor
                .insert(sessions, insert_plans, &update_ids)
                .await?;
        }
        Ok(connect_plans)
    }

    async fn health_check(
        hawk_actor: &mut HawkActor,
        sessions: &mut BothEyes<Vec<HawkSessionRef>>,
        job_failed: bool,
    ) -> Result<()> {
        if job_failed {
            // There is some error so the sessions may be somehow invalid. Make new ones.
            *sessions = hawk_actor.new_sessions().await?;
        }

        // Validate the common state after processing the requests.
        try_join!(
            HawkSession::state_check(&sessions[LEFT][0]),
            HawkSession::state_check(&sessions[RIGHT][0]),
        )?;
        Ok(())
    }
}

#[derive(Default)]
struct Consensus {
    next_session_id: u32,
}

impl Consensus {
    fn next_session_id(&mut self) -> SessionId {
        let id = SessionId(self.next_session_id);
        self.next_session_id += 1;
        id
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

        let all_results = izip!(irises, handles.clone())
            .map(|(shares, mut handle)| async move {
                // TODO: different test irises for each eye.

                let shares_right_cloned = shares.clone();
                let shares_left_cloned = shares.clone();

                let shares_right = shares_right_cloned.clone().into_iter().map(|(share, _)| share).collect();
                let shares_right_mirrored = shares_right_cloned.into_iter().map(|(_, share)| share).collect();

                let shares_left = shares_left_cloned.clone().into_iter().map(|(share, _)| share).collect();
                let shares_left_mirrored = shares_left_cloned.into_iter().map(|(_, share)| share).collect();

                let [left_iris_requests, left_iris_rotated_requests, left_iris_interpolated_requests, left_mirrored_iris_interpolated_requests] = receive_batch_shares(shares_right, shares_right_mirrored);
                let [right_iris_requests, right_iris_rotated_requests, right_iris_interpolated_requests, right_mirrored_iris_interpolated_requests] = receive_batch_shares(shares_left, shares_left_mirrored);

                let batch = BatchQuery {
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

                    // Batch details to be just copied to the result.
                    request_ids: vec!["X".to_string(); batch_size],
                    request_types: vec![UNIQUENESS_MESSAGE_TYPE.to_string(); batch_size],
                    metadata: vec![
                        BatchMetadata {
                            node_id: "X".to_string(),
                            trace_id: "X".to_string(),
                            span_id: "X".to_string(),
                        };
                        batch_size
                    ],

                    or_rule_indices: vec![vec![]; batch_size],
                    luc_lookback_records: 2,
                    skip_persistence: vec![false; batch_size],

                    ..BatchQuery::default()
                };
                handle.submit_batch_query(batch).await.await
            })
            .collect::<JoinSet<_>>()
            .join_all()
            .await
            .into_iter()
            .collect::<Result<Vec<ServerJobResult>>>()?;

        let result = assert_all_equal(all_results);

        assert_eq!(batch_size, result.merged_results.len());
        assert_eq!(result.merged_results, (0..batch_size as u32).collect_vec());
        assert_eq!(batch_size, result.request_ids.len());
        assert_eq!(batch_size, result.request_types.len());
        assert_eq!(batch_size, result.metadata.len());
        assert_eq!(batch_size, result.matches.len());
        assert_eq!(batch_size, result.matches_with_skip_persistence.len());
        assert_eq!(batch_size, result.match_ids.len());
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
        assert_eq!(batch_size, result.actor_data.0[0].len());
        assert_eq!(batch_size, result.actor_data.0[1].len());

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
            SortedNeighborhood,
        },
        shares::share::DistanceShare,
    };
    type ConnectPlanLayer = ConnectPlanLayerV<Aby3Store>;

    #[tokio::test]
    async fn test_graph_load() -> Result<()> {
        // The test data is a sequence of mutations on the graph.
        let vectors = (0..5).map(VectorId::from_0_index).collect_vec();
        let distance = DistanceShare::new(Default::default(), Default::default());

        let make_plans = |side| {
            let side = side as usize; // Make some difference between sides.

            vectors
                .iter()
                .enumerate()
                .map(|(i, vector)| ConnectPlan {
                    inserted_vector: *vector,
                    layers: vec![ConnectPlanLayer {
                        neighbors: SortedNeighborhood::from_ascending_vec(vec![(
                            vectors[side],
                            distance.clone(),
                        )]),
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
            let mutation = HawkMutation(STORE_IDS.map(make_plans));
            let mut graph_tx = graph_store.tx().await?;
            mutation.persist(&mut graph_tx).await?;
            graph_tx.tx.commit().await?;
        }

        // Start an actor and load the graph from SQL to memory.
        let args = HawkArgs {
            party_index: 0,
            addresses: vec!["0.0.0.0:1234".to_string()],
            request_parallelism: 4,
            stream_parallelism: 2,
            connection_parallelism: 2,
            hnsw_param_ef_constr: 320,
            hnsw_param_M: 256,
            hnsw_param_ef_search: 256,
            hnsw_prng_seed: None,
            match_distances_buffer_size: 64,
            n_buckets: 10,
            disable_persistence: false,
        };
        let mut hawk_actor = HawkActor::from_cli(&args).await?;
        let (_, graph_loader) = hawk_actor.as_iris_loader().await;
        graph_loader.load_graph_store(&graph_store).await?;

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
