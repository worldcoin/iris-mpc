use super::player::Identity;
pub use crate::hawkers::aby3::aby3_store::VectorId;
use crate::{
    execution::{
        local::generate_local_identities,
        player::{Role, RoleAssignment},
        session::{NetworkSession, Session, SessionId},
    },
    hawkers::aby3::aby3_store::{
        Aby3Store, Query, QueryRef, SharedIrises, SharedIrisesMut, SharedIrisesRef,
    },
    hnsw::{
        graph::{graph_store, neighborhood::SortedNeighborhoodV},
        searcher::ConnectPlanV,
        GraphMem, HnswSearcher, VectorStore,
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
    smpc_request::{REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE},
    statistics::BucketStatistics,
};
use iris_mpc_common::job::Eye;
use iris_mpc_common::{
    helpers::inmemory_store::InMemoryStore,
    job::{BatchQuery, JobSubmissionHandle},
    ROTATIONS,
};
use itertools::{izip, Itertools};
use matching::{Filter, Match, MatchId};
use rand::{thread_rng, Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use scheduler::parallelize;
use siphasher::sip::SipHasher13;
use std::{
    collections::HashMap,
    future::Future,
    hash::{Hash, Hasher},
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    ops::{Deref, Not},
    slice,
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

mod intra_batch;
mod is_match_batch;
mod matching;
mod rot;
mod scheduler;
mod search;
pub mod state_check;
use crate::protocol::ops::{
    compare_threshold_buckets, open_ring, translate_threshold_a, MATCH_THRESHOLD_RATIO,
};
use crate::shares::share::DistanceShare;
use is_match_batch::calculate_missing_is_match;
use rot::VecRots;

#[derive(Clone, Parser)]
pub struct HawkArgs {
    #[clap(short, long)]
    pub party_index: usize,

    #[clap(short, long, value_delimiter = ',')]
    pub addresses: Vec<String>,

    #[clap(short, long, default_value_t = 2)]
    pub request_parallelism: usize,

    #[clap(long, default_value_t = 2)]
    pub connection_parallelism: usize,

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
    search_params: Arc<HnswSearcher>,
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
const LEFT: usize = 0;
const RIGHT: usize = 1;
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
type VecRequests<T> = Vec<T>;
type VecBuckets = Vec<u32>;
/// VecEdges are lists of things for each neighbor of a vector (graph edges).
type VecEdges<T> = Vec<T>;
/// MapEdges are maps from neighbor IDs to something.
type MapEdges<T> = HashMap<VectorId, T>;

type GraphRef = Arc<RwLock<GraphMem<Aby3Store>>>;
pub type GraphMut<'a> = RwLockWriteGuard<'a, GraphMem<Aby3Store>>;

/// HawkSession is a unit of parallelism when operating on the HawkActor.
pub struct HawkSession {
    aby3_store: Aby3Store,
    graph_store: GraphRef,
    shared_rng: Box<dyn RngCore + Send + Sync>,
}

type HawkSessionRef = Arc<RwLock<HawkSession>>;

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
        let search_params = Arc::new(HnswSearcher::default());

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
            search_params,
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
        plans: Vec<InsertPlan>,
    ) -> Result<Vec<ConnectPlan>> {
        let insert_plans = join_plans(plans);
        let mut connect_plans = vec![];
        for plan in insert_plans {
            // Parallel insertions are not supported, so only one session is needed.
            let mut session = sessions[0].write().await;
            let cp = self.insert_one(&mut session, plan).await?;
            connect_plans.push(cp);
        }
        Ok(connect_plans)
    }

    async fn insert_one(
        &mut self,
        session: &mut HawkSession,
        insert_plan: InsertPlan,
    ) -> Result<ConnectPlan> {
        let inserted = session.aby3_store.storage.insert(&insert_plan.query).await;
        let mut graph_store = session.graph_store.write().await;

        let connect_plan = self
            .search_params
            .insert_prepare(
                &mut session.aby3_store,
                graph_store.deref(),
                inserted,
                insert_plan.links,
                insert_plan.set_ep,
            )
            .await?;

        graph_store.insert_apply(connect_plan.clone()).await;

        Ok(connect_plan)
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
    queries: Arc<BothEyes<VecRequests<VecRots<QueryRef>>>>,
    queries_mirror: Arc<BothEyes<VecRequests<VecRots<QueryRef>>>>,
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
    fn queries(&self, orient: Orientation) -> Arc<BothEyes<VecRequests<VecRots<QueryRef>>>> {
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HawkResult {
    batch: BatchQuery,
    matches: VecRequests<Vec<Match>>,
    match_results: matching::BatchStep3,
    connect_plans: HawkMutation,
    is_matches: VecRequests<bool>,
    anonymized_bucket_statistics: BothEyes<BucketStatistics>,
}

impl HawkResult {
    fn new(
        batch: BatchQuery,
        match_results: matching::BatchStep3,
        anonymized_bucket_statistics: BothEyes<BucketStatistics>,
    ) -> Self {
        // Get matches from the graph.
        let is_matches = match_results.is_matches();
        let n_requests = is_matches.len();

        HawkResult {
            batch,
            matches: match_results.match_list(),
            match_results,
            connect_plans: HawkMutation([vec![None; n_requests], vec![None; n_requests]]),
            is_matches,
            anonymized_bucket_statistics,
        }
    }

    fn filter_for_insertion<T>(
        &self,
        both_insert_plans: BothEyes<VecRequests<T>>,
    ) -> (VecRequests<usize>, BothEyes<VecRequests<T>>) {
        let filtered = both_insert_plans.map(|plans| {
            izip!(plans, self.is_matches())
                .filter_map(|(plan, &is_match)| is_match.not().then_some(plan))
                .collect_vec()
        });

        let indices = self
            .is_matches()
            .iter()
            .enumerate()
            .filter_map(|(index, &is_match)| is_match.not().then_some(index))
            .collect();

        (indices, filtered)
    }

    fn set_connect_plan(&mut self, request_i: usize, side: StoreId, plan: ConnectPlan) {
        self.connect_plans.0[side as usize][request_i] = Some(plan);
    }

    fn is_matches(&self) -> &[bool] {
        &self.is_matches
    }

    fn merged_results(&self) -> Vec<u32> {
        let match_ids = self.match_ids();
        self.connect_plans.0[0]
            .iter()
            .enumerate()
            .map(|(idx, plan)| match plan {
                Some(plan) => plan.inserted_vector.index(),
                None => match_ids[idx][0],
            })
            .collect()
    }

    fn inserted_id(&self, request_i: usize) -> Option<VectorId> {
        self.connect_plans.0[LEFT][request_i]
            .as_ref()
            .map(|plan| plan.inserted_vector)
    }

    fn match_ids(&self) -> Vec<Vec<u32>> {
        let per_request = |matches: slice::Iter<Match>| {
            matches
                .filter_map(|m| match m.id {
                    MatchId::Search(id) => Some(id),
                    MatchId::Luc(id) => Some(id),
                    MatchId::IntraBatch(req_i) => self.inserted_id(req_i),
                })
                .map(|id| id.index())
                .unique()
                .collect_vec()
        };

        self.matches
            .iter()
            .map(|matches| per_request(matches.iter()))
            .collect_vec()
    }

    fn matched_batch_request_ids(&self) -> Vec<Vec<String>> {
        let per_match = |m: &Match| match m.id {
            MatchId::IntraBatch(req_i) => Some(self.batch.request_ids[req_i].clone()),
            _ => None,
        };

        self.matches
            .iter()
            .map(|matches| matches.iter().filter_map(per_match).collect_vec())
            .collect_vec()
    }

    fn select_indices(&self, filter: Filter) -> VecRequests<Vec<u32>> {
        self.match_results
            .select(filter)
            .iter()
            .map(|matches| {
                matches
                    .iter()
                    .filter_map(|&m| match m {
                        MatchId::Search(id) => Some(id),
                        MatchId::Luc(id) => Some(id),
                        MatchId::IntraBatch(req_i) => self.inserted_id(req_i),
                    })
                    .map(|id| id.index())
                    .collect_vec()
            })
            .collect_vec()
    }

    fn select_count(&self, filter: Filter) -> VecRequests<usize> {
        self.match_results
            .select(filter)
            .iter()
            .map(|matches| matches.len())
            .collect_vec()
    }

    fn job_result(self) -> ServerJobResult {
        use OnlyOrBoth::{Both, Only};
        use Orientation::{Mirror, Normal};
        use StoreId::{Left, Right};

        let n_requests = self.is_matches.len();

        let matches = self.is_matches().to_vec();
        let match_ids = self.match_ids();

        let partial_match_ids_left = self
            .match_results
            .filter_map(|(id, [l, _r])| l.then_some(id.index()));
        let partial_match_ids_right = self
            .match_results
            .filter_map(|(id, [_l, r])| r.then_some(id.index()));
        let partial_match_counters_left = partial_match_ids_left.iter().map(Vec::len).collect();
        let partial_match_counters_right = partial_match_ids_right.iter().map(Vec::len).collect();

        let merged_results = self.merged_results();
        let matched_batch_request_ids = self.matched_batch_request_ids();

        let anonymized_bucket_statistics_left = self.anonymized_bucket_statistics[LEFT].clone();
        let anonymized_bucket_statistics_right = self.anonymized_bucket_statistics[RIGHT].clone();

        ServerJobResult {
            merged_results,
            matches_with_skip_persistence: matches.clone(), // TODO
            matches,

            match_ids,
            partial_match_ids_left,
            partial_match_ids_right,

            full_face_mirror_match_ids: self.select_indices(Filter {
                eyes: Both,
                orient: Only(Mirror),
            }),
            full_face_mirror_partial_match_ids_left: self.select_indices(Filter {
                eyes: Only(Left),
                orient: Only(Mirror),
            }),
            full_face_mirror_partial_match_counters_left: self.select_count(Filter {
                eyes: Only(Left),
                orient: Only(Mirror),
            }),
            full_face_mirror_partial_match_ids_right: self.select_indices(Filter {
                eyes: Only(Right),
                orient: Only(Mirror),
            }),
            full_face_mirror_partial_match_counters_right: self.select_count(Filter {
                eyes: Only(Right),
                orient: Only(Mirror),
            }),

            partial_match_counters_left,
            partial_match_counters_right,
            left_iris_requests: self.batch.left_iris_requests,
            right_iris_requests: self.batch.right_iris_requests,
            deleted_ids: vec![], // TODO.
            matched_batch_request_ids,
            anonymized_bucket_statistics_left,
            anonymized_bucket_statistics_right,
            successful_reauths: vec![false; n_requests], // TODO.
            reauth_target_indices: Default::default(),   // TODO.
            reauth_or_rule_used: Default::default(),     // TODO.
            reset_update_indices: vec![],                // TODO.
            reset_update_request_ids: vec![],            // TODO.
            reset_update_shares: vec![],                 // TODO.

            request_ids: self.batch.request_ids,
            request_types: self.batch.request_types,
            metadata: self.batch.metadata,
            modifications: self.batch.modifications,

            actor_data: self.connect_plans,
            full_face_mirror_attack_detected: vec![false; n_requests], // TODO.
        }
    }
}

pub type ServerJobResult = iris_mpc_common::job::ServerJobResult<HawkMutation>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HawkMutation(BothEyes<Vec<Option<ConnectPlan>>>);

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

        let luc_ids = request.luc_ids(hawk_actor.iris_store[LEFT].read().await.deref());

        let mut do_search = async |orient| -> Result<_> {
            let search_queries = &request.queries(orient);

            let intra_results = intra_batch_is_match(sessions, search_queries).await?;

            // Search for nearest neighbors.
            // For both eyes, all requests, and rotations.
            let search_results: BothEyes<VecRequests<VecRots<InsertPlan>>> =
                search::search(sessions, search_queries, hawk_actor.search_params.clone()).await?;

            hawk_actor
                .update_anon_stats(&sessions[0][0], &search_results)
                .await?;

            let match_result = {
                let step1 = matching::BatchStep1::new(&search_results, &luc_ids);

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

        let mut results = HawkResult::new(
            request.batch,
            match_result,
            hawk_actor.anonymized_bucket_statistics.clone(),
        );

        let (insert_indices, search_results) = results.filter_for_insertion(search_results);

        // Insert into the database.
        if !hawk_actor.args.disable_persistence {
            // For both eyes.
            for (side, sessions, search_results) in izip!(&STORE_IDS, sessions, search_results) {
                // Focus on the main results (forget rotations).
                let insert_plans = search_results
                    .into_iter()
                    .map(VecRots::into_center)
                    .collect();

                // Insert in memory, and return the plans to update the persistent database.
                let plans = hawk_actor.insert(sessions, insert_plans).await?;

                // Convert to Vec<Option> matching the request order.
                for (i, plan) in izip!(&insert_indices, plans) {
                    results.set_connect_plan(*i, *side, plan);
                }
            }
        } else {
            tracing::info!("Persistence is disabled, not writing to DB");
        }

        metrics::histogram!("job_duration").record(now.elapsed().as_secs_f64());
        metrics::gauge!("db_size").set(hawk_actor.db_size as f64);
        let query_count = results.batch.request_ids.len();
        metrics::gauge!("search_queries_left").set(query_count as f64);
        metrics::gauge!("search_queries_right").set(query_count as f64);

        Ok(results)
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
    next_session_id: u64,
}

impl Consensus {
    fn next_session_id(&mut self) -> SessionId {
        let id = SessionId(self.next_session_id);
        self.next_session_id += 1;
        id
    }
}

/// Combine insert plans from parallel searches, repairing any conflict.
fn join_plans(mut plans: Vec<InsertPlan>) -> Vec<InsertPlan> {
    let set_ep = plans.iter().any(|plan| plan.set_ep);
    if set_ep {
        // There can be at most one new entry point.
        let highest = plans
            .iter()
            .map(|plan| plan.links.len())
            .position_max()
            .unwrap();

        for plan in &mut plans {
            plan.set_ep = false;
        }
        plans[highest].set_ep = true;
    }
    plans
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
            .map(|iris| GaloisRingSharedIris::generate_shares_locally(iris_rng, iris))
            .collect_vec();
        // Unzip: party -> iris_id -> share
        let irises = (0..n_parties)
            .map(|party_index| {
                irises
                    .iter()
                    .map(|iris| iris[party_index].clone())
                    .collect_vec()
            })
            .collect_vec();

        let all_results = izip!(irises, handles.clone())
        .map(|(shares, mut handle)| async move {
                // TODO: different test irises for each eye.
                let shares_right = shares.clone();
                let [left_iris_requests, left_iris_rotated_requests, left_iris_interpolated_requests, left_mirrored_iris_interpolated_requests] = receive_batch_shares(shares);
                let [right_iris_requests, right_iris_rotated_requests, right_iris_interpolated_requests, right_mirrored_iris_interpolated_requests] = receive_batch_shares(shares_right);

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
    fn receive_batch_shares(shares: Vec<GaloisRingSharedIris>) -> [IrisQueryBatchEntries; 4] {
        let mut out = [(); 4].map(|_| IrisQueryBatchEntries::default());
        for share in shares {
            let one = preprocess_iris_message_shares(share.code, share.mask).unwrap();
            out[0].code.push(one.code);
            out[0].mask.push(one.mask);
            out[1].code.extend(one.code_rotated);
            out[1].mask.extend(one.mask_rotated);
            out[2].code.extend(one.code_interpolated.clone());
            out[2].mask.extend(one.mask_interpolated.clone());
            // TODO: mirrored.
            out[3].code.extend(one.code_interpolated);
            out[3].mask.extend(one.mask_interpolated);
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
            connection_parallelism: 2,
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
