use super::player::Identity;
pub use crate::hawkers::aby3::aby3_store::VectorId;
use crate::{
    execution::{
        local::generate_local_identities,
        player::{Role, RoleAssignment},
        session::{BootSession, Session, SessionId},
    },
    hawkers::aby3::aby3_store::{Aby3Store, Query, QueryRef, SharedIrisesMut, SharedIrisesRef},
    hnsw::{
        graph::{graph_store, neighborhood::SortedNeighborhoodV},
        searcher::ConnectPlanV,
        GraphMem, HnswSearcher, VectorStore,
    },
    network::grpc::{GrpcConfig, GrpcHandle, GrpcNetworking},
    proto_generated::party_node::party_node_server::PartyNodeServer,
    protocol::{ops::setup_replicated_prf, shared_iris::GaloisRingSharedIris},
};
use aes_prng::AesRng;
use clap::Parser;
use eyre::Result;
use iris_mpc_common::helpers::statistics::BucketStatistics;
use iris_mpc_common::job::Eye;
use iris_mpc_common::{
    helpers::inmemory_store::InMemoryStore,
    job::{BatchQuery, JobSubmissionHandle},
    ROTATIONS,
};
use itertools::{izip, Itertools};
use rand::{thread_rng, Rng, SeedableRng};
use std::{
    collections::HashMap,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    ops::{Deref, Not},
    sync::Arc,
    time::Duration,
    vec,
};
use tokio::{
    sync::{mpsc, oneshot, RwLock, RwLockWriteGuard},
    task::JoinSet,
};
use tonic::transport::Server;

pub type GraphStore = graph_store::GraphPg<Aby3Store>;
pub type GraphTx<'a> = graph_store::GraphTx<'a, Aby3Store>;

mod is_match_batch;
mod matching;
mod rot;
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

    // ---- My network setup ----
    networking: GrpcHandle,
    own_identity: Identity,
    party_id: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum StoreId {
    Left = 0,
    Right = 1,
}
const LEFT: usize = 0;
const RIGHT: usize = 1;
pub const STORE_IDS: BothEyes<StoreId> = [StoreId::Left, StoreId::Right];

/// BothEyes is an alias for types that apply to both left and right eyes.
pub type BothEyes<T> = [T; 2];
/// VecRequests are lists of things for each request of a batch.
type VecRequests<T> = Vec<T>;
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
    shared_rng: AesRng,
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
            GraphMem::<Aby3Store>::new(),
            [(); 2].map(|_| SharedIrisesRef::default()),
        )
        .await
    }

    pub async fn from_cli_with_graph_and_store(
        args: &HawkArgs,
        graph: GraphMem<Aby3Store>,
        iris_store: BothEyes<SharedIrisesRef>,
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

        let graph_store = [(); 2].map(|_| Arc::new(RwLock::new(graph.clone())));
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
            role_assignments: Arc::new(role_assignments),
            consensus: Consensus::default(),
            networking,
            own_identity: my_identity.clone(),
            party_id: my_index,
        })
    }

    fn iris_store(&self, store_id: StoreId) -> SharedIrisesRef {
        self.iris_store[store_id as usize].clone()
    }

    fn graph_store(&self, store_id: StoreId) -> GraphRef {
        self.graph_store[store_id as usize].clone()
    }

    pub async fn new_session(&mut self, store_id: StoreId) -> Result<HawkSession> {
        let session_id = self.consensus.next_session_id();
        self.create_session(store_id, session_id).await
    }

    async fn create_session(
        &self,
        store_id: StoreId,
        session_id: SessionId,
    ) -> Result<HawkSession> {
        // TODO: cleanup of dropped sessions.
        self.networking.create_session(session_id).await?;

        // Wait until others ceated their side of the session.
        self.networking.wait_for_session(session_id).await?;

        let boot_session = BootSession {
            session_id,
            role_assignments: self.role_assignments.clone(),
            networking: Arc::new(self.networking.clone()),
            own_identity: self.own_identity.clone(),
        };

        let my_session_seed = thread_rng().gen();
        let prf = setup_replicated_prf(&boot_session, my_session_seed).await?;

        let session = Session {
            boot_session,
            setup: prf,
        };

        let aby3_store = Aby3Store {
            session,
            storage: self.iris_store(store_id),
            owner: self.own_identity.clone(),
        };

        // TODO: Use a better seed?
        let shared_rng = AesRng::seed_from_u64(session_id.0);

        Ok(HawkSession {
            aby3_store,
            graph_store: self.graph_store(store_id),
            shared_rng,
        })
    }

    pub async fn search_both_eyes(
        &self,
        sessions: &BothEyes<Vec<HawkSessionRef>>,
        queries: &BothEyes<VecRequests<VecRots<QueryRef>>>,
    ) -> Result<BothEyes<VecRequests<VecRots<InsertPlan>>>> {
        let (left, right) = futures::join!(
            self.search_rotations(&sessions[LEFT], &queries[LEFT]),
            self.search_rotations(&sessions[RIGHT], &queries[RIGHT]),
        );
        Ok([left?, right?])
    }

    async fn search_rotations(
        &self,
        sessions: &[HawkSessionRef],
        queries: &VecRequests<VecRots<QueryRef>>,
    ) -> Result<VecRequests<VecRots<InsertPlan>>> {
        // Flatten the rotations from all requests.
        let flat_queries = VecRots::flatten(queries);
        // Do it all in parallel.
        let flat_results = self.search_parallel(sessions, flat_queries).await?;
        // Nest the results per request again.
        Ok(VecRots::unflatten(flat_results))
    }

    async fn search_parallel(
        &self,
        sessions: &[HawkSessionRef],
        queries: Vec<QueryRef>,
    ) -> Result<Vec<InsertPlan>> {
        let mut plans = vec![];

        // Distribute the requests over the given sessions in parallel.
        for chunk in queries.chunks(sessions.len()) {
            let tasks = izip!(chunk, sessions)
                .map(|(query, session)| {
                    let search_params = Arc::clone(&self.search_params);
                    let session = Arc::clone(session);
                    let query = query.clone();

                    tokio::spawn(async move {
                        let mut session = session.write().await;
                        let graph_store = Arc::clone(&session.graph_store);
                        let graph_store = graph_store.read().await;
                        Self::search_to_insert_one(
                            &search_params,
                            &graph_store,
                            &mut session,
                            query,
                        )
                        .await
                    })
                })
                .collect_vec();

            // Wait between chunks for determinism (not relying on mutex).
            for t in tasks {
                plans.push(t.await?);
            }
        }

        Ok(plans)
    }

    async fn search_to_insert_one(
        search_params: &HnswSearcher,
        graph_store: &GraphMem<Aby3Store>,
        session: &mut HawkSession,
        query: QueryRef,
    ) -> InsertPlan {
        let insertion_layer = search_params.select_layer(&mut session.shared_rng);

        let (links, set_ep) = search_params
            .search_to_insert(
                &mut session.aby3_store,
                graph_store,
                &query,
                insertion_layer,
            )
            .await;

        let match_count = search_params
            .match_count(&mut session.aby3_store, &links)
            .await;

        InsertPlan {
            query,
            links,
            match_count,
            set_ep,
        }
    }

    // TODO: Implement actual parallelism.
    pub async fn insert(
        &mut self,
        sessions: &[HawkSessionRef],
        plans: Vec<InsertPlan>,
    ) -> Result<Vec<ConnectPlan>> {
        let insert_plans = join_plans(plans);
        let mut connect_plans = vec![];
        for plan in insert_plans {
            // TODO: Parallel insertions are not supported, so only one session is needed.
            let mut session = sessions[0].write().await;
            let cp = self.insert_one(&mut session, plan).await?;
            connect_plans.push(cp);
        }
        Ok(connect_plans)
    }

    // TODO: Remove `&mut self` requirement to support parallel sessions.
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
            .await;

        graph_store.insert_apply(connect_plan.clone()).await;

        Ok(connect_plan)
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

pub struct IrisLoader<'a> {
    party_id: usize,
    db_size: &'a mut usize,
    irises: BothEyes<SharedIrisesMut<'a>>,
}

#[allow(clippy::needless_lifetimes)]
impl<'a> InMemoryStore for IrisLoader<'a> {
    fn load_single_record_from_db(
        &mut self,
        index: usize,
        left_code: &[u16],
        left_mask: &[u16],
        right_code: &[u16],
        right_mask: &[u16],
    ) {
        let vector_id = VectorId::from_serial_id(index as u32);
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
    queries: BothEyes<VecRequests<VecRots<QueryRef>>>,
}

// TODO: Unify `BatchQuery` and `HawkRequest`.
// TODO: Unify `BatchQueryEntries` and `Vec<GaloisRingSharedIris>`.
impl From<&BatchQuery> for HawkRequest {
    fn from(batch: &BatchQuery) -> Self {
        Self {
            queries: [
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
            ]
            .map(|(codes, masks, codes_proc, masks_proc)| {
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
            }),
        }
    }
}

impl HawkRequest {
    fn search_queries(&self) -> &BothEyes<VecRequests<VecRots<QueryRef>>> {
        &self.queries
    }

    fn filter_for_insertion<T>(
        &self,
        both_insert_plans: BothEyes<VecRequests<T>>,
        is_matches: &[bool],
    ) -> (VecRequests<usize>, BothEyes<VecRequests<T>>) {
        let filtered = both_insert_plans.map(|plans| {
            izip!(plans, is_matches)
                .filter_map(|(plan, &is_match)| is_match.not().then_some(plan))
                .collect_vec()
        });

        let indices = is_matches
            .iter()
            .enumerate()
            .filter_map(|(index, &is_match)| is_match.not().then_some(index))
            .collect();

        (indices, filtered)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HawkResult {
    match_results: matching::BatchStep2,
    connect_plans: HawkMutation,
    is_matches: VecRequests<bool>,
    anonymized_bucket_statistics: BothEyes<BucketStatistics>,
}

impl HawkResult {
    fn new(
        match_results: matching::BatchStep2,
        anonymized_bucket_statistics: BothEyes<BucketStatistics>,
    ) -> Self {
        let is_matches = match_results.is_matches();
        let n_requests = is_matches.len();
        HawkResult {
            match_results,
            connect_plans: HawkMutation([vec![None; n_requests], vec![None; n_requests]]),
            is_matches,
            anonymized_bucket_statistics,
        }
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
                Some(plan) => plan.inserted_vector.to_serial_id(),
                None => match_ids[idx][0],
            })
            .collect()
    }

    fn match_ids(&self) -> Vec<Vec<u32>> {
        self.match_results
            .filter_map(|(id, [l, r])| (*l && *r).then_some(id.to_serial_id()))
    }

    fn job_result(self, batch: BatchQuery) -> ServerJobResult {
        let n_requests = self.is_matches.len();

        let match_ids = self.match_ids();

        let partial_match_ids_left = self
            .match_results
            .filter_map(|(id, [l, _r])| l.then_some(id.to_serial_id()));
        let partial_match_ids_right = self
            .match_results
            .filter_map(|(id, [_l, r])| r.then_some(id.to_serial_id()));
        let partial_match_counters_left = partial_match_ids_left.iter().map(Vec::len).collect();
        let partial_match_counters_right = partial_match_ids_right.iter().map(Vec::len).collect();

        let merged_results = self.merged_results();

        ServerJobResult {
            merged_results,
            request_ids: batch.request_ids,
            request_types: batch.request_types,
            metadata: batch.metadata,
            matches: self.is_matches().to_vec(),
            matches_with_skip_persistence: self.is_matches().to_vec(), // TODO
            match_ids,
            partial_match_ids_left,
            partial_match_ids_right,
            partial_match_counters_left,
            partial_match_counters_right,
            left_iris_requests: batch.left_iris_requests,
            right_iris_requests: batch.right_iris_requests,
            deleted_ids: vec![],                                 // TODO.
            matched_batch_request_ids: vec![vec![]; n_requests], // TODO.
            anonymized_bucket_statistics_left: self.anonymized_bucket_statistics[0].clone(),
            anonymized_bucket_statistics_right: self.anonymized_bucket_statistics[1].clone(),
            successful_reauths: vec![false; n_requests], // TODO.
            reauth_target_indices: Default::default(),   // TODO.
            reauth_or_rule_used: Default::default(),     // TODO.
            modifications: batch.modifications,
            actor_data: self.connect_plans,
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
                graph_tx.with_graph(side).insert_apply(plan).await;
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
    ) -> impl std::future::Future<Output = ServerJobResult> {
        let request = HawkRequest::from(&batch);
        let result = self.submit(request).await.unwrap();

        async move { result.job_result(batch) }
    }
}

impl HawkHandle {
    pub async fn new(mut hawk_actor: HawkActor, request_parallelism: usize) -> Result<Self> {
        let sessions = [
            Self::new_sessions(&mut hawk_actor, request_parallelism, StoreId::Left).await?,
            Self::new_sessions(&mut hawk_actor, request_parallelism, StoreId::Right).await?,
        ];
        Self::new_with_sessions(hawk_actor, sessions).await
    }

    pub async fn new_with_sessions(
        mut hawk_actor: HawkActor,
        sessions: [Vec<HawkSessionRef>; 2],
    ) -> Result<Self> {
        let (tx, mut rx) = mpsc::channel::<HawkJob>(1);

        // ---- Request Handler ----
        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                tracing::info!("Processing an Hawk job…");

                let search_queries: &BothEyes<VecRequests<VecRots<QueryRef>>> =
                    job.request.search_queries();

                // Search for nearest neighbors.
                // For both eyes, all requests, and rotations.
                let search_results: BothEyes<VecRequests<VecRots<InsertPlan>>> = hawk_actor
                    .search_both_eyes(&sessions, search_queries)
                    .await
                    .unwrap();

                let match_result = {
                    let step1 = matching::BatchStep1::new(&search_results);
                    // Go fetch the missing vector IDs and calculate their is_match.
                    let missing_is_match = calculate_missing_is_match(
                        search_queries,
                        step1.missing_vector_ids(),
                        &sessions,
                    )
                    .await;
                    step1.step2(&missing_is_match)
                };

                let mut results = HawkResult::new(
                    match_result,
                    hawk_actor.anonymized_bucket_statistics.clone(),
                );

                let (insert_indices, search_results) = job
                    .request
                    .filter_for_insertion(search_results, results.is_matches());

                // Insert into the database.
                if !hawk_actor.args.disable_persistence {
                    // For both eyes.
                    for (side, sessions, search_results) in
                        izip!(&STORE_IDS, &sessions, search_results)
                    {
                        // Focus on the main results (forget rotations).
                        let insert_plans = search_results
                            .into_iter()
                            .map(VecRots::into_center)
                            .collect();

                        // Insert in memory, and return the plans to update the persistent database.
                        let plans = hawk_actor.insert(sessions, insert_plans).await.unwrap();

                        // Convert to Vec<Option> matching the request order.
                        for (i, plan) in izip!(&insert_indices, plans) {
                            results.set_connect_plan(*i, *side, plan);
                        }
                    }
                }

                let _ = job.return_channel.send(Ok(results));
            }
        });

        Ok(Self { job_queue: tx })
    }

    pub async fn new_sessions(
        hawk_actor: &mut HawkActor,
        request_parallelism: usize,
        store_id: StoreId,
    ) -> Result<Vec<HawkSessionRef>> {
        tracing::debug!("Creating {} MPC sessions…", request_parallelism);
        let mut sessions = vec![];
        for _ in 0..request_parallelism {
            let session = hawk_actor.new_session(store_id).await?;
            sessions.push(Arc::new(RwLock::new(session)));
        }
        tracing::debug!("…created {} MPC sessions.", request_parallelism);
        Ok(sessions)
    }

    pub async fn submit(&self, request: HawkRequest) -> Result<HawkResult> {
        let (tx, rx) = oneshot::channel();
        let job = HawkJob {
            request,
            return_channel: tx,
        };
        self.job_queue.send(job).await?;
        rx.await?
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
    HawkHandle::new(hawk_actor, args.request_parallelism).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        execution::local::get_free_local_addresses, protocol::shared_iris::GaloisRingSharedIris,
    };
    use futures::future::JoinAll;
    use iris_mpc_common::{
        galois_engine::degree4::preprocess_iris_message_shares,
        helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE,
        iris_db::db::IrisDB,
        job::{BatchMetadata, IrisQueryBatchEntries},
    };
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
                let [left_iris_requests, left_iris_rotated_requests, left_iris_interpolated_requests] = receive_batch_shares(shares);
                let [right_iris_requests, right_iris_rotated_requests, right_iris_interpolated_requests] = receive_batch_shares(shares_right);

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
                    ..BatchQuery::default()
                };
                let res = handle.submit_batch_query(batch).await.await;
                Ok(res)
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
    fn receive_batch_shares(shares: Vec<GaloisRingSharedIris>) -> [IrisQueryBatchEntries; 3] {
        let mut out = [(); 3].map(|_| IrisQueryBatchEntries::default());
        for share in shares {
            let one = preprocess_iris_message_shares(share.code, share.mask).unwrap();
            out[0].code.push(one.code);
            out[0].mask.push(one.mask);
            out[1].code.extend(one.code_rotated);
            out[1].mask.extend(one.mask_rotated);
            out[2].code.extend(one.code_interpolated);
            out[2].mask.extend(one.mask_interpolated);
        }
        out
    }

    fn assert_all_equal(mut all_results: Vec<ServerJobResult>) -> ServerJobResult {
        // Ignore the actual secret shares because they are different for each party.
        for i in 1..all_results.len() {
            all_results[i].left_iris_requests = all_results[0].left_iris_requests.clone();
            all_results[i].right_iris_requests = all_results[0].right_iris_requests.clone();
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
            graph::graph_store::test_utils::TestGraphPg, searcher::ConnectPlanLayerV,
            SortedNeighborhood,
        },
        shares::share::DistanceShare,
    };
    type ConnectPlanLayer = ConnectPlanLayerV<Aby3Store>;

    #[tokio::test]
    async fn test_graph_load() -> Result<()> {
        // The test data is a sequence of mutations on the graph.
        let vectors = (0..5).map(VectorId::from).collect_vec();
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
                        nb_links: vec![SortedNeighborhood::from_ascending_vec(vec![(
                            *vector,
                            distance.clone(),
                        )])],
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
                links.deref(),
                &[(expected_ep, distance.clone())],
                "vec_2 connects to the entry point"
            );
        }

        graph_store.cleanup().await.unwrap();
        Ok(())
    }
}
