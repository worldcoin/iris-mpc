use super::player::Identity;
use crate::{
    database_generators::GaloisRingSharedIris,
    execution::{
        local::generate_local_identities,
        player::{Role, RoleAssignment},
        session::{BootSession, Session, SessionId},
    },
    hawkers::aby3_store::{Aby3Store, SharedIrisesMut, SharedIrisesRef},
    hnsw::{
        graph::{graph_store::GraphPg, neighborhood::SortedNeighborhoodV},
        searcher::ConnectPlanV,
        GraphMem, HnswSearcher, VectorStore,
    },
    network::grpc::{GrpcConfig, GrpcNetworking},
    proto_generated::party_node::party_node_server::PartyNodeServer,
    protocol::ops::setup_replicated_prf,
};
use aes_prng::AesRng;
use clap::Parser;
use eyre::Result;
use iris_mpc_common::{
    helpers::inmemory_store::InMemoryStore,
    job::{BatchQuery, JobSubmissionHandle},
};
use itertools::{izip, Itertools};
use rand::{thread_rng, Rng, SeedableRng};
use std::{
    collections::HashMap,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    ops::Deref,
    sync::Arc,
    time::Duration,
    vec,
};
use tokio::{
    sync::{mpsc, oneshot, RwLock, RwLockWriteGuard},
    task::JoinSet,
};
use tonic::transport::Server;
pub type GraphStore = GraphPg<Aby3Store>;

#[derive(Parser)]
pub struct HawkArgs {
    #[clap(short, long)]
    pub party_index: usize,

    #[clap(short, long, value_delimiter = ',')]
    pub addresses: Vec<String>,

    #[clap(short, long, default_value_t = 2)]
    pub request_parallelism: usize,
}

/// HawkActor manages the state of the HNSW database and connections to other
/// MPC nodes.
pub struct HawkActor {
    // ---- Shared setup ----
    search_params:    Arc<HnswSearcher>,
    role_assignments: Arc<HashMap<Role, Identity>>,
    consensus:        Consensus,

    // ---- My state ----
    // TODO: Persistence.
    db_size:     usize,
    iris_store:  BothEyes<SharedIrisesRef>,
    graph_store: BothEyes<GraphRef>,

    // ---- My network setup ----
    networking:   GrpcNetworking,
    own_identity: Identity,
    party_id:     usize,
}

#[derive(Clone, Copy, Debug)]
pub enum StoreId {
    Left,
    Right,
}

pub const STORE_IDS: BothEyes<StoreId> = [StoreId::Left, StoreId::Right];

pub type BothEyes<T> = [T; 2];

type GraphRef = Arc<RwLock<GraphMem<Aby3Store>>>;
pub type GraphMut<'a> = RwLockWriteGuard<'a, GraphMem<Aby3Store>>;

/// HawkSession is a unit of parallelism when operating on the HawkActor.
pub struct HawkSession {
    aby3_store:  Aby3Store,
    graph_store: GraphRef,
    shared_rng:  AesRng,
}

type HawkSessionRef = Arc<RwLock<HawkSession>>;

pub type SearchResult = (
    <Aby3Store as VectorStore>::VectorRef,
    <Aby3Store as VectorStore>::DistanceRef,
);

pub type InsertPlan = InsertPlanV<Aby3Store>;
pub type ConnectPlan = ConnectPlanV<Aby3Store>;

#[derive(Debug)]
pub struct InsertPlanV<V: VectorStore> {
    query:    V::QueryRef,
    links:    Vec<SortedNeighborhoodV<V>>,
    set_ep:   bool,
    is_match: bool,
}

impl HawkActor {
    pub async fn from_cli(args: &HawkArgs) -> Result<Self> {
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
        };

        let networking = GrpcNetworking::new(my_identity.clone(), grpc_config);

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

        let iris_store = [(); 2].map(|_| SharedIrisesRef::default());
        let graph_store = [(); 2].map(|_| Arc::new(RwLock::new(GraphMem::<Aby3Store>::new())));

        Ok(HawkActor {
            search_params,
            db_size: 0,
            iris_store,
            graph_store,
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
        self.networking.wait_for_session(session_id).await;

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

    pub async fn search(
        &self,
        sessions: &[HawkSessionRef],
        iris_shares: Vec<GaloisRingSharedIris>,
    ) -> Result<Vec<SearchResult>> {
        Ok(self
            .search_to_insert(sessions, iris_shares)
            .await?
            .iter()
            .map(|plan| {
                plan.links
                    .first()
                    .and_then(|layer| layer.get_nearest())
                    .cloned()
            })
            .collect::<Option<Vec<SearchResult>>>()
            .unwrap_or_default())
    }

    pub async fn search_to_insert(
        &self,
        sessions: &[HawkSessionRef],
        iris_shares: Vec<GaloisRingSharedIris>,
    ) -> Result<Vec<InsertPlan>> {
        let mut plans = vec![];

        // Distribute the requests over the given sessions in parallel.
        for chunk in iris_shares.chunks(sessions.len()) {
            let tasks = izip!(chunk, sessions)
                .map(|(iris, session)| {
                    let search_params = Arc::clone(&self.search_params);
                    let session = Arc::clone(session);
                    let iris = iris.clone();

                    tokio::spawn(async move {
                        let mut session = session.write().await;
                        let graph_store = Arc::clone(&session.graph_store);
                        let graph_store = graph_store.read().await;
                        Self::search_to_insert_one(&search_params, &graph_store, &mut session, iris)
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
        iris: GaloisRingSharedIris,
    ) -> InsertPlan {
        let insertion_layer = search_params.select_layer(&mut session.shared_rng);
        let query = session.aby3_store.prepare_query(iris);

        let (links, set_ep) = search_params
            .search_to_insert(
                &mut session.aby3_store,
                graph_store,
                &query,
                insertion_layer,
            )
            .await;

        let is_match = search_params
            .is_match(&mut session.aby3_store, &links)
            .await;

        InsertPlan {
            query,
            links,
            set_ep,
            is_match,
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
        let inserted = session.aby3_store.insert(&insert_plan.query).await;
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
                db_size:  &mut self.db_size,
                irises:   [
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
    db_size:  &'a mut usize,
    irises:   BothEyes<SharedIrisesMut<'a>>,
}

impl<'a> InMemoryStore for IrisLoader<'a> {
    fn load_single_record_from_db(
        &mut self,
        index: usize,
        left_code: &[u16],
        left_mask: &[u16],
        right_code: &[u16],
        right_mask: &[u16],
    ) {
        for (side, code, mask) in izip!(&mut self.irises, [left_code, right_code], [
            left_mask, right_mask
        ]) {
            if index >= side.points.len() {
                side.points.resize(
                    index + 1,
                    GaloisRingSharedIris::default_for_party(self.party_id),
                );
            }
            side.points[index].code.coefs = code.try_into().unwrap();
            side.points[index].mask.coefs = mask.try_into().unwrap();
        }
    }

    fn increment_db_size(&mut self, _index: usize) {
        *self.db_size += 1;
    }

    fn reserve(&mut self, additional: usize) {
        for side in &mut self.irises {
            side.points.reserve(additional);
        }
    }

    fn current_db_sizes(&self) -> impl std::fmt::Debug {
        *self.db_size
    }

    fn fake_db(&mut self, size: usize) {
        *self.db_size = size;
        for side in &mut self.irises {
            side.points
                .resize(size, GaloisRingSharedIris::default_for_party(self.party_id));
        }
    }
}

pub struct GraphLoader<'a>(BothEyes<GraphMut<'a>>);

impl<'a> GraphLoader<'a> {
    pub async fn load_graph_store(self, graph_store: &GraphStore) -> Result<()> {
        for (side, mut graph) in izip!(STORE_IDS, self.0) {
            *graph = graph_store.tx(side).await?.load_to_mem().await?;
        }
        Ok(())
    }
}

struct HawkJob {
    request:        HawkRequest,
    return_channel: oneshot::Sender<Result<HawkResult>>,
}

/// HawkRequest contains a batch of items to search.
#[derive(Clone, Debug)]
pub struct HawkRequest {
    pub shares: BothEyes<Vec<GaloisRingSharedIris>>,
}

// TODO: Unify `BatchQuery` and `HawkRequest`.
// TODO: Unify `BatchQueryEntries` and `Vec<GaloisRingSharedIris>`.
impl From<&BatchQuery> for HawkRequest {
    fn from(batch: &BatchQuery) -> Self {
        Self {
            shares: [
                GaloisRingSharedIris::from_batch(batch.query_left.clone()),
                GaloisRingSharedIris::from_batch(batch.query_right.clone()),
            ],
        }
    }
}

impl HawkRequest {
    fn shares_to_search(&self) -> &BothEyes<Vec<GaloisRingSharedIris>> {
        // TODO: obtain rotated and mirrored versions.
        &self.shares
    }

    /// *AND* policy: only match, if both eyes match (like `mergeDbResults`).
    // TODO: Account for rotated and mirrored versions.
    fn is_insertion(both_insert_plans: &BothEyes<Vec<InsertPlan>>) -> Vec<bool> {
        izip!(&both_insert_plans[0], &both_insert_plans[1])
            .map(|(left, right)| !(left.is_match && right.is_match))
            .collect_vec()
    }

    fn filter_for_insertion(
        &self,
        both_insert_plans: BothEyes<Vec<InsertPlan>>,
        is_insertion: &[bool],
    ) -> BothEyes<Vec<InsertPlan>> {
        // TODO: Report the insertions versus rejections.

        both_insert_plans.map(|plans| {
            izip!(plans, is_insertion)
                .filter_map(|(plan, &is_insertion)| is_insertion.then_some(plan))
                .collect_vec()
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HawkResult {
    connect_plans: HawkMutation,
    is_insertion:  Vec<bool>,
}

impl HawkResult {
    fn matches(&self) -> Vec<bool> {
        self.is_insertion.iter().map(|&insert| !insert).collect()
    }
}

pub type ServerJobResult = iris_mpc_common::job::ServerJobResult<HawkMutation>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HawkMutation(BothEyes<Vec<ConnectPlan>>);

impl HawkMutation {
    pub async fn persist(self, graph_store: &GraphStore) -> Result<()> {
        let mut graph_tx = graph_store.tx(StoreId::Left).await?;
        for (side, plans) in izip!(STORE_IDS, self.0) {
            graph_tx = graph_tx.select_graph(side);
            for plan in plans {
                graph_tx.insert_apply(plan).await;
            }
        }
        graph_tx.tx.commit().await?;
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

        async move {
            ServerJobResult {
                merged_results: vec![], // TODO.
                request_ids: batch.request_ids,
                request_types: batch.request_types,
                metadata: batch.metadata,
                matches: result.matches(),
                match_ids: vec![],                    // TODO.
                partial_match_ids_left: vec![],       // TODO.
                partial_match_ids_right: vec![],      // TODO.
                partial_match_counters_left: vec![],  // TODO.
                partial_match_counters_right: vec![], // TODO.
                store_left: batch.store_left,
                store_right: batch.store_right,
                deleted_ids: vec![],                                    // TODO.
                matched_batch_request_ids: vec![],                      // TODO.
                anonymized_bucket_statistics_left: Default::default(),  // TODO.
                anonymized_bucket_statistics_right: Default::default(), // TODO.
                successful_reauths: vec![],                             // TODO.
                reauth_target_indices: Default::default(),              // TODO.
                reauth_or_rule_used: Default::default(),                // TODO.
                modifications: batch.modifications,
                actor_data: result.connect_plans,
            }
        }
    }
}

impl HawkHandle {
    pub async fn new(mut hawk_actor: HawkActor, request_parallelism: usize) -> Result<Self> {
        let sessions = [
            Self::new_sessions(&mut hawk_actor, request_parallelism, StoreId::Left).await?,
            Self::new_sessions(&mut hawk_actor, request_parallelism, StoreId::Right).await?,
        ];

        let (tx, mut rx) = tokio::sync::mpsc::channel::<HawkJob>(1);

        // ---- Request Handler ----
        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                tracing::debug!("Processing an Hawk job…");
                let mut both_insert_plans = [vec![], vec![]];

                // For both eyes.
                for (sessions, iris_shares, insert_plans) in izip!(
                    &sessions,
                    job.request.shares_to_search(),
                    &mut both_insert_plans
                ) {
                    // Search for nearest neighbors.
                    *insert_plans = hawk_actor
                        .search_to_insert(sessions, iris_shares.clone())
                        .await
                        .unwrap();

                    // TODO: Optimize for pure searches (rotations).
                }

                let is_insertion = HawkRequest::is_insertion(&both_insert_plans);
                let both_insert_plans = job
                    .request
                    .filter_for_insertion(both_insert_plans, &is_insertion);

                // Insert into the database.
                let mut results = HawkResult {
                    connect_plans: HawkMutation([vec![], vec![]]),
                    is_insertion,
                };

                // For both eyes.
                for (sessions, insert_plans, connect_plans) in
                    izip!(&sessions, both_insert_plans, &mut results.connect_plans.0)
                {
                    // Insert in memory, and return the plans to update the persistent database.
                    *connect_plans = hawk_actor.insert(sessions, insert_plans).await.unwrap();
                }

                println!("🎉 Inserted items into the database");

                let _ = job.return_channel.send(Ok(results));
            }
        });

        Ok(Self { job_queue: tx })
    }

    async fn new_sessions(
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
        plans.swap(0, highest);
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
        database_generators::generate_galois_iris_shares,
        execution::local::get_free_local_addresses,
    };
    use iris_mpc_common::iris_db::db::IrisDB;
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

                hawk_main(args).await
            }
        };

        let n_parties = 3;
        let addresses = get_free_local_addresses(n_parties).await?;

        let handles = (0..n_parties)
            .map(|i| go(addresses.clone(), i))
            .collect::<JoinSet<_>>()
            .join_all()
            .await
            .into_iter()
            .collect::<Result<Vec<HawkHandle>>>()?;

        // ---- Send requests ----

        let batch_size = 5;
        let iris_rng = &mut AesRng::seed_from_u64(1337);

        // Generate: iris_id -> party -> share
        let irises = IrisDB::new_random_rng(batch_size, iris_rng)
            .db
            .into_iter()
            .map(|iris| generate_galois_iris_shares(iris_rng, iris))
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

        let all_plans = izip!(irises, handles.clone())
            .map(|(share, mut handle)| async move {
                let batch = BatchQuery {
                    query_left: GaloisRingSharedIris::to_batch(&share),
                    query_right: GaloisRingSharedIris::to_batch(&share), // TODO: different eyes.
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

        assert!(
            all_plans.iter().all_equal(),
            "All parties must agree on the graph changes"
        );

        Ok(())
    }
}

#[cfg(test)]
#[cfg(feature = "db_dependent")]
mod tests_db {
    use super::*;
    use crate::{
        hawkers::aby3_store::VectorId,
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
                    layers:          vec![ConnectPlanLayer {
                        neighbors: SortedNeighborhood::from_ascending_vec(vec![(
                            vectors[side],
                            distance.clone(),
                        )]),
                        n_links:   vec![SortedNeighborhood::from_ascending_vec(vec![(
                            *vector,
                            distance.clone(),
                        )])],
                    }],
                    set_ep:          i == side,
                })
                .collect_vec()
        };

        // Populate the SQL store with test data.
        let graph_store = TestGraphPg::<Aby3Store>::new().await.unwrap();
        let mutation = HawkMutation(STORE_IDS.map(make_plans));
        mutation.persist(&graph_store).await?;

        // Start an actor and load the graph from SQL to memory.
        let args = HawkArgs {
            party_index:         0,
            addresses:           vec!["0.0.0.0:1234".to_string()],
            request_parallelism: 2,
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
