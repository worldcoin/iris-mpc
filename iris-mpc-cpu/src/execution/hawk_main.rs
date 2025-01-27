use super::player::Identity;
use crate::{
    database_generators::GaloisRingSharedIris,
    execution::{
        local::generate_local_identities,
        player::{Role, RoleAssignment},
        session::{BootSession, Session, SessionId},
    },
    hawkers::aby3_store::{Aby3Store, SharedIrisesRef},
    hnsw::{searcher::ConnectPlanV, HnswSearcher},
    network::grpc::{GrpcConfig, GrpcNetworking},
    proto_generated::party_node::party_node_server::PartyNodeServer,
    protocol::ops::setup_replicated_prf,
};
use aes_prng::AesRng;
use clap::Parser;
use eyre::Result;
use hawk_pack::{graph_store::GraphMem, hawk_searcher::FurthestQueue, VectorStore};
use itertools::{izip, Itertools};
use rand::{thread_rng, Rng, SeedableRng};
use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
    sync::Arc,
    time::Duration,
    vec,
};
use tokio::{
    sync::{mpsc, oneshot, RwLock},
    task::JoinSet,
};
use tonic::transport::Server;

#[derive(Parser)]
pub struct HawkArgs {
    #[clap(short, long)]
    party_index: usize,

    #[clap(short, long, value_delimiter = ',')]
    addresses: Vec<String>,

    #[clap(short, long, default_value_t = 2)]
    request_parallelism: usize,
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
    iris_store:  BothEyes<SharedIrisesRef>,
    graph_store: BothEyes<GraphRef>,

    // ---- My network setup ----
    networking:   GrpcNetworking,
    own_identity: Identity,
}

#[derive(Clone, Copy, Debug)]
pub enum StoreId {
    Left,
    Right,
}

type BothEyes<T> = [T; 2];

type GraphRef = Arc<RwLock<GraphMem<Aby3Store>>>;

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
    links:    Vec<FurthestQueue<V::VectorRef, V::DistanceRef>>,
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
            let socket = my_address.parse().unwrap();
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
                    player.connect_to_party(identity, &url).await?;
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
            iris_store,
            graph_store,
            role_assignments: Arc::new(role_assignments),
            consensus: Consensus::default(),
            networking,
            own_identity: my_identity.clone(),
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

        self.search_params
            .insert_apply(graph_store.deref_mut(), connect_plan.clone())
            .await;

        Ok(connect_plan)
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
    connect_plans: BothEyes<Vec<ConnectPlan>>,
    is_insertion:  Vec<bool>,
}

/// HawkHandle is a handle to the HawkActor managing concurrency.
#[derive(Clone, Debug)]
pub struct HawkHandle {
    job_queue: mpsc::Sender<HawkJob>,
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
                    connect_plans: [vec![], vec![]],
                    is_insertion,
                };

                // For both eyes.
                for (sessions, insert_plans, connect_plans) in
                    izip!(&sessions, both_insert_plans, &mut results.connect_plans)
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
        let mut sessions = vec![];
        for _ in 0..request_parallelism {
            let session = hawk_actor.new_session(store_id).await?;
            sessions.push(Arc::new(RwLock::new(session)));
        }
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
            .map(|(share, handle)| async move {
                let plans = handle
                    .submit(HawkRequest {
                        shares: [share.clone(), share], // TODO: different eyes.
                    })
                    .await?;
                Ok(plans)
            })
            .collect::<JoinSet<_>>()
            .join_all()
            .await
            .into_iter()
            .collect::<Result<Vec<HawkResult>>>()?;

        assert!(
            all_plans.iter().all_equal(),
            "All parties must agree on the graph changes"
        );

        Ok(())
    }
}
