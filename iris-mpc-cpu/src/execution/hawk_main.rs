use super::player::Identity;
use crate::{
    database_generators::{generate_galois_iris_shares, GaloisRingSharedIris},
    execution::{
        local::generate_local_identities,
        player::{Role, RoleAssignment},
        session::{BootSession, Session, SessionId},
    },
    hawkers::aby3_store::{Aby3Store, SharedIrisesRef},
    hnsw::HnswSearcher,
    network::grpc::{GrpcConfig, GrpcNetworking},
    proto_generated::party_node::party_node_server::PartyNodeServer,
    protocol::ops::setup_replicated_prf,
};
use aes_prng::AesRng;
use clap::Parser;
use eyre::Result;
use hawk_pack::graph_store::GraphMem;
use iris_mpc_common::iris_db::db::IrisDB;
use itertools::{izip, Itertools};
use rand::{thread_rng, Rng, RngCore, SeedableRng};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::{task::JoinSet, time::sleep};
use tonic::transport::Server;

const TEST_WAIT: Duration = Duration::from_secs(3);

#[derive(Parser)]
pub struct HawkArgs {
    #[clap(short, long)]
    party_index: usize,

    #[clap(short, long, value_delimiter = ',')]
    addresses: Vec<String>,
}

/// HawkActor manages the state of the HNSW database and connections to other
/// MPC nodes.
pub struct HawkActor {
    // ---- Shared setup ----
    search_params:    HnswSearcher,
    role_assignments: Arc<HashMap<Role, Identity>>,

    // ---- My state ----
    // TODO: Persistence.
    iris_store:  SharedIrisesRef,
    graph_store: GraphMem<Aby3Store>,
    graph_rng:   AesRng,

    // ---- My network setup ----
    networking:   GrpcNetworking,
    own_identity: Identity,
}

/// HawkSession is a unit of parallelism when operating on the HawkActor.
pub struct HawkSession {
    aby3_store: Aby3Store,
}

/// HawkRequest contains a batch of items to search.
pub struct HawkRequest {
    pub my_iris_shares: Vec<GaloisRingSharedIris>,
}

impl HawkActor {
    pub async fn from_cli(args: &HawkArgs) -> Result<Self> {
        let search_params = HnswSearcher::default();

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
            println!("Listening on {my_address}");
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

        // TODO: Retry until all servers are up.
        sleep(TEST_WAIT).await;

        // Connect to other players.
        izip!(&identities, &args.addresses)
            .filter(|(_, address)| address != &my_address)
            .map(|(identity, address)| {
                let player = networking.clone();
                let identity = identity.clone();
                let url = format!("http://{}", address);
                println!("Connecting to {url}");
                async move {
                    player.connect_to_party(identity, &url).await?;
                    println!("Connected to {url}");
                    Ok(())
                }
            })
            .collect::<JoinSet<_>>()
            .join_all()
            .await
            .into_iter()
            .collect::<Result<()>>()?;

        // TODO: Wait until others connected to me.
        sleep(TEST_WAIT).await;

        Ok(HawkActor {
            search_params,
            iris_store: SharedIrisesRef::default(),
            graph_store: GraphMem::<Aby3Store>::new(),
            graph_rng: AesRng::seed_from_u64(123),
            role_assignments: Arc::new(role_assignments),
            networking,
            own_identity: my_identity.clone(),
        })
    }

    pub async fn new_session(&self, session_id: SessionId) -> Result<HawkSession> {
        // TODO: cleanup of dropped sessions.
        self.networking.create_session(session_id).await?;

        // TODO: Wait until others ceated their side of the session.
        sleep(TEST_WAIT).await;

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
            storage: self.iris_store.clone(),
            owner: self.own_identity.clone(),
        };

        Ok(HawkSession { aby3_store })
    }

    // TODO: Remove `&mut self` requirement to support parallel sessions.
    pub async fn request(&mut self, session: &mut HawkSession, req: HawkRequest) -> Result<()> {
        for iris in req.my_iris_shares {
            let query = session.aby3_store.prepare_query(iris);

            self.search_params
                .insert(
                    &mut session.aby3_store,
                    &mut self.graph_store,
                    &query,
                    &mut self.graph_rng,
                )
                .await;
        }

        Ok(())
    }
}

pub async fn hawk_main(args: HawkArgs) -> Result<()> {
    println!("🦅 Starting Hawk node {}", args.party_index);
    let mut hawk_actor = HawkActor::from_cli(&args).await?;

    // ---- Requests ----
    // TODO: Listen for external requests.

    let parallelism = 2;
    let n_batches = 3;
    let batch_size = 5;
    let iris_rng = &mut AesRng::seed_from_u64(1337);

    let mut sessions = vec![];
    for _ in 0..parallelism {
        let session_id = SessionId::from(iris_rng.next_u64());
        sessions.push(hawk_actor.new_session(session_id).await?);
    }

    for i in 0..n_batches {
        let my_iris_shares = IrisDB::new_random_rng(batch_size, iris_rng)
            .db
            .into_iter()
            .map(|iris| generate_galois_iris_shares(iris_rng, iris)[args.party_index].clone())
            .collect_vec();
        let req = HawkRequest { my_iris_shares };

        hawk_actor
            .request(&mut sessions[i % parallelism], req)
            .await?;

        println!("🎉 Inserted {batch_size} items into the database");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::local::get_free_local_addresses;

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

        (0..n_parties)
            .map(|i| go(addresses.clone(), i))
            .collect::<JoinSet<_>>()
            .join_all()
            .await
            .into_iter()
            .collect::<Result<()>>()
    }
}
