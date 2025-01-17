use crate::{
    database_generators::generate_galois_iris_shares,
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
use rand::SeedableRng;
use std::{sync::Arc, time::Duration};
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

pub async fn hawk_main(args: HawkArgs) -> Result<()> {
    // ---- Shared setup ----

    let search_params = HnswSearcher::default();

    let identities = generate_local_identities();

    let role_assignments: RoleAssignment = identities
        .iter()
        .enumerate()
        .map(|(index, id)| (Role::new(index), id.clone()))
        .collect();

    // ---- My network setup ----

    let my_index = args.party_index;
    let my_identity = identities[my_index].clone();
    let my_address = &args.addresses[my_index];

    println!("🦅 Starting Hawk node {my_index}");

    let grpc_config = GrpcConfig {
        timeout_duration: Duration::from_secs(10),
    };

    let player = GrpcNetworking::new(my_identity.clone(), grpc_config);

    // Start server.
    {
        println!("Listening on {my_address}");
        let player = player.clone();
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
            let player = player.clone();
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

    // ---- My state ----
    // TODO: Persistence.

    let iris_store = SharedIrisesRef::default();
    let mut graph_store = GraphMem::<Aby3Store>::new();
    let mut graph_rng = AesRng::seed_from_u64(123);

    // ---- MPC session ----
    // TODO: Manage parallel sessions.

    let session_id = SessionId::from(0_u64);
    let my_session_seed = [0_u8; 16];

    player.create_session(session_id).await?;
    println!("Created {session_id:?}");

    // TODO: Wait until others ceated their side of the session.
    sleep(TEST_WAIT).await;

    let boot_session = BootSession {
        session_id,
        role_assignments: Arc::new(role_assignments.clone()),
        networking: Arc::new(player.clone()),
        own_identity: my_identity.clone(),
    };

    let prf = setup_replicated_prf(&boot_session, my_session_seed).await?;

    let session = Session {
        boot_session,
        setup: prf,
    };

    let mut aby3_store = Aby3Store {
        session,
        storage: iris_store,
        owner: my_identity,
    };
    assert_eq!(aby3_store.get_owner_index(), my_index);

    // ---- Requests ----
    // TODO: Listen for external requests.

    let n_inserts = 10;
    let iris_rng = &mut AesRng::seed_from_u64(1337);

    let my_iris_shares = IrisDB::new_random_rng(n_inserts, iris_rng)
        .db
        .into_iter()
        .map(|iris| generate_galois_iris_shares(iris_rng, iris)[my_index].clone())
        .collect_vec();

    for iris in my_iris_shares {
        let query = aby3_store.prepare_query(iris);

        search_params
            .insert(&mut aby3_store, &mut graph_store, &query, &mut graph_rng)
            .await;
    }

    println!("🎉 Inserted {n_inserts} items into the database");
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
                let args = HawkArgs::parse_from(&[
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
