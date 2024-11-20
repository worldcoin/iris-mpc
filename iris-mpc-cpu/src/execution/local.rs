use crate::{
    execution::{
        player::*,
        session::{BootSession, Session, SessionHandles, SessionId},
    },
    network::{grpc::setup_local_grpc_networking, local::LocalNetworkingStore, NetworkType},
    protocol::{ops::setup_replicated_prf, prf::PrfSeed},
};
use std::{collections::HashMap, sync::Arc};
use tokio::task::JoinSet;

pub fn generate_local_identities() -> Vec<Identity> {
    vec![
        Identity::from("alice"),
        Identity::from("bob"),
        Identity::from("charlie"),
    ]
}

fn get_free_port() -> eyre::Result<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?.port())
}

pub fn get_free_local_addresses(num_ports: usize) -> eyre::Result<Vec<String>> {
    let free_ports = (0..num_ports)
        .map(|_| get_free_port())
        .collect::<eyre::Result<Vec<u16>>>()?;
    Ok(free_ports.iter().map(|p| format!("[::1]:{p}")).collect())
}

#[derive(Debug, Clone)]
pub struct LocalRuntime {
    pub identities:       Vec<Identity>,
    pub role_assignments: RoleAssignment,
    pub seeds:            Vec<PrfSeed>,
    // only one session per player is created
    pub sessions:         HashMap<Identity, Session>,
}

impl LocalRuntime {
    pub async fn mock_setup(network_t: NetworkType) -> eyre::Result<Self> {
        let num_parties = 3;
        let identities = generate_local_identities();
        let mut seeds = Vec::new();
        for i in 0..num_parties {
            let mut seed = [0_u8; 16];
            seed[0] = i;
            seeds.push(seed);
        }
        LocalRuntime::new_with_network_type(identities, seeds, network_t).await
    }

    pub async fn mock_setup_with_channel() -> eyre::Result<Self> {
        Self::mock_setup(NetworkType::LocalChannel).await
    }

    pub async fn new_with_network_type(
        identities: Vec<Identity>,
        seeds: Vec<PrfSeed>,
        network_type: NetworkType,
    ) -> eyre::Result<Self> {
        let role_assignments: RoleAssignment = identities
            .iter()
            .enumerate()
            .map(|(index, id)| (Role::new(index), id.clone()))
            .collect();
        let sess_id = SessionId::from(0_u64);
        let boot_sessions = match network_type {
            NetworkType::LocalChannel => {
                let network = LocalNetworkingStore::from_host_ids(&identities);
                let boot_sessions: Vec<BootSession> = (0..seeds.len())
                    .map(|i| {
                        let identity = identities[i].clone();
                        BootSession {
                            session_id:       sess_id,
                            role_assignments: Arc::new(role_assignments.clone()),
                            networking:       Arc::new(network.get_local_network(identity.clone())),
                            own_identity:     identity,
                        }
                    })
                    .collect();
                boot_sessions
            }
            NetworkType::GrpcChannel => {
                let networks = setup_local_grpc_networking(identities.clone()).await?;
                let mut jobs = JoinSet::new();
                for player in networks.iter() {
                    let player = player.clone();
                    jobs.spawn(async move {
                        player.create_session(sess_id).await.unwrap();
                    });
                }
                jobs.join_all().await;
                let boot_sessions: Vec<BootSession> = (0..seeds.len())
                    .map(|i| {
                        let identity = identities[i].clone();
                        BootSession {
                            session_id:       sess_id,
                            role_assignments: Arc::new(role_assignments.clone()),
                            networking:       Arc::new(networks[i].clone()),
                            own_identity:     identity,
                        }
                    })
                    .collect();
                boot_sessions
            }
        };

        let mut jobs = JoinSet::new();
        for (player_id, boot_session) in boot_sessions.iter().enumerate() {
            let player_seed = seeds[player_id];
            let sess = boot_session.clone();
            jobs.spawn(async move {
                let prf = setup_replicated_prf(&sess, player_seed).await.unwrap();
                (sess, prf)
            });
        }
        let mut sessions = HashMap::new();
        while let Some(t) = jobs.join_next().await {
            let (boot_session, prf) = t.unwrap();
            sessions.insert(boot_session.own_identity(), Session {
                boot_session,
                setup: prf,
            });
        }
        Ok(LocalRuntime {
            identities,
            role_assignments,
            seeds,
            sessions,
        })
    }

    pub async fn new(identities: Vec<Identity>, seeds: Vec<PrfSeed>) -> eyre::Result<Self> {
        Self::new_with_network_type(identities, seeds, NetworkType::LocalChannel).await
    }
}
