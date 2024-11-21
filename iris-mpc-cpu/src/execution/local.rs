use crate::{
    execution::{
        player::*,
        session::{BootSession, Session, SessionHandles, SessionId},
    },
    network::{grpc::setup_local_grpc_networking, local::LocalNetworkingStore, NetworkType},
    protocol::{ops::setup_replicated_prf, prf::PrfSeed},
};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, LazyLock},
};
use tokio::{sync::Mutex, task::JoinSet};

pub fn generate_local_identities() -> Vec<Identity> {
    vec![
        Identity::from("alice"),
        Identity::from("bob"),
        Identity::from("charlie"),
    ]
}

static USED_PORTS: LazyLock<Mutex<HashSet<u16>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

pub async fn get_free_local_addresses(num_ports: usize) -> eyre::Result<Vec<String>> {
    let mut addresses = vec![];
    let mut listeners = vec![];
    while addresses.len() < num_ports {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        if USED_PORTS.lock().await.insert(port) {
            addresses.push(format!("127.0.0.1:{port}"));
            listeners.push(listener);
        } else {
            tracing::warn!("Port {port} already in use, retrying");
        }
    }
    tracing::info!("Found free addresses: {addresses:?}");
    Ok(addresses)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_free_local_addresses() {
        let mut jobs = JoinSet::new();
        let num_ports = 3;

        for _ in 0..100 {
            jobs.spawn(async move {
                let mut addresses = get_free_local_addresses(num_ports).await.unwrap();
                assert_eq!(addresses.len(), num_ports);
                addresses.sort();
                addresses.dedup();
                assert_eq!(addresses.len(), num_ports);
            });
        }
        jobs.join_all().await;
    }
}
