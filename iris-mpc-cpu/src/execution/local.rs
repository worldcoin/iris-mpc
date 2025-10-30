use crate::{
    execution::{
        player::*,
        session::{NetworkSession, Session, SessionId},
    },
    network::{
        local::LocalNetworkingStore,
        tcp::testing::{interleave_vecs, setup_local_tcp_networking},
        NetworkType,
    },
    protocol::{
        ops::setup_replicated_prf,
        prf::{Prf, PrfSeed},
    },
};
use eyre::Result;
use futures::future::join_all;
use std::{
    collections::HashSet,
    sync::{Arc, LazyLock},
};
use tokio::{sync::Mutex, task::JoinHandle};

pub fn generate_local_identities() -> Vec<Identity> {
    vec![
        Identity::from("alice"),
        Identity::from("bob"),
        Identity::from("charlie"),
    ]
}

static USED_PORTS: LazyLock<Mutex<HashSet<u16>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

pub async fn get_free_local_addresses(num_ports: usize) -> Result<Vec<String>> {
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

type SessionRef = Arc<Mutex<Session>>;

#[derive(Debug)]
pub struct LocalRuntime {
    // only one session per player is created
    pub sessions: Vec<Session>,
}

impl LocalRuntime {
    pub async fn mock_setup(network_t: NetworkType) -> Result<Self> {
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

    pub async fn mock_setup_with_channel() -> Result<Self> {
        Self::mock_setup(NetworkType::Local).await
    }

    async fn new_with_network_type(
        identities: Vec<Identity>,
        seeds: Vec<PrfSeed>,
        network_type: NetworkType,
    ) -> Result<Self> {
        let role_assignments: RoleAssignment = identities
            .iter()
            .enumerate()
            .map(|(index, id)| (Role::new(index), id.clone()))
            .collect();
        let network_sessions = match network_type {
            NetworkType::Local => {
                let sess_id = SessionId::from(0_u32);
                let network = LocalNetworkingStore::from_host_ids(&identities);
                let network_sessions: Vec<NetworkSession> = (0..seeds.len())
                    .map(|i| {
                        let identity = identities[i].clone();
                        NetworkSession {
                            session_id: sess_id,
                            role_assignments: Arc::new(role_assignments.clone()),
                            networking: Box::new(network.get_local_network(identity.clone())),
                            own_role: Role::new(i),
                        }
                    })
                    .collect();
                network_sessions
            }
            NetworkType::Tcp {
                connection_parallelism,
                request_parallelism,
            } => {
                let (handles, network_sessions) = setup_local_tcp_networking(
                    identities.clone(),
                    connection_parallelism,
                    request_parallelism,
                )
                .await?;
                // the TcpNetworkHandle needs to live as long as its sessions do. the GrpcHandle
                // doesn't have to worry about this because each handle is also a gRPC server,
                // which lives in another task.
                //
                // std::mem::forget will prevent drop from being called and the memory will be reclaimed
                // when the process ends.
                let h = Arc::new(handles);
                std::mem::forget(h);

                interleave_vecs(network_sessions)
            }
        };

        let mut jobs = vec![];
        for (player_id, mut network_session) in network_sessions.into_iter().enumerate() {
            let player_seed = seeds[player_id % seeds.len()];
            let task: JoinHandle<Result<(NetworkSession, Prf)>> = tokio::spawn(async move {
                let prf = setup_replicated_prf(&mut network_session, player_seed).await?;
                Ok((network_session, prf))
            });
            jobs.push(task);
        }
        let sessions = join_all(jobs)
            .await
            .into_iter()
            .map(|t| {
                let (network_session, prf) = t??;
                Ok(Session {
                    network_session,
                    prf,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(LocalRuntime { sessions })
    }

    pub async fn new(identities: Vec<Identity>, seeds: Vec<PrfSeed>) -> Result<Self> {
        Self::new_with_network_type(identities, seeds, NetworkType::Local).await
    }

    fn into_sessions(self) -> Vec<SessionRef> {
        self.sessions
            .into_iter()
            .map(|s| Arc::new(Mutex::new(s)))
            .collect()
    }

    async fn mock_sessions(network_type: NetworkType) -> Result<Vec<SessionRef>> {
        Self::mock_setup(network_type)
            .await
            .map(|rt| rt.into_sessions())
    }

    pub async fn mock_sessions_with_channel() -> Result<Vec<SessionRef>> {
        Self::mock_sessions(NetworkType::Local).await
    }

    pub async fn mock_sessions_with_tcp(
        connection_parallelism: usize,
        request_parallelism: usize,
    ) -> Result<Vec<SessionRef>> {
        Self::mock_sessions(NetworkType::Tcp {
            connection_parallelism,
            request_parallelism,
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use tokio::task::JoinSet;

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
