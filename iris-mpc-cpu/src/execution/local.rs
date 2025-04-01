use crate::{
    execution::{
        player::*,
        session::{NetworkSession, Session, SessionId},
    },
    network::{grpc::setup_local_grpc_networking, local::LocalNetworkingStore, NetworkType},
    protocol::{ops::setup_replicated_prf, prf::PrfSeed},
};
use futures::future::join_all;
use std::{
    collections::HashSet,
    sync::{Arc, LazyLock},
};
use tokio::sync::Mutex;

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

type SessionRef = Arc<Mutex<Session>>;

#[derive(Debug)]
pub struct LocalRuntime {
    // only one session per player is created
    pub sessions: Vec<Session>,
}

impl LocalRuntime {
    pub(crate) async fn mock_setup(network_t: NetworkType) -> eyre::Result<Self> {
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

    pub async fn mock_setup_with_grpc() -> eyre::Result<Self> {
        Self::mock_setup(NetworkType::GrpcChannel).await
    }

    async fn new_with_network_type(
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
        let network_sessions = match network_type {
            NetworkType::LocalChannel => {
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
            NetworkType::GrpcChannel => {
                let networks = setup_local_grpc_networking(identities.clone()).await?;
                let mut jobs = vec![];
                for player in networks.iter() {
                    let player = player.clone();
                    let task =
                        tokio::spawn(async move { player.create_session(sess_id).await.unwrap() });
                    jobs.push(task);
                }
                let grpc_sessions = join_all(jobs)
                    .await
                    .into_iter()
                    .map(|r| r.map_err(eyre::Report::new))
                    .collect::<eyre::Result<Vec<_>>>()?;
                let network_sessions: Vec<NetworkSession> = grpc_sessions
                    .into_iter()
                    .enumerate()
                    .map(|(id, session)| NetworkSession {
                        session_id: sess_id,
                        role_assignments: Arc::new(role_assignments.clone()),
                        networking: Box::new(session),
                        own_role: Role::new(id),
                    })
                    .collect();
                network_sessions
            }
        };

        let mut jobs = vec![];
        for (player_id, mut network_session) in network_sessions.into_iter().enumerate() {
            let player_seed = seeds[player_id];
            let task = tokio::spawn(async move {
                let prf = setup_replicated_prf(&mut network_session, player_seed)
                    .await
                    .unwrap();
                (network_session, prf)
            });
            jobs.push(task);
        }
        let sessions = join_all(jobs)
            .await
            .into_iter()
            .map(|t| {
                let (network_session, prf) = t?;
                Ok(Session {
                    network_session,
                    prf,
                })
            })
            .collect::<eyre::Result<Vec<_>>>()?;
        Ok(LocalRuntime { sessions })
    }

    pub async fn new(identities: Vec<Identity>, seeds: Vec<PrfSeed>) -> eyre::Result<Self> {
        Self::new_with_network_type(identities, seeds, NetworkType::LocalChannel).await
    }

    fn into_sessions(self) -> Vec<SessionRef> {
        self.sessions
            .into_iter()
            .map(|s| Arc::new(Mutex::new(s)))
            .collect()
    }

    async fn mock_sessions(network_type: NetworkType) -> eyre::Result<Vec<SessionRef>> {
        Self::mock_setup(network_type)
            .await
            .map(|rt| rt.into_sessions())
    }

    pub async fn mock_sessions_with_channel() -> eyre::Result<Vec<SessionRef>> {
        Self::mock_sessions(NetworkType::LocalChannel).await
    }

    pub async fn mock_sessions_with_grpc() -> eyre::Result<Vec<SessionRef>> {
        Self::mock_sessions(NetworkType::GrpcChannel).await
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
