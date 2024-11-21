use crate::{
    execution::{
        player::*,
        session::{BootSession, Session, SessionHandles, SessionId},
    },
    network::local::LocalNetworkingStore,
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

#[derive(Debug, Clone)]
pub struct LocalRuntime {
    pub identities:       Vec<Identity>,
    pub role_assignments: RoleAssignment,
    pub seeds:            Vec<PrfSeed>,
    // only one session per player is created
    pub sessions:         HashMap<Identity, Session>,
}

impl LocalRuntime {
    pub async fn replicated_test_config() -> eyre::Result<Self> {
        let num_parties = 3;
        let identities = generate_local_identities();
        let mut seeds = Vec::new();
        for i in 0..num_parties {
            let mut seed = [0_u8; 16];
            seed[0] = i;
            seeds.push(seed);
        }
        LocalRuntime::new(identities, seeds).await
    }

    pub async fn new(identities: Vec<Identity>, seeds: Vec<PrfSeed>) -> eyre::Result<Self> {
        let role_assignments: RoleAssignment = identities
            .iter()
            .enumerate()
            .map(|(index, id)| (Role::new(index), id.clone()))
            .collect();
        let network = LocalNetworkingStore::from_host_ids(&identities);
        let sess_id = SessionId::from(0_u128);
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
}
