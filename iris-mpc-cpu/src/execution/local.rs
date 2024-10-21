use crate::{
    execution::{
        player::*,
        session::{BootSession, Session, SessionHandles, SessionId},
    },
    network::local::LocalNetworkingStore,
    protocol::{
        ops::setup_replicated_prf,
        prf::{Prf, PrfSeed},
    },
};
use std::{collections::HashMap, sync::Arc};
use tokio::task::JoinSet;

#[derive(Debug, Clone)]
pub struct LocalRuntime {
    pub identities:       Vec<Identity>,
    pub role_assignments: RoleAssignment,
    pub prf_setups:       Option<HashMap<Role, Prf>>,
    pub seeds:            Vec<PrfSeed>,
}

impl LocalRuntime {
    pub fn replicated_test_config() -> Self {
        let num_parties = 3;
        let identities: Vec<Identity> = vec!["alice".into(), "bob".into(), "charlie".into()];
        let mut seeds = Vec::new();
        for i in 0..num_parties {
            let mut seed = [0_u8; 16];
            seed[0] = i;
            seeds.push(seed);
        }
        LocalRuntime::new(identities, seeds)
    }
    pub fn new(identities: Vec<Identity>, seeds: Vec<PrfSeed>) -> Self {
        let role_assignments: RoleAssignment = identities
            .iter()
            .enumerate()
            .map(|(index, id)| (Role::new(index), id.clone()))
            .collect();
        LocalRuntime {
            identities,
            role_assignments,
            prf_setups: None,
            seeds,
        }
    }

    pub async fn create_player_sessions(&self) -> eyre::Result<HashMap<Identity, Session>> {
        let network = LocalNetworkingStore::from_host_ids(&self.identities);
        let sess_id = SessionId::from(0_u128);
        let boot_sessions: Vec<BootSession> = (0..self.seeds.len())
            .map(|i| {
                let identity = self.identities[i].clone();
                BootSession {
                    session_id:       sess_id,
                    role_assignments: Arc::new(self.role_assignments.clone()),
                    networking:       Arc::new(network.get_local_network(identity.clone())),
                    own_identity:     identity,
                }
            })
            .collect();

        let mut jobs = JoinSet::new();
        for (player_id, boot_session) in boot_sessions.iter().enumerate() {
            let player_seed = self.seeds[player_id];
            let sess = boot_session.clone();
            jobs.spawn(async move {
                let prf = setup_replicated_prf(&sess, player_seed).await.unwrap();
                (sess, prf)
            });
        }
        let mut complete_sessions = HashMap::new();
        while let Some(t) = jobs.join_next().await {
            let (boot_session, prf) = t.unwrap();
            complete_sessions.insert(boot_session.own_identity(), Session {
                boot_session,
                setup: prf,
            });
        }
        Ok(complete_sessions)
    }
}
