use crate::{
    execution::player::{Identity, Role},
    next_gen::Networking,
    protocol::prf::Prf,
};
use eyre::eyre;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionId(pub u128);

impl From<u128> for SessionId {
    fn from(id: u128) -> Self {
        SessionId(id)
    }
}

pub type NetworkingImpl = Arc<dyn Networking + Send + Sync>;

#[derive(Clone)]
pub struct Session {
    pub boot_session: BootSession,
    pub setup:        Prf,
}

#[derive(Clone)]
pub struct BootSession {
    pub session_id:       SessionId,
    pub role_assignments: Arc<HashMap<Role, Identity>>,
    pub networking:       NetworkingImpl,
    pub own_identity:     Identity,
}

pub trait SessionHandles {
    fn session_id(&self) -> SessionId;
    fn own_role(&self) -> eyre::Result<Role>;
    fn own_identity(&self) -> Identity;
    fn identity(&self, role: &Role) -> eyre::Result<&Identity>;
    fn network(&self) -> &NetworkingImpl;
}

impl SessionHandles for BootSession {
    fn session_id(&self) -> SessionId {
        self.session_id
    }

    fn own_role(&self) -> eyre::Result<Role> {
        let role: Vec<&Role> = self
            .role_assignments
            .iter()
            .filter_map(|(role, identity)| {
                if identity == &self.own_identity {
                    Some(role)
                } else {
                    None
                }
            })
            .collect();
        if role.len() != 1 {
            Err(eyre!(
                "Couldn't find exact match in role assignment hashmap to retrieve own role"
            ))
        } else {
            Ok((*role[0]).clone())
        }
    }

    fn own_identity(&self) -> Identity {
        self.own_identity.clone()
    }

    fn identity(&self, role: &Role) -> eyre::Result<&Identity> {
        match self.role_assignments.get(role) {
            Some(id) => Ok(id),
            None => Err(eyre!("Couldn't find role in role assignment map")),
        }
    }

    fn network(&self) -> &NetworkingImpl {
        &self.networking
    }
}

impl SessionHandles for Session {
    fn session_id(&self) -> SessionId {
        self.boot_session.session_id
    }
    fn identity(&self, role: &Role) -> eyre::Result<&Identity> {
        self.boot_session.identity(role)
    }
    fn network(&self) -> &NetworkingImpl {
        self.boot_session.network()
    }
    fn own_identity(&self) -> Identity {
        self.boot_session.own_identity()
    }
    fn own_role(&self) -> eyre::Result<Role> {
        self.boot_session.own_role()
    }
}

impl Session {
    pub fn prf_as_mut(&mut self) -> &mut Prf {
        &mut self.setup
    }
}
