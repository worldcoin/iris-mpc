use crate::{
    execution::player::{Identity, Role},
    network::Networking,
    protocol::prf::Prf,
};
use eyre::eyre;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt::Debug, sync::Arc};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionId(pub u64);

impl From<u64> for SessionId {
    fn from(id: u64) -> Self {
        SessionId(id)
    }
}

pub type NetworkingImpl = Box<dyn Networking + Send + Sync>;

#[derive(Debug)]
pub struct Session {
    pub network_session: NetworkSession,
    pub prf: Prf,
}

pub struct NetworkSession {
    pub session_id: SessionId,
    pub role_assignments: Arc<HashMap<Role, Identity>>,
    pub networking: NetworkingImpl,
    pub own_identity: Identity,
}

impl NetworkSession {
    async fn send(&self, value: Vec<u8>, receiver: &Identity) -> eyre::Result<()> {
        self.networking.send(value, receiver).await
    }

    async fn receive(&mut self, sender: &Identity) -> eyre::Result<Vec<u8>> {
        self.networking.receive(sender).await
    }

    pub async fn send_next(&self, value: Vec<u8>) -> eyre::Result<()> {
        let next_identity = self.next_identity()?;
        self.send(value, &next_identity).await
    }

    pub async fn send_prev(&self, value: Vec<u8>) -> eyre::Result<()> {
        let prev_identity = self.prev_identity()?;
        self.send(value, &prev_identity).await
    }

    pub async fn receive_next(&mut self) -> eyre::Result<Vec<u8>> {
        let next_identity = self.next_identity()?;
        self.receive(&next_identity).await
    }

    pub async fn receive_prev(&mut self) -> eyre::Result<Vec<u8>> {
        let prev_identity = self.prev_identity()?;
        self.receive(&prev_identity).await
    }
}

impl Debug for NetworkSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // TODO: incorporate networking into debug output
        f.debug_struct("NetworkSession")
            .field("session_id", &self.session_id)
            .field("role_assignments", &self.role_assignments)
            .field("own_identity", &self.own_identity)
            .finish()
    }
}

pub trait SessionHandles {
    fn session_id(&self) -> SessionId;
    fn own_role(&self) -> eyre::Result<Role>;
    fn own_identity(&self) -> Identity;
    fn identity(&self, role: &Role) -> eyre::Result<&Identity>;
    fn next_identity(&self) -> eyre::Result<Identity>;
    fn prev_identity(&self) -> eyre::Result<Identity>;
}

impl SessionHandles for NetworkSession {
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

    fn prev_identity(&self) -> eyre::Result<Identity> {
        let prev_role = self.own_role()?.prev(self.role_assignments.len() as u8);
        match self.role_assignments.get(&prev_role) {
            Some(id) => Ok(id.clone()),
            None => Err(eyre!(
                "Couldn't find role in role assignment map for prev_identity"
            )),
        }
    }

    fn next_identity(&self) -> eyre::Result<Identity> {
        let next_role = self.own_role()?.next(self.role_assignments.len() as u8);
        match self.role_assignments.get(&next_role) {
            Some(id) => Ok(id.clone()),
            None => Err(eyre!(
                "Couldn't find role in role assignment map for next_identity"
            )),
        }
    }
}

impl SessionHandles for Session {
    fn session_id(&self) -> SessionId {
        self.network_session.session_id
    }
    fn identity(&self, role: &Role) -> eyre::Result<&Identity> {
        self.network_session.identity(role)
    }
    fn own_identity(&self) -> Identity {
        self.network_session.own_identity()
    }
    fn own_role(&self) -> eyre::Result<Role> {
        self.network_session.own_role()
    }
    fn prev_identity(&self) -> eyre::Result<Identity> {
        self.network_session.prev_identity()
    }
    fn next_identity(&self) -> eyre::Result<Identity> {
        self.network_session.next_identity()
    }
}
