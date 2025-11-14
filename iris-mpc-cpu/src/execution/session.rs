use crate::{
    execution::player::{Identity, Role},
    network::{
        value::{NetworkInt, NetworkValue},
        Networking,
    },
    protocol::prf::Prf,
    shares::ring_impl::VecRingElement,
};
use eyre::{eyre, Result};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt::Debug, sync::Arc};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionId(pub u32);

impl From<u32> for SessionId {
    fn from(id: u32) -> Self {
        SessionId(id)
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StreamId(pub u32);

impl From<u32> for StreamId {
    fn from(id: u32) -> Self {
        StreamId(id)
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
    pub own_role: Role,
}

impl NetworkSession {
    async fn send(&mut self, value: NetworkValue, receiver: &Identity) -> Result<()> {
        self.networking.send(value, receiver).await
    }

    async fn receive(&mut self, sender: &Identity) -> Result<NetworkValue> {
        self.networking.receive(sender).await
    }

    pub async fn send_next(&mut self, value: NetworkValue) -> Result<()> {
        let next_identity = self.next_identity()?;
        self.send(value, &next_identity).await
    }

    pub async fn send_prev(&mut self, value: NetworkValue) -> Result<()> {
        let prev_identity = self.prev_identity()?;
        self.send(value, &prev_identity).await
    }

    pub async fn receive_next(&mut self) -> Result<NetworkValue> {
        let next_identity = self.next_identity()?;
        self.receive(&next_identity).await
    }

    pub async fn receive_prev(&mut self) -> Result<NetworkValue> {
        let prev_identity = self.prev_identity()?;
        self.receive(&prev_identity).await
    }
}

// Helper methods for sending and receiving VecRingElement<T>.
impl NetworkSession {
    async fn send_ring_vec<T: NetworkInt>(
        &mut self,
        data: &VecRingElement<T>,
        receiver: &Identity,
    ) -> Result<()> {
        let message = if data.len() == 1 {
            T::new_network_element(data.0[0])
        } else {
            T::new_network_vec(data.0.clone())
        };
        self.send(message, receiver).await
    }

    pub async fn send_ring_vec_next<T: NetworkInt>(
        &mut self,
        data: &VecRingElement<T>,
    ) -> Result<()> {
        self.send_ring_vec(data, &self.next_identity()?).await
    }

    pub async fn send_ring_vec_prev<T: NetworkInt>(
        &mut self,
        data: &VecRingElement<T>,
    ) -> Result<()> {
        self.send_ring_vec(data, &self.prev_identity()?).await
    }

    async fn receive_ring_vec<T: NetworkInt>(
        &mut self,
        receiver: &Identity,
    ) -> Result<VecRingElement<T>> {
        let m = self.receive(receiver).await?;
        Ok(T::into_vec(m)?.into())
    }

    pub async fn receive_ring_vec_next<T: NetworkInt>(&mut self) -> Result<VecRingElement<T>> {
        self.receive_ring_vec(&self.next_identity()?).await
    }

    pub async fn receive_ring_vec_prev<T: NetworkInt>(&mut self) -> Result<VecRingElement<T>> {
        self.receive_ring_vec(&self.prev_identity()?).await
    }
}

impl Debug for NetworkSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // TODO: incorporate networking into debug output
        f.debug_struct("NetworkSession")
            .field("session_id", &self.session_id)
            .field("role_assignments", &self.role_assignments)
            .field("own_identity", &self.own_identity())
            .finish()
    }
}

pub trait SessionHandles {
    fn session_id(&self) -> SessionId;
    fn own_role(&self) -> Role;
    fn own_identity(&self) -> Identity;
    fn identity(&self, role: &Role) -> Result<&Identity>;
    fn next_identity(&self) -> Result<Identity>;
    fn prev_identity(&self) -> Result<Identity>;
}

impl SessionHandles for NetworkSession {
    fn session_id(&self) -> SessionId {
        self.session_id
    }

    fn own_role(&self) -> Role {
        self.own_role
    }

    fn own_identity(&self) -> Identity {
        self.role_assignments.get(&self.own_role()).unwrap().clone()
    }

    fn identity(&self, role: &Role) -> Result<&Identity> {
        match self.role_assignments.get(role) {
            Some(id) => Ok(id),
            None => Err(eyre!("Couldn't find role in role assignment map")),
        }
    }

    fn prev_identity(&self) -> Result<Identity> {
        let prev_role = self.own_role().prev(self.role_assignments.len() as u8);
        match self.role_assignments.get(&prev_role) {
            Some(id) => Ok(id.clone()),
            None => Err(eyre!(
                "Couldn't find role in role assignment map for prev_identity"
            )),
        }
    }

    fn next_identity(&self) -> Result<Identity> {
        let next_role = self.own_role().next(self.role_assignments.len() as u8);
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
    fn identity(&self, role: &Role) -> Result<&Identity> {
        self.network_session.identity(role)
    }
    fn own_identity(&self) -> Identity {
        self.network_session.own_identity()
    }
    fn own_role(&self) -> Role {
        self.network_session.own_role()
    }
    fn prev_identity(&self) -> Result<Identity> {
        self.network_session.prev_identity()
    }
    fn next_identity(&self) -> Result<Identity> {
        self.network_session.next_identity()
    }
}
