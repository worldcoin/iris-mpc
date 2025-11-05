use crate::{
    execution::player::{Identity, Role},
    network::{value::NetworkValue, Networking},
    protocol::prf::Prf,
};
use eyre::{eyre, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use std::{collections::HashMap, fmt::Debug, sync::Arc};
use tokio::time::sleep;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionId(pub u32);

impl From<u32> for SessionId {
    fn from(id: u32) -> Self {
        SessionId(id)
    }
}
/// Added to introduce delay
const ARTIFICIAL_LINK_DELAY: Duration = Duration::from_millis(20); // this is for dev tests
                                                                   // use std::sync::OnceLock; // use this for local tests

// fn artificial_link_delay() -> Duration {
//     static DELAY: OnceLock<Duration> = OnceLock::new();
//     *DELAY.get_or_init(|| {
//         std::env::var("NET_DELAY")
//             .ok()
//             .and_then(|s| s.parse::<u64>().ok())
//             .map(Duration::from_millis)
//             .unwrap_or(Duration::from_millis(30)) // default if NET_DELAY unset
//     })
// }

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
        // let d = artificial_link_delay();
        // sleep(d).await;

        sleep(ARTIFICIAL_LINK_DELAY).await; //uncomment this for dev tests
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
