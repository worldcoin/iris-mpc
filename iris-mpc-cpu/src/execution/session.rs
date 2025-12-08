use crate::{
    execution::player::{Identity, Role},
    network::{value::NetworkValue, Networking},
    protocol::prf::Prf,
};
use eyre::{eyre, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use std::{collections::HashMap, fmt::Debug, sync::Arc};
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
    time::sleep,
};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionId(pub u32);

impl From<u32> for SessionId {
    fn from(id: u32) -> Self {
        SessionId(id)
    }
}
/// Added to introduce delay
const ARTIFICIAL_LINK_DELAY: Duration = Duration::from_millis(60); // this is for dev tests
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

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LaneId(pub usize);

impl From<usize> for LaneId {
    fn from(id: usize) -> Self {
        LaneId(id)
    }
}

pub type NetworkingImpl = Arc<dyn Networking + Send + Sync>;

#[derive(Debug)]
pub struct Session {
    pub network_session: NetworkSession,
    pub prf: Prf,
    pub(crate) network_task: JoinHandle<()>,
}

pub struct NetworkSessionInner {
    pub session_id: SessionId,
    pub role_assignments: Arc<HashMap<Role, Identity>>,
    pub networking: NetworkingImpl,
    pub own_role: Role,
}

impl NetworkSessionInner {
    async fn send(&self, value: NetworkValue, receiver: &Identity) -> Result<()> {
        // let d = artificial_link_delay();
        // sleep(d).await;

        // sleep(ARTIFICIAL_LINK_DELAY).await; //uncomment this for dev tests
        self.send_on_lane(value, receiver, LaneId(0)).await
    }

    async fn receive(&self, sender: &Identity) -> Result<NetworkValue> {
        self.receive_on_lane(sender, LaneId(0)).await
    }

    async fn receive_on_lane(&self, sender: &Identity, lane: LaneId) -> Result<NetworkValue> {
        self.networking.receive_on_lane(sender, lane.0).await
    }

    async fn send_on_lane(
        &self,
        value: NetworkValue,
        receiver: &Identity,
        lane: LaneId,
    ) -> Result<()> {
        self.networking.send_on_lane(value, receiver, lane.0).await
    }

    pub fn lane_count(&self) -> usize {
        self.networking.num_lanes()
    }

    pub async fn send_next(&self, value: NetworkValue) -> Result<()> {
        let next_identity = self.next_identity()?;
        self.send(value, &next_identity).await
    }

    pub async fn send_prev(&self, value: NetworkValue) -> Result<()> {
        let prev_identity = self.prev_identity()?;
        self.send(value, &prev_identity).await
    }

    pub async fn receive_next(&self) -> Result<NetworkValue> {
        let next_identity = self.next_identity()?;
        self.receive(&next_identity).await
    }

    pub async fn receive_prev(&self) -> Result<NetworkValue> {
        let prev_identity = self.prev_identity()?;
        self.receive(&prev_identity).await
    }

    pub async fn receive_next_on_lane(&self, lane: LaneId) -> Result<NetworkValue> {
        let next_identity = self.next_identity()?;
        self.receive_on_lane(&next_identity, lane).await
    }

    pub async fn receive_prev_on_lane(&self, lane: LaneId) -> Result<NetworkValue> {
        let prev_identity = self.prev_identity()?;
        self.receive_on_lane(&prev_identity, lane).await
    }

    pub async fn send_next_on_lane(&self, lane: LaneId, value: NetworkValue) -> Result<()> {
        let next_identity = self.next_identity()?;
        self.send_on_lane(value, &next_identity, lane).await
    }

    pub async fn send_prev_on_lane(&self, lane: LaneId, value: NetworkValue) -> Result<()> {
        let prev_identity = self.prev_identity()?;
        self.send_on_lane(value, &prev_identity, lane).await
    }
}

impl Debug for NetworkSessionInner {
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

impl SessionHandles for NetworkSessionInner {
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
        self.network_session.session_id()
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

/// Internal message sent to the networking actor.
enum NetworkRequest {
    SendNext {
        value: NetworkValue,
        resp: oneshot::Sender<Result<()>>,
    },
    SendPrev {
        value: NetworkValue,
        resp: oneshot::Sender<Result<()>>,
    },
    SendNextOnLane {
        lane: LaneId,
        value: NetworkValue,
        resp: oneshot::Sender<Result<()>>,
    },
    SendPrevOnLane {
        lane: LaneId,
        value: NetworkValue,
        resp: oneshot::Sender<Result<()>>,
    },
    ReceiveNext {
        resp: oneshot::Sender<Result<NetworkValue>>,
    },
    ReceivePrev {
        resp: oneshot::Sender<Result<NetworkValue>>,
    },
    ReceiveNextOnLane {
        lane: LaneId,
        resp: oneshot::Sender<Result<NetworkValue>>,
    },
    ReceivePrevOnLane {
        lane: LaneId,
        resp: oneshot::Sender<Result<NetworkValue>>,
    },
}

/// Cloneable client that forwards networking operations to a dedicated actor task.
#[derive(Clone, Debug)]
pub struct NetworkClient {
    tx: mpsc::Sender<NetworkRequest>,
    lane_count: usize,
}

impl NetworkClient {
    pub async fn send_next(&self, value: NetworkValue) -> Result<()> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.tx
            .send(NetworkRequest::SendNext {
                value,
                resp: resp_tx,
            })
            .await
            .map_err(|_| eyre!("network actor dropped"))?;
        resp_rx.await.map_err(|_| eyre!("network actor stopped"))?
    }

    pub async fn send_prev(&self, value: NetworkValue) -> Result<()> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.tx
            .send(NetworkRequest::SendPrev {
                value,
                resp: resp_tx,
            })
            .await
            .map_err(|_| eyre!("network actor dropped"))?;
        resp_rx.await.map_err(|_| eyre!("network actor stopped"))?
    }

    pub async fn receive_next(&self) -> Result<NetworkValue> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.tx
            .send(NetworkRequest::ReceiveNext { resp: resp_tx })
            .await
            .map_err(|_| eyre!("network actor dropped"))?;
        resp_rx.await.map_err(|_| eyre!("network actor stopped"))?
    }

    pub async fn receive_prev(&self) -> Result<NetworkValue> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.tx
            .send(NetworkRequest::ReceivePrev { resp: resp_tx })
            .await
            .map_err(|_| eyre!("network actor dropped"))?;
        resp_rx.await.map_err(|_| eyre!("network actor stopped"))?
    }

    pub async fn receive_next_on_lane(&self, lane: LaneId) -> Result<NetworkValue> {
        if lane.0 >= self.lane_count {
            return Err(eyre!(
                "lane {} out of range (available: {})",
                lane.0,
                self.lane_count
            ));
        }
        let (resp_tx, resp_rx) = oneshot::channel();
        self.tx
            .send(NetworkRequest::ReceiveNextOnLane {
                lane,
                resp: resp_tx,
            })
            .await
            .map_err(|_| eyre!("network actor dropped"))?;
        resp_rx.await.map_err(|_| eyre!("network actor stopped"))?
    }

    pub async fn receive_prev_on_lane(&self, lane: LaneId) -> Result<NetworkValue> {
        if lane.0 >= self.lane_count {
            return Err(eyre!(
                "lane {} out of range (available: {})",
                lane.0,
                self.lane_count
            ));
        }
        let (resp_tx, resp_rx) = oneshot::channel();
        self.tx
            .send(NetworkRequest::ReceivePrevOnLane {
                lane,
                resp: resp_tx,
            })
            .await
            .map_err(|_| eyre!("network actor dropped"))?;
        resp_rx.await.map_err(|_| eyre!("network actor stopped"))?
    }

    pub async fn send_next_on_lane(&self, lane: LaneId, value: NetworkValue) -> Result<()> {
        if lane.0 >= self.lane_count {
            return Err(eyre!(
                "lane {} out of range (available: {})",
                lane.0,
                self.lane_count
            ));
        }
        let (resp_tx, resp_rx) = oneshot::channel();
        self.tx
            .send(NetworkRequest::SendNextOnLane {
                lane,
                value,
                resp: resp_tx,
            })
            .await
            .map_err(|_| eyre!("network actor dropped"))?;
        resp_rx.await.map_err(|_| eyre!("network actor stopped"))?
    }

    pub async fn send_prev_on_lane(&self, lane: LaneId, value: NetworkValue) -> Result<()> {
        if lane.0 >= self.lane_count {
            return Err(eyre!(
                "lane {} out of range (available: {})",
                lane.0,
                self.lane_count
            ));
        }
        let (resp_tx, resp_rx) = oneshot::channel();
        self.tx
            .send(NetworkRequest::SendPrevOnLane {
                lane,
                value,
                resp: resp_tx,
            })
            .await
            .map_err(|_| eyre!("network actor dropped"))?;
        resp_rx.await.map_err(|_| eyre!("network actor stopped"))?
    }

    pub fn lane_count(&self) -> usize {
        self.lane_count
    }
}

/// Spawn an async task that owns `network_session` and handles networking requests.
pub fn spawn_network_client(
    network_session: NetworkSessionInner,
) -> (NetworkClient, JoinHandle<()>) {
    let lane_count = network_session.lane_count();
    let (tx, mut rx) = mpsc::channel::<NetworkRequest>(64);
    let session = Arc::new(network_session);
    let handle = tokio::spawn(async move {
        while let Some(req) = rx.recv().await {
            let session = session.clone();
            tokio::spawn(async move {
                match req {
                    NetworkRequest::SendNext { value, resp } => {
                        let _ = resp.send(session.send_next(value).await);
                    }
                    NetworkRequest::SendPrev { value, resp } => {
                        let _ = resp.send(session.send_prev(value).await);
                    }
                    NetworkRequest::SendNextOnLane { lane, value, resp } => {
                        let _ = resp.send(session.send_next_on_lane(lane, value).await);
                    }
                    NetworkRequest::SendPrevOnLane { lane, value, resp } => {
                        let _ = resp.send(session.send_prev_on_lane(lane, value).await);
                    }
                    NetworkRequest::ReceiveNext { resp } => {
                        let _ = resp.send(session.receive_next().await);
                    }
                    NetworkRequest::ReceivePrev { resp } => {
                        let _ = resp.send(session.receive_prev().await);
                    }
                    NetworkRequest::ReceiveNextOnLane { lane, resp } => {
                        let _ = resp.send(session.receive_next_on_lane(lane).await);
                    }
                    NetworkRequest::ReceivePrevOnLane { lane, resp } => {
                        let _ = resp.send(session.receive_prev_on_lane(lane).await);
                    }
                }
            });
        }
    });
    (NetworkClient { tx, lane_count }, handle)
}

/// Handle used by protocol code; forwards operations to the shared `NetworkClient`.
#[derive(Clone, Debug)]
pub struct NetworkSession {
    client: NetworkClient,
    session_id: SessionId,
    role_assignments: Arc<HashMap<Role, Identity>>,
    own_role: Role,
    lane_count: usize,
}

impl NetworkSession {
    pub fn new(
        client: NetworkClient,
        session_id: SessionId,
        role_assignments: Arc<HashMap<Role, Identity>>,
        own_role: Role,
    ) -> Self {
        let lane_count = client.lane_count();
        Self {
            client,
            session_id,
            role_assignments,
            own_role,
            lane_count,
        }
    }

    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    pub fn own_role(&self) -> Role {
        self.own_role
    }

    pub fn own_identity(&self) -> Identity {
        self.role_assignments
            .get(&self.own_role)
            .expect("own identity")
            .clone()
    }

    pub fn role_assignments(&self) -> Arc<HashMap<Role, Identity>> {
        self.role_assignments.clone()
    }

    pub fn network_client(&self) -> NetworkClient {
        self.client.clone()
    }

    pub fn lane_count(&self) -> usize {
        self.lane_count
    }

    pub async fn send_next(&self, value: NetworkValue) -> Result<()> {
        self.client.send_next(value).await
    }

    pub async fn send_prev(&self, value: NetworkValue) -> Result<()> {
        self.client.send_prev(value).await
    }

    pub async fn receive_next(&self) -> Result<NetworkValue> {
        self.client.receive_next().await
    }

    pub async fn receive_prev(&self) -> Result<NetworkValue> {
        self.client.receive_prev().await
    }

    pub async fn send_next_on_lane(&self, lane: LaneId, value: NetworkValue) -> Result<()> {
        self.client.send_next_on_lane(lane, value).await
    }

    pub async fn send_prev_on_lane(&self, lane: LaneId, value: NetworkValue) -> Result<()> {
        self.client.send_prev_on_lane(lane, value).await
    }

    pub async fn receive_next_on_lane(&self, lane: LaneId) -> Result<NetworkValue> {
        self.client.receive_next_on_lane(lane).await
    }

    pub async fn receive_prev_on_lane(&self, lane: LaneId) -> Result<NetworkValue> {
        self.client.receive_prev_on_lane(lane).await
    }
}

impl SessionHandles for NetworkSession {
    fn session_id(&self) -> SessionId {
        self.session_id
    }

    fn own_role(&self) -> Role {
        self.own_role
    }

    fn own_identity(&self) -> Identity {
        self.own_identity()
    }

    fn identity(&self, role: &Role) -> Result<&Identity> {
        self.role_assignments
            .get(role)
            .ok_or_else(|| eyre!("Couldn't find role in role assignment map"))
    }

    fn prev_identity(&self) -> Result<Identity> {
        let prev_role = self.own_role.prev(self.role_assignments.len() as u8);
        self.role_assignments
            .get(&prev_role)
            .cloned()
            .ok_or_else(|| eyre!("Couldn't find role in role assignment map for prev_identity"))
    }

    fn next_identity(&self) -> Result<Identity> {
        let next_role = self.own_role.next(self.role_assignments.len() as u8);
        self.role_assignments
            .get(&next_role)
            .cloned()
            .ok_or_else(|| eyre!("Couldn't find role in role assignment map for next_identity"))
    }
}
