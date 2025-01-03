use crate::{
    execution::{player::Identity, session::SessionId},
    network::Networking,
};
use async_trait::async_trait;
use dashmap::DashMap;
use eyre::eyre;
use std::sync::Arc;

#[derive(Debug, Clone)]
struct Value {
    value: Vec<u8>,
}

type P2PChannels = Arc<
    DashMap<
        (Identity, Identity),
        (
            Arc<async_channel::Sender<Value>>,
            Arc<async_channel::Receiver<Value>>,
        ),
    >,
>;

#[derive(Debug, Clone)]
pub struct LocalNetworkingStore {
    p2p_channels: P2PChannels,
}

impl LocalNetworkingStore {
    pub fn from_host_ids(identities: &[Identity]) -> Self {
        let p2p = DashMap::new();
        for v1 in identities.to_owned().iter() {
            for v2 in identities.to_owned().iter() {
                if v1 != v2 {
                    let (tx, rx) = async_channel::unbounded::<Value>();
                    p2p.insert((v1.clone(), v2.clone()), (Arc::new(tx), Arc::new(rx)));
                }
            }
        }
        LocalNetworkingStore {
            p2p_channels: Arc::new(p2p),
        }
    }

    pub fn get_local_network(&self, owner: Identity) -> LocalNetworking {
        LocalNetworking {
            p2p_channels: Arc::clone(&self.p2p_channels),
            owner,
        }
    }
}

#[derive(Debug)]
pub struct LocalNetworking {
    p2p_channels: P2PChannels,
    pub owner:    Identity,
}

#[async_trait]
impl Networking for LocalNetworking {
    async fn send(
        &self,
        val: Vec<u8>,
        receiver: &Identity,
        _session_id: &SessionId,
    ) -> eyre::Result<(), eyre::Error> {
        let (tx, _) = self
            .p2p_channels
            .get(&(self.owner.clone(), receiver.clone()))
            .ok_or_else(|| {
                eyre!(format!(
                    "p2p channel retrieve error when sending: owner: {:?}, receiver: {:?}. \
                     session {:?}",
                    self.owner, receiver, _session_id
                ))
            })?
            .value()
            .clone();

        let ready_to_send_value = Value { value: val };
        tx.send(ready_to_send_value).await.map_err(|e| e.into())
    }

    async fn receive(&self, sender: &Identity, _session_id: &SessionId) -> eyre::Result<Vec<u8>> {
        let (_, rx) = self
            .p2p_channels
            .get(&(sender.clone(), self.owner.clone()))
            .ok_or_else(|| {
                eyre!(format!(
                    "p2p channel retrieve error when receiving: owner: {:?}, sender: {:?}",
                    self.owner, sender
                ))
            })?
            .value()
            .clone();

        let received_value = rx.recv().await?;
        Ok(received_value.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::value::NetworkValue;
    use std::num::Wrapping;

    #[tokio::test]
    async fn test_network_send_receive() {
        let identities: Vec<Identity> = vec!["alice".into(), "bob".into(), "charlie".into()];
        let networking_store = LocalNetworkingStore::from_host_ids(&identities);

        let alice = networking_store.get_local_network("alice".into());
        let bob = networking_store.get_local_network("bob".into());

        let task1 = tokio::spawn(async move {
            let recv = bob.receive(&"alice".into(), &1_u64.into()).await;
            assert_eq!(
                NetworkValue::from_network(recv).unwrap(),
                NetworkValue::Ring16(Wrapping::<u16>(777))
            );
        });
        let task2 = tokio::spawn(async move {
            let value = NetworkValue::Ring16(Wrapping::<u16>(777));
            alice
                .send(value.to_network(), &"bob".into(), &1_u64.into())
                .await
        });

        let _ = tokio::try_join!(task1, task2).unwrap();
    }
}
