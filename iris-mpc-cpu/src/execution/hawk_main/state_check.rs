use eyre::{eyre, Result};
use serde::{Deserialize, Serialize};
use siphasher::sip::SipHasher13;
use std::hash::{Hash, Hasher};

use super::{HawkSession, HawkSessionRef};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SetHash {
    hash: u64,
}

impl SetHash {
    pub fn add(&mut self, value: &impl Hash) {
        let mut hasher = SipHasher13::default();
        value.hash(&mut hasher);

        self.hash ^= hasher.finish();
    }

    pub fn digest(&self) -> u64 {
        self.hash
    }
}

impl HawkSession {
    pub async fn sync(session: &HawkSessionRef) -> Result<()> {
        let mut session = session.write().await;

        let my_state = session
            .aby3_store
            .storage
            .digest()
            .await
            .to_le_bytes()
            .to_vec();

        let net = &mut session.aby3_store.session.network_session;
        net.send_prev(my_state.clone()).await?;
        net.send_next(my_state.clone()).await?;
        let prev_state = net.receive_prev().await?;
        let next_state = net.receive_next().await?;

        if prev_state != my_state || next_state != my_state {
            return Err(eyre!(
                "Out-of-sync: my_state={my_state:?} prev_state={prev_state:?} next_state={next_state:?}"
            ));
        }
        Ok(())
    }
}
