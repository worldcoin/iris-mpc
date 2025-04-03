use eyre::{eyre, Result};
use itertools::{chain, Itertools};
use serde::{Deserialize, Serialize};
use siphasher::sip::SipHasher13;
use std::hash::{Hash, Hasher};

use super::{HawkSession, HawkSessionRef};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SetHash {
    hash: u64,
}

impl SetHash {
    pub fn add_unordered(&mut self, value: impl Hash) {
        let mut hasher = SipHasher13::default();
        value.hash(&mut hasher);
        let h = hasher.finish();
        self.hash = self.hash.wrapping_add(h);
    }

    pub fn digest(&self) -> u64 {
        self.hash
    }
}

impl HawkSession {
    pub async fn state_check(session: &HawkSessionRef) -> Result<()> {
        let mut session = session.write().await;

        let my_state = session.digest().await;

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

    async fn digest(&self) -> Vec<u8> {
        let iris_digest = self.aby3_store.storage.digest().await;

        let graph_digest = self.graph_store.read().await.digest_slow();

        chain(iris_digest.to_le_bytes(), graph_digest.to_le_bytes()).collect_vec()
    }
}
