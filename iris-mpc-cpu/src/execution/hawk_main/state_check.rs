use eyre::{eyre, Result};
use itertools::{chain, Itertools};
use serde::{Deserialize, Serialize};
use siphasher::sip::SipHasher13;
use std::hash::{Hash, Hasher};

use super::{HawkSession, HawkSessionRef};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SetHash {
    accumulator: u64,
}

impl SetHash {
    pub fn add_unordered(&mut self, value: impl Hash) {
        self.accumulator = self.accumulator.wrapping_add(Self::hash(value));
    }

    pub fn remove(&mut self, value: impl Hash) {
        self.accumulator = self.accumulator.wrapping_sub(Self::hash(value));
    }

    pub fn digest(&self) -> u64 {
        self.accumulator
    }

    fn hash(value: impl Hash) -> u64 {
        let mut hasher = SipHasher13::default();
        value.hash(&mut hasher);
        hasher.finish()
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

        let graph_digest = self.graph_store.read().await.digest();

        chain(iris_digest.to_le_bytes(), graph_digest.to_le_bytes()).collect_vec()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::hnsw::graph::neighborhood::SortedEdgeIds;
    use iris_mpc_common::vector_id::VectorId;

    #[test]
    fn test_set_hash() {
        let mut digests = vec![];

        let a = 12_u64;
        let b = VectorId::from_serial_id(34);
        let c = (a, &SortedEdgeIds::from_ascending_vec(vec![b; 10]));

        let mut set_hash = SetHash::default();
        digests.push(set_hash.digest());

        set_hash.add_unordered(a);
        digests.push(set_hash.digest());

        set_hash.add_unordered(b);
        digests.push(set_hash.digest());

        set_hash.add_unordered(c);
        digests.push(set_hash.digest());

        assert!(digests.iter().all_unique());

        let different_order = {
            let mut set_hash = SetHash::default();
            set_hash.add_unordered(c);
            set_hash.add_unordered(a);
            set_hash.remove(c);
            set_hash.add_unordered(c);
            set_hash.add_unordered(b);
            set_hash.digest()
        };
        assert_eq!(digests.pop().unwrap(), different_order);

        set_hash.remove(c);
        assert_eq!(digests.pop().unwrap(), set_hash.digest());

        set_hash.remove(b);
        assert_eq!(digests.pop().unwrap(), set_hash.digest());

        set_hash.remove(a);
        assert_eq!(digests.pop().unwrap(), set_hash.digest());
        assert_eq!(SetHash::default().digest(), set_hash.digest());
    }
}
