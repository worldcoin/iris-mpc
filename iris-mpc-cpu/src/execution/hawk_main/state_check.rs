use eyre::{bail, eyre, Result};
use futures::join;
use serde::{Deserialize, Serialize};
use siphasher::sip::SipHasher13;
use std::hash::{Hash, Hasher};

use crate::{
    execution::hawk_main::{LEFT, RIGHT},
    network::value::{NetworkValue, StateChecksum},
};

use super::{BothEyes, HawkSession};

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

    pub fn checksum(&self) -> u64 {
        self.accumulator
    }

    fn hash(value: impl Hash) -> u64 {
        let mut hasher = SipHasher13::default();
        value.hash(&mut hasher);
        hasher.finish()
    }
}

impl HawkSession {
    pub async fn state_check(sessions: BothEyes<&HawkSession>) -> Result<()> {
        let (left_state, right_state) = join!(
            HawkSession::state_check_side(sessions[LEFT]),
            HawkSession::state_check_side(sessions[RIGHT]),
        );

        let left_state = left_state?;
        let right_state = right_state?;
        left_state.check_left_vs_right(&right_state)?;
        Ok(())
    }

    async fn state_check_side(session: &HawkSession) -> Result<StateChecksum> {
        let my_state = session.checksum().await;
        let net = &mut session.aby3_store.write().await.session.network_session;

        // Send my state to others.
        let my_msg = || NetworkValue::StateChecksum(my_state.clone());
        net.send_prev(my_msg()).await?;
        net.send_next(my_msg()).await?;

        // Receive their state.
        let decode = |msg| match msg {
            Ok(NetworkValue::StateChecksum(c)) => Ok(c),
            other => {
                tracing::error!("Unexpected message format: {:?}", other);
                Err(eyre!("Could not deserialize StateChecksum"))
            }
        };
        let prev_state = decode(net.receive_prev().await)?;
        let next_state = decode(net.receive_next().await)?;

        if prev_state != my_state || next_state != my_state {
            bail!(
                "Party states have diverged: my_state={my_state:?} prev_state={prev_state:?} next_state={next_state:?}"
            );
        }
        Ok(my_state)
    }

    async fn checksum(&self) -> StateChecksum {
        StateChecksum {
            irises: self.aby3_store.read().await.storage.checksum().await,
            graph: self.graph_store.read().await.checksum(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::hnsw::graph::neighborhood::SortedEdgeIds;
    use iris_mpc_common::vector_id::VectorId;
    use itertools::Itertools;

    #[test]
    fn test_set_hash() {
        let mut digests = vec![];

        let a = 12_u64;
        let b = VectorId::from_serial_id(34);
        let c = (a, &SortedEdgeIds::from_ascending_vec(vec![b; 10]));

        let mut set_hash = SetHash::default();
        digests.push(set_hash.checksum());

        set_hash.add_unordered(a);
        digests.push(set_hash.checksum());

        set_hash.add_unordered(b);
        digests.push(set_hash.checksum());

        set_hash.add_unordered(c);
        digests.push(set_hash.checksum());

        assert!(digests.iter().all_unique());

        let different_order = {
            let mut set_hash = SetHash::default();
            set_hash.add_unordered(c);
            set_hash.add_unordered(a);
            set_hash.remove(c);
            set_hash.add_unordered(c);
            set_hash.add_unordered(b);
            set_hash.checksum()
        };
        assert_eq!(digests.pop().unwrap(), different_order);

        set_hash.remove(c);
        assert_eq!(digests.pop().unwrap(), set_hash.checksum());

        set_hash.remove(b);
        assert_eq!(digests.pop().unwrap(), set_hash.checksum());

        set_hash.remove(a);
        assert_eq!(digests.pop().unwrap(), set_hash.checksum());
        assert_eq!(SetHash::default().checksum(), set_hash.checksum());
    }
}
