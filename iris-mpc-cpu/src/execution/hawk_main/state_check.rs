use ampc_secret_sharing::{shares::bit::Bit, RingElement};
use eyre::{bail, eyre, Result};
use futures::join;
use serde::{Deserialize, Serialize};
use siphasher::sip::SipHasher13;
use std::hash::{Hash, Hasher};

use crate::{
    execution::hawk_main::{BothOrient, LEFT, RIGHT},
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
    // returns true if there is a mismatch
    pub async fn sync_peers(shutdown: bool, sessions: &BothEyes<Vec<HawkSession>>) -> Result<bool> {
        let session = &sessions[0][0];
        let mut store = session.aby3_store.write().await;
        let msg = NetworkValue::RingElementBit(RingElement(Bit::new(shutdown)));
        let net = &mut store.session.network_session;
        net.send_prev(msg.clone()).await?;
        net.send_next(msg).await?;

        let decode = |msg| match msg {
            Ok(NetworkValue::RingElementBit(elem)) => Ok(elem.0.convert()),
            other => {
                tracing::error!("Unexpected message format: {:?}", other);
                Err(eyre!("Could not deserialize sync result"))
            }
        };
        let prev_share = decode(net.receive_prev().await)?;
        let next_share = decode(net.receive_next().await)?;

        Ok(prev_share == shutdown && next_share == shutdown)
    }

    pub async fn prf_check(sessions: &BothOrient<BothEyes<Vec<HawkSession>>>) -> Result<()> {
        // make a function because the borrow checker can't track the lifetimes properly if this was a closure
        async fn squeeze_rng(session: &HawkSession) -> Result<()> {
            let mut store = session.aby3_store.write().await;
            let prf = &mut store.session.prf;

            let my_share = prf.gen_zero_share::<u128>();
            let my_msg = || NetworkValue::PrfCheck(my_share);

            let net = &mut store.session.network_session;
            net.send_prev(my_msg()).await?;
            net.send_next(my_msg()).await?;

            let decode = |msg| match msg {
                Ok(NetworkValue::PrfCheck(c)) => Ok(c),
                other => {
                    tracing::error!("Unexpected message format: {:?}", other);
                    Err(eyre!("Could not deserialize PrfCheck"))
                }
            };
            let prev_share = decode(net.receive_prev().await)?;
            let next_share = decode(net.receive_next().await)?;

            if (prev_share + my_share + next_share).convert() != 0_u128 {
                bail!("PRFs are out of sync");
            }
            Ok(())
        }

        let _ = futures::future::try_join_all(
            sessions
                .iter()
                .flat_map(|orient| orient.iter().flat_map(|eyes| eyes.iter()))
                .map(squeeze_rng),
        )
        .await?;
        Ok(())
    }

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
            irises: self.aby3_store.read().await.checksum().await,
            graph: self.graph_store.read().await.checksum(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use iris_mpc_common::vector_id::VectorId;
    use itertools::Itertools;

    #[test]
    fn test_set_hash() {
        let mut digests = vec![];

        let a = 12_u64;
        let b = VectorId::from_serial_id(34);
        let c = (a, &vec![b; 10]);

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
