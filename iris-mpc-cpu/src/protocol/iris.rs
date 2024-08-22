use super::prf::{Prf, PrfSeed};
use crate::{error::Error, networks::network_trait::NetworkTrait};
use bytes::{Buf, Bytes, BytesMut};
use iris_mpc_common::id::PartyID;

pub(crate) const IRIS_CODE_SIZE: usize =
    iris_mpc_common::iris_db::iris::IrisCodeArray::IRIS_CODE_SIZE;
pub(crate) const MATCH_THRESHOLD_RATIO: f64 = iris_mpc_common::iris_db::iris::MATCH_THRESHOLD_RATIO;
pub(crate) const B_BITS: u64 = 16;
pub(crate) const B: u64 = 1 << B_BITS;
pub(crate) const A: u64 = ((1. - 2. * MATCH_THRESHOLD_RATIO) * B as f64) as u64;
pub(crate) const A_BITS: u32 = u64::BITS - A.leading_zeros();

pub(crate) struct WorkerThread<N: NetworkTrait> {
    pub(crate) id:      usize,
    pub(crate) network: N,
    pub(crate) prf:     Prf,
}

impl<N: NetworkTrait> WorkerThread<N> {
    pub(crate) fn create(id: usize, network: N) -> Self {
        Self {
            id,
            network,
            prf: Prf::default(),
        }
    }

    pub fn get_party_id(&self) -> PartyID {
        self.network.get_id()
    }

    pub(crate) fn bytes_to_seed(mut bytes: BytesMut) -> Result<PrfSeed, Error> {
        if bytes.len() != std::mem::size_of::<PrfSeed>() {
            Err(Error::InvalidMessageSize)
        } else {
            let mut their_seed: PrfSeed = PrfSeed::default();
            bytes.copy_to_slice(&mut their_seed);
            Ok(their_seed)
        }
    }

    pub(crate) async fn setup_prf(&mut self) -> Result<(), Error> {
        let seed = Prf::gen_seed();
        let data = Bytes::from_iter(seed.into_iter());
        self.network.send_next_id(data).await?;
        let response = self.network.receive_prev_id().await?;
        let their_seed = Self::bytes_to_seed(response)?;
        self.prf = Prf::new(seed, their_seed);
        Ok(())
    }
}
