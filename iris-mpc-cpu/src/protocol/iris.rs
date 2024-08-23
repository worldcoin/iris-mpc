use super::prf::{Prf, PrfSeed};
use crate::{
    error::Error,
    networks::network_trait::NetworkTrait,
    shares::{
        share::Share,
        vecshare::{SliceShare, VecShare},
    },
};
use bytes::{Buf, Bytes, BytesMut};
use iris_mpc_common::{id::PartyID, iris_db::iris::IrisCodeArray};

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

    pub(crate) fn get_cmp_diff(&self, dot: &mut Share<u16>, mask_ones: usize) {
        let threshold = (mask_ones as f64 * (1. - 2. * MATCH_THRESHOLD_RATIO)) as usize;
        *dot = dot.sub_from_const(
            threshold
                .try_into()
                .expect("Sizes are checked in constructor"),
            self.network.get_id(),
        )
    }

    pub(crate) fn combine_masks(mask_a: &IrisCodeArray, mask_b: &IrisCodeArray) -> IrisCodeArray {
        *mask_a & *mask_b
    }

    pub(crate) fn rep3_compare_iris_public_mask_many(
        &mut self,
        a: SliceShare<'_, u16>,
        b: &[VecShare<u16>],
        mask_a: &IrisCodeArray,
        mask_b: &[IrisCodeArray],
    ) -> Result<VecShare<u64>, Error> {
        let amount = b.len();
        if (amount != mask_b.len()) || (amount == 0) {
            return Err(Error::InvalidSize);
        }

        let masks = mask_b
            .iter()
            .map(|b| Self::combine_masks(mask_a, b))
            .collect::<Vec<_>>();
        let mask_lens: Vec<_> = masks.iter().map(|m| m.count_ones()).collect();

        let mut dots = self.rep3_dot_many(a, b)?;
        // self.compare_threshold_many(dots, mask_lens)

        // a < b <=> msb(a - b)
        // Given no overflow, which is enforced in constructor
        for (dot, mask_len) in dots.iter_mut().zip(mask_lens) {
            self.get_cmp_diff(dot, mask_len);
        }

        self.extract_msb_u16::<{ u16::BITS as usize }>(dots)
    }
}
