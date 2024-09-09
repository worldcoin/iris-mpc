use super::prf::{Prf, PrfSeed};
use crate::{
    networks::network_trait::NetworkTrait,
    shares::{
        bit::Bit,
        int_ring::IntRing2k,
        ring_impl::RingElement,
        share::Share,
        vecshare::{SliceShare, VecShare},
    },
    utils::Utils,
};
use bytes::{Buf, Bytes, BytesMut};
use eyre::{eyre, Error, Result};
use iris_mpc_common::{
    galois_engine::degree4::GaloisRingIrisCodeShare, id::PartyID, iris_db::iris::IrisCodeArray,
};
use static_assertions::const_assert;
use std::ops::SubAssign;

pub(crate) const MATCH_THRESHOLD_RATIO: f64 = iris_mpc_common::iris_db::iris::MATCH_THRESHOLD_RATIO;
pub(crate) const B_BITS: u64 = 16;
pub(crate) const B: u64 = 1 << B_BITS;
pub(crate) const A: u64 = ((1. - 2. * MATCH_THRESHOLD_RATIO) * B as f64) as u64;
pub(crate) const A_BITS: u32 = u64::BITS - A.leading_zeros();

pub struct IrisWorker<N: NetworkTrait> {
    pub(crate) network: N,
    pub(crate) prf:     Prf,
}

impl<N: NetworkTrait> IrisWorker<N> {
    pub fn new(network: N) -> Self {
        const_assert!(MATCH_THRESHOLD_RATIO > 0.0 && MATCH_THRESHOLD_RATIO < 1.0);
        // checking that code size fits on 15 bits.
        const_assert!(
            u16::MAX as usize >= iris_mpc_common::iris_db::iris::IrisCodeArray::IRIS_CODE_SIZE
        );
        Self {
            network,
            prf: Prf::default(),
        }
    }

    pub fn get_party_id(&self) -> PartyID {
        self.network.get_id()
    }

    pub(crate) fn bytes_to_seed(mut bytes: BytesMut) -> Result<PrfSeed, Error> {
        if bytes.len() != std::mem::size_of::<PrfSeed>() {
            Err(eyre!("InvalidMessageSize"))
        } else {
            let mut their_seed: PrfSeed = PrfSeed::default();
            bytes.copy_to_slice(&mut their_seed);
            Ok(their_seed)
        }
    }

    pub async fn setup_prf(&mut self) -> Result<(), Error> {
        let seed = Prf::gen_seed();
        let data = Bytes::from_iter(seed.into_iter());
        self.network.send_next_id(data).await?;
        let response = self.network.receive_prev_id().await?;
        let their_seed = Self::bytes_to_seed(response)?;
        self.prf = Prf::new(seed, their_seed);
        Ok(())
    }

    pub(crate) fn rep3_get_cmp_diff(&self, dot: &mut Share<u16>, mask_ones: usize) {
        let threshold: u16 = ((mask_ones as f64 * (1. - 2. * MATCH_THRESHOLD_RATIO)) as usize)
            .try_into()
            .expect("Sizes are checked in constructor");
        *dot = dot.sub_from_const(threshold, self.network.get_id());
    }

    pub(crate) fn shamir_get_cmp_diff(&self, dot: &mut RingElement<u16>, mask_ones: usize) {
        let threshold: u16 = ((mask_ones as f64 * (1. - 2. * MATCH_THRESHOLD_RATIO)) as usize)
            .try_into()
            .expect("Sizes are checked in constructor");
        *dot = RingElement(threshold) - *dot;
    }

    pub fn combine_masks(mask_a: &IrisCodeArray, mask_b: &IrisCodeArray) -> IrisCodeArray {
        *mask_a & *mask_b
    }

    pub fn shamir_to_rep3_many(
        &mut self,
        inp: Vec<RingElement<u16>>,
    ) -> Result<VecShare<u16>, Error> {
        let len = inp.len();
        let shares_a = inp;
        let bytes = Utils::blocking_send_slice_and_receive(&mut self.network, &shares_a)?;
        let shares_b = Utils::ring_iter_from_bytes(bytes, len)?;
        let res = VecShare::from_avec_biter(shares_a, shares_b);
        Ok(res)
    }

    pub fn shamir_to_rep3(&mut self, inp: RingElement<u16>) -> Result<Share<u16>, Error> {
        let share_a = inp;
        let bytes_to_send = Utils::ring_to_bytes(&share_a);
        let response = Utils::blocking_send_and_receive(&mut self.network, bytes_to_send)?;
        let share_b = Utils::ring_from_bytes(response)?;

        let res = Share::new(share_a, share_b);
        Ok(res)
    }

    pub fn rep3_compare_iris_public_mask(
        &mut self,
        a: SliceShare<'_, u16>,
        b: SliceShare<'_, u16>,
        mask_a: &IrisCodeArray,
        mask_b: &IrisCodeArray,
    ) -> Result<Share<Bit>, Error> {
        let mask = Self::combine_masks(mask_a, mask_b);
        let mask_len = mask.count_ones();

        let mut dot = self.rep3_dot(a, b)?;

        // a < b <=> msb(a - b)
        // Given no overflow, which is enforced in constructor
        self.rep3_get_cmp_diff(&mut dot, mask_len);

        self.single_extract_msb_u16::<{ u16::BITS as usize }>(dot)
    }

    pub fn rep3_compare_iris_public_mask_many(
        &mut self,
        a: SliceShare<'_, u16>,
        b: &[VecShare<u16>],
        mask_a: &IrisCodeArray,
        mask_b: &[IrisCodeArray],
    ) -> Result<VecShare<u64>, Error> {
        let amount = b.len();
        if (amount != mask_b.len()) || (amount == 0) {
            return Err(eyre!("InvalidSize"));
        }

        let masks = mask_b
            .iter()
            .map(|b| Self::combine_masks(mask_a, b))
            .collect::<Vec<_>>();
        let mask_lens: Vec<_> = masks.iter().map(|m| m.count_ones()).collect();

        let mut dots = self.rep3_dot_many(a, b)?;

        // a < b <=> msb(a - b)
        // Given no overflow, which is enforced in constructor
        for (dot, mask_len) in dots.iter_mut().zip(mask_lens) {
            self.rep3_get_cmp_diff(dot, mask_len);
        }

        self.extract_msb_u16::<{ u16::BITS as usize }>(dots)
    }

    pub fn rep3_compare_iris_private_mask(
        &mut self,
        a: SliceShare<'_, u16>,
        b: SliceShare<'_, u16>,
        mask_a: SliceShare<'_, u16>,
        mask_b: SliceShare<'_, u16>,
    ) -> Result<Share<Bit>, Error> {
        let amount = b.len();
        if (amount != mask_b.len()) || (amount == 0) {
            return Err(eyre!("InvalidSize"));
        }

        let code_dots = self.rep3_dot(a, b)?;
        let mask_dots = self.rep3_dot(mask_a, mask_b)?;

        // Compute code_dots > a/b * mask_dots
        // via MSB(a * mask_dots - b * code_dots)
        self.compare_threshold_masked(code_dots, mask_dots)
    }

    pub fn rep3_compare_iris_private_mask_many(
        &mut self,
        a: SliceShare<'_, u16>,
        b: &[VecShare<u16>],
        mask_a: SliceShare<'_, u16>,
        mask_b: &[VecShare<u16>],
    ) -> Result<VecShare<u64>, Error> {
        let amount = b.len();
        if (amount != mask_b.len()) || (amount == 0) {
            return Err(eyre!("InvalidSize"));
        }

        let code_dots = self.rep3_dot_many(a, b)?;
        let mask_dots = self.rep3_dot_many(mask_a, mask_b)?;

        // Compute code_dots > a/b * mask_dots
        // via MSB(a * mask_dots - b * code_dots)
        self.compare_threshold_masked_many(code_dots, mask_dots)
    }

    pub async fn rep3_distance_non_blocking(
        &mut self,
        x: &VecShare<u16>,
        y: &VecShare<u16>,
    ) -> Result<Share<u16>, Error> {
        self.rep3_dot_non_blocking(x, y).await
    }

    pub async fn rep3_pairwise_distance(
        &mut self,
        lhs_shares: &VecShare<u16>,
        rhs_shares: &VecShare<u16>,
        lhs_mask: &IrisCodeArray,
        rhs_mask: &IrisCodeArray,
    ) -> Result<(Share<u16>, usize), Error> {
        let combined_mask = Self::combine_masks(lhs_mask, rhs_mask);
        let mask_dots = combined_mask.count_ones();
        let code_dots = self.rep3_dot_non_blocking(lhs_shares, rhs_shares).await?;
        Ok((code_dots, mask_dots))
    }

    pub fn rep3_compute_cross_mul(
        &mut self,
        d1: Share<u16>,
        t1: u32,
        d2: Share<u16>,
        t2: u32,
    ) -> Result<Share<u32>, Error> {
        let protocol = self;

        let mut vd1 = VecShare::<u16>::with_capacity(1);
        // Do preprocessing to lift d1
        vd1.push(d1);
        // Compute (d1 + 2^{15}) % 2^{16}
        for x in vd1.iter_mut() {
            x.add_assign_const(1_u16 << 15, protocol.get_party_id());
        }
        let mut lifted_d1 = protocol.lift::<16_usize>(vd1)?;
        // Now we got shares of d1' over 2^32 such that d1' = (d1'_1 + d1'_2 + d1'_3) %
        // 2^{16} = d1 Next we subtract the 2^15 term we've added previously to
        // get signed shares over 2^{32}
        for x in lifted_d1.iter_mut() {
            x.add_assign_const(
                ((1_u64 << 32) - (1_u64 << 15)) as u32,
                protocol.get_party_id(),
            );
        }

        // Compute d1 * t2
        for x in lifted_d1.iter_mut() {
            *x *= t2;
        }

        // Do preprocessing to lift d2
        let mut vd2 = VecShare::<u16>::with_capacity(1);
        vd2.push(d2);
        // Same process for d2, compute (d2 + 2^{15}) % 2^{16}
        for x in vd2.iter_mut() {
            x.add_assign_const(1_u16 << 15, protocol.get_party_id());
        }
        let mut lifted_d2 = protocol.lift::<16_usize>(vd2)?;
        // Now get rid of the 2^{15} term to get signed shares over 2^{32}
        for x in lifted_d2.iter_mut() {
            x.add_assign_const(
                ((1_u64 << 32) - (1_u64 << 15)) as u32,
                protocol.get_party_id(),
            );
        }
        // Compute d2 * t1
        for x in lifted_d2.iter_mut() {
            *x *= t1;
        }
        // Compute d2*t1 - d1*t2
        lifted_d2.sub_assign(lifted_d1);
        Ok(lifted_d2.get_at(0))
    }

    pub fn rep3_lift_and_cross_mul(
        &mut self,
        d1: Share<u16>,
        t1: u32,
        d2: Share<u16>,
        t2: u32,
    ) -> Result<bool, Error> {
        let diff = self.rep3_compute_cross_mul(d1, t1, d2, t2)?;
        let protocol = self;
        // Compute bit <- MSB(D2 * T1 - D1 * T2)
        let bit = protocol.single_extract_msb_u32::<32>(diff)?;

        // Open bit
        let opened_b = protocol.open_bin(bit)?;
        Ok(opened_b.convert())
    }

    pub fn rep3_single_iris_match_public_output(
        &mut self,
        iris_to_match: SliceShare<'_, u16>,
        ground_truth: VecShare<u16>,
        mask_iris: &IrisCodeArray,
        mask_ground_truth: IrisCodeArray,
    ) -> Result<bool, Error> {
        let res =
            self.rep3_compare_iris_public_mask_many(iris_to_match, &[ground_truth], mask_iris, &[
                mask_ground_truth,
            ])?;
        let bit = self.open_t_many(res)?;
        Ok(bit[0] != 0)
    }

    pub fn shamir_compare_iris_public_mask(
        &mut self,
        a: &mut GaloisRingIrisCodeShare,
        b: &GaloisRingIrisCodeShare,
        mask_a: &IrisCodeArray,
        mask_b: &IrisCodeArray,
    ) -> Result<Share<Bit>, Error> {
        // We have to add the lagrange coefficient here
        a.preprocess_iris_code_query_share();

        let mask = Self::combine_masks(mask_a, mask_b);
        let mask_len = mask.count_ones();

        let mut dot = self.shamir_dot(a, b)?;

        // a < b <=> msb(a - b)
        // Given no overflow, which is enforced in constructor
        self.shamir_get_cmp_diff(&mut dot, mask_len);

        // Network: reshare
        let dot = self.shamir_to_rep3(dot)?;

        self.single_extract_msb_u16::<{ u16::BITS as usize }>(dot)
    }

    pub fn shamir_compare_iris_public_mask_many(
        &mut self,
        a: &mut GaloisRingIrisCodeShare,
        b: &[GaloisRingIrisCodeShare],
        mask_a: &IrisCodeArray,
        mask_b: &[IrisCodeArray],
    ) -> Result<VecShare<u64>, Error> {
        let amount = b.len();
        if (amount != mask_b.len()) || (amount == 0) {
            return Err(eyre!("InvalidSize"));
        }

        // We have to add the lagrange coefficient here
        a.preprocess_iris_code_query_share();

        let masks = mask_b
            .iter()
            .map(|b| Self::combine_masks(mask_a, b))
            .collect::<Vec<_>>();
        let mask_lens: Vec<_> = masks.iter().map(|m| m.count_ones()).collect();

        let mut dots = self.shamir_dot_many(a, b)?;

        // a < b <=> msb(a - b)
        // Given no overflow, which is enforced in constructor
        for (dot, mask_len) in dots.iter_mut().zip(mask_lens) {
            self.shamir_get_cmp_diff(dot, mask_len);
        }

        // Network: reshare
        let dots = self.shamir_to_rep3_many(dots)?;

        self.extract_msb_u16::<{ u16::BITS as usize }>(dots)
    }

    pub fn shamir_compare_iris_private_mask(
        &mut self,
        a: &mut GaloisRingIrisCodeShare,
        b: &GaloisRingIrisCodeShare,
        mask_a: &mut GaloisRingIrisCodeShare,
        mask_b: &GaloisRingIrisCodeShare,
    ) -> Result<Share<Bit>, Error> {
        // We have to add the lagrange coefficient here
        a.preprocess_iris_code_query_share();
        mask_a.preprocess_iris_code_query_share();

        let code_dot = self.shamir_dot(a, b)?;
        let mask_dot = self.shamir_dot(mask_a, mask_b)?;

        // Network: reshare
        let code_dot = self.shamir_to_rep3(code_dot)?;
        let mask_dot = self.shamir_to_rep3(mask_dot)?;

        // Compute code_dots > a/b * mask_dots
        // via MSB(a * mask_dots - b * code_dots)
        self.compare_threshold_masked(code_dot, mask_dot)
    }

    pub fn shamir_compare_iris_private_mask_many(
        &mut self,
        a: &mut GaloisRingIrisCodeShare,
        b: &[GaloisRingIrisCodeShare],
        mask_a: &mut GaloisRingIrisCodeShare,
        mask_b: &[GaloisRingIrisCodeShare],
    ) -> Result<VecShare<u64>, Error> {
        let amount = b.len();
        if (amount != mask_b.len()) || (amount == 0) {
            return Err(eyre!("InvalidSize"));
        }

        // We have to add the lagrange coefficient here
        a.preprocess_iris_code_query_share();
        mask_a.preprocess_iris_code_query_share();

        let code_dots = self.shamir_dot_many(a, b)?;
        let mask_dots = self.shamir_dot_many(mask_a, mask_b)?;

        // Network: reshare
        let code_dots = self.shamir_to_rep3_many(code_dots)?;
        let mask_dots = self.shamir_to_rep3_many(mask_dots)?;

        // Compute code_dots > a/b * mask_dots
        // via MSB(a * mask_dots - b * code_dots)
        self.compare_threshold_masked_many(code_dots, mask_dots)
    }

    pub fn open<T: IntRing2k>(&mut self, share: Share<T>) -> Result<T, Error> {
        let c = Utils::blocking_send_and_receive_value(&mut self.network, &share.b)?;
        Ok((share.a + share.b + c).convert())
    }

    pub fn open_t_many<T: IntRing2k>(&mut self, shares: VecShare<T>) -> Result<Vec<T>, Error> {
        let shares_b = shares.iter().map(|s| &s.b);
        let bytes = Utils::blocking_send_iter_and_receive(&mut self.network, shares_b)?;
        let shares_c = Utils::ring_iter_from_bytes(bytes, shares.len())?;
        let res = shares
            .into_iter()
            .zip(shares_c)
            .map(|(s, c)| {
                let (a, b) = s.get_ab();
                (a + b + c).convert()
            })
            .collect();
        Ok(res)
    }

    pub async fn open_async<T: IntRing2k>(
        &mut self,
        share: Share<T>,
    ) -> Result<RingElement<T>, Error> {
        let bytes_to_send = Utils::ring_to_bytes(&share.b);
        let _ = self.network.send_next_id(bytes_to_send).await;
        let response = self.network.receive_prev_id().await?;
        let (a, b) = share.get_ab();
        Ok(a + b + Utils::ring_from_bytes(response)?)
    }

    pub async fn open_async_t_many<T: IntRing2k>(
        &mut self,
        shares: VecShare<T>,
    ) -> Result<Vec<RingElement<T>>, Error> {
        let n = shares.len();
        let shares_b: Vec<_> = shares.iter().map(|s| s.b).collect();
        let bytes_to_send = Utils::ring_slice_to_bytes(&shares_b);

        self.network.send_next_id(bytes_to_send).await?;
        let response = self.network.receive_prev_id().await?;
        let shares_c = Utils::ring_iter_from_bytes(response, n)?;

        let res = shares
            .into_iter()
            .zip(shares_c)
            .map(|(s, c)| {
                let (a, b) = s.get_ab();
                a + b + c
            })
            .collect();
        Ok(res)
    }
}
