use super::iris::WorkerThread;
use crate::{
    error::Error,
    networks::network_trait::NetworkTrait,
    protocol::iris::{A, A_BITS, B_BITS},
    shares::{
        bit::Bit,
        int_ring::IntRing2k,
        ring_impl::RingElement,
        share::Share,
        vecshare::{SliceShare, VecShare},
    },
    utils::Utils,
};
use iris_mpc_common::id::PartyID;
use num_traits::{One, Zero};
use rand::{distributions::Standard, prelude::Distribution, Rng};
use std::ops::SubAssign;

impl<N: NetworkTrait> WorkerThread<N> {
    pub(crate) fn mul_lift_2k<T: IntRing2k, const K: u64>(vals: SliceShare<T>) -> VecShare<u32>
    where
        u32: From<T>,
    {
        VecShare::new_vec(
            vals.iter()
                .map(|val| {
                    let a = (u32::from(val.a.0)) << K;
                    let b = (u32::from(val.b.0)) << K;
                    Share::new(RingElement(a), RingElement(b))
                })
                .collect(),
        )
    }

    pub(crate) fn a2b_pre<T: IntRing2k>(&self, x: Share<T>) -> (Share<T>, Share<T>, Share<T>) {
        let (a, b) = x.get_ab();

        let mut x1 = Share::zero();
        let mut x2 = Share::zero();
        let mut x3 = Share::zero();

        match self.network.get_id() {
            PartyID::ID0 => {
                x1.a = a;
                x3.b = b;
            }
            PartyID::ID1 => {
                x2.a = a;
                x1.b = b;
            }
            PartyID::ID2 => {
                x3.a = a;
                x2.b = b;
            }
        }
        (x1, x2, x3)
    }

    pub(crate) fn and_many_send<T: IntRing2k>(
        &mut self,
        a: SliceShare<'_, T>,
        b: SliceShare<'_, T>,
    ) -> Result<Vec<RingElement<T>>, Error>
    where
        Standard: Distribution<T>,
    {
        if a.len() != b.len() {
            return Err(Error::InvalidSize);
        }

        let mut shares_a = Vec::with_capacity(a.len());
        for (a_, b_) in a.iter().zip(b.iter()) {
            let rand = self.prf.gen_binary_zero_share::<T>();
            let mut c = a_ & b_;
            c ^= rand;
            shares_a.push(c);
        }

        self.network
            .blocking_send_next_id(Utils::ring_slice_to_bytes(&shares_a))?;
        Ok(shares_a)
    }

    pub(crate) fn and_many_receive<T: IntRing2k>(
        &mut self,
        shares_a: Vec<RingElement<T>>,
    ) -> Result<VecShare<T>, Error>
    where
        Standard: Distribution<T>,
    {
        let len = shares_a.len();
        let response = self.network.blocking_receive_prev_id()?;
        let shares_b = Utils::ring_iter_from_bytes(response, len)?;

        let res = VecShare::from_avec_biter(shares_a, shares_b);
        Ok(res)
    }

    pub(crate) fn and_many<T: IntRing2k>(
        &mut self,
        a: SliceShare<'_, T>,
        b: SliceShare<'_, T>,
    ) -> Result<VecShare<T>, Error>
    where
        Standard: Distribution<T>,
    {
        let shares_a = self.and_many_send(a, b)?;
        self.and_many_receive(shares_a)
    }

    pub(crate) fn transposed_pack_and_send<T: IntRing2k>(
        &mut self,
        x1: Vec<VecShare<T>>,
        x2: Vec<VecShare<T>>,
    ) -> Result<Vec<Vec<RingElement<T>>>, Error>
    where
        Standard: Distribution<T>,
    {
        let len = x1.len();
        debug_assert_eq!(len, x2.len());

        let mut send = Vec::with_capacity(len);
        for (x1, x2) in x1.iter().zip(x2.iter()) {
            let send_ = self.and_many_send(x1.as_slice(), x2.as_slice())?;
            send.push(send_);
        }
        Ok(send)
    }

    pub(crate) fn transposed_pack_and_receive<T: IntRing2k>(
        &mut self,
        shares_a: Vec<Vec<RingElement<T>>>,
    ) -> Result<Vec<VecShare<T>>, Error>
    where
        Standard: Distribution<T>,
    {
        let mut x3 = Vec::with_capacity(shares_a.len());

        for shares_a_ in shares_a {
            let res = self.and_many_receive(shares_a_)?;
            x3.push(res);
        }

        Ok(x3)
    }

    pub(crate) fn transposed_pack_and<T: IntRing2k>(
        &mut self,
        x1: Vec<VecShare<T>>,
        x2: Vec<VecShare<T>>,
    ) -> Result<Vec<VecShare<T>>, Error>
    where
        Standard: Distribution<T>,
    {
        let send = self.transposed_pack_and_send(x1, x2)?;
        self.transposed_pack_and_receive(send)
    }

    pub(crate) fn transposed_pack_xor<T: IntRing2k>(
        x1: &[VecShare<T>],
        x2: &[VecShare<T>],
    ) -> Vec<VecShare<T>> {
        let len = x1.len();
        debug_assert_eq!(len, x2.len());

        let mut res = Vec::with_capacity(len);
        for (x1, x2) in x1.iter().zip(x2.iter()) {
            res.push(x1.as_slice() ^ x2.as_slice());
        }
        res
    }

    pub(crate) fn transposed_pack_xor_assign<T: IntRing2k>(
        x1: &mut [VecShare<T>],
        x2: &[VecShare<T>],
    ) {
        let len = x1.len();
        debug_assert_eq!(len, x2.len());

        for (x1, x2) in x1.iter_mut().zip(x2.iter()) {
            *x1 ^= x2.as_slice();
        }
    }

    fn bit_inject_ot_2round_sender<T: IntRing2k>(
        &mut self,
        input: VecShare<Bit>,
    ) -> Result<VecShare<T>, Error>
    where
        Standard: Distribution<T>,
    {
        let len = input.len();
        let mut m0 = Vec::with_capacity(len);
        let mut m1 = Vec::with_capacity(len);
        let mut shares = VecShare::with_capacity(len);

        for inp in input.into_iter() {
            let (a, b) = inp.get_ab();
            // new shares
            let (c3, c2) = self.prf.gen_rands::<RingElement<T>>();
            // mask of the ot
            let w0 = self.prf.get_my_prf().gen::<RingElement<T>>();
            let w1 = self.prf.get_my_prf().gen::<RingElement<T>>();

            shares.push(Share::new(c3, c2));
            let c = c3 + c2;
            let xor = RingElement(T::from((a ^ b).convert().convert()));
            let m0_ = xor - c;
            let m1_ = (xor ^ RingElement::one()) - c;
            m0.push(m0_ ^ w0);
            m1.push(m1_ ^ w1);
        }
        self.network
            .blocking_send_prev_id(Utils::ring_slice_to_bytes(&m0))?;
        self.network
            .blocking_send_prev_id(Utils::ring_slice_to_bytes(&m1))?;

        Ok(shares)
    }

    fn bit_inject_ot_2round_receiver<T: IntRing2k>(
        &mut self,
        input: VecShare<Bit>,
    ) -> Result<VecShare<T>, Error>
    where
        Standard: Distribution<T>,
    {
        let len = input.len();
        let m0_bytes = self.network.blocking_receive_next_id()?;
        let m1_bytes = self.network.blocking_receive_next_id()?;
        let wc_bytes = self.network.blocking_receive_prev_id()?;

        let m0 = Utils::ring_iter_from_bytes(m0_bytes, len)?;
        let m1 = Utils::ring_iter_from_bytes(m1_bytes, len)?;

        let wc = Utils::ring_iter_from_bytes(wc_bytes, len)?;

        let mut shares = VecShare::with_capacity(len);
        let mut send = Vec::with_capacity(len);

        for ((inp, wc), (m0, m1)) in input.into_iter().zip(wc).zip(m0.zip(m1)) {
            // new share
            let c2 = self.prf.get_my_prf().gen::<RingElement<T>>();

            let choice = inp.get_b().convert().convert();
            let xor = if choice { wc ^ m1 } else { wc ^ m0 };

            send.push(xor);
            shares.push(Share::new(c2, xor));
        }

        // Reshare to Helper
        self.network
            .blocking_send_prev_id(Utils::ring_slice_to_bytes(&send))?;

        Ok(shares)
    }

    fn bit_inject_ot_2round_helper<T: IntRing2k>(
        &mut self,
        input: VecShare<Bit>,
    ) -> Result<VecShare<T>, Error>
    where
        Standard: Distribution<T>,
    {
        let len = input.len();
        let mut wc = Vec::with_capacity(len);
        let mut shares = VecShare::with_capacity(len);

        for inp in input.into_iter() {
            // new share
            let c3 = self.prf.get_prev_prf().gen::<RingElement<T>>();
            shares.push(Share::new(RingElement::zero(), c3));

            // mask of the ot
            let w0 = self.prf.get_prev_prf().gen::<RingElement<T>>();
            let w1 = self.prf.get_prev_prf().gen::<RingElement<T>>();

            let choice = inp.get_a().convert().convert();
            if choice {
                wc.push(w1);
            } else {
                wc.push(w0);
            }
        }
        self.network
            .blocking_send_next_id(Utils::ring_slice_to_bytes(&wc))?;

        // Receive Reshare
        let c1_bytes = self.network.blocking_receive_next_id()?;
        let c1 = Utils::ring_iter_from_bytes(c1_bytes, len)?;

        for (s, c1) in shares.iter_mut().zip(c1) {
            s.a = c1;
        }
        Ok(shares)
    }

    // TODO this is inbalanced, so a real implementation should actually rotate
    // parties around
    pub(crate) fn bit_inject_ot_2round<T: IntRing2k>(
        &mut self,
        input: VecShare<Bit>,
    ) -> Result<VecShare<T>, Error>
    where
        Standard: Distribution<T>,
    {
        let res = match self.get_party_id() {
            PartyID::ID0 => {
                // OT Helper
                self.bit_inject_ot_2round_helper(input)?
            }
            PartyID::ID1 => {
                // OT Receiver
                self.bit_inject_ot_2round_receiver(input)?
            }
            PartyID::ID2 => {
                // OT Sender
                self.bit_inject_ot_2round_sender(input)?
            }
        };
        Ok(res)
    }

    pub fn extract_msb_u16<const K: usize>(
        &mut self,
        x_: VecShare<u16>,
    ) -> Result<VecShare<u64>, Error> {
        // let truncate_len = x_.len();
        let x = x_.transpose_pack_u64_with_len::<K>();
        self.extract_msb::<K>(x)
    }

    pub fn extract_msb_u32<const K: usize>(
        &mut self,
        x_: VecShare<u32>,
    ) -> Result<VecShare<u64>, Error> {
        // let truncate_len = x_.len();
        let x = x_.transpose_pack_u64_with_len::<K>();
        self.extract_msb::<K>(x)
    }

    // Extracts bit at position K
    fn extract_msb<const K: usize>(
        &mut self,
        x: Vec<VecShare<u64>>,
    ) -> Result<VecShare<u64>, Error> {
        let len = x.len();

        let mut x1 = Vec::with_capacity(len);
        let mut x2 = Vec::with_capacity(len);
        let mut x3 = Vec::with_capacity(len);

        for x_ in x.into_iter() {
            let len_ = x_.len();
            let mut x1_ = VecShare::with_capacity(len_);
            let mut x2_ = VecShare::with_capacity(len_);
            let mut x3_ = VecShare::with_capacity(len_);
            for x__ in x_.into_iter() {
                let (x1__, x2__, x3__) = self.a2b_pre(x__);
                x1_.push(x1__);
                x2_.push(x2__);
                x3_.push(x3__);
            }
            x1.push(x1_);
            x2.push(x2_);
            x3.push(x3_);
        }

        self.binary_add_3_get_msb(x1, x2, x3)
    }

    fn binary_add_3_get_two_carries<T: IntRing2k>(
        &mut self,
        x1: Vec<VecShare<T>>,
        x2: Vec<VecShare<T>>,
        x3: Vec<VecShare<T>>,
        truncate_len: usize,
    ) -> Result<(VecShare<Bit>, VecShare<Bit>), Error>
    where
        Standard: Distribution<T>,
    {
        let len = x1.len();
        debug_assert!(len == x2.len() && len == x3.len());

        // Full adder to get 2 * c and s
        let mut x2x3 = x2;
        Self::transposed_pack_xor_assign(&mut x2x3, &x3);
        let s = Self::transposed_pack_xor(&x1, &x2x3);
        let mut x1x3 = x1;
        Self::transposed_pack_xor_assign(&mut x1x3, &x3);
        let mut c = self.transposed_pack_and(x1x3, x2x3)?;
        Self::transposed_pack_xor_assign(&mut c, &x3);

        // Add 2c + s via a ripple carry adder
        // LSB of c is 0
        // First round: half adder can be skipped due to LSB of c being 0
        let mut a = s;
        let mut b = c;

        // First full adder (carry is 0)
        let mut c = self.and_many(a[1].as_slice(), b[0].as_slice())?;

        // For last round
        let mut b_msb = b.pop().expect("Enough elements present");

        // 2 -> k
        for (a_, b_) in a.iter_mut().skip(2).zip(b.iter_mut().skip(1)) {
            *a_ ^= c.as_slice();
            *b_ ^= c.as_slice();
            let tmp_c = self.and_many(a_.as_slice(), b_.as_slice())?;
            c ^= tmp_c;
        }

        // Finally, last bit of a is 0
        let res2 = self.and_many(b_msb.as_slice(), c.as_slice())?;
        b_msb ^= c;

        // Extract bits for outputs
        let mut res1 = b_msb.convert_to_bits();
        res1.truncate(truncate_len);
        let mut res2 = res2.convert_to_bits();
        res2.truncate(truncate_len);

        Ok((res1, res2))
    }

    pub(crate) fn binary_add_3_get_msb<T: IntRing2k>(
        &mut self,
        x1: Vec<VecShare<T>>,
        x2: Vec<VecShare<T>>,
        mut x3: Vec<VecShare<T>>,
        // truncate_len: usize,
    ) -> Result<VecShare<T>, Error>
    where
        Standard: Distribution<T>,
    {
        let len = x1.len();
        debug_assert!(len == x2.len() && len == x3.len());

        // Full adder to get 2 * c and s
        let mut x2x3 = x2;
        Self::transposed_pack_xor_assign(&mut x2x3, &x3);
        let s = Self::transposed_pack_xor(&x1, &x2x3);
        let mut x1x3 = x1;
        Self::transposed_pack_xor_assign(&mut x1x3, &x3);
        // 2 * c
        x1x3.pop().expect("Enough elements present");
        x2x3.pop().expect("Enough elements present");
        x3.pop().expect("Enough elements present");
        let mut c = self.transposed_pack_and(x1x3, x2x3)?;
        Self::transposed_pack_xor_assign(&mut c, &x3);

        // Add 2c + s via a ripple carry adder
        // LSB of c is 0
        // First round: half adder can be skipped due to LSB of c being 0
        let mut a = s;
        let mut b = c;

        // First full adder (carry is 0)
        let mut c = self.and_many(a[1].as_slice(), b[0].as_slice())?;

        // For last round
        let mut a_msb = a.pop().expect("Enough elements present");
        let b_msb = b.pop().expect("Enough elements present");

        // 2 -> k-1
        for (a_, b_) in a.iter_mut().skip(2).zip(b.iter_mut().skip(1)) {
            *a_ ^= c.as_slice();
            *b_ ^= c.as_slice();
            let tmp_c = self.and_many(a_.as_slice(), b_.as_slice())?;
            c ^= tmp_c;
        }

        a_msb ^= b_msb;
        a_msb ^= c;

        // Extract bits for outputs
        let res = a_msb;
        // let mut res = a_msb.convert_to_bits();
        // res.truncate(truncate_len);

        Ok(res)
    }

    pub(crate) fn transposed_padded_len(len: usize) -> usize {
        let padded_len = (len + 63) / 64;
        padded_len * 64
    }

    pub(crate) fn lift<const K: usize>(
        &mut self,
        shares: VecShare<u16>,
    ) -> Result<VecShare<u32>, Error> {
        let len = shares.len();
        let padded_len = Self::transposed_padded_len(len);

        let mut x_a = VecShare::with_capacity(padded_len);
        for share in shares.iter() {
            x_a.push(Share::new(
                RingElement(share.a.0 as u32),
                RingElement(share.b.0 as u32),
            ));
        }

        let x = shares.transpose_pack_u64();

        let len_ = x.len();
        let mut x1 = Vec::with_capacity(len_);
        let mut x2 = Vec::with_capacity(len_);
        let mut x3 = Vec::with_capacity(len_);

        for x_ in x.into_iter() {
            let len__ = x_.len();
            let mut x1_ = VecShare::with_capacity(len__);
            let mut x2_ = VecShare::with_capacity(len__);
            let mut x3_ = VecShare::with_capacity(len__);
            for x__ in x_.into_iter() {
                let (x1__, x2__, x3__) = self.a2b_pre(x__);
                x1_.push(x1__);
                x2_.push(x2__);
                x3_.push(x3__);
            }
            x1.push(x1_);
            x2.push(x2_);
            x3.push(x3_);
        }

        let (mut b1, b2) = self.binary_add_3_get_two_carries(x1, x2, x3, len)?;

        b1.extend(b2);

        // We need bit1 * 2^{ShareRing::K+1} mod 2^{ShareRing::K+K}  and bit2 *
        // 2^{ShareRing::K+1} mod 2^{ShareRing::K+K} So we inject bit to mod
        // 2^{K-1} and mod 2^{K-2} and use the mul_lift_2k function TODO: This
        // one is not optimized: We send too much, since we need less than K
        // bits
        debug_assert!(K <= 16); // otherwise u16 does not work
        let mut b = self.bit_inject_ot_2round::<u16>(b1)?;
        let (b1, b2) = b.split_at_mut(len);

        // Make the result mod 2^{K-1} and mod 2^{K-2} (Not required since we bitextract
        // the correct one later) Self::share_bit_mod(&mut b1, K as u32);
        // Self::share_bit_mod(&mut b2, K as u32 - 1);

        let b1 = Self::mul_lift_2k::<_, { u16::K as u64 }>(b1.to_slice());
        let b2 = Self::mul_lift_2k::<_, { u16::K as u64 + 1 }>(b2.to_slice());

        // Finally, compute the result
        x_a.sub_assign(b1);
        x_a.sub_assign(b2);
        Ok(x_a)
    }

    // Compute code_dots > a/b * mask_dots
    // via MSB(a * mask_dots - b * code_dots)
    pub fn compare_threshold_masked_many(
        &mut self,
        code_dots: VecShare<u16>,
        mask_dots: VecShare<u16>,
    ) -> Result<VecShare<u64>, Error> {
        debug_assert!(A_BITS as u64 <= B_BITS);
        let len = code_dots.len();
        assert_eq!(len, mask_dots.len());

        let y = Self::mul_lift_2k::<_, B_BITS>(code_dots.as_slice());
        let mut x = self.lift::<{ B_BITS as usize }>(mask_dots)?;
        for x_ in x.iter_mut() {
            *x_ *= A as u32;
        }

        x.sub_assign(y);
        self.extract_msb_u32::<{ u16::K + B_BITS as usize }>(x)
    }
}
