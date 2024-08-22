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

impl<N: NetworkTrait> WorkerThread<N> {
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

    // Extracts bit at position K
    pub fn extract_msb<const K: usize>(
        &mut self,
        x_: VecShare<u32>,
    ) -> Result<VecShare<u64>, Error> {
        // let truncate_len = x_.len();
        let x = x_.transpose_pack_u64_with_len::<K>();

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
        transposed_pack_xor_assign(&mut x2x3, &x3);
        let s = transposed_pack_xor(&x1, &x2x3);
        let mut x1x3 = x1;
        transposed_pack_xor_assign(&mut x1x3, &x3);
        // 2 * c
        x1x3.pop().expect("Enough elements present");
        x2x3.pop().expect("Enough elements present");
        x3.pop().expect("Enough elements present");
        let mut c = self.transposed_pack_and(x1x3, x2x3)?;
        transposed_pack_xor_assign(&mut c, &x3);

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
        self.extract_msb::<{ u16::K + B_BITS as usize }>(x)
    }
}
