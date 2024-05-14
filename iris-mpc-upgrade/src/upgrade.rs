use aes_prng::AesRng;
use color_eyre::eyre::{bail, Error};
use itertools::{izip, Itertools};
use rand::SeedableRng;

use crate::{
    packets::{
        FpMulManyMessage, PackedBinaryAndMessage, ShamirSharesMessage, TwoToThreeIrisCodeMessage,
    },
    prf::Prf,
    shamir::{Shamir, P, P32},
    share::{RepShare, RepSharedBits},
    NewIrisShareSink, PartyID, Seed, IRIS_CODE_LEN,
};

pub type Stage1Local = Vec<RepShare<u64>>;
pub type Stage2Local = (Vec<RepShare<u16>>, Vec<RepShare<u16>>);

#[derive(Clone)]
pub struct IrisCodeUpgrader {
    seed1: Seed,
    seed2: Seed,
    party_id: PartyID,
}

impl IrisCodeUpgrader {
    /// Creates a new IrisCodeUpgrader with the given seeds and party id.
    /// The seeds are used to create the PRFs used in the protocol and seed1 of party i must be equal to seed2 of party i+1.
    ///
    pub fn new(seed1: Seed, seed2: Seed, party_id: PartyID) -> Self {
        Self {
            seed1,
            seed2,
            party_id,
        }
    }

    /// Stage one of the upgrade protocol. Takes as input two TwoToThreeIrisCodeMessage messages
    /// with the same id, sent from party 0 and 1 respectively.
    ///
    /// # Returns
    /// A tuple of ([Stage1Local], [PackedBinaryAndMessage]), the first one is to be sent to ourselves and the second one to party i+1.
    /// These are the inputs to stage2.
    pub fn stage1(
        &self,
        msg1: TwoToThreeIrisCodeMessage,
        msg2: TwoToThreeIrisCodeMessage,
    ) -> Result<(Stage1Local, PackedBinaryAndMessage), Error> {
        // TODO: sanity checks for messages
        let id = msg1.id;
        if id != msg2.id {
            bail!("mismatches message id in stage 1");
        }

        // create a new Prf based on the master seed, the message id and the stage
        let mut prf = self.make_prf(id, 1);

        let mut data1 = msg1.data;
        let data2 = msg2.data;
        data1
            .iter_mut()
            .zip(data2.into_iter())
            .for_each(|(a, b)| *a += b);
        let shares = data1;
        debug_assert_eq!(shares.len() % 64, 0);
        let len = shares.len() / 64;

        let x = Self::transpose_pack_u64_2bits(&shares);

        let mut x1: [_; 2] = core::array::from_fn(|_| Vec::with_capacity(len));
        let mut x2: [_; 2] = core::array::from_fn(|_| Vec::with_capacity(len));
        let mut x3: [_; 2] = core::array::from_fn(|_| Vec::with_capacity(len));

        for (i, x) in x.into_iter().enumerate() {
            for x_ in x.into_iter() {
                let (x1_, x2_, x3_) = x_.a2b_pre(self.party_id);
                x1[i].push(x1_);
                x2[i].push(x2_);
                x3[i].push(x3_);
            }
        }
        // Full adder to get 2 * c and s, but I just need s[1] ^ c[0]
        // let s1 = x1[1] ^ x2[1] ^ x3[1];
        // let c0 = x3[0] ^ ((x1[0] ^ x3[0]) & (x2[0] ^ x3[0]));
        let [mut x10, mut x11] = x1;
        let [mut x20, x21] = x2;
        let [x30, x31] = x3;

        // x10 -> x10^x30, x20-> x20^x30
        for (des1, des2, src) in izip!(&mut x10, &mut x20, &x30) {
            *des1 ^= src;
            *des2 ^= src;
        }
        // x10 &= x20
        let local_result = self.and_many_pre(&x10, &x20, &mut prf);

        for (des, src1, src2, src3, src4) in izip!(&mut x11, x21, x31, x30, &local_result) {
            *des ^= src1 ^ src2 ^ src3;
            // add the local result of and here, we add the remote result in stage2
            des.a ^= src4;
        }

        Ok((
            x11,
            PackedBinaryAndMessage {
                id,
                party_id: self.party_id.prev_id().into(),
                from: self.party_id.into(),
                data: local_result,
            },
        ))
    }

    /// Stage two of the upgrade protocol. Takes as input a [PackedBinaryAndMessage] and a [Stage1Local]
    /// with the same id. The message is received from the previous party.
    ///
    /// # Returns
    /// A tuple of ([Stage2Local], [FpMulManyMessage]), the first one is to be sent to ourselves and the second one to party i+1.
    /// These are the inputs to stage3.
    pub fn stage2(
        &self,
        msg_local: Stage1Local,
        msg_prev: PackedBinaryAndMessage,
    ) -> Result<(Stage2Local, FpMulManyMessage), Error> {
        // TODO: sanity checks for messages
        let id = msg_prev.id;
        let mut prf = self.make_prf(id, 2);

        let mut stage1_result = msg_local;
        // finish up binary AND
        for (dst, src) in izip!(&mut stage1_result, &msg_prev.data) {
            dst.b ^= src;
        }

        // bitinject

        let bitlen = stage1_result.len() * 64;
        assert!(bitlen == IRIS_CODE_LEN);
        let mut x1 = Vec::with_capacity(bitlen);
        let mut x2 = Vec::with_capacity(bitlen);
        let mut x3 = Vec::with_capacity(bitlen);

        let iter = RepSharedBits::new(&stage1_result);
        for (a, b) in iter.take(bitlen) {
            let (x1_, x2_, x3_) = RepShare::bitinject_pre(a, b, self.party_id);
            x1.push(x1_);
            x2.push(x2_);
            x3.push(x3_);
        }
        let local_result = self.mul_fp_many_pre(&x1, &x2, &mut prf);
        // local part of xor_arith_many
        for (a, b, &p) in izip!(x1.iter_mut(), x2, &local_result) {
            let res_a = (a.a as u32 + b.a as u32 + P32 + P32 - p as u32 - p as u32) % P32;
            let res_b = (a.b as u32 + b.b as u32) % P32; // we leave the b part as 0, and stage 3 fixes this
            a.a = res_a as u16;
            a.b = res_b as u16;
        }

        Ok((
            (x1, x3),
            FpMulManyMessage {
                id,
                party_id: self.party_id.next_id().into(),
                from: self.party_id.into(),
                data: local_result,
            },
        ))
    }

    /// Stage three of the upgrade protocol. Takes as input a [FpMulManyMessage] and a [Stage2Local]
    /// with the same id. The message is received from the previous party.
    ///
    /// # Returns
    /// A array of 3 ([ShamirSharesMessage]), where element i is to be sent to party i.
    /// These are the inputs to [Self::finalize].
    pub fn stage3(
        &self,
        msg_local: Stage2Local,
        msg_prev: FpMulManyMessage,
    ) -> Result<[ShamirSharesMessage; 3], Error> {
        // TODO: sanity checks for messages
        let id = msg_prev.id;
        let (mut x1, x3) = msg_local;
        for (a, b) in izip!(&mut x1, msg_prev.data) {
            a.b = ((a.b as u32 + P32 + P32 - b as u32 - b as u32) % P32) as u16;
        }
        let res = self.arithmetic_xor_many_rep_to_add(&x1, &x3)?;

        let len = res.len();
        let mut shares: [_; 3] = core::array::from_fn(|_| Vec::with_capacity(len));
        let mut rng = AesRng::from_entropy();
        for sh in res.iter() {
            let s = Shamir::share_d1(*sh, &mut rng);
            for (share, s) in shares.iter_mut().zip(s.into_iter()) {
                share.push(s);
            }
        }
        Ok(shares
            .into_iter()
            .enumerate()
            .map(|(i, data)| ShamirSharesMessage {
                id,
                party_id: i as u8,
                from: self.party_id.into(),
                data,
            })
            .collect_vec()
            .try_into()
            .unwrap())
    }

    /// Finalizes the upgrade protocol.
    /// Takes 3 [ShamirSharesMessages] from the 3 parties and combines them into a final shamir share.
    /// Also takes a [ShamirSharesMessage] with the masks.
    /// Both the new shared code and mask are then stored in the [NewIrisShareSink].
    pub fn finalize(
        [msg1, msg2, msg3]: [ShamirSharesMessage; 3],
        masks: ShamirSharesMessage,
        s: &impl NewIrisShareSink,
    ) -> Result<(), Error> {
        // todo: sanity checks
        let id = msg1.id;
        let mut result = msg1.data;
        let part2 = msg2.data;
        let part3 = msg3.data;
        let mask = masks.data;
        for (a, b, c, m) in izip!(result.iter_mut(), part2.iter(), part3.iter(), mask.iter()) {
            *a = ((*a as u32 + *b as u32 + *c as u32) % P32) as u16;
            *a = ((*m as u32 + P32 + P32 - *a as u32 - *a as u32) % P32) as u16;
        }

        s.store_code_share(id, result)?;
        s.store_mask_share(id, mask)?;
        Ok(())
    }

    fn share_transpose2x64(a: &[RepShare<u16>; 64]) -> [RepShare<u64>; 2] {
        let mut res: [RepShare<u64>; 2] = core::array::from_fn(|_| RepShare::default());

        // pack results into Share64 datatypes
        for (i, bb) in res.iter_mut().enumerate() {
            for j in (0..64).step_by(2) {
                bb.a |= ((a[j + i].a as u64) & 0x3) << j;
                bb.b |= ((a[j + i].b as u64) & 0x3) << j;
            }
        }

        // version of 64x64 transpose that only does the swaps needed for 2 bits
        let t = ((&res[0] >> 1) ^ &res[1]) & 0x5555555555555555;
        res[1] ^= &t;
        res[0] ^= t << 1;

        res
    }
    fn transpose_pack_u64_2bits(input: &[RepShare<u16>]) -> [Vec<RepShare<u64>>; 2] {
        debug_assert_eq!(input.len() % 64, 0);
        let len = input.len() / 64;

        let mut res = core::array::from_fn(|_| vec![RepShare::default(); len]);

        for (j, x) in input.chunks_exact(64).enumerate() {
            let trans = Self::share_transpose2x64(x.try_into().unwrap());
            for (src, des) in trans.into_iter().zip(res.iter_mut()) {
                des[j] = src;
            }
        }
        res
    }

    fn and_many_pre(&self, a: &[RepShare<u64>], b: &[RepShare<u64>], prf: &mut Prf) -> Vec<u64> {
        let len = a.len();
        debug_assert_eq!(len, b.len());
        let mut res = Vec::with_capacity(len);

        for (a, b) in a.iter().zip(b.iter()) {
            let rand: u64 = prf.gen_binary_zero_share();
            let mut c = a & b;
            c ^= rand;
            res.push(c);
        }

        res
    }

    fn mul_fp_many_pre(&self, a: &[RepShare<u16>], b: &[RepShare<u16>], prf: &mut Prf) -> Vec<u16> {
        let len = a.len();
        debug_assert_eq!(len, b.len());
        let mut res = Vec::with_capacity(len);

        for (a, b) in a.iter().zip(b.iter()) {
            let my_rand = Shamir::random_fp(prf.get_my_prf());
            let prev_rand = Shamir::random_fp(prf.get_prev_prf());
            let c = (a.a as u64 * b.a as u64)
                + (a.b as u64 * b.a as u64)
                + (a.a as u64 * b.b as u64)
                + my_rand as u64
                + P as u64
                - prev_rand as u64;
            res.push((c % P as u64) as u16);
        }

        res
    }

    fn mul_fp_many_pre_without_rand(a: &[RepShare<u16>], b: &[RepShare<u16>]) -> Vec<u16> {
        let len = a.len();
        debug_assert_eq!(len, b.len());
        let mut res = Vec::with_capacity(len);

        for (a, b) in a.iter().zip(b.iter()) {
            let c =
                (a.a as u64 * b.a as u64) + (a.b as u64 * b.a as u64) + (a.a as u64 * b.b as u64);
            res.push((c % P as u64) as u16);
        }

        res
    }
    fn arithmetic_xor_many_rep_to_add(
        &self,
        a: &[RepShare<u16>],
        b: &[RepShare<u16>],
    ) -> Result<Vec<u16>, Error> {
        let mut prod = Self::mul_fp_many_pre_without_rand(a, b);
        for (a, b, p) in izip!(a, b, &mut prod) {
            let res_a = (a.a as u32 + b.a as u32 + P32 + P32 - *p as u32 - *p as u32) % P32;
            *p = res_a as u16;
        }
        Ok(prod)
    }

    fn make_prf(&self, id: u64, stage: u8) -> Prf {
        let seed1 = u128::from_be_bytes(self.seed1.0) + u128::from(id) << 8 + u128::from(stage);
        let seed2 = u128::from_be_bytes(self.seed2.0) + u128::from(id) << 8 + u128::from(stage);
        Prf::new(seed1.to_be_bytes(), seed2.to_be_bytes())
    }
}
