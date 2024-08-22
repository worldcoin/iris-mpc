use super::{ring_impl::RingElement, share::Share, vecshare::VecShare};

impl VecShare<u16> {
    fn share64_from_share16s(
        a: &Share<u16>,
        b: &Share<u16>,
        c: &Share<u16>,
        d: &Share<u16>,
    ) -> Share<u64> {
        let a_ = (a.a.0 as u64)
            | ((b.a.0 as u64) << 16)
            | ((c.a.0 as u64) << 32)
            | ((d.a.0 as u64) << 48);
        let b_ = (a.b.0 as u64)
            | ((b.b.0 as u64) << 16)
            | ((c.b.0 as u64) << 32)
            | ((d.b.0 as u64) << 48);

        Share {
            a: RingElement(a_),
            b: RingElement(b_),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn share128_from_share16s(
        a: &Share<u16>,
        b: &Share<u16>,
        c: &Share<u16>,
        d: &Share<u16>,
        e: &Share<u16>,
        f: &Share<u16>,
        g: &Share<u16>,
        h: &Share<u16>,
    ) -> Share<u128> {
        let a_ = (a.a.0 as u128)
            | ((b.a.0 as u128) << 16)
            | ((c.a.0 as u128) << 32)
            | ((d.a.0 as u128) << 48)
            | ((e.a.0 as u128) << 64)
            | ((f.a.0 as u128) << 80)
            | ((g.a.0 as u128) << 96)
            | ((h.a.0 as u128) << 112);
        let b_ = (a.b.0 as u128)
            | ((b.b.0 as u128) << 16)
            | ((c.b.0 as u128) << 32)
            | ((d.b.0 as u128) << 48)
            | ((e.b.0 as u128) << 64)
            | ((f.b.0 as u128) << 80)
            | ((g.b.0 as u128) << 96)
            | ((h.b.0 as u128) << 112);

        Share {
            a: RingElement(a_),
            b: RingElement(b_),
        }
    }

    fn share_transpose16x128(a: &[Share<u16>; 128]) -> [Share<u128>; 16] {
        let mut j: u32;
        let mut k: usize;
        let mut m: u128;
        let mut t: Share<u128>;

        let mut res = core::array::from_fn(|_| Share::default());

        // pack results into Share128 datatypes
        for (i, bb) in res.iter_mut().enumerate() {
            *bb = Self::share128_from_share16s(
                &a[i],
                &a[i + 16],
                &a[i + 32],
                &a[i + 48],
                &a[i + 64],
                &a[i + 80],
                &a[i + 96],
                &a[i + 112],
            );
        }

        // version of 128x128 transpose that only does the swaps needed for 16 bits
        m = 0x00ff00ff00ff00ff00ff00ff00ff00ff;
        j = 8;
        while j != 0 {
            k = 0;
            while k < 16 {
                t = ((&res[k] >> j) ^ &res[k + j as usize]) & m;
                res[k + j as usize] ^= &t;
                res[k] ^= t << j;
                k = (k + j as usize + 1) & !(j as usize);
            }
            j >>= 1;
            m = m ^ (m << j);
        }

        res
    }

    fn share_transpose16x64(a: &[Share<u16>; 64]) -> [Share<u64>; 16] {
        let mut j: u32;
        let mut k: usize;
        let mut m: u64;
        let mut t: Share<u64>;

        let mut res = core::array::from_fn(|_| Share::default());

        // pack results into Share64 datatypes
        for (i, bb) in res.iter_mut().enumerate() {
            *bb = Self::share64_from_share16s(&a[i], &a[16 + i], &a[32 + i], &a[48 + i]);
        }

        // version of 64x64 transpose that only does the swaps needed for 16 bits
        m = 0x00ff00ff00ff00ff;
        j = 8;
        while j != 0 {
            k = 0;
            while k < 16 {
                t = ((&res[k] >> j) ^ &res[k + j as usize]) & m;
                res[k + j as usize] ^= &t;
                res[k] ^= t << j;
                k = (k + j as usize + 1) & !(j as usize);
            }
            j >>= 1;
            m = m ^ (m << j);
        }

        res
    }

    pub fn transpose_pack_u64(self) -> Vec<VecShare<u64>> {
        self.transpose_pack_u64_with_len::<{ u16::BITS as usize }>()
    }

    pub fn transpose_pack_u64_with_len<const L: usize>(mut self) -> Vec<VecShare<u64>> {
        // Pad to multiple of 64
        let len = (self.shares.len() + 63) / 64;
        self.shares.resize(len * 64, Share::default());

        let mut res = (0..L)
            .map(|_| VecShare::new_vec(vec![Share::default(); len]))
            .collect::<Vec<_>>();

        for (j, x) in self.shares.chunks_exact(64).enumerate() {
            let trans = Self::share_transpose16x64(x.try_into().unwrap());
            for (src, des) in trans.into_iter().zip(res.iter_mut()) {
                des.shares[j] = src;
            }
        }
        debug_assert_eq!(res.len(), L);
        res
    }

    pub fn transpose_pack_u128(self) -> Vec<VecShare<u128>> {
        self.transpose_pack_u128_with_len::<{ u16::BITS as usize }>()
    }

    pub fn transpose_pack_u128_with_len<const L: usize>(mut self) -> Vec<VecShare<u128>> {
        // Pad to multiple of 128
        let len = (self.shares.len() + 127) / 128;
        self.shares.resize(len * 128, Share::default());

        let mut res = (0..L)
            .map(|_| VecShare::new_vec(vec![Share::default(); len]))
            .collect::<Vec<_>>();

        for (j, x) in self.shares.chunks_exact(128).enumerate() {
            let trans = Self::share_transpose16x128(x.try_into().unwrap());
            for (src, des) in trans.into_iter().zip(res.iter_mut()) {
                des.shares[j] = src;
            }
        }
        debug_assert_eq!(res.len(), L);
        res
    }
}

impl VecShare<u32> {
    fn share64_from_share32s(a: &Share<u32>, b: &Share<u32>) -> Share<u64> {
        let a_ = (a.a.0 as u64) | ((b.a.0 as u64) << 32);
        let b_ = (a.b.0 as u64) | ((b.b.0 as u64) << 32);

        Share {
            a: RingElement(a_),
            b: RingElement(b_),
        }
    }

    fn share128_from_share32s(
        a: &Share<u32>,
        b: &Share<u32>,
        c: &Share<u32>,
        d: &Share<u32>,
    ) -> Share<u128> {
        let a_ = (a.a.0 as u128)
            | ((b.a.0 as u128) << 32)
            | ((c.a.0 as u128) << 64)
            | ((d.a.0 as u128) << 96);
        let b_ = (a.b.0 as u128)
            | ((b.b.0 as u128) << 32)
            | ((c.b.0 as u128) << 64)
            | ((d.b.0 as u128) << 96);

        Share {
            a: RingElement(a_),
            b: RingElement(b_),
        }
    }

    fn share_transpose32x128(a: &[Share<u32>; 128]) -> [Share<u128>; 32] {
        let mut j: u32;
        let mut k: usize;
        let mut m: u128;
        let mut t: Share<u128>;

        let mut res = core::array::from_fn(|_| Share::default());

        // pack results into Share128 datatypes
        for (i, bb) in res.iter_mut().enumerate() {
            *bb = Self::share128_from_share32s(&a[i], &a[32 + i], &a[64 + i], &a[96 + i]);
        }

        // version of 128x128 transpose that only does the swaps needed for 32 bits
        m = 0x0000ffff0000ffff0000ffff0000ffff;
        j = 16;
        while j != 0 {
            k = 0;
            while k < 32 {
                t = ((&res[k] >> j) ^ &res[k + j as usize]) & m;
                res[k + j as usize] ^= &t;
                res[k] ^= t << j;
                k = (k + j as usize + 1) & !(j as usize);
            }
            j >>= 1;
            m = m ^ (m << j);
        }

        res
    }

    fn share_transpose32x64(a: &[Share<u32>; 64]) -> [Share<u64>; 32] {
        let mut j: u32;
        let mut k: usize;
        let mut m: u64;
        let mut t: Share<u64>;

        let mut res = core::array::from_fn(|_| Share::default());

        // pack results into Share64 datatypes
        for (i, bb) in res.iter_mut().enumerate() {
            *bb = Self::share64_from_share32s(&a[i], &a[32 + i]);
        }

        // version of 64x64 transpose that only does the swaps needed for 32 bits
        m = 0x0000ffff0000ffff;
        j = 16;
        while j != 0 {
            k = 0;
            while k < 32 {
                t = ((&res[k] >> j) ^ &res[k + j as usize]) & m;
                res[k + j as usize] ^= &t;
                res[k] ^= t << j;
                k = (k + j as usize + 1) & !(j as usize);
            }
            j >>= 1;
            m = m ^ (m << j);
        }

        res
    }

    pub fn transpose_pack_u64(self) -> Vec<VecShare<u64>> {
        self.transpose_pack_u64_with_len::<{ u32::BITS as usize }>()
    }

    pub fn transpose_pack_u64_with_len<const L: usize>(mut self) -> Vec<VecShare<u64>> {
        // Pad to multiple of 64
        let len = (self.shares.len() + 63) / 64;
        self.shares.resize(len * 64, Share::default());

        let mut res = (0..L)
            .map(|_| VecShare::new_vec(vec![Share::default(); len]))
            .collect::<Vec<_>>();

        for (j, x) in self.shares.chunks_exact(64).enumerate() {
            let trans = Self::share_transpose32x64(x.try_into().unwrap());
            for (src, des) in trans.into_iter().zip(res.iter_mut()) {
                des.shares[j] = src;
            }
        }
        debug_assert_eq!(res.len(), L);
        res
    }

    pub fn transpose_pack_u128(self) -> Vec<VecShare<u128>> {
        self.transpose_pack_u128_with_len::<{ u32::BITS as usize }>()
    }

    pub fn transpose_pack_u128_with_len<const L: usize>(mut self) -> Vec<VecShare<u128>> {
        // Pad to multiple of 128
        let len = (self.shares.len() + 127) / 128;
        self.shares.resize(len * 128, Share::default());

        let mut res = (0..L)
            .map(|_| VecShare::new_vec(vec![Share::default(); len]))
            .collect::<Vec<_>>();

        for (j, x) in self.shares.chunks_exact(128).enumerate() {
            let trans = Self::share_transpose32x128(x.try_into().unwrap());
            for (src, des) in trans.into_iter().zip(res.iter_mut()) {
                des.shares[j] = src;
            }
        }
        debug_assert_eq!(res.len(), L);
        res
    }
}

impl VecShare<u64> {
    fn share128_from_share64s(a: &Share<u64>, b: &Share<u64>) -> Share<u128> {
        let a_ = (a.a.0 as u128) | ((b.a.0 as u128) << 64);
        let b_ = (a.b.0 as u128) | ((b.b.0 as u128) << 64);

        Share {
            a: RingElement(a_),
            b: RingElement(b_),
        }
    }

    fn share_transpose64x128(a: &[Share<u64>; 128]) -> [Share<u128>; 64] {
        let mut j: u32;
        let mut k: usize;
        let mut m: u128;
        let mut t: Share<u128>;

        let mut res = core::array::from_fn(|_| Share::default());

        // pack results into Share128 datatypes
        for (i, bb) in res.iter_mut().enumerate() {
            *bb = Self::share128_from_share64s(&a[i], &a[i + 64]);
        }

        // version of 128x128 transpose that only does the swaps needed for 64 bits
        m = 0x00000000ffffffff00000000ffffffff;
        j = 32;
        while j != 0 {
            k = 0;
            while k < 64 {
                t = ((&res[k] >> j) ^ &res[k + j as usize]) & m;
                res[k + j as usize] ^= &t;
                res[k] ^= t << j;
                k = (k + j as usize + 1) & !(j as usize);
            }
            j >>= 1;
            m = m ^ (m << j);
        }

        res
    }

    fn share_transpose64x64(a: &mut [Share<u64>; 64]) {
        let mut j: u32;
        let mut k: usize;
        let mut m: u64;
        let mut t: Share<u64>;

        m = 0x00000000ffffffff;
        j = 32;
        while j != 0 {
            k = 0;
            while k < 64 {
                t = ((&a[k] >> j) ^ &a[k + j as usize]) & m;
                a[k + j as usize] ^= &t;
                a[k] ^= t << j;
                k = (k + j as usize + 1) & !(j as usize);
            }
            j >>= 1;
            m = m ^ (m << j);
        }
    }

    pub fn transpose_pack_u64(self) -> Vec<VecShare<u64>> {
        self.transpose_pack_u64_with_len::<{ u64::BITS as usize }>()
    }

    pub fn transpose_pack_u64_with_len<const L: usize>(mut self) -> Vec<VecShare<u64>> {
        // Pad to multiple of 64
        let len = (self.shares.len() + 63) / 64;
        self.shares.resize(len * 64, Share::default());

        let mut res = (0..L)
            .map(|_| VecShare::new_vec(vec![Share::default(); len]))
            .collect::<Vec<_>>();

        for (j, x) in self.shares.chunks_exact_mut(64).enumerate() {
            Self::share_transpose64x64(x.try_into().unwrap());
            for (src, des) in x.iter().cloned().zip(res.iter_mut()) {
                des.shares[j] = src;
            }
        }
        debug_assert_eq!(res.len(), L);
        res
    }

    pub fn transpose_pack_u128(self) -> Vec<VecShare<u128>> {
        self.transpose_pack_u128_with_len::<{ u64::BITS as usize }>()
    }

    pub fn transpose_pack_u128_with_len<const L: usize>(mut self) -> Vec<VecShare<u128>> {
        // Pad to multiple of 128
        let len = (self.shares.len() + 127) / 128;
        self.shares.resize(len * 128, Share::default());

        let mut res = (0..L)
            .map(|_| VecShare::new_vec(vec![Share::default(); len]))
            .collect::<Vec<_>>();

        for (j, x) in self.shares.chunks_exact(128).enumerate() {
            let trans = Self::share_transpose64x128(x.try_into().unwrap());
            for (src, des) in trans.into_iter().zip(res.iter_mut()) {
                des.shares[j] = src;
            }
        }
        debug_assert_eq!(res.len(), L);
        res
    }
}
