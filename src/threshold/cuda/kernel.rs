pub(crate) const B_BITS: u64 = 20;

pub fn shared_xor<T>(
    res_a: &mut [T],
    res_b: &mut [T],
    lhs_a: &[T],
    lhs_b: &[T],
    rhs_a: &[T],
    rhs_b: &[T],
) where
    for<'a> &'a T: std::ops::BitXor<&'a T, Output = T>,
{
    let n = res_a.len();
    assert_eq!(n, res_b.len());
    assert_eq!(n, lhs_a.len());
    assert_eq!(n, lhs_b.len());
    assert_eq!(n, rhs_a.len());
    assert_eq!(n, rhs_b.len());

    for i in 0..n {
        res_a[i] = &lhs_a[i] ^ &rhs_a[i];
        res_b[i] = &lhs_b[i] ^ &rhs_b[i];
    }
}

pub fn shared_xor_assign<T>(lhs_a: &mut [T], lhs_b: &mut [T], rhs_a: &[T], rhs_b: &[T])
where
    for<'a> T: std::ops::BitXorAssign<&'a T>,
{
    let n = lhs_a.len();
    assert_eq!(n, lhs_b.len());
    assert_eq!(n, rhs_a.len());
    assert_eq!(n, rhs_b.len());

    for i in 0..n {
        lhs_a[i] ^= &rhs_a[i];
        lhs_b[i] ^= &rhs_b[i];
    }
}

// Computes the local part of the multiplication (including randomness)
pub fn shared_and_pre<T>(
    res_a: &mut [T],
    lhs_a: &[T],
    lhs_b: &[T],
    rhs_a: &[T],
    rhs_b: &[T],
    rand: &[T],
) where
    for<'a> &'a T: std::ops::BitAnd<&'a T, Output = T>,
    for<'a> T: std::ops::BitXor<&'a T, Output = T>,
    T: std::ops::BitXor<Output = T>,
{
    let n = res_a.len();
    assert_eq!(n, lhs_a.len());
    assert_eq!(n, lhs_b.len());
    assert_eq!(n, rhs_a.len());
    assert_eq!(n, rhs_b.len());
    assert_eq!(n, rand.len());

    for i in 0..n {
        res_a[i] =
            (&lhs_a[i] & &rhs_a[i]) ^ (&lhs_b[i] & &rhs_a[i]) ^ (&lhs_a[i] & &rhs_b[i]) ^ &rand[i];
    }
}

pub fn shared_mul_lift_b(res_a: &mut [u64], res_b: &mut [u64], lhs_a: &[u16], lhs_b: &[u16]) {
    let n = res_a.len();
    assert_eq!(n, res_b.len());
    assert_eq!(n, lhs_a.len());
    assert_eq!(n, lhs_b.len());

    for i in 0..n {
        res_a[i] = (lhs_a[i] as u64) << B_BITS;
        res_b[i] = (lhs_b[i] as u64) << B_BITS;
    }
}

fn u64_from_u16s(a: &u16, b: &u16, c: &u16, d: &u16) -> u64 {
    (*a as u64) | ((*b as u64) << 16) | ((*c as u64) << 32) | ((*d as u64) << 48)
}

fn share_transpose16x64(inp: &[u16; 64], outp: &mut [u64; 16]) {
    let mut j: u32;
    let mut k: usize;
    let mut m: u64;
    let mut t: u64;

    // pack results into Share64 datatypes
    for (i, bb) in outp.iter_mut().enumerate() {
        *bb = u64_from_u16s(&inp[i], &inp[16 + i], &inp[32 + i], &inp[48 + i]);
    }

    // version of 64x64 transpose that only does the swaps needed for 16 bits
    m = 0x00FF00FF00FF00FF;
    j = 8;
    while j != 0 {
        k = 0;
        while k < 16 {
            t = ((outp[k] >> j) ^ outp[k + j as usize]) & m;
            outp[k + j as usize] ^= &t;
            outp[k] ^= t << j;
            k = (k + j as usize + 1) & !(j as usize);
        }
        j >>= 1;
        m = m ^ (m << j);
    }
}

pub fn u16_transpose_pack_u64_with_len<const L: usize>(inp: &[u16]) -> Vec<Vec<u64>> {
    assert!(inp.len() % 64 == 0);
    let len = inp.len() / 64;

    let mut res = (0..L).map(|_| vec![0; len]).collect::<Vec<_>>();

    for (j, x) in inp.chunks_exact(64).enumerate() {
        let mut trans = [0; 16];
        share_transpose16x64(x.try_into().unwrap(), &mut trans);
        for (src, des) in trans.into_iter().zip(res.iter_mut()) {
            des[j] = src;
        }
    }
    debug_assert_eq!(res.len(), L);
    res
}

fn u64_from_u32s(a: &u32, b: &u32) -> u64 {
    (*a as u64) | ((*b as u64) << 32)
}

fn share_transpose32x64(inp: &[u32; 64], outp: &mut [u64; 32]) {
    let mut j: u32;
    let mut k: usize;
    let mut m: u64;
    let mut t: u64;

    // pack results into Share64 datatypes
    for (i, bb) in outp.iter_mut().enumerate() {
        *bb = u64_from_u32s(&inp[i], &inp[32 + i]);
    }

    // version of 64x64 transpose that only does the swaps needed for 16 bits
    m = 0x0000FFFF0000FFFF;
    j = 16;
    while j != 0 {
        k = 0;
        while k < 32 {
            t = ((outp[k] >> j) ^ outp[k + j as usize]) & m;
            outp[k + j as usize] ^= &t;
            outp[k] ^= t << j;
            k = (k + j as usize + 1) & !(j as usize);
        }
        j >>= 1;
        m = m ^ (m << j);
    }
}

pub fn u32_transpose_pack_u64_with_len<const L: usize>(inp: &[u32]) -> Vec<Vec<u64>> {
    assert!(inp.len() % 64 == 0);
    let len = inp.len() / 64;

    let mut res = (0..L).map(|_| vec![0; len]).collect::<Vec<_>>();

    for (j, x) in inp.chunks_exact(64).enumerate() {
        let mut trans = [0; 32];
        share_transpose32x64(x.try_into().unwrap(), &mut trans);
        for (src, des) in trans.into_iter().zip(res.iter_mut()) {
            des[j] = src;
        }
    }
    debug_assert_eq!(res.len(), L);
    res
}

fn share_transpose64x64(inout: &mut [u64; 64]) {
    let mut j: u32;
    let mut k: usize;
    let mut m: u64;
    let mut t: u64;

    // version of 64x64 transpose that only does the swaps needed for 16 bits
    m = 0x00000000FFFFFFFF;
    j = 32;
    while j != 0 {
        k = 0;
        while k < 64 {
            t = ((inout[k] >> j) ^ inout[k + j as usize]) & m;
            inout[k + j as usize] ^= &t;
            inout[k] ^= t << j;
            k = (k + j as usize + 1) & !(j as usize);
        }
        j >>= 1;
        m = m ^ (m << j);
    }
}

pub fn u64_transpose_pack_u64_with_len<const L: usize>(inp: &mut [u64]) -> Vec<Vec<u64>> {
    assert!(inp.len() % 64 == 0);
    let len = inp.len() / 64;

    let mut res = (0..L).map(|_| vec![0; len]).collect::<Vec<_>>();

    for (j, x) in inp.chunks_exact_mut(64).enumerate() {
        share_transpose64x64(x.try_into().unwrap());
        for (src, des) in x.iter().zip(res.iter_mut()) {
            des[j] = *src;
        }
    }
    debug_assert_eq!(res.len(), L);
    res
}
