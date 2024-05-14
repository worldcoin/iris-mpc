use rand::{CryptoRng, Rng};

use crate::PartyID;

pub const P: u16 = ((1u32 << 16) - 17) as u16;
pub const P32: u32 = P as u32;

pub struct Shamir {}

impl Shamir {
    pub fn random_fp<R: Rng + CryptoRng>(rng: &mut R) -> u16 {
        let mut r = rng.gen::<u16>();
        while r >= P {
            r = rng.gen::<u16>();
        }
        r
    }

    // Euclid with inputs reversed, which is optimized for getting inverses
    fn extended_euclid_rev(a: u32, b: u32) -> (u32, u32) {
        let mut r1 = a;
        let mut r0 = b;
        let mut s1: u32 = 1;
        let mut s0: u32 = 0;

        while r1 != 0 {
            let q = r0 / r1;
            let tmp = r0 + P32 - (q * r1) % P32;
            r0 = r1;
            r1 = tmp % P32;
            let tmp = s0 + P32 - (q * s1) % P32;
            s0 = s1;
            s1 = tmp % P32;
        }
        (r0, s0)
    }

    fn mod_inverse(x: u16) -> u16 {
        let inv = Self::extended_euclid_rev(x as u32, P32).1;
        debug_assert_eq!((x as u32 * inv) % P32, 1);
        inv as u16
    }

    pub fn share_d1<R: Rng + CryptoRng>(secret: u16, rng: &mut R) -> [u16; 3] {
        let mut shares = [0; 3];
        let coeff = Self::random_fp(rng) as u32;

        shares[0] = ((secret as u32 + coeff) % P32) as u16;
        shares[1] = ((shares[0] as u32 + coeff) % P32) as u16;
        shares[2] = ((shares[1] as u32 + coeff) % P32) as u16;
        shares
    }

    pub fn my_lagrange_coeff_d1(id: PartyID, other: PartyID) -> u16 {
        let mut den = 1;
        let i = (usize::from(id) + 1) as u32;
        let j = (usize::from(other) + 1) as u32;

        let num = j;
        den *= (j + P32 - i) % P32;
        den %= P32;

        ((num * Self::mod_inverse(den as u16) as u32) % P32) as u16
    }

    pub fn my_lagrange_coeff_d2(id: PartyID) -> u16 {
        let mut num = 1;
        let mut den = 1;
        let i = (usize::from(id) + 1) as u32;
        for j in 1..=3u32 {
            if i != j {
                num = (num * j) % P32;
                den *= (j + P32 - i) % P32;
                den %= P32;
            }
        }
        ((num * Self::mod_inverse(den as u16) as u32) % P32) as u16
    }
}

pub fn share_bool_vec<R: Rng + CryptoRng>(data: &[bool], rng: &mut R) -> [Vec<u16>; 3] {
    let len = data.len();
    let mut results = [
        Vec::with_capacity(len),
        Vec::with_capacity(len),
        Vec::with_capacity(len),
    ];

    for code in data.iter() {
        let [share0, share1, share2] = Shamir::share_d1(*code as u16, rng);
        results[0].push(share0);
        results[1].push(share1);
        results[2].push(share2);
    }

    results
}

#[cfg(test)]
mod shamir_test {
    use crate::PartyID;

    use super::*;

    const TESTRUNS: usize = 5;

    #[test]
    fn test_shamir() {
        let mut rng = rand::thread_rng();
        for _ in 0..TESTRUNS {
            let secret1 = Shamir::random_fp(&mut rng);
            let secret2 = Shamir::random_fp(&mut rng);
            let mul = ((secret1 as u32 * secret2 as u32) % P32) as u16;

            let shares1 = Shamir::share_d1(secret1, &mut rng); // Degree-1 shares
            let shares2 = Shamir::share_d1(secret2, &mut rng); // Degree-1 shares
            let mul_shares = [
                ((shares1[0] as u32 * shares2[0] as u32) % P32) as u16,
                ((shares1[1] as u32 * shares2[1] as u32) % P32) as u16,
                ((shares1[2] as u32 * shares2[2] as u32) % P32) as u16,
            ]; // Degree-2 shares

            // Reconstruct d1 shares with all combinations
            let lagrange_d1 = [
                Shamir::my_lagrange_coeff_d1(PartyID::ID0, PartyID::ID1) as u32,
                Shamir::my_lagrange_coeff_d1(PartyID::ID1, PartyID::ID0) as u32,
            ];
            let s1 = (((shares1[0] as u32 * lagrange_d1[0]) % P32
                + (shares1[1] as u32 * lagrange_d1[1]) % P32)
                % P32) as u16;
            assert_eq!(s1, secret1);
            let s2 = (((shares2[0] as u32 * lagrange_d1[0]) % P32
                + (shares2[1] as u32 * lagrange_d1[1]) % P32)
                % P32) as u16;
            assert_eq!(s2, secret2);

            let lagrange_d1 = [
                Shamir::my_lagrange_coeff_d1(PartyID::ID0, PartyID::ID2) as u32,
                Shamir::my_lagrange_coeff_d1(PartyID::ID2, PartyID::ID0) as u32,
            ];
            let s1 = (((shares1[0] as u32 * lagrange_d1[0]) % P32
                + (shares1[2] as u32 * lagrange_d1[1]) % P32)
                % P32) as u16;
            assert_eq!(s1, secret1);
            let s2 = (((shares2[0] as u32 * lagrange_d1[0]) % P32
                + (shares2[2] as u32 * lagrange_d1[1]) % P32)
                % P32) as u16;
            assert_eq!(s2, secret2);

            let lagrange_d1 = [
                Shamir::my_lagrange_coeff_d1(PartyID::ID1, PartyID::ID2) as u32,
                Shamir::my_lagrange_coeff_d1(PartyID::ID2, PartyID::ID1) as u32,
            ];
            let s1 = (((shares1[1] as u32 * lagrange_d1[0]) % P32
                + (shares1[2] as u32 * lagrange_d1[1]) % P32)
                % P32) as u16;
            assert_eq!(s1, secret1);
            let s2 = (((shares2[1] as u32 * lagrange_d1[0]) % P32
                + (shares2[2] as u32 * lagrange_d1[1]) % P32)
                % P32) as u16;
            assert_eq!(s2, secret2);

            let lagrange_d2 = [
                Shamir::my_lagrange_coeff_d2(PartyID::ID0) as u32,
                Shamir::my_lagrange_coeff_d2(PartyID::ID1) as u32,
                Shamir::my_lagrange_coeff_d2(PartyID::ID2) as u32,
            ];

            let reconstructed = ((0..3).fold(0u32, |acc, i| {
                acc + (mul_shares[i] as u32 * lagrange_d2[i]) % P32
            }) % P32) as u16;

            assert_eq!(mul, reconstructed);
        }
    }
}
