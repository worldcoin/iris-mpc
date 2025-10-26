// prf.rs — drop-in, faster PRF with ChaCha20 backend.
// - Primary seed type: PrfSeed = [u8; 32]  (matches ChaCha20Rng and many runtime APIs).
// - If you need a 16-byte key on the wire, use `prf_seed32_to_16(seed32)`.
// - If you receive a 16-byte seed from the wire, expand with `prf_expand_seed_16_to_32(seed16)`.
//
// Dependencies (Cargo.toml):
// [dependencies]
// rand = "0.8"
// rand_chacha = "0.3"
// rand_core = "0.6"
// blake3 = "1"

use crate::shares::{int_ring::IntRing2k, ring_impl::RingElement};
use rand::rngs::OsRng;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng};
use rand_core::{CryptoRng, RngCore};

// Fast PRG backend. We call it AesRng to preserve the rest of your code.
pub use rand_chacha::ChaCha20Rng as AesRng;

// --- Seed types --------------------------------------------------------------

/// Primary seed type used internally and for runtime APIs that expect 32 bytes.
pub type PrfSeed = <AesRng as SeedableRng>::Seed; // [u8; 32]
/// Wire-format legacy (if you still have 16-byte keys in messages).
pub type PrfSeed16 = [u8; 16];

// --- Helpers for (de)normalizing seed sizes ----------------------------------

/// Expand a 16-byte wire seed into a 32-byte ChaCha20 seed (deterministic, via BLAKE3).
#[inline]
pub fn prf_expand_seed_16_to_32(short: PrfSeed16) -> PrfSeed {
    use blake3::Hasher;
    let mut h = Hasher::new();
    h.update(&short);
    let digest = h.finalize(); // 32 bytes
    let mut out = [0u8; 32];
    out.copy_from_slice(digest.as_bytes());
    out
}

/// Derive a 16-byte key from a 32-byte seed (for wire formats that still require [u8;16]).
/// This is a *derivation*, not truncation: stable and domain-separated.
#[inline]
pub fn prf_seed32_to_16(seed32: PrfSeed) -> PrfSeed16 {
    use blake3::Hasher;
    let mut h = Hasher::new();
    h.update(b"prf:seed32->key16");
    h.update(&seed32);
    let digest = h.finalize(); // 32 bytes
    let mut out = [0u8; 16];
    out.copy_from_slice(&digest.as_bytes()[..16]);
    out
}

/// Convenience: expand a Vec<[u8;16]> to Vec<[u8;32]> (useful before runtime calls).
#[inline]
pub fn prf_expand_seeds_16_to_32_vec(seeds16: Vec<PrfSeed16>) -> Vec<PrfSeed> {
    seeds16.into_iter().map(prf_expand_seed_16_to_32).collect()
}

// --- PRF object --------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Prf {
    pub my_prf: AesRng,
    pub prev_prf: AesRng,
}

impl Default for Prf {
    fn default() -> Self {
        Self {
            my_prf: AesRng::from_rng(OsRng).expect("OsRng"),
            prev_prf: AesRng::from_rng(OsRng).expect("OsRng"),
        }
    }
}

impl Prf {
    /// Construct from 32-byte seeds (native ChaCha20). Prefer this in new code.
    pub fn new(my_key: PrfSeed, prev_key: PrfSeed) -> Self {
        Self {
            my_prf: AesRng::from_seed(my_key),
            prev_prf: AesRng::from_seed(prev_key),
        }
    }

    /// Construct from 16-byte wire seeds (legacy): expands internally to 32 bytes.
    pub fn new_from_16(my_key16: PrfSeed16, prev_key16: PrfSeed16) -> Self {
        let my_key = prf_expand_seed_16_to_32(my_key16);
        let prev_key = prf_expand_seed_16_to_32(prev_key16);
        Self::new(my_key, prev_key)
    }

    /// 32-byte seed from OS CSPRNG (preferred).
    #[inline]
    pub fn gen_seed() -> PrfSeed {
        let mut s = PrfSeed::default();
        OsRng.fill_bytes(&mut s);
        s
    }

    /// 16-byte seed from OS CSPRNG (for wire formats that still require 16 bytes).
    #[inline]
    pub fn gen_seed16() -> PrfSeed16 {
        let mut s = [0u8; 16];
        OsRng.fill_bytes(&mut s);
        s
    }

    #[inline]
    pub fn get_my_prf(&mut self) -> &mut AesRng {
        &mut self.my_prf
    }

    #[inline]
    pub fn get_prev_prf(&mut self) -> &mut AesRng {
        &mut self.prev_prf
    }

    /// Generate one sample from each stream (additive sharing).
    pub fn gen_rands<T>(&mut self) -> (T, T)
    where
        Standard: Distribution<T>,
    {
        let a = self.my_prf.gen::<T>();
        let b = self.prev_prf.gen::<T>();
        (a, b)
    }

    /// Additive zero-share in ring T: a - b
    pub fn gen_zero_share<T: IntRing2k>(&mut self) -> RingElement<T>
    where
        Standard: Distribution<T>,
    {
        let (a, b) = self.gen_rands::<RingElement<T>>();
        a - b
    }

    /// Binary/XOR zero-share in ring T: a ^ b
    pub fn gen_binary_zero_share<T: IntRing2k>(&mut self) -> RingElement<T>
    where
        Standard: Distribution<T>,
    {
        let (a, b) = self.gen_rands::<RingElement<T>>();
        a ^ b
    }
}
