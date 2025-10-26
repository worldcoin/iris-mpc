// prf.rs — fast PRF with ChaCha20 backend while keeping PrfSeed = [u8; 16] on the wire.
//
// Why this file:
// - Keeps your external types the same: PrfSeed = [u8; 16].
// - NetworkValue::PrfKey([u8; 16]) continues to compile.
// - LocalRuntime::new_with_network_type(..., Vec<PrfSeed>, ...) now expects Vec<[u8; 16]>.
// - Internally expands 16-byte seeds into 32-bytes for ChaCha20 (deterministic via BLAKE3).
//
// Dependencies (Cargo.toml):
//   rand = "0.8"
//   rand_core = "0.6"
//   rand_chacha = "0.3"
//   blake3 = "1"

use crate::shares::{int_ring::IntRing2k, ring_impl::RingElement};
use rand::rngs::OsRng;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng};
use rand_chacha::rand_core;
use rand_core::RngCore;

// Keep the old name visible to the rest of the codebase.
pub use rand_chacha::ChaCha20Rng as AesRng;

/// Wire / external seed type. Stays 16 bytes to match existing code and messages.
pub type PrfSeed = [u8; 16];

/// Internal 32-byte seed for ChaCha20.
type PrfSeed32 = <AesRng as SeedableRng>::Seed; // [u8; 32]

#[inline]
fn expand_seed_16_to_32(short: PrfSeed) -> PrfSeed32 {
    use blake3::Hasher;
    let mut h = Hasher::new();
    h.update(&short);
    let digest = h.finalize(); // 32 bytes
    let mut out = [0u8; 32];
    out.copy_from_slice(digest.as_bytes());
    out
}

#[derive(Clone, Debug)]
pub struct Prf {
    pub my_prf: AesRng,
    pub prev_prf: AesRng,
}

impl Default for Prf {
    fn default() -> Self {
        let my16 = Prf::gen_seed();
        let prev16 = Prf::gen_seed();
        Self::new(my16, prev16)
    }
}

impl Prf {
    /// Construct from 16-byte seeds (wire format). Internally expanded to 32 bytes.
    pub fn new(my_key: PrfSeed, prev_key: PrfSeed) -> Self {
        let my_seed32 = expand_seed_16_to_32(my_key);
        let prev_seed32 = expand_seed_16_to_32(prev_key);
        Self {
            my_prf: AesRng::from_seed(my_seed32),
            prev_prf: AesRng::from_seed(prev_seed32),
        }
    }

    /// Generate a fresh 16-byte seed from OS CSPRNG (wire format).
    #[inline]
    pub fn gen_seed() -> PrfSeed {
        let mut s = [0u8; 16];
        OsRng.fill_bytes(&mut s);
        s
        // Note: Security is bounded by 128 bits of entropy here (wire format).
        // Internally we expand to 256-bit ChaCha seeds via BLAKE3.
    }

    /// Accessors
    #[inline]
    pub fn get_my_prf(&mut self) -> &mut AesRng {
        &mut self.my_prf
    }

    #[inline]
    pub fn get_prev_prf(&mut self) -> &mut AesRng {
        &mut self.prev_prf
    }

    /// Draw one sample from each stream.
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

// Optional helpers if you ever need them elsewhere:

/// Expand a slice of 16-byte seeds to 32-byte seeds (not used here, but handy).
#[allow(dead_code)]
pub fn expand_seeds_vec_16_to_32(seeds16: Vec<PrfSeed>) -> Vec<PrfSeed32> {
    seeds16.into_iter().map(expand_seed_16_to_32).collect()
}

/// Derive a 16-byte key from a 32-byte seed (domain-separated), if needed for wire formats.
#[allow(dead_code)]
pub fn derive_key16_from_seed32(seed32: PrfSeed32) -> PrfSeed {
    use blake3::Hasher;
    let mut h = Hasher::new();
    h.update(b"prf:seed32->key16");
    h.update(&seed32);
    let digest = h.finalize();
    let mut out = [0u8; 16];
    out.copy_from_slice(&digest.as_bytes()[..16]);
    out
}
