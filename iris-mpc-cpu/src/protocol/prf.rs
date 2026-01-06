use crate::shares::{int_ring::IntRing2k, ring_impl::RingElement};
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng};

#[cfg(not(feature = "chacha_prf"))]
type PrfRng = aes_prng::AesRng;

#[cfg(feature = "chacha_prf")]
type PrfRng = rand_chacha::ChaCha20Rng;

pub type PrfSeed = [u8; 16];

#[derive(Clone, Debug)]
pub struct Prf {
    pub my_prf: PrfRng,
    pub prev_prf: PrfRng,
}

impl Default for Prf {
    fn default() -> Self {
        Self {
            my_prf: PrfRng::from_entropy(),
            prev_prf: PrfRng::from_entropy(),
        }
    }
}

impl Prf {
    #[cfg(feature = "chacha_prf")]
    pub fn new(my_key: PrfSeed, prev_key: PrfSeed) -> Self {
        let my_seed32 = expand_seed_16_to_32(my_key);
        let prev_seed32 = expand_seed_16_to_32(prev_key);
        Self {
            my_prf: PrfRng::from_seed(Self::expand_seed(my_key)),
            prev_prf: PrfRng::from_seed(Self::expand_seed(prev_key)),
        }
    }

    #[cfg(not(feature = "chacha_prf"))]
    pub fn new(my_key: PrfSeed, prev_key: PrfSeed) -> Self {
        Self {
            my_prf: PrfRng::from_seed(my_key),
            prev_prf: PrfRng::from_seed(prev_key),
        }
    }

    #[cfg(feature = "chacha_prf")]
    fn expand_seed(seed: PrfSeed) -> [u8; 32] {
        use blake3::Hasher;
        let mut h = Hasher::new();
        h.update(&seed);
        let digest = h.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(digest.as_bytes());
        out
    }

    pub fn get_my_prf(&mut self) -> &mut PrfRng {
        &mut self.my_prf
    }

    pub fn get_prev_prf(&mut self) -> &mut PrfRng {
        &mut self.prev_prf
    }

    pub fn gen_seed() -> PrfSeed {
        let mut rng = PrfRng::from_entropy();
        rng.gen::<PrfSeed>()
    }

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
