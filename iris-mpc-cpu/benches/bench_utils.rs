use iris_mpc_cpu::shares::{IntRing2k, RingElement, Share};
use rand::{Rng, RngCore};
use rand_distr::{Distribution, Standard};

pub fn create_random_sharing<R, ShareRing>(rng: &mut R, input: ShareRing) -> Vec<Share<ShareRing>>
where
    R: RngCore,
    ShareRing: IntRing2k + std::fmt::Display,
    Standard: Distribution<ShareRing>,
{
    let val = RingElement(input);
    let a = RingElement(rng.gen());
    let b = RingElement(rng.gen());
    let c = val - a - b;

    let share1 = Share::new(a, c);
    let share2 = Share::new(b, a);
    let share3 = Share::new(c, b);

    vec![share1, share2, share3]
}
