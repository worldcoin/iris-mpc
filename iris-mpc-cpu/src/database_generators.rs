use crate::shares::{ring_impl::RingElement, share::Share};
use rand::{Rng, RngCore};

type ShareRing = u16;
type ShareRingPlain = RingElement<ShareRing>;

pub fn create_random_sharing<R: RngCore>(rng: &mut R, input: u16) -> Vec<Share<u16>> {
    let val = RingElement(input);
    let a = rng.gen::<ShareRingPlain>();
    let b = rng.gen::<ShareRingPlain>();
    let c = val - a - b;

    let share1 = Share::new(a, c);
    let share2 = Share::new(b, a);
    let share3 = Share::new(c, b);

    vec![share1, share2, share3]
}
