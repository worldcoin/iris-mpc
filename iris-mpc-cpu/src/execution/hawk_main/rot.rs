use itertools::Itertools;
use std::ops::Deref;

#[allow(dead_code)]
/// All rotations from -15 to 15
pub const ALL_ROTATIONS_MASK: u32 = (1 << 31) - 1;

#[allow(dead_code)]
/// Rotation 0 only
pub const CENTER_ONLY_MASK: u32 = 1 << 15;

#[allow(dead_code)]
/// Rotations -10, 0, 10. These base rotations cover [-15, 15] when used with MinFhd5/6
pub const CENTER_AND_10_MASK: u32 = (1 << 5) + (1 << 15) + (1 << 25);

/// VecRotationSupport is an abstraction for functions that work with or without rotations,
/// controlled by the const generic ROTMASK which is a u32 where the i-th bit is set iff
/// the i-th rotation is present in this collection.
/// Under the hood it is a Vec of length `ROTMASK.count_ones()`.
#[derive(Clone, Debug)]
pub struct VecRotationSupport<R, const ROTMASK: u32> {
    rotations: Vec<R>,
}

impl<R, const ROTMASK: u32> Deref for VecRotationSupport<R, ROTMASK> {
    type Target = Vec<R>;

    fn deref(&self) -> &Self::Target {
        &self.rotations
    }
}

impl<R, const ROTMASK: u32> From<Vec<R>> for VecRotationSupport<R, ROTMASK> {
    fn from(rotations: Vec<R>) -> Self {
        assert_eq!(rotations.len(), Self::n_rotations());
        Self { rotations }
    }
}

impl<R> VecRotationSupport<R, { 1 << 15 }> {
    pub fn new_center_only(r: R) -> Self {
        Self { rotations: vec![r] }
    }
}

impl<R, const ROTMASK: u32> VecRotationSupport<R, ROTMASK> {
    pub const fn n_rotations() -> usize {
        ROTMASK.count_ones() as usize
    }
    /// Get the item attached to the center rotation.
    pub fn center(&self) -> &R {
        &self.rotations[self.rotations.len() / 2]
    }

    /// Flatten a batch of something with rotations into a concatenated Vec.
    /// Attach a copy of the corresponding `B` to each rotation.
    pub fn flatten_broadcast<'a, B>(batch: impl IntoIterator<Item = (&'a Self, B)>) -> Vec<(R, B)>
    where
        B: Clone + 'a,
        R: Clone + 'a,
    {
        batch
            .into_iter()
            .flat_map(|(rots, b)| rots.rotations.iter().map(move |r| (r.clone(), b.clone())))
            .collect_vec()
    }

    /// The opposite of flatten.
    /// Split a concatenated Vec into a batch of something with rotations.
    pub fn unflatten(batch: Vec<R>) -> Vec<Self> {
        let mut rots = Vec::with_capacity(batch.len() / Self::n_rotations());
        let mut it = batch.into_iter();
        loop {
            let rot = it.by_ref().take(Self::n_rotations()).collect_vec();
            if rot.is_empty() {
                break;
            }
            rots.push(Self::from(rot));
        }
        rots
    }
}
