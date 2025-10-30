use std::{marker::PhantomData, ops::Deref};

use iris_mpc_common::ROTATIONS;
use itertools::Itertools;

pub trait Rotations: Send + Sync + 'static {
    /// The number of rotations.
    const N_ROTATIONS: usize;
}

#[derive(Clone, Debug)]
pub struct AllRotations {}

impl Rotations for AllRotations {
    const N_ROTATIONS: usize = ROTATIONS;
}

#[derive(Clone, Debug)]
pub struct CenterOnly {}

impl Rotations for CenterOnly {
    const N_ROTATIONS: usize = 1;
}

/// VecRotationSupport is an abstraction for functions that work with or without rotations,
/// controlled by the generic ROT = AllRotations or CenterOnly.
/// Under the hood it is a Vec of length either 1 or ROTATIONS.
#[derive(Clone, Debug)]
pub struct VecRotationSupport<R, ROT> {
    rotations: Vec<R>,
    phantom: PhantomData<ROT>,
}

impl<R, ROT> Deref for VecRotationSupport<R, ROT> {
    type Target = Vec<R>;

    fn deref(&self) -> &Self::Target {
        &self.rotations
    }
}

impl<R, ROT: Rotations> From<Vec<R>> for VecRotationSupport<R, ROT> {
    fn from(rotations: Vec<R>) -> Self {
        assert_eq!(rotations.len(), ROT::N_ROTATIONS);
        Self {
            rotations,
            phantom: PhantomData,
        }
    }
}

impl<R> VecRotationSupport<R, CenterOnly> {
    pub fn new_center_only(center: R) -> Self {
        Self {
            rotations: vec![center],
            phantom: PhantomData,
        }
    }
}

impl<R, ROT: Rotations> VecRotationSupport<R, ROT> {
    pub const fn n_rotations() -> usize {
        ROT::N_ROTATIONS
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
        let mut rots = Vec::with_capacity(batch.len() / ROT::N_ROTATIONS);
        let mut it = batch.into_iter();
        loop {
            let rot = it.by_ref().take(ROT::N_ROTATIONS).collect_vec();
            if rot.is_empty() {
                break;
            }
            rots.push(Self::from(rot));
        }
        rots
    }
}
