use std::{marker::PhantomData, ops::Deref};

use iris_mpc_common::ROTATIONS;
use itertools::Itertools;

pub trait Rotations: Send + Sync + 'static {
    /// The number of rotations.
    fn n_rotations() -> usize;
}

#[derive(Clone, Debug)]
pub struct WithRot {}

impl Rotations for WithRot {
    fn n_rotations() -> usize {
        ROTATIONS
    }
}

#[derive(Clone, Debug)]
pub struct WithoutRot {}

impl Rotations for WithoutRot {
    fn n_rotations() -> usize {
        1
    }
}

/// VecRots are lists of things for each rotation.
#[derive(Clone, Debug)]
pub struct VecRots<R, ROT = WithRot> {
    rotations: Vec<R>,
    phantom: PhantomData<ROT>,
}

impl<R, ROT> Deref for VecRots<R, ROT> {
    type Target = Vec<R>;

    fn deref(&self) -> &Self::Target {
        &self.rotations
    }
}

impl<R, ROT: Rotations> From<Vec<R>> for VecRots<R, ROT> {
    fn from(rotations: Vec<R>) -> Self {
        assert_eq!(rotations.len(), ROT::n_rotations());
        Self {
            rotations,
            phantom: PhantomData,
        }
    }
}

impl<R> VecRots<R, WithoutRot> {
    pub fn new_center_only(center: R) -> Self {
        Self {
            rotations: vec![center],
            phantom: PhantomData,
        }
    }
}

impl<R, ROT: Rotations> VecRots<R, ROT> {
    /// Get the item attached to the center rotation.
    pub fn center(&self) -> &R {
        &self.rotations[self.rotations.len() / 2]
    }

    /// Get the item attached to the center rotation.
    pub fn into_center(self) -> R {
        let middle = self.rotations.len() / 2;
        self.rotations.into_iter().nth(middle).unwrap()
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
        let mut rots = Vec::with_capacity(batch.len() / ROT::n_rotations());
        let mut it = batch.into_iter();
        loop {
            let rot = it.by_ref().take(ROT::n_rotations()).collect_vec();
            if rot.is_empty() {
                break;
            }
            rots.push(VecRots::from(rot));
        }
        rots
    }
}
