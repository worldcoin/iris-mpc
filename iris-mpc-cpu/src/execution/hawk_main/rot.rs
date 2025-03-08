use std::ops::Deref;

use iris_mpc_common::ROTATIONS;
use itertools::Itertools;

/// VecRots are lists of things for each rotation.
pub struct VecRots<R> {
    rotations: Vec<R>,
}

impl<R> Deref for VecRots<R> {
    type Target = Vec<R>;

    fn deref(&self) -> &Self::Target {
        &self.rotations
    }
}

impl<R> VecRots<R> {
    pub fn new(rotations: Vec<R>) -> Self {
        assert_eq!(rotations.len(), ROTATIONS);
        Self { rotations }
    }

    /// Get the item attached to the center rotation.
    pub fn center(&self) -> &R {
        &self.rotations[self.rotations.len() / 2]
    }

    /// Get the item attached to the center rotation.
    pub fn into_center(self) -> R {
        let middle = self.rotations.len() / 2;
        self.rotations.into_iter().nth(middle).unwrap()
    }

    pub fn flatten(batch: &[VecRots<R>]) -> Vec<R>
    where
        R: Clone,
    {
        batch
            .iter()
            .flat_map(|x| &x.rotations)
            .cloned()
            .collect_vec()
    }

    pub fn nest(batch: Vec<R>) -> Vec<VecRots<R>> {
        let mut rots = Vec::with_capacity(batch.len() / ROTATIONS);
        let mut it = batch.into_iter();
        loop {
            let rot = it.by_ref().take(ROTATIONS).collect_vec();
            if rot.is_empty() {
                break;
            }
            rots.push(VecRots::new(rot));
        }
        rots
    }
}
