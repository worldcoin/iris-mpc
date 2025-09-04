use std::{collections::HashMap, sync::Arc};

use iris_mpc_common::vector_id::{SerialId, VectorId, VersionId};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::execution::hawk_main::state_check::SetHash;

/// Storage of inserted irises.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SharedIrises<I: Clone> {
    pub points: Vec<Option<(VersionId, I)>>,
    pub size: usize, // Number of Some() stored in points
    pub next_id: u32,
    pub empty_iris: I,
    pub set_hash: SetHash,
}

impl<I: Clone + Default> Default for SharedIrises<I> {
    fn default() -> Self {
        Self {
            points: Default::default(),
            size: 0,
            next_id: 1,
            empty_iris: Default::default(),
            set_hash: Default::default(),
        }
    }
}

impl<I: Clone> SharedIrises<I> {
    pub fn new(points_map: HashMap<VectorId, I>, empty_iris: I) -> Self {
        let size = points_map.keys().map(|v| v.serial_id()).unique().count();
        let next_id = points_map.keys().map(|v| v.serial_id()).max().unwrap_or(0) + 1;

        let mut points = vec![None; next_id as usize];
        for (v, iris) in points_map.into_iter() {
            points[v.serial_id() as usize] = Some((v.version_id(), iris));
        }

        SharedIrises {
            points,
            size,
            next_id,
            empty_iris,
            set_hash: SetHash::default(),
        }
    }

    /// Inserts the given iris into the database with the specified id.  If an
    /// entry is already present with the given id, the iris is overwritten by `iris`.
    ///
    /// Updates the checksum hash to reflect the new or replaced entry for the
    /// associated serial id, and updates the `next_id` field to be the next
    /// value after the inserted serial id if this value is larger than the
    /// current value of `next_id`.
    pub fn insert(&mut self, vector_id: VectorId, iris: I) -> VectorId {
        let serial_id = vector_id.serial_id() as usize;

        // Extend underlying Vec to accomodate the new serial_id
        if self.points.len() <= serial_id {
            self.points
                .extend((0..serial_id - self.points.len() + 1).map(|_| None));
        }

        // If overwriting entry, remove previous vector id from set_hash
        if let Some((version, _)) = self.points[serial_id] {
            let prev_vector_id = VectorId::new(serial_id as u32, version);
            self.set_hash.remove(prev_vector_id);
            self.size -= 1;
        }

        self.size += 1;
        self.points[serial_id] = Some((vector_id.version_id(), iris));
        self.set_hash.add_unordered(vector_id);
        self.next_id = self.next_id.max(serial_id as u32 + 1);

        vector_id
    }

    /// Insert the given iris at the next unused serial ID, with version
    /// initialized to 0.
    pub fn append(&mut self, iris: I) -> VectorId {
        let new_id = self.next_id();
        self.insert(new_id, iris);
        new_id
    }

    /// Insert the given iris at ID given by `original_id.next_version()`, i.e.
    /// with identical serial number, and one higher version number.
    pub fn update(&mut self, original_id: VectorId, iris: I) -> VectorId {
        let new_id = original_id.next_version();
        self.insert(new_id, iris);
        new_id
    }

    pub fn db_size(&self) -> usize {
        self.size
    }

    /// Return the next id for new insertions, which should have the serial id
    /// following the largest previously inserted serial id, and version 0.
    fn next_id(&self) -> VectorId {
        VectorId::from_serial_id(self.next_id)
    }

    pub fn reserve(&mut self, additional: usize) {
        self.points.reserve(additional);
    }

    pub fn get_current_version(&self, serial_id: SerialId) -> Option<VersionId> {
        match &self.points.get(serial_id as usize) {
            Some(Some((version, _))) => Some(*version),
            _ => None,
        }
    }

    pub fn get_vector_or_empty(&self, vector: &VectorId) -> I {
        self.get_vector(vector).unwrap_or(self.empty_iris.clone())
    }

    pub fn get_vector_by_serial_id(&self, serial_id: SerialId) -> Option<I> {
        match &self.points.get(serial_id as usize) {
            Some(Some((_, iris))) => Some(iris.clone()),
            _ => None,
        }
    }

    pub fn get_vector(&self, vector: &VectorId) -> Option<I> {
        match &self.points.get(vector.serial_id() as usize) {
            Some(Some((version, iris))) if vector.version_matches(*version) => Some(iris.clone()),
            _ => None,
        }
    }

    pub fn contains(&self, vector: &VectorId) -> bool {
        matches!(self.points.get(vector.serial_id() as usize),
            Some(Some((version, _))) if vector.version_matches(*version))
    }

    pub fn get_sorted_serial_ids(&self) -> Vec<SerialId> {
        self.points
            .iter()
            .enumerate()
            .filter_map(|(i, op)| match op {
                Some(_) => Some(i as SerialId),
                _ => None,
            })
            .collect_vec()
    }

    pub fn to_arc(self) -> SharedIrisesRef<I> {
        SharedIrisesRef {
            data: Arc::new(RwLock::new(self)),
        }
    }

    pub fn last_vector_ids(&self, n: usize) -> Vec<VectorId> {
        // Mihai: was this logic correct previously?
        (1..self.next_id)
            .rev()
            .take(n)
            .filter_map(|serial_id| match self.points.get(serial_id as usize) {
                Some(Some((version, _))) => Some(VectorId::new(serial_id, *version)),
                _ => None,
            })
            .collect_vec()
    }

    pub fn from_0_indices(&self, indices: &[u32]) -> Vec<VectorId> {
        indices
            .iter()
            .map(|index| {
                let v = VectorId::from_0_index(*index);
                if let Some(version) = self.get_current_version(v.serial_id()) {
                    VectorId::new(v.serial_id(), version)
                } else {
                    v
                }
            })
            .collect_vec()
    }
}

/// Reference to inserted irises.
#[derive(Clone)]
pub struct SharedIrisesRef<I: Clone> {
    pub(crate) data: Arc<RwLock<SharedIrises<I>>>,
}

impl<I: Clone> std::fmt::Debug for SharedIrisesRef<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt("SharedIrisesRef", f)
    }
}

// Getters, iterators and mutators of the iris storage.
impl<I: Clone> SharedIrisesRef<I> {
    pub async fn write(&self) -> RwLockWriteGuard<'_, SharedIrises<I>> {
        self.data.write().await
    }

    pub async fn read(&self) -> RwLockReadGuard<'_, SharedIrises<I>> {
        self.data.read().await
    }

    pub async fn get_vector_id(&self, serial_id: SerialId) -> Option<VectorId> {
        *self.get_vector_ids(&[serial_id]).await.first().unwrap()
    }

    pub async fn get_vector_or_empty(&self, vector: &VectorId) -> I {
        self.data.read().await.get_vector_or_empty(vector)
    }

    pub async fn get_vector(&self, vector: &VectorId) -> Option<I> {
        self.data.read().await.get_vector(vector)
    }

    pub async fn get_vector_ids(&self, serial_ids: &[SerialId]) -> Vec<Option<VectorId>> {
        let body = self.data.read().await;

        serial_ids
            .iter()
            .map(|serial_id| {
                body.get_current_version(*serial_id)
                    .map(|version_id| VectorId::new(*serial_id, version_id))
            })
            .collect()
    }

    pub async fn get_vectors(
        &self,
        vector_ids: impl IntoIterator<Item = &VectorId>,
    ) -> Vec<Option<I>> {
        let body = self.data.read().await;
        vector_ids
            .into_iter()
            .map(|v| body.get_vector(v))
            .collect_vec()
    }

    pub async fn get_vectors_or_empty(
        &self,
        vector_ids: impl IntoIterator<Item = &VectorId>,
    ) -> Vec<I> {
        let body = self.data.read().await;
        vector_ids
            .into_iter()
            .map(|v| body.get_vector_or_empty(v))
            .collect_vec()
    }

    /// Obtain a write lock for the underlying irises data, and insert the given
    /// `query` iris at the specified `id`.
    ///
    /// Returns the `VectorId` at which the query is inserted.
    pub async fn insert(&mut self, id: VectorId, iris_ref: &I) -> VectorId {
        self.data.write().await.insert(id, iris_ref.clone())
    }

    /// Obtain a write lock for the underlying irises data, and insert the given
    /// `query` iris at the next unused `VectorId` serial number, with version 0.
    ///
    /// Returns the `VectorId` at which the query is inserted.
    pub async fn append(&mut self, iris_ref: &I) -> VectorId {
        self.data.write().await.append(iris_ref.clone())
    }

    /// Obtain a write lock for the underlying irises data, and insert the given
    /// `query` iris at the id `original_id.next_version()`, that is, the `VectorId`
    /// with equal serial id and incremented version number.
    ///
    /// Returns the `VectorId` at which the query is inserted.
    pub async fn update(&mut self, original_id: VectorId, iris_ref: &I) -> VectorId {
        self.data
            .write()
            .await
            .update(original_id, iris_ref.clone())
    }

    pub async fn checksum(&self) -> u64 {
        self.data.read().await.set_hash.checksum()
    }
}
