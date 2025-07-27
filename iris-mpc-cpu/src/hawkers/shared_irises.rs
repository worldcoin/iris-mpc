use std::{collections::HashMap, sync::Arc};

use iris_mpc_common::vector_id::{SerialId, VectorId, VersionId};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::{
    execution::hawk_main::state_check::SetHash, hawkers::aby3::aby3_store::Aby3Query,
    protocol::shared_iris::GaloisRingSharedIris,
};

type IrisRef = Arc<GaloisRingSharedIris>;

/// Storage of inserted irises.
#[derive(Clone, Serialize, Deserialize)]
pub struct SharedIrises {
    pub points: HashMap<SerialId, (VersionId, IrisRef)>,
    pub next_id: u32,
    pub empty_iris: IrisRef,
    pub set_hash: SetHash,
}

impl Default for SharedIrises {
    fn default() -> Self {
        SharedIrises::new(HashMap::new())
    }
}

impl SharedIrises {
    pub fn new(points: HashMap<VectorId, IrisRef>) -> Self {
        let next_id = points.keys().map(|v| v.serial_id()).max().unwrap_or(0) + 1;

        let points = points
            .into_iter()
            .map(|(v, iris)| (v.serial_id(), (v.version_id(), iris)))
            .collect::<HashMap<_, _>>();

        SharedIrises {
            points,
            next_id,
            empty_iris: Arc::new(GaloisRingSharedIris::default_for_party(0)),
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
    pub fn insert(&mut self, vector_id: VectorId, iris: IrisRef) -> VectorId {
        let prev_entry = self
            .points
            .insert(vector_id.serial_id(), (vector_id.version_id(), iris));

        self.next_id = self.next_id.max(vector_id.serial_id() + 1);

        // If overwriting entry, remove previous vector id from set_hash
        if let Some((version, _)) = prev_entry {
            let prev_vector_id = VectorId::new(vector_id.serial_id(), version);
            self.set_hash.remove(prev_vector_id);
        }
        self.set_hash.add_unordered(vector_id);

        vector_id
    }

    /// Insert the given iris at the next unused serial ID, with version
    /// initialized to 0.
    pub fn append(&mut self, iris: IrisRef) -> VectorId {
        let new_id = self.next_id();
        self.insert(new_id, iris);
        new_id
    }

    /// Insert the given iris at ID given by `original_id.next_version()`, i.e.
    /// with identical serial number, and one higher version number.
    pub fn update(&mut self, original_id: VectorId, iris: IrisRef) -> VectorId {
        let new_id = original_id.next_version();
        self.insert(new_id, iris);
        new_id
    }

    pub fn db_size(&self) -> usize {
        self.points.len()
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
        self.points.get(&serial_id).map(|(version, _iris)| *version)
    }

    fn get_vector_or_empty(&self, vector: &VectorId) -> IrisRef {
        match self.points.get(&vector.serial_id()) {
            Some((version, iris)) if vector.version_matches(*version) => Arc::clone(iris),
            _ => Arc::clone(&self.empty_iris),
        }
    }

    fn get_vector(&self, vector: &VectorId) -> Option<IrisRef> {
        match self.points.get(&vector.serial_id()) {
            Some((version, iris)) if vector.version_matches(*version) => Some(Arc::clone(iris)),
            _ => None,
        }
    }

    pub fn contains(&self, vector: &VectorId) -> bool {
        matches!(self.points.get(&vector.serial_id()),
            Some((version, _)) if vector.version_matches(*version))
    }

    pub fn to_arc(self) -> SharedIrisesRef {
        SharedIrisesRef {
            data: Arc::new(RwLock::new(self)),
        }
    }

    pub fn last_vector_ids(&self, n: usize) -> Vec<VectorId> {
        (1..self.next_id)
            .rev()
            .take(n)
            .filter_map(|serial_id| {
                self.points
                    .get(&serial_id)
                    .map(|(version, _)| VectorId::new(serial_id, *version))
            })
            .collect_vec()
    }

    pub fn from_0_indices(&self, indices: &[u32]) -> Vec<VectorId> {
        indices
            .iter()
            .map(|index| {
                let v = VectorId::from_0_index(*index);
                if let Some((version, _)) = self.points.get(&v.serial_id()) {
                    VectorId::new(v.serial_id(), *version)
                } else {
                    v
                }
            })
            .collect_vec()
    }
}

/// Reference to inserted irises.
#[derive(Clone)]
pub struct SharedIrisesRef {
    pub data: Arc<RwLock<SharedIrises>>,
}

/// Mutable reference to inserted irises.
pub type SharedIrisesMut<'a> = RwLockWriteGuard<'a, SharedIrises>;

impl std::fmt::Debug for SharedIrisesRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt("SharedIrisesRef", f)
    }
}

// Getters, iterators and mutators of the iris storage.
impl SharedIrisesRef {
    pub async fn write(&self) -> SharedIrisesMut {
        self.data.write().await
    }

    pub async fn read(&self) -> RwLockReadGuard<'_, SharedIrises> {
        self.data.read().await
    }

    pub async fn get_vector_id(&self, serial_id: SerialId) -> Option<VectorId> {
        *self.get_vector_ids(&[serial_id]).await.first().unwrap()
    }

    pub async fn get_vector_or_empty(&self, vector: &VectorId) -> IrisRef {
        self.data.read().await.get_vector_or_empty(vector)
    }

    pub async fn get_vector(&self, vector: &VectorId) -> Option<IrisRef> {
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
    ) -> Vec<Option<IrisRef>> {
        let body = self.data.read().await;
        vector_ids
            .into_iter()
            .map(|v| body.get_vector(v))
            .collect_vec()
    }

    pub async fn get_vectors_or_empty(
        &self,
        vector_ids: impl IntoIterator<Item = &VectorId>,
    ) -> Vec<IrisRef> {
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
    pub async fn insert(&mut self, id: VectorId, query: &Aby3Query) -> VectorId {
        self.data.write().await.insert(id, query.iris.clone())
    }

    /// Obtain a write lock for the underlying irises data, and insert the given
    /// `query` iris at the next unused `VectorId` serial number, with version 0.
    ///
    /// Returns the `VectorId` at which the query is inserted.
    pub async fn append(&mut self, query: &Aby3Query) -> VectorId {
        self.data.write().await.append(query.iris.clone())
    }

    /// Obtain a write lock for the underlying irises data, and insert the given
    /// `query` iris at the id `original_id.next_version()`, that is, the `VectorId`
    /// with equal serial id and incremented version number.
    ///
    /// Returns the `VectorId` at which the query is inserted.
    pub async fn update(&mut self, original_id: VectorId, query: &Aby3Query) -> VectorId {
        self.data
            .write()
            .await
            .update(original_id, query.iris.clone())
    }

    pub async fn checksum(&self) -> u64 {
        self.data.read().await.set_hash.checksum()
    }
}
