use crate::protocol::shared_iris::GaloisRingSharedIris;
use iris_mpc_store::DbStoredIris;

// An iris pair identifier.
pub type IrisSerialId = u64;

// Galois field element shares over raw iris biometric data.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct IrisGaloisShares {
    // Left share of the iris Galois field element.
    left: GaloisRingSharedIris,

    // Right share of the iris Galois field element.
    right: GaloisRingSharedIris,
}

// Accessors.
#[allow(dead_code)]
impl IrisGaloisShares {
    pub fn left(&self) -> &GaloisRingSharedIris {
        &self.left
    }

    pub fn right(&self) -> &GaloisRingSharedIris {
        &self.right
    }
}

// Constructors.
#[allow(dead_code)]
impl IrisGaloisShares {
    /// Creates a new instance of `IrisGaloisShares` from data fetched from a store.
    pub(crate) fn new(party_id: usize, data: &DbStoredIris) -> Self {
        let left = GaloisRingSharedIris::try_from_buffers_inner(
            party_id,
            data.left_code(),
            data.left_mask(),
        )
        .unwrap();
        let right = GaloisRingSharedIris::try_from_buffers_inner(
            party_id,
            data.right_code(),
            data.right_mask(),
        )
        .unwrap();

        Self { left, right }
    }
}
