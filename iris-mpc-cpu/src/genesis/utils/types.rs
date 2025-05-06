use iris_mpc_store::DbStoredIris;

// An iris pair identifier.
pub type IrisSerialId = u64;

// An iris pair version identifier.
pub type IrisVersionId = i16;

// Iris code identifiers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IrisIdentifier {
    serial_id: IrisSerialId,
    version_id: IrisVersionId,
}

// Convertor: DbStoredIris -> IrisIdentifiers.
impl From<&DbStoredIris> for IrisIdentifier {
    fn from(value: &DbStoredIris) -> Self {
        IrisIdentifier {
            serial_id: value.serial_id() as IrisSerialId,
            version_id: value.version_id(),
        }
    }
}
