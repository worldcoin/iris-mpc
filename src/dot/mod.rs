pub mod device_manager;
pub mod distance_comparator;
pub mod share_db;

pub const IRIS_CODE_LENGTH: usize = 12_800;
pub const ROTATIONS: usize = 31;
pub(crate) const P: u16 = 65519;
