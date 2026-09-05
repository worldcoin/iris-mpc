//! Minimal data containers for compiling the production kernels without the
//! service dependency graph. MixedPlaneIris and its packing come from the PR.
pub mod shared_iris {
    use crate::{Record, CODE_LEN, MASK_LEN};
    use std::sync::Arc;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Share {
        pub id: usize,
        pub coefs: Vec<u16>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct GaloisRingSharedIris {
        pub code: Share,
        pub mask: Share,
    }

    pub type ArcIris = Arc<GaloisRingSharedIris>;

    impl GaloisRingSharedIris {
        pub fn default_for_party(party: usize) -> Self {
            Self {
                code: Share {
                    id: party + 1,
                    coefs: vec![0; CODE_LEN],
                },
                mask: Share {
                    id: party + 1,
                    coefs: vec![0; MASK_LEN],
                },
            }
        }

        pub(crate) fn from_record(record: &Record) -> Self {
            Self {
                code: Share {
                    id: 1,
                    coefs: record.code.clone(),
                },
                mask: Share {
                    id: 1,
                    coefs: record.mask.clone(),
                },
            }
        }
    }

    include!(concat!(env!("OUT_DIR"), "/mixed_iris.rs"));
}
