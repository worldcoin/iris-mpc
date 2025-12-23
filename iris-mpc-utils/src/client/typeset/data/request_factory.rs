use uuid;

use iris_mpc_common::{helpers::smpc_request, IrisSerialId};

/// A set of variants over an associated uniqueness request. Pertinent when creating requests
/// of the following types: identity_deletion ^ reauth ^ reset_update.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum ParentUniquenessRequest {
    // A concrete uniqueness request instance uuid.
    RequestUuid(uuid::Uuid),
    // A serial identifier correlated with a previously processed uniqueness request.
    IrisSerialId(IrisSerialId),
}

impl ParentUniquenessRequest {
    pub fn is_valid(kind: &str, parent: &Option<Self>) -> bool {
        match kind {
            smpc_request::RESET_CHECK_MESSAGE_TYPE | smpc_request::UNIQUENESS_MESSAGE_TYPE => {
                parent.is_none()
            }
            smpc_request::IDENTITY_DELETION_MESSAGE_TYPE
            | smpc_request::REAUTH_MESSAGE_TYPE
            | smpc_request::RESET_UPDATE_MESSAGE_TYPE => parent.is_some(),
            _ => false,
        }
    }
}
