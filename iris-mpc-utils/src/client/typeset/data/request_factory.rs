use uuid;

use iris_mpc_common::{helpers::smpc_request, IrisSerialId};

use super::{request_info::RequestInfo, Request, RequestBatch};

/// A set of variants over an associated uniqueness request. Pertinent when creating requests
/// of the following types: identity_deletion ^ reauth ^ reset_update.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum ParentUniquenessRequest {
    // A concrete uniqueness request instance.
    Instance(Request),
    // A serial identifier correlated with a previously processed uniqueness request.
    IrisSerialId(IrisSerialId),
}

impl ParentUniquenessRequest {
    fn is_valid(kind: &str, parent: &Option<Self>) -> bool {
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

    pub fn new_maybe(
        batch: &RequestBatch,
        kind: &str,
        serial_id: Option<IrisSerialId>,
    ) -> Option<Self> {
        match kind {
            smpc_request::RESET_CHECK_MESSAGE_TYPE | smpc_request::UNIQUENESS_MESSAGE_TYPE => None,
            smpc_request::IDENTITY_DELETION_MESSAGE_TYPE
            | smpc_request::REAUTH_MESSAGE_TYPE
            | smpc_request::RESET_UPDATE_MESSAGE_TYPE => match serial_id {
                None => Some(Self::Instance(RequestFactory::new_uniqueness(batch))),
                Some(serial_id) => Some(Self::IrisSerialId(serial_id)),
            },
            _ => panic!("Invalid request kind"),
        }
    }
}

/// Encapsulates instantion of request variants.
pub struct RequestFactory {}

impl RequestFactory {
    pub fn new_request(
        batch: &RequestBatch,
        kind: &str,
        parent: Option<ParentUniquenessRequest>,
    ) -> Request {
        if ParentUniquenessRequest::is_valid(kind, &parent) {
            match kind {
                smpc_request::IDENTITY_DELETION_MESSAGE_TYPE => {
                    Self::new_identity_deletion(batch, parent.unwrap())
                }
                smpc_request::REAUTH_MESSAGE_TYPE => {
                    Self::new_reauthorisation(batch, parent.unwrap())
                }
                smpc_request::RESET_CHECK_MESSAGE_TYPE => Self::new_reset_check(batch),
                smpc_request::RESET_UPDATE_MESSAGE_TYPE => {
                    Self::new_reset_update(batch, parent.unwrap())
                }
                smpc_request::UNIQUENESS_MESSAGE_TYPE => Self::new_uniqueness(batch),
                _ => unreachable!(),
            }
        } else {
            panic!("Invalid parent request association");
        }
    }

    pub fn new_identity_deletion(batch: &RequestBatch, parent: ParentUniquenessRequest) -> Request {
        match parent {
            ParentUniquenessRequest::Instance(parent) => Request::IdentityDeletion {
                info: RequestInfo::new(batch, Some(parent.request_id())),
                uniqueness_serial_id: None,
            },
            ParentUniquenessRequest::IrisSerialId(serial_id) => Request::IdentityDeletion {
                info: RequestInfo::new(batch, None),
                uniqueness_serial_id: Some(serial_id),
            },
        }
    }

    pub fn new_reauthorisation(batch: &RequestBatch, parent: ParentUniquenessRequest) -> Request {
        match parent {
            ParentUniquenessRequest::Instance(parent) => Request::Reauthorization {
                info: RequestInfo::new(batch, Some(parent.request_id())),
                reauth_id: uuid::Uuid::new_v4(),
                uniqueness_serial_id: None,
            },
            ParentUniquenessRequest::IrisSerialId(serial_id) => Request::Reauthorization {
                info: RequestInfo::new(batch, None),
                reauth_id: uuid::Uuid::new_v4(),
                uniqueness_serial_id: Some(serial_id),
            },
        }
    }

    pub fn new_reset_check(batch: &RequestBatch) -> Request {
        Request::ResetCheck {
            info: RequestInfo::new(batch, None),
            reset_id: uuid::Uuid::new_v4(),
        }
    }

    pub fn new_reset_update(batch: &RequestBatch, parent: ParentUniquenessRequest) -> Request {
        match parent {
            ParentUniquenessRequest::Instance(parent) => Request::ResetUpdate {
                info: RequestInfo::new(batch, Some(parent.request_id())),
                reset_id: uuid::Uuid::new_v4(),
                uniqueness_serial_id: None,
            },
            ParentUniquenessRequest::IrisSerialId(serial_id) => Request::ResetUpdate {
                info: RequestInfo::new(batch, None),
                reset_id: uuid::Uuid::new_v4(),
                uniqueness_serial_id: Some(serial_id),
            },
        }
    }

    pub fn new_uniqueness(batch: &RequestBatch) -> Request {
        Request::Uniqueness {
            info: RequestInfo::new(batch, None),
            signup_id: uuid::Uuid::new_v4(),
        }
    }
}
