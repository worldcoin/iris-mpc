use uuid;

use iris_mpc_common::{
    helpers::smpc_request::{
        IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE,
        RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
    },
    IrisSerialId,
};

use super::{request::Request, request_batch::RequestBatch, request_info::RequestInfo};

/// Set of variants over a request association.
enum AssociatedRequest {
    // A request that has no associations.
    None,
    // A request that is associated with a uniqueness request derived from it's Iris serial id.
    UniquenessFromIrisSerialId(IrisSerialId),
    // A request that is associated with a new uniqueness request.
    UniquenessFromNew,
}

pub struct RequestFactory {}

impl RequestFactory {
    /// Returns a two member tuple of a request plus maybe it's associated parent.
    /// The tuple is primarily derived from the request kind.
    pub fn new_from_kind(
        batch: &RequestBatch,
        kind: &str,
        parent_iris_serial_id: Option<IrisSerialId>,
    ) -> (Request, Option<Request>) {
        let associated = match kind {
            IDENTITY_DELETION_MESSAGE_TYPE | REAUTH_MESSAGE_TYPE | RESET_UPDATE_MESSAGE_TYPE => {
                if let Some(parent_iris_serial_id) = parent_iris_serial_id {
                    AssociatedRequest::UniquenessFromIrisSerialId(parent_iris_serial_id)
                } else {
                    AssociatedRequest::UniquenessFromNew
                }
            }
            RESET_CHECK_MESSAGE_TYPE | UNIQUENESS_MESSAGE_TYPE => AssociatedRequest::None,
            _ => panic!("Invalid batch kind"),
        };

        match associated {
            AssociatedRequest::None => match kind {
                RESET_CHECK_MESSAGE_TYPE => (
                    Request::ResetCheck {
                        info: RequestInfo::new(batch, None),
                        reset_id: uuid::Uuid::new_v4(),
                    },
                    None,
                ),
                UNIQUENESS_MESSAGE_TYPE => (Request::new_uniqueness(batch), None),
                _ => unreachable!(),
            },
            AssociatedRequest::UniquenessFromIrisSerialId(serial_id) => match kind {
                IDENTITY_DELETION_MESSAGE_TYPE => (
                    Request::IdentityDeletion {
                        info: RequestInfo::new(batch, None),
                        uniqueness_serial_id: Some(serial_id),
                    },
                    None,
                ),
                REAUTH_MESSAGE_TYPE => (
                    Request::Reauthorization {
                        info: RequestInfo::new(batch, None),
                        reauth_id: uuid::Uuid::new_v4(),
                        uniqueness_serial_id: Some(serial_id),
                    },
                    None,
                ),
                RESET_UPDATE_MESSAGE_TYPE => (
                    Request::ResetUpdate {
                        info: RequestInfo::new(batch, None),
                        reset_id: uuid::Uuid::new_v4(),
                        uniqueness_serial_id: Some(serial_id),
                    },
                    None,
                ),
                _ => unreachable!(),
            },
            AssociatedRequest::UniquenessFromNew => {
                let parent = Request::new_uniqueness(batch);
                match kind {
                    IDENTITY_DELETION_MESSAGE_TYPE => (
                        Request::IdentityDeletion {
                            info: RequestInfo::new(batch, Some(&parent)),
                            uniqueness_serial_id: None,
                        },
                        Some(parent),
                    ),
                    REAUTH_MESSAGE_TYPE => (
                        Request::Reauthorization {
                            info: RequestInfo::new(batch, Some(&parent)),
                            reauth_id: uuid::Uuid::new_v4(),
                            uniqueness_serial_id: None,
                        },
                        Some(parent),
                    ),
                    RESET_UPDATE_MESSAGE_TYPE => (
                        Request::ResetUpdate {
                            info: RequestInfo::new(batch, Some(&parent)),
                            reset_id: uuid::Uuid::new_v4(),
                            uniqueness_serial_id: None,
                        },
                        Some(parent),
                    ),
                    _ => unreachable!(),
                }
            }
        }
    }
}
