use uuid;

use iris_mpc_common::{
    helpers::smpc_request::{
        IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE,
        RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
    },
    IrisSerialId,
};

use super::{request::Request, request_batch::RequestBatch, request_info::RequestInfo};

pub struct RequestFactory {}

impl RequestFactory {
    pub fn new_from_kind(
        batch: &RequestBatch,
        kind: &str,
        parent_iris_serial_id: Option<IrisSerialId>,
    ) -> (Request, Option<Request>) {
        match kind {
            IDENTITY_DELETION_MESSAGE_TYPE => {
                if let Some(parent_iris_serial_id) = parent_iris_serial_id {
                    (
                        RequestFactory::new_identity_deletion_2(batch, parent_iris_serial_id),
                        None,
                    )
                } else {
                    let r1 = RequestFactory::new_uniqueness(batch);
                    let r2 = RequestFactory::new_identity_deletion_1(batch, &r1);
                    (r1, Some(r2))
                }
            }
            RESET_CHECK_MESSAGE_TYPE => (RequestFactory::new_reset_check(batch), None),
            REAUTH_MESSAGE_TYPE => {
                if let Some(parent_iris_serial_id) = parent_iris_serial_id {
                    (
                        RequestFactory::new_reauthorisation_2(batch, parent_iris_serial_id),
                        None,
                    )
                } else {
                    let r1 = RequestFactory::new_uniqueness(batch);
                    let r2 = RequestFactory::new_reauthorisation_1(batch, &r1);
                    (r1, Some(r2))
                }
            }
            RESET_UPDATE_MESSAGE_TYPE => {
                if let Some(parent_iris_serial_id) = parent_iris_serial_id {
                    (
                        RequestFactory::new_reset_update_2(batch, parent_iris_serial_id),
                        None,
                    )
                } else {
                    let r1 = RequestFactory::new_uniqueness(batch);
                    let r2 = RequestFactory::new_reset_update_1(batch, &r1);
                    (r1, Some(r2))
                }
            }
            UNIQUENESS_MESSAGE_TYPE => (RequestFactory::new_uniqueness(batch), None),
            _ => panic!("Invalid batch kind"),
        }
    }

    fn new_identity_deletion_1(batch: &RequestBatch, parent: &Request) -> Request {
        match parent {
            Request::Uniqueness { .. } => {}
            _ => panic!("Invalid request parent"),
        };

        Request::IdentityDeletion {
            info: RequestInfo::new(batch, Some(parent)),
            uniqueness_serial_id: None,
        }
    }

    fn new_identity_deletion_2(
        batch: &RequestBatch,
        parent_iris_serial_id: IrisSerialId,
    ) -> Request {
        Request::IdentityDeletion {
            info: RequestInfo::new(batch, None),
            uniqueness_serial_id: Some(parent_iris_serial_id),
        }
    }

    fn new_reauthorisation_1(batch: &RequestBatch, parent: &Request) -> Request {
        match parent {
            Request::Uniqueness { .. } => {}
            _ => panic!("Invalid request parent"),
        };

        Request::Reauthorization {
            info: RequestInfo::new(batch, Some(parent)),
            reauth_id: uuid::Uuid::new_v4(),
            uniqueness_serial_id: None,
        }
    }

    fn new_reauthorisation_2(batch: &RequestBatch, parent_iris_serial_id: IrisSerialId) -> Request {
        Request::Reauthorization {
            info: RequestInfo::new(batch, None),
            reauth_id: uuid::Uuid::new_v4(),
            uniqueness_serial_id: Some(parent_iris_serial_id),
        }
    }

    fn new_reset_check(batch: &RequestBatch) -> Request {
        Request::ResetCheck {
            info: RequestInfo::new(batch, None),
            reset_id: uuid::Uuid::new_v4(),
        }
    }

    fn new_reset_update_1(batch: &RequestBatch, parent: &Request) -> Request {
        match parent {
            Request::Uniqueness { .. } => {}
            _ => panic!("Invalid request parent"),
        };

        Request::ResetUpdate {
            info: RequestInfo::new(batch, Some(parent)),
            reset_id: uuid::Uuid::new_v4(),
            uniqueness_serial_id: None,
        }
    }

    fn new_reset_update_2(batch: &RequestBatch, parent_iris_serial_id: IrisSerialId) -> Request {
        Request::ResetUpdate {
            info: RequestInfo::new(batch, None),
            reset_id: uuid::Uuid::new_v4(),
            uniqueness_serial_id: Some(parent_iris_serial_id),
        }
    }

    fn new_uniqueness(batch: &RequestBatch) -> Request {
        Request::Uniqueness {
            info: RequestInfo::new(batch, None),
            signup_id: uuid::Uuid::new_v4(),
        }
    }
}
