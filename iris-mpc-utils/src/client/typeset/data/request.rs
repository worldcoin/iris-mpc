use std::{fmt, time::Instant};

use uuid;

use iris_mpc_common::{helpers::smpc_request, IrisSerialId};

use super::{request_batch::RequestBatch, request_info::RequestInfo, response::ResponseBody};

/// Encapsulates data pertinent to a system processing request.
#[derive(Clone, Debug)]
pub enum Request {
    IdentityDeletion {
        // Standard request information.
        info: RequestInfo,
        // Iris serial identifier.
        uniqueness_serial_id: Option<IrisSerialId>,
    },
    Reauthorization {
        // Standard request information.
        info: RequestInfo,
        // Operation identifier.
        reauth_id: uuid::Uuid,
        // Iris serial identifier.
        uniqueness_serial_id: Option<IrisSerialId>,
    },
    ResetCheck {
        // Standard request information.
        info: RequestInfo,
        // Operation identifier.
        reset_id: uuid::Uuid,
    },
    ResetUpdate {
        // Standard request information.
        info: RequestInfo,
        // Operation identifier.
        reset_id: uuid::Uuid,
        // Iris serial identifier.
        uniqueness_serial_id: Option<IrisSerialId>,
    },
    Uniqueness {
        // Standard request information.
        info: RequestInfo,
        // Iris sign-up identifier.
        signup_id: uuid::Uuid,
    },
}

impl Request {
    pub fn info(&self) -> &RequestInfo {
        match self {
            Self::IdentityDeletion { info, .. }
            | Self::Reauthorization { info, .. }
            | Self::ResetCheck { info, .. }
            | Self::ResetUpdate { info, .. }
            | Self::Uniqueness { info, .. } => info,
        }
    }

    fn info_mut(&mut self) -> &mut RequestInfo {
        match self {
            Self::IdentityDeletion { info, .. }
            | Self::Reauthorization { info, .. }
            | Self::ResetCheck { info, .. }
            | Self::ResetUpdate { info, .. }
            | Self::Uniqueness { info, .. } => info,
        }
    }

    /// Returns identifier to be assigned to associated iris shares.
    pub fn iris_shares_id(&self) -> Option<&uuid::Uuid> {
        match self {
            Self::IdentityDeletion { .. } => None,
            Self::Reauthorization { reauth_id, .. } => Some(reauth_id),
            Self::ResetCheck { reset_id, .. } => Some(reset_id),
            Self::ResetUpdate { reset_id, .. } => Some(reset_id),
            Self::Uniqueness { signup_id, .. } => Some(signup_id),
        }
    }

    /// Returns true if a system response is deemed to be correlated with this system request.
    pub fn is_correlation(&self, response: &ResponseBody) -> bool {
        match (self, response) {
            (
                Self::IdentityDeletion {
                    uniqueness_serial_id: serial_id,
                    ..
                },
                ResponseBody::IdentityDeletion(result),
            ) => serial_id.unwrap() == result.serial_id,
            (Self::Reauthorization { reauth_id, .. }, ResponseBody::Reauthorization(result)) => {
                reauth_id.to_string() == result.reauth_id
            }
            (Self::ResetCheck { reset_id, .. }, ResponseBody::ResetCheck(result)) => {
                reset_id.to_string() == result.reset_id
            }
            (Self::ResetUpdate { reset_id, .. }, ResponseBody::ResetUpdate(result)) => {
                reset_id.to_string() == result.reset_id
            }
            (Self::Uniqueness { signup_id, .. }, ResponseBody::Uniqueness(result)) => {
                signup_id.to_string() == result.signup_id
            }
            _ => false,
        }
    }

    /// Returns true if request has been enqueued for system processing.
    pub fn is_correlated(&self) -> bool {
        self.info().is_correlated()
    }

    /// Returns true if request has been enqueued for system processing.
    pub fn is_enqueued(&self) -> bool {
        matches!(self.info().status(), RequestStatus::Enqueued(_))
    }

    /// Returns true if generated and not awaiting data returned from a parent request.
    pub fn is_enqueueable(&self) -> bool {
        matches!(self.info().status(), RequestStatus::SharesUploaded(_))
            && match self {
                Self::IdentityDeletion {
                    uniqueness_serial_id: serial_id,
                    ..
                } => serial_id.is_some(),
                Self::Reauthorization {
                    uniqueness_serial_id: serial_id,
                    ..
                } => serial_id.is_some(),
                Self::ResetUpdate {
                    uniqueness_serial_id: serial_id,
                    ..
                } => serial_id.is_some(),
                _ => true,
            }
    }

    fn label(&self) -> &str {
        match self {
            Self::IdentityDeletion { .. } => "IdentityDeletion",
            Self::Reauthorization { .. } => "Reauthorization",
            Self::ResetCheck { .. } => "ResetCheck",
            Self::ResetUpdate { .. } => "ResetUpdate",
            Self::Uniqueness { .. } => "Uniqueness",
        }
    }

    pub fn request_id(&self) -> &uuid::Uuid {
        self.info().request_id()
    }

    pub fn request_id_of_parent(&self) -> &Option<uuid::Uuid> {
        self.info().request_id_of_parent()
    }

    pub fn set_correlation(&mut self, response: &ResponseBody) {
        tracing::info!("{} :: Correlated -> Node-{}", &self, response.node_id());
        self.info_mut().set_correlation(response);
        if self.is_correlated() {
            self.set_status(RequestStatus::new_correlated());
        }
    }

    pub fn set_data_from_parent_response(&mut self, response: &ResponseBody) {
        match self {
            Self::IdentityDeletion {
                uniqueness_serial_id,
                ..
            } => match response {
                ResponseBody::Uniqueness(result) => {
                    *uniqueness_serial_id = result.serial_id;
                }
                _ => {}
            },
            Self::Reauthorization {
                uniqueness_serial_id,
                ..
            } => match response {
                ResponseBody::Uniqueness(result) => {
                    *uniqueness_serial_id = result.serial_id;
                }
                _ => {}
            },
            Self::ResetUpdate {
                uniqueness_serial_id,
                ..
            } => match response {
                ResponseBody::Uniqueness(result) => {
                    *uniqueness_serial_id = result.serial_id;
                }
                _ => {}
            },
            _ => panic!("Unsupported parent data"),
        }
    }

    pub fn set_status(&mut self, new_state: RequestStatus) {
        tracing::info!("{} :: State -> {}", &self, new_state);
        self.info_mut().set_status(new_state);
    }
}

impl fmt::Display for Request {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}.{}", self.info(), self.label())
    }
}

/// Enumeration over request message body for dispatch to system ingress queue.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum RequestBody {
    IdentityDeletion(smpc_request::IdentityDeletionRequest),
    Reauthorization(smpc_request::ReAuthRequest),
    ResetCheck(smpc_request::ResetCheckRequest),
    ResetUpdate(smpc_request::ResetUpdateRequest),
    Uniqueness(smpc_request::UniquenessRequest),
}

pub struct RequestFactory {}

impl RequestFactory {
    pub fn new_identity_deletion_1(batch: &RequestBatch, parent: &Request) -> Request {
        match parent {
            Request::Uniqueness { .. } => {}
            _ => panic!("Invalid request parent"),
        };

        Request::IdentityDeletion {
            info: RequestInfo::new(batch, Some(parent)),
            uniqueness_serial_id: None,
        }
    }

    pub fn new_identity_deletion_2(
        batch: &RequestBatch,
        known_iris_serial_id: IrisSerialId,
    ) -> Request {
        Request::IdentityDeletion {
            info: RequestInfo::new(batch, None),
            uniqueness_serial_id: Some(known_iris_serial_id),
        }
    }

    pub fn new_reauthorisation_1(batch: &RequestBatch, parent: &Request) -> Request {
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

    pub fn new_reauthorisation_2(
        batch: &RequestBatch,
        known_iris_serial_id: IrisSerialId,
    ) -> Request {
        Request::Reauthorization {
            info: RequestInfo::new(batch, None),
            reauth_id: uuid::Uuid::new_v4(),
            uniqueness_serial_id: Some(known_iris_serial_id),
        }
    }

    pub fn new_reset_check(batch: &RequestBatch) -> Request {
        Request::ResetCheck {
            info: RequestInfo::new(batch, None),
            reset_id: uuid::Uuid::new_v4(),
        }
    }

    pub fn new_reset_update_1(batch: &RequestBatch, parent: &Request) -> Request {
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

    pub fn new_reset_update_2(batch: &RequestBatch, known_iris_serial_id: IrisSerialId) -> Request {
        Request::ResetUpdate {
            info: RequestInfo::new(batch, None),
            reset_id: uuid::Uuid::new_v4(),
            uniqueness_serial_id: Some(known_iris_serial_id),
        }
    }

    pub fn new_uniqueness(batch: &RequestBatch) -> Request {
        Request::Uniqueness {
            info: RequestInfo::new(batch, None),
            signup_id: uuid::Uuid::new_v4(),
        }
    }
}

/// Enumeration over request processing states plus time of state change for statisitical purposes.
#[derive(Debug, Clone)]
pub enum RequestStatus {
    // Associated responses have been correlated.
    Correlated(Instant),
    // Enqueued upon system ingress queue.
    Enqueued(Instant),
    // Newly generated by client.
    New(Instant),
    // Iris shares uploaded in advance of enqueuement.
    SharesUploaded(Instant),
}

impl RequestStatus {
    pub const VARIANT_COUNT: usize = 4;

    pub fn new_correlated() -> Self {
        Self::Correlated(Instant::now())
    }

    pub fn new_enqueued() -> Self {
        Self::Enqueued(Instant::now())
    }

    pub fn new_shares_uploaded() -> Self {
        Self::SharesUploaded(Instant::now())
    }

    fn label(&self) -> &str {
        match self {
            Self::Correlated(_) => "Correlated",
            Self::Enqueued(_) => "Enqueued",
            Self::New(_) => "New",
            Self::SharesUploaded(_) => "SharesUploaded",
        }
    }
}

impl Default for RequestStatus {
    fn default() -> Self {
        Self::New(Instant::now())
    }
}

impl fmt::Display for RequestStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}
