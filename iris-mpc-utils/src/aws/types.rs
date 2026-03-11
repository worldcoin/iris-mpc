use std::fmt;

use iris_mpc_common::helpers::smpc_request;
use serde::ser::Serialize;
use serde_json;
use uuid;

use crate::client::{Request, RequestPayload};

const ENROLLMENT_REQUEST_TYPE: &str = "enrollment";

// Helper type encapsulating AWS-S3 object information.
#[derive(Debug)]
pub struct S3ObjectInfo {
    // S3 object data.
    body: Vec<u8>,

    // S3 bucket name.
    bucket: String,

    // S3 key name.
    key: String,
}

impl S3ObjectInfo {
    pub fn body(&self) -> &[u8] {
        &self.body
    }

    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn new<T>(bucket: &str, key: &str, body: &T) -> Self
    where
        T: ?Sized + Serialize,
    {
        Self {
            body: serde_json::to_vec(body).unwrap(),
            bucket: bucket.to_owned(),
            key: key.to_owned(),
        }
    }
}

impl fmt::Display for S3ObjectInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.bucket)
    }
}

// Helper type encapsulating an AWS-SQS message.
#[derive(Debug)]
pub struct SnsMessageInfo {
    // SNS message body - a JSON encoded string.
    body: String,

    // SNS message group - e.g. "enrollment".
    group_id: String,

    // SNS message kind - e.g. "uniqueness".
    kind: String,

    // SNS message deduplication ID for FIFO topics.
    deduplication_id: String,
}

impl SnsMessageInfo {
    pub fn body(&self) -> &str {
        &self.body
    }

    pub fn group_id(&self) -> &str {
        &self.group_id
    }

    pub fn kind(&self) -> &str {
        &self.kind
    }

    pub fn deduplication_id(&self) -> &str {
        &self.deduplication_id
    }

    pub fn new<T>(group_id: &str, kind: &str, body: &T, deduplication_id: String) -> Self
    where
        T: ?Sized + Serialize,
    {
        Self {
            body: serde_json::to_string(body).unwrap(),
            group_id: String::from(group_id),
            kind: String::from(kind),
            deduplication_id,
        }
    }
}

impl fmt::Display for SnsMessageInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}.{}", self.group_id, self.kind)
    }
}

// Helper type encapsulating an AWS-SQS message.
#[derive(Debug)]
pub struct SqsMessageInfo {
    // SNS message body - a JSON encoded string.
    body: String,

    // SQS message kind - e.g. "uniqueness".
    kind: String,

    // SQS queue URL from which the message was received.
    queue_url: String,

    // SQS message receipt handle for subsequent purging.
    receipt_handle: String,
}

impl SqsMessageInfo {
    pub fn body(&self) -> &str {
        &self.body
    }

    pub fn kind(&self) -> &str {
        &self.kind
    }

    pub fn queue_url(&self) -> &str {
        &self.queue_url
    }

    pub fn receipt_handle(&self) -> &str {
        &self.receipt_handle
    }

    pub fn new(kind: String, body: String, queue_url: String, receipt_handle: String) -> Self {
        Self {
            body,
            kind,
            queue_url,
            receipt_handle,
        }
    }
}

impl fmt::Display for SqsMessageInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl From<&Request> for SnsMessageInfo {
    fn from(request: &Request) -> Self {
        let dedup_id = match request {
            Request::IdentityDeletion { deletion_id, .. } => deletion_id.to_string(),
            Request::Reauthorization { reauth_id, .. } => reauth_id.to_string(),
            Request::RecoveryCheck { request_id, .. } => request_id.to_string(),
            Request::ResetCheck { reset_id, .. } => reset_id.to_string(),
            Request::ResetUpdate { reset_id, .. } => reset_id.to_string(),
            Request::RecoveryUpdate { recovery_id, .. } => recovery_id.to_string(),
            Request::Uniqueness { signup_id, .. } => signup_id.to_string(),
        };

        let payload = RequestPayload::from(request);
        Self::from_payload(payload, dedup_id)
    }
}

impl From<RequestPayload> for SnsMessageInfo {
    fn from(payload: RequestPayload) -> Self {
        // Generate a fresh UUID for deduplication when creating from payload directly
        let dedup_id = uuid::Uuid::new_v4().to_string();
        Self::from_payload(payload, dedup_id)
    }
}

impl SnsMessageInfo {
    fn from_payload(body: RequestPayload, dedup_id: String) -> Self {
        match body {
            RequestPayload::IdentityDeletion(body) => Self::new(
                ENROLLMENT_REQUEST_TYPE,
                smpc_request::IDENTITY_DELETION_MESSAGE_TYPE,
                &body,
                dedup_id,
            ),
            RequestPayload::Reauthorization(body) => Self::new(
                ENROLLMENT_REQUEST_TYPE,
                smpc_request::REAUTH_MESSAGE_TYPE,
                &body,
                dedup_id,
            ),
            RequestPayload::RecoveryCheck(body) => Self::new(
                ENROLLMENT_REQUEST_TYPE,
                smpc_request::RECOVERY_CHECK_MESSAGE_TYPE,
                &body,
                dedup_id,
            ),
            RequestPayload::ResetCheck(body) => Self::new(
                ENROLLMENT_REQUEST_TYPE,
                smpc_request::RESET_CHECK_MESSAGE_TYPE,
                &body,
                dedup_id,
            ),
            RequestPayload::ResetUpdate(body) => Self::new(
                ENROLLMENT_REQUEST_TYPE,
                smpc_request::RESET_UPDATE_MESSAGE_TYPE,
                &body,
                dedup_id,
            ),
            RequestPayload::RecoveryUpdate(body) => Self::new(
                ENROLLMENT_REQUEST_TYPE,
                smpc_request::RECOVERY_UPDATE_MESSAGE_TYPE,
                &body,
                dedup_id,
            ),
            RequestPayload::Uniqueness(body) => Self::new(
                ENROLLMENT_REQUEST_TYPE,
                smpc_request::UNIQUENESS_MESSAGE_TYPE,
                &body,
                dedup_id,
            ),
        }
    }
}
