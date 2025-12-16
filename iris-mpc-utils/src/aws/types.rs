use std::fmt;

use serde::ser::Serialize;
use serde_json;

// Helper type encpasulating AWS-S3 object information.
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
    pub fn body(&self) -> &Vec<u8> {
        &self.body
    }

    pub fn bucket(&self) -> &String {
        &self.bucket
    }

    pub fn key(&self) -> &String {
        &self.key
    }

    pub fn new<T>(bucket: &String, key: &String, body: &T) -> Self
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

// Helper type encpasulating an AWS-SQS message.
#[derive(Debug)]
pub struct SnsMessageInfo {
    // SNS message body - a JSON encoded string.
    body: String,

    // SNS message group - e.g. "enrollment".
    group_id: String,

    // SNS message kind - e.g. "uniqueness".
    kind: String,
}

impl SnsMessageInfo {
    pub fn body(&self) -> &String {
        &self.body
    }

    pub fn group_id(&self) -> &String {
        &self.group_id
    }

    pub fn kind(&self) -> &String {
        &self.kind
    }

    pub fn new<T>(group_id: &str, kind: &str, body: &T) -> Self
    where
        T: ?Sized + Serialize,
    {
        Self {
            body: serde_json::to_string(body).unwrap(),
            group_id: String::from(group_id),
            kind: String::from(kind),
        }
    }
}

impl fmt::Display for SnsMessageInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}.{}", self.group_id, self.kind)
    }
}

// Helper type encpasulating an AWS-SQS message.
#[derive(Debug)]
pub struct SqsMessageInfo {
    // SNS message body - a JSON encoded string.
    body: String,

    // SQS message kind - e.g. "uniqueness".
    kind: String,

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

    pub fn receipt_handle(&self) -> &str {
        &self.receipt_handle
    }

    pub fn new(kind: String, body: String, receipt_handle: String) -> Self {
        Self {
            body,
            kind,
            receipt_handle,
        }
    }
}

impl fmt::Display for SqsMessageInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}
