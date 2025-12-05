use std::fmt;

use serde::ser::Serialize;
use serde_json;

// Helper type encpasulating AWS-S3 object information.
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
        write!(f, "S3ObjectInfo:{}.{}", self.bucket, self.key)
    }
}

// Helper type encpasulating an AWS-SQS message.
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
        write!(f, "SnsMessageInfo:{}.{}", self.group_id, self.kind)
    }
}
