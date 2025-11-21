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

// Helper type encpasulating an AWS-SQS message.
pub struct SnsMessageInfo {
    body: Vec<u8>,
    group_id: String,
    kind: String,
}

impl SnsMessageInfo {
    pub fn body(&self) -> &Vec<u8> {
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
            body: serde_json::to_vec(body).unwrap(),
            group_id: String::from(group_id),
            kind: String::from(kind),
        }
    }
}
