use serde::ser::Serialize;
use serde_json;

// Helper type that encpasulates an AWS-SQS message.
pub struct S3ObjectInfo {
    body: Vec<u8>,
    bucket: String,
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

    pub fn new(bucket: String, key: String, body: Vec<u8>) -> Self {
        Self { body, bucket, key }
    }

    pub fn new_from_jsonic<T>(bucket: &String, key: &String, body: &T) -> Self
    where
        T: ?Sized + Serialize,
    {
        Self::new(
            bucket.to_owned(),
            key.to_owned(),
            serde_json::to_vec(body).unwrap(),
        )
    }
}

// Helper type that encpasulates an AWS-SQS message.
pub struct SnsMessageInfo<T>
where
    T: Sized + Serialize,
{
    body: T,
    group_id: String,
    kind: String,
}

impl<T> SnsMessageInfo<T>
where
    T: Sized + Serialize,
{
    pub fn body(&self) -> &T {
        &self.body
    }

    pub fn group_id(&self) -> &String {
        &self.group_id
    }

    pub fn kind(&self) -> &String {
        &self.kind
    }

    pub fn new(body: T, group_id: String, kind: String) -> Self {
        Self {
            body,
            group_id,
            kind,
        }
    }
}
