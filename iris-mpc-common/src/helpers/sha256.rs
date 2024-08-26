use sha2::{Digest, Sha256};

pub fn calculate_sha256<T: AsRef<[u8]>>(data: T) -> String {
    hex::encode(Sha256::digest(data.as_ref()))
}
