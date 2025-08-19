use sha2::{Digest, Sha256};

pub fn sha256_as_hex_string<T: AsRef<[u8]>>(data: T) -> String {
    hex::encode(sha256_bytes(data))
}

pub fn sha256_bytes<T: AsRef<[u8]>>(data: T) -> [u8; 32] {
    Sha256::digest(data.as_ref()).into()
}
