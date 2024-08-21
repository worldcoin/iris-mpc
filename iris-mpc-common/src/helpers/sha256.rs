use sha2::{Digest, Sha256};
use std::fmt::Write;
pub fn calculate_sha256<T: AsRef<[u8]>>(data: T) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();

    let mut hex_string = String::new();
    for byte in result {
        write!(&mut hex_string, "{:02x}", byte).expect("Unable to write");
    }
    hex_string
}
