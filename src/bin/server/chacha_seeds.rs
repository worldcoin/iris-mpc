use core::sync::atomic::Ordering::SeqCst;
use lazy_static::lazy_static;
use ring::hkdf::{Algorithm, Okm, Salt, HKDF_SHA256};
use std::sync::atomic::AtomicUsize;

lazy_static! {
    static ref KDF_NONCE: AtomicUsize = AtomicUsize::new(0);
    static ref KDF_SALT: Salt = Salt::new(HKDF_SHA256, b"IRIS_MPC");
}

/// Internal helper function to derive a new seed from the given seed and nonce.
fn derive_seed(seed: [u32; 8], nonce: usize) -> eyre::Result<[u32; 8]> {
    let pseudo_rand_key = KDF_SALT.extract(bytemuck::cast_slice(&seed));
    let nonce = nonce.to_be_bytes();
    let context = vec![nonce.as_slice()];
    let output_key_material: Okm<Algorithm> =
        pseudo_rand_key.expand(&context, HKDF_SHA256).unwrap();
    let mut result = [0u32; 8];
    output_key_material
        .fill(bytemuck::cast_slice_mut(&mut result))
        .unwrap();
    Ok(result)
}

/// Applies a KDF to the given seeds to derive new seeds.
pub fn next_chacha_seeds(seeds: ([u32; 8], [u32; 8])) -> eyre::Result<([u32; 8], [u32; 8])> {
    let nonce = KDF_NONCE.fetch_add(1, SeqCst);
    Ok((derive_seed(seeds.0, nonce)?, derive_seed(seeds.1, nonce)?))
}
