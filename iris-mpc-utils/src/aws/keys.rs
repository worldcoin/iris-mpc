use base64::{engine::general_purpose, Engine};
use eyre::Result;
use sodiumoxide::crypto::box_::PublicKey;
use thiserror::Error;

use iris_mpc_common::helpers::key_pair;

use crate::{
    constants::{PARTY_IDX_0, PARTY_IDX_1, PARTY_IDX_2},
    types::PublicKeyset,
};

#[derive(Error, Debug)]
#[allow(clippy::enum_variant_names)]
pub enum PublicKeyError {
    #[error("Download public key decoding error: {0}")]
    DecodeError(String),

    #[error("Download public key error: {0}")]
    DownloadError(String),

    #[error("Public key format error: {0}")]
    FormatError(String),
}

/// Returns downloaded & deserialised MPC party public keyset.
pub(super) async fn download_public_keyset(base_url: &str) -> Result<PublicKeyset> {
    async fn download_public_key(
        party_idx: usize,
        base_url: &str,
    ) -> Result<PublicKey, PublicKeyError> {
        let pbk_b64 = key_pair::download_public_key(base_url.to_owned(), party_idx.to_string())
            .await
            .map_err(|e| {
                tracing::error!("MPC party {} public key download error: {}", party_idx, e);
                PublicKeyError::DownloadError(e.to_string())
            })?;
        tracing::info!("MPC party {} public key downloaded", party_idx);

        let pbk_bytes = general_purpose::STANDARD
            .decode(pbk_b64)
            .map_err(|e| PublicKeyError::DecodeError(e.to_string()))?;

        PublicKey::from_slice(&pbk_bytes).ok_or_else(|| {
            PublicKeyError::FormatError(format!(
                "Invalid public key format for party {}",
                party_idx
            ))
        })
    }

    let (key0, key1, key2) = tokio::join!(
        download_public_key(PARTY_IDX_0, base_url),
        download_public_key(PARTY_IDX_1, base_url),
        download_public_key(PARTY_IDX_2, base_url)
    );

    Ok([key0?, key1?, key2?])
}
