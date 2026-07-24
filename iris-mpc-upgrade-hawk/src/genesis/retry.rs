//! Bounded retry for idempotent genesis operations.
//!
//! A single transient (an Aurora connection blip, an S3 5xx) otherwise aborts
//! the whole run — cheap to survive when the operation can be replayed cleanly.

use eyre::{eyre, Result};
use std::{future::Future, time::Duration};

/// Total attempts for the DB scans and the deferred row/cursor writes.
pub(super) const DB_RETRY_ATTEMPTS: u32 = 3;
/// Total attempts for a whole checkpoint upload (fresh S3 key per attempt).
pub(super) const CHECKPOINT_UPLOAD_ATTEMPTS: u32 = 3;

/// Run `op` up to `attempts` times, sleeping with exponential backoff between
/// failures. The caller must guarantee `op` is idempotent: it re-runs from
/// scratch on every attempt.
pub(super) async fn with_retry<T, F, Fut>(label: &str, attempts: u32, mut op: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let attempts = attempts.max(1);
    let mut attempt = 1;
    loop {
        match op().await {
            Ok(value) => return Ok(value),
            Err(e) if attempt < attempts => {
                let delay = Duration::from_millis(500u64 << (attempt - 1).min(5));
                tracing::warn!(
                    "{label}: attempt {attempt}/{attempts} failed: {e:?}; retrying in {delay:?}"
                );
                tokio::time::sleep(delay).await;
                attempt += 1;
            }
            Err(e) => {
                return Err(eyre!("{label}: failed after {attempts} attempts: {e:?}"));
            }
        }
    }
}
