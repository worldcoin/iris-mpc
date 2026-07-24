//! Bounded retry for idempotent genesis operations.
//!
//! A single transient (an Aurora connection blip) otherwise aborts the whole
//! run — cheap to survive when the operation can be replayed cleanly.

use eyre::{eyre, Result};
use std::{future::Future, time::Duration};

/// Total attempts for the idempotent DB scans.
pub(super) const DB_RETRY_ATTEMPTS: u32 = 3;

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
