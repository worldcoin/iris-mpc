use std::future::Future;
use std::time::Duration;

const MAX_BACKOFF: Duration = Duration::from_secs(5);

/// Retry a fallible async AWS operation with exponential backoff to absorb
/// transient failures (DNS/connection blips, throttling, brief 5xx) that the
/// SDK's own bounded retry window did not outlast. After `max_attempts` the
/// last error is returned, so a persistent failure still surfaces with the
/// original semantics rather than looping forever.
///
/// Callers must only pass operations that are safe to repeat (reads, or writes
/// the downstream deduplicates).
pub async fn retry_transient<T, E, F, Fut>(
    op_name: &str,
    max_attempts: u32,
    initial_backoff: Duration,
    mut op: F,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Debug,
{
    let mut attempt = 0u32;
    let mut backoff = initial_backoff;
    loop {
        attempt += 1;
        match op().await {
            Ok(value) => return Ok(value),
            Err(err) if attempt >= max_attempts => return Err(err),
            Err(err) => {
                tracing::warn!(
                    "{} failed (attempt {}/{}): {:?}. Retrying in {:?}...",
                    op_name,
                    attempt,
                    max_attempts,
                    err,
                    backoff
                );
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(MAX_BACKOFF);
            }
        }
    }
}
