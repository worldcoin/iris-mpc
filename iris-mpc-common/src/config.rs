// ...existing code...

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // ...existing code...
    #[serde(default = "default_batch_sync_delay_ms")]
    pub batch_sync_delay_ms: u64,

    #[serde(default = "default_sqs_poll_wait_seconds")]
    pub sqs_poll_wait_seconds: i32,

    #[serde(default = "default_no_messages_retry_delay_secs")]
    pub no_messages_retry_delay_secs: u64,
    // ...existing code...
}

// ...existing code...

fn default_batch_sync_delay_ms() -> u64 {
    50 // Reduced from 200ms to 50ms for faster synchronization
}

fn default_sqs_poll_wait_seconds() -> i32 {
    1 // Use 1 second long polling
}

fn default_no_messages_retry_delay_secs() -> u64 {
    1 // Reduced from 3 seconds to 1 second
}

// ...existing code...
