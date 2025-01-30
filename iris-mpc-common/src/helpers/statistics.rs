use chrono::{
    serde::{ts_seconds, ts_seconds_option},
    DateTime, Utc,
};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Eye {
    Left,
    Right,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketResult {
    pub count:                   usize,
    pub hamming_distance_bucket: [f64; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketStatistics {
    pub buckets: Vec<BucketResult>,
    pub n_buckets: usize,
    pub match_distances_buffer_size: usize,
    pub party_id: usize,
    pub eye: Eye,
    #[serde(with = "ts_seconds")]
    pub start_timestamp: DateTime<Utc>,
    #[serde(with = "ts_seconds_option")]
    pub end_timestamp: Option<DateTime<Utc>>,
    #[serde(skip_serializing)]
    #[serde(with = "ts_seconds_option")]
    pub next_start_timestamp: Option<DateTime<Utc>>,
}

impl BucketStatistics {
    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }
}

impl fmt::Display for BucketStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for bucket in &self.buckets {
            writeln!(
                f,
                "    {:.3}-{:.3}: {}",
                bucket.hamming_distance_bucket[0], bucket.hamming_distance_bucket[1], bucket.count
            )?;
        }
        Ok(())
    }
}

impl BucketStatistics {
    /// Create a new `BucketStatistics` with your desired metadata.
    pub fn new(
        match_distances_buffer_size: usize,
        n_buckets: usize,
        party_id: usize,
        eye: Eye,
    ) -> Self {
        Self {
            buckets: Vec::with_capacity(n_buckets),
            n_buckets,
            eye,
            match_distances_buffer_size,
            party_id,
            start_timestamp: Utc::now(),
            end_timestamp: None,
            next_start_timestamp: None,
        }
    }

    /// `buckets_array` array of buckets
    /// `buckets`, which for i=0..n_buckets might be a cumulative count (or
    /// partial sum).
    ///
    /// `match_threshold_ratio` is the upper bound you used for the last bucket.
    /// If e.g. you want hamming-distance thresholds in [0.0,
    /// MATCH_THRESHOLD_RATIO], we subdivide that interval by `n_buckets`.
    pub fn fill_buckets(
        &mut self,
        buckets_array: &[u32],
        match_threshold_ratio: f64,
        start_timestamp: Option<DateTime<Utc>>,
    ) {
        tracing::info!("Filling buckets: {:?}", buckets_array);

        let now_timestamp = Utc::now();

        // clear just in case, we already clear it on sending the message
        self.buckets.clear();
        self.end_timestamp = Some(now_timestamp);

        let step = match_threshold_ratio / (self.n_buckets as f64);
        for i in 0..buckets_array.len() {
            let previous_threshold = step * (i as f64);
            let threshold = step * ((i + 1) as f64);

            // The difference between buckets[i] and buckets[i - 1], except when i=0
            let previous_count = if i == 0 { 0 } else { buckets_array[i - 1] };
            let count = buckets_array[i].saturating_sub(previous_count);

            self.buckets.push(BucketResult {
                hamming_distance_bucket: [previous_threshold, threshold],
                count:                   count as usize,
            });
        }

        // If the start timestamp is provided, we use it as the start timestamp,
        // otherwise, it means it was the first iteration (ServerActor
        // instantiation)
        if let Some(start_timestamp) = start_timestamp {
            self.start_timestamp = start_timestamp;
        }
        // Set the next start timestamp to now
        self.next_start_timestamp = Some(now_timestamp);
    }
}
