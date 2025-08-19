use crate::job::Eye;
use chrono::{
    serde::{ts_seconds, ts_seconds_option},
    DateTime, Utc,
};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketResult {
    pub count: usize,
    pub hamming_distance_bucket: [f64; 2],
}

impl Eq for BucketResult {}
impl PartialEq for BucketResult {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count
            && (self.hamming_distance_bucket[0] - other.hamming_distance_bucket[0]).abs() <= 1e-9
            && (self.hamming_distance_bucket[1] - other.hamming_distance_bucket[1]).abs() <= 1e-9
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BucketStatistics {
    pub buckets: Vec<BucketResult>,
    pub n_buckets: usize,
    // The number of matches gathered before sending the statistics
    pub match_distances_buffer_size: usize,
    pub party_id: usize,
    pub eye: Eye,
    #[serde(with = "ts_seconds")]
    // Start timestamp at which we start recording the statistics
    pub start_time_utc_timestamp: DateTime<Utc>,
    #[serde(with = "ts_seconds_option")]
    // End timestamp at which we stop recording the statistics
    pub end_time_utc_timestamp: Option<DateTime<Utc>>,
    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    #[serde(with = "ts_seconds_option")]
    pub next_start_time_utc_timestamp: Option<DateTime<Utc>>,
    // Flag to indicate if these statistics are from mirror orientation processing
    pub is_mirror_orientation: bool,
}

impl BucketStatistics {
    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }
}

impl fmt::Display for BucketStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    party_id: {}", self.party_id)?;
        writeln!(f, "    eye: {:?}", self.eye)?;
        writeln!(f, "    start_time_utc: {}", self.start_time_utc_timestamp)?;
        match &self.end_time_utc_timestamp {
            Some(end) => writeln!(f, "    end_time_utc: {}", end)?,
            None => writeln!(f, "    end_time_utc: <none>")?,
        }
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
            start_time_utc_timestamp: Utc::now(),
            end_time_utc_timestamp: None,
            next_start_time_utc_timestamp: None,
            is_mirror_orientation: false,
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
        self.end_time_utc_timestamp = Some(now_timestamp);

        let step = match_threshold_ratio / (self.n_buckets as f64);
        for i in 0..buckets_array.len() {
            let previous_threshold = step * (i as f64);
            let threshold = step * ((i + 1) as f64);

            // The difference between buckets[i] and buckets[i - 1], except when i=0
            let previous_count = if i == 0 { 0 } else { buckets_array[i - 1] };
            // Ensure non-decreasing cumulative counts to avoid underflow
            let count = if buckets_array[i] >= previous_count {
                buckets_array[i] - previous_count
            } else {
                tracing::warn!(
                    "Non-monotonic cumulative bucket counts at index {}: current {}, previous {}; clamping to zero",
                    i,
                    buckets_array[i],
                    previous_count,
                );
                0
            };

            self.buckets.push(BucketResult {
                hamming_distance_bucket: [previous_threshold, threshold],
                count: count as usize,
            });
        }

        // If the start timestamp is provided, we use it as the start timestamp,
        // otherwise, it means it was the first iteration (ServerActor
        // instantiation)
        if let Some(start_timestamp) = start_timestamp {
            self.start_time_utc_timestamp = start_timestamp;
        }
        // Set the next start timestamp to now
        self.next_start_time_utc_timestamp = Some(now_timestamp);
    }
}
