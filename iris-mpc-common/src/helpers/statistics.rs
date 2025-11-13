use crate::job::Eye;
use chrono::{
    serde::{ts_seconds, ts_seconds_option},
    DateTime, Utc,
};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnonStatsResultSource {
    Legacy,
    Aggregator,
}

impl Default for AnonStatsResultSource {
    fn default() -> Self {
        Self::Legacy
    }
}

// 1D anonymized statistics types
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
    #[serde(default)]
    pub source: AnonStatsResultSource,
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
        writeln!(f, "    source: {:?}", self.source)?;
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
            source: AnonStatsResultSource::Legacy,
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

// 2D anonymized statistics types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bucket2DResult {
    pub count: usize,
    pub left_hamming_distance_bucket: [f64; 2],
    pub right_hamming_distance_bucket: [f64; 2],
}

impl Eq for Bucket2DResult {}
impl PartialEq for Bucket2DResult {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count
            && (self.left_hamming_distance_bucket[0] - other.left_hamming_distance_bucket[0]).abs()
                <= 1e-9
            && (self.left_hamming_distance_bucket[1] - other.left_hamming_distance_bucket[1]).abs()
                <= 1e-9
            && (self.right_hamming_distance_bucket[0] - other.right_hamming_distance_bucket[0])
                .abs()
                <= 1e-9
            && (self.right_hamming_distance_bucket[1] - other.right_hamming_distance_bucket[1])
                .abs()
                <= 1e-9
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BucketStatistics2D {
    pub buckets: Vec<Bucket2DResult>,
    pub n_buckets_per_side: usize,
    // The number of two-sided matches gathered before sending the statistics
    pub match_distances_buffer_size: usize,
    pub party_id: usize,
    #[serde(default)]
    pub source: AnonStatsResultSource,
    #[serde(with = "ts_seconds")]
    pub start_time_utc_timestamp: DateTime<Utc>,
    #[serde(with = "ts_seconds_option")]
    pub end_time_utc_timestamp: Option<DateTime<Utc>>,
    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    #[serde(with = "ts_seconds_option")]
    pub next_start_time_utc_timestamp: Option<DateTime<Utc>>,
}

impl BucketStatistics2D {
    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }
}

impl fmt::Display for BucketStatistics2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    party_id: {}", self.party_id)?;
        writeln!(f, "    source: {:?}", self.source)?;
        writeln!(f, "    start_time_utc: {}", self.start_time_utc_timestamp)?;
        match &self.end_time_utc_timestamp {
            Some(end) => writeln!(f, "    end_time_utc: {}", end)?,
            None => writeln!(f, "    end_time_utc: <none>")?,
        }
        for bucket in &self.buckets {
            writeln!(
                f,
                "    L({:.3}-{:.3}), R({:.3}-{:.3}): {}",
                bucket.left_hamming_distance_bucket[0],
                bucket.left_hamming_distance_bucket[1],
                bucket.right_hamming_distance_bucket[0],
                bucket.right_hamming_distance_bucket[1],
                bucket.count
            )?;
        }
        Ok(())
    }
}

impl BucketStatistics2D {
    pub fn new(
        match_distances_buffer_size: usize,
        n_buckets_per_side: usize,
        party_id: usize,
    ) -> Self {
        Self {
            buckets: Vec::with_capacity(n_buckets_per_side * n_buckets_per_side),
            n_buckets_per_side,
            match_distances_buffer_size,
            party_id,
            source: AnonStatsResultSource::Legacy,
            start_time_utc_timestamp: Utc::now(),
            end_time_utc_timestamp: None,
            next_start_time_utc_timestamp: None,
        }
    }

    // Fill bucket counts for the 2D histogram.
    // buckets_2d is expected in row-major order (left index major):
    // buckets_2d[left_idx * n_buckets_per_side + right_idx]
    // flattened, row-major, INCLUSIVE 2D cumulative sums
    pub fn fill_buckets(
        &mut self,
        buckets_2d: &[u32],
        match_threshold_ratio: f64,
        start_timestamp: Option<DateTime<Utc>>,
    ) {
        tracing::info!("Filling 2D buckets : {} entries", buckets_2d.len());
        let now_timestamp = Utc::now();
        self.buckets.clear();
        self.end_time_utc_timestamp = Some(now_timestamp);

        let n = self.n_buckets_per_side;
        let nn = n * n; // safe from overflow, worst case its 375*375=140625
        if buckets_2d.len() != nn {
            tracing::warn!(
                "Expected {} cumulative entries ({}x{}), got {}. Missing cells treated as 0.",
                nn,
                n,
                n,
                buckets_2d.len()
            );
        }

        let step = match_threshold_ratio / (self.n_buckets_per_side as f64);
        self.buckets.reserve(nn);
        for idx in 0..nn {
            let i = idx / n; // row (left)
            let j = idx % n; // col (right)

            // C is the cumulative 2D bucket counts
            // C[i][j]
            let c_ij = *buckets_2d.get(idx).unwrap_or(&0) as i64;

            // C[i-1][j]
            let c_im1_j = if i > 0 {
                *buckets_2d.get(idx - n).unwrap_or(&0) as i64
            } else {
                0
            };

            // C[i][j-1]
            let c_i_jm1 = if j > 0 {
                *buckets_2d.get(idx - 1).unwrap_or(&0) as i64
            } else {
                0
            };

            // C[i-1][j-1]
            let c_im1_jm1 = if i > 0 && j > 0 {
                *buckets_2d.get(idx - n - 1).unwrap_or(&0) as i64
            } else {
                0
            };

            // Inclusion–exclusion to recover the true cell count [i][j]
            let mut cell = c_ij - c_im1_j - c_i_jm1 + c_im1_jm1;

            if cell < 0 {
                tracing::warn!(
                    "Negative decumulated count at ({},{}) after inclusion–exclusion; clamping to 0 \
                     (C[i,j]={}, C[i-1,j]={}, C[i,j-1]={}, C[i-1,j-1]={})",
                    i, j, c_ij, c_im1_j, c_i_jm1, c_im1_jm1
                );
                cell = 0;
            }

            let left_range = [step * (i as f64), step * ((i + 1) as f64)];
            let right_range = [step * (j as f64), step * ((j + 1) as f64)];

            // Only add the bucket if the count is greater than 0
            // This is because our end goal is to create a 2D histogram, and buckets with 0 count are not meaningful.
            // By avoiding them, we can dramatically reduce the size of the resulting message.
            if cell > 0 {
                self.buckets.push(Bucket2DResult {
                    count: cell as usize,
                    left_hamming_distance_bucket: left_range,
                    right_hamming_distance_bucket: right_range,
                });
            }
        }

        if let Some(start_timestamp) = start_timestamp {
            self.start_time_utc_timestamp = start_timestamp;
        }
        self.next_start_time_utc_timestamp = Some(now_timestamp);
    }
}
