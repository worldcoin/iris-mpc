use std::{fmt, time::Instant};

const FLUSH_AFTER_COUNT: u64 = 1000001;

pub struct FastHistogram {
    name: String,
    metrics_count: metrics::Counter,
    metrics_count_rate: metrics::Histogram,

    metrics_sum: metrics::Histogram,
    metrics_sum_rate: metrics::Histogram,

    metrics_min: metrics::Histogram,
    metrics_max: metrics::Histogram,
    metrics_avg: metrics::Histogram,

    count: u64,
    sum: f64,
    min: f64,
    max: f64,
    start: Instant,
}

impl FastHistogram {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),

            metrics_count: metrics::counter!(format!("{}.count", name)),
            metrics_count_rate: metrics::histogram!(format!("{}.count_rate", name)),

            metrics_sum: metrics::histogram!(format!("{}.sum", name)),
            metrics_sum_rate: metrics::histogram!(format!("{}.sum_rate", name)),

            metrics_min: metrics::histogram!(format!("{}.min", name)),
            metrics_max: metrics::histogram!(format!("{}.max", name)),
            metrics_avg: metrics::histogram!(format!("{}.avg", name)),

            count: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            start: Instant::now(),
        }
    }

    pub fn record(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }

        if self.count >= FLUSH_AFTER_COUNT {
            self.flush();
        }
    }

    pub fn flush(&mut self) {
        if self.count == 0 {
            return;
        }
        let elapsed = self.start.elapsed().as_secs_f64().max(1e-9);

        self.metrics_count.increment(self.count);
        self.metrics_count_rate.record(self.count as f64 / elapsed);

        self.metrics_sum.record(self.sum);
        self.metrics_sum_rate.record(self.sum / elapsed);

        self.metrics_min.record(self.min);
        self.metrics_max.record(self.max);
        self.metrics_avg.record(self.sum / self.count as f64);

        self.count = 0;
        self.sum = 0.0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
        self.start = Instant::now();
    }

    /// Get the current count of recorded values (before flushing).
    pub fn get_count(&self) -> u64 {
        self.count
    }

    /// Get the current sum of recorded values (before flushing).
    pub fn get_sum(&self) -> f64 {
        self.sum
    }

    /// Get the minimum recorded value (before flushing).
    pub fn get_min(&self) -> f64 {
        self.min
    }

    /// Get the maximum recorded value (before flushing).
    pub fn get_max(&self) -> f64 {
        self.max
    }

    /// Get the average of recorded values (before flushing).
    pub fn get_avg(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    /// Get the metric name.
    pub fn get_name(&self) -> &str {
        &self.name
    }
}

impl Clone for FastHistogram {
    fn clone(&self) -> Self {
        Self::new(&self.name)
    }
}

// Flush on drop.
impl Drop for FastHistogram {
    fn drop(&mut self) {
        self.flush();
    }
}

impl fmt::Debug for FastHistogram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FastHistogram")
            .field("count", &self.count)
            .field("sum", &self.sum)
            .field("min", &self.min)
            .field("max", &self.max)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrics::{
        with_local_recorder, Counter, CounterFn, Gauge, GaugeFn, Histogram, HistogramFn, Key,
        KeyName, Metadata, Recorder, SharedString, Unit,
    };
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::sync::Arc;

    #[test]
    fn test_fast_histogram() {
        let recorder = TestRecorder::new();

        let mut hist = with_local_recorder(&recorder, || FastHistogram::new("test_histogram"));

        hist.record(1.0);
        hist.record(2.0);
        hist.record(3.0);
        assert_eq!(hist.count, 3);
        assert_eq!(hist.sum, 6.0);
        assert_eq!(hist.min, 1.0);
        assert_eq!(hist.max, 3.0);
        hist.flush();

        assert_eq!(hist.count, 0);
        assert_eq!(hist.sum, 0.0);
        assert_eq!(hist.min, f64::INFINITY);
        assert_eq!(hist.max, f64::NEG_INFINITY);
        hist.record(4.0);
        drop(hist); // Flush.

        let mut events = recorder.events();

        // Handle time-dependent values.
        const RATE: f64 = 123.0;
        for i in [1, 3, 8, 10] {
            assert!(events[i].2 > 0.0);
            events[i].2 = RATE;
        }

        assert_eq!(
            events,
            vec![
                // Records: 1, 2, 3.
                (COUNT, "test_histogram.count".to_string(), 3.0),
                (HISTO, "test_histogram.count_rate".to_string(), RATE),
                (HISTO, "test_histogram.sum".to_string(), 6.0),
                (HISTO, "test_histogram.sum_rate".to_string(), RATE),
                (HISTO, "test_histogram.min".to_string(), 1.0),
                (HISTO, "test_histogram.max".to_string(), 3.0),
                (HISTO, "test_histogram.avg".to_string(), 2.0),
                // Records: 4.
                (COUNT, "test_histogram.count".to_string(), 1.0),
                (HISTO, "test_histogram.count_rate".to_string(), RATE),
                (HISTO, "test_histogram.sum".to_string(), 4.0),
                (HISTO, "test_histogram.sum_rate".to_string(), RATE),
                (HISTO, "test_histogram.min".to_string(), 4.0),
                (HISTO, "test_histogram.max".to_string(), 4.0),
                (HISTO, "test_histogram.avg".to_string(), 4.0),
            ]
        );
    }

    struct TestRecorder {
        tx: Sender<(bool, String, f64)>,
        rx: Receiver<(bool, String, f64)>,
    }

    impl TestRecorder {
        fn new() -> Self {
            let (tx, rx) = mpsc::channel();
            Self { tx, rx }
        }

        fn events(&self) -> Vec<(bool, String, f64)> {
            self.rx.try_iter().collect()
        }
    }

    impl Recorder for TestRecorder {
        fn describe_counter(
            &self,
            _name: KeyName,
            _unit: Option<Unit>,
            _description: SharedString,
        ) {
        }

        fn describe_gauge(&self, _name: KeyName, _unit: Option<Unit>, _description: SharedString) {}

        fn describe_histogram(
            &self,
            _name: KeyName,
            _unit: Option<Unit>,
            _description: SharedString,
        ) {
        }

        fn register_counter(&self, key: &Key, _meta: &Metadata<'_>) -> Counter {
            Counter::from_arc(Arc::new(TestCounter {
                is_histogram: false,
                name: key.name().to_string(),
                events: self.tx.clone(),
            }))
        }

        fn register_gauge(&self, key: &Key, _meta: &Metadata<'_>) -> Gauge {
            Gauge::from_arc(Arc::new(TestCounter {
                is_histogram: false,
                name: key.name().to_string(),
                events: self.tx.clone(),
            }))
        }

        fn register_histogram(&self, key: &Key, _meta: &Metadata<'_>) -> Histogram {
            Histogram::from_arc(Arc::new(TestCounter {
                is_histogram: true,
                name: key.name().to_string(),
                events: self.tx.clone(),
            }))
        }
    }

    struct TestCounter {
        is_histogram: bool,
        name: String,
        events: Sender<(bool, String, f64)>,
    }
    const COUNT: bool = false;
    const HISTO: bool = true;

    impl CounterFn for TestCounter {
        fn increment(&self, value: u64) {
            self.events
                .send((self.is_histogram, self.name.clone(), value as f64))
                .unwrap();
        }

        fn absolute(&self, _value: u64) {
            unimplemented!()
        }
    }

    impl GaugeFn for TestCounter {
        fn set(&self, value: f64) {
            self.events
                .send((self.is_histogram, self.name.clone(), value))
                .unwrap();
        }

        fn increment(&self, _value: f64) {
            unimplemented!();
        }

        fn decrement(&self, _value: f64) {
            unimplemented!();
        }
    }

    impl HistogramFn for TestCounter {
        fn record(&self, value: f64) {
            self.events
                .send((self.is_histogram, self.name.clone(), value))
                .unwrap();
        }
    }
}
