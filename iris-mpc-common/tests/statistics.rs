mod tests {
    use ampc_anon_stats::types::{AnonStatsResultSource, Eye};
    use ampc_anon_stats::{
        AnonStatsOperation, Bucket2DResult, BucketResult, BucketStatistics, BucketStatistics2D,
    };
    use chrono::{TimeZone, Utc};
    use serde_json::{json, Value};

    #[test]
    fn test_serde_serialization() {
        // Let's pick some fixed timestamps for reproducible tests
        let known_start_time = Utc.timestamp_opt(1_700_000_000, 0).single().unwrap(); // ~ 2023-11-28T00:00:00Z
        let known_end_time = Utc.timestamp_opt(1_700_000_100, 0).single().unwrap(); // 100s later

        // Create a struct with some data
        let stats = BucketStatistics {
            buckets: vec![
                BucketResult {
                    count: 10,
                    hamming_distance_bucket: [0.00, 0.10],
                },
                BucketResult {
                    count: 20,
                    hamming_distance_bucket: [0.10, 0.20],
                },
            ],
            n_buckets: 2,
            match_distances_buffer_size: 128,
            party_id: 999,
            eye: Some(Eye::Right),
            operation: AnonStatsOperation::default(),
            source: AnonStatsResultSource::Aggregator,
            start_time_utc_timestamp: known_start_time,
            end_time_utc_timestamp: Some(known_end_time),
            // This field is #[serde(skip_serializing)]
            next_start_time_utc_timestamp: Some(Utc::now()),
            is_mirror_orientation: false,
        };

        // Serialize to JSON
        let json_str = serde_json::to_string(&stats).unwrap();
        let value: Value = serde_json::from_str(&json_str).unwrap();

        // 1) Check that the required fields appear in JSON
        //
        // Because of `#[serde(with = "ts_seconds")]`, `start_time_utc`
        // should be an integer with the timestamp in seconds
        assert_eq!(value["start_time_utc_timestamp"], json!(1_700_000_000_i64));

        // `end_time_utc` should likewise be an integer
        assert_eq!(value["end_time_utc_timestamp"], json!(1_700_000_100_i64));

        // The optional next_start_time_utc has `#[serde(skip_serializing)]` so it
        // should NOT appear
        assert!(!value
            .as_object()
            .unwrap()
            .contains_key("next_start_time_utc_timestamp"));

        // Eye is an enum with `#[derive(Serialize)]`, so it should appear as a string
        assert_eq!(value["eye"], json!("Right"));

        // We can check the buckets array
        assert_eq!(value["buckets"].as_array().unwrap().len(), 2);
        // For instance, check the first bucket
        assert_eq!(
            value["buckets"][0],
            json!({
                "count": 10,
                "hamming_distance_bucket": [0.00, 0.10]
            })
        );

        // 2) Check that party_id, n_buckets, etc. appear as expected
        assert_eq!(value["party_id"], json!(999));
        assert_eq!(value["n_buckets"], json!(2));
        assert_eq!(value["match_distances_buffer_size"], json!(128));
        assert_eq!(value["is_mirror_orientation"], json!(false));
        assert_eq!(value["source"], json!("aggregator"));
    }

    #[test]
    fn test_serde_deserialization() {
        // Imagine we receive this JSON from somewhere
        // Notice that we do NOT include next_start_time_utc
        // because it's skip_serializing in your struct
        let incoming_json = json!({
            "buckets": [
                {
                    "count": 50,
                    "hamming_distance_bucket": [0.0, 0.3]
                }
            ],
            "n_buckets": 1,
            "match_distances_buffer_size": 1024,
            "party_id": 123,
            "eye": "Left",
            "start_time_utc_timestamp": 1700000000,
            "end_time_utc_timestamp": null,
            "is_mirror_orientation": false
        })
        .to_string();

        let stats: BucketStatistics = serde_json::from_str(&incoming_json).unwrap();

        // Check that fields were deserialized
        assert_eq!(stats.buckets.len(), 1);
        assert_eq!(stats.buckets[0].count, 50);
        assert_eq!(stats.buckets[0].hamming_distance_bucket, [0.0, 0.3]);

        assert_eq!(stats.n_buckets, 1);
        assert_eq!(stats.match_distances_buffer_size, 1024);
        assert_eq!(stats.party_id, 123);
        assert!(matches!(stats.eye, Some(Eye::Left)));

        // start_time_utc is with seconds, so check that
        let expected_start = Utc.timestamp_opt(1_700_000_000, 0).single().unwrap();
        assert_eq!(stats.start_time_utc_timestamp, expected_start);

        // end_time_utc was null
        assert_eq!(stats.end_time_utc_timestamp, None);

        // next_start_time_utc is skip_serializing, so it wouldn't appear in JSON.
        // If your struct allows it in deserialization, it would default to None
        assert_eq!(stats.next_start_time_utc_timestamp, None);

        assert!(!stats.is_mirror_orientation);
        assert_eq!(stats.source, AnonStatsResultSource::Legacy);
    }

    #[test]
    fn test_serde_roundtrip() {
        // Create an instance
        let original_stats = BucketStatistics {
            buckets: vec![BucketResult {
                count: 100,
                hamming_distance_bucket: [0.33, 0.66],
            }],
            n_buckets: 1,
            match_distances_buffer_size: 42,
            party_id: 777,
            operation: AnonStatsOperation::default(),
            eye: Some(Eye::Right),
            source: AnonStatsResultSource::Aggregator,
            start_time_utc_timestamp: Utc.timestamp_opt(1_700_000_000, 0).single().unwrap(),
            end_time_utc_timestamp: Some(
                Utc.timestamp_opt(1_700_000_000, 0).single().unwrap()
                    + chrono::Duration::seconds(15),
            ),
            next_start_time_utc_timestamp: None,
            is_mirror_orientation: false,
        };

        // Serialize
        let json_str = serde_json::to_string(&original_stats).unwrap();
        println!("{}", json_str);

        // Deserialize
        let roundtrip_stats: BucketStatistics = serde_json::from_str(&json_str).unwrap();

        // Because next_start_time_utc is skip_serializing, it does NOT appear in the
        // JSON. So after deserialization, it won't be recovered. All other
        // fields, however, should match.
        assert_eq!(roundtrip_stats.buckets.len(), 1);
        assert_eq!(roundtrip_stats.buckets[0].count, 100);
        assert_eq!(
            roundtrip_stats.buckets[0].hamming_distance_bucket,
            [0.33, 0.66]
        );

        assert_eq!(roundtrip_stats.n_buckets, 1);
        assert_eq!(roundtrip_stats.match_distances_buffer_size, 42);
        assert_eq!(roundtrip_stats.party_id, 777);
        assert!(matches!(roundtrip_stats.eye, Some(Eye::Right)));

        // Timestamps (except next_start_time_utc) should match
        assert_eq!(
            roundtrip_stats.start_time_utc_timestamp,
            original_stats.start_time_utc_timestamp
        );
        assert_eq!(
            roundtrip_stats.end_time_utc_timestamp,
            original_stats.end_time_utc_timestamp
        );

        assert_eq!(roundtrip_stats.source, AnonStatsResultSource::Aggregator);

        // next_start_time_utc won't match because it was not serialized
        // So it should come back as None
        assert_eq!(roundtrip_stats.next_start_time_utc_timestamp, None);

        // is_mirror_orientation should be preserved correctly
        assert_eq!(
            roundtrip_stats.is_mirror_orientation,
            original_stats.is_mirror_orientation
        );
    }

    #[test]
    fn test_2d_fill_buckets_decumulate_and_ranges() {
        // n = 3 thresholds per side, match_threshold_ratio = 1.0 → step = 1/3
        let n = 3usize;
        let step = 1.0 / n as f64; // 0.333...

        // Cumulative matrix B (row-major):
        // [ [1,1,2],
        //   [1,2,4],
        //   [2,3,7] ]
        let buckets_2d_cumulative = vec![1, 1, 2, 1, 2, 4, 2, 3, 7];

        let mut stats = BucketStatistics2D::new(
            128,
            n,
            42,
            AnonStatsResultSource::Legacy,
            Some(AnonStatsOperation::default()),
        );
        stats.fill_buckets(&buckets_2d_cumulative, 1.0, None);

        // Expected per-cell histogram H (row-major, skipping zeros in output order):
        // H = [ [1,0,1],
        //       [0,1,1],
        //       [1,0,2] ]
        // Pushed in row-major skipping zeros → counts: [1,1,1,1,1,2]
        let expected = vec![
            Bucket2DResult {
                // (i,j) = (0,0)
                count: 1,
                left_hamming_distance_bucket: [0.0 * step, 1.0 * step],
                right_hamming_distance_bucket: [0.0 * step, 1.0 * step],
            },
            Bucket2DResult {
                // (0,2)
                count: 1,
                left_hamming_distance_bucket: [0.0 * step, 1.0 * step],
                right_hamming_distance_bucket: [2.0 * step, 3.0 * step],
            },
            Bucket2DResult {
                // (1,1)
                count: 1,
                left_hamming_distance_bucket: [1.0 * step, 2.0 * step],
                right_hamming_distance_bucket: [1.0 * step, 2.0 * step],
            },
            Bucket2DResult {
                // (1,2)
                count: 1,
                left_hamming_distance_bucket: [1.0 * step, 2.0 * step],
                right_hamming_distance_bucket: [2.0 * step, 3.0 * step],
            },
            Bucket2DResult {
                // (2,0)
                count: 1,
                left_hamming_distance_bucket: [2.0 * step, 3.0 * step],
                right_hamming_distance_bucket: [0.0 * step, 1.0 * step],
            },
            Bucket2DResult {
                // (2,2)
                count: 2,
                left_hamming_distance_bucket: [2.0 * step, 3.0 * step],
                right_hamming_distance_bucket: [2.0 * step, 3.0 * step],
            },
        ];

        assert_eq!(stats.buckets.len(), expected.len());
        assert_eq!(stats.buckets, expected);
    }
}
