mod tests {
    use chrono::{TimeZone, Utc};
    use iris_mpc_common::{
        helpers::statistics::{BucketResult, BucketStatistics},
        job::Eye,
    };
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
            eye: Eye::Right,
            start_time_utc_timestamp: known_start_time,
            end_time_utc_timestamp: Some(known_end_time),
            // This field is #[serde(skip_serializing)]
            next_start_time_utc_timestamp: Some(Utc::now()),
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
            "end_time_utc_timestamp": null
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
        assert!(matches!(stats.eye, Eye::Left));

        // start_time_utc is with seconds, so check that
        let expected_start = Utc.timestamp_opt(1_700_000_000, 0).single().unwrap();
        assert_eq!(stats.start_time_utc_timestamp, expected_start);

        // end_time_utc was null
        assert_eq!(stats.end_time_utc_timestamp, None);

        // next_start_time_utc is skip_serializing, so it wouldn't appear in JSON.
        // If your struct allows it in deserialization, it would default to None
        assert_eq!(stats.next_start_time_utc_timestamp, None);
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
            eye: Eye::Right,
            start_time_utc_timestamp: Utc.timestamp_opt(1_700_000_000, 0).single().unwrap(),
            end_time_utc_timestamp: Some(
                Utc.timestamp_opt(1_700_000_000, 0).single().unwrap()
                    + chrono::Duration::seconds(15),
            ),
            next_start_time_utc_timestamp: None,
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
        assert!(matches!(roundtrip_stats.eye, Eye::Right));

        // Timestamps (except next_start_time_utc) should match
        assert_eq!(
            roundtrip_stats.start_time_utc_timestamp,
            original_stats.start_time_utc_timestamp
        );
        assert_eq!(
            roundtrip_stats.end_time_utc_timestamp,
            original_stats.end_time_utc_timestamp
        );

        // next_start_time_utc won't match because it was not serialized
        // So it should come back as None
        assert_eq!(roundtrip_stats.next_start_time_utc_timestamp, None);
    }
}
