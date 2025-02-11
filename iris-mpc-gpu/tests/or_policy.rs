mod or_policy_test {
    use iris_mpc_gpu::server::{generate_luc_records, prepare_or_policy_bitmap};
    use std::collections::HashSet;

    const MAX_DB_SIZE: usize = 8 * 1000;

    #[test]
    fn test_or_policy_bitmap_creation() {
        let or_policy_bitmap = vec![vec![1, 5, 10], vec![], vec![3]];
        let batch_size = or_policy_bitmap.len();
        let bitmap: Vec<u64> = prepare_or_policy_bitmap(MAX_DB_SIZE, or_policy_bitmap, batch_size);
        assert_eq!(bitmap.len() * 64, MAX_DB_SIZE * batch_size);

        // Positions 1, 5, 10 should be set in the bitmap
        // in u64, that is: 2^1 + 2^5 + 2^10 = 2 + 32 + 1024 = 1058
        let row_0_bitmap = "0000000000000000000000000000000000000000000000000000010000100010";
        // Nothing should be set
        let row_1_bitmap = "0000000000000000000000000000000000000000000000000000000000000000";
        // Position 3 should be set in the bitmap
        // in u64, that is: 2^3 = 8
        let row_2_bitmap = "0000000000000000000000000000000000000000000000000000000000001000";

        // Row stride: 8000/64 = 125, and we'll have 3 rows (batch_size)
        assert_eq!(bitmap[0], 1058);
        assert_eq!(format!("{:064b}", bitmap[0]), row_0_bitmap);

        assert_eq!(bitmap[125], 0);
        assert_eq!(format!("{:064b}", bitmap[125]), row_1_bitmap);

        assert_eq!(bitmap[250], 8);
        assert_eq!(format!("{:064b}", bitmap[250]), row_2_bitmap);
    }

    #[test]
    fn test_only_lookback_records() {
        let latest_index = 10;
        let or_rule_indices = vec![];
        let lookback_records = 5;

        let result = generate_luc_records(
            latest_index,
            or_rule_indices,
            lookback_records,
            2,
            HashSet::new(),
        );

        let expected = vec![vec![5, 6, 7, 8, 9, 10], vec![5, 6, 7, 8, 9, 10]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_empty_zero_lookback_records() {
        let latest_index = 10;
        let or_rule_indices = vec![vec![1, 3], vec![4, 5, 9]];
        let lookback_records = 0;

        let result = generate_luc_records(
            latest_index,
            or_rule_indices.clone(),
            lookback_records,
            1,
            HashSet::new(),
        );

        // No lookback, so we expect the same as the input.
        let expected = or_rule_indices.clone();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_basic_lookback() {
        // If latest_index=10 and lookback_records=3,
        // then lookback range is [7, 8, 9].
        let latest_lookback_index = 10;
        let lookback_records = 3;

        // Suppose our existing IDs are:
        let or_rule_indices = vec![vec![1, 3], vec![4, 5, 9]];

        let result = generate_luc_records(
            latest_lookback_index,
            or_rule_indices,
            lookback_records,
            2,
            HashSet::new(),
        );
        // We expect 7, 8, 9 to be appended (9 is already in second vector).
        let expected = vec![
            vec![1, 3, 7, 8, 9, 10], // was [1, 3] + [7, 8, 9, 10]
            vec![4, 5, 7, 8, 9, 10], // was [4, 5, 9] + [7, 8, 9, 10] => duplicates removed
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_duplicate_ids() {
        // Check that duplicates (both pre-existing and from the lookback) are removed
        // We use a small range so we can see the effect clearly.
        let latest_index = 5; // Suppose we want [3, 4] as lookback
        let lookback_records = 2;

        // Already has duplicates in the first vector
        let or_rule_indices = vec![vec![1, 1, 2, 3], vec![3, 3, 4]];

        let result = generate_luc_records(
            latest_index,
            or_rule_indices,
            lookback_records,
            2,
            HashSet::new(),
        );
        // After merging, each vector should include [3, 4] (some of which are
        // duplicates). Then we sort and deduplicate.
        let expected = vec![vec![1, 2, 3, 4, 5], vec![3, 4, 5]];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_skip_lookback_requests() {
        // We'll choose a latest index of 10, and a lookback of 3, which generates [8,
        // 9, 10].
        let latest_index = 10;
        let lookback_records = 3;

        // Suppose we have two requests in batch with OR-rule indices initially:
        //   Index 0: [1, 2]
        //   Index 1: [4, 5]
        // We'll skip adding the lookback to request 0, but allow it for request 1.
        let or_rule_indices = vec![vec![1, 2], vec![4, 5]];
        let batch_size = or_rule_indices.len();

        // Skip lookback for request 0.
        let skip_positions = std::collections::HashSet::from([0]);

        let result = generate_luc_records(
            latest_index,
            or_rule_indices.clone(),
            lookback_records,
            batch_size,
            skip_positions,
        );

        // Request 0 should remain unchanged: [1, 2].
        // Request 1 should have [7, 8, 9, 10] appended (then deduplicated and sorted).
        let expected = vec![vec![1, 2], vec![4, 5, 7, 8, 9, 10]];
        assert_eq!(result, expected);
    }
}
