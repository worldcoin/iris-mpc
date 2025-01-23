mod or_policy_test {

    use iris_mpc_gpu::server::prepare_or_policy_bitmap;

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
}
