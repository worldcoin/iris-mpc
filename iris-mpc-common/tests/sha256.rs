mod tests {
    use iris_mpc_common::helpers::sha256::calculate_sha256;

    #[test]
    fn test_calculate_sha256() {
        // Arrange
        let data = "Hello, world!";
        let expected_hash = "315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3";

        // Act
        let calculated_hash = calculate_sha256(data);

        // Assert
        assert_eq!(
            calculated_hash, expected_hash,
            "The calculated SHA-256 hash should match the expected hash"
        );
    }

    #[test]
    fn test_calculate_sha256_empty_string() {
        // Arrange
        let data = "";
        let expected_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

        // Act
        let calculated_hash = calculate_sha256(data);

        // Assert
        assert_eq!(
            calculated_hash, expected_hash,
            "The calculated SHA-256 hash for an empty string should match the expected hash"
        );
    }

    #[test]
    fn test_calculate_sha256_different_data() {
        // Arrange
        let data_1 = "Data 1";
        let data_2 = "Data 2";
        let hash_1 = calculate_sha256(data_1);
        let hash_2 = calculate_sha256(data_2);

        // Act & Assert
        assert_ne!(
            hash_1, hash_2,
            "SHA-256 hashes of different data should not be equal"
        );
    }
}
