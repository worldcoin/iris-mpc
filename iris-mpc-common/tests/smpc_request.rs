mod tests {
    use base64::{engine::general_purpose::STANDARD, Engine};
    use http::StatusCode;
    use iris_mpc_common::helpers::{
        key_pair::{SharesDecodingError, SharesEncryptionKeyPair},
        smpc_request::{IrisCodesJSON, SMPCRequest},
    };
    use serde_json::json;
    use sodiumoxide::crypto::{box_::PublicKey, sealedbox};
    use wiremock::{
        matchers::{method, path},
        Mock, MockServer, ResponseTemplate,
    };

    const PUBLIC_KEY: &str = "HDp962tQyZIG9t+GX4JM0i1wgJx/YGpHGsuDSD34KBA=";
    const PRIVATE_KEY: &str = "14Z6Zijg3kbFN//R9BRKLeTS/wCiZMfK6AurEr/nAZg=";

    fn get_key_pair() -> SharesEncryptionKeyPair {
        SharesEncryptionKeyPair::from_b64_strings(
            PUBLIC_KEY.to_string().clone(),
            PRIVATE_KEY.to_string().clone(),
        )
        .unwrap()
    }

    fn get_mock_request() -> SMPCRequest {
        SMPCRequest {
            batch_size:              None,
            signup_id:               "test_signup_id".to_string(),
            s3_presigned_url:        "https://example.com/package".to_string(),
            iris_shares_file_hashes: [
                "hash_0".to_string(),
                "hash_1".to_string(),
                "hash_2".to_string(),
            ],
        }
    }

    #[tokio::test]
    async fn test_retrieve_iris_shares_from_s3_success() {
        let mock_server = MockServer::start().await;

        // Simulate a successful response from the presigned URL
        let response_body = json!({
            "iris_share_0": "share_0_data",
            "iris_share_1": "share_1_data",
            "iris_share_2": "share_2_data"
        });

        let template = ResponseTemplate::new(StatusCode::OK).set_body_json(response_body.clone());

        Mock::given(method("GET"))
            .and(path("/test_presign_url"))
            .respond_with(template)
            .mount(&mock_server)
            .await;

        let smpc_request = SMPCRequest {
            batch_size:              None,
            signup_id:               "test_signup_id".to_string(),
            s3_presigned_url:        mock_server.uri().clone() + "/test_presign_url",
            iris_shares_file_hashes: [
                "hash_0".to_string(),
                "hash_1".to_string(),
                "hash_2".to_string(),
            ],
        };

        let result = smpc_request.get_iris_data_by_party_id(0).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "share_0_data".to_string());
    }

    #[tokio::test]
    async fn test_decrypt_iris_share_success() {
        // Mocked base64 encoded JSON string
        let iris_codes_json = IrisCodesJSON {
            iris_version:           "1.0".to_string(),
            left_iris_code_shares:  "left_code".to_string(),
            right_iris_code_shares: "right_code".to_string(),
            left_iris_mask_shares:  "left_mask".to_string(),
            right_iris_mask_shares: "right_mask".to_string(),
        };

        let decoded_public_key = STANDARD.decode(PUBLIC_KEY.as_bytes()).unwrap();
        let shares_encryption_public_key = PublicKey::from_slice(&decoded_public_key).unwrap();

        // convert iris code to JSON string, sealbox and encode as BASE64
        let json_string = serde_json::to_string(&iris_codes_json).unwrap();
        let sealed_box = sealedbox::seal(json_string.as_bytes(), &shares_encryption_public_key);
        let encoded_share = STANDARD.encode(sealed_box);

        let smpc_request = get_mock_request();
        let key_pair = get_key_pair();

        let result = smpc_request.decrypt_iris_share(encoded_share, key_pair);

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), iris_codes_json);
    }

    #[tokio::test]
    async fn test_decrypt_iris_share_invalid_base64() {
        let invalid_base64 = "InvalidBase64String";
        let key_pair = get_key_pair();
        let smpc_request = get_mock_request();

        let result = smpc_request.decrypt_iris_share(invalid_base64.to_string(), key_pair);

        assert!(matches!(
            result,
            Err(SharesDecodingError::Base64DecodeError)
        ));
    }

    #[tokio::test]
    async fn test_decrypt_iris_share_invalid_utf8() {
        let invalid_utf8 = vec![0, 159, 146, 150]; // Not valid UTF-8

        let decoded_public_key = STANDARD.decode(PUBLIC_KEY.as_bytes()).unwrap();
        let shares_encryption_public_key = PublicKey::from_slice(&decoded_public_key).unwrap();
        let sealed_box = sealedbox::seal(&invalid_utf8, &shares_encryption_public_key);
        let encoded_share = STANDARD.encode(&sealed_box);

        let key_pair = get_key_pair();
        let smpc_request = get_mock_request();

        let result = smpc_request.decrypt_iris_share(encoded_share, key_pair);

        assert!(matches!(
            result,
            Err(SharesDecodingError::DecodedShareParsingToUTF8Error(_))
        ));
    }

    #[tokio::test]
    async fn test_decrypt_iris_share_invalid_json() {
        let invalid_json = "totally-not-a-json-string";

        let decoded_public_key = STANDARD.decode(PUBLIC_KEY.as_bytes()).unwrap();
        let shares_encryption_public_key = PublicKey::from_slice(&decoded_public_key).unwrap();
        let sealed_box = sealedbox::seal(invalid_json.as_bytes(), &shares_encryption_public_key);
        let encoded_share = STANDARD.encode(&sealed_box);

        let key_pair = get_key_pair();
        let smpc_request = get_mock_request();

        let result = smpc_request.decrypt_iris_share(encoded_share, key_pair);

        assert!(matches!(result, Err(SharesDecodingError::SerdeError(_))));
    }
}
