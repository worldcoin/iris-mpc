use super::key_pair::{SharesDecodingError, SharesEncryptionKeyPair};
use base64::{engine::general_purpose::STANDARD, Engine};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct SQSMessage {
    #[serde(rename = "Type")]
    pub notification_type: String,
    #[serde(rename = "MessageId")]
    pub message_id:        String,
    #[serde(rename = "SequenceNumber")]
    pub sequence_number:   String,
    #[serde(rename = "TopicArn")]
    pub topic_arn:         String,
    #[serde(rename = "Message")]
    pub message:           String,
    #[serde(rename = "Timestamp")]
    pub timestamp:         String,
    #[serde(rename = "UnsubscribeURL")]
    pub unsubscribe_url:   String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SMPCRequest {
    pub signup_id:               String,
    pub s3_presigned_url:        String,
    pub iris_shares_file_hashes: [String; 3],
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SharesS3Object {
    pub iris_share_0: String,
    pub iris_share_1: String,
    pub iris_share_2: String,
}

#[derive(PartialEq, Serialize, Deserialize, Debug)]
pub struct IrisCodesJSON {
    #[serde(rename = "IRIS_version")]
    pub iris_version:    String,
    pub left_iris_code:  String,
    pub right_iris_code: String,
    pub left_iris_mask:  String,
    pub right_iris_mask: String,
}

impl SharesS3Object {
    pub fn get(&self, party_id: &String) -> Option<&String> {
        match party_id.as_str() {
            "0" => Some(&self.iris_share_0),
            "1" => Some(&self.iris_share_1),
            "2" => Some(&self.iris_share_2),
            _ => None,
        }
    }
}

impl SMPCRequest {
    pub async fn get_iris_data_by_party_id(
        &self,
        node_id: String,
        presign_url: String,
    ) -> Result<String, SharesDecodingError> {
        // Send a GET request to the presigned URL
        let response = match reqwest::get(presign_url.clone()).await {
            Ok(response) => response,
            Err(e) => {
                eprintln!("Failed to send request: {}", e);
                return Err(SharesDecodingError::RequestError(e));
            }
        };

        // Ensure the request was successful
        if response.status().is_success() {
            // Parse the JSON response into the SharesS3Object struct
            let shares_file: SharesS3Object = match response.json().await {
                Ok(file) => file,
                Err(e) => {
                    eprintln!("Failed to parse JSON: {}", e);
                    return Err(SharesDecodingError::RequestError(e));
                }
            };

            // Construct the field name dynamically
            let field_name = format!("iris_share_{}", node_id);
            // Access the field dynamically
            if let Some(value) = shares_file.get(&node_id) {
                Ok(value.to_string())
            } else {
                eprintln!("Failed to find field: {}", field_name);
                Err(SharesDecodingError::SecretStringNotFound)
            }
        } else {
            eprintln!("Failed to download file: {}", response.status());
            Err(SharesDecodingError::ResponseContent {
                status:  response.status(),
                url:     presign_url,
                message: response.text().await.unwrap_or_default(),
            })
        }
    }

    pub fn decrypt_iris_share(
        &self,
        share: String,
        key_pair: SharesEncryptionKeyPair,
    ) -> Result<IrisCodesJSON, SharesDecodingError> {
        let share_bytes = STANDARD
            .decode(share.as_bytes())
            .map_err(|_| SharesDecodingError::Base64DecodeError)?;

        let decrypted = key_pair.open_sealed_box(share_bytes);

        let iris_share = match decrypted {
            Ok(bytes) => {
                let json_string = String::from_utf8(bytes)
                    .map_err(|_| SharesDecodingError::DecodedShareParsingToUTF8Error)?;

                let iris_share: IrisCodesJSON = serde_json::from_str(&json_string)
                    .map_err(|_| SharesDecodingError::DecodedShareParsingToJSONError)?;
                iris_share
            }
            Err(e) => return Err(e),
        };

        Ok(iris_share)
    }

    fn validate_hashes(&self, hashes: [String; 3]) -> bool {
        self.iris_shares_file_hashes == hashes
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResultEvent {
    pub node_id:    usize,
    pub db_index:   u32,
    pub is_match:   bool,
    pub request_id: String,
}

impl ResultEvent {
    pub fn new(node_id: usize, db_index: u32, is_match: bool, request_id: String) -> Self {
        Self {
            node_id,
            db_index,
            is_match,
            request_id,
        }
    }
}
mod tests {
    use crate::helpers::{
        key_pair::{SharesDecodingError, SharesEncryptionKeyPair},
        sqs::{IrisCodesJSON, SMPCRequest},
    };
    use base64::{engine::general_purpose::STANDARD, Engine};
    use http::StatusCode;
    use serde_json::json;
    use sodiumoxide::crypto::{box_::PublicKey, sealedbox};
    use wiremock::{
        matchers::{method, path},
        Mock, MockServer, ResponseTemplate,
    };

    const PUBLIC_KEY: &str = "HDp962tQyZIG9t+GX4JM0i1wgJx/YGpHGsuDSD34KBA=";
    const PRIVATE_KEY: &str = "14Z6Zijg3kbFN//R9BRKLeTS/wCiZMfK6AurEr/nAZg=";

    fn get_key_pair() -> SharesEncryptionKeyPair {
        return SharesEncryptionKeyPair::from_b64_strings(
            PUBLIC_KEY.to_string().clone(),
            PRIVATE_KEY.to_string().clone(),
        )
        .unwrap();
    }

    fn get_mock_request() -> SMPCRequest {
        SMPCRequest {
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
            signup_id:               "test_signup_id".to_string(),
            s3_presigned_url:        mock_server.uri().clone() + "/test_presign_url",
            iris_shares_file_hashes: [
                "hash_0".to_string(),
                "hash_1".to_string(),
                "hash_2".to_string(),
            ],
        };

        let result = smpc_request
            .get_iris_data_by_party_id("0".to_string(), smpc_request.s3_presigned_url.clone())
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "share_0_data".to_string());
    }

    #[tokio::test]
    async fn test_decrypt_iris_share_success() {
        // Mocked base64 encoded JSON string
        let iris_codes_json = IrisCodesJSON {
            iris_version:    "1.0".to_string(),
            left_iris_code:  "left_code".to_string(),
            right_iris_code: "right_code".to_string(),
            left_iris_mask:  "left_mask".to_string(),
            right_iris_mask: "right_mask".to_string(),
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
            Err(SharesDecodingError::DecodedShareParsingToUTF8Error)
        ));
    }

    #[tokio::test]
    async fn test_decrypt_iris_share_invalid_json() {
        let invalid_json = "totally-not-a-json-string";

        let decoded_public_key = STANDARD.decode(PUBLIC_KEY.as_bytes()).unwrap();
        let shares_encryption_public_key = PublicKey::from_slice(&decoded_public_key).unwrap();
        let sealed_box = sealedbox::seal(&invalid_json.as_bytes(), &shares_encryption_public_key);
        let encoded_share = STANDARD.encode(&sealed_box);

        let key_pair = get_key_pair();
        let smpc_request = get_mock_request();

        let result = smpc_request.decrypt_iris_share(encoded_share, key_pair);

        assert!(matches!(
            result,
            Err(SharesDecodingError::DecodedShareParsingToJSONError)
        ));
    }
}
