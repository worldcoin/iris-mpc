use super::aws_sigv4::{
    authorization_header, canonical_request, scope_string_with_service, signed_header_string,
    signing_key, string_to_sign, LONG_DATETIME,
};
use aws_sdk_kms::Client;
use base64::{prelude::BASE64_STANDARD, Engine};
use eyre::ContextCompat;
use http::{
    header::{AUTHORIZATION, CONTENT_TYPE},
    HeaderMap,
};
use reqwest::Url;
use ring::{
    digest::{digest, SHA256},
    hmac,
};
use serde_json::{json, Value};
use std::env;
use time::OffsetDateTime;

/// Derive a shared secret from two KMS keys
/// Unfortunately, this is not yet implemented in the AWS SDK for Rust, so we
/// have to do it manually with the API and SigV4
pub async fn derive_shared_secret(own_key_id: &str, other_key_id: &str) -> eyre::Result<[u8; 32]> {
    let shared_config = aws_config::from_env().load().await;
    let other_pub_key = get_public_key(other_key_id).await?;

    let access_key = env::var("AWS_ACCESS_KEY_ID")?;
    let secret_key = env::var("AWS_SECRET_ACCESS_KEY")?;
    let region = shared_config.region().context("No region set!")?.as_ref();
    let hostname = format!("kms.{}.amazonaws.com", region);
    let url = format!("https://{}/", hostname);
    let ts = OffsetDateTime::now_utc();
    let ts_formatted = ts.format(LONG_DATETIME)?;

    let mut headers = HeaderMap::new();
    headers.insert("Host", hostname.parse().unwrap());
    headers.insert("X-Amz-Date", ts_formatted.parse().unwrap());
    headers.insert(CONTENT_TYPE, "application/x-amz-json-1.1".parse().unwrap());

    headers.insert(
        "x-amz-target",
        "TrentService.DeriveSharedSecret".parse().unwrap(),
    );

    let json_data = json!({
        "KeyId": own_key_id,
        "PublicKey": other_pub_key,
        "KeyAgreementAlgorithm": "ECDH"
    });

    let json_string = serde_json::to_string(&json_data).unwrap();
    let url: Url = url.parse().unwrap();
    let canonical_req = canonical_request(
        "POST",
        &url,
        &headers,
        &hex::encode(digest(&SHA256, json_string.as_bytes())),
    )?;
    let scope = scope_string_with_service(&ts, &region, "kms")?;
    let string = string_to_sign(&ts, &canonical_req, &scope)?;
    let signing_key = signing_key(&ts, &secret_key, &region, "kms")?;
    let key = hmac::Key::new(hmac::HMAC_SHA256, &signing_key);
    let tag = hmac::sign(&key, string.as_bytes());
    let signature = hex::encode(tag);
    let auth_header = authorization_header(
        &access_key,
        &scope,
        &signed_header_string(&headers),
        &signature,
    )?;

    // Use the above constructed auth header
    headers.insert(AUTHORIZATION, auth_header.parse().unwrap());

    let client = reqwest::Client::new();
    let res = client
        .post(url)
        .headers(headers)
        .body(json_string)
        .send()
        .await?;

    res.error_for_status_ref()?;

    let value: Value = serde_json::from_str(res.text().await?.as_str())?;
    let shared_secret = value["SharedSecret"]
        .as_str()
        .context("No key `SharedSecret` found in KMS response.")?;
    let shared_secret_decoded = BASE64_STANDARD.decode(shared_secret)?;

    let mut buffer = [0u8; 32];
    buffer.copy_from_slice(&shared_secret_decoded);

    Ok(buffer)
}

async fn get_public_key(key_id: &str) -> eyre::Result<String> {
    let shared_config = aws_config::from_env().load().await;
    let client = Client::new(&shared_config);

    let res = client.get_public_key().key_id(key_id).send().await?;

    Ok(BASE64_STANDARD.encode(res.public_key().context("No public key found")?))
}
