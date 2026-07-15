//! Backend-independent object storage construction.
//!
//! Existing configuration values historically contain an S3 bucket name.  To
//! preserve compatibility, values without a URL scheme are interpreted as S3
//! buckets.  Full URLs can be used to select any backend supported by the
//! `object_store` crate, for example `gs://bucket`, `az://container`, or
//! `file:///var/lib/iris-mpc/objects`.

use object_store::{
    aws::{AmazonS3Builder, AmazonS3ConfigKey, AwsCredentialProvider},
    path::Path,
    prefix::PrefixStore,
    ObjectStore, ObjectStoreScheme,
};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};
use url::Url;

#[cfg(feature = "helpers")]
use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
#[cfg(feature = "helpers")]
use object_store::{aws::AwsCredential, CredentialProvider};
#[cfg(feature = "helpers")]
use std::time::{Duration, SystemTime};

pub use object_store::ObjectStoreExt;

/// A shared object store rooted at the configured bucket/container and prefix.
pub type ObjectStoreRef = Arc<dyn ObjectStore>;

/// Lazily constructs and caches object stores for configured storage locations.
///
/// This retains the old client-plus-bucket calling convention while removing
/// the storage implementation's dependency on `aws-sdk-s3`.
#[derive(Clone, Debug)]
pub struct ObjectStoreClient {
    region: Option<String>,
    force_path_style: bool,
    aws_credentials: Option<AwsCredentialProvider>,
    options: Arc<Vec<(String, String)>>,
    stores: Arc<RwLock<HashMap<String, ObjectStoreRef>>>,
}

impl ObjectStoreClient {
    pub fn new(region: Option<String>, force_path_style: bool) -> Self {
        Self {
            region,
            force_path_style,
            aws_credentials: None,
            options: Arc::default(),
            stores: Arc::default(),
        }
    }

    /// Use an already-resolved AWS SDK credential chain for S3 stores.
    ///
    /// This preserves shared-profile, SSO, `credential_process`, and assume-role
    /// support while still constructing the storage backend through
    /// `object_store`. The provider remains dynamic, so expiring credentials can
    /// be refreshed by the AWS SDK chain.
    #[cfg(feature = "helpers")]
    pub fn with_aws_sdk_config(mut self, sdk_config: &aws_config::SdkConfig) -> Self {
        if self.region.is_none() {
            self.region = sdk_config.region().map(ToString::to_string);
        }
        if let Some(endpoint) = sdk_config.endpoint_url() {
            Arc::make_mut(&mut self.options).push(("aws_endpoint".to_owned(), endpoint.to_owned()));
        }
        if let Some(provider) = sdk_config.credentials_provider() {
            self.aws_credentials = Some(Arc::new(AwsSdkCredentialAdapter::new(provider)));
        }
        self
    }

    pub fn with_option(mut self, key: impl Into<String>, value: impl ToString) -> Self {
        Arc::make_mut(&mut self.options).push((key.into(), value.to_string()));
        self
    }

    /// Returns the store for `location`.
    ///
    /// A value without a URL scheme is treated as an S3 bucket for backwards
    /// compatibility. A URL may also contain a path prefix; returned stores are
    /// rooted at that prefix.
    pub fn store(&self, location: &str) -> object_store::Result<ObjectStoreRef> {
        if let Some(store) = self.stores.read().unwrap().get(location).cloned() {
            return Ok(store);
        }

        let url = storage_url(location)?;
        let (scheme, prefix) = ObjectStoreScheme::parse(&url)?;
        let mut options: Vec<(String, String)> = std::env::vars().collect();
        if let Some(region) = &self.region {
            options.push(("aws_region".to_owned(), region.clone()));
        }
        let mut addressing_options = options.clone();
        addressing_options.extend(self.options.iter().cloned());
        options.push((
            "aws_virtual_hosted_style_request".to_owned(),
            automatic_virtual_hosted_style(&url, self.force_path_style, &addressing_options)?
                .to_string(),
        ));
        options.extend(self.options.iter().cloned());

        if scheme == ObjectStoreScheme::AmazonS3 {
            normalize_s3_options(&url, &mut options)?;
        }

        let store: Box<dyn ObjectStore> = if scheme == ObjectStoreScheme::AmazonS3 {
            let mut builder = AmazonS3Builder::new().with_url(url.as_str());
            for (key, value) in options {
                if let Ok(key) = key.to_ascii_lowercase().parse::<AmazonS3ConfigKey>() {
                    builder = builder.with_config(key, value);
                }
            }
            if let Some(credentials) = &self.aws_credentials {
                builder = builder.with_credentials(credentials.clone());
            }
            Box::new(builder.build()?)
        } else {
            object_store::parse_url_opts(&url, options)?.0
        };
        let store: ObjectStoreRef = if prefix.as_ref().is_empty() {
            Arc::from(store)
        } else {
            Arc::new(PrefixStore::new(store, prefix))
        };

        let mut stores = self.stores.write().unwrap();
        Ok(stores
            .entry(location.to_owned())
            .or_insert_with(|| store.clone())
            .clone())
    }

    /// Registers a pre-built store, primarily for backend-independent tests.
    pub fn insert(&self, location: impl Into<String>, store: ObjectStoreRef) {
        self.stores.write().unwrap().insert(location.into(), store);
    }
}

#[cfg(feature = "helpers")]
#[derive(Debug)]
struct AwsSdkCredentialAdapter {
    provider: SharedCredentialsProvider,
    cached: tokio::sync::Mutex<Option<CachedAwsCredential>>,
}

#[cfg(feature = "helpers")]
#[derive(Debug)]
struct CachedAwsCredential {
    credential: Arc<AwsCredential>,
    expires_at: SystemTime,
}

#[cfg(feature = "helpers")]
impl AwsSdkCredentialAdapter {
    // Match the AWS SDK lazy identity cache defaults so switching storage
    // implementations does not change credential loading behavior.
    const LOAD_TIMEOUT: Duration = Duration::from_secs(5);
    const DEFAULT_EXPIRATION: Duration = Duration::from_secs(15 * 60);
    const REFRESH_BUFFER: Duration = Duration::from_secs(10);

    fn new(provider: SharedCredentialsProvider) -> Self {
        Self {
            provider,
            cached: tokio::sync::Mutex::new(None),
        }
    }

    fn is_fresh(cached: &CachedAwsCredential) -> bool {
        cached
            .expires_at
            .duration_since(SystemTime::now())
            .is_ok_and(|remaining| remaining > Self::REFRESH_BUFFER)
    }

    async fn get_credential_with_timeout(
        &self,
        timeout: Duration,
    ) -> object_store::Result<Arc<AwsCredential>> {
        tokio::time::timeout(timeout, self.get_credential_inner())
            .await
            .map_err(|source| object_store::Error::Generic {
                store: "AWS SDK credential provider",
                source: Box::new(source),
            })?
    }

    async fn get_credential_inner(&self) -> object_store::Result<Arc<AwsCredential>> {
        let mut cached = self.cached.lock().await;
        if let Some(cached) = cached.as_ref().filter(|cached| Self::is_fresh(cached)) {
            return Ok(cached.credential.clone());
        }

        let credentials = self
            .provider
            .provide_credentials()
            .await
            .map_err(|source| object_store::Error::Generic {
                store: "AWS SDK credential provider",
                source: Box::new(source),
            })?;
        let expires_at = credentials
            .expiry()
            .unwrap_or_else(|| SystemTime::now() + Self::DEFAULT_EXPIRATION);
        let credential = Arc::new(AwsCredential {
            key_id: credentials.access_key_id().to_owned(),
            secret_key: credentials.secret_access_key().to_owned(),
            token: credentials.session_token().map(ToOwned::to_owned),
        });
        *cached = Some(CachedAwsCredential {
            credential: credential.clone(),
            expires_at,
        });
        Ok(credential)
    }
}

#[cfg(feature = "helpers")]
#[async_trait::async_trait]
impl CredentialProvider for AwsSdkCredentialAdapter {
    type Credential = AwsCredential;

    async fn get_credential(&self) -> object_store::Result<Arc<Self::Credential>> {
        self.get_credential_with_timeout(Self::LOAD_TIMEOUT).await
    }
}

/// Parse a configured object key using the portable `object_store` path rules.
pub fn path(key: &str) -> object_store::Result<Path> {
    Path::parse(key).map_err(Into::into)
}

fn storage_url(location: &str) -> object_store::Result<Url> {
    let value = if location.contains("://") {
        location.to_owned()
    } else {
        format!("s3://{location}")
    };
    Url::parse(&value).map_err(|source| object_store::Error::Generic {
        store: "configuration",
        source: Box::new(source),
    })
}

fn normalize_s3_options(
    storage_url: &Url,
    options: &mut Vec<(String, String)>,
) -> object_store::Result<()> {
    let endpoint = effective_s3_endpoint(options).map(str::to_owned);

    if endpoint
        .as_deref()
        .is_some_and(|value| value.starts_with("http://"))
        && !has_option(options, &["aws_allow_http", "allow_http"])
    {
        options.push(("aws_allow_http".to_owned(), "true".to_owned()));
    }

    let virtual_hosted = option_value(
        options,
        &[
            "aws_virtual_hosted_style_request",
            "virtual_hosted_style_request",
        ],
    )
    .and_then(|value| value.parse::<bool>().ok())
    .unwrap_or(false);
    if !virtual_hosted {
        return Ok(());
    }

    let Some(bucket) = s3_bucket(storage_url) else {
        return Ok(());
    };
    if !is_virtual_hostable_s3_bucket(&bucket, endpoint.as_deref())? {
        options.push((
            "aws_virtual_hosted_style_request".to_owned(),
            "false".to_owned(),
        ));
        return Ok(());
    }
    let Some(endpoint) = endpoint else {
        return Ok(());
    };
    match bucket_qualified_endpoint(&endpoint, &bucket)? {
        Some(endpoint) => options.push(("aws_endpoint_url_s3".to_owned(), endpoint)),
        None => options.push((
            "aws_virtual_hosted_style_request".to_owned(),
            "false".to_owned(),
        )),
    }
    Ok(())
}

fn automatic_virtual_hosted_style(
    storage_url: &Url,
    force_path_style: bool,
    options: &[(String, String)],
) -> object_store::Result<bool> {
    if force_path_style {
        return Ok(false);
    }
    let Some(bucket) = s3_bucket(storage_url) else {
        return Ok(false);
    };
    is_virtual_hostable_s3_bucket(&bucket, effective_s3_endpoint(options))
}

/// Mirrors the AWS SDK's automatic choice between virtual-hosted and path-style
/// S3 addressing. HTTPS requires a single DNS-compatible label so wildcard TLS
/// certificates remain valid; custom HTTP endpoints may use dotted buckets.
fn is_virtual_hostable_s3_bucket(
    bucket: &str,
    endpoint: Option<&str>,
) -> object_store::Result<bool> {
    let allow_subdomains = if let Some(endpoint) = endpoint {
        let url = Url::parse(endpoint).map_err(|source| object_store::Error::Generic {
            store: "S3 endpoint configuration",
            source: Box::new(source),
        })?;
        match url.host() {
            Some(url::Host::Domain(host)) if !host.eq_ignore_ascii_case("localhost") => {
                url.scheme() == "http"
            }
            _ => return Ok(false),
        }
    } else {
        false
    };

    if allow_subdomains {
        Ok(bucket.split('.').all(is_virtual_hostable_s3_segment))
    } else {
        Ok(!bucket.contains('.') && is_virtual_hostable_s3_segment(bucket))
    }
}

fn is_virtual_hostable_s3_segment(segment: &str) -> bool {
    let bytes = segment.as_bytes();
    (3..=63).contains(&bytes.len())
        && bytes
            .first()
            .is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        && bytes
            .last()
            .is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'-')
}

fn effective_s3_endpoint(options: &[(String, String)]) -> Option<&str> {
    option_value(options, &["aws_endpoint_url_s3"]).or_else(|| {
        option_value(
            options,
            &[
                "aws_endpoint_url",
                "aws_endpoint",
                "endpoint_url",
                "endpoint",
            ],
        )
    })
}

fn option_value<'a>(options: &'a [(String, String)], keys: &[&str]) -> Option<&'a str> {
    options.iter().rev().find_map(|(key, value)| {
        keys.iter()
            .any(|candidate| key.eq_ignore_ascii_case(candidate))
            .then_some(value.as_str())
    })
}

fn has_option(options: &[(String, String)], keys: &[&str]) -> bool {
    option_value(options, keys).is_some()
}

fn s3_bucket(url: &Url) -> Option<String> {
    match url.scheme() {
        "s3" | "s3a" => url.host_str().map(ToOwned::to_owned),
        "https" => {
            let host = url.host_str()?;
            if host.starts_with("s3") || host.ends_with("r2.cloudflarestorage.com") {
                url.path_segments()?.next().map(ToOwned::to_owned)
            } else {
                host.split_once(".s3").map(|(bucket, _)| bucket.to_owned())
            }
        }
        _ => None,
    }
}

fn bucket_qualified_endpoint(endpoint: &str, bucket: &str) -> object_store::Result<Option<String>> {
    let mut url = Url::parse(endpoint).map_err(|source| object_store::Error::Generic {
        store: "S3 endpoint configuration",
        source: Box::new(source),
    })?;
    let Some(host) = url.host() else {
        return Ok(None);
    };
    let url::Host::Domain(host) = host else {
        return Ok(None);
    };
    if host.eq_ignore_ascii_case("localhost") {
        return Ok(None);
    }
    if host == bucket || host.starts_with(&format!("{bucket}.")) {
        return Ok(Some(url.to_string().trim_end_matches('/').to_owned()));
    }
    url.set_host(Some(&format!("{bucket}.{host}")))
        .map_err(|_| object_store::Error::Generic {
            store: "S3 endpoint configuration",
            source: "unable to add bucket to S3 endpoint host".into(),
        })?;
    Ok(Some(url.to_string().trim_end_matches('/').to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;
    use wiremock::{
        matchers::{method, path as request_path},
        Mock, MockServer, ResponseTemplate,
    };

    #[test]
    fn caches_registered_stores() {
        let client = ObjectStoreClient::new(None, false);
        let expected: ObjectStoreRef = Arc::new(InMemory::new());
        client.insert("test", expected.clone());

        let actual = client.store("test").unwrap();
        assert!(Arc::ptr_eq(&expected, &actual));
    }

    #[test]
    fn plain_locations_remain_s3_buckets() {
        assert_eq!(storage_url("bucket").unwrap().as_str(), "s3://bucket");
    }

    #[test]
    fn service_specific_http_endpoint_is_allowed() {
        let url = storage_url("bucket").unwrap();
        let mut options = vec![
            (
                "AWS_ENDPOINT_URL".to_owned(),
                "https://global.example".to_owned(),
            ),
            (
                "AWS_ENDPOINT_URL_S3".to_owned(),
                "http://s3.example".to_owned(),
            ),
            (
                "aws_virtual_hosted_style_request".to_owned(),
                "false".to_owned(),
            ),
        ];

        normalize_s3_options(&url, &mut options).unwrap();

        assert_eq!(effective_s3_endpoint(&options), Some("http://s3.example"));
        assert_eq!(option_value(&options, &["aws_allow_http"]), Some("true"));
    }

    #[test]
    fn virtual_hosted_custom_endpoint_is_bucket_qualified() {
        let url = storage_url("bucket").unwrap();
        let mut options = vec![
            (
                "aws_endpoint_url_s3".to_owned(),
                "https://storage.example:9443/base".to_owned(),
            ),
            (
                "aws_virtual_hosted_style_request".to_owned(),
                "true".to_owned(),
            ),
        ];

        normalize_s3_options(&url, &mut options).unwrap();

        assert_eq!(
            effective_s3_endpoint(&options),
            Some("https://bucket.storage.example:9443/base")
        );
    }

    #[test]
    fn automatic_addressing_falls_back_for_non_virtual_hostable_buckets() {
        let options = Vec::new();
        assert!(automatic_virtual_hosted_style(
            &storage_url("valid-bucket").unwrap(),
            false,
            &options,
        )
        .unwrap());

        for bucket in ["bucket.name", "BucketName", "aa"] {
            assert!(
                !automatic_virtual_hosted_style(&storage_url(bucket).unwrap(), false, &options,)
                    .unwrap(),
                "{bucket} must use path-style addressing"
            );
        }
        assert!(!automatic_virtual_hosted_style(
            &storage_url("valid-bucket").unwrap(),
            true,
            &options,
        )
        .unwrap());
    }

    #[test]
    fn dotted_buckets_are_virtual_hosted_only_for_custom_http_endpoints() {
        let url = storage_url("bucket.name").unwrap();
        let https_options = vec![(
            "aws_endpoint_url_s3".to_owned(),
            "https://storage.example".to_owned(),
        )];
        let http_options = vec![(
            "aws_endpoint_url_s3".to_owned(),
            "http://storage.example".to_owned(),
        )];

        assert!(!automatic_virtual_hosted_style(&url, false, &https_options).unwrap());
        assert!(automatic_virtual_hosted_style(&url, false, &http_options).unwrap());
    }

    #[test]
    fn already_qualified_custom_endpoint_is_unchanged() {
        assert_eq!(
            bucket_qualified_endpoint("https://bucket.storage.example", "bucket").unwrap(),
            Some("https://bucket.storage.example".to_owned())
        );
        assert_eq!(
            bucket_qualified_endpoint("http://localhost:4566", "bucket").unwrap(),
            None
        );
    }

    #[cfg(feature = "helpers")]
    #[tokio::test]
    async fn aws_sdk_credentials_are_bridged() {
        use aws_credential_types::Credentials;

        let credentials = Credentials::new(
            "profile-access-key",
            "profile-secret-key",
            Some("profile-session-token".to_owned()),
            None,
            "test-profile",
        );
        let sdk_config = aws_config::SdkConfig::builder()
            .credentials_provider(SharedCredentialsProvider::new(credentials))
            .endpoint_url("https://profile-endpoint.example")
            .build();
        let client = ObjectStoreClient::new(None, false).with_aws_sdk_config(&sdk_config);

        assert_eq!(
            option_value(&client.options, &["aws_endpoint"]),
            Some("https://profile-endpoint.example")
        );
        let provider = client.aws_credentials.as_ref().unwrap();
        let credentials = provider.get_credential().await.unwrap();
        assert_eq!(credentials.key_id, "profile-access-key");
        assert_eq!(credentials.secret_key, "profile-secret-key");
        assert_eq!(credentials.token.as_deref(), Some("profile-session-token"));
        let cached_credentials = provider.get_credential().await.unwrap();
        assert!(Arc::ptr_eq(&credentials, &cached_credentials));
    }

    #[cfg(feature = "helpers")]
    #[tokio::test]
    async fn credentials_without_expiry_are_refreshed() {
        use aws_credential_types::{credential_fn::provide_credentials_fn, Credentials};
        use std::sync::atomic::{AtomicUsize, Ordering};

        let calls = Arc::new(AtomicUsize::new(0));
        let provider_calls = calls.clone();
        let provider = SharedCredentialsProvider::new(provide_credentials_fn(move || {
            let call = provider_calls.fetch_add(1, Ordering::SeqCst) + 1;
            async move {
                Ok(Credentials::new(
                    format!("access-key-{call}"),
                    "secret-key",
                    None,
                    None,
                    "rotating-test-provider",
                ))
            }
        }));
        let adapter = AwsSdkCredentialAdapter::new(provider);

        let first = adapter.get_credential().await.unwrap();
        let cached = adapter.get_credential().await.unwrap();
        assert_eq!(first.key_id, "access-key-1");
        assert!(Arc::ptr_eq(&first, &cached));
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // Providers such as credential_process may omit an expiry. They still
        // receive a finite cache lifetime, matching the AWS SDK identity cache.
        adapter.cached.lock().await.as_mut().unwrap().expires_at = SystemTime::UNIX_EPOCH;

        let refreshed = adapter.get_credential().await.unwrap();
        assert_eq!(refreshed.key_id, "access-key-2");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[cfg(feature = "helpers")]
    #[tokio::test]
    async fn credential_loading_has_a_timeout() {
        use aws_credential_types::credential_fn::provide_credentials_fn;

        let provider = SharedCredentialsProvider::new(provide_credentials_fn(|| async {
            std::future::pending::<aws_credential_types::provider::Result>().await
        }));
        let adapter = AwsSdkCredentialAdapter::new(provider);

        let error = adapter
            .get_credential_with_timeout(Duration::from_millis(1))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("AWS SDK credential provider"));
    }

    #[tokio::test]
    async fn http_ip_endpoint_falls_back_to_path_style() {
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(request_path("/bucket/checkpoint.bin"))
            .respond_with(ResponseTemplate::new(200).insert_header("ETag", "\"test-etag\""))
            .mount(&server)
            .await;

        let client = ObjectStoreClient::new(Some("us-east-1".to_owned()), false)
            .with_option("aws_endpoint_url_s3", server.uri())
            .with_option("aws_access_key_id", "test")
            .with_option("aws_secret_access_key", "test");
        let store = client.store("bucket").unwrap();
        store
            .put(
                &path("checkpoint.bin").unwrap(),
                b"checkpoint".to_vec().into(),
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn full_urls_select_non_s3_backends_and_prefixes() {
        let client = ObjectStoreClient::new(None, false);
        let store = client.store("memory:///iris-mpc").unwrap();
        let location = path("checkpoint.bin").unwrap();

        store
            .put(&location, b"checkpoint".to_vec().into())
            .await
            .unwrap();
        let bytes = store.get(&location).await.unwrap().bytes().await.unwrap();

        assert_eq!(bytes.as_ref(), b"checkpoint");
    }
}
