//! Backend-independent object storage construction.
//!
//! Existing configuration values historically contain an S3 bucket name.  To
//! preserve compatibility, values without a URL scheme are interpreted as S3
//! buckets.  Full URLs can be used to select any backend supported by the
//! `object_store` crate, for example `gs://bucket`, `az://container`, or
//! `file:///var/lib/iris-mpc/objects`.

use object_store::{path::Path, prefix::PrefixStore, ObjectStore};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};
use url::Url;

pub use object_store::ObjectStoreExt;

/// A shared object store rooted at the configured bucket/container and prefix.
pub type ObjectStoreRef = Arc<dyn ObjectStore>;

/// Lazily constructs and caches object stores for configured storage locations.
///
/// This retains the old client-plus-bucket calling convention while removing
/// the storage implementation's dependency on the AWS SDK.
#[derive(Clone, Debug)]
pub struct ObjectStoreClient {
    region: Option<String>,
    force_path_style: bool,
    options: Arc<Vec<(String, String)>>,
    stores: Arc<RwLock<HashMap<String, ObjectStoreRef>>>,
}

impl ObjectStoreClient {
    pub fn new(region: Option<String>, force_path_style: bool) -> Self {
        Self {
            region,
            force_path_style,
            options: Arc::default(),
            stores: Arc::default(),
        }
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
        let mut options: Vec<(String, String)> = std::env::vars().collect();
        if options.iter().any(|(key, value)| {
            matches!(key.as_str(), "AWS_ENDPOINT" | "AWS_ENDPOINT_URL")
                && value.starts_with("http://")
        }) {
            options.push(("aws_allow_http".to_owned(), "true".to_owned()));
        }
        if let Some(region) = &self.region {
            options.push(("aws_region".to_owned(), region.clone()));
        }
        options.push((
            "aws_virtual_hosted_style_request".to_owned(),
            (!self.force_path_style).to_string(),
        ));
        options.extend(self.options.iter().cloned());

        let (store, prefix) = object_store::parse_url_opts(&url, options)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

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
