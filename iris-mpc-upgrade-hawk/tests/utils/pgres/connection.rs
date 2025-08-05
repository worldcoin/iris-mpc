use iris_mpc_common::{config::Config as NodeConfig, postgres::AccessMode};

/// Encapsulates information required to connect to a database.
pub struct DbConnectionInfo {
    /// Connection schema.
    access_mode: AccessMode,

    /// Connection schema.
    schema_name: String,

    /// Connection URL.
    url: String,
}

/// Constructors.
impl DbConnectionInfo {
    fn new(config: &NodeConfig, schema_suffix: &String, access_mode: AccessMode) -> Self {
        Self {
            access_mode,
            schema_name: config.get_db_schema(schema_suffix),
            url: config.get_db_url(),
        }
    }

    #[allow(dead_code)]
    pub fn new_read_only(config: &NodeConfig, schema_suffix: &String) -> Self {
        Self::new(config, schema_suffix, AccessMode::ReadOnly)
    }

    pub fn new_read_write(config: &NodeConfig, schema_suffix: &String) -> Self {
        Self::new(config, schema_suffix, AccessMode::ReadWrite)
    }
}

/// Accessors.
impl DbConnectionInfo {
    pub fn access_mode(&self) -> AccessMode {
        self.access_mode
    }

    pub fn schema_name(&self) -> &String {
        &self.schema_name
    }

    pub fn url(&self) -> &String {
        &self.url
    }
}
