#[derive(Clone, Serialize, Deserialize, Default)]
pub struct DbConfig {
    pub url: String,

    #[serde(default)]
    pub migrate: bool,

    #[serde(default)]
    pub create: bool,
}

impl fmt::Debug for DbConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DbConfig")
            .field("url", &"********") // Mask the URL
            .field("migrate", &self.migrate)
            .field("create", &self.create)
            .finish()
    }
}
