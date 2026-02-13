use std::path::Path;

use eyre::Result;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use toml;

/// Writes serialised toml data to local file system.
pub async fn write_toml<T>(entity: &T, path_to_file: &Path) -> Result<()>
where
    T: serde::ser::Serialize,
{
    let serialized = toml::to_string_pretty(entity).unwrap();
    let mut fhandle = File::create(path_to_file).await?;
    fhandle.write_all(serialized.as_bytes()).await?;
    fhandle.flush().await?;

    Ok(())
}
