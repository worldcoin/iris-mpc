use eyre::ensure;
use std::path::Path;
use tokio::fs;

/// The name of the file that stores the previous iris index as a string.
pub const PREV_IRIS_INDEX_FILE: &str = "prev_iris_index.txt";

pub type IrisSerialId = i64;

pub async fn fetch_height_of_indexed() -> IrisSerialId {
    async fn try_fetch_from_disk() -> eyre::Result<IrisSerialId> {
        let prev_iris_index_path = Path::new(PREV_IRIS_INDEX_FILE);
        prev_iris_index_path.try_exists()?;
        ensure!(
            prev_iris_index_path.is_file(),
            format!("{} is not a file.", prev_iris_index_path.display())
        );
        let file_content = fs::read_to_string(prev_iris_index_path).await?;
        Ok(file_content.parse::<IrisSerialId>()?)
    }
    try_fetch_from_disk().await.unwrap_or(1)
}
