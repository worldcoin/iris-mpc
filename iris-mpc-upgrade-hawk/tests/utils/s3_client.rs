use aws_sdk_s3::{primitives::ByteStream, Client as S3_Client};
use eyre::Result;
use iris_mpc_common::{config::Config, IrisSerialId};

pub async fn clear_s3_iris_deletions(config: &Config, s3_client: &S3_Client) -> Result<()> {
    let s3_bucket = format!("wf-smpcv2-{}-sync-protocol", config.environment);
    let s3_key = format!("{}_deleted_serial_ids.json", config.environment);

    let body = ByteStream::from(Vec::new());

    s3_client
        .put_object()
        .bucket(&s3_bucket)
        .key(&s3_key)
        .body(body)
        .send()
        .await?;

    Ok(())
}
