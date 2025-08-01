use aws_sdk_s3::{primitives::ByteStream, Client as S3_Client};
use eyre::Result;
use iris_mpc_common::config::Config;
use iris_mpc_cpu::genesis::state_accessor::{
    get_s3_iris_deletions_key, get_s3_sync_protocol_bucket, S3IrisDeletions,
};

pub async fn clear_s3_iris_deletions(config: &Config, s3_client: &S3_Client) -> Result<()> {
    let s3_bucket = get_s3_sync_protocol_bucket(config);
    let s3_key = get_s3_iris_deletions_key(config);

    let obj = S3IrisDeletions {
        deleted_serial_ids: vec![],
    };
    let obj_bytes = serde_json::to_vec(&obj)?;
    let body = ByteStream::from(obj_bytes);

    s3_client
        .put_object()
        .bucket(&s3_bucket)
        .key(&s3_key)
        .body(body)
        .send()
        .await?;

    Ok(())
}
