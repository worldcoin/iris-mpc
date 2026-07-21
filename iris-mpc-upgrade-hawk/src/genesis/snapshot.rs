//! Database snapshot phase for genesis.
//!
//! Optionally taken after indexation completes: creates an RDS cluster
//! snapshot tagged with the indexed range. The entry point is
//! [`exec_snapshot`].

use aws_sdk_rds::Client as RDSClient;
use chrono::Utc;
use iris_mpc_cpu::genesis::IndexationError;

use super::ExecutionContextInfo;

/// Takes a dB snapshot.
///
/// # Arguments
///
/// * `ctx` - Execution context information.
/// * `aws_rds_client` - AWS RDS SDK client.
///
pub(super) async fn exec_snapshot(
    ctx: &ExecutionContextInfo,
    aws_rds_client: &RDSClient,
) -> Result<(), IndexationError> {
    tracing::info!("Db snapshot begins");

    // Set snapshot ID.
    let unix_timestamp = Utc::now().timestamp();
    let snapshot_id = format!(
        "genesis-{}-{}-{}-{}",
        ctx.last_indexed_id,
        ctx.args.max_indexation_id,
        ctx.args.batch_size_config.to_aws_identifier(),
        unix_timestamp
    );

    // Set cluster ID.
    let db_config = ctx.config.cpu_database.as_ref().unwrap();
    let url = db_config
        .url
        .strip_prefix("postgresql://")
        .ok_or(IndexationError::AwsRdsInvalidClusterURL)?;
    let at_pos = url
        .rfind('@')
        .ok_or(IndexationError::AwsRdsInvalidClusterURL)?;
    let host_and_db = &url[at_pos + 1..];
    let slash_pos = host_and_db.find('/').unwrap_or(host_and_db.len());
    let cluster_endpoint = &host_and_db[..slash_pos];
    let resp = aws_rds_client
        .describe_db_clusters()
        .send()
        .await
        .map_err(|_| IndexationError::AwsRdsGetClusterURLs)?;
    let cluster_id = resp
        .db_clusters()
        .iter()
        .find(|cluster| cluster.endpoint() == Some(cluster_endpoint))
        .and_then(|cluster| cluster.db_cluster_identifier())
        .ok_or(IndexationError::AwsRdsClusterIdNotFound)?;

    // Create cluster snapshot.
    tracing::info!(
        "Creating RDS snapshot for cluster: cluster-id={} :: snapshot-id={}",
        cluster_id,
        snapshot_id.clone()
    );
    aws_rds_client
        .create_db_cluster_snapshot()
        .db_cluster_identifier(cluster_id)
        .db_cluster_snapshot_identifier(snapshot_id.clone())
        .send()
        .await
        .map_err(|err| {
            tracing::error!("Failed to create db snapshot: {}", err);
            IndexationError::AwsRdsCreateSnapshotFailure(err.to_string())
        })?;
    tracing::info!(
        "Created RDS snapshot for cluster: cluster-id={} :: snapshot-id={}",
        cluster_id,
        snapshot_id
    );

    Ok(())
}
