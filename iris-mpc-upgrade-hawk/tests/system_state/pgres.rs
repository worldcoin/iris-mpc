use crate::utils::{constants::PARTY_IDX_SET, pgres::NetDbProvider, types::NodeType};
use eyre::Result;
use futures::future::join_all;
use iris_mpc_common::IrisSerialId;
use iris_mpc_cpu::protocol::shared_iris::{GaloisRingSharedIrisPair, GaloisRingSharedIrisPairSet};
use iris_mpc_store::{Store as IrisStore, StoredIrisRef};
use itertools::{IntoChunks, Itertools};

/// Returns a set of iris counts in each GPU node store.
///
/// # Arguments
///
/// * `db_provider` - Network wide dB provider.
/// * `node_type` - Type of node.
///
/// # Returns
///
/// A set of iris counts in each GPU node store.
///
pub async fn get_iris_counts(
    db_provider: &NetDbProvider,
    node_type: &NodeType,
) -> Result<Vec<usize>> {
    Ok(join_all(
        db_provider
            .iris_stores(node_type)
            .into_iter()
            .map(|iris_store| async { iris_store.count_irises().await.unwrap() }),
    )
    .await)
}

/// Persists a stream of Iris shares batches to remote databases.
///
/// # Arguments
///
/// * `db_provider` - Network wide dB provider.
/// * `node_type` - Type of node.
/// * `iris_shares_stream` - A generator of batches of Iris shares to be inserted into stores.
/// * `tx_batch_size` - Constraint over number of Iris shares to persist in a single pgres transaction.
///
/// # Returns
///
/// A set of ranges of inserted Iris serial identifiers.
///
pub async fn insert_iris_shares(
    db_provider: &NetDbProvider,
    node_type: NodeType,
    iris_shares_stream: &IntoChunks<impl Iterator<Item = Box<GaloisRingSharedIrisPairSet>>>,
    tx_batch_size: usize,
) -> Result<Vec<Vec<(IrisSerialId, IrisSerialId)>>> {
    let mut result = Vec::new();
    for chunk in iris_shares_stream.into_iter() {
        result.push(
            insert_iris_shares_batch(
                db_provider,
                &node_type,
                chunk.into_iter().map(|x| x.to_vec()).collect_vec(),
                tx_batch_size,
            )
            .await
            .unwrap(),
        );
    }

    Ok(result)
}

/// Persists a batch of Iris shares to remote databases.
async fn insert_iris_shares_batch(
    db_provider: &NetDbProvider,
    node_type: &NodeType,
    batch: Vec<Vec<GaloisRingSharedIrisPair>>,
    tx_batch_size: usize,
) -> Result<Vec<(IrisSerialId, IrisSerialId)>> {
    let mut result = Vec::new();
    for party_idx in PARTY_IDX_SET {
        let iris_shares = batch.iter().map(|i| i[party_idx].to_owned()).collect_vec();
        result.push(
            insert_iris_shares_batch_item(
                db_provider.iris_store(party_idx, node_type),
                iris_shares,
                tx_batch_size,
            )
            .await
            .unwrap(),
        );
    }

    Ok(result)
}

/// Persists Iris shares to a remote database.
async fn insert_iris_shares_batch_item(
    iris_store: &IrisStore,
    iris_shares: Vec<GaloisRingSharedIrisPair>,
    tx_batch_size: usize,
) -> Result<(IrisSerialId, IrisSerialId)> {
    // Set insertion identifier range.
    let start_serial_id = iris_store.get_max_serial_id().await.unwrap_or(0) + 1;
    let end_serial_id = start_serial_id + iris_shares.len() - 1;

    // Insert batches commiting a pgres tx for each one.
    let mut tx = iris_store.tx().await?;
    for batch in &iris_shares.iter().enumerate().chunks(tx_batch_size) {
        let iris_refs: Vec<_> = batch
            .map(|(idx, (iris_l, iris_r))| StoredIrisRef {
                id: (start_serial_id + idx) as i64,
                left_code: &iris_l.code.coefs,
                left_mask: &iris_l.mask.coefs,
                right_code: &iris_r.code.coefs,
                right_mask: &iris_r.mask.coefs,
            })
            .collect();

        iris_store.insert_irises(&mut tx, &iris_refs).await?;
        tx.commit().await?;
        tx = iris_store.tx().await?;
    }

    Ok((
        start_serial_id as IrisSerialId,
        end_serial_id as IrisSerialId,
    ))
}
