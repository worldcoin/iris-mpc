use crate::utils::pgres::NetDbProvider;
use eyre::Result;
use iris_mpc_common::{IrisSerialId, PARTY_IDX_SET};
use iris_mpc_cpu::protocol::shared_iris::{GaloisRingSharedIrisPair, GaloisRingSharedIrisPairSet};
use iris_mpc_store::{Store as IrisStore, StoredIrisRef};
use itertools::{IntoChunks, Itertools};

/// Persists a stream of Iris shares batches to remote databases.
///
/// # Arguments
///
/// * `db_provider` - Network wide dB provider.
/// * `iris_shares_stream` - A generator of batches of Iris shares to be inserted into stores.
/// * `tx_batch_size` - Constraint over number of Iris shares to persist in a single pgres transaction.
///
/// # Returns
///
/// A set of ranges of inserted Iris serial identifiers.
///
pub async fn insert_iris_shares(
    db_provider: &NetDbProvider,
    iris_shares_stream: &IntoChunks<impl Iterator<Item = Box<GaloisRingSharedIrisPairSet>>>,
    tx_batch_size: usize,
) -> Result<Vec<Vec<(IrisSerialId, IrisSerialId)>>> {
    // TODO: refactor using iter.map.collect ...etc.
    let mut result = Vec::new();
    for chunk in iris_shares_stream.into_iter() {
        result.push(
            insert_iris_shares_batch(
                chunk.into_iter().map(|x| x.to_vec()).collect_vec(),
                db_provider,
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
    batch: Vec<Vec<GaloisRingSharedIrisPair>>,
    db_provider: &NetDbProvider,
    tx_batch_size: usize,
) -> Result<Vec<(IrisSerialId, IrisSerialId)>> {
    // TODO: refactor using iter.map.collect ...etc.
    let mut result = Vec::new();
    for party_idx in PARTY_IDX_SET {
        let iris_store = db_provider.of_node(party_idx).gpu_store().iris_store();
        let iris_shares = batch.iter().map(|i| i[party_idx].to_owned()).collect_vec();
        result.push(
            insert_iris_shares_batch_item(iris_shares, iris_store, tx_batch_size)
                .await
                .unwrap(),
        );
    }

    Ok(result)
}

/// Persists Iris shares to a remote database.
async fn insert_iris_shares_batch_item(
    iris_shares: Vec<GaloisRingSharedIrisPair>,
    iris_store: &IrisStore,
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
