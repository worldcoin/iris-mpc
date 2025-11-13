use iris_mpc_cpu::{execution::session::Session, network::value::NetworkValue};
use itertools::Itertools;

pub async fn sync_on_job_sizes(session: &mut Session, job_size: usize) -> eyre::Result<usize> {
    tracing::info!("Synchronizing on job size: {}", job_size);
    let sizes = broadcast_usize(session, job_size).await?;
    let min_size = sizes.iter().cloned().min().unwrap();
    tracing::info!(
        "Synchronized job sizes: local = {}, broadcasted = [{}], using min = {}",
        job_size,
        sizes.iter().join(","),
        min_size
    );
    Ok(min_size)
}

pub async fn sync_on_id_hash(session: &mut Session, id_hash: u64) -> eyre::Result<bool> {
    tracing::info!("Synchronizing on job id hash: {}", id_hash);
    let broadcast_hashes = broadcast_u64(session, id_hash).await?;
    if broadcast_hashes.iter().all(|&h| h == id_hash) {
        Ok(true)
    } else {
        tracing::info!(
            "Mismatched id hashes detected: local hash = {}, broadcasted hashes = {:?}",
            id_hash,
            broadcast_hashes
        );
        Ok(false)
    }
}

async fn broadcast(session: &mut Session, value: NetworkValue) -> eyre::Result<[NetworkValue; 3]> {
    session.network_session.send_next(value.clone()).await?;
    session.network_session.send_prev(value.clone()).await?;
    let next_value = session.network_session.receive_next().await?;
    let prev_value = session.network_session.receive_prev().await?;

    match session.network_session.own_role.index() {
        0 => Ok([value, next_value, prev_value]),
        1 => Ok([prev_value, value, next_value]),
        2 => Ok([next_value, prev_value, value]),
        _ => Err(eyre::eyre!("Invalid party id")),
    }
}

async fn broadcast_usize(session: &mut Session, value: usize) -> eyre::Result<[usize; 3]> {
    let size = u64::try_from(value).expect("value fits in u64");
    let broadcasted_sizes = broadcast_u64(session, size).await?;
    Ok(broadcasted_sizes.map(|x| usize::try_from(x).expect("value fits into usize")))
}

async fn broadcast_u64(session: &mut Session, value: u64) -> eyre::Result<[u64; 3]> {
    let network_value = NetworkValue::Bytes(value.to_le_bytes().to_vec());
    let broadcasted_values = broadcast(session, network_value).await?;

    let mut result = [0u64; 3];
    for (i, nv) in broadcasted_values.iter().enumerate() {
        if let NetworkValue::Bytes(b) = nv {
            if b.len() != 8 {
                return Err(eyre::eyre!("Unexpected byte length for usize"));
            }
            let mut arr = [0u8; 8];
            arr.copy_from_slice(&b[..8]);
            result[i] = u64::from_le_bytes(arr);
        } else {
            return Err(eyre::eyre!("Unexpected network value type"));
        }
    }
    Ok(result)
}
