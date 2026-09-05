//! Restartable, three-party migration of authoritative raw iris shares.
//!
//! A generation names one jointly created sharing, not merely its plaintext
//! version. After a partial write all parties must regenerate the affected
//! batch: mixing old and fresh Shamir evaluations would give incorrect scores.
use super::{convert_irises, SpectralIris};
use crate::protocol::shared_iris::GaloisRingSharedIris;
use ampc_actor_utils::{
    execution::session::{Session, SessionHandles},
    network::mpc::NetworkValue,
    protocol::ops::setup_shared_seed,
};
use eyre::{ensure, Result};
use iris_mpc_common::VectorId;
use iris_mpc_store::Store;
use rand::Rng;
use std::sync::Arc;

/// Changing the field, root, coordinate order, or byte-plane layout requires a
/// new version. Format 2 is F_52201, root 43061, frequency-major, eight lanes.
pub const FORMAT_VERSION: i16 = 2;

#[derive(sqlx::FromRow)]
struct StoredSpectralIris {
    serial_id: i64,
    version_id: i16,
    party: i16,
    format_version: i16,
    generation: Vec<u8>,
    payload: Vec<u8>,
    payload_hash: Vec<u8>,
}

/// Persist a newly created sharing before publishing it to the resident scan.
/// Original iris persistence remains authoritative: startup accepts only exact
/// versions whose generation agrees on all three parties.
pub async fn persist_batch(
    store: &Store,
    side: usize,
    party: usize,
    generation: &[u8; 16],
    records: &[(VectorId, Arc<SpectralIris>)],
) -> Result<()> {
    ensure!(side < 2 && party < 3, "invalid spectral party or side");
    let mut tx = store.tx().await?;
    for (id, iris) in records {
        let payload: &[u8] = bytemuck::cast_slice(iris.packed_bytes());
        sqlx::query(
            "INSERT INTO cpu_spectral_irises
             (serial_id, version_id, side, party, format_version, generation, payload, payload_hash)
             VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
             ON CONFLICT (serial_id, side) DO UPDATE SET
             version_id=EXCLUDED.version_id, party=EXCLUDED.party,
             format_version=EXCLUDED.format_version, generation=EXCLUDED.generation,
             payload=EXCLUDED.payload, payload_hash=EXCLUDED.payload_hash",
        )
        .bind(i64::from(id.serial_id()))
        .bind(id.version_id())
        .bind(side as i16)
        .bind(party as i16)
        .bind(FORMAT_VERSION)
        .bind(generation.as_slice())
        .bind(payload)
        .bind(blake3::hash(payload).as_bytes().as_slice())
        .execute(&mut *tx)
        .await?;
    }
    tx.commit().await?;
    Ok(())
}

/// Exchange only public migration metadata (record versions and generation
/// identifiers). Sending both directions ensures all parties choose the same
/// branch even when precisely one party has an incomplete cached generation.
async fn agree(session: &mut Session, payload: &[u8]) -> Result<bool> {
    let network = &mut session.network_session;
    network
        .send_next(NetworkValue::Bytes(payload.to_vec().into()))
        .await?;
    network
        .send_prev(NetworkValue::Bytes(payload.to_vec().into()))
        .await?;
    let NetworkValue::Bytes(prev) = network.receive_prev().await? else {
        eyre::bail!("expected spectral migration metadata");
    };
    let NetworkValue::Bytes(next) = network.receive_next().await? else {
        eyre::bail!("expected spectral migration metadata");
    };
    Ok(prev.as_ref() == payload && next.as_ref() == payload)
}

/// Public, fresh identifier shared by a newly converted batch. Independent of
/// PRF masking randomness; never reuse this identifier for another conversion.
pub async fn new_generation(session: &mut Session) -> Result<[u8; 16]> {
    let seed = rand::thread_rng().gen();
    setup_shared_seed(&mut session.network_session, seed).await
}

/// Load a batch of exact-version cached shares if every party has the same
/// generation. Otherwise privately reconvert and atomically replace the local
/// batch. A crash during the distributed write is repaired on the next startup.
///
/// Callers assign ordered batches to fixed sessions on all parties. The server
/// must remain unavailable to queries until every batch has completed.
pub async fn load_or_convert_batch(
    session: &mut Session,
    store: &Store,
    side: usize,
    ids: &[VectorId],
) -> Result<Vec<(VectorId, Arc<SpectralIris>)>> {
    ensure!(side < 2, "invalid spectral eye");
    ensure!(
        ids.windows(2).all(|w| w[0].serial_id() < w[1].serial_id()),
        "spectral migration IDs must be strictly ordered"
    );
    let party = session.own_role().index();
    // Check the authoritative version manifest before any secret-dependent
    // exchange, including on cache hits. Mismatched snapshots fail closed.
    let mut manifest = vec![side as u8];
    for id in ids {
        manifest.extend_from_slice(&id.serial_id().to_le_bytes());
        manifest.extend_from_slice(&id.version_id().to_le_bytes());
    }
    ensure!(
        agree(session, &manifest).await?,
        "spectral migration requires matching iris versions on all parties"
    );
    let serial_ids: Vec<_> = ids.iter().map(|id| i64::from(id.serial_id())).collect();
    let rows: Vec<StoredSpectralIris> = sqlx::query_as(
        "SELECT serial_id, version_id, party, format_version, generation, payload, payload_hash
         FROM cpu_spectral_irises WHERE side=$1 AND serial_id=ANY($2) ORDER BY serial_id",
    )
    .bind(side as i16)
    .bind(&serial_ids)
    .fetch_all(&store.pool)
    .await?;
    let mut cached = Vec::with_capacity(ids.len());
    let mut generations = Vec::with_capacity(ids.len() * 16 + 1);
    // The leading byte prevents an empty or incomplete cache from appearing to
    // be a complete generation manifest.
    generations.push(1);
    for (id, row) in ids.iter().zip(&rows) {
        if row.serial_id != i64::from(id.serial_id())
            || row.version_id != id.version_id()
            || row.party != party as i16
            || row.format_version != FORMAT_VERSION
            || row.generation.len() != 16
            || row.payload_hash != blake3::hash(&row.payload).as_bytes()
        {
            break;
        }
        let packed = row.payload.iter().map(|&x| x as i8).collect();
        let Ok(iris) = SpectralIris::from_packed(packed) else {
            break;
        };
        generations.extend_from_slice(&row.generation);
        cached.push((*id, Arc::new(iris)));
    }
    let complete = cached.len() == ids.len();
    if !complete {
        generations = vec![0];
    }
    let same_generation = agree(session, &generations).await?;
    if complete && same_generation {
        metrics::counter!("ntt_migration_cached_records_total").increment(ids.len() as u64);
        return Ok(cached);
    }
    drop(cached);
    drop(rows);
    let raw = store
        .get_iris_data_by_ids_for_side(&serial_ids, side)
        .await?;
    ensure!(
        raw.len() == ids.len() && raw.iter().zip(ids).all(|(row, id)| row.vector_id() == *id),
        "authoritative iris version changed during spectral migration"
    );
    let irises = raw
        .iter()
        .map(|row| GaloisRingSharedIris::try_from_buffers_inner(party, row.code(), row.mask()))
        .collect::<Result<Vec<_>>>()?;
    let generation = new_generation(session).await?;
    let field = convert_irises(session, &irises.iter().collect::<Vec<_>>()).await?;
    // Transform work does not block the asynchronous MPC runtime.
    let packed = tokio::task::spawn_blocking(move || {
        field
            .iter()
            .map(|iris| Arc::new(SpectralIris::prepare(iris)))
            .collect::<Vec<_>>()
    })
    .await?;
    let records: Vec<_> = ids.iter().copied().zip(packed).collect();
    persist_batch(store, side, party, &generation, &records).await?;
    metrics::counter!("ntt_migration_converted_records_total").increment(ids.len() as u64);
    Ok(records)
}
