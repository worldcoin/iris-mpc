//! Full serving-memory consistency check for the CPU/Hawk backend.

use std::sync::Arc;

use ampc_actor_utils::execution::session::NetworkSession;
use eyre::{bail, ensure, Result};
use iris_mpc_common::consistency_canary::{
    derive_challenge, fresh_challenge_contribution, private_any_nonzero, BooleanMpcTransport,
    CanaryAccumulator,
};
use iris_mpc_common::VectorId;
use rayon::prelude::*;

use super::iris_worker::{IrisWorkerInventory, IrisWorkerPool};
use super::{BothEyes, LEFT, RIGHT};
use crate::network::mpc::NetworkValue;
use crate::protocol::shared_iris::ArcIris;

const FETCH_CHUNK_SIZE: usize = 1024;
const INVENTORY_LEN: usize = 8 + 32;
const BARRIER_PACKET_LEN: usize = 3 * INVENTORY_LEN + 32;
const INVALID_INVENTORY: IrisWorkerInventory = IrisWorkerInventory {
    entries: u64::MAX,
    digest: [0xff; 32],
};

fn barrier_packet(
    registry: IrisWorkerInventory,
    left: Option<IrisWorkerInventory>,
    right: Option<IrisWorkerInventory>,
    contribution: [u8; 32],
) -> [u8; BARRIER_PACKET_LEN] {
    let mut packet = [0; BARRIER_PACKET_LEN];
    for (offset, inventory) in [Some(registry), left, right].into_iter().enumerate() {
        let inventory = inventory.unwrap_or(INVALID_INVENTORY);
        let start = offset * INVENTORY_LEN;
        packet[start..start + 8].copy_from_slice(&inventory.entries.to_le_bytes());
        packet[start + 8..start + INVENTORY_LEN].copy_from_slice(&inventory.digest);
    }
    packet[3 * INVENTORY_LEN..].copy_from_slice(&contribution);
    packet
}

fn decode_barrier_packet(packet: NetworkValue) -> Result<([IrisWorkerInventory; 3], [u8; 32])> {
    let NetworkValue::Bytes(packet) = packet else {
        bail!("canary readiness barrier received an unexpected message")
    };
    ensure!(
        packet.len() == BARRIER_PACKET_LEN,
        "canary readiness barrier packet has invalid length"
    );
    let decode = |offset: usize| IrisWorkerInventory {
        entries: u64::from_le_bytes(packet[offset..offset + 8].try_into().expect("checked")),
        digest: packet[offset + 8..offset + INVENTORY_LEN]
            .try_into()
            .expect("checked"),
    };
    Ok((
        [decode(0), decode(INVENTORY_LEN), decode(2 * INVENTORY_LEN)],
        packet[3 * INVENTORY_LEN..].try_into().expect("checked"),
    ))
}

/// The one readiness barrier. Every party publishes its registry inventory,
/// both observed worker inventories (or an error marker), and fresh randomness.
/// Thus a local inventory failure makes all parties fail before scanning.
async fn agree_inventory_and_challenge(
    net: &mut NetworkSession,
    registry: IrisWorkerInventory,
    left: Option<IrisWorkerInventory>,
    right: Option<IrisWorkerInventory>,
    context: [u8; 32],
) -> Result<[u8; 32]> {
    let mine = fresh_challenge_contribution();
    let own = [
        registry,
        left.unwrap_or(INVALID_INVENTORY),
        right.unwrap_or(INVALID_INVENTORY),
    ];
    let packet = NetworkValue::Bytes(barrier_packet(registry, left, right, mine).to_vec());
    net.send_next(packet.clone()).await?;
    net.send_prev(packet).await?;
    let (previous_inventories, previous) = decode_barrier_packet(net.receive_prev().await?)?;
    let (next_inventories, next) = decode_barrier_packet(net.receive_next().await?)?;
    ensure!(
        [own, previous_inventories, next_inventories]
            .iter()
            .all(|inventories| inventories.iter().all(|inventory| *inventory == registry)),
        "canary registry/worker inventory mismatch at readiness barrier"
    );
    Ok(derive_challenge(context, [mine, previous, next]))
}

struct SessionBooleanTransport<'a> {
    party_id: usize,
    net: &'a mut NetworkSession,
}

impl BooleanMpcTransport for SessionBooleanTransport<'_> {
    fn party_id(&self) -> usize {
        self.party_id
    }

    async fn exchange_next(&mut self, message: Vec<u8>) -> Result<Vec<u8>> {
        self.net.send_next(NetworkValue::Bytes(message)).await?;
        match self.net.receive_prev().await? {
            NetworkValue::Bytes(message) => Ok(message),
            _ => bail!("canary Boolean MPC received an unexpected message"),
        }
    }

    async fn exchange_previous(&mut self, message: Vec<u8>) -> Result<Vec<u8>> {
        self.net.send_prev(NetworkValue::Bytes(message)).await?;
        match self.net.receive_next().await? {
            NetworkValue::Bytes(message) => Ok(message),
            _ => bail!("canary Boolean MPC received an unexpected message"),
        }
    }
}

fn accumulate_row(
    accumulator: &mut CanaryAccumulator,
    id: VectorId,
    left: &ArcIris,
    right: &ArcIris,
) -> Result<()> {
    accumulator.accumulate(
        id.serial_id(),
        &left.code.coefs,
        &left.mask.coefs,
        &right.code.coefs,
        &right.mask.coefs,
    )
}

pub async fn run_consistency_canary(
    worker_pools: &BothEyes<Arc<dyn IrisWorkerPool>>,
    vector_ids: &[VectorId],
    registry_consistent: bool,
    party_id: usize,
    repetitions: usize,
    context: [u8; 32],
    net: &mut NetworkSession,
) -> Result<()> {
    let registry = if registry_consistent {
        IrisWorkerInventory::from_sorted_ids(vector_ids)
    } else {
        INVALID_INVENTORY
    };
    let inventories = tokio::join!(
        worker_pools[LEFT].iris_inventory(),
        worker_pools[RIGHT].iris_inventory(),
    );
    let challenge = agree_inventory_and_challenge(
        net,
        registry,
        inventories.0.as_ref().ok().copied(),
        inventories.1.as_ref().ok().copied(),
        context,
    )
    .await?;

    let scan: Result<CanaryAccumulator> = async {
        let mut accumulator = CanaryAccumulator::new(party_id, repetitions, challenge);
        for ids in vector_ids.chunks(FETCH_CHUNK_SIZE) {
            let (left, right) = tokio::try_join!(
                worker_pools[LEFT].fetch_irises_strict(ids.to_vec()),
                worker_pools[RIGHT].fetch_irises_strict(ids.to_vec()),
            )?;
            ensure!(
                left.len() == ids.len() && right.len() == ids.len(),
                "canary strict fetch returned an incomplete chunk"
            );
            let chunk = (0..ids.len())
                .into_par_iter()
                .try_fold(
                    || CanaryAccumulator::new(party_id, repetitions, challenge),
                    |mut local, index| -> Result<_> {
                        accumulate_row(&mut local, ids[index], &left[index], &right[index])?;
                        Ok(local)
                    },
                )
                .try_reduce(
                    || CanaryAccumulator::new(party_id, repetitions, challenge),
                    |mut left, right| -> Result<_> {
                        left.merge(right)?;
                        Ok(left)
                    },
                )?;
            accumulator.merge(chunk)?;
        }
        Ok(accumulator)
    }
    .await;

    let local_scan_failed = scan.is_err();
    let rows = scan.as_ref().map_or(0, CanaryAccumulator::rows);
    let mut share = scan.map_or_else(
        |_| vec![0; repetitions * 4],
        CanaryAccumulator::into_syndrome_share,
    );
    share.push(u16::from(local_scan_failed));
    let mut transport = SessionBooleanTransport { party_id, net };
    let nonzero = private_any_nonzero(share, repetitions, challenge, &mut transport).await?;
    ensure!(
        !nonzero,
        "CONSISTENCY CANARY FAILED after checking {rows} rows; refusing to serve"
    );
    tracing::info!(rows, repetitions, "Full consistency canary passed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::local::LocalRuntime;
    use iris_mpc_common::consistency_canary::{
        private_any_nonzero, DEFAULT_CANARY_REPETITIONS, STARTUP_CANARY_CONTEXT,
    };
    use iris_mpc_common::iris_db::db::IrisDB;
    use rand::SeedableRng;
    use tokio::task::JoinSet;

    use crate::protocol::shared_iris::GaloisRingSharedIris;

    #[derive(Clone, Copy)]
    enum Fault {
        None,
        CorruptPartyOne,
        OmitPartyOne,
    }

    async fn run_canary(fault: Fault) -> Vec<Result<()>> {
        let sessions = LocalRuntime::mock_sessions_with_channel().await.unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let left = IrisDB::new_random_rng(1, &mut rng).db;
        let right = IrisDB::new_random_rng(1, &mut rng).db;
        let mut shares: Vec<Vec<(ArcIris, ArcIris)>> = vec![Vec::new(); 3];
        let left_shares = GaloisRingSharedIris::generate_shares_locally(&mut rng, left[0].clone());
        let right_shares =
            GaloisRingSharedIris::generate_shares_locally(&mut rng, right[0].clone());
        for party in 0..3 {
            shares[party].push((
                Arc::new(left_shares[party].clone()),
                Arc::new(right_shares[party].clone()),
            ));
        }
        if matches!(fault, Fault::CorruptPartyOne) {
            let iris = Arc::make_mut(&mut shares[1][0].0);
            iris.code.coefs[41] = iris.code.coefs[41].wrapping_add(1);
        }

        let mut tasks = JoinSet::new();
        for (party, session) in sessions.iter().enumerate() {
            let session = session.clone();
            let rows = shares[party].clone();
            tasks.spawn(async move {
                let mut guard = session.lock().await;
                let net = &mut guard.network_session;
                let inventory =
                    IrisWorkerInventory::from_sorted_ids(&[VectorId::from_serial_id(1)]);
                let challenge = agree_inventory_and_challenge(
                    net,
                    inventory,
                    Some(inventory),
                    Some(inventory),
                    STARTUP_CANARY_CONTEXT,
                )
                .await?;
                let mut accumulator =
                    CanaryAccumulator::new(party, DEFAULT_CANARY_REPETITIONS, challenge);
                if !(party == 1 && matches!(fault, Fault::OmitPartyOne)) {
                    let (left, right) = &rows[0];
                    accumulate_row(&mut accumulator, VectorId::from_serial_id(1), left, right)?;
                }
                let mut share = accumulator.into_syndrome_share();
                share.push(0);
                let mut transport = SessionBooleanTransport {
                    party_id: party,
                    net,
                };
                let nonzero = private_any_nonzero(
                    share,
                    DEFAULT_CANARY_REPETITIONS,
                    challenge,
                    &mut transport,
                )
                .await?;
                ensure!(!nonzero, "canary detected an inconsistent one-row database");
                Ok(())
            });
        }
        tasks.join_all().await
    }

    #[tokio::test]
    async fn one_row_valid_database_passes_private_zero_test() {
        for result in run_canary(Fault::None).await {
            result.unwrap();
        }
    }

    #[tokio::test]
    async fn one_row_corrupt_share_fails_without_opening_syndrome() {
        assert!(run_canary(Fault::CorruptPartyOne)
            .await
            .iter()
            .all(Result::is_err));
    }

    #[tokio::test]
    async fn one_row_missing_share_fails_without_opening_syndrome() {
        assert!(run_canary(Fault::OmitPartyOne)
            .await
            .iter()
            .all(Result::is_err));
    }

    #[tokio::test]
    async fn inventory_mismatch_fails_every_party_without_deadlock() {
        let sessions = LocalRuntime::mock_sessions_with_channel().await.unwrap();
        let mut tasks = JoinSet::new();
        for (party, session) in sessions.iter().enumerate() {
            let session = session.clone();
            tasks.spawn(async move {
                let ids = if party == 1 {
                    Vec::new()
                } else {
                    vec![VectorId::from_serial_id(1)]
                };
                let inventory = IrisWorkerInventory::from_sorted_ids(&ids);
                let mut guard = session.lock().await;
                agree_inventory_and_challenge(
                    &mut guard.network_session,
                    inventory,
                    Some(inventory),
                    Some(inventory),
                    STARTUP_CANARY_CONTEXT,
                )
                .await
            });
        }
        assert!(tasks.join_all().await.iter().all(Result::is_err));
    }

    #[tokio::test]
    async fn local_scan_failure_sets_only_the_private_zero_test_bit() {
        let sessions = LocalRuntime::mock_sessions_with_channel().await.unwrap();
        let mut tasks = JoinSet::new();
        for (party, session) in sessions.iter().enumerate() {
            let session = session.clone();
            tasks.spawn(async move {
                let mut guard = session.lock().await;
                let mut share = vec![0; DEFAULT_CANARY_REPETITIONS * 4];
                share.push(u16::from(party == 1));
                let mut transport = SessionBooleanTransport {
                    party_id: party,
                    net: &mut guard.network_session,
                };
                private_any_nonzero(share, DEFAULT_CANARY_REPETITIONS, [5; 32], &mut transport)
                    .await
            });
        }
        for result in tasks.join_all().await {
            assert!(result.unwrap());
        }
    }
}
