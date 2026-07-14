//! Private end-of-epoch consistency check for persisted rerandomized shares.
//!
//! The three sweepers for one store kind establish a dedicated private MPC
//! session before opening database snapshots. Each snapshot must remain at
//! repeatable-read isolation from inventory collection through accumulation.

use std::future::Future;

use ampc_actor_utils::network::{
    mpc::{
        handle::{build_network_handle, control_channel::ControlChannel, NetworkHandleArgs},
        NetworkValue,
    },
    tcp::{connection::client::TlsClient, TlsClientConfig, TlsConfig},
};
use eyre::{bail, ensure, eyre, Result};
use iris_mpc_common::consistency_canary::{
    derive_challenge, fresh_challenge_contribution, private_any_nonzero, BooleanMpcTransport,
    CanaryAccumulator, DEFAULT_CANARY_REPETITIONS,
};
use iris_mpc_store::rerand::{
    begin_rerand_verification_snapshot, RerandContext, RerandVerificationSnapshot,
    RERAND_CHECK_PROTOCOL_VERSION,
};
use iris_mpc_store::Store;
use tokio_util::sync::CancellationToken;

use super::coordination::OFFSET_GENERATION;
use super::{RerandStoreKind, RerandSweeperConfig};

const CONTEXT_DOMAIN: &str = "iris-mpc/rerand-v2/persisted-epoch-check/context/v1";
const BARRIER_MAGIC: &[u8; 8] = b"IRECHK01";
const BARRIER_LEN: usize = BARRIER_MAGIC.len() + 1 + 32 + 8 + 32 + 32;
const INVALID_INVENTORY: EpochCheckInventory = EpochCheckInventory {
    rows: u64::MAX,
    digest: [0xff; 32],
};

struct ControlChannelBooleanTransport<'a> {
    party_id: usize,
    channel: &'a mut dyn ControlChannel,
}

impl BooleanMpcTransport for ControlChannelBooleanTransport<'_> {
    fn party_id(&self) -> usize {
        self.party_id
    }

    async fn exchange_next(&mut self, message: Vec<u8>) -> Result<Vec<u8>> {
        self.channel.send_next(NetworkValue::Bytes(message)).await?;
        match self.channel.recv_prev().await? {
            NetworkValue::Bytes(message) => Ok(message),
            _ => bail!("epoch-check MPC received an unexpected message"),
        }
    }

    async fn exchange_previous(&mut self, message: Vec<u8>) -> Result<Vec<u8>> {
        self.channel.send_prev(NetworkValue::Bytes(message)).await?;
        match self.channel.recv_next().await? {
            NetworkValue::Bytes(message) => Ok(message),
            _ => bail!("epoch-check MPC received an unexpected message"),
        }
    }
}

/// Public inventory committed by the database snapshot before the random
/// compression challenge is known. Its digest covers ordered `(id,
/// version_id)` entries only. Epoch and semantic metadata are validated
/// locally, since those values legitimately differ between physical stores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EpochCheckInventory {
    pub rows: u64,
    pub digest: [u8; 32],
}

/// Common, non-secret values which distinguish this check from every other
/// rerandomization epoch and store-kind protocol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct EpochCheckBinding {
    environment: String,
    coordination_id: String,
    store_kind: RerandStoreKind,
    store_registry_commitment: [u8; 32],
    epoch: u32,
    seed_commitment: [u8; 32],
}

impl EpochCheckBinding {
    pub(super) fn new(
        config: &RerandSweeperConfig,
        store_registry_commitment: [u8; 32],
        epoch: u32,
        seed_commitment: [u8; 32],
    ) -> Self {
        Self {
            environment: config.environment.clone(),
            coordination_id: config.coordination_id.clone(),
            store_kind: config.store_kind,
            store_registry_commitment,
            epoch,
            seed_commitment,
        }
    }
}

/// A live repeatable-read snapshot. Implementations must compute `inventory`
/// before returning the guard and keep the same database snapshot alive until
/// `accumulate` and `finish` return.
///
/// The rows passed to [`CanaryAccumulator`] must be normalized to epoch zero.
/// This permits legitimate per-party CAS races and post-pass epoch-zero writes
/// while checking that the persisted data still forms valid three-party
/// sharings.
#[allow(async_fn_in_trait)]
trait EpochCheckSnapshot {
    fn inventory(&self) -> EpochCheckInventory;

    async fn accumulate(
        &mut self,
        party_id: usize,
        repetitions: usize,
        challenge: [u8; 32],
    ) -> Result<CanaryAccumulator>;

    /// Close the read-only transaction. A close failure is included in the
    /// private failure bit rather than allowing peers to block.
    async fn finish(self) -> Result<()>;
}

/// Adapter for the store crate's owned repeatable-read guard. Keeping this
/// small wrapper here avoids making the persistence crate depend on the
/// upgrade protocol crate.
struct StoreEpochCheckSnapshot<'a> {
    snapshot: RerandVerificationSnapshot,
    rerand: &'a RerandContext,
}

impl EpochCheckSnapshot for StoreEpochCheckSnapshot<'_> {
    fn inventory(&self) -> EpochCheckInventory {
        let inventory = self.snapshot.inventory();
        EpochCheckInventory {
            rows: inventory.rows,
            digest: inventory.digest,
        }
    }

    async fn accumulate(
        &mut self,
        party_id: usize,
        repetitions: usize,
        challenge: [u8; 32],
    ) -> Result<CanaryAccumulator> {
        ensure!(
            self.rerand.party_id() == party_id,
            "epoch-check snapshot belongs to another party"
        );
        self.snapshot
            .accumulate(self.rerand, repetitions, challenge)
            .await
    }

    async fn finish(self) -> Result<()> {
        self.snapshot.finish().await
    }
}

/// Open and bind the concrete store snapshot used by
/// [`run_persisted_epoch_check`]. The database guard itself enforces
/// repeatable-read isolation and validates all row-local rerandomization
/// metadata before returning.
async fn open_store_epoch_check_snapshot<'a>(
    store: &Store,
    rerand: &'a RerandContext,
    expected: &EpochCheckBinding,
) -> Result<StoreEpochCheckSnapshot<'a>> {
    let snapshot = begin_rerand_verification_snapshot(&store.pool, expected.epoch).await?;
    let binding = snapshot.binding();
    ensure!(
        binding.epoch == expected.epoch && binding.seed_commitment == expected.seed_commitment,
        "database epoch-check binding differs from the verified protocol binding"
    );
    Ok(StoreEpochCheckSnapshot { snapshot, rerand })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BarrierPacket {
    ready: bool,
    context: [u8; 32],
    inventory: EpochCheckInventory,
    contribution: [u8; 32],
}

impl BarrierPacket {
    fn encode(self) -> Vec<u8> {
        let mut output = Vec::with_capacity(BARRIER_LEN);
        output.extend_from_slice(BARRIER_MAGIC);
        output.push(u8::from(self.ready));
        output.extend_from_slice(&self.context);
        output.extend_from_slice(&self.inventory.rows.to_le_bytes());
        output.extend_from_slice(&self.inventory.digest);
        output.extend_from_slice(&self.contribution);
        debug_assert_eq!(output.len(), BARRIER_LEN);
        output
    }

    fn decode(value: NetworkValue) -> Result<Self> {
        let NetworkValue::Bytes(packet) = value else {
            bail!("epoch-check barrier received an unexpected message")
        };
        ensure!(
            packet.len() == BARRIER_LEN && &packet[..BARRIER_MAGIC.len()] == BARRIER_MAGIC,
            "epoch-check barrier packet is malformed"
        );
        let ready_offset = BARRIER_MAGIC.len();
        ensure!(
            packet[ready_offset] <= 1,
            "epoch-check ready flag is not a bit"
        );
        let context_offset = ready_offset + 1;
        let rows_offset = context_offset + 32;
        let digest_offset = rows_offset + 8;
        let contribution_offset = digest_offset + 32;
        Ok(Self {
            ready: packet[ready_offset] != 0,
            context: packet[context_offset..rows_offset].try_into()?,
            inventory: EpochCheckInventory {
                rows: u64::from_le_bytes(packet[rows_offset..digest_offset].try_into()?),
                digest: packet[digest_offset..contribution_offset].try_into()?,
            },
            contribution: packet[contribution_offset..].try_into()?,
        })
    }
}

fn hash_len_prefixed(hasher: &mut blake3::Hasher, value: &[u8]) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

/// Derive the public context to which both the readiness barrier and private
/// Boolean frames are bound.
fn epoch_check_context(
    binding: &EpochCheckBinding,
    inventory: EpochCheckInventory,
    repetitions: usize,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(CONTEXT_DOMAIN);
    hasher.update(&OFFSET_GENERATION.to_le_bytes());
    hasher.update(&RERAND_CHECK_PROTOCOL_VERSION.to_le_bytes());
    hash_len_prefixed(&mut hasher, binding.environment.as_bytes());
    hash_len_prefixed(&mut hasher, binding.coordination_id.as_bytes());
    hasher.update(&[match binding.store_kind {
        RerandStoreKind::Gpu => 0,
        RerandStoreKind::Hnsw => 1,
    }]);
    hasher.update(&binding.store_registry_commitment);
    hasher.update(&binding.epoch.to_le_bytes());
    hasher.update(&binding.seed_commitment);
    hasher.update(&(repetitions as u64).to_le_bytes());
    hasher.update(&inventory.rows.to_le_bytes());
    hasher.update(&inventory.digest);
    *hasher.finalize().as_bytes()
}

async fn agree_inventory_and_challenge<T: BooleanMpcTransport>(
    transport: &mut T,
    binding: &EpochCheckBinding,
    ready: bool,
    inventory: EpochCheckInventory,
    repetitions: usize,
) -> Result<[u8; 32]> {
    let own = BarrierPacket {
        ready,
        context: epoch_check_context(binding, inventory, repetitions),
        inventory,
        contribution: fresh_challenge_contribution(),
    };
    let message = own.encode();
    // The first ring exchange commits each party's contribution to one peer
    // before it receives the other peer's contribution. Sending inconsistent
    // packets in the second direction produces different challenges and makes
    // the domain-bound private frames fail closed.
    let previous = BarrierPacket::decode(NetworkValue::Bytes(
        transport.exchange_next(message.clone()).await?,
    ))?;
    let next = BarrierPacket::decode(NetworkValue::Bytes(
        transport.exchange_previous(message).await?,
    ))?;
    ensure!(
        [own, previous, next].iter().all(|packet| {
            packet.ready && packet.context == own.context && packet.inventory == own.inventory
        }),
        "epoch-check binding or inventory mismatch at readiness barrier"
    );
    Ok(derive_challenge(
        own.context,
        [own.contribution, previous.contribution, next.contribution],
    ))
}

fn validated_tls(config: &RerandSweeperConfig) -> Result<TlsConfig> {
    ensure!(
        config.party_id < 3
            && config.check_addresses.len() == 3
            && config.check_outbound_addresses.len() == 3
            && config
                .check_addresses
                .iter()
                .chain(&config.check_outbound_addresses)
                .all(|address| !address.trim().is_empty()),
        "epoch check requires one inbound and outbound address for each of three parties"
    );
    let tls = config
        .check_tls
        .as_tls_config()
        .ok_or_else(|| eyre!("epoch-check networking requires mutual TLS"))?;
    ensure!(
        tls.private_key
            .as_ref()
            .is_some_and(|value| !value.trim().is_empty())
            && tls
                .leaf_cert
                .as_ref()
                .is_some_and(|value| !value.trim().is_empty())
            && !tls.root_certs.is_empty()
            && tls.root_certs.iter().all(|value| !value.trim().is_empty()),
        "epoch-check mutual TLS configuration is incomplete"
    );
    Ok(tls)
}

/// Fail before a sweeper mutates any rows if its mandatory private check
/// network is not completely configured.
pub(super) async fn validate_epoch_check_config(config: &RerandSweeperConfig) -> Result<()> {
    let tls = validated_tls(config)?;
    ampc_actor_utils::network::tcp::init_rustls_crypto_provider();
    TlsClient::new(TlsClientConfig::Mutual {
        root_certs: tls.root_certs,
        key_file: tls.private_key.expect("validated private key"),
        cert_file: tls.leaf_cert.expect("validated leaf certificate"),
    })
    .await?;
    Ok(())
}

async fn run_with_transport<S: EpochCheckSnapshot, T: BooleanMpcTransport>(
    party_id: usize,
    binding: &EpochCheckBinding,
    repetitions: usize,
    transport: &mut T,
    snapshot: Result<S>,
) -> Result<()> {
    ensure!(
        transport.party_id() == party_id,
        "epoch-check transport belongs to another party"
    );
    let (ready, inventory) = snapshot
        .as_ref()
        .map(|snapshot| (true, snapshot.inventory()))
        .unwrap_or((false, INVALID_INVENTORY));
    let agreement =
        agree_inventory_and_challenge(transport, binding, ready, inventory, repetitions).await;
    let mut snapshot = match snapshot {
        Ok(snapshot) => snapshot,
        Err(open_error) => {
            // Still complete the barrier first, so a local database failure
            // cannot leave healthy peers waiting forever for our packet.
            let _ = agreement;
            return Err(open_error);
        }
    };
    let challenge = agreement?;
    let scan = snapshot
        .accumulate(party_id, repetitions, challenge)
        .await
        .and_then(|accumulator| {
            ensure!(
                accumulator.rows() == inventory.rows,
                "epoch-check scan row count differs from its fixed inventory"
            );
            Ok(accumulator)
        });
    let finish = snapshot.finish().await;
    let local_failure = scan.is_err() || finish.is_err();
    if let Err(error) = &scan {
        tracing::error!(?error, "Persisted epoch-check scan failed locally");
    }
    if let Err(error) = &finish {
        tracing::error!(
            ?error,
            "Persisted epoch-check snapshot close failed locally"
        );
    }
    let checked_rows = scan.as_ref().map_or(0, CanaryAccumulator::rows);
    let mut syndrome = scan.map_or_else(
        |_| vec![0; repetitions * 4],
        CanaryAccumulator::into_syndrome_share,
    );
    syndrome.push(u16::from(local_failure));
    let nonzero = private_any_nonzero(syndrome, repetitions, challenge, transport).await?;
    ensure!(
        !nonzero,
        "PERSISTED EPOCH CHECK FAILED for epoch {} after checking {checked_rows} rows",
        binding.epoch
    );
    tracing::info!(
        epoch = binding.epoch,
        store_kind = %binding.store_kind,
        checked_rows,
        repetitions,
        "Persisted rerandomization epoch check passed"
    );
    Ok(())
}

/// Establish the dedicated three-party private network first, then open the
/// caller-provided repeatable-read snapshot and check its normalized rows.
/// Opening the snapshot only after all peers connect prevents an unavailable
/// peer from holding database resources on healthy parties.
async fn run_persisted_epoch_check<Open, OpenFuture, Snapshot>(
    config: &RerandSweeperConfig,
    binding: &EpochCheckBinding,
    open_snapshot: Open,
) -> Result<()>
where
    Open: FnOnce() -> OpenFuture,
    OpenFuture: Future<Output = Result<Snapshot>>,
    Snapshot: EpochCheckSnapshot,
{
    ensure!(binding.epoch > 0, "epoch check requires a positive epoch");
    ensure!(
        binding.environment == config.environment
            && binding.coordination_id == config.coordination_id
            && binding.store_kind == config.store_kind,
        "epoch-check binding differs from sweeper configuration"
    );
    let tls = validated_tls(config)?;
    let shutdown = CancellationToken::new();
    let mut network = build_network_handle(
        NetworkHandleArgs {
            party_index: config.party_id as usize,
            addresses: config.check_addresses.clone(),
            outbound_addresses: config.check_outbound_addresses.clone(),
            connection_parallelism: 1,
            request_parallelism: 1,
            sessions_per_request: 1,
            tls: Some(tls),
        },
        shutdown,
    )
    .await?;
    let mut channel = network.control_channel().await?;
    let mut transport = ControlChannelBooleanTransport {
        party_id: config.party_id as usize,
        channel: channel.as_mut(),
    };
    let snapshot = open_snapshot().await;
    run_with_transport(
        config.party_id as usize,
        binding,
        DEFAULT_CANARY_REPETITIONS,
        &mut transport,
        snapshot,
    )
    .await
}

/// Convenience entry point for the production store implementation.
pub(super) async fn run_store_persisted_epoch_check(
    config: &RerandSweeperConfig,
    binding: &EpochCheckBinding,
    store: &Store,
    rerand: &RerandContext,
) -> Result<()> {
    run_persisted_epoch_check(config, binding, || {
        open_store_epoch_check_snapshot(store, rerand, binding)
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use ampc_actor_utils::execution::local::LocalRuntime;
    use clap::Parser;
    use iris_mpc_common::consistency_canary::NetworkSessionBooleanTransport;
    use iris_mpc_common::{IRIS_CODE_LENGTH, MASK_CODE_LENGTH};
    use tokio::task::JoinSet;

    #[derive(Clone, Copy)]
    enum Fault {
        None,
        CorruptPartyOne,
        ScanPartyOne,
        InventoryPartyOne,
        OpenPartyOne,
    }

    struct FakeSnapshot {
        inventory: EpochCheckInventory,
        corrupt: bool,
        fail_scan: bool,
    }

    impl EpochCheckSnapshot for FakeSnapshot {
        fn inventory(&self) -> EpochCheckInventory {
            self.inventory
        }

        async fn accumulate(
            &mut self,
            party_id: usize,
            repetitions: usize,
            challenge: [u8; 32],
        ) -> Result<CanaryAccumulator> {
            ensure!(!self.fail_scan, "injected local scan failure");
            let mut left_code = vec![0; IRIS_CODE_LENGTH];
            if self.corrupt {
                left_code[41] = 1;
            }
            let mask = vec![0; MASK_CODE_LENGTH];
            let right_code = vec![0; IRIS_CODE_LENGTH];
            let mut accumulator = CanaryAccumulator::new(party_id, repetitions, challenge);
            accumulator.accumulate(1, &left_code, &mask, &right_code, &mask)?;
            Ok(accumulator)
        }

        async fn finish(self) -> Result<()> {
            Ok(())
        }
    }

    fn config() -> RerandSweeperConfig {
        RerandSweeperConfig::try_parse_from([
            "rerand-sweeper",
            "--party-id",
            "0",
            "--db-url",
            "postgres://rerand@example/db",
            "--schema-name",
            "SMPC_stage_0",
            "--store-id",
            "party-0-gpu",
            "--store-kind",
            "gpu",
            "--environment",
            "stage",
            "--coordination-id",
            "generation-7",
            "--s3-bucket",
            "rerand",
            "--expected-store-registry",
            "[]",
            "--check-addresses",
            "127.0.0.1:9000,127.0.0.1:9001,127.0.0.1:9002",
            "--check-outbound-addresses",
            "party-0:9000,party-1:9001,party-2:9002",
            "--rerand-check-tls-private-key",
            "/tls/key.pem",
            "--rerand-check-tls-leaf-cert",
            "/tls/cert.pem",
            "--rerand-check-tls-root-certs",
            "/tls/root-a.pem,/tls/root-b.pem",
        ])
        .unwrap()
    }

    fn binding() -> EpochCheckBinding {
        EpochCheckBinding {
            environment: "stage".to_owned(),
            coordination_id: "generation-7".to_owned(),
            store_kind: RerandStoreKind::Gpu,
            store_registry_commitment: [8; 32],
            epoch: 4,
            seed_commitment: [9; 32],
        }
    }

    async fn run_fake_check(fault: Fault) -> Vec<Result<()>> {
        let sessions = LocalRuntime::mock_sessions_with_channel().await.unwrap();
        let mut tasks = JoinSet::new();
        for (party_id, session) in sessions.into_iter().enumerate() {
            let binding = binding();
            tasks.spawn(async move {
                let inventory = EpochCheckInventory {
                    rows: 1,
                    digest: if party_id == 1 && matches!(fault, Fault::InventoryPartyOne) {
                        [8; 32]
                    } else {
                        [7; 32]
                    },
                };
                let snapshot = if party_id == 1 && matches!(fault, Fault::OpenPartyOne) {
                    Err(eyre!("injected snapshot-open failure"))
                } else {
                    Ok(FakeSnapshot {
                        inventory,
                        corrupt: party_id == 1 && matches!(fault, Fault::CorruptPartyOne),
                        fail_scan: party_id == 1 && matches!(fault, Fault::ScanPartyOne),
                    })
                };
                let mut session = session.lock().await;
                let mut transport =
                    NetworkSessionBooleanTransport::new(party_id, &mut session.network_session);
                run_with_transport(
                    party_id,
                    &binding,
                    DEFAULT_CANARY_REPETITIONS,
                    &mut transport,
                    snapshot,
                )
                .await
            });
        }

        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            let mut results = Vec::new();
            while let Some(result) = tasks.join_next().await {
                results.push(result.unwrap());
            }
            results
        })
        .await
        .expect("epoch-check peers blocked")
    }

    #[test]
    fn context_binds_epoch_store_inventory_and_deployment() {
        let inventory = EpochCheckInventory {
            rows: 10,
            digest: [3; 32],
        };
        let baseline = epoch_check_context(&binding(), inventory, 12);
        let mut changed = binding();
        changed.epoch += 1;
        assert_ne!(baseline, epoch_check_context(&changed, inventory, 12));
        changed = binding();
        changed.store_kind = RerandStoreKind::Hnsw;
        assert_ne!(baseline, epoch_check_context(&changed, inventory, 12));
        changed = binding();
        changed.coordination_id.push_str("-replacement");
        assert_ne!(baseline, epoch_check_context(&changed, inventory, 12));
        assert_ne!(
            baseline,
            epoch_check_context(
                &binding(),
                EpochCheckInventory {
                    rows: 11,
                    ..inventory
                },
                12
            )
        );
    }

    #[test]
    fn barrier_round_trip_rejects_wrong_magic_and_non_bit_ready() {
        let packet = BarrierPacket {
            ready: true,
            context: [1; 32],
            inventory: EpochCheckInventory {
                rows: 8,
                digest: [2; 32],
            },
            contribution: [3; 32],
        };
        assert_eq!(
            BarrierPacket::decode(NetworkValue::Bytes(packet.encode())).unwrap(),
            packet
        );
        let mut malformed = packet.encode();
        malformed[0] ^= 1;
        assert!(BarrierPacket::decode(NetworkValue::Bytes(malformed)).is_err());
        let mut malformed = packet.encode();
        malformed[BARRIER_MAGIC.len()] = 2;
        assert!(BarrierPacket::decode(NetworkValue::Bytes(malformed)).is_err());
    }

    #[test]
    fn production_network_config_requires_complete_mutual_tls() {
        let mut config = config();
        assert_eq!(config.check_addresses.len(), 3);
        assert_eq!(config.check_outbound_addresses.len(), 3);
        validated_tls(&config).unwrap();
        config.check_tls.private_key = None;
        assert!(validated_tls(&config).is_err());
        config.environment = "testing".to_owned();
        config.check_tls = Default::default();
        assert!(validated_tls(&config).is_err());
    }

    #[tokio::test]
    async fn three_party_check_accepts_valid_one_row_and_rejects_failures() {
        assert!(run_fake_check(Fault::None)
            .await
            .into_iter()
            .all(|result| result.is_ok()));
        for fault in [
            Fault::CorruptPartyOne,
            Fault::ScanPartyOne,
            Fault::InventoryPartyOne,
            Fault::OpenPartyOne,
        ] {
            assert!(run_fake_check(fault)
                .await
                .into_iter()
                .all(|result| result.is_err()));
        }
    }
}
