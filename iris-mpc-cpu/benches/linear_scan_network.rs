use aes_prng::AesRng;
use ampc_actor_utils::network::mpc::{NetworkType, NetworkValue, Networking};
use async_trait::async_trait;
use eyre::{ensure, Result};
use iris_mpc_common::{iris_db::iris::IrisCode, VectorId};
use iris_mpc_cpu::{
    execution::{
        hawk_main::iris_worker::{IrisWorkerPool, LocalIrisWorkerPool},
        local::LocalRuntime,
    },
    hawkers::aby3::aby3_store::{Aby3Store, DistanceMode, FhdOps},
    protocol::shared_iris::{ArcIris, GaloisRingSharedIris},
};
use rand::SeedableRng;
use std::{
    collections::HashMap,
    env,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

const DEFAULT_COMPARISONS: usize = 4096;
const TCP_SESSION_ID_BYTES: u64 = 4;

#[derive(Debug, Default)]
struct SendCounter {
    sent_payload_bytes: AtomicU64,
    sent_messages: AtomicU64,
    received_payload_bytes: AtomicU64,
    received_messages: AtomicU64,
}

#[derive(Clone, Copy, Debug, Default)]
struct SendSnapshot {
    sent_payload_bytes: u64,
    sent_messages: u64,
    received_payload_bytes: u64,
    received_messages: u64,
}

impl SendCounter {
    fn snapshot(&self) -> SendSnapshot {
        SendSnapshot {
            sent_payload_bytes: self.sent_payload_bytes.load(Ordering::Relaxed),
            sent_messages: self.sent_messages.load(Ordering::Relaxed),
            received_payload_bytes: self.received_payload_bytes.load(Ordering::Relaxed),
            received_messages: self.received_messages.load(Ordering::Relaxed),
        }
    }
}

impl std::ops::Sub for SendSnapshot {
    type Output = SendSnapshot;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            sent_payload_bytes: self.sent_payload_bytes - rhs.sent_payload_bytes,
            sent_messages: self.sent_messages - rhs.sent_messages,
            received_payload_bytes: self.received_payload_bytes - rhs.received_payload_bytes,
            received_messages: self.received_messages - rhs.received_messages,
        }
    }
}

impl std::ops::Add for SendSnapshot {
    type Output = SendSnapshot;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            sent_payload_bytes: self.sent_payload_bytes + rhs.sent_payload_bytes,
            sent_messages: self.sent_messages + rhs.sent_messages,
            received_payload_bytes: self.received_payload_bytes + rhs.received_payload_bytes,
            received_messages: self.received_messages + rhs.received_messages,
        }
    }
}

impl SendSnapshot {
    fn sent_framed_bytes(self) -> u64 {
        self.sent_payload_bytes + TCP_SESSION_ID_BYTES * self.sent_messages
    }

    fn received_framed_bytes(self) -> u64 {
        self.received_payload_bytes + TCP_SESSION_ID_BYTES * self.received_messages
    }
}

struct CountingNetworking {
    inner: Box<dyn Networking + Send + Sync>,
    counter: Arc<SendCounter>,
}

#[async_trait]
impl Networking for CountingNetworking {
    async fn send(
        &mut self,
        value: NetworkValue,
        receiver: &ampc_actor_utils::execution::player::Identity,
    ) -> Result<()> {
        self.counter
            .sent_payload_bytes
            .fetch_add(value.byte_len() as u64, Ordering::Relaxed);
        self.counter.sent_messages.fetch_add(1, Ordering::Relaxed);
        self.inner.send(value, receiver).await
    }

    async fn receive(
        &mut self,
        sender: &ampc_actor_utils::execution::player::Identity,
    ) -> Result<NetworkValue> {
        let value = self.inner.receive(sender).await?;
        self.counter
            .received_payload_bytes
            .fetch_add(value.byte_len() as u64, Ordering::Relaxed);
        self.counter
            .received_messages
            .fetch_add(1, Ordering::Relaxed);
        Ok(value)
    }
}

struct UnusedNetworking;

#[async_trait]
impl Networking for UnusedNetworking {
    async fn send(
        &mut self,
        _value: NetworkValue,
        _receiver: &ampc_actor_utils::execution::player::Identity,
    ) -> Result<()> {
        unreachable!("placeholder networking must never be used")
    }

    async fn receive(
        &mut self,
        _sender: &ampc_actor_utils::execution::player::Identity,
    ) -> Result<NetworkValue> {
        unreachable!("placeholder networking must never be used")
    }
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    match env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(err) => Err(err.into()),
    }
}

fn snapshots(counters: &[Arc<SendCounter>]) -> Vec<SendSnapshot> {
    counters.iter().map(|counter| counter.snapshot()).collect()
}

fn deltas(before: &[SendSnapshot], after: &[SendSnapshot]) -> Vec<SendSnapshot> {
    after
        .iter()
        .zip(before)
        .map(|(&after, &before)| after - before)
        .collect()
}

fn print_stage(stage: &str, comparisons: usize, stats: &[SendSnapshot]) {
    for (party, stat) in stats.iter().copied().enumerate() {
        println!(
            "NETWORK_RESULT stage={stage} party={party} comparisons={comparisons} \
             sent_payload_bytes={} sent_messages={} sent_framed_bytes={} \
             sent_framed_bytes_per_comparison={:.6} received_payload_bytes={} \
             received_messages={} received_framed_bytes={} \
             received_framed_bytes_per_comparison={:.6}",
            stat.sent_payload_bytes,
            stat.sent_messages,
            stat.sent_framed_bytes(),
            stat.sent_framed_bytes() as f64 / comparisons as f64,
            stat.received_payload_bytes,
            stat.received_messages,
            stat.received_framed_bytes(),
            stat.received_framed_bytes() as f64 / comparisons as f64,
        );
    }
    let max_sent_framed = stats
        .iter()
        .copied()
        .map(SendSnapshot::sent_framed_bytes)
        .max()
        .unwrap_or_default();
    let max_received_framed = stats
        .iter()
        .copied()
        .map(SendSnapshot::received_framed_bytes)
        .max()
        .unwrap_or_default();
    let aggregate_sent_framed = stats
        .iter()
        .copied()
        .map(SendSnapshot::sent_framed_bytes)
        .sum::<u64>();
    let aggregate_received_framed = stats
        .iter()
        .copied()
        .map(SendSnapshot::received_framed_bytes)
        .sum::<u64>();
    assert_eq!(
        aggregate_sent_framed, aggregate_received_framed,
        "all sent protocol bytes must be received within the measured stage"
    );
    println!(
        "NETWORK_SUMMARY stage={stage} comparisons={comparisons} \
         max_instance_sent_framed_bytes_per_comparison={:.6} \
         max_instance_received_framed_bytes_per_comparison={:.6} \
         aggregate_sent_framed_bytes_per_comparison={:.6}",
        max_sent_framed as f64 / comparisons as f64,
        max_received_framed as f64 / comparisons as f64,
        aggregate_sent_framed as f64 / comparisons as f64,
    );
}

async fn setup(
    comparisons: usize,
) -> Result<(
    Vec<Aby3Store<FhdOps>>,
    Vec<iris_mpc_cpu::hawkers::aby3::aby3_store::Aby3Query>,
    Vec<VectorId>,
    Vec<Arc<SendCounter>>,
)> {
    let mut runtime = LocalRuntime::mock_setup(NetworkType::Local).await?;
    let counters = (0..3)
        .map(|_| Arc::new(SendCounter::default()))
        .collect::<Vec<_>>();

    // PRF setup is complete before counters are installed, so only the scan
    // protocol itself is measured.
    for (session, counter) in runtime.sessions.iter_mut().zip(&counters) {
        let inner = std::mem::replace(
            &mut session.network_session.networking,
            Box::new(UnusedNetworking),
        );
        session.network_session.networking = Box::new(CountingNetworking {
            inner,
            counter: counter.clone(),
        });
    }

    let mut rng = AesRng::seed_from_u64(0x006e_6574_776f_726b_u64);
    let iris = IrisCode::random_rng(&mut rng);
    let shares = GaloisRingSharedIris::generate_shares_locally(&mut rng, iris);
    let vector_ids = (0..comparisons)
        .map(|index| VectorId::from_0_index(index as u32))
        .collect::<Vec<_>>();

    let mut stores = Vec::with_capacity(3);
    for (party, session) in runtime.sessions.into_iter().enumerate() {
        let points = vector_ids
            .iter()
            .copied()
            .map(|id| (id, Arc::new(shares[party].clone()) as ArcIris))
            .collect::<HashMap<_, _>>();
        let storage = Aby3Store::<FhdOps>::new_storage(Some(points)).to_arc();
        let workers: Arc<dyn IrisWorkerPool> = Arc::new(LocalIrisWorkerPool::new_local(
            storage.clone(),
            DistanceMode::MinRotation,
            party,
        ));
        let registry = storage.read().await.to_registry().to_arc();
        stores.push(Aby3Store::new(
            registry,
            session,
            workers,
            DistanceMode::MinRotation,
        ));
    }

    let query_id = vector_ids[0];
    let mut queries = Vec::with_capacity(3);
    for store in &stores {
        queries.push(store.cache_query_from_store(&query_id).await?);
    }
    Ok((stores, queries, vector_ids, counters))
}

#[tokio::main]
async fn main() -> Result<()> {
    let comparisons = env_usize("IRIS_MPC_NETWORK_BENCH_COMPARISONS", DEFAULT_COMPARISONS)?;
    ensure!(comparisons > 0, "comparison count must be positive");
    ensure!(
        comparisons <= u32::MAX as usize,
        "comparison count exceeds VectorId range"
    );
    println!(
        "NETWORK_CONFIG comparisons={comparisons} rotations=31 parties=3 \
         framing=tcp_session_id_4_bytes excludes=tcp_ip_tls_headers_and_prf_setup"
    );

    let (mut stores, queries, vector_ids, counters) = setup(comparisons).await?;

    let before = snapshots(&counters);
    let results =
        futures::future::try_join_all(stores.iter_mut().zip(&queries).map(|(store, query)| {
            store.eval_distance_batch_full_rotation_thresholds(query, &vector_ids)
        }))
        .await?;
    ensure!(
        results
            .iter()
            .all(|result| result.matches.len() == comparisons),
        "match result count mismatch"
    );
    ensure!(
        results
            .iter()
            .all(|result| result.matches.iter().all(Option::is_some)),
        "identical records must all match"
    );
    ensure!(
        results
            .iter()
            .all(|result| result.match_rotations.len() == comparisons),
        "rotation result count mismatch"
    );
    let traffic = deltas(&before, &snapshots(&counters));
    print_stage("gpu_equivalent_threshold_scan", comparisons, &traffic);

    Ok(())
}
