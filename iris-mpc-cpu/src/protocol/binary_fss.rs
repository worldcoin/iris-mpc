// same imports as in binary.rs

use crate::{
    execution::{
        player::Role,
        session::{LaneId, NetworkSession, Session, SessionHandles},
    },
    network::value::{NetworkInt, NetworkValue},
    shares::{
        bit::Bit,
        int_ring::IntRing2k,
        ring_impl::RingElement,
        share::Share,
        vecshare::{SliceShare, VecShare},
    },
};
use std::io::Write;

use aes_prng::AesRng;
use ark_std::{end_timer, start_timer};
use eyre::{bail, eyre, Error, Result};
use iris_mpc_common::fast_metrics::FastHistogram;
use itertools::{izip, Itertools};
use num_traits::{One, Zero};
use rand::prelude::*;
use rand::{distributions::Standard, prelude::Distribution, Rng};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    OnceLock,
};
use std::{
    cell::RefCell,
    collections::VecDeque,
    ops::SubAssign,
    sync::{Arc, Mutex},
};
use tokio::{sync::mpsc, task::JoinHandle};
use tracing::{info, instrument, trace_span, Instrument};

use fss_rs::icf::{IcShare, Icf, InG, IntvFn, OutG};
use fss_rs::prg::Aes128MatyasMeyerOseasPrg;

#[derive(Clone)]
pub struct BinaryFssContext {
    role: Role,
    network_session: NetworkSession,
}

impl BinaryFssContext {
    pub fn new(session: &Session) -> Self {
        Self {
            role: session.own_role(),
            network_session: session.network_session.clone(),
        }
    }

    pub fn own_role(&self) -> Role {
        self.role
    }

    pub fn network_session(&self) -> NetworkSession {
        self.network_session.clone()
    }

    pub fn lane_count(&self) -> usize {
        self.network_session.lane_count()
    }
}

macro_rules! maybe_perf_scoped_for_party {
    ($enabled:expr, $base_label:literal, $party:expr, $n:expr, $bucket:expr) => {{
        if $enabled {
            Some($crate::perf_scoped_for_party!(
                $base_label,
                $party,
                $n,
                $bucket
            ))
        } else {
            None
        }
    }};
}

#[inline]
fn bits_to_network_vec(bits: &[RingElement<Bit>]) -> Vec<NetworkValue> {
    bits.iter()
        .copied()
        .map(NetworkValue::RingElementBit)
        .collect()
}

#[inline]
fn encode_bit_chunk(start: usize, bits: &[RingElement<Bit>]) -> NetworkValue {
    let mut payload = Vec::with_capacity(bits.len() + 2);
    payload.push(NetworkValue::RingElement32(RingElement(start as u32)));
    payload.push(NetworkValue::RingElement32(RingElement(bits.len() as u32)));
    payload.extend(bits_to_network_vec(bits));
    NetworkValue::vec_to_network(payload)
}

fn decode_bit_chunk(value: NetworkValue) -> Result<(usize, Vec<RingElement<Bit>>), Error> {
    let mut entries = NetworkValue::vec_from_network(value)?;
    if entries.len() < 2 {
        bail!("invalid bit chunk payload len {}", entries.len());
    }

    let len_entry = entries
        .get(1)
        .ok_or_else(|| eyre!("missing len entry"))?
        .clone();
    let start_entry = entries
        .get(0)
        .ok_or_else(|| eyre!("missing start entry"))?
        .clone();
    let bits_entries = entries.split_off(2);

    let start = match start_entry {
        NetworkValue::RingElement32(val) => val.convert() as usize,
        other => bail!("expected chunk start, got {:?}", other),
    };
    let expected_len = match len_entry {
        NetworkValue::RingElement32(val) => val.convert() as usize,
        other => bail!("expected chunk len, got {:?}", other),
    };
    let bits = network_vec_to_bits(bits_entries).map_err(|e| {
        eyre!(
            "cannot decode chunk bits for chunk starting at {}: {e}",
            start
        )
    })?;
    if bits.len() != expected_len {
        bail!(
            "chunk starting at {} expected len {}, got {}",
            start,
            expected_len,
            bits.len()
        );
    }
    Ok((start, bits))
}

#[inline]
fn network_vec_to_bits(values: Vec<NetworkValue>) -> Result<Vec<RingElement<Bit>>, Error> {
    values
        .into_iter()
        .map(|nv| match nv {
            NetworkValue::RingElementBit(bit) => Ok(bit),
            other => Err(eyre!("expected RingElementBit, got {:?}", other)),
        })
        .collect()
}

#[inline]
fn decode_key_package(
    pkg: NetworkValue,
) -> Result<(Vec<RingElement<u32>>, Vec<RingElement<u32>>), Error> {
    let mut entries = NetworkValue::vec_from_network(pkg)?;
    if entries.len() != 2 {
        bail!("invalid key package length {}", entries.len());
    }
    let blob = entries
        .pop()
        .ok_or_else(|| eyre!("missing key blob in package"))?;
    let lens = entries
        .pop()
        .ok_or_else(|| eyre!("missing key lengths in package"))?;

    let lens = match lens {
        NetworkValue::VecRing32(v) => Ok(v),
        other => Err(eyre!("expected VecRing32 for lens, got {:?}", other)),
    }?;
    let blob = match blob {
        NetworkValue::VecRing32(v) => Ok(v),
        other => Err(eyre!("expected VecRing32 for blob, got {:?}", other)),
    }?;
    Ok((lens, blob))
}

#[inline]
fn encode_key_package(lens: Vec<RingElement<u32>>, blob: Vec<RingElement<u32>>) -> NetworkValue {
    NetworkValue::vec_to_network(vec![
        NetworkValue::VecRing32(lens),
        NetworkValue::VecRing32(blob),
    ])
}

struct EvaluatorKeyState {
    my_keys: Vec<IcShare>,
    lens_usize: Vec<usize>,
    offs: Vec<usize>,
    key_blob_words: Vec<u32>,
}

impl EvaluatorKeyState {
    fn new(lens_words: Vec<u32>, key_blob_words: Vec<u32>) -> Result<Self, Error> {
        let mut offs = Vec::with_capacity(lens_words.len());
        let mut my_keys = Vec::with_capacity(lens_words.len());
        let mut lens_usize = Vec::with_capacity(lens_words.len());

        let mut cursor = 0usize;
        for &len_u32 in &lens_words {
            let len = len_u32 as usize;
            lens_usize.push(len);
            offs.push(cursor);

            let key = IcShare::deserialize(&key_blob_words[cursor..cursor + len])
                .map_err(|e| eyre!("failed to deserialize ICF key: {e}"))?;
            my_keys.push(key);
            cursor += len;
        }

        Ok(Self {
            my_keys,
            lens_usize,
            offs,
            key_blob_words,
        })
    }

    fn key_slice(&self, idx: usize) -> &[u32] {
        let start = self.offs[idx];
        let len = self.lens_usize[idx];
        &self.key_blob_words[start..start + len]
    }

    fn key(&self, idx: usize) -> &IcShare {
        &self.my_keys[idx]
    }
}

enum KeySource {
    Prev,
    Next,
}

const FSS_PIPELINE_MIN_CHUNK: usize = 128;
const FSS_PIPELINE_MAX_CHUNK: usize = 4096;
const FSS_PIPELINE_WIDTH: usize = 3;
const FSS_MAX_PENDING_RECONS: usize = 2;

#[derive(Default)]
struct EvalTrafficStats {
    key_recv_bytes: usize,
    recon_send_bytes: usize,
    recon_recv_bytes: usize,
    post_send_prev_bytes: usize,
    post_send_next_bytes: usize,
    post_recv_bytes: usize,
}

impl EvalTrafficStats {
    fn total(&self) -> usize {
        self.key_recv_bytes
            + self.recon_send_bytes
            + self.recon_recv_bytes
            + self.post_send_prev_bytes
            + self.post_send_next_bytes
            + self.post_recv_bytes
    }

    fn send_total(&self) -> usize {
        self.recon_send_bytes + self.post_send_prev_bytes + self.post_send_next_bytes
    }

    fn recv_total(&self) -> usize {
        self.key_recv_bytes + self.recon_recv_bytes + self.post_recv_bytes
    }
}

#[derive(Default)]
struct DealerTrafficStats {
    key_send_prev_bytes: usize,
    key_send_next_bytes: usize,
    share_recv_from_p0_bytes: usize,
    share_recv_from_p1_bytes: usize,
}

impl DealerTrafficStats {
    fn total(&self) -> usize {
        self.key_send_prev_bytes
            + self.key_send_next_bytes
            + self.share_recv_from_p0_bytes
            + self.share_recv_from_p1_bytes
    }

    fn send_total(&self) -> usize {
        self.key_send_prev_bytes + self.key_send_next_bytes
    }

    fn recv_total(&self) -> usize {
        self.share_recv_from_p0_bytes + self.share_recv_from_p1_bytes
    }
}

#[derive(Default)]
struct DealerCollectorStats {
    from_prev_bytes: usize,
    from_next_bytes: usize,
}

fn bytes_to_mb(bytes: usize) -> f64 {
    bytes_to_mb_u64(bytes as u64)
}

fn bytes_to_mb_u64(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

const TRAFFIC_DETAIL_DEFAULT_ENABLED: bool = false;
const TRAFFIC_TOTAL_DEFAULT_ENABLED: bool = false;
static TRAFFIC_DETAIL_ENABLED: OnceLock<bool> = OnceLock::new();
static TRAFFIC_TOTAL_ENABLED: OnceLock<bool> = OnceLock::new();
static TRAFFIC_TOTALS: OnceLock<TrafficTotals> = OnceLock::new();
static KEY_PACKAGE_BYTES: OnceLock<KeyPackageBytes> = OnceLock::new();
static LAST_EVAL_SNAPSHOTS: OnceLock<Mutex<[EvalTrafficSnapshot; 2]>> = OnceLock::new();
static LAST_DEALER_SNAPSHOT: OnceLock<Mutex<DealerTrafficSnapshot>> = OnceLock::new();

struct KeyPackageBytes {
    prev: AtomicU64,
    next: AtomicU64,
}

impl KeyPackageBytes {
    const fn new() -> Self {
        Self {
            prev: AtomicU64::new(0),
            next: AtomicU64::new(0),
        }
    }

    fn store(&self, prev: u64, next: u64) {
        self.prev.store(prev, Ordering::Relaxed);
        self.next.store(next, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, Default)]
pub struct EvalTrafficSnapshot {
    pub key_recv: u64,
    pub recon_send: u64,
    pub recon_recv: u64,
    pub post_send_prev: u64,
    pub post_send_next: u64,
    pub post_recv: u64,
}

impl EvalTrafficSnapshot {
    pub fn total_send(&self) -> u64 {
        self.recon_send + self.post_send_prev + self.post_send_next
    }

    pub fn total_recv(&self) -> u64 {
        self.key_recv + self.recon_recv + self.post_recv
    }
}

#[derive(Clone, Copy, Default)]
pub struct DealerTrafficSnapshot {
    pub key_send_prev: u64,
    pub key_send_next: u64,
    pub recv_from_p0: u64,
    pub recv_from_p1: u64,
}

impl DealerTrafficSnapshot {
    pub fn total_send(&self) -> u64 {
        self.key_send_prev + self.key_send_next
    }

    pub fn total_recv(&self) -> u64 {
        self.recv_from_p0 + self.recv_from_p1
    }
}

struct TrafficTotals {
    send: [AtomicU64; 3],
    recv: [AtomicU64; 3],
}

impl TrafficTotals {
    const fn new() -> Self {
        Self {
            send: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
            recv: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
        }
    }
}

fn env_flag(var: &str, default: bool) -> bool {
    std::env::var(var)
        .ok()
        .map(|val| {
            let lower = val.to_ascii_lowercase();
            matches!(
                lower.as_str(),
                "1" | "true" | "yes" | "y" | "on" | "enable" | "enabled"
            )
        })
        .unwrap_or(default)
}

fn traffic_detail_enabled() -> bool {
    *TRAFFIC_DETAIL_ENABLED.get_or_init(|| {
        if std::env::var("IRIS_FSS_TRAFFIC").is_ok() {
            env_flag("IRIS_FSS_TRAFFIC", TRAFFIC_DETAIL_DEFAULT_ENABLED)
        } else {
            env_flag("IRIS_FSS_TRAFFIC_DETAIL", TRAFFIC_DETAIL_DEFAULT_ENABLED)
        }
    })
}

fn traffic_total_enabled() -> bool {
    *TRAFFIC_TOTAL_ENABLED
        .get_or_init(|| env_flag("IRIS_FSS_TRAFFIC_TOTAL", TRAFFIC_TOTAL_DEFAULT_ENABLED))
}

fn traffic_totals() -> &'static TrafficTotals {
    TRAFFIC_TOTALS.get_or_init(TrafficTotals::new)
}

fn record_traffic_totals(role: usize, send_bytes: usize, recv_bytes: usize) {
    if role >= 3 {
        return;
    }
    let totals = traffic_totals();
    let send_total =
        totals.send[role].fetch_add(send_bytes as u64, Ordering::Relaxed) + send_bytes as u64;
    let recv_total =
        totals.recv[role].fetch_add(recv_bytes as u64, Ordering::Relaxed) + recv_bytes as u64;
    if traffic_total_enabled() {
        log_total_stats(role, send_total, recv_total, send_bytes, recv_bytes);
    }
}

pub fn traffic_totals_per_party() -> Option<[u64; 3]> {
    TRAFFIC_TOTALS.get().map(|totals| {
        let mut out = [0u64; 3];
        for role in 0..3 {
            let send = totals.send[role].load(Ordering::Relaxed);
            let recv = totals.recv[role].load(Ordering::Relaxed);
            out[role] = send + recv;
        }
        out
    })
}

pub fn format_traffic_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * KB;
    const GB: f64 = 1024.0 * MB;
    const TB: f64 = 1024.0 * GB;
    let bytes_f = bytes as f64;
    if bytes_f >= TB {
        format!("{:.0} TB", bytes_f / TB)
    } else if bytes_f >= GB {
        format!("{:.0} GB", bytes_f / GB)
    } else if bytes_f >= MB {
        format!("{:.0} MB", bytes_f / MB)
    } else if bytes_f >= KB {
        format!("{:.0} KB", bytes_f / KB)
    } else {
        format!("{} B", bytes)
    }
}

fn record_key_package_bytes(prev_bytes: usize, next_bytes: usize) {
    let stats = KEY_PACKAGE_BYTES.get_or_init(|| KeyPackageBytes::new());
    stats.store(prev_bytes as u64, next_bytes as u64);
}

pub fn fss_key_package_bytes() -> Option<(u64, u64)> {
    KEY_PACKAGE_BYTES.get().map(|stats| {
        (
            stats.prev.load(Ordering::Relaxed),
            stats.next.load(Ordering::Relaxed),
        )
    })
}

fn eval_snapshots_store() -> &'static Mutex<[EvalTrafficSnapshot; 2]> {
    LAST_EVAL_SNAPSHOTS.get_or_init(|| Mutex::new([EvalTrafficSnapshot::default(); 2]))
}

fn dealer_snapshot_store() -> &'static Mutex<DealerTrafficSnapshot> {
    LAST_DEALER_SNAPSHOT.get_or_init(|| Mutex::new(DealerTrafficSnapshot::default()))
}

fn store_eval_snapshot(role: usize, stats: &EvalTrafficStats) {
    if role > 1 {
        return;
    }
    let mut guard = eval_snapshots_store().lock().unwrap();
    guard[role].key_recv += stats.key_recv_bytes as u64;
    guard[role].recon_send += stats.recon_send_bytes as u64;
    guard[role].recon_recv += stats.recon_recv_bytes as u64;
    guard[role].post_send_prev += stats.post_send_prev_bytes as u64;
    guard[role].post_send_next += stats.post_send_next_bytes as u64;
    guard[role].post_recv += stats.post_recv_bytes as u64;
}

fn store_dealer_snapshot(stats: &DealerTrafficStats) {
    let mut guard = dealer_snapshot_store().lock().unwrap();
    guard.key_send_prev += stats.key_send_prev_bytes as u64;
    guard.key_send_next += stats.key_send_next_bytes as u64;
    guard.recv_from_p0 += stats.share_recv_from_p0_bytes as u64;
    guard.recv_from_p1 += stats.share_recv_from_p1_bytes as u64;
}

pub fn eval_traffic_breakdown() -> Option<[EvalTrafficSnapshot; 2]> {
    LAST_EVAL_SNAPSHOTS
        .get()
        .map(|mutex| *mutex.lock().unwrap())
}

pub fn dealer_traffic_breakdown() -> Option<DealerTrafficSnapshot> {
    LAST_DEALER_SNAPSHOT
        .get()
        .map(|mutex| *mutex.lock().unwrap())
}

fn log_eval_stats(role: usize, label: &str, stats: &EvalTrafficStats) {
    let send_mb = bytes_to_mb(stats.send_total());
    let recv_mb = bytes_to_mb(stats.recv_total());
    let total_mb = bytes_to_mb(stats.total());
    info!(
        "FSS traffic role {} ({}): send={:.3}MB recv={:.3}MB total={:.3}MB | key_recv={:.3}MB recon_send={:.3}MB recon_recv={:.3}MB post_send_prev={:.3}MB post_send_next={:.3}MB post_recv={:.3}MB",
        role,
        label,
        send_mb,
        recv_mb,
        total_mb,
        bytes_to_mb(stats.key_recv_bytes),
        bytes_to_mb(stats.recon_send_bytes),
        bytes_to_mb(stats.recon_recv_bytes),
        bytes_to_mb(stats.post_send_prev_bytes),
        bytes_to_mb(stats.post_send_next_bytes),
        bytes_to_mb(stats.post_recv_bytes),
    );
}

fn log_total_stats(
    role: usize,
    send_total: u64,
    recv_total: u64,
    delta_send: usize,
    delta_recv: usize,
) {
    info!(
        "FSS cumulative traffic role {}: send={:.3}MB recv={:.3}MB total={:.3}MB (delta send={:.3}MB delta recv={:.3}MB)",
        role,
        bytes_to_mb_u64(send_total),
        bytes_to_mb_u64(recv_total),
        bytes_to_mb_u64(send_total + recv_total),
        bytes_to_mb(delta_send),
        bytes_to_mb(delta_recv),
    );
}

fn log_dealer_stats(role: usize, stats: &DealerTrafficStats) {
    let send_mb = bytes_to_mb(stats.send_total());
    let recv_mb = bytes_to_mb(stats.recv_total());
    let total_mb = bytes_to_mb(stats.total());
    info!(
        "FSS traffic role {} (dealer): send={:.3}MB recv={:.3}MB total={:.3}MB | key_send_prev={:.3}MB key_send_next={:.3}MB recv_from_p0={:.3}MB recv_from_p1={:.3}MB",
        role,
        send_mb,
        recv_mb,
        total_mb,
        bytes_to_mb(stats.key_send_prev_bytes),
        bytes_to_mb(stats.key_send_next_bytes),
        bytes_to_mb(stats.share_recv_from_p0_bytes),
        bytes_to_mb(stats.share_recv_from_p1_bytes),
    );
}

fn chunk_ranges(len: usize, chunk_size: usize) -> Vec<(usize, usize)> {
    if len == 0 {
        return Vec::new();
    }
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < len {
        let end = (start + chunk_size).min(len);
        ranges.push((start, end));
        start = end;
    }
    ranges
}

#[derive(Clone)]
struct ChunkPlan {
    ranges: Arc<Vec<(usize, usize)>>,
}

impl ChunkPlan {
    fn new(total_len: usize, parallel_threshold: usize) -> Self {
        let chunk_size = compute_chunk_size(total_len, parallel_threshold);
        let ranges = chunk_ranges(total_len, chunk_size);
        Self {
            ranges: Arc::new(ranges),
        }
    }

    fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    fn ranges(&self) -> &[(usize, usize)] {
        &self.ranges
    }

    fn ranges_arc(&self) -> Arc<Vec<(usize, usize)>> {
        Arc::clone(&self.ranges)
    }

    fn len(&self) -> usize {
        self.ranges.len()
    }

    fn total_bits(&self) -> usize {
        self.ranges.last().map(|(_, end)| *end).unwrap_or(0)
    }
}

fn compute_chunk_size(total: usize, parallel_threshold: usize) -> usize {
    if total == 0 {
        return 0;
    }
    let width = FSS_PIPELINE_WIDTH.max(1);
    let per_width = (total + width - 1) / width;
    let upper_capped = per_width.min(FSS_PIPELINE_MAX_CHUNK);
    let base = parallel_threshold.max(FSS_PIPELINE_MIN_CHUNK).max(1);
    upper_capped.max(base)
}

struct ReconTask {
    start: usize,
    end: usize,
    handle: JoinHandle<Result<(Vec<RingElement<u32>>, usize, usize), Error>>,
}

fn spawn_dealer_stream_reader<const TIMED: bool>(
    ctx: BinaryFssContext,
    role: usize,
    ranges: Arc<Vec<(usize, usize)>>,
    bucket_bound: usize,
    from_prev: bool,
) -> (
    mpsc::Receiver<(usize, Vec<RingElement<Bit>>)>,
    JoinHandle<Result<usize, Error>>,
) {
    let (tx, rx) = mpsc::channel(ranges.len());
    let lane_count = ctx.lane_count().max(1);
    let mut lane_indices: Vec<Vec<usize>> = vec![Vec::new(); lane_count];
    for idx in 0..ranges.len() {
        lane_indices[idx % lane_count].push(idx);
    }

    let mut handles = Vec::new();
    for (lane, indices) in lane_indices.into_iter().enumerate() {
        if indices.is_empty() {
            continue;
        }
        let ctx_clone = ctx.clone();
        let ranges_clone = ranges.clone();
        let tx_clone = tx.clone();
        handles.push(tokio::spawn(async move {
            let network = ctx_clone.network_session();
            let lane_id = LaneId(lane);
            let mut lane_bytes = 0usize;
            for idx in indices {
                let (start, end) = ranges_clone[idx];
                let chunk_bits_count = end - start;
                let _tt_net = if from_prev {
                    maybe_perf_scoped_for_party!(
                        TIMED,
                        "fss.network.dealer.recv_P1",
                        role,
                        chunk_bits_count,
                        bucket_bound
                    )
                } else {
                    maybe_perf_scoped_for_party!(
                        TIMED,
                        "fss.network.dealer.recv_P0",
                        role,
                        chunk_bits_count,
                        bucket_bound
                    )
                };
                let msg_res = if from_prev {
                    network.receive_prev_on_lane(lane_id).await
                } else {
                    network.receive_next_on_lane(lane_id).await
                };
                drop(_tt_net);

                let msg = match msg_res {
                    Ok(m) => m,
                    Err(e) => {
                        return Err(eyre!(
                            "Dealer cannot receive bit chunk from {} on lane {}: {e}",
                            if from_prev { "P1" } else { "P0" },
                            lane
                        ));
                    }
                };
                lane_bytes += msg.byte_len();

                let (chunk_start, chunk_bits) = match decode_bit_chunk(msg) {
                    Ok(res) => res,
                    Err(err) => {
                        return Err(err.wrap_err("Dealer cannot decode bit chunk"));
                    }
                };
                if chunk_start != start {
                    return Err(eyre!(
                        "Dealer expected chunk start {} but got {} on lane {}",
                        start,
                        chunk_start,
                        lane
                    ));
                }
                if chunk_bits.len() != chunk_bits_count {
                    return Err(eyre!(
                        "Dealer expected {} bits for chunk starting at {}, got {}",
                        chunk_bits_count,
                        chunk_start,
                        chunk_bits.len()
                    ));
                }
                if tx_clone.send((idx, chunk_bits)).await.is_err() {
                    return Ok(lane_bytes);
                }
            }
            Ok(lane_bytes)
        }));
    }
    drop(tx);

    let handle = tokio::spawn(async move {
        let mut total_bytes = 0usize;
        for handle in handles {
            total_bytes += handle.await??;
        }
        Ok(total_bytes)
    });

    (rx, handle)
}

async fn dealer_collect_bit_chunks<const TIMED: bool>(
    ctx: &BinaryFssContext,
    role: usize,
    chunk_plan: &ChunkPlan,
    bucket_bound: usize,
) -> Result<
    (
        Vec<RingElement<Bit>>,
        Vec<RingElement<Bit>>,
        DealerCollectorStats,
    ),
    Error,
> {
    if chunk_plan.is_empty() {
        return Ok((Vec::new(), Vec::new(), DealerCollectorStats::default()));
    }

    let ranges_prev = chunk_plan.ranges_arc();
    let ranges_next = chunk_plan.ranges_arc();
    let (mut rx_prev, handle_prev) =
        spawn_dealer_stream_reader::<TIMED>(ctx.clone(), role, ranges_prev, bucket_bound, true);
    let (mut rx_next, handle_next) =
        spawn_dealer_stream_reader::<TIMED>(ctx.clone(), role, ranges_next, bucket_bound, false);

    let chunk_count = chunk_plan.len();
    let total_bits = chunk_plan.total_bits();
    let mut prev_chunks: Vec<Option<Vec<RingElement<Bit>>>> = vec![None; chunk_count];
    let mut next_chunks: Vec<Option<Vec<RingElement<Bit>>>> = vec![None; chunk_count];
    let mut received_prev = 0usize;
    let mut received_next = 0usize;
    let mut flush_idx = 0usize;
    let mut out_prev: Vec<RingElement<Bit>> = Vec::with_capacity(total_bits);
    let mut out_next: Vec<RingElement<Bit>> = Vec::with_capacity(total_bits);

    while flush_idx < chunk_count {
        let progress = tokio::select! {
            Some(msg) = rx_prev.recv(), if received_prev < chunk_count => {
                let (idx, bits) = msg;
                if prev_chunks[idx].is_some() {
                    return Err(eyre!("Dealer received duplicate chunk index {} from P1", idx));
                }
                prev_chunks[idx] = Some(bits);
                received_prev += 1;
                true
            },
            Some(msg) = rx_next.recv(), if received_next < chunk_count => {
                let (idx, bits) = msg;
                if next_chunks[idx].is_some() {
                    return Err(eyre!("Dealer received duplicate chunk index {} from P0", idx));
                }
                next_chunks[idx] = Some(bits);
                received_next += 1;
                true
            },
            else => false,
        };

        if !progress {
            break;
        }

        while flush_idx < chunk_count {
            if prev_chunks[flush_idx].is_some() && next_chunks[flush_idx].is_some() {
                let chunk0 = prev_chunks[flush_idx].take().unwrap();
                let chunk1 = next_chunks[flush_idx].take().unwrap();
                out_prev.extend(chunk0);
                out_next.extend(chunk1);
                flush_idx += 1;
            } else {
                break;
            }
        }
    }

    if flush_idx < chunk_count {
        return Err(eyre!(
            "Dealer stream closed before all chunks were received ({} of {})",
            flush_idx,
            chunk_count
        ));
    }

    let bytes_prev = handle_prev.await??;
    let bytes_next = handle_next.await??;

    Ok((
        out_prev,
        out_next,
        DealerCollectorStats {
            from_prev_bytes: bytes_prev,
            from_next_bytes: bytes_next,
        },
    ))
}

async fn receive_key_state<const TIMED: bool>(
    ctx: &BinaryFssContext,
    role: usize,
    n: usize,
    bucket_bound: usize,
    src: KeySource,
) -> Result<(EvaluatorKeyState, usize), Error> {
    let _tt_net_recv =
        maybe_perf_scoped_for_party!(TIMED, "fss.network.start_recv_keys", role, n, bucket_bound);
    let pkg = match src {
        KeySource::Prev => ctx.network_session().receive_prev().await,
        KeySource::Next => ctx.network_session().receive_next().await,
    }
    .map_err(|e| eyre!("Party {role} cannot receive FSS key package: {e}"))?;
    drop(_tt_net_recv);

    let pkg_size = pkg.byte_len();
    let (lens_re, key_blob_re) = decode_key_package(pkg)?;
    let lens = RingElement::<u32>::convert_vec(lens_re);
    let key_blob = RingElement::<u32>::convert_vec(key_blob_re);
    Ok((EvaluatorKeyState::new(lens, key_blob)?, pkg_size))
}

async fn reconstruct_masks_p0<const TIMED: bool>(
    ctx: &BinaryFssContext,
    batch: &[Share<u32>],
    role: usize,
    n: usize,
    bucket_bound: usize,
    lane_idx: usize,
) -> Result<(Vec<RingElement<u32>>, usize, usize), Error> {
    let send_d2r2: Vec<RingElement<u32>> = batch.iter().map(|x| x.b).collect();
    let network = ctx.network_session();
    let lane = LaneId(lane_idx);
    let send_future = {
        let client = network.clone();
        let payload = NetworkInt::new_network_vec(send_d2r2);
        let payload_size = payload.byte_len();
        let lane = lane;
        async move {
            let _tt_net_recon = maybe_perf_scoped_for_party!(
                TIMED,
                "fss.network.recon.send",
                role,
                n,
                bucket_bound
            );
            let res = client
                .send_next_on_lane(lane, payload)
                .await
                .map(|r| (r, payload_size));
            drop(_tt_net_recon);
            res
        }
    };
    let recv_future = {
        let lane = lane;
        async move {
            let _tt_net_recon = maybe_perf_scoped_for_party!(
                TIMED,
                "fss.network.recon.recv",
                role,
                n,
                bucket_bound
            );
            let msg = network
                .receive_next_on_lane(lane)
                .await
                .map_err(|e| eyre!("FSS: Party 0 cannot receive d1+r1 vector from P1: {e}"))?;
            let msg_size = msg.byte_len();
            drop(_tt_net_recon);
            Ok::<(NetworkValue, usize), Error>((msg, msg_size))
        }
    };

    let ((_, send_bytes), (d1r1_msg, recv_bytes)) = futures::try_join!(send_future, recv_future)?;
    Ok((u32::into_vec(d1r1_msg)?, send_bytes, recv_bytes))
}

async fn reconstruct_masks_p1<const TIMED: bool>(
    ctx: &BinaryFssContext,
    batch: &[Share<u32>],
    role: usize,
    n: usize,
    bucket_bound: usize,
    lane_idx: usize,
) -> Result<(Vec<RingElement<u32>>, usize, usize), Error> {
    let send_d1r1: Vec<RingElement<u32>> = batch.iter().map(|x| x.a).collect();
    let network = ctx.network_session();
    let lane = LaneId(lane_idx);
    let send_future = {
        let client = network.clone();
        let payload = NetworkInt::new_network_vec(send_d1r1);
        let payload_size = payload.byte_len();
        let lane = lane;
        async move {
            let _tt_net_recon = maybe_perf_scoped_for_party!(
                TIMED,
                "fss.network.recon.send",
                role,
                n,
                bucket_bound
            );
            let res = client
                .send_prev_on_lane(lane, payload)
                .await
                .map(|r| (r, payload_size));
            drop(_tt_net_recon);
            res
        }
    };
    let recv_future = {
        let lane = lane;
        async move {
            let _tt_net_recon = maybe_perf_scoped_for_party!(
                TIMED,
                "fss.network.recon.recv",
                role,
                n,
                bucket_bound
            );
            let msg = network
                .receive_prev_on_lane(lane)
                .await
                .map_err(|e| eyre!("FSS: Party 1 cannot receive d2+r2 vector from P0: {e}"))?;
            let msg_size = msg.byte_len();
            drop(_tt_net_recon);
            Ok::<(NetworkValue, usize), Error>((msg, msg_size))
        }
    };

    let ((_, send_bytes), (d2r2_msg, recv_bytes)) = futures::try_join!(send_future, recv_future)?;
    Ok((u32::into_vec(d2r2_msg)?, send_bytes, recv_bytes))
}

#[inline]
fn evaluate_bits_stage<const TIMED: bool>(
    role: usize,
    n: usize,
    bucket_bound: usize,
    n_half_u32: u32,
    parallel_threshold: usize,
    is_party1: bool,
    p: InG,
    q: InG,
    icf: &Icf<Aes128MatyasMeyerOseasPrg<16, 2, 4>>,
    key_state: &EvaluatorKeyState,
    batch: &[Share<u32>],
    masks: &[RingElement<u32>],
    base_idx: usize,
) -> Vec<RingElement<Bit>> {
    #[cfg(not(feature = "parallel-msb"))]
    let _ = parallel_threshold;

    #[inline]
    fn sequential_eval<const TIMED_INNER: bool>(
        role: usize,
        n: usize,
        bucket_bound: usize,
        n_half_u32: u32,
        is_party1: bool,
        icf: &Icf<Aes128MatyasMeyerOseasPrg<16, 2, 4>>,
        key_state: &EvaluatorKeyState,
        batch: &[Share<u32>],
        masks: &[RingElement<u32>],
        base_idx: usize,
    ) -> Vec<RingElement<Bit>> {
        let _tt = maybe_perf_scoped_for_party!(
            TIMED_INNER,
            "fss.add3.non-parallel",
            role,
            n,
            bucket_bound
        );

        let mut bits: Vec<RingElement<Bit>> = Vec::with_capacity(batch.len());
        for i in 0..batch.len() {
            let global_idx = base_idx + i;
            let x = &batch[i];
            let y = x.a + masks[i] + x.b + RingElement(n_half_u32);

            crate::perf_time_let_for_party!(
                "fss.add3.icf.eval",
                role,
                n,
                bucket_bound,
                    let f = icf.eval(is_party1, key_state.key(global_idx), fss_rs::group::int::U32Group(y.0))
            );

            let f_u128 = u128::from_le_bytes(f.0);
            let b = (f_u128 & 1) != 0;
            bits.push(RingElement(Bit::new(b)));
        }
        bits
    }

    #[cfg(feature = "parallel-msb")]
    if batch.len() >= parallel_threshold {
        use rayon::prelude::*;

        (0..batch.len())
            .into_par_iter()
            .map(|i| {
                let global_idx = base_idx + i;
                let x = &batch[i];
                let y = x.a + masks[i] + x.b + RingElement(n_half_u32);

                let seed = [[0u8; 16]; 4];
                let prg_i = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                    &seed[0], &seed[1], &seed[2], &seed[3],
                ]);
                let icf_i = Icf::new(p, q, prg_i);
                let key_i = IcShare::deserialize(key_state.key_slice(global_idx))
                    .expect("deserialize IcShare for evaluator");
                let f = icf_i.eval(is_party1, &key_i, fss_rs::group::int::U32Group(y.0));
                let f_u128 = u128::from_le_bytes(f.0);
                let b = (f_u128 & 1) != 0;
                RingElement(Bit::new(b))
            })
            .collect()
    } else {
        sequential_eval::<TIMED>(
            role,
            n,
            bucket_bound,
            n_half_u32,
            is_party1,
            icf,
            key_state,
            batch,
            masks,
            base_idx,
        )
    }

    #[cfg(not(feature = "parallel-msb"))]
    sequential_eval::<TIMED>(
        role,
        n,
        bucket_bound,
        n_half_u32,
        is_party1,
        icf,
        key_state,
        batch,
        masks,
        base_idx,
    )
}

async fn exchange_bits_p0<const TIMED: bool>(
    ctx: &BinaryFssContext,
    bits: &[RingElement<Bit>],
    chunk_start: usize,
    role: usize,
    n: usize,
    bucket_bound: usize,
    lane_idx: usize,
) -> Result<(Vec<RingElement<Bit>>, usize, usize, usize), Error> {
    let encoded_chunk = encode_bit_chunk(chunk_start, bits);
    let chunk_bytes = encoded_chunk.byte_len();
    let network = ctx.network_session();
    let lane = LaneId(lane_idx);
    let send_prev_future = {
        let client = network.clone();
        let msg = encoded_chunk.clone();
        let lane = lane;
        async move {
            let _tt_net = maybe_perf_scoped_for_party!(
                TIMED,
                "fss.network.post-icf.send_prev",
                role,
                n,
                bucket_bound
            );
            let res = client.send_prev_on_lane(lane, msg).await;
            drop(_tt_net);
            res
        }
    };
    let send_next_future = {
        let client = network.clone();
        let msg = encoded_chunk;
        let lane = lane;
        async move {
            let _tt_net = maybe_perf_scoped_for_party!(
                TIMED,
                "fss.network.post-icf.send_next",
                role,
                n,
                bucket_bound
            );
            let res = client.send_next_on_lane(lane, msg).await;
            drop(_tt_net);
            res
        }
    };
    let recv_future = {
        let lane = lane;
        async move {
            let _tt_net = maybe_perf_scoped_for_party!(
                TIMED,
                "fss.network.post-icf.recv_to_eval",
                role,
                n,
                bucket_bound
            );
            let msg = network
                .receive_next_on_lane(lane)
                .await
                .map_err(|e| eyre!("Party 0 cannot receive bit vector from P1: {e}"))?;
            let recv_bytes = msg.byte_len();
            drop(_tt_net);
            Ok::<(NetworkValue, usize), Error>((msg, recv_bytes))
        }
    };

    let (_, _, (recv_msg, recv_bytes)) =
        futures::try_join!(send_prev_future, send_next_future, recv_future)?;
    let (peer_start, peer_bits) = decode_bit_chunk(recv_msg)?;
    if peer_start != chunk_start {
        return Err(eyre!(
            "Party 0 received chunk start {} but expected {}",
            peer_start,
            chunk_start
        ));
    }
    Ok((peer_bits, chunk_bytes, chunk_bytes, recv_bytes))
}

async fn exchange_bits_p1<const TIMED: bool>(
    ctx: &BinaryFssContext,
    bits: &[RingElement<Bit>],
    chunk_start: usize,
    role: usize,
    n: usize,
    bucket_bound: usize,
    lane_idx: usize,
) -> Result<(Vec<RingElement<Bit>>, usize, usize, usize), Error> {
    let encoded_chunk = encode_bit_chunk(chunk_start, bits);
    let chunk_bytes = encoded_chunk.byte_len();
    let network = ctx.network_session();
    let lane = LaneId(lane_idx);
    let send_next_future = {
        let client = network.clone();
        let msg = encoded_chunk.clone();
        let lane = lane;
        async move {
            let _tt_net = maybe_perf_scoped_for_party!(
                TIMED,
                "fss.network.post-icf.send_next",
                role,
                n,
                bucket_bound
            );
            let res = client.send_next_on_lane(lane, msg).await;
            drop(_tt_net);
            res
        }
    };
    let send_prev_future = {
        let client = network.clone();
        let msg = encoded_chunk;
        let lane = lane;
        async move {
            let _tt_net = maybe_perf_scoped_for_party!(
                TIMED,
                "fss.network.post-icf.send_prev",
                role,
                n,
                bucket_bound
            );
            let res = client.send_prev_on_lane(lane, msg).await;
            drop(_tt_net);
            res
        }
    };
    let recv_future = {
        let lane = lane;
        async move {
            let _tt_net = maybe_perf_scoped_for_party!(
                TIMED,
                "fss.network.post-icf.recv_to_eval",
                role,
                n,
                bucket_bound
            );
            let msg = network
                .receive_prev_on_lane(lane)
                .await
                .map_err(|e| eyre!("Party 1 cannot receive bit vector from P0: {e}"))?;
            let recv_bytes = msg.byte_len();
            drop(_tt_net);
            Ok::<(NetworkValue, usize), Error>((msg, recv_bytes))
        }
    };

    let (_, _, (recv_msg, recv_bytes)) =
        futures::try_join!(send_next_future, send_prev_future, recv_future)?;
    let (peer_start, peer_bits) = decode_bit_chunk(recv_msg)?;
    if peer_start != chunk_start {
        return Err(eyre!(
            "Party 1 received chunk start {} but expected {}",
            peer_start,
            chunk_start
        ));
    }
    Ok((peer_bits, chunk_bytes, chunk_bytes, recv_bytes))
}

async fn add_3_get_msb_fss_batch_parallel_threshold_impl<const TIMED: bool>(
    ctx: BinaryFssContext,
    batch: Vec<Share<u32>>,
    parallel_threshold: usize,
) -> Result<Vec<Share<Bit>>, Error> {
    use eyre::eyre;
    use fss_rs::icf::{IcShare, Icf, InG, IntvFn, OutG};
    use fss_rs::prg::Aes128MatyasMeyerOseasPrg;
    use rand::thread_rng;

    let role = ctx.own_role().index();
    let n = batch.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let chunk_plan = ChunkPlan::new(n, parallel_threshold);
    if chunk_plan.is_empty() {
        return Ok(Vec::new());
    }
    let chunk_ranges = chunk_plan.ranges();
    let batch_arc = Arc::new(batch);
    let bucket_bound = 150;
    #[cfg(not(feature = "parallel-msb"))]
    let _ = parallel_threshold;

    #[inline]
    fn re_vec_to_u32(v: Vec<RingElement<u32>>) -> Vec<u32> {
        RingElement::<u32>::convert_vec(v)
    }
    #[inline]
    fn u32_to_re_vec(v: Vec<u32>) -> Vec<RingElement<u32>> {
        RingElement::<u32>::convert_vec_rev(v)
    }

    // We test y ∈ [0, 2^31-1] where y = x + r_in + 2^31 (mod 2^32)
    let p = InG::from(0u32);
    //let q = InG::from((1u32 << 31) - 1);
    let q = InG::from(1u32 << 31); // [0, 2^31)
    let n_half_u32: u32 = 1u32 << 31;

    match role {
        // =======================
        // Party 0 (Evaluator)
        // =======================
        0 => {
            let (key_state, key_bytes) =
                receive_key_state::<TIMED>(&ctx, role, n, bucket_bound, KeySource::Prev).await?;
            let detail_enabled = traffic_detail_enabled();
            let mut traffic = EvalTrafficStats::default();
            traffic.key_recv_bytes += key_bytes;

            // Build a single PRG/ICF for eval
            let seed = [[0u8; 16]; 4];
            let prg = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                &seed[0], &seed[1], &seed[2], &seed[3],
            ]);
            let icf = Icf::new(p, q, prg);

            let mut outputs: Vec<Share<Bit>> = Vec::with_capacity(n);
            let mut next_chunk_idx = 0usize;
            let batch_data = batch_arc.clone();
            let mut pending_recon: VecDeque<ReconTask> = VecDeque::new();
            let lane_count = ctx.lane_count().max(1);

            let mut launch_recon = |pending: &mut VecDeque<ReconTask>, next_idx: &mut usize| {
                while pending.len() < FSS_MAX_PENDING_RECONS && *next_idx < chunk_ranges.len() {
                    let (start, end) = chunk_ranges[*next_idx];
                    let chunk_len = end - start;
                    let chunk_idx = *next_idx;
                    let lane_idx = chunk_idx % lane_count;
                    let ctx_clone = ctx.clone();
                    let batch_clone = batch_data.clone();
                    let handle = tokio::spawn(async move {
                        reconstruct_masks_p0::<TIMED>(
                            &ctx_clone,
                            &batch_clone[start..end],
                            role,
                            chunk_len,
                            bucket_bound,
                            lane_idx,
                        )
                        .await
                    });
                    pending.push_back(ReconTask { start, end, handle });
                    *next_idx += 1;
                }
            };

            launch_recon(&mut pending_recon, &mut next_chunk_idx);

            let mut processed_idx = 0usize;
            for _ in 0..chunk_ranges.len() {
                let ReconTask { start, end, handle } = pending_recon
                    .pop_front()
                    .expect("missing pending reconstruction task");
                let (masks, send_bytes, recv_bytes) = handle
                    .await
                    .map_err(|e| eyre!("Party 0 recon task panicked: {e}"))??;
                traffic.recon_send_bytes += send_bytes;
                traffic.recon_recv_bytes += recv_bytes;
                let chunk_len = end - start;

                let bit0s_vec = evaluate_bits_stage::<TIMED>(
                    role,
                    chunk_len,
                    bucket_bound,
                    n_half_u32,
                    parallel_threshold,
                    false,
                    p,
                    q,
                    &icf,
                    &key_state,
                    &batch_data[start..end],
                    &masks,
                    start,
                );

                let local_bits = Arc::new(bit0s_vec);
                let send_bits = local_bits.clone();
                let ctx_clone = ctx.clone();
                let chunk_start = start;
                let lane_idx = processed_idx % lane_count;
                let chunk_bits = chunk_len;
                let exchange_future = tokio::spawn(async move {
                    exchange_bits_p0::<TIMED>(
                        &ctx_clone,
                        send_bits.as_slice(),
                        chunk_start,
                        role,
                        chunk_bits,
                        bucket_bound,
                        lane_idx,
                    )
                    .await
                });

                launch_recon(&mut pending_recon, &mut next_chunk_idx);

                let (peer_bits, send_prev_bytes, send_next_bytes, recv_bytes) = exchange_future
                    .await
                    .map_err(|e| eyre!("Party 0 exchange task panicked: {e}"))??;
                traffic.post_send_prev_bytes += send_prev_bytes;
                traffic.post_send_next_bytes += send_next_bytes;
                traffic.post_recv_bytes += recv_bytes;
                let local_vec = Arc::try_unwrap(local_bits).unwrap_or_else(|arc| (*arc).clone());
                outputs.extend(
                    local_vec
                        .into_iter()
                        .zip(peer_bits.into_iter())
                        .map(|(b0, b1)| Share::new(b0, b1)),
                );
                processed_idx += 1;
            }

            if detail_enabled {
                log_eval_stats(role, "P0", &traffic);
            }
            record_traffic_totals(role, traffic.send_total(), traffic.recv_total());
            store_eval_snapshot(role, &traffic);
            Ok(outputs)
        }

        // =======================
        // Party 1 (Evaluator)
        // =======================
        1 => {
            let (key_state, key_bytes) =
                receive_key_state::<TIMED>(&ctx, role, n, bucket_bound, KeySource::Next).await?;
            let detail_enabled = traffic_detail_enabled();
            let mut traffic = EvalTrafficStats::default();
            traffic.key_recv_bytes += key_bytes;

            let seed = [[0u8; 16]; 4];
            let prg = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                &seed[0], &seed[1], &seed[2], &seed[3],
            ]);
            let icf = Icf::new(p, q, prg);

            let mut outputs: Vec<Share<Bit>> = Vec::with_capacity(n);
            let mut next_chunk_idx = 0usize;
            let batch_data = batch_arc.clone();
            let mut pending_recon: VecDeque<ReconTask> = VecDeque::new();
            let lane_count = ctx.lane_count().max(1);

            let mut launch_recon = |pending: &mut VecDeque<ReconTask>, next_idx: &mut usize| {
                while pending.len() < FSS_MAX_PENDING_RECONS && *next_idx < chunk_ranges.len() {
                    let (start, end) = chunk_ranges[*next_idx];
                    let chunk_len = end - start;
                    let chunk_idx = *next_idx;
                    let lane_idx = chunk_idx % lane_count;
                    let ctx_clone = ctx.clone();
                    let batch_clone = batch_data.clone();
                    let handle = tokio::spawn(async move {
                        reconstruct_masks_p1::<TIMED>(
                            &ctx_clone,
                            &batch_clone[start..end],
                            role,
                            chunk_len,
                            bucket_bound,
                            lane_idx,
                        )
                        .await
                    });
                    pending.push_back(ReconTask { start, end, handle });
                    *next_idx += 1;
                }
            };

            launch_recon(&mut pending_recon, &mut next_chunk_idx);

            let mut processed_idx = 0usize;
            for _ in 0..chunk_ranges.len() {
                let ReconTask { start, end, handle } = pending_recon
                    .pop_front()
                    .expect("missing pending reconstruction task");
                let (masks, send_bytes, recv_bytes) = handle
                    .await
                    .map_err(|e| eyre!("Party 1 recon task panicked: {e}"))??;
                traffic.recon_send_bytes += send_bytes;
                traffic.recon_recv_bytes += recv_bytes;
                let chunk_len = end - start;

                let bit1s_vec = evaluate_bits_stage::<TIMED>(
                    role,
                    chunk_len,
                    bucket_bound,
                    n_half_u32,
                    parallel_threshold,
                    true,
                    p,
                    q,
                    &icf,
                    &key_state,
                    &batch_data[start..end],
                    &masks,
                    start,
                );

                let local_bits = Arc::new(bit1s_vec);
                let send_bits = local_bits.clone();
                let ctx_clone = ctx.clone();
                let chunk_start = start;
                let lane_idx = processed_idx % lane_count;
                let exchange_future = tokio::spawn(async move {
                    exchange_bits_p1::<TIMED>(
                        &ctx_clone,
                        send_bits.as_slice(),
                        chunk_start,
                        role,
                        chunk_len,
                        bucket_bound,
                        lane_idx,
                    )
                    .await
                });

                launch_recon(&mut pending_recon, &mut next_chunk_idx);

                let (peer_bits, send_prev_bytes, send_next_bytes, recv_bytes) = exchange_future
                    .await
                    .map_err(|e| eyre!("Party 1 exchange task panicked: {e}"))??;
                traffic.post_send_prev_bytes += send_prev_bytes;
                traffic.post_send_next_bytes += send_next_bytes;
                traffic.post_recv_bytes += recv_bytes;
                let local_vec = Arc::try_unwrap(local_bits).unwrap_or_else(|arc| (*arc).clone());
                outputs.extend(
                    peer_bits
                        .into_iter()
                        .zip(local_vec.into_iter())
                        .map(|(b0, b1)| Share::new(b0, b1)),
                );
                processed_idx += 1;
            }

            if detail_enabled {
                log_eval_stats(role, "P1", &traffic);
            }
            record_traffic_totals(role, traffic.send_total(), traffic.recv_total());
            store_eval_snapshot(role, &traffic);
            Ok(outputs)
        }

        // =======================
        // Party 2 (Dealer)
        // =======================
        2 => {
            // // Build r_in (input masks) using the ctx PRF; must match evaluators' PRF consumption
            // let mut r_in_list: Vec<u32> = Vec::with_capacity(n);
            // for _ in 0..n {
            //     let (r_next, r_prev) = ctx.prf.gen_rands::<RingElement<u32>>().clone();
            //     let r_in = (r_next + r_prev).convert();
            //     r_in_list.push(r_in);
            // }

            // This variant assumes r1 = r2 = 0 on evaluators, so use r_in = 0 here as well.
            // Otherwise keys would be generated for x + r_in while evaluators feed x.
            let r_in_list: Vec<u32> = vec![0u32; n];

            // Fresh ICF.Gen per index, r_out = 0; adaptive parallelism by threshold
            // First collect key pairs, then pre-size and flatten to avoid reallocs.
            let pairs: Vec<(Vec<u32>, Vec<u32>)> = {
                #[cfg(feature = "parallel-msb")]
                {
                    //metrics: measure the genkeys time
                    let _tt_gen = maybe_perf_scoped_for_party!(
                        TIMED,
                        "fss.dealer.genkeys",
                        role,
                        n,            // bucket on the items this block processes
                        bucket_bound  // your desired bucket cap
                    );

                    if n >= parallel_threshold {
                        use rayon::prelude::*;
                        r_in_list
                            .par_iter()
                            .map(|&r_in_u32| {
                                let seed = [[0u8; 16]; 4];
                                let prg_i = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                                    &seed[0], &seed[1], &seed[2], &seed[3],
                                ]);
                                let icf = Icf::new(p, q, prg_i);
                                let mut rng = thread_rng();
                                let f = IntvFn {
                                    //r_in: InG::from(r_in_u32),
                                    r_in: InG::from(0u32),
                                    r_out: OutG::from(0u128),
                                };
                                let (k0, k1) = icf.gen(f, &mut rng);
                                (k0.serialize().unwrap(), k1.serialize().unwrap())
                            })
                            .collect()
                    } else {
                        let mut tmp = Vec::with_capacity(n);
                        for &r_in_u32 in &r_in_list {
                            let seed = [[0u8; 16]; 4];
                            let prg_i = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                                &seed[0], &seed[1], &seed[2], &seed[3],
                            ]);
                            let icf = Icf::new(p, q, prg_i);
                            let mut rng = thread_rng();
                            let f = IntvFn {
                                //r_in: InG::from(r_in_u32),
                                r_in: InG::from(0u32),
                                r_out: OutG::from(0u128),
                            };
                            let (k0, k1) = icf.gen(f, &mut rng);
                            tmp.push((k0.serialize()?, k1.serialize()?));
                        }
                        tmp
                    }
                }

                #[cfg(not(feature = "parallel-msb"))]
                {
                    //metrics: measure the genkeys time
                    let _tt_gen = maybe_perf_scoped_for_party!(
                        TIMED,
                        "fss.dealer.genkeys",
                        role,
                        n,            // bucket on the items this block processes
                        bucket_bound  // your desired bucket cap
                    );

                    let mut tmp = Vec::with_capacity(n);
                    for &r_in_u32 in &r_in_list {
                        let seed = [[0u8; 16]; 4];
                        let prg_i = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                            &seed[0], &seed[1], &seed[2], &seed[3],
                        ]);
                        let icf = Icf::new(p, q, prg_i);
                        let mut rng = thread_rng();
                        let f = IntvFn {
                            //r_in: InG::from(r_in_u32),
                            r_in: InG::from(0u32),
                            r_out: OutG::from(0u128),
                        };
                        let (k0, k1) = icf.gen(f, &mut rng);
                        tmp.push((k0.serialize()?, k1.serialize()?));
                    }
                    tmp
                }
            };

            // Flatten and send to P0 (prev) and P1 (next)
            let lens_p0: Vec<u32> = pairs.iter().map(|(k0, _)| k0.len() as u32).collect();
            let lens_p1: Vec<u32> = pairs.iter().map(|(_, k1)| k1.len() as u32).collect();
            let key_blob_for_p0: Vec<u32> = pairs.iter().flat_map(|(k0, _)| k0.clone()).collect();
            let key_blob_for_p1: Vec<u32> = pairs.iter().flat_map(|(_, k1)| k1.clone()).collect();

            let detail_enabled = traffic_detail_enabled();
            let mut traffic = DealerTrafficStats::default();
            let network = ctx.network_session();
            let pkg_prev = encode_key_package(
                u32_to_re_vec(lens_p0.clone()),
                u32_to_re_vec(key_blob_for_p0),
            );
            let pkg_prev_bytes = pkg_prev.byte_len();
            let pkg_next = encode_key_package(
                u32_to_re_vec(lens_p1.clone()),
                u32_to_re_vec(key_blob_for_p1),
            );
            let pkg_next_bytes = pkg_next.byte_len();
            record_key_package_bytes(pkg_prev_bytes, pkg_next_bytes);
            traffic.key_send_prev_bytes += pkg_prev_bytes;
            traffic.key_send_next_bytes += pkg_next_bytes;

            let send_prev_future = {
                let client = network.clone();
                let pkg = pkg_prev;
                async move {
                    let _tt_net0a = maybe_perf_scoped_for_party!(
                        TIMED,
                        "fss.network.dealer.send_P0a",
                        role,
                        n,
                        bucket_bound
                    );
                    let res = client.send_prev(pkg).await;
                    drop(_tt_net0a);
                    res
                }
            };

            let send_next_future = {
                let client = network.clone();
                let pkg = pkg_next;
                async move {
                    let _tt_net1a = maybe_perf_scoped_for_party!(
                        TIMED,
                        "fss.network.dealer.send_P1a",
                        role,
                        n,
                        bucket_bound
                    );
                    let res = client.send_next(pkg).await;
                    drop(_tt_net1a);
                    res
                }
            };

            let collect_future = {
                let ctx_clone = ctx.clone();
                let chunk_plan_clone = chunk_plan.clone();
                async move {
                    dealer_collect_bit_chunks::<TIMED>(
                        &ctx_clone,
                        role,
                        &chunk_plan_clone,
                        bucket_bound,
                    )
                    .await
                }
            };

            let (_, _, (bit0s_ringbit, bit1s_ringbit, collector_stats)) =
                futures::try_join!(send_prev_future, send_next_future, collect_future,)?;
            traffic.share_recv_from_p0_bytes += collector_stats.from_prev_bytes;
            traffic.share_recv_from_p1_bytes += collector_stats.from_next_bytes;

            // Keep the (P0_bits, P1_bits) ordering to match P0/P1
            let out: Vec<Share<Bit>> = bit0s_ringbit
                .into_iter()
                .zip(bit1s_ringbit.into_iter())
                .map(|(b0, b1)| Share::new(b0, b1))
                .collect();

            if detail_enabled {
                log_dealer_stats(role, &traffic);
            }
            store_dealer_snapshot(&traffic);
            record_traffic_totals(role, traffic.send_total(), traffic.recv_total());
            Ok(out)
        }

        _ => Err(eyre!("invalid role index {}", role).into()),
    }
}

pub(crate) async fn add_3_get_msb_fss_batch_parallel_threshold_timers(
    ctx: BinaryFssContext,
    batch: Vec<Share<u32>>,
    parallel_threshold: usize,
) -> Result<Vec<Share<Bit>>, Error> {
    add_3_get_msb_fss_batch_parallel_threshold_impl::<true>(ctx, batch, parallel_threshold).await
}

pub(crate) async fn add_3_get_msb_fss_batch_parallel_threshold(
    ctx: BinaryFssContext,
    batch: Vec<Share<u32>>,
    parallel_threshold: usize,
) -> Result<Vec<Share<Bit>>, Error> {
    add_3_get_msb_fss_batch_parallel_threshold_impl::<false>(ctx, batch, parallel_threshold).await
}

// Evaluation (P0/P1) and generation (P2) becomes parallel under `parallel-msb` if batch.len > parallel_threshold
// Here r_2 = r_1 = 0, so d2r2=d2 and d1r1=d1

/// Batched version of the function above
/// Instead of handling one request at a time, get a batch of size ???
/// Main differences: each party
pub(crate) async fn add_3_get_msb_fss_batch_timers(
    session: &mut Session,
    x: &[Share<u32>],
) -> Result<Vec<Share<Bit>>, Error>
where
    Standard: Distribution<u32>,
{
    // Input is Share {a,b}, in the notation below we have:
    // Party0: a=d0, b=d2
    // Party1: a=d1, b=d0
    // Party2: a=d2, b=d1

    // Get party number
    let role = session.own_role().index();
    let n = x.len();
    let batch_size = x.len();
    let bucket_bound = 150;

    // Depending on the role, do different stuff
    match role {
        0 => {
            //Generate all r2 prf key, keep the r0 keys for later
            // println!("party 0: Batch size is {}", batch_size);
            let mut r_prime_keys = Vec::with_capacity(batch_size);
            let mut d2r2_vec = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let (r_prime_temp, _) = session.prf.gen_rands::<RingElement<u32>>().clone(); //_ is r2
                r_prime_keys.push(RingElement::<u128>(u128::from(r_prime_temp.0))); //convet to this for later
                d2r2_vec.push(x[i].b + RingElement(0)); // change this to take the second thing gen_rands returns
            }

            // Send the vector of d2+r2 to party 1
            let clone_d2r2_vec = d2r2_vec.clone();

            //metrics: measure the network for share reconstruction
            let _tt_net_recon = crate::perf_scoped_for_party!(
                "fss.network.recon.send",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session
                .network_session
                .send_next(u32::new_network_vec(clone_d2r2_vec))
                .await?;

            // drop timer
            drop(_tt_net_recon);

            //metrics: measure the network for share reconstruction
            let _tt_net_recon = crate::perf_scoped_for_party!(
                "fss.network.recon.recv",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Receive d1+r1 from party 1
            let d1r1 = match session.network_session.receive_next().await {
                Ok(v) => u32::into_vec(v),
                Err(e) => Err(eyre!("FSS: Party 0 cannot receive d1+r1 from party 1: {e}")),
            }?;

            // drop timer
            drop(_tt_net_recon);

            // metrics: measure the network for share reconstruction
            let _tt_net_recv = crate::perf_scoped_for_party!(
                "fss.network.start_recv_keys",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Receive batch_size number of fss keys from dealer
            let k_fss_0_vec = match session.network_session.receive_prev().await {
                Ok(v) => u32::into_vec(v),
                Err(e) => Err(eyre!("Party 0 cannot receive my fss key from dealer {e}")),
            }?;

            // drop timer
            drop(_tt_net_recv);

            // Set up the function for FSS
            // we need this below to handle signed numbers, if input is unsigned no need to add N/2
            let n_falf_u32 = 1u32 << 31;
            let n_half = InG::from(n_falf_u32);
            // make the interval so that we return 1 when MSB == 1
            // this is (our number + n/2 ) % n, modulo is handled by U32Group
            let p = InG::from(1u32 << 31) + n_half;
            let q = InG::from(u32::MAX) + n_half; // modulo is handled by U32Group
            let keys: Vec<[u8; 16]> = vec![[0u8; 16]; 4];
            let mut f_x_0_bits = Vec::with_capacity(batch_size); // store all the eval results

            // Deserialize each to find original IcShare and call eval
            let key_words_fss_0: Vec<u32> = RingElement::<u32>::convert_vec(k_fss_0_vec); //need to un-flatten key vector
            let mut offset: usize = 0;
            for i in 0..batch_size {
                // // Need to "unflatten" to get batch_size number of fss keys
                let curr_key_byte_len = 1 + (key_words_fss_0[offset] as usize + 3) / 4; // offset index has the byte length, then find total u32s for this key

                // Get current key
                let k_fss_0_icshare: IcShare =
                    IcShare::deserialize(&key_words_fss_0[offset..offset + curr_key_byte_len])?;
                offset += curr_key_byte_len; //update offset to point to next cell that contains size of next key

                // reconstruct the input d+r [recall x.a=d0] for each x[i]
                let d_plus_r: RingElement<u32> =
                    d1r1[i] + d2r2_vec[i] + x[i].a + RingElement(n_falf_u32);
                // this should be wrapping addition, implemented by RingElement

                // Now we're ready to call eval
                let prg =
                    Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&std::array::from_fn(|i| &keys[i]));
                let icf = Icf::new(p, q, prg);
                //Call eval & convert from from ByteGroup<16> to RingElement<u128>
                let temp_eval = RingElement::<u128>(u128::from_le_bytes(
                    icf.eval(
                        false,
                        &k_fss_0_icshare,
                        fss_rs::group::int::U32Group(d_plus_r.0),
                    )
                    .0,
                ));

                // Add the respective r_prime and add to the vector of results and take only the LSB
                // Make them RingElements so it's easy to send to network
                f_x_0_bits.push(RingElement(Bit::new(
                    ((temp_eval ^ r_prime_keys[i]).0 & 1) != 0,
                )));
            }

            // Prepare them in a vector to send to dealer and next party
            let f_0_res_network: Vec<NetworkValue> = f_x_0_bits
                .iter()
                .copied()
                .map(NetworkValue::RingElementBit)
                .collect();

            let cloned_f_0_res_network = f_0_res_network.clone();

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.send_next",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session // send to party 1
                .network_session
                .send_next(NetworkValue::vec_to_network(cloned_f_0_res_network))
                .await?;

            // drop timer
            drop(_tt_net);

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.send_prev",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session // send to the dealer (party 2)
                .network_session
                .send_prev(NetworkValue::vec_to_network(f_0_res_network))
                .await?;

            // drop timer
            drop(_tt_net);

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.recv",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Receive Bits of share of party 1 --> this is a vec of network values
            let f_x_1_bits_net = match session.network_session.receive_next().await {
                Ok(v) => NetworkValue::vec_from_network(v),
                Err(e) => return Err(eyre!("Party 0 cannot receive bit shares from party 1: {e}")),
            }?;

            // drop timer
            drop(_tt_net);

            // Convert Vec<NetworkValue> to Vec<RingElement<Bit>>
            let f_x_1_bits: Vec<RingElement<Bit>> = f_x_1_bits_net
                .into_iter()
                .map(|nv| match nv {
                    NetworkValue::RingElementBit(b) => Ok(b),
                    other => Err(eyre!("expected RingElementBit, got {:?}", other)),
                })
                .collect::<Result<_, _>>()?;

            // Return a vector of Share<Bit> where the a is from f_x_0_bits
            // and the b is from f_x_1_bits
            let shares: Vec<Share<Bit>> = f_x_0_bits
                .into_iter()
                .zip(f_x_1_bits)
                .map(|(a, b)| Share { a, b })
                .collect();
            Ok(shares)
        }
        1 => {
            // eprintln!("party 1: Batch size is {}", batch_size);
            std::io::stderr().flush().ok();
            let mut r_prime_keys = Vec::with_capacity(batch_size);
            let mut d1r1_vec = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let (_, r_prime_temp) = session.prf.gen_rands::<RingElement<u32>>().clone(); //_ is r1
                r_prime_keys.push(RingElement::<u128>(u128::from(r_prime_temp.0))); //convet to this for later
                d1r1_vec.push(x[i].a + RingElement(0)); // change this to take the first thing gen_rands returns
            }

            // Send the vector of d1+r1 to party 0
            let cloned_d1r1_vec = d1r1_vec.clone();

            //metrics: measure the network for share reconstruction
            let _tt_net_recon = crate::perf_scoped_for_party!(
                "fss.network.recon.send",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session
                .network_session
                .send_prev(u32::new_network_vec(cloned_d1r1_vec))
                .await?;

            // drop timer
            drop(_tt_net_recon);

            //metrics: measure the network for share reconstruction
            let _tt_net_recon = crate::perf_scoped_for_party!(
                "fss.network.recon.recv",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Receive d2+r2 from party 0
            let d2r2_vec = match session.network_session.receive_prev().await {
                Ok(v) => u32::into_vec(v),
                Err(e) => Err(eyre!("FSS: Party 1 cannot receive d2+r2 from party 0: {e}")),
            }?;

            // drop timer
            drop(_tt_net_recon);

            // metrics: measure the network for share reconstruction
            let _tt_net_recv = crate::perf_scoped_for_party!(
                "fss.network.start_recv_keys",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Receive batch_size number of fss keys from dealer
            let k_fss_1_vec = match session.network_session.receive_next().await {
                Ok(v) => u32::into_vec(v),
                Err(e) => Err(eyre!("Party 1 cannot receive my fss key from dealer {e}")),
            }?;

            // drop timer
            drop(_tt_net_recv);

            // Set up the function for FSS
            // we need this below to handle signed numbers, if input is unsigned no need to add N/2
            let n_falf_u32 = 1u32 << 31;
            let n_half = InG::from(n_falf_u32);
            // make the interval so that we return 1 when MSB == 1
            // this is (our number + n/2 ) % n, modulo is handled by U32Group
            let p = InG::from(1u32 << 31) + n_half;
            let q = InG::from(u32::MAX) + n_half; // modulo is handled by U32Group
            let keys: Vec<[u8; 16]> = vec![[0u8; 16]; 4];
            let mut f_x_1_bits = Vec::with_capacity(batch_size); // store all the eval results

            // Deserialize each to find original IcShare and call eval
            // Deserialize each to find original IcShare and call eval
            let key_words_fss_1: Vec<u32> = RingElement::<u32>::convert_vec(k_fss_1_vec); //need to un-flatten key vector
            let mut offset: usize = 0;
            for i in 0..batch_size {
                // // Need to "unflatten" to get batch_size number of fss keys
                let curr_key_byte_len = 1 + (key_words_fss_1[offset] as usize + 3) / 4; // offset index has the byte length, then find total u32s for this key

                // Get current key
                let k_fss_1_icshare: IcShare =
                    IcShare::deserialize(&key_words_fss_1[offset..offset + curr_key_byte_len])?;
                offset += curr_key_byte_len; //update offset to point to next cell that contains size of next key

                // reconstruct the input d+r [recall d0=x.b] for each x[i]
                let d_plus_r: RingElement<u32> =
                    d1r1_vec[i] + d2r2_vec[i] + x[i].b + RingElement(n_falf_u32);
                // this should be wrapping addition, implemented by RingElement

                // Now we're ready to call eval
                let prg =
                    Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&std::array::from_fn(|i| &keys[i]));
                let icf = Icf::new(p, q, prg);
                //Call eval & convert from from ByteGroup<16> to RingElement<u128>
                let temp_eval = RingElement::<u128>(u128::from_le_bytes(
                    icf.eval(
                        true,
                        &k_fss_1_icshare,
                        fss_rs::group::int::U32Group(d_plus_r.0),
                    )
                    .0,
                ));

                // Add the respective r_prime and add to the vector of results and take only the LSB
                // Make them RingElements so it's easy to send to network
                f_x_1_bits.push(RingElement(Bit::new(
                    ((temp_eval ^ r_prime_keys[i]).0 & 1) != 0,
                )));
            }

            // Prepare them in a vector to send to dealer and next party
            let f_1_res_network: Vec<NetworkValue> = f_x_1_bits
                .iter()
                .copied()
                .map(NetworkValue::RingElementBit)
                .collect();

            let cloned_f_1_res_network = f_1_res_network.clone();

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.send_prev",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session // send to party 0
                .network_session
                .send_prev(NetworkValue::vec_to_network(cloned_f_1_res_network))
                .await?;

            // drop timer
            drop(_tt_net);

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.send_next",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session // send to the dealer (party 2)
                .network_session
                .send_next(NetworkValue::vec_to_network(f_1_res_network))
                .await?;

            // drop timer
            drop(_tt_net);

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.recv",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Receive Bits of share of party 0 --> this is a vec of network values
            let f_x_0_bits_net = match session.network_session.receive_prev().await {
                Ok(v) => NetworkValue::vec_from_network(v),
                Err(e) => return Err(eyre!("Party 0 cannot receive bit shares from party 1: {e}")),
            }?;

            // drop timer
            drop(_tt_net);

            // Convert Vec<NetworkValue> to Vec<RingElement<Bit>>
            let f_x_0_bits: Vec<RingElement<Bit>> = f_x_0_bits_net
                .into_iter()
                .map(|nv| match nv {
                    NetworkValue::RingElementBit(b) => Ok(b),
                    other => Err(eyre!("expected RingElementBit, got {:?}", other)),
                })
                .collect::<Result<_, _>>()?;

            // Return a vector of Share<Bit> where the a is from f_x_0_bits
            // and the b is from f_x_1_bits
            let shares: Vec<Share<Bit>> = f_x_0_bits
                .into_iter()
                .zip(f_x_1_bits)
                .map(|(a, b)| Share { a, b })
                .collect();
            Ok(shares)
        }
        2 => {
            // Setting up the Interval Containment function
            // we need this to handle signed numbers, if input is unsigned no need to add N/2
            let n_half = InG::from(1u32 << 31);
            let keys: Vec<[u8; 16]> = vec![[0u8; 16]; 4];

            // make the interval so that we return 1 when MSB == 1
            // this is (our number + n/2 ) % n, modulo is handled by U32Group
            let p = InG::from(1u32 << 31) + n_half;
            let q = InG::from(u32::MAX) + n_half; // modulo is handled by U32Group
                                                  // println!("Interval is p={p:?}, q={q:?}");

            let mut k_fss_0_vec_flat = Vec::with_capacity(batch_size); // to store the fss keys
            let mut k_fss_1_vec_flat = Vec::with_capacity(batch_size);

            //metrics: measure the genkeys time
            let _tt_gen = crate::perf_scoped_for_party!(
                "fss.dealer.genkeys",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            for _i in 0..batch_size {
                // Draw r1 + r2 (aka r_in)
                let (_r2, _r1) = session.prf.gen_rands::<RingElement<u32>>().clone();
                let r2 = RingElement(0);
                let r1 = RingElement(0);

                let r1_plus_r2_u32: u32 = (r1 + r2).convert();
                // Defining the function f using r_in
                let f = IntvFn {
                    r_in: InG::from(r1_plus_r2_u32), //rin = r1+r2
                    r_out: OutG::from(0u128),        // rout=0
                };
                // now we can call gen to generate the FSS keys for each party
                let prg =
                    Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&std::array::from_fn(|i| &keys[i]));
                let icf = Icf::new(p, q, prg);
                let (k_fss_0_pre_ser, k_fss_1_pre_ser): (IcShare, IcShare) = {
                    let mut rng = rand::thread_rng();
                    icf.gen(f, &mut rng)
                };

                let temp_key0 = k_fss_0_pre_ser.serialize()?;
                k_fss_0_vec_flat.extend(RingElement::<u32>::convert_vec_rev(temp_key0.clone()));

                let temp_key1 = k_fss_1_pre_ser.serialize()?;
                k_fss_1_vec_flat.extend(RingElement::<u32>::convert_vec_rev(temp_key1.clone()));
            }

            // drop timer
            drop(_tt_gen);

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.dealer.send_P0",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Send the flattened FSS keys to parties 0 and 1, so they can do Eval
            session
                .network_session
                .send_next(NetworkInt::new_network_vec(k_fss_0_vec_flat))
                .await?; //next is party 0

            // drop timer
            drop(_tt_net);

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.dealer.send_P1",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session
                .network_session
                .send_prev(NetworkInt::new_network_vec(k_fss_1_vec_flat))
                .await?; //previous is party 1

            // drop timer
            drop(_tt_net);

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.dealer.recv_P0",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Receive bit of share from party 0
            let f_x_0_bits_net = match session.network_session.receive_next().await {
                Ok(v) => NetworkValue::vec_from_network(v),
                Err(e) => return Err(eyre!("Party 0 cannot receive bit shares from party 1: {e}")),
            }?;

            // drop timer
            drop(_tt_net);

            // Convert Vec<NetworkValue> to Vec<RingElement<Bit>>
            let f_x_0_bits: Vec<RingElement<Bit>> = f_x_0_bits_net
                .into_iter()
                .map(|nv| match nv {
                    NetworkValue::RingElementBit(b) => Ok(b),
                    other => Err(eyre!("expected RingElementBit, got {:?}", other)),
                })
                .collect::<Result<_, _>>()?;

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.dealer.recv_P1",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Receive Bits of share of party 1 --> this is a vec of network values
            let f_x_1_bits_net = match session.network_session.receive_prev().await {
                Ok(v) => NetworkValue::vec_from_network(v),
                Err(e) => return Err(eyre!("Party 0 cannot receive bit shares from party 1: {e}")),
            }?;

            // drop timer
            drop(_tt_net);

            // Convert Vec<NetworkValue> to Vec<RingElement<Bit>>
            let f_x_1_bits: Vec<RingElement<Bit>> = f_x_1_bits_net
                .into_iter()
                .map(|nv| match nv {
                    NetworkValue::RingElementBit(b) => Ok(b),
                    other => Err(eyre!("expected RingElementBit, got {:?}", other)),
                })
                .collect::<Result<_, _>>()?;

            // Return a vector of Share<Bit> where the a is from f_x_0_bits
            // and the b is from f_x_1_bits
            let shares: Vec<Share<Bit>> = f_x_0_bits
                .into_iter()
                .zip(f_x_1_bits)
                .map(|(a, b)| Share { a, b })
                .collect();
            Ok(shares)
        }
        _ => {
            // this is not a valid party number
            Err(eyre!("Party no is invalid for FSS."))
        }
    }
}
