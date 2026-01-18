use super::BatchQuery;
use crate::{
    dot::{
        distance_comparator::DistanceComparator,
        share_db::{preprocess_query, DBChunkBuffers, ShareDB, SlicedProcessedDatabase},
        PartialResultsWithRotations, IRIS_CODE_LENGTH, MASK_CODE_LENGTH, ROTATIONS,
    },
    helpers::{
        self,
        comm::NcclComm,
        device_manager::DeviceManager,
        htod_on_stream_sync,
        query_processor::{
            CompactQuery, CudaVec2DSlicerRawPointer, DeviceCompactQuery, DeviceCompactSums,
        },
    },
    server::{
        anon_stats::{
            CpuDistanceShare, DistanceCache, OneSidedDistanceCache, TwoSidedDistanceCache,
        },
        PreprocessedBatchQuery,
    },
    threshold_ring::protocol::{ChunkShare, ChunkShareView, Circuits},
};
use ampc_anon_stats::types::Eye;
use ampc_anon_stats::{
    AnonStatsContext, AnonStatsOperation, AnonStatsOrientation, AnonStatsOrigin, AnonStatsStore,
};
use cudarc::{
    cublas::CudaBlas,
    driver::{
        result::{self, event::elapsed, mem_get_info, stream},
        sys::CUevent,
        CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice,
    },
};
use eyre::{bail, eyre, Result};
use futures::{Future, FutureExt};
use iris_mpc_common::galois_engine::degree4::{
    GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare,
};
use iris_mpc_common::{
    helpers::{
        inmemory_store::InMemoryStore,
        sha256::sha256_bytes,
        smpc_request::{REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE},
    },
    iris_db::get_dummy_shares_for_deletion,
    job::{JobSubmissionHandle, ServerJobResult},
    vector_id::VectorId,
};
use iris_mpc_cpu::shares::{
    share::{DistanceShare, Share},
    RingElement,
};
use itertools::{izip, Itertools};
use ring::hkdf::{Algorithm, Okm, Salt, HKDF_SHA256};
use std::{
    collections::{HashMap, HashSet},
    convert::TryFrom,
    fmt, mem,
    sync::Arc,
    time::Instant,
};
use tokio::runtime::Handle;
use tokio::sync::{mpsc, oneshot};

macro_rules! record_stream_time {
    ($manager:expr, $streams:expr, $map:expr, $label:expr, $enable_timing:expr, $block:block) => {{
        if $enable_timing {
            let evt0 = $manager.create_events();
            let evt1 = $manager.create_events();
            $manager.record_event($streams, &evt0);
            let res = $block;
            $manager.record_event($streams, &evt1);
            $map.entry($label).or_default().extend(vec![evt0, evt1]);
            res
        } else {
            $block
        }
    }};
}

#[derive(Debug)]
struct ServerJob {
    pub batch: BatchQuery,
    pub return_channel: oneshot::Sender<ServerJobResult>,
}

#[derive(Debug, Clone)]
pub struct ServerActorHandle {
    job_queue: mpsc::Sender<ServerJob>,
}

impl JobSubmissionHandle for ServerActorHandle {
    type A = ();

    async fn submit_batch_query(
        &mut self,
        batch: BatchQuery,
    ) -> impl Future<Output = Result<ServerJobResult>> {
        let (tx, rx) = oneshot::channel();
        let job = ServerJob {
            batch,
            return_channel: tx,
        };
        self.job_queue.send(job).await.unwrap();
        rx.map(|x| Ok(x?))
    }
}

pub(crate) const DB_CHUNK_SIZE: usize = 1 << 15;
const KDF_SALT: &str = "111a1a93518f670e9bb0c2c68888e2beb9406d4c4ed571dc77b801e676ae3091"; // Random 32 byte salt
const SUPERMATCH_THRESHOLD: usize = 4_000;

// Orientation enum to indicate the orientation of the iris code during the batch processing.
// Normal: Normal orientation of the iris code.
// Mirror: Mirrored orientation of the iris code: Used to detect full-face mirror attacks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Orientation {
    Normal,
    Mirror,
}

impl fmt::Display for Orientation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Orientation::Normal => write!(f, "Normal"),
            Orientation::Mirror => write!(f, "Mirror"),
        }
    }
}

pub struct ServerActor {
    job_queue: mpsc::Receiver<ServerJob>,
    pub device_manager: Arc<DeviceManager>,
    party_id: usize,
    // engines
    codes_engine: ShareDB,
    masks_engine: ShareDB,
    batch_codes_engine: ShareDB,
    batch_masks_engine: ShareDB,
    phase2: Circuits,
    phase2_batch: Circuits,
    distance_comparator: DistanceComparator,
    comms: Vec<Arc<NcclComm>>,
    // DB slices
    pub left_code_db_slices: SlicedProcessedDatabase,
    pub left_mask_db_slices: SlicedProcessedDatabase,
    pub right_code_db_slices: SlicedProcessedDatabase,
    pub right_mask_db_slices: SlicedProcessedDatabase,
    streams: Vec<Vec<CudaStream>>,
    cublas_handles: Vec<Vec<CudaBlas>>,
    results: Vec<CudaSlice<u32>>,
    batch_results: Vec<CudaSlice<u32>>,
    final_results: Vec<CudaSlice<u32>>,
    db_match_list_left: Vec<CudaSlice<u64>>,
    db_match_list_right: Vec<CudaSlice<u64>>,
    batch_match_list_left: Vec<CudaSlice<u64>>,
    batch_match_list_right: Vec<CudaSlice<u64>>,
    current_db_sizes: Vec<usize>,
    query_db_size: Vec<usize>,
    max_batch_size: usize,
    max_db_size: usize,
    match_distances_buffer_size: usize,
    match_distances_buffer_size_extra_percent: usize,
    match_distances_2d_buffer_size: usize,
    return_partial_results: bool,
    disable_persistence: bool,
    enable_debug_timing: bool,
    code_chunk_buffers: Vec<DBChunkBuffers>,
    mask_chunk_buffers: Vec<DBChunkBuffers>,
    phase1_events: Vec<Vec<CUevent>>,
    phase2_events: Vec<Vec<CUevent>>,
    // counter that has number of queries that have been processed
    // this is used to determine the "global" query id for the current query for bucket statistics
    internal_batch_counter: u64,
    // Normal orientation buffers
    match_distances_buffer: DistanceCache,
    // Mirror orientation buffers
    match_distances_buffer_mirror: DistanceCache,
    // 2D anon stats buffer
    both_side_match_distances_buffer: Vec<TwoSidedDistanceCache>,
    full_scan_side: Eye,
    full_scan_side_switching_enabled: bool,
    // Per-batch flag propagated from SQS to disable anonymized statistics collection and computation
    anon_stats_writer: Option<AnonStatsWriter>,
}

const NON_MATCH_ID: u32 = u32::MAX;

type DistanceBundle1D = Vec<DistanceShare<u16>>;
type DistanceBundle2D = (DistanceBundle1D, DistanceBundle1D);

struct AnonStatsWriter {
    store: AnonStatsStore,
    runtime: Handle,
}

impl AnonStatsWriter {
    fn new(store: AnonStatsStore, runtime: Handle) -> Self {
        Self { store, runtime }
    }

    fn insert_1d(
        &self,
        origin: AnonStatsOrigin,
        operation: AnonStatsOperation,
        data: Vec<(i64, DistanceBundle1D)>,
    ) {
        let store = self.store.clone();
        self.runtime.spawn(async move {
            if let Err(err) = store
                .insert_anon_stats_batch_1d(&data, origin, operation)
                .await
            {
                tracing::warn!(?err, ?origin, "Failed to persist 1D anon stats batch");
            }
        });
    }

    fn insert_2d(
        &self,
        origin: AnonStatsOrigin,
        operation: AnonStatsOperation,
        data: Vec<(i64, DistanceBundle2D)>,
    ) {
        let store = self.store.clone();
        self.runtime.spawn(async move {
            if let Err(err) = store
                .insert_anon_stats_batch_2d(&data, origin, operation)
                .await
            {
                tracing::warn!(?err, ?origin, "Failed to persist 2D anon stats batch");
            }
        });
    }
}

impl ServerActor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        party_id: usize,
        chacha_seeds: ([u32; 8], [u32; 8]),
        job_queue_size: usize,
        max_db_size: usize,
        max_batch_size: usize,
        match_distances_buffer_size: usize,
        match_distances_buffer_size_extra_percent: usize,
        match_distances_2d_buffer_size: usize,
        return_partial_results: bool,
        disable_persistence: bool,
        enable_debug_timing: bool,
        full_scan_side: Eye,
        full_scan_side_switching_enabled: bool,
        anon_stats_writer: Option<(AnonStatsStore, Handle)>,
    ) -> Result<(Self, ServerActorHandle)> {
        tracing::info!("GPU Actor: Starting Device Manager");
        let device_manager = Arc::new(DeviceManager::init());
        Self::new_with_device_manager(
            party_id,
            chacha_seeds,
            device_manager,
            job_queue_size,
            max_db_size,
            max_batch_size,
            match_distances_buffer_size,
            match_distances_buffer_size_extra_percent,
            match_distances_2d_buffer_size,
            return_partial_results,
            disable_persistence,
            enable_debug_timing,
            full_scan_side,
            full_scan_side_switching_enabled,
            anon_stats_writer,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_device_manager(
        party_id: usize,
        chacha_seeds: ([u32; 8], [u32; 8]),
        device_manager: Arc<DeviceManager>,
        job_queue_size: usize,
        max_db_size: usize,
        max_batch_size: usize,
        match_distances_buffer_size: usize,
        match_distances_buffer_size_extra_percent: usize,
        match_distances_2d_buffer_size: usize,
        return_partial_results: bool,
        disable_persistence: bool,
        enable_debug_timing: bool,
        full_scan_side: Eye,
        full_scan_side_switching_enabled: bool,
        anon_stats_writer: Option<(AnonStatsStore, Handle)>,
    ) -> Result<(Self, ServerActorHandle)> {
        tracing::info!("GPU Actor: Initializing NCCL");
        let ids = device_manager.get_ids_from_magic(0);
        let comms = device_manager.instantiate_network_from_ids(party_id, &ids)?;
        Self::new_with_device_manager_and_comms(
            party_id,
            chacha_seeds,
            device_manager,
            comms,
            job_queue_size,
            max_db_size,
            max_batch_size,
            match_distances_buffer_size,
            match_distances_buffer_size_extra_percent,
            match_distances_2d_buffer_size,
            return_partial_results,
            disable_persistence,
            enable_debug_timing,
            full_scan_side,
            full_scan_side_switching_enabled,
            anon_stats_writer,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_device_manager_and_comms(
        party_id: usize,
        chacha_seeds: ([u32; 8], [u32; 8]),
        device_manager: Arc<DeviceManager>,
        comms: Vec<Arc<NcclComm>>,
        job_queue_size: usize,
        max_db_size: usize,
        max_batch_size: usize,
        match_distances_buffer_size: usize,
        match_distances_buffer_size_extra_percent: usize,
        match_distances_2d_buffer_size: usize,
        return_partial_results: bool,
        disable_persistence: bool,
        enable_debug_timing: bool,
        full_scan_side: Eye,
        full_scan_side_switching_enabled: bool,
        anon_stats_writer: Option<(AnonStatsStore, Handle)>,
    ) -> Result<(Self, ServerActorHandle)> {
        let (tx, rx) = mpsc::channel(job_queue_size);
        let actor = Self::init(
            party_id,
            chacha_seeds,
            device_manager,
            comms,
            rx,
            max_db_size,
            max_batch_size,
            match_distances_buffer_size,
            match_distances_buffer_size_extra_percent,
            match_distances_2d_buffer_size,
            return_partial_results,
            disable_persistence,
            enable_debug_timing,
            full_scan_side,
            full_scan_side_switching_enabled,
            anon_stats_writer,
        )?;
        Ok((actor, ServerActorHandle { job_queue: tx }))
    }

    #[allow(clippy::too_many_arguments)]
    fn init(
        party_id: usize,
        chacha_seeds: ([u32; 8], [u32; 8]),
        device_manager: Arc<DeviceManager>,
        comms: Vec<Arc<NcclComm>>,
        job_queue: mpsc::Receiver<ServerJob>,
        max_db_size: usize,
        max_batch_size: usize,
        match_distances_buffer_size: usize,
        match_distances_buffer_size_extra_percent: usize,
        match_distances_2d_buffer_size: usize,
        return_partial_results: bool,
        disable_persistence: bool,
        enable_debug_timing: bool,
        full_scan_side: Eye,
        full_scan_side_switching_enabled: bool,
        anon_stats_writer: Option<(AnonStatsStore, Handle)>,
    ) -> Result<Self> {
        let anon_stats_writer =
            anon_stats_writer.map(|(store, runtime)| AnonStatsWriter::new(store, runtime));
        assert_ne!(max_batch_size, 0);
        let mut kdf_nonce = 0;
        let kdf_salt: Salt = Salt::new(HKDF_SHA256, &hex::decode(KDF_SALT)?);
        let n_queries = max_batch_size * ROTATIONS;

        // helper closure to generate the next chacha seeds
        let mut next_chacha_seeds = |seeds: ([u32; 8], [u32; 8])| -> Result<([u32; 8], [u32; 8])> {
            let nonce = kdf_nonce;
            kdf_nonce += 1;
            Ok((
                derive_seed(seeds.0, &kdf_salt, nonce)?,
                derive_seed(seeds.1, &kdf_salt, nonce)?,
            ))
        };

        tracing::info!("GPU actor: Starting engines...");

        // Phase 1 Setup
        let codes_engine = ShareDB::init(
            party_id,
            device_manager.clone(),
            DB_CHUNK_SIZE,
            n_queries,
            IRIS_CODE_LENGTH,
            next_chacha_seeds(chacha_seeds)?,
            comms.clone(),
        );

        let masks_engine = ShareDB::init(
            party_id,
            device_manager.clone(),
            DB_CHUNK_SIZE,
            n_queries,
            MASK_CODE_LENGTH,
            next_chacha_seeds(chacha_seeds)?,
            comms.clone(),
        );

        let now = Instant::now();

        let left_code_db_slices = codes_engine.alloc_db(max_db_size);
        let left_mask_db_slices = masks_engine.alloc_db(max_db_size);
        let right_code_db_slices = codes_engine.alloc_db(max_db_size);
        let right_mask_db_slices = masks_engine.alloc_db(max_db_size);

        tracing::info!("GPU actor: Allocated db in {:?}", now.elapsed());

        // Engines for inflight queries
        let batch_codes_engine = ShareDB::init(
            party_id,
            device_manager.clone(),
            n_queries,
            n_queries,
            IRIS_CODE_LENGTH,
            next_chacha_seeds(chacha_seeds)?,
            comms.clone(),
        );

        let batch_masks_engine = ShareDB::init(
            party_id,
            device_manager.clone(),
            n_queries,
            n_queries,
            MASK_CODE_LENGTH,
            next_chacha_seeds(chacha_seeds)?,
            comms.clone(),
        );

        // Phase 2 Setup
        let phase2_chunk_size = n_queries * DB_CHUNK_SIZE;

        // Not divided by GPU_COUNT since we do the work on all GPUs for simplicity,
        // also not padded to 2048 since we only require it to be a multiple of 64
        let phase2_batch_chunk_size = n_queries * n_queries;
        assert_eq!(
            phase2_batch_chunk_size % 64,
            0,
            "Phase2 batch chunk size must be a multiple of 64"
        );
        assert_eq!(
            phase2_chunk_size % 64,
            0,
            "Phase2 chunk size must be a multiple of 64"
        );

        let phase2_batch = Circuits::new(
            party_id,
            phase2_batch_chunk_size,
            phase2_batch_chunk_size / 64,
            next_chacha_seeds(chacha_seeds)?,
            device_manager.clone(),
            comms.clone(),
        );

        let phase2 = Circuits::new(
            party_id,
            phase2_chunk_size,
            phase2_chunk_size / 64,
            next_chacha_seeds(chacha_seeds)?,
            device_manager.clone(),
            comms.clone(),
        );

        // NOTE: bucket computation (1D/mirror/2D) has been deprecated and moved to the
        // anon-stats-server. The online GPU pipeline only persists raw distance bundles.

        let distance_comparator =
            DistanceComparator::init(n_queries, max_db_size, device_manager.clone());
        // Prepare streams etc.
        let mut streams = vec![];
        let mut cublas_handles = vec![];
        for _ in 0..2 {
            let tmp_streams = device_manager.fork_streams();
            cublas_handles.push(device_manager.create_cublas(&tmp_streams));
            streams.push(tmp_streams);
        }

        let final_results = distance_comparator.prepare_final_results();
        let results = distance_comparator.prepare_results();
        let batch_results = distance_comparator.prepare_results();

        let db_match_list_left =
            distance_comparator.prepare_db_match_list(max_db_size / device_manager.device_count());
        let db_match_list_right =
            distance_comparator.prepare_db_match_list(max_db_size / device_manager.device_count());
        let batch_match_list_left = distance_comparator.prepare_db_match_list(n_queries);
        let batch_match_list_right = distance_comparator.prepare_db_match_list(n_queries);

        let query_db_size = vec![n_queries; device_manager.device_count()];
        let current_db_sizes = vec![0; device_manager.device_count()];

        let code_chunk_buffers = vec![
            codes_engine.alloc_db_chunk_buffer(DB_CHUNK_SIZE),
            codes_engine.alloc_db_chunk_buffer(DB_CHUNK_SIZE),
        ];
        let mask_chunk_buffers = vec![
            masks_engine.alloc_db_chunk_buffer(DB_CHUNK_SIZE),
            masks_engine.alloc_db_chunk_buffer(DB_CHUNK_SIZE),
        ];

        // Create all needed events
        let phase1_events = vec![device_manager.create_events(); 2];
        let phase2_events = vec![device_manager.create_events(); 2];

        // Buffers and counters for match distribution
        let distance_buffer_len =
            match_distances_buffer_size * (100 + match_distances_buffer_size_extra_percent) / 100;
        let bucket_distance_cache = DistanceCache::init(&device_manager, distance_buffer_len);
        // Mirror orientation buffers
        let bucket_distance_cache_mirror =
            DistanceCache::init(&device_manager, distance_buffer_len);

        for dev in device_manager.devices() {
            dev.synchronize().unwrap();
        }
        tracing::info!("GPU actor: Initialized");

        let both_side_match_distances_buffer =
            vec![TwoSidedDistanceCache::default(); device_manager.device_count()];

        Ok(Self {
            party_id,
            job_queue,
            device_manager,
            codes_engine,
            masks_engine,
            phase2,
            phase2_batch,
            distance_comparator,
            batch_codes_engine,
            batch_masks_engine,
            comms,
            left_code_db_slices,
            left_mask_db_slices,
            right_code_db_slices,
            right_mask_db_slices,
            streams,
            cublas_handles,
            results,
            batch_results,
            final_results,
            current_db_sizes,
            query_db_size,
            db_match_list_left,
            db_match_list_right,
            batch_match_list_left,
            batch_match_list_right,
            max_batch_size,
            max_db_size,
            match_distances_buffer_size,
            match_distances_buffer_size_extra_percent,
            match_distances_2d_buffer_size,
            return_partial_results,
            disable_persistence,
            enable_debug_timing,
            code_chunk_buffers,
            mask_chunk_buffers,
            phase1_events,
            phase2_events,
            internal_batch_counter: 0,
            match_distances_buffer: bucket_distance_cache,
            match_distances_buffer_mirror: bucket_distance_cache_mirror,
            full_scan_side,
            full_scan_side_switching_enabled,
            both_side_match_distances_buffer,
            anon_stats_writer,
        })
    }

    pub fn run(mut self) {
        while let Some(job) = self.job_queue.blocking_recv() {
            let ServerJob {
                batch,
                return_channel,
            } = job;
            let now = Instant::now();
            if batch.full_face_mirror_attacks_detection_enabled {
                tracing::info!("Full face mirror attack detection enabled");
                match self.process_batch_query(batch.clone(), Orientation::Mirror, None) {
                    Ok(mirrored_results) => {
                        match self.process_batch_query(
                            batch,
                            Orientation::Normal,
                            Some(mirrored_results),
                        ) {
                            Ok(combined_results) => {
                                // Send the combined results to the return channel
                                let _ = return_channel.send(combined_results);
                            }
                            Err(e) => {
                                tracing::error!(
                                    "Error processing batch query (normal flow): {:?}",
                                    e
                                );
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Error processing batch query (mirror flow): {:?}", e);
                    }
                }
            } else {
                tracing::info!("Full face mirror attack detection disabled");
                let result = self.process_batch_query(batch, Orientation::Normal, None);
                match result {
                    Ok(results) => {
                        // Send the results to the return channel
                        let _ = return_channel.send(results);
                    }
                    Err(e) => {
                        tracing::error!("Error processing batch query: {:?}", e);
                    }
                }
            }

            tracing::info!(
                "Full batch duration took:  {:?}",
                now.elapsed().as_secs_f64()
            );

            metrics::histogram!("full_batch_duration").record(now.elapsed().as_secs_f64());

            if self.full_scan_side_switching_enabled {
                // Alternate the full scan side for the next batch
                self.full_scan_side = self.full_scan_side.other();
                tracing::info!("Switching full scan side to {}", self.full_scan_side);
            }
        }
        tracing::info!("Server Actor finished due to all job queues being closed");
    }

    fn register_host_memory(&self) {
        let page_lock_ts = Instant::now();
        tracing::info!("Starting page lock");
        self.device_manager.register_host_memory(
            &self.left_code_db_slices,
            self.max_db_size,
            IRIS_CODE_LENGTH,
        );
        self.device_manager.register_host_memory(
            &self.right_code_db_slices,
            self.max_db_size,
            IRIS_CODE_LENGTH,
        );
        tracing::info!("Page locking completed for code slice");
        self.device_manager.register_host_memory(
            &self.left_mask_db_slices,
            self.max_db_size,
            MASK_CODE_LENGTH,
        );
        self.device_manager.register_host_memory(
            &self.right_mask_db_slices,
            self.max_db_size,
            MASK_CODE_LENGTH,
        );
        tracing::info!("Page locking completed for mask slice");
        tracing::info!("Page locking completed in {:?}", page_lock_ts.elapsed());
    }

    fn process_batch_query(
        &mut self,
        batch: BatchQuery,
        orientation: Orientation,
        previous_results: Option<ServerJobResult>,
    ) -> Result<ServerJobResult> {
        let now = Instant::now();
        // we only want to perform deletions and reset updates for the first query
        // this ensures we do not perform the same request twice and enables faster processing
        let is_first_query = previous_results.is_none();

        let batch = PreprocessedBatchQuery::from(batch);
        let mut events: HashMap<&str, Vec<Vec<CUevent>>> = HashMap::new();

        let n_reauths = batch
            .request_types
            .iter()
            .filter(|x| *x == REAUTH_MESSAGE_TYPE)
            .count();
        let n_reset_checks = batch
            .request_types
            .iter()
            .filter(|x| *x == RESET_CHECK_MESSAGE_TYPE)
            .count();
        tracing::info!(
            "Started processing batch: {} uniqueness, {} reauth, {} reset_check, {} reset_update, {} deletion requests",
            batch.request_types.len() - n_reauths - n_reset_checks,
            n_reauths,
            n_reset_checks,
            batch.reset_update_request_ids.len(),
            batch.deletion_requests_indices.len(),
        );

        let mut batch = batch;
        let mut batch_size = batch.left_iris_requests.code.len();
        assert!(batch_size > 0 && batch_size <= self.max_batch_size);
        assert!(
            batch_size == batch.left_iris_requests.mask.len()
                && batch_size == batch.request_ids.len()
                && batch_size == batch.request_types.len()
                && batch_size == batch.metadata.len()
                && batch_size == batch.right_iris_requests.code.len()
                && batch_size == batch.right_iris_requests.mask.len()
                && batch_size * ROTATIONS == batch.left_iris_interpolated_requests.code.len()
                && batch_size * ROTATIONS == batch.left_iris_interpolated_requests.mask.len()
                && batch_size * ROTATIONS == batch.right_iris_interpolated_requests.code.len()
                && batch_size * ROTATIONS == batch.right_iris_interpolated_requests.mask.len()
                && batch_size * ROTATIONS
                    == batch.left_mirrored_iris_interpolated_requests.code.len()
                && batch_size * ROTATIONS
                    == batch.left_mirrored_iris_interpolated_requests.mask.len()
                && batch_size * ROTATIONS
                    == batch.right_mirrored_iris_interpolated_requests.code.len()
                && batch_size * ROTATIONS
                    == batch.right_mirrored_iris_interpolated_requests.mask.len()
                && batch_size * ROTATIONS == batch.left_iris_rotated_requests.code.len()
                && batch_size * ROTATIONS == batch.left_iris_rotated_requests.mask.len()
                && batch_size * ROTATIONS == batch.right_iris_rotated_requests.code.len()
                && batch_size * ROTATIONS == batch.right_iris_rotated_requests.mask.len()
                && batch_size * ROTATIONS
                    == batch.left_iris_interpolated_requests_preprocessed.len()
                && batch_size * ROTATIONS
                    == batch.right_iris_interpolated_requests_preprocessed.len()
                && batch_size * ROTATIONS == batch.left_iris_rotated_requests_preprocessed.len()
                && batch_size * ROTATIONS == batch.right_iris_rotated_requests_preprocessed.len()
                && batch_size * ROTATIONS
                    == batch
                        .left_mirrored_iris_interpolated_requests_preprocessed
                        .len()
                && batch_size * ROTATIONS
                    == batch
                        .right_mirrored_iris_interpolated_requests_preprocessed
                        .len()
                && batch_size == batch.skip_persistence.len(),
            "Query batch sizes mismatch"
        );

        let n_reset_updates = batch.reset_update_request_ids.len();
        assert!(
            n_reset_updates == batch.reset_update_shares.len()
                && n_reset_updates == batch.reset_update_indices.len(),
            "Reset update batch sizes mismatch"
        );

        if (!batch.or_rule_indices.is_empty() || batch.luc_lookback_records > 0)
            && orientation == Orientation::Normal
        {
            assert!(
                (batch.or_rule_indices.len() == batch_size) || (batch.luc_lookback_records > 0)
            );
            let skip_lookback_requests: HashSet<usize> = batch
                .request_types
                .iter()
                .enumerate()
                .filter(|(_, req_type)| {
                    req_type.as_str() == REAUTH_MESSAGE_TYPE
                        || req_type.as_str() == RESET_CHECK_MESSAGE_TYPE
                })
                .map(|(index, _)| index)
                .collect();
            batch.or_rule_indices = generate_luc_records(
                (self.current_db_sizes.iter().sum::<usize>() - 1) as u32,
                batch.or_rule_indices.clone(),
                batch.luc_lookback_records,
                batch_size,
                skip_lookback_requests,
            );
        }

        ///////////////////////////////////////////////////////////////////
        // SYNC BATCH CONTENTS AND FILTER OUT INVALID ENTRIES
        ///////////////////////////////////////////////////////////////////
        let tmp_now = Instant::now();
        tracing::info!("Syncing batch entries");

        // Compute hash of the SNS message ids concatenated + currently used scan side
        let batch_hash = sha256_bytes(format!(
            "{}{}",
            batch.sns_message_ids.join(""),
            self.full_scan_side
        ));
        tracing::info!("Current batch hash: {}", hex::encode(&batch_hash[0..4]));

        let valid_entries =
            self.sync_batch_entries(&batch.valid_entries, self.max_batch_size, &batch_hash)?;
        let valid_entry_idxs = valid_entries.iter().positions(|&x| x).collect::<Vec<_>>();
        if valid_entry_idxs.len() != batch_size {
            tracing::warn!(
                "Batch size reduced from {} to {} due to invalid entries. Valid entries: {:?}",
                batch_size,
                valid_entry_idxs.len(),
                valid_entry_idxs,
            );
        }
        batch_size = valid_entry_idxs.len();
        batch.retain(&valid_entry_idxs);
        let batch_operations = batch
            .request_types
            .iter()
            .map(|request_type| {
                if request_type.as_str() == REAUTH_MESSAGE_TYPE {
                    AnonStatsOperation::Reauth
                } else {
                    AnonStatsOperation::Uniqueness
                }
            })
            .collect::<Vec<_>>();
        // For each valid query in the batch, grab the reauth target index if it has one
        // Vec of (query_idx, reauth_db_target_idx)
        let batch_reauth_targets = batch
            .request_types
            .iter()
            .zip(batch.request_ids.iter())
            .map(|(request_type, request_id)| {
                if request_type.as_str() == REAUTH_MESSAGE_TYPE {
                    batch.reauth_target_indices.get(request_id).copied()
                } else {
                    None
                }
            })
            .enumerate()
            .filter_map(|(query_idx, reauth_target)| match reauth_target {
                Some(reauth_target_idx) => Some((query_idx as u32, reauth_target_idx)),
                None => None,
            })
            .collect_vec();
        tracing::info!("Sync and filter done in {:?}", tmp_now.elapsed());
        self.internal_batch_counter += 1;

        ///////////////////////////////////////////////////////////////////
        // PERFORM DELETIONS (IF ANY)
        ///////////////////////////////////////////////////////////////////
        if !batch.deletion_requests_indices.is_empty() && is_first_query {
            tracing::info!("Performing deletions");
            // Prepare dummy deletion shares
            let (dummy_code_share, dummy_mask_share) = get_dummy_shares_for_deletion(self.party_id);
            let (dummy_queries, dummy_sums) =
                self.prepare_device_query_for_shares(&dummy_code_share, &dummy_mask_share)?;

            // Overwrite the in-memory db
            for deletion_index in batch.deletion_requests_indices.clone() {
                let device_index = deletion_index % self.device_manager.device_count() as u32;
                let device_db_index = deletion_index / self.device_manager.device_count() as u32;
                if device_db_index as usize >= self.current_db_sizes[device_index as usize] {
                    tracing::warn!(
                        "Deletion index {} is out of bounds for device {}",
                        deletion_index,
                        device_index
                    );
                    continue;
                }
                self.device_manager
                    .device(device_index as usize)
                    .bind_to_thread()
                    .unwrap();
                write_db_at_index(
                    &self.left_code_db_slices,
                    &self.left_mask_db_slices,
                    &self.right_code_db_slices,
                    &self.right_mask_db_slices,
                    &dummy_queries,
                    &dummy_sums,
                    &dummy_queries,
                    &dummy_sums,
                    0,
                    device_db_index as usize,
                    device_index as usize,
                    &self.streams[0],
                );
            }
        }

        ///////////////////////////////////////////////////////////////////
        // PERFORM RESET UPDATES (IF ANY)
        ///////////////////////////////////////////////////////////////////
        if !batch.reset_update_request_ids.is_empty() && is_first_query {
            tracing::info!("Performing reset updates");

            // Overwrite the in-memory db
            for (reset_index, shares) in izip!(
                batch.reset_update_indices.clone(),
                batch.reset_update_shares.clone()
            ) {
                let (queries_left, sums_left) =
                    self.prepare_device_query_for_shares(&shares.code_left, &shares.mask_left)?;
                let (queries_right, sums_right) =
                    self.prepare_device_query_for_shares(&shares.code_right, &shares.mask_right)?;

                let device_index = reset_index % self.device_manager.device_count() as u32;
                let device_db_index = reset_index / self.device_manager.device_count() as u32;
                if device_db_index as usize >= self.current_db_sizes[device_index as usize] {
                    tracing::warn!(
                        "Reset index {} is out of bounds for device {}",
                        reset_index,
                        device_index
                    );
                    continue;
                }
                self.device_manager
                    .device(device_index as usize)
                    .bind_to_thread()
                    .unwrap();
                write_db_at_index(
                    &self.left_code_db_slices,
                    &self.left_mask_db_slices,
                    &self.right_code_db_slices,
                    &self.right_mask_db_slices,
                    &queries_left,
                    &sums_left,
                    &queries_right,
                    &sums_right,
                    0,
                    device_db_index as usize,
                    device_index as usize,
                    &self.streams[0],
                );
            }
        }

        ///////////////////////////////////////////////////////////////////
        // COMPARE FULL SCAN EYE QUERIES
        ///////////////////////////////////////////////////////////////////
        tracing::info!("Comparing {} eye queries", self.full_scan_side);
        // *Query* variant including Lagrange interpolation.
        let compact_query_side1 = CompactQuery {
            code_query: batch
                .get_iris_interpolated_requests_preprocessed(self.full_scan_side, orientation)
                .code
                .clone(),
            mask_query: batch
                .get_iris_interpolated_requests_preprocessed(self.full_scan_side, orientation)
                .mask
                .clone(),
            code_query_insert: batch
                .get_iris_requests_rotated_preprocessed(self.full_scan_side)
                .code
                .clone(),
            mask_query_insert: batch
                .get_iris_requests_rotated_preprocessed(self.full_scan_side)
                .mask
                .clone(),
        };

        let (compact_device_queries_side1, compact_device_sums_side1) = record_stream_time!(
            &self.device_manager,
            &self.streams[0],
            events,
            "query_preprocess",
            self.enable_debug_timing,
            {
                // This needs to be max_batch_size, even though the query can be shorter to have
                // enough padding for GEMM
                let compact_device_queries_side1 = compact_query_side1.htod_transfer(
                    &self.device_manager,
                    &self.streams[0],
                    self.max_batch_size,
                )?;

                let compact_device_sums_side1 = compact_device_queries_side1.query_sums(
                    &self.codes_engine,
                    &self.masks_engine,
                    &self.streams[0],
                    &self.cublas_handles[0],
                )?;

                (compact_device_queries_side1, compact_device_sums_side1)
            }
        );

        tracing::info!(
            "Comparing {} eye queries against DB and self",
            self.full_scan_side
        );

        let (partial_results_with_rotations_side1, one_sided_distance_cache_side1) = self
            .compare_query_against_db_and_self(
                &compact_device_queries_side1,
                &compact_device_sums_side1,
                &mut events,
                self.full_scan_side,
                batch_size,
                orientation,
                &batch_operations,
                &batch_reauth_targets,
            );

        ///////////////////////////////////////////////////////////////////
        // FETCH PARTIAL FULL SCAN PARTIAL RESULTS
        ///////////////////////////////////////////////////////////////////
        tracing::info!("Fetching partial {} results", self.full_scan_side);
        let mut partial_matches_side1 = self.distance_comparator.get_partial_results(
            match self.full_scan_side {
                Eye::Left => &self.db_match_list_left,
                Eye::Right => &self.db_match_list_right,
            },
            &self.current_db_sizes,
            &self.streams[0],
        );

        // also add the OR rule indices to the partial matches
        let or_indices = batch
            .or_rule_indices
            .iter()
            .flatten()
            .copied()
            .unique()
            .collect_vec();

        for or_idx in or_indices {
            let device_idx = or_idx % self.device_manager.device_count() as u32;
            let db_idx = or_idx / self.device_manager.device_count() as u32;
            if db_idx as usize >= self.current_db_sizes[device_idx as usize] {
                tracing::warn!(
                    "OR rule index {} is out of bounds for device {}",
                    or_idx,
                    device_idx
                );
                continue;
            }
            partial_matches_side1[device_idx as usize].push(db_idx);
        }

        ///////////////////////////////////////////////////////////////////
        // COMPARE OTHER EYE QUERIES
        ///////////////////////////////////////////////////////////////////
        let other_side = self.full_scan_side.other();
        tracing::info!("Comparing {} eye queries", other_side);
        // *Query* variant including Lagrange interpolation.
        let compact_query_side2 = CompactQuery {
            code_query: batch
                .get_iris_interpolated_requests_preprocessed(other_side, orientation)
                .code
                .clone(),
            mask_query: batch
                .get_iris_interpolated_requests_preprocessed(other_side, orientation)
                .mask
                .clone(),
            code_query_insert: batch
                .get_iris_requests_rotated_preprocessed(other_side)
                .code
                .clone(),
            mask_query_insert: batch
                .get_iris_requests_rotated_preprocessed(other_side)
                .mask
                .clone(),
        };

        let (compact_device_queries_side2, compact_device_sums_side2) = record_stream_time!(
            &self.device_manager,
            &self.streams[0],
            events,
            "query_preprocess",
            self.enable_debug_timing,
            {
                // This needs to be MAX_BATCH_SIZE, even though the query can be shorter to have
                // enough padding for GEMM
                let compact_device_queries_side2 = compact_query_side2.htod_transfer(
                    &self.device_manager,
                    &self.streams[0],
                    self.max_batch_size,
                )?;

                let compact_device_sums_side2 = compact_device_queries_side2.query_sums(
                    &self.codes_engine,
                    &self.masks_engine,
                    &self.streams[0],
                    &self.cublas_handles[0],
                )?;

                (compact_device_queries_side2, compact_device_sums_side2)
            }
        );

        let (partial_results_with_rotations_side2, one_sided_distance_cache_side2) =
            if partial_matches_side1
                .iter()
                .any(|x| x.len() >= DB_CHUNK_SIZE)
            {
                tracing::warn!(
                    "Partial matches {} too large, doing full match: {} > {}",
                    self.full_scan_side,
                    partial_matches_side1.len(),
                    DB_CHUNK_SIZE
                );

                tracing::info!("Comparing {} eye queries against DB and self", other_side);
                self.compare_query_against_db_and_self(
                    &compact_device_queries_side2,
                    &compact_device_sums_side2,
                    &mut events,
                    other_side,
                    batch_size,
                    orientation,
                    &batch_operations,
                    &batch_reauth_targets,
                )
            } else {
                tracing::info!("Comparing {} eye queries against DB subset", other_side);
                self.compare_query_against_db_subset_and_self(
                    &compact_device_queries_side2,
                    &compact_device_sums_side2,
                    &mut events,
                    other_side,
                    batch_size,
                    &partial_matches_side1,
                    orientation,
                    &batch_operations,
                    &batch_reauth_targets,
                )
            };

        ///////////////////////////////////////////////////////////////////
        // MERGE LEFT & RIGHT results
        ///////////////////////////////////////////////////////////////////

        tracing::info!("Joining both sides");
        // Merge results and fetch matching indices
        // Format: host_results[device_index][query_index]

        // Initialize bitmap with OR rule, if exists
        if !batch.or_rule_indices.is_empty()
            && !batch.or_rule_indices.iter().all(|inner| inner.is_empty())
        {
            assert_eq!(batch.or_rule_indices.len(), batch_size);

            let now = Instant::now();
            tracing::info!("Preparing and allocating OR policy bitmap");
            // Populate the pre-allocated OR policy bitmap with the serial ids
            let host_or_policy_bitmap = prepare_or_policy_bitmap(
                self.max_db_size,
                batch.or_rule_indices.clone(),
                self.max_batch_size,
            );

            let device_or_policy_bitmap =
                self.allocate_or_policy_bitmap(host_or_policy_bitmap.clone());
            tracing::info!("OR policy bitmap prepared in {:?}", now.elapsed());

            self.distance_comparator.join_db_matches_with_bitmaps(
                self.max_db_size,
                &self.db_match_list_left,
                &self.db_match_list_right,
                &self.final_results,
                &self.current_db_sizes,
                &self.streams[0],
                &device_or_policy_bitmap,
            );
        } else {
            self.distance_comparator.join_db_matches(
                &self.db_match_list_left,
                &self.db_match_list_right,
                &self.final_results,
                &self.current_db_sizes,
                &self.streams[0],
            );
        }

        self.distance_comparator.join_batch_matches(
            &self.batch_match_list_left,
            &self.batch_match_list_right,
            &self.final_results,
            &self.streams[0],
        );

        self.device_manager.await_streams(&self.streams[0]);
        self.device_manager.await_streams(&self.streams[1]);

        // Iterate over a list of tracing payloads, and create logs with mappings to
        // payloads Log at least a "start" event using a log with trace.id
        // and parent.trace.id
        for tracing_payload in batch.metadata.iter() {
            tracing::info!(
                node_id = tracing_payload.node_id,
                dd.trace_id = tracing_payload.trace_id,
                dd.span_id = tracing_payload.span_id,
                "Protocol finished",
            );
        }

        // Fetch the final results (blocking)
        let mut host_results = self
            .distance_comparator
            .fetch_final_results(&self.final_results);

        // Truncate the results to the batch size
        host_results.iter_mut().for_each(|x| x.truncate(batch_size));

        // Fetch and truncate the match counters
        let match_counters_devices = self
            .distance_comparator
            .fetch_match_counters(&self.distance_comparator.match_counters)
            .into_iter()
            .map(|x| x[..batch_size].to_vec())
            .collect::<Vec<_>>();

        // Aggregate across devices
        let match_counters =
            match_counters_devices
                .iter()
                .fold(vec![0usize; batch_size], |mut acc, counters| {
                    for (i, &value) in counters.iter().enumerate() {
                        acc[i] += value as usize;
                    }
                    acc
                });

        // Transfer all match ids
        let match_ids = self.distance_comparator.fetch_all_match_ids(
            &match_counters_devices,
            &self.distance_comparator.all_matches,
        );

        // Check if there are more matches than we fetch
        // TODO: In the future we might want to dynamically allocate more memory here
        // and retry.
        for i in 0..match_counters.len() {
            if match_counters[i] > match_ids[i].len() {
                tracing::warn!(
                    "More matches than fetched (actual: {}, fetched: {}).",
                    match_counters[i],
                    match_ids[i].len()
                );
            }
        }

        // Fetch the partial matches
        let (
            partial_match_ids_left,
            partial_match_counters_left,
            partial_match_ids_right,
            partial_match_counters_right,
        ) = if self.return_partial_results {
            // Transfer the partial results to the host
            let partial_match_counters_left = self
                .distance_comparator
                .fetch_match_counters(&self.distance_comparator.match_counters_left)
                .into_iter()
                .map(|x| x[..batch_size].to_vec())
                .collect::<Vec<_>>();
            let partial_match_counters_right = self
                .distance_comparator
                .fetch_match_counters(&self.distance_comparator.match_counters_right)
                .into_iter()
                .map(|x| x[..batch_size].to_vec())
                .collect::<Vec<_>>();

            let partial_results_left = self.distance_comparator.fetch_all_match_ids(
                &partial_match_counters_left,
                &self.distance_comparator.partial_results_left,
            );
            let partial_results_right = self.distance_comparator.fetch_all_match_ids(
                &partial_match_counters_right,
                &self.distance_comparator.partial_results_right,
            );
            (
                partial_results_left,
                partial_match_counters_left,
                partial_results_right,
                partial_match_counters_right,
            )
        } else {
            (vec![], vec![], vec![], vec![])
        };

        let partial_match_counters_left = partial_match_counters_left.iter().fold(
            vec![0usize; batch_size],
            |mut acc, counters| {
                for (i, &value) in counters.iter().enumerate() {
                    acc[i] += value as usize;
                }
                acc
            },
        );

        let partial_match_counters_right = partial_match_counters_right.iter().fold(
            vec![0usize; batch_size],
            |mut acc, counters| {
                for (i, &value) in counters.iter().enumerate() {
                    acc[i] += value as usize;
                }
                acc
            },
        );

        // Gather rotation indices of db matches for each query
        // Format: partial_match_rotation_indices[query_index][db_id][rotation_index]
        let (partial_match_rotation_indices_left, partial_match_rotation_indices_right) =
            if self.return_partial_results {
                let partial_match_rotation_indices_full_scan = self
                    .map_rotation_indices_to_db_ids_per_query(
                        partial_results_with_rotations_side1,
                        match self.full_scan_side {
                            Eye::Left => &partial_match_ids_left,
                            Eye::Right => &partial_match_ids_right,
                        },
                        batch_size,
                    );
                let partial_match_rotation_indices_other = self
                    .map_rotation_indices_to_db_ids_per_query(
                        partial_results_with_rotations_side2,
                        match self.full_scan_side {
                            Eye::Left => &partial_match_ids_right,
                            Eye::Right => &partial_match_ids_left,
                        },
                        batch_size,
                    );

                match self.full_scan_side {
                    Eye::Left => (
                        partial_match_rotation_indices_full_scan,
                        partial_match_rotation_indices_other,
                    ),
                    Eye::Right => (
                        partial_match_rotation_indices_other,
                        partial_match_rotation_indices_full_scan,
                    ),
                }
            } else {
                (vec![vec![]; batch_size], vec![vec![]; batch_size])
            };

        // Evaluate the results across devices
        // Format: merged_results[query_index]
        let mut merged_results =
            get_merged_results(&host_results, self.device_manager.device_count());

        // Check for mirror attack detection when we have previous results and are in normal orientation
        let request_count = batch.request_ids.len();
        let mut full_face_mirror_match_ids: Vec<Vec<u32>> = vec![vec![]; request_count];
        let mut full_face_mirror_partial_match_ids_left: Vec<Vec<u32>> =
            vec![vec![]; request_count];
        let mut full_face_mirror_partial_match_ids_right: Vec<Vec<u32>> =
            vec![vec![]; request_count];
        let mut full_face_mirror_partial_match_counters_left: Vec<usize> =
            vec![0usize; request_count];
        let mut full_face_mirror_partial_match_counters_right: Vec<usize> =
            vec![0usize; request_count];
        let full_face_mirror_attack_detected: Vec<bool> =
            if orientation == Orientation::Normal && previous_results.is_some() {
                let mirror_results = previous_results.clone().unwrap();
                let attack_detected = (0..request_count)
                    .map(|i| {
                        // Here we check that the normal merged result is non-match while the mirrored merged result shows a match.
                        merged_results[i] == NON_MATCH_ID
                            && mirror_results.matches_with_skip_persistence[i]
                            // Ensures that mirror attack detection is only applied to uniqueness requests.
                            // This constraint is necessary due to the implementation of the `matches` and
                            // `matches_with_skip_persistence` vectors:
                            // 1. The `matches` vector is initialized by the `calculate_insertion_indices()` function
                            //    with all elements set to `true` by default.
                            // 2. During iteration over the `uniqueness_insertion_list`, only elements corresponding
                            //    to unique requests have their value set to `false` in the `matches` vector.
                            // 3. Consequently, non-uniqueness requests (reauth, reset, deletion) retain their
                            //    initial `true` value in the `matches` vector, which would incorrectly cause
                            //    the mirror attack detection algorithm to classify them as full face mirror attacks.
                            && batch.request_types[i] == UNIQUENESS_MESSAGE_TYPE
                    })
                    .collect();
                full_face_mirror_match_ids = mirror_results.match_ids;
                full_face_mirror_partial_match_ids_left = mirror_results.partial_match_ids_left;
                full_face_mirror_partial_match_ids_right = mirror_results.partial_match_ids_right;
                full_face_mirror_partial_match_counters_left =
                    mirror_results.partial_match_counters_left;
                full_face_mirror_partial_match_counters_right =
                    mirror_results.partial_match_counters_right;

                // Edge case: when both normal and mirrored matches occur
                // This happens when both merged_results and mirror_results.merged_results
                // indicate a match (i.e. not equal to NON_MATCH_ID).
                let both_matched_count = (0..request_count)
                    .filter(|&i| {
                        merged_results[i] != NON_MATCH_ID
                            && mirror_results.matches_with_skip_persistence[i]
                    })
                    .count();

                // Log and count the edge cases if any are detected
                if both_matched_count > 0 {
                    tracing::info!(
                        "Detected {} cases where both normal and mirrored matches occurred",
                        both_matched_count
                    );
                    metrics::counter!("mirror.attack.both_matched")
                        .increment(both_matched_count as u64);
                }

                attack_detected
            } else {
                vec![false; request_count]
            };

        // sync the results across nodes, since these are the ones which the insertions are based upon
        self.sync_match_results(self.max_batch_size, &merged_results)?;

        // List the indices of the uniqueness requests that did not match as well as the
        // skipped requests that did not match We do not insert the skipped
        // requests into the DB
        // Full face mirror attack detection additional check:
        // This is being executed for both mirrored and normal mode. In the second run(normal mode) we have the previous results(mirror_results) available
        // We only add entries in the uniqueness_insertion_list if they did not match in the mirror orientation as well
        let (uniqueness_insertion_list, skipped_unique_insertions): (Vec<_>, Vec<_>) =
            merged_results
                .iter()
                .enumerate()
                .filter(|&(idx, &num)| {
                    // Basic condition: must be a uniqueness request, with no match, and below supermatcher threshold
                    let basic_condition = batch.request_types[idx] == UNIQUENESS_MESSAGE_TYPE
                        && num == NON_MATCH_ID
                        && partial_match_counters_left[idx] <= SUPERMATCH_THRESHOLD
                        && partial_match_counters_right[idx] <= SUPERMATCH_THRESHOLD;

                    // When in normal mode and we have mirrored results, only consider that
                    // the entry was unique if it did not match in the mirror orientation as well
                    match (orientation, &previous_results) {
                        (Orientation::Normal, Some(mirror_results)) => {
                            basic_condition && !mirror_results.matches_with_skip_persistence[idx]
                        }
                        _ => basic_condition, // In mirror mode or without previous results
                    }
                })
                .map(|(idx, _num)| idx)
                .partition(|&idx| !batch.skip_persistence[idx]);

        // Spread the insertions across devices.
        let uniqueness_insertion_list =
            distribute_insertions(&uniqueness_insertion_list, &self.current_db_sizes);

        // Calculate the new indices for the inserted uniqueness queries
        let matches = calculate_insertion_indices(
            &mut merged_results,
            &uniqueness_insertion_list,
            &self.current_db_sizes,
            batch_size,
        );

        // create a seperate matches list that includes the matches for skip persistence
        let mut matches_with_skip_persistence = matches.clone();
        skipped_unique_insertions.iter().for_each(|&idx| {
            matches_with_skip_persistence[idx] = false;
            tracing::info!(
                "Matches with skip insertion request ID {}",
                batch.request_ids[idx],
            );
        });

        // Check for batch matches
        let matched_batch_request_ids = match_ids
            .iter()
            .map(|ids| {
                ids.iter()
                    .filter(|&&x| x > (u32::MAX - batch_size as u32)) // ignore matches outside the batch size (dummy matches)
                    .map(|&x| batch.request_ids[(u32::MAX - x) as usize].clone())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let match_ids_filtered = match_ids
            .into_iter()
            .map(|ids| {
                ids.into_iter()
                    .filter(|&x| x <= (u32::MAX - self.max_batch_size as u32))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let successful_reauths = match_ids_filtered
            .iter()
            .enumerate()
            .map(|(idx, matches)| {
                if batch.request_types[idx] != REAUTH_MESSAGE_TYPE {
                    return false;
                }
                let reauth_id = batch.request_ids[idx].clone();
                // Expect a match with target reauth index
                matches.contains(batch.reauth_target_indices.get(&reauth_id).unwrap())
            })
            .collect::<Vec<bool>>();
        let mut reauth_updates_per_device = vec![vec![]; self.device_manager.device_count()];

        // reauth_updates_per_device only used in the normal orientation during the db write, we can skip in mirrored orientation
        if orientation == Orientation::Normal {
            for (reauth_pos, success) in successful_reauths.clone().iter().enumerate() {
                if !*success {
                    continue;
                }
                let reauth_id = batch.request_ids[reauth_pos].clone();
                let reauth_index = *batch.reauth_target_indices.get(&reauth_id).unwrap();
                let device_index = reauth_index % self.device_manager.device_count() as u32;
                reauth_updates_per_device[device_index as usize].push(reauth_pos);
            }
        }

        // Write back to in-memory db - only in normal mode
        let previous_total_db_size = self.current_db_sizes.iter().sum::<usize>();
        let n_insertions = uniqueness_insertion_list
            .iter()
            .map(|x| x.len())
            .sum::<usize>();

        // Check if we actually have space left to write the new entries - only relevant for normal mode
        if orientation == Orientation::Normal
            && previous_total_db_size + n_insertions > self.max_db_size
        {
            tracing::error!(
                "Cannot write new entries, since DB size would be exceeded, current: {}, batch \
                 insertions: {}, max: {}",
                previous_total_db_size,
                n_insertions,
                self.max_db_size
            );
            eyre::bail!("DB size exceeded");
        }

        // Only persist in normal mode, never in mirror mode
        if orientation == Orientation::Mirror {
            tracing::info!("Mirror mode - not persisting results to in-memory DB");
        } else if self.disable_persistence {
            tracing::info!("Persistence is disabled, not writing to DB");
        } else {
            record_stream_time!(
                &self.device_manager,
                &self.streams[0],
                events,
                "db_write",
                self.enable_debug_timing,
                {
                    for i in 0..self.device_manager.device_count() {
                        self.device_manager.device(i).bind_to_thread().unwrap();
                        for insertion_idx in uniqueness_insertion_list[i].clone() {
                            write_db_at_index(
                                match self.full_scan_side {
                                    Eye::Left => &self.left_code_db_slices,
                                    Eye::Right => &self.right_code_db_slices,
                                },
                                match self.full_scan_side {
                                    Eye::Left => &self.left_mask_db_slices,
                                    Eye::Right => &self.right_mask_db_slices,
                                },
                                match self.full_scan_side {
                                    Eye::Left => &self.right_code_db_slices,
                                    Eye::Right => &self.left_code_db_slices,
                                },
                                match self.full_scan_side {
                                    Eye::Left => &self.right_mask_db_slices,
                                    Eye::Right => &self.left_mask_db_slices,
                                },
                                &compact_device_queries_side1,
                                &compact_device_sums_side1,
                                &compact_device_queries_side2,
                                &compact_device_sums_side2,
                                insertion_idx,
                                self.current_db_sizes[i],
                                i,
                                &self.streams[0],
                            );
                            self.current_db_sizes[i] += 1;
                        }

                        tracing::info!(
                            "Updating DB size on device {}: {:?}",
                            i,
                            self.current_db_sizes[i]
                        );

                        for reauth_pos in reauth_updates_per_device[i].clone() {
                            let reauth_id = batch.request_ids[reauth_pos].clone();
                            let reauth_index =
                                *batch.reauth_target_indices.get(&reauth_id).unwrap();
                            let device_db_index =
                                reauth_index / self.device_manager.device_count() as u32;
                            tracing::info!(
                                "Writing succesful reauth index {} at device {} to {}",
                                reauth_index,
                                i,
                                device_db_index
                            );
                            if device_db_index as usize >= self.current_db_sizes[i] {
                                tracing::error!(
                                    "Reauth index {} is out of bounds for device {}",
                                    reauth_index,
                                    i
                                );
                                continue;
                            }
                            write_db_at_index(
                                match self.full_scan_side {
                                    Eye::Left => &self.left_code_db_slices,
                                    Eye::Right => &self.right_code_db_slices,
                                },
                                match self.full_scan_side {
                                    Eye::Left => &self.left_mask_db_slices,
                                    Eye::Right => &self.right_mask_db_slices,
                                },
                                match self.full_scan_side {
                                    Eye::Left => &self.right_code_db_slices,
                                    Eye::Right => &self.left_code_db_slices,
                                },
                                match self.full_scan_side {
                                    Eye::Left => &self.right_mask_db_slices,
                                    Eye::Right => &self.left_mask_db_slices,
                                },
                                &compact_device_queries_side1,
                                &compact_device_sums_side1,
                                &compact_device_queries_side2,
                                &compact_device_sums_side2,
                                reauth_pos,
                                device_db_index as usize,
                                i,
                                &self.streams[0],
                            );
                        }
                    }
                }
            );
        }

        // Attempt for 2D anonymized bundle persistence (bucket computation is handled by
        // anon-stats-server).
        let (one_sided_distance_cache_left, one_sided_distance_cache_right) =
            if self.full_scan_side == Eye::Left {
                (
                    one_sided_distance_cache_side1,
                    one_sided_distance_cache_side2,
                )
            } else {
                (
                    one_sided_distance_cache_side2,
                    one_sided_distance_cache_side1,
                )
            };
        let two_sided_match_distances = one_sided_distance_cache_left
            .into_iter()
            .zip(one_sided_distance_cache_right)
            .map(|(left, right)| TwoSidedDistanceCache::merge(left, right))
            .collect::<Vec<_>>();

        for (new, cache) in two_sided_match_distances
            .into_iter()
            .zip(self.both_side_match_distances_buffer.iter_mut())
        {
            cache.extend(new);
        }

        // check if we have enough results to persist a 2D batch
        let match_distance_2d_count = self
            .both_side_match_distances_buffer
            .iter()
            .map(|x| x.len())
            .sum::<usize>();
        tracing::info!(
            "Match distance 2D count: {match_distance_2d_count}/{}",
            self.match_distances_2d_buffer_size
        );
        if match_distance_2d_count >= self.match_distances_2d_buffer_size {
            tracing::info!("Persisting 2D anon stats bundles for both sides");
            let mut both_side_match_distances_buffer =
                vec![TwoSidedDistanceCache::default(); self.device_manager.device_count()];
            mem::swap(
                &mut self.both_side_match_distances_buffer,
                &mut both_side_match_distances_buffer,
            );

            self.persist_two_sided_caches(&both_side_match_distances_buffer);
        }

        // Instead of sending to return_channel, we'll return this at the end
        let result = ServerJobResult {
            merged_results,
            request_ids: batch.request_ids,
            request_types: batch.request_types,
            metadata: batch.metadata,
            matches,
            matches_with_skip_persistence,
            match_ids: match_ids_filtered,
            full_face_mirror_match_ids,
            partial_match_ids_left,
            partial_match_ids_right,
            partial_match_rotation_indices_left,
            partial_match_rotation_indices_right,
            full_face_mirror_partial_match_ids_left,
            full_face_mirror_partial_match_ids_right,
            partial_match_counters_left,
            partial_match_counters_right,
            full_face_mirror_partial_match_counters_left,
            full_face_mirror_partial_match_counters_right,
            left_iris_requests: batch.left_iris_requests,
            right_iris_requests: batch.right_iris_requests,
            deleted_ids: batch.deletion_requests_indices,
            matched_batch_request_ids,
            successful_reauths,
            reauth_target_indices: batch.reauth_target_indices,
            reauth_or_rule_used: batch.reauth_use_or_rule,
            reset_update_indices: batch.reset_update_indices,
            reset_update_request_ids: batch.reset_update_request_ids,
            reset_update_shares: batch.reset_update_shares,
            modifications: batch.modifications,
            actor_data: (),
            full_face_mirror_attack_detected,
        };

        // Reset the results buffers for reuse
        for dst in [
            &self.db_match_list_left,
            &self.db_match_list_right,
            &self.batch_match_list_left,
            &self.batch_match_list_right,
        ] {
            reset_slice(self.device_manager.devices(), dst, 0, &self.streams[0]);
        }

        for dst in [
            &self.distance_comparator.all_matches,
            &self.distance_comparator.match_counters,
            &self.distance_comparator.match_counters_left,
            &self.distance_comparator.match_counters_right,
            &self.distance_comparator.partial_results_left,
            &self.distance_comparator.partial_results_right,
        ] {
            reset_slice(self.device_manager.devices(), dst, 0, &self.streams[0]);
        }

        self.device_manager.await_streams(&self.streams[0]);
        self.device_manager.await_streams(&self.streams[1]);

        // ---- END RESULT PROCESSING ----
        if self.enable_debug_timing {
            log_timers(events);
        }
        let processed_mil_elements_per_second = (self.max_batch_size * previous_total_db_size)
            as f64
            / now.elapsed().as_secs_f64()
            / 1e6;
        tracing::info!(
            "Batch took {:?} [{:.2} Melems/s] (orientation: {:?})",
            now.elapsed(),
            processed_mil_elements_per_second,
            orientation
        );

        metrics::histogram!("batch_duration").record(now.elapsed().as_secs_f64());
        metrics::histogram!("processed_melems_per_second")
            .record(processed_mil_elements_per_second);

        let new_db_size = self.current_db_sizes.iter().sum::<usize>();
        tracing::info!(
            "Old DB size: {}, New DB size: {}",
            previous_total_db_size,
            new_db_size
        );

        metrics::gauge!("db_size").set(new_db_size as f64);
        metrics::gauge!("remaining_db_size").set((self.max_db_size - new_db_size) as f64);
        metrics::gauge!("batch_size").set(batch_size as f64);
        metrics::gauge!("max_batch_size").set(self.max_batch_size as f64);

        // Update GPU memory metrics
        let mut sum_free = 0;
        let mut sum_total = 0;
        for i in 0..self.device_manager.device_count() {
            let device = self.device_manager.device(i);
            unsafe { result::ctx::set_current(*device.cu_primary_ctx()) }.unwrap();
            let (free, total) = mem_get_info()?;
            metrics::gauge!(format!("gpu_memory_free_{}", i)).set(free as f64);
            metrics::gauge!(format!("gpu_memory_total_{}", i)).set(total as f64);
            sum_free += free;
            sum_total += total;
        }
        metrics::gauge!("gpu_memory_free_sum").set(sum_free as f64);
        metrics::gauge!("gpu_memory_total_sum").set(sum_total as f64);

        Ok(result)
    }

    fn compare_query_against_self(
        &mut self,
        compact_device_queries: &DeviceCompactQuery,
        compact_device_sums: &DeviceCompactSums,
        events: &mut HashMap<&str, Vec<Vec<CUevent>>>,
        eye_db: Eye,
    ) {
        let batch_streams = &self.streams[0];
        let batch_cublas = &self.cublas_handles[0];

        let batch_match_bitmap = match eye_db {
            Eye::Left => &self.batch_match_list_left,
            Eye::Right => &self.batch_match_list_right,
        };

        // ---- START BATCH DEDUP ----
        tracing::info!(party_id = self.party_id, "Starting batch deduplication");

        record_stream_time!(
            &self.device_manager,
            batch_streams,
            events,
            "batch_dot",
            self.enable_debug_timing,
            {
                tracing::info!(party_id = self.party_id, "batch_dot start");

                compact_device_queries.compute_dot_products(
                    &mut self.batch_codes_engine,
                    &mut self.batch_masks_engine,
                    &self.query_db_size,
                    0,
                    batch_streams,
                    batch_cublas,
                );
                tracing::info!(party_id = self.party_id, "compute_dot_reducers start");

                compact_device_sums.compute_dot_reducers(
                    &mut self.batch_codes_engine,
                    &mut self.batch_masks_engine,
                    &self.query_db_size,
                    0,
                    batch_streams,
                );
                tracing::info!(party_id = self.party_id, "batch_dot end");
            }
        );

        record_stream_time!(
            &self.device_manager,
            batch_streams,
            events,
            "batch_reshare",
            self.enable_debug_timing,
            {
                tracing::info!(party_id = self.party_id, "batch_reshare start");
                self.batch_codes_engine
                    .reshare_results(&self.query_db_size, batch_streams);
                tracing::info!(party_id = self.party_id, "batch_reshare masks start");
                self.batch_masks_engine
                    .reshare_results(&self.query_db_size, batch_streams);
                tracing::info!(party_id = self.party_id, "batch_reshare end");
            }
        );

        let db_sizes_batch =
            vec![self.max_batch_size * ROTATIONS; self.device_manager.device_count()];
        let code_dots_batch = self.batch_codes_engine.result_chunk_shares(&db_sizes_batch);
        let mask_dots_batch = self.batch_masks_engine.result_chunk_shares(&db_sizes_batch);

        record_stream_time!(
            &self.device_manager,
            batch_streams,
            events,
            "batch_threshold",
            self.enable_debug_timing,
            {
                tracing::info!(party_id = self.party_id, "batch_threshold start");
                self.phase2_batch.compare_threshold_masked_many(
                    &code_dots_batch,
                    &mask_dots_batch,
                    batch_streams,
                );
                tracing::info!(party_id = self.party_id, "batch_threshold end");
            }
        );

        tracing::info!(party_id = self.party_id, "phase2_batch start");

        let res = self.phase2_batch.take_result_buffer();
        let chunk_size = self.phase2_batch.chunk_size();
        open_batch(
            &mut self.phase2_batch,
            &res,
            &self.distance_comparator,
            batch_match_bitmap,
            chunk_size,
            &db_sizes_batch,
            &db_sizes_batch,
            0,
            &db_sizes_batch,
            &vec![false; self.device_manager.device_count()],
            batch_streams,
        );
        self.phase2_batch.return_result_buffer(res);

        tracing::info!(party_id = self.party_id, "Finished batch deduplication");
    }

    #[allow(clippy::too_many_arguments)]
    fn compare_query_against_db_subset_and_self(
        &mut self,
        compact_device_queries: &DeviceCompactQuery,
        compact_device_sums: &DeviceCompactSums,
        events: &mut HashMap<&str, Vec<Vec<CUevent>>>,
        eye_db: Eye,
        batch_size: usize,
        db_subset_idx: &[Vec<u32>],
        orientation: Orientation,
        operations: &[AnonStatsOperation],
        batch_reauth_targets: &[(u32, u32)],
    ) -> (PartialResultsWithRotations, Vec<OneSidedDistanceCache>) {
        let old_distance_cache_counters = match orientation {
            Orientation::Normal => Some(
                self.match_distances_buffer
                    .load_counters(&self.device_manager, eye_db),
            ),
            Orientation::Mirror => {
                None // Do not work on mirror
            }
        };

        // ---- START BATCH DEDUP ----
        self.compare_query_against_self(
            compact_device_queries,
            compact_device_sums,
            events,
            eye_db,
        );
        // ---- END BATCH DEDUP ----

        // if the subset is completely empty, we can skip the whole process after we do the batch check
        if db_subset_idx.iter().all(|x| x.is_empty()) {
            return (
                HashMap::new(),
                vec![OneSidedDistanceCache::default(); self.device_manager.device_count()],
            );
        }

        // which database are we querying against
        let (code_db_slices, mask_db_slices) = match eye_db {
            Eye::Left => (&self.left_code_db_slices, &self.left_mask_db_slices),
            Eye::Right => (&self.right_code_db_slices, &self.right_mask_db_slices),
        };

        // We copied over a subset of the db, so we match against DB chunks of the given sizes
        let chunk_size = db_subset_idx.iter().map(|x| x.len()).collect::<Vec<_>>();
        let dot_chunk_size = chunk_size
            .iter()
            .map(|&s| s.max(1).div_ceil(64) * 64)
            .collect::<Vec<_>>();

        record_stream_time!(
            &self.device_manager,
            &self.streams[0],
            events,
            "prefetch_db_chunk",
            self.enable_debug_timing,
            {
                self.codes_engine.prefetch_db_subset_into_chunk_buffers(
                    code_db_slices,
                    &self.code_chunk_buffers[0],
                    db_subset_idx,
                    &self.streams[0],
                );
                self.masks_engine.prefetch_db_subset_into_chunk_buffers(
                    mask_db_slices,
                    &self.mask_chunk_buffers[0],
                    db_subset_idx,
                    &self.streams[0],
                );
            }
        );

        record_stream_time!(
            &self.device_manager,
            &self.streams[0],
            events,
            "db_dot",
            self.enable_debug_timing,
            {
                compact_device_queries.dot_products_against_db(
                    &mut self.codes_engine,
                    &mut self.masks_engine,
                    &CudaVec2DSlicerRawPointer::from(&self.code_chunk_buffers[0]),
                    &CudaVec2DSlicerRawPointer::from(&self.mask_chunk_buffers[0]),
                    &dot_chunk_size,
                    0,
                    &self.streams[0],
                    &self.cublas_handles[0],
                );
            }
        );

        record_stream_time!(
            &self.device_manager,
            &self.streams[0],
            events,
            "db_reduce",
            self.enable_debug_timing,
            {
                compact_device_sums.compute_dot_reducer_against_prepared_db(
                    &mut self.codes_engine,
                    &mut self.masks_engine,
                    &self.code_chunk_buffers[0].sums,
                    &self.mask_chunk_buffers[0].sums,
                    &dot_chunk_size,
                    &self.streams[0],
                );
            }
        );

        record_stream_time!(
            &self.device_manager,
            &self.streams[0],
            events,
            "db_reshare",
            self.enable_debug_timing,
            {
                self.codes_engine
                    .reshare_results(&dot_chunk_size, &self.streams[0]);
                self.masks_engine
                    .reshare_results(&dot_chunk_size, &self.streams[0]);
            }
        );

        // ---- END PHASE 1 ----

        // ---- START PHASE 2 ----
        let max_chunk_size = dot_chunk_size.iter().max().copied().unwrap();
        let phase_2_chunk_sizes = vec![max_chunk_size; self.device_manager.device_count()];
        let code_dots = self.codes_engine.result_chunk_shares(&phase_2_chunk_sizes);
        let mask_dots = self.masks_engine.result_chunk_shares(&phase_2_chunk_sizes);

        assert_eq!(
            (max_chunk_size * self.max_batch_size * ROTATIONS) % 64,
            0,
            "Phase 2 input size must be a multiple of 64"
        );
        self.phase2
            .set_chunk_size(max_chunk_size * self.max_batch_size * ROTATIONS / 64);

        record_stream_time!(
            &self.device_manager,
            &self.streams[0],
            events,
            "db_threshold",
            self.enable_debug_timing,
            {
                self.phase2
                    .compare_threshold_masked_many(&code_dots, &mask_dots, &self.streams[0]);
            }
        );

        let res = self.phase2.take_result_buffer();

        let db_match_bitmap = match eye_db {
            Eye::Left => &self.db_match_list_left,
            Eye::Right => &self.db_match_list_right,
        };

        let (
            match_distances_buffers_codes,
            match_distances_buffers_masks,
            match_distances_counters,
            match_distances_indices,
        ) = match orientation {
            Orientation::Normal => self.match_distances_buffer.get_buffers(eye_db),
            Orientation::Mirror => self.match_distances_buffer_mirror.get_buffers(eye_db),
        };

        // ignore all device results where the chunk size is 0
        let ignore_device_results: Vec<bool> = chunk_size.iter().map(|&s| s == 0).collect();

        record_stream_time!(
            &self.device_manager,
            &self.streams[0],
            events,
            "db_open",
            self.enable_debug_timing,
            {
                open_subset_results(
                    &mut self.phase2,
                    &res,
                    &self.distance_comparator,
                    db_match_bitmap,
                    max_chunk_size * self.max_batch_size * ROTATIONS / 64,
                    &dot_chunk_size,
                    &chunk_size,
                    &self.current_db_sizes,
                    &ignore_device_results,
                    match_distances_buffers_codes,
                    match_distances_buffers_masks,
                    match_distances_counters,
                    match_distances_indices,
                    self.internal_batch_counter,
                    &code_dots,
                    &mask_dots,
                    batch_size,
                    self.match_distances_buffer_size
                        * (100 + self.match_distances_buffer_size_extra_percent)
                        / 100,
                    &self.streams[0],
                    db_subset_idx,
                );
                self.phase2.return_result_buffer(res);
            }
        );

        // Retrieve partial results
        let partial_results_with_rotations = self
            .distance_comparator
            .get_partial_results_with_rotations(&self.streams[0]);

        // Reset the partial results buffers and counter for re-use
        for dst in [
            &self.distance_comparator.partial_results_query_indices,
            &self.distance_comparator.partial_results_db_indices,
            &self.distance_comparator.partial_match_counter,
        ] {
            reset_slice(self.device_manager.devices(), dst, 0, &self.streams[0]);
        }

        // Reset rotations buffer separately due to different type (i8 vs u32)
        reset_slice(
            self.device_manager.devices(),
            &self.distance_comparator.partial_results_rotations,
            0,
            &self.streams[0],
        );
        let new_partial_match_buffer = match old_distance_cache_counters {
            Some(counters) => self.match_distances_buffer.load_additions_since(
                &self.device_manager,
                eye_db,
                counters,
                self.match_distances_buffer_size
                    * (100 + self.match_distances_buffer_size_extra_percent)
                    / 100,
                &self.streams[0],
                operations,
                self.distance_comparator.query_length as u64,
            ),
            None => {
                vec![OneSidedDistanceCache::default(); self.device_manager.device_count()]
            }
        };

        self.persist_one_sided_caches(eye_db, orientation, &new_partial_match_buffer);
        match orientation {
            Orientation::Normal => {
                self.reset_anon_stats_distance_cache(eye_db, Orientation::Normal)
            }
            Orientation::Mirror => {
                self.reset_anon_stats_distance_cache(eye_db, Orientation::Mirror)
            }
        };
        (partial_results_with_rotations, new_partial_match_buffer)
    }
    #[allow(clippy::too_many_arguments)]
    fn compare_query_against_db_and_self(
        &mut self,
        compact_device_queries: &DeviceCompactQuery,
        compact_device_sums: &DeviceCompactSums,
        events: &mut HashMap<&str, Vec<Vec<CUevent>>>,
        eye_db: Eye,
        batch_size: usize,
        orientation: Orientation,
        operations: &[AnonStatsOperation],
        batch_reauth_targets: &[(u32, u32)],
    ) -> (PartialResultsWithRotations, Vec<OneSidedDistanceCache>) {
        let old_distance_cache_counters = match orientation {
            Orientation::Normal => Some(
                self.match_distances_buffer
                    .load_counters(&self.device_manager, eye_db),
            ),
            Orientation::Mirror => {
                None // Do not work on mirror
            }
        };

        // ---- START BATCH DEDUP ----
        self.compare_query_against_self(
            compact_device_queries,
            compact_device_sums,
            events,
            eye_db,
        );
        // ---- END BATCH DEDUP ----

        // which database are we querying against
        let (code_db_slices, mask_db_slices) = match eye_db {
            Eye::Left => (&self.left_code_db_slices, &self.left_mask_db_slices),
            Eye::Right => (&self.right_code_db_slices, &self.right_mask_db_slices),
        };

        // partial results, left or right depending on the eye that we are running
        // it gets filled by the open_results kernel based on the results from the 3 mpc nodes
        let db_match_bitmap = match eye_db {
            Eye::Left => &self.db_match_list_left,
            Eye::Right => &self.db_match_list_right,
        };

        let chunk_sizes = |chunk_idx: usize| {
            self.current_db_sizes
                .iter()
                .map(|s| (s - DB_CHUNK_SIZE * chunk_idx).clamp(0, DB_CHUNK_SIZE))
                .collect::<Vec<_>>()
        };

        record_stream_time!(
            &self.device_manager,
            &self.streams[0],
            events,
            "prefetch_db_chunk",
            self.enable_debug_timing,
            {
                self.codes_engine.prefetch_db_chunk(
                    code_db_slices,
                    &self.code_chunk_buffers[0],
                    &chunk_sizes(0),
                    &vec![0; self.device_manager.device_count()],
                    &self.current_db_sizes,
                    &self.streams[0],
                );
                self.masks_engine.prefetch_db_chunk(
                    mask_db_slices,
                    &self.mask_chunk_buffers[0],
                    &chunk_sizes(0),
                    &vec![0; self.device_manager.device_count()],
                    &self.current_db_sizes,
                    &self.streams[0],
                );
            }
        );

        // ---- START DATABASE DEDUP ----
        tracing::info!(party_id = self.party_id, "Start DB deduplication");
        let ignore_device_results: Vec<bool> =
            self.current_db_sizes.iter().map(|&s| s == 0).collect();
        let mut db_chunk_idx = 0;
        loop {
            let request_streams = &self.streams[db_chunk_idx % 2];
            let next_request_streams = &self.streams[(db_chunk_idx + 1) % 2];
            let request_cublas_handles = &self.cublas_handles[db_chunk_idx % 2];

            let offset = db_chunk_idx * DB_CHUNK_SIZE;
            let chunk_size = chunk_sizes(db_chunk_idx);
            let next_chunk_size = chunk_sizes(db_chunk_idx + 1);

            // We need to pad the chunk size for two reasons:
            // 1. Chunk size needs to be a multiple of 4, because the underlying
            // `gemm_ex` expects this.
            // 2. We are running into NCCL issues if the bytes sent/received are not a
            //    multiple of 64.
            // We filter out potential "phantom matches" for the padded data in the `open`
            // later.
            let dot_chunk_size = chunk_size
                .iter()
                .map(|&s| s.max(1).div_ceil(64) * 64)
                .collect::<Vec<_>>();

            // First stream doesn't need to wait
            if db_chunk_idx == 0 {
                self.device_manager
                    .record_event(request_streams, &self.phase1_events[db_chunk_idx % 2]);
                self.device_manager
                    .record_event(request_streams, &self.phase2_events[db_chunk_idx % 2]);
            }

            // Prefetch next chunk
            record_stream_time!(
                &self.device_manager,
                next_request_streams,
                events,
                "prefetch_db_chunk",
                self.enable_debug_timing,
                {
                    self.codes_engine.prefetch_db_chunk(
                        code_db_slices,
                        &self.code_chunk_buffers[(db_chunk_idx + 1) % 2],
                        &next_chunk_size,
                        &chunk_size.iter().map(|s| offset + s).collect::<Vec<_>>(),
                        &self.current_db_sizes,
                        next_request_streams,
                    );
                    self.masks_engine.prefetch_db_chunk(
                        mask_db_slices,
                        &self.mask_chunk_buffers[(db_chunk_idx + 1) % 2],
                        &next_chunk_size,
                        &chunk_size.iter().map(|s| offset + s).collect::<Vec<_>>(),
                        &self.current_db_sizes,
                        next_request_streams,
                    );
                }
            );

            self.device_manager
                .await_event(request_streams, &self.phase1_events[db_chunk_idx % 2]);

            // ---- START PHASE 1 ----
            record_stream_time!(
                &self.device_manager,
                request_streams,
                events,
                "db_dot",
                self.enable_debug_timing,
                {
                    compact_device_queries.dot_products_against_db(
                        &mut self.codes_engine,
                        &mut self.masks_engine,
                        &CudaVec2DSlicerRawPointer::from(
                            &self.code_chunk_buffers[db_chunk_idx % 2],
                        ),
                        &CudaVec2DSlicerRawPointer::from(
                            &self.mask_chunk_buffers[db_chunk_idx % 2],
                        ),
                        &dot_chunk_size,
                        0,
                        request_streams,
                        request_cublas_handles,
                    );
                }
            );

            self.device_manager
                .await_event(request_streams, &self.phase2_events[db_chunk_idx % 2]);

            record_stream_time!(
                &self.device_manager,
                request_streams,
                events,
                "db_reduce",
                self.enable_debug_timing,
                {
                    compact_device_sums.compute_dot_reducer_against_db(
                        &mut self.codes_engine,
                        &mut self.masks_engine,
                        code_db_slices,
                        mask_db_slices,
                        &dot_chunk_size,
                        offset,
                        request_streams,
                    );
                }
            );

            self.device_manager
                .record_event(request_streams, &self.phase1_events[(db_chunk_idx + 1) % 2]);

            record_stream_time!(
                &self.device_manager,
                request_streams,
                events,
                "db_reshare",
                self.enable_debug_timing,
                {
                    self.codes_engine
                        .reshare_results(&dot_chunk_size, request_streams);
                    self.masks_engine
                        .reshare_results(&dot_chunk_size, request_streams);
                }
            );

            // ---- END PHASE 1 ----

            // ---- START PHASE 2 ----
            let max_chunk_size = dot_chunk_size.iter().max().copied().unwrap();
            let phase_2_chunk_sizes = vec![max_chunk_size; self.device_manager.device_count()];
            let code_dots = self.codes_engine.result_chunk_shares(&phase_2_chunk_sizes);
            let mask_dots = self.masks_engine.result_chunk_shares(&phase_2_chunk_sizes);
            {
                assert_eq!(
                    (max_chunk_size * self.max_batch_size * ROTATIONS) % 64,
                    0,
                    "Phase 2 input size must be a multiple of 64"
                );
                self.phase2
                    .set_chunk_size(max_chunk_size * self.max_batch_size * ROTATIONS / 64);

                record_stream_time!(
                    &self.device_manager,
                    request_streams,
                    events,
                    "db_threshold",
                    self.enable_debug_timing,
                    {
                        self.phase2.compare_threshold_masked_many(
                            &code_dots,
                            &mask_dots,
                            request_streams,
                        );
                    }
                );

                let res = self.phase2.take_result_buffer();

                let (
                    match_distances_buffers_codes,
                    match_distances_buffers_masks,
                    match_distances_counters,
                    match_distances_indices,
                ) = self.match_distances_buffer.get_buffers(eye_db);

                record_stream_time!(
                    &self.device_manager,
                    request_streams,
                    events,
                    "db_open",
                    self.enable_debug_timing,
                    {
                        open(
                            &mut self.phase2,
                            &res,
                            &self.distance_comparator,
                            db_match_bitmap,
                            max_chunk_size * self.max_batch_size * ROTATIONS / 64,
                            &dot_chunk_size,
                            &chunk_size,
                            offset,
                            &self.current_db_sizes,
                            &ignore_device_results,
                            match_distances_buffers_codes,
                            match_distances_buffers_masks,
                            match_distances_counters,
                            match_distances_indices,
                            self.internal_batch_counter,
                            &code_dots,
                            &mask_dots,
                            batch_size,
                            self.match_distances_buffer_size
                                * (100 + self.match_distances_buffer_size_extra_percent)
                                / 100,
                            request_streams,
                        );
                        self.phase2.return_result_buffer(res);
                    }
                );
            }
            self.device_manager
                .record_event(request_streams, &self.phase2_events[(db_chunk_idx + 1) % 2]);

            // ---- END PHASE 2 ----

            // Increment chunk index
            db_chunk_idx += 1;

            // Break if we reached the end of the database
            if db_chunk_idx * DB_CHUNK_SIZE >= *self.current_db_sizes.iter().max().unwrap() {
                break;
            }
        }
        // ---- END DATABASE DEDUP ----

        // Wait for protocol to finish
        tracing::info!(party_id = self.party_id, "waiting for db search to finish");
        self.device_manager.await_streams(&self.streams[0]);
        self.device_manager.await_streams(&self.streams[1]);
        tracing::info!(party_id = self.party_id, "db search finished");

        // Retrieve partial results with rotations
        let partial_results_with_rotations = self
            .distance_comparator
            .get_partial_results_with_rotations(&self.streams[0]);

        // Reset the partial results buffers and counter for re-use
        for dst in [
            &self.distance_comparator.partial_results_query_indices,
            &self.distance_comparator.partial_results_db_indices,
            &self.distance_comparator.partial_match_counter,
        ] {
            reset_slice(self.device_manager.devices(), dst, 0, &self.streams[0]);
        }

        // Reset rotations buffer separately due to different type (i8 vs u32)
        reset_slice(
            self.device_manager.devices(),
            &self.distance_comparator.partial_results_rotations,
            0,
            &self.streams[0],
        );

        // Reset the results buffers for reuse
        for dst in &[&self.results, &self.batch_results, &self.final_results] {
            reset_slice(self.device_manager.devices(), dst, 0xff, &self.streams[0]);
        }

        let new_partial_match_buffer = match old_distance_cache_counters {
            Some(counters) => self.match_distances_buffer.load_additions_since(
                &self.device_manager,
                eye_db,
                counters,
                self.match_distances_buffer_size
                    * (100 + self.match_distances_buffer_size_extra_percent)
                    / 100,
                &self.streams[0],
                operations,
                self.distance_comparator.query_length as u64,
            ),
            None => {
                vec![OneSidedDistanceCache::default(); self.device_manager.device_count()]
            }
        };

        self.persist_one_sided_caches(eye_db, orientation, &new_partial_match_buffer);
        match orientation {
            Orientation::Normal => {
                self.reset_anon_stats_distance_cache(eye_db, Orientation::Normal)
            }
            Orientation::Mirror => {
                self.reset_anon_stats_distance_cache(eye_db, Orientation::Mirror)
            }
        };
        (partial_results_with_rotations, new_partial_match_buffer)
    }

    fn sync_match_results(&mut self, max_batch_size: usize, match_results: &[u32]) -> Result<()> {
        assert!(match_results.len() <= max_batch_size);
        let mut buffer = self
            .device_manager
            .device(0)
            .alloc_zeros(max_batch_size * self.comms[0].world_size())
            .unwrap();

        let mut host_buffer = vec![0u32; max_batch_size];
        host_buffer[..match_results.len()].copy_from_slice(match_results);

        let buffer_self = self.device_manager.device(0).htod_copy(host_buffer)?;
        self.device_manager.device(0).synchronize()?;
        self.comms[0]
            .all_gather(&buffer_self, &mut buffer)
            .map_err(|e| eyre!(format!("{:?}", e)))?;
        self.device_manager.device(0).synchronize()?;

        let results = self.device_manager.device(0).dtoh_sync_copy(&buffer)?;
        let results: Vec<_> = results
            .chunks_exact(results.len() / self.comms[0].world_size())
            .collect();

        // check that the results are the same on all nodes
        for i in 0..self.comms[0].world_size() {
            if &results[i][..match_results.len()] != match_results {
                tracing::error!(
                    party_id = self.party_id,
                    "Match results mismatch with node {}. MPC protocol produced out of sync results.",
                    i
                );
                metrics::counter!("mpc.mismatch").increment(1);
                bail!(
                    "Match results mismatch with node {}. MPC protocol produced out of sync results.",
                    i
                );
            }
        }
        Ok(())
    }

    fn sync_batch_entries(
        &mut self,
        valid_entries: &[bool],
        max_batch_size: usize,
        batch_hash: &[u8],
    ) -> Result<Vec<bool>> {
        assert!(valid_entries.len() <= max_batch_size);
        let hash_len = batch_hash.len();
        let mut buffer = self
            .device_manager
            .device(0)
            .alloc_zeros((max_batch_size + hash_len) * self.comms[0].world_size())
            .unwrap();

        let mut host_buffer = vec![0u8; max_batch_size + hash_len];
        host_buffer[..valid_entries.len()]
            .copy_from_slice(&valid_entries.iter().map(|&x| x as u8).collect::<Vec<u8>>());
        host_buffer[max_batch_size..].copy_from_slice(batch_hash);

        let buffer_self = self.device_manager.device(0).htod_copy(host_buffer)?;

        // Use all_gather to sync the buffer across all nodes (only using device 0)
        self.device_manager.device(0).synchronize()?;
        self.comms[0]
            .all_gather(&buffer_self, &mut buffer)
            .map_err(|e| eyre!(format!("{:?}", e)))?;
        self.device_manager.device(0).synchronize()?;

        let results = self.device_manager.device(0).dtoh_sync_copy(&buffer)?;
        let results: Vec<_> = results
            .chunks_exact(results.len() / self.comms[0].world_size())
            .collect();

        // Only keep entries that are valid on all nodes
        let mut valid_merged = vec![true; max_batch_size];
        for i in 0..self.comms[0].world_size() {
            for j in 0..max_batch_size {
                valid_merged[j] &= results[i][j] == 1;
            }
        }

        // Check that the hash is the same on nodes
        for i in 0..self.comms[0].world_size() {
            if &results[i][max_batch_size..] != batch_hash {
                tracing::error!(
                    party_id = self.party_id,
                    "Batch mismatch with node {}. Queues seem to be out of sync (check requests and full scan side).",
                    i
                );
                metrics::counter!("batch.mismatch").increment(1);
                bail!(
                    "Batch mismatch with node {}. Queues seem to be out of sync (check requests and full scan side).",
                    i
                );
            }
        }

        Ok(valid_merged)
    }

    fn prepare_device_query_for_shares(
        &self,
        code_share: &GaloisRingIrisCodeShare,
        mask_share: &GaloisRingTrimmedMaskCodeShare,
    ) -> Result<(DeviceCompactQuery, DeviceCompactSums)> {
        let compact_query = {
            let code = preprocess_query(
                &code_share
                    .all_rotations()
                    .into_iter()
                    .flat_map(|e| e.coefs)
                    .collect::<Vec<_>>(),
            );
            let mask = preprocess_query(
                &mask_share
                    .all_rotations()
                    .into_iter()
                    .flat_map(|e| e.coefs)
                    .collect::<Vec<_>>(),
            );
            CompactQuery {
                code_query: code.clone(),
                mask_query: mask.clone(),
                code_query_insert: code,
                mask_query_insert: mask,
            }
        };
        let compact_device_queries = compact_query.htod_transfer(
            &self.device_manager,
            &self.streams[0],
            self.max_batch_size,
        )?;

        let compact_device_sums = compact_device_queries.query_sums(
            &self.codes_engine,
            &self.masks_engine,
            &self.streams[0],
            &self.cublas_handles[0],
        )?;

        Ok((compact_device_queries, compact_device_sums))
    }

    fn allocate_or_policy_bitmap(&mut self, bitmap: Vec<u64>) -> Vec<CudaSlice<u64>> {
        let devices = self.device_manager.devices();

        let mut or_policy_bitmap = Vec::with_capacity(devices.len());

        for (device_idx, dev) in devices.iter().enumerate() {
            // Transfer the bitmap to the device. It will be the same for each of the
            // devices
            let _bitmap = htod_on_stream_sync(&bitmap, dev, &self.streams[0][device_idx]).unwrap();
            or_policy_bitmap.push(_bitmap);
        }
        or_policy_bitmap
    }

    fn map_rotation_indices_to_db_ids_per_query(
        &self,
        partial_results_with_rotations: PartialResultsWithRotations,
        partial_match_ids: &[Vec<u32>],
        batch_size: usize,
    ) -> Vec<Vec<Vec<i8>>> {
        let mut partial_match_rotation_indices = vec![vec![]; batch_size];

        for query_idx in 0..batch_size {
            if let Some(db_matches_with_rotations) =
                partial_results_with_rotations.get(&(query_idx as u32))
            {
                let match_ids = &partial_match_ids[query_idx];

                // Create rotation indices mapped to match IDs
                for &db_id in match_ids {
                    if let Some(rotations) = db_matches_with_rotations.get(&db_id) {
                        partial_match_rotation_indices[query_idx].push(rotations.clone());
                    }
                }
            }
        }

        partial_match_rotation_indices
    }
}

/// Internal helper function to log the timers of measured cuda streams.
fn log_timers(events: HashMap<&str, Vec<Vec<CUevent>>>) {
    for (name, event_vecs) in &events {
        let duration: f32 = event_vecs
            .chunks(2)
            .map(|pair| {
                // Calculate the average duration per device
                let (start_events, end_events) = (&pair[0], &pair[1]);
                let total_duration: f32 = start_events
                    .iter()
                    .zip(end_events.iter())
                    .map(|(start, end)| unsafe { elapsed(*start, *end) }.unwrap())
                    .sum();

                total_duration / start_events.len() as f32
            })
            .sum();

        tracing::info!("Event {}: {:?} ms", name, duration);
        metrics::histogram!("event_duration", "event_name" => name.to_string()).record(duration);
    }
}

/// Internal helper function to derive a new seed from the given seed and nonce.
fn derive_seed(seed: [u32; 8], kdf_salt: &Salt, nonce: usize) -> Result<[u32; 8]> {
    let pseudo_rand_key = kdf_salt.extract(bytemuck::cast_slice(&seed));
    let nonce = nonce.to_be_bytes();
    let context = vec![nonce.as_slice()];
    let output_key_material: Okm<Algorithm> =
        pseudo_rand_key.expand(&context, HKDF_SHA256).unwrap();
    let mut result = [0u32; 8];
    output_key_material
        .fill(bytemuck::cast_slice_mut(&mut result))
        .unwrap();
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn open(
    party: &mut Circuits,
    x: &[ChunkShare<u64>],
    distance_comparator: &DistanceComparator,
    matches_bitmap: &[CudaSlice<u64>],
    chunk_size: usize,
    db_sizes: &[usize],
    real_db_sizes: &[usize],
    offset: usize,
    total_db_sizes: &[usize],
    ignore_db_results: &[bool],
    match_distances_buffers_codes: &[ChunkShare<u16>],
    match_distances_buffers_masks: &[ChunkShare<u16>],
    match_distances_counters: &[CudaSlice<u32>],
    match_distances_indices: &[CudaSlice<u64>],
    batch_id: u64,
    code_dots: &[ChunkShareView<u16>],
    mask_dots: &[ChunkShareView<u16>],
    batch_size: usize,
    max_bucket_distances: usize,
    streams: &[CudaStream],
) {
    let n_devices = x.len();
    let mut a = Vec::with_capacity(n_devices);
    let mut b = Vec::with_capacity(n_devices);
    let mut c = Vec::with_capacity(n_devices);

    cudarc::nccl::result::group_start().unwrap();
    for (idx, res) in x.iter().enumerate() {
        // Result is in bit 0
        let res = res.get_offset(0, chunk_size);
        party.comms()[idx]
            .send_view(&res.b, party.next_id(), &streams[idx])
            .unwrap();
        a.push(res.a);
        b.push(res.b);
    }
    for (idx, res) in x.iter().enumerate() {
        let mut res = res.get_offset(1, chunk_size);
        party.comms()[idx]
            .receive_view(&mut res.a, party.prev_id(), &streams[idx])
            .unwrap();
        c.push(res.a);
    }
    cudarc::nccl::result::group_end().unwrap();

    distance_comparator.open_results(
        &a,
        &b,
        &c,
        matches_bitmap,
        db_sizes,
        real_db_sizes,
        offset,
        total_db_sizes,
        ignore_db_results,
        match_distances_buffers_codes,
        match_distances_buffers_masks,
        match_distances_counters,
        match_distances_indices,
        batch_id,
        code_dots,
        mask_dots,
        batch_size,
        max_bucket_distances,
        streams,
    );
}

#[allow(clippy::too_many_arguments)]
fn open_subset_results(
    party: &mut Circuits,
    x: &[ChunkShare<u64>],
    distance_comparator: &DistanceComparator,
    matches_bitmap: &[CudaSlice<u64>],
    chunk_size: usize,
    db_sizes: &[usize],
    real_db_sizes: &[usize],
    total_db_sizes: &[usize],
    ignore_db_results: &[bool],
    match_distances_buffers_codes: &[ChunkShare<u16>],
    match_distances_buffers_masks: &[ChunkShare<u16>],
    match_distances_counters: &[CudaSlice<u32>],
    match_distances_indices: &[CudaSlice<u64>],
    batch_id: u64,
    code_dots: &[ChunkShareView<u16>],
    mask_dots: &[ChunkShareView<u16>],
    batch_size: usize,
    max_bucket_distances: usize,
    streams: &[CudaStream],
    index_mapping: &[Vec<u32>],
) {
    let n_devices = x.len();
    let mut a = Vec::with_capacity(n_devices);
    let mut b = Vec::with_capacity(n_devices);
    let mut c = Vec::with_capacity(n_devices);

    cudarc::nccl::result::group_start().unwrap();
    for (idx, res) in x.iter().enumerate() {
        // Result is in bit 0
        let res = res.get_offset(0, chunk_size);
        party.comms()[idx]
            .send_view(&res.b, party.next_id(), &streams[idx])
            .unwrap();
        a.push(res.a);
        b.push(res.b);
    }
    for (idx, res) in x.iter().enumerate() {
        let mut res = res.get_offset(1, chunk_size);
        party.comms()[idx]
            .receive_view(&mut res.a, party.prev_id(), &streams[idx])
            .unwrap();
        c.push(res.a);
    }
    cudarc::nccl::result::group_end().unwrap();

    distance_comparator.open_results_with_index_mapping(
        &a,
        &b,
        &c,
        matches_bitmap,
        db_sizes,
        real_db_sizes,
        total_db_sizes,
        ignore_db_results,
        match_distances_buffers_codes,
        match_distances_buffers_masks,
        match_distances_counters,
        match_distances_indices,
        batch_id,
        code_dots,
        mask_dots,
        batch_size,
        max_bucket_distances,
        streams,
        index_mapping,
    );
}

#[allow(clippy::too_many_arguments)]
fn open_batch(
    party: &mut Circuits,
    x: &[ChunkShare<u64>],
    distance_comparator: &DistanceComparator,
    matches_bitmap: &[CudaSlice<u64>],
    chunk_size: usize,
    db_sizes: &[usize],
    real_db_sizes: &[usize],
    offset: usize,
    total_db_sizes: &[usize],
    ignore_db_results: &[bool],
    streams: &[CudaStream],
) {
    let n_devices = x.len();
    let mut a = Vec::with_capacity(n_devices);
    let mut b = Vec::with_capacity(n_devices);
    let mut c = Vec::with_capacity(n_devices);

    cudarc::nccl::result::group_start().unwrap();
    for (idx, res) in x.iter().enumerate() {
        // Result is in bit 0
        let res = res.get_offset(0, chunk_size);
        party.comms()[idx]
            .send_view(&res.b, party.next_id(), &streams[idx])
            .unwrap();
        a.push(res.a);
        b.push(res.b);
    }
    for (idx, res) in x.iter().enumerate() {
        let mut res = res.get_offset(1, chunk_size);
        party.comms()[idx]
            .receive_view(&mut res.a, party.prev_id(), &streams[idx])
            .unwrap();
        c.push(res.a);
    }
    cudarc::nccl::result::group_end().unwrap();

    distance_comparator.open_batch_results(
        &a,
        &b,
        &c,
        matches_bitmap,
        db_sizes,
        real_db_sizes,
        offset,
        total_db_sizes,
        ignore_db_results,
        streams,
    );
}

fn get_merged_results(host_results: &[Vec<u32>], n_devices: usize) -> Vec<u32> {
    let mut results = vec![];
    for j in 0..host_results[0].len() {
        let mut match_entry = NON_MATCH_ID;
        for i in 0..host_results.len() {
            let match_idx = host_results[i][j] * n_devices as u32 + i as u32;
            if host_results[i][j] != NON_MATCH_ID && match_idx < match_entry {
                match_entry = match_idx;
            }
        }

        results.push(match_entry);

        tracing::info!(
            "Query {}: match={} [index: {}]",
            j,
            match_entry != NON_MATCH_ID,
            match_entry
        );
    }
    results
}

fn distribute_insertions(results: &[usize], db_sizes: &[usize]) -> Vec<Vec<usize>> {
    let mut ret = vec![vec![]; db_sizes.len()];
    let start = db_sizes
        .iter()
        .position(|&x| x == *db_sizes.iter().min().unwrap())
        .unwrap();

    let mut c = start;
    for &r in results {
        ret[c].push(r);
        c = (c + 1) % db_sizes.len();
    }
    ret
}

fn reset_single_share<T>(
    devs: &[Arc<CudaDevice>],
    dst: &ChunkShare<T>,
    value: u8,
    streams: &[CudaStream],
    i: usize,
) {
    devs[i].bind_to_thread().unwrap();
    unsafe {
        result::memset_d8_async(
            *dst.a.device_ptr(),
            value,
            dst.a.num_bytes(),
            streams[i].stream,
        )
        .unwrap();

        result::memset_d8_async(
            *dst.b.device_ptr(),
            value,
            dst.b.num_bytes(),
            streams[i].stream,
        )
        .unwrap();
    };
}

pub(crate) fn reset_share<T>(
    devs: &[Arc<CudaDevice>],
    dst: &[ChunkShare<T>],
    value: u8,
    streams: &[CudaStream],
) {
    for i in 0..devs.len() {
        reset_single_share(devs, &dst[i], value, streams, i);
    }
}

pub(crate) fn reset_slice<T>(
    devs: &[Arc<CudaDevice>],
    dst: &[CudaSlice<T>],
    value: u8,
    streams: &[CudaStream],
) {
    for i in 0..devs.len() {
        devs[i].bind_to_thread().unwrap();
        unsafe {
            result::memset_d8_async(
                *dst[i].device_ptr(),
                value,
                dst[i].num_bytes(),
                streams[i].stream,
            )
            .unwrap();
        };
    }
}

fn calculate_insertion_indices(
    merged_results: &mut [u32],
    insertion_list: &[Vec<usize>],
    db_sizes: &[usize],
    batch_size: usize,
) -> Vec<bool> {
    let mut matches = vec![true; batch_size];
    let mut last_db_index = db_sizes.iter().sum::<usize>() as u32;
    let (mut min_index, mut min_index_val) = (0, usize::MAX);
    for (i, list) in insertion_list.iter().enumerate() {
        if let Some(&first_val) = list.first() {
            if first_val < min_index_val {
                min_index_val = first_val;
                min_index = i;
            }
        }
    }
    let mut c: usize = 0;
    loop {
        for i in 0..insertion_list.len() {
            let idx = (i + min_index) % insertion_list.len();

            if c >= insertion_list[idx].len() {
                return matches;
            }

            let insert_idx = insertion_list[idx][c];
            merged_results[insert_idx] = last_db_index;
            matches[insert_idx] = false;
            last_db_index += 1;
        }
        c += 1;
    }
}

impl ServerActor {
    fn reset_anon_stats_distance_cache(&self, eye: Eye, orientation: Orientation) {
        let streams = &self.streams[0];
        let (codes, masks, counters, indices) = match orientation {
            Orientation::Normal => self.match_distances_buffer.get_buffers(eye),
            Orientation::Mirror => self.match_distances_buffer_mirror.get_buffers(eye),
        };

        reset_slice(self.device_manager.devices(), counters, 0, streams);
        reset_slice(self.device_manager.devices(), indices, 0xff, streams);
        reset_share(self.device_manager.devices(), masks, 0xff, streams);
        reset_share(self.device_manager.devices(), codes, 0xff, streams);
        self.device_manager.await_streams(streams);
    }

    fn persist_one_sided_caches(
        &self,
        eye: Eye,
        orientation: Orientation,
        caches: &[OneSidedDistanceCache],
    ) {
        tracing::info!(
            "Persisting one-sided anon stats caches for eye {:?} and orientation {:?}",
            eye,
            orientation
        );
        if orientation != Orientation::Normal {
            return;
        }
        let writer = match &self.anon_stats_writer {
            Some(writer) => writer,
            None => {
                tracing::info!("No writer configured for anon stats caches");
                return;
            }
        };

        let origin = AnonStatsOrigin {
            side: Some(eye),
            orientation: AnonStatsOrientation::Normal,
            context: AnonStatsContext::GPU,
        };

        let mut uniqueness_bundles = Vec::new();
        let mut reauth_bundles = Vec::new();
        for cache in caches {
            for (key, values) in cache.iter() {
                if values.is_empty() {
                    continue;
                }
                let match_id = match i64::try_from(*key) {
                    Ok(id) => id,
                    Err(_) => {
                        tracing::warn!(
                            match_id = *key,
                            "Skipping anon stats entry with id exceeding i64"
                        );
                        continue;
                    }
                };
                let mut sorted = values.clone();
                sorted.sort_by_key(|share| share.idx);
                let operation = sorted
                    .first()
                    .map(|share| share.operation)
                    .unwrap_or(AnonStatsOperation::Uniqueness);
                if sorted.iter().any(|share| share.operation != operation) {
                    tracing::warn!(
                        "Mixed anonymized stats operations detected for match_id {} on eye {:?}",
                        match_id,
                        eye
                    );
                }
                let distance_bundle = sorted
                    .into_iter()
                    .map(|share| {
                        DistanceShare::new(
                            Share::new(RingElement(share.code_a), RingElement(share.code_b)),
                            Share::new(RingElement(share.mask_a), RingElement(share.mask_b)),
                        )
                    })
                    .collect::<DistanceBundle1D>();
                match operation {
                    AnonStatsOperation::Reauth => reauth_bundles.push((match_id, distance_bundle)),
                    _ => uniqueness_bundles.push((match_id, distance_bundle)),
                }
            }
        }

        if !uniqueness_bundles.is_empty() {
            tracing::info!(
                "Inserting {} uniqueness anon stats bundles",
                uniqueness_bundles.len()
            );
            writer.insert_1d(
                origin,
                AnonStatsOperation::Uniqueness,
                uniqueness_bundles.clone(),
            );
        }
        if !reauth_bundles.is_empty() {
            tracing::info!(
                "Inserting {} reauth anon stats bundles",
                reauth_bundles.len()
            );
            writer.insert_1d(origin, AnonStatsOperation::Reauth, reauth_bundles.clone());
        }

        if uniqueness_bundles.is_empty() && reauth_bundles.is_empty() {
            tracing::info!("No anon stats bundles to insert");
        }
    }

    fn persist_two_sided_caches(&self, caches: &[TwoSidedDistanceCache]) {
        tracing::info!("Persisting two-sided anon stats caches");
        let writer = match &self.anon_stats_writer {
            Some(writer) => writer,
            None => return,
        };

        let origin = AnonStatsOrigin {
            side: None,
            orientation: AnonStatsOrientation::Normal,
            context: AnonStatsContext::GPU,
        };

        let mut uniqueness_bundles = Vec::new();
        let mut reauth_bundles = Vec::new();
        for cache in caches {
            for (key, (left_values, right_values)) in cache.iter() {
                if left_values.is_empty() || right_values.is_empty() {
                    continue;
                }
                let match_id = match i64::try_from(*key) {
                    Ok(id) => id,
                    Err(_) => {
                        tracing::warn!(
                            match_id = *key,
                            "Skipping 2D anon stats entry with id exceeding i64"
                        );
                        continue;
                    }
                };
                let mut left_sorted = left_values.clone();
                left_sorted.sort_by_key(|share| share.idx);
                let operation = left_sorted
                    .first()
                    .map(|share| share.operation)
                    .unwrap_or(AnonStatsOperation::Uniqueness);
                if left_sorted.iter().any(|share| share.operation != operation)
                    || right_values
                        .iter()
                        .any(|share| share.operation != operation)
                {
                    tracing::warn!(
                        "Mixed operations detected while persisting 2D anon stats for match_id {}",
                        match_id
                    );
                }
                let mut right_sorted = right_values.clone();
                right_sorted.sort_by_key(|share| share.idx);

                let left_bundle = left_sorted
                    .into_iter()
                    .map(|share| {
                        DistanceShare::new(
                            Share::new(RingElement(share.code_a), RingElement(share.code_b)),
                            Share::new(RingElement(share.mask_a), RingElement(share.mask_b)),
                        )
                    })
                    .collect::<DistanceBundle1D>();

                let right_bundle = right_sorted
                    .into_iter()
                    .map(|share| {
                        DistanceShare::new(
                            Share::new(RingElement(share.code_a), RingElement(share.code_b)),
                            Share::new(RingElement(share.mask_a), RingElement(share.mask_b)),
                        )
                    })
                    .collect::<DistanceBundle1D>();

                match operation {
                    AnonStatsOperation::Reauth => {
                        reauth_bundles.push((match_id, (left_bundle, right_bundle)))
                    }
                    _ => uniqueness_bundles.push((match_id, (left_bundle, right_bundle))),
                }
            }
        }

        if !uniqueness_bundles.is_empty() {
            writer.insert_2d(origin, AnonStatsOperation::Uniqueness, uniqueness_bundles);
        }
        if !reauth_bundles.is_empty() {
            writer.insert_2d(origin, AnonStatsOperation::Reauth, reauth_bundles);
        }
    }
}

fn copy_distance_shares_for_indices(
    batch_reauth_targets: &[(u32, u32)],
    code_dots: &[ChunkShareView<u16>],
    mask_dots: &[ChunkShareView<u16>],
    current_db_offset: usize,
    chunk_sizes: &[usize],
    query_size: usize,
    distance_cache: &mut [OneSidedDistanceCache],
    streams: &[CudaStream],
) {
    if batch_reauth_targets.is_empty() {
        return;
    }
    // map each reauth target to its corresponding GPU
    let num_devices = streams.len();
    let mut gpu_targets = vec![vec![]; num_devices];
    for (query_idx, reauth_target) in batch_reauth_targets {
        let device_idx = (*reauth_target as usize) % num_devices;
        let device_db_idx = (*reauth_target as usize) / num_devices;
        if (current_db_offset..current_db_offset + chunk_sizes[device_idx])
            .contains(&(device_db_idx as usize))
        {
            let chunk_db_idx = device_db_idx - current_db_offset;
            let target_idx =
                ROTATIONS * query_size * chunk_db_idx + ROTATIONS * *query_idx as usize;
            gpu_targets[device_idx].push(target_idx);
        }
    }
    // nothing to do for this chunk
    if gpu_targets.iter().all(|x| x.is_empty()) {
        return;
    }
    // prepare target buffers for each GPU
    let mut codes_buf = gpu_targets
        .iter()
        .map(|targets| vec![0u16; targets.len() * ROTATIONS * 2])
        .collect_vec();
    let mut masks_buf = codes_buf.clone();

    // for each device, schedule copies on stream
    for (code_dot, mask_dot, code_buf, mask_buf, stream, targets) in izip!(
        code_dots,
        mask_dots,
        &mut codes_buf,
        &mut masks_buf,
        streams,
        &gpu_targets
    ) {
        for (idx, chunk_idx) in targets.iter().enumerate() {
            // copy codes a share
            // SAFETY: We wait on streams below, so the target buffers are still in scope and valid
            unsafe {
                helpers::dtoh_at_offset(
                    code_buf.as_mut_ptr() as u64,
                    idx * ROTATIONS * 2,
                    *code_dot.a.device_ptr(),
                    *chunk_idx as usize,
                    ROTATIONS,
                    stream.stream,
                );
            }
            // copy codes b share
            // SAFETY: We wait on streams below, so the target buffers are still in scope and valid
            unsafe {
                helpers::dtoh_at_offset(
                    code_buf.as_mut_ptr() as u64,
                    idx * ROTATIONS * 2 + ROTATIONS,
                    *code_dot.b.device_ptr(),
                    *chunk_idx as usize,
                    ROTATIONS,
                    stream.stream,
                );
            }
            // copy mask a share
            // SAFETY: We wait on streams below, so the target buffers are still in scope and valid
            unsafe {
                helpers::dtoh_at_offset(
                    mask_buf.as_mut_ptr() as u64,
                    idx * ROTATIONS * 2,
                    *mask_dot.a.device_ptr(),
                    *chunk_idx as usize,
                    ROTATIONS,
                    stream.stream,
                );
            }
            // copy mask b share
            // SAFETY: We wait on streams below, so the target buffers are still in scope and valid
            unsafe {
                helpers::dtoh_at_offset(
                    mask_buf.as_mut_ptr() as u64,
                    idx * ROTATIONS * 2 + ROTATIONS,
                    *mask_dot.b.device_ptr(),
                    *chunk_idx as usize,
                    ROTATIONS,
                    stream.stream,
                );
            }
        }
    }
    // wait for all copies to be finished
    for stream in streams {
        // SAFETY: these streams have already been created, and the caller holds a
        // reference to their CudaDevice, which makes sure they aren't dropped.
        unsafe {
            stream::synchronize(stream.stream).unwrap();
        }
    }

    // Bundle the results and add them to the distance cache
    for (codes, masks, targets, cache) in izip!(codes_buf, masks_buf, gpu_targets, distance_cache) {
        for (idx, target) in targets.iter().enumerate() {
            let mut shares = Vec::with_capacity(ROTATIONS);
            for i in 0..ROTATIONS {
                shares.push(CpuDistanceShare {
                    idx: (*target as u64) * ROTATIONS as u64 + i as u64,
                    code_a: codes[idx * 2 * ROTATIONS + i],
                    code_b: codes[idx * 2 * ROTATIONS + ROTATIONS + i],
                    mask_a: masks[idx * 2 * ROTATIONS + i],
                    mask_b: masks[idx * 2 * ROTATIONS + ROTATIONS + i],
                    operation: AnonStatsOperation::Reauth,
                })
            }
            if cache.insert(*target as u64, shares).is_some() {
                tracing::error!(
                    "we should not have duplicate values in reauth anon stats insertion"
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn write_db_at_index(
    left_code_db_slices: &SlicedProcessedDatabase,
    left_mask_db_slices: &SlicedProcessedDatabase,
    right_code_db_slices: &SlicedProcessedDatabase,
    right_mask_db_slices: &SlicedProcessedDatabase,
    compact_device_queries_left: &DeviceCompactQuery,
    compact_device_sums_left: &DeviceCompactSums,
    compact_device_queries_right: &DeviceCompactQuery,
    compact_device_sums_right: &DeviceCompactSums,
    src_index: usize,
    dst_index: usize,
    device_index: usize,
    streams: &[CudaStream],
) {
    for (code_length, db, query, sums) in [
        (
            IRIS_CODE_LENGTH,
            left_code_db_slices,
            &compact_device_queries_left.code_query_insert,
            &compact_device_sums_left.code_query_insert,
        ),
        (
            MASK_CODE_LENGTH,
            left_mask_db_slices,
            &compact_device_queries_left.mask_query_insert,
            &compact_device_sums_left.mask_query_insert,
        ),
        (
            IRIS_CODE_LENGTH,
            right_code_db_slices,
            &compact_device_queries_right.code_query_insert,
            &compact_device_sums_right.code_query_insert,
        ),
        (
            MASK_CODE_LENGTH,
            right_mask_db_slices,
            &compact_device_queries_right.mask_query_insert,
            &compact_device_sums_right.mask_query_insert,
        ),
    ] {
        unsafe {
            helpers::dtoh_at_offset(
                db.code_gr.limb_0[device_index],
                dst_index * code_length,
                *query.limb_0[device_index].device_ptr(),
                code_length * 15 + src_index * code_length * ROTATIONS,
                code_length,
                streams[device_index].stream,
            );

            helpers::dtoh_at_offset(
                db.code_gr.limb_1[device_index],
                dst_index * code_length,
                *query.limb_1[device_index].device_ptr(),
                code_length * 15 + src_index * code_length * ROTATIONS,
                code_length,
                streams[device_index].stream,
            );

            helpers::dtod_at_offset(
                *db.code_sums_gr.limb_0[device_index].device_ptr(),
                dst_index * mem::size_of::<u32>(),
                *sums.limb_0[device_index].device_ptr(),
                mem::size_of::<u32>() * 15 + src_index * mem::size_of::<u32>() * ROTATIONS,
                mem::size_of::<u32>(),
                streams[device_index].stream,
            );

            helpers::dtod_at_offset(
                *db.code_sums_gr.limb_1[device_index].device_ptr(),
                dst_index * mem::size_of::<u32>(),
                *sums.limb_1[device_index].device_ptr(),
                size_of::<u32>() * 15 + src_index * mem::size_of::<u32>() * ROTATIONS,
                size_of::<u32>(),
                streams[device_index].stream,
            );
        }
    }
}

pub fn prepare_or_policy_bitmap(
    max_db_size: usize,
    or_rule_indices: Vec<Vec<u32>>,
    batch_size: usize,
) -> Vec<u64> {
    let row_stride64 = max_db_size.div_ceil(64);
    let total_size = row_stride64 * batch_size;

    // Create the bitmap on the host
    let mut bitmap = vec![0u64; total_size];

    for (query_idx, db_indices) in or_rule_indices.iter().enumerate() {
        for &db_idx in db_indices {
            let row_start = query_idx * row_stride64;
            let word_idx = row_start + (db_idx as usize / 64);
            let bit_offset = db_idx as usize % 64;
            bitmap[word_idx] |= 1 << bit_offset;
        }
    }
    bitmap
}

pub fn generate_luc_records(
    latest_luc_index: u32,
    mut or_rule_indices: Vec<Vec<u32>>,
    lookback_records: usize,
    batch_size: usize,
    skip_lookback_requests: HashSet<usize>,
) -> Vec<Vec<u32>> {
    // If lookback_records is 0, return the original or_rule_indices
    if lookback_records == 0 {
        return or_rule_indices;
    }
    // Generate the lookback serial IDs: [current_db_size - luc_lookback_records,
    // current_db_size)
    let lookback_start = latest_luc_index.saturating_sub(lookback_records as u32); // ensure no underflow
    let lookback_ids: Vec<u32> = (lookback_start..=latest_luc_index).collect();
    let lookback_records: Vec<Vec<u32>> = vec![lookback_ids; batch_size];

    // If there are no OR rules, return only the lookback records
    if or_rule_indices.is_empty() {
        return lookback_records;
    }

    // Otherwise, merge them into each inner vector of or_rule_indices
    for (idx, (or_ids, luc_ids)) in
        izip!(or_rule_indices.iter_mut(), lookback_records.iter()).enumerate()
    {
        if skip_lookback_requests.contains(&idx) {
            continue;
        }
        // Add the lookback IDs
        or_ids.extend_from_slice(luc_ids);
        // Sort and remove duplicates
        or_ids.sort_unstable();
        or_ids.dedup();
    }
    or_rule_indices
}

impl InMemoryStore for ServerActor {
    fn load_single_record_from_db(
        &mut self,
        index: usize,
        _vector_id: VectorId,
        left_code: &[u16],
        left_mask: &[u16],
        right_code: &[u16],
        right_mask: &[u16],
    ) {
        ShareDB::load_single_record_from_db(
            index,
            &self.left_code_db_slices.code_gr,
            left_code,
            self.device_manager.device_count(),
            IRIS_CODE_LENGTH,
        );
        ShareDB::load_single_record_from_db(
            index,
            &self.left_mask_db_slices.code_gr,
            left_mask,
            self.device_manager.device_count(),
            MASK_CODE_LENGTH,
        );
        ShareDB::load_single_record_from_db(
            index,
            &self.right_code_db_slices.code_gr,
            right_code,
            self.device_manager.device_count(),
            IRIS_CODE_LENGTH,
        );
        ShareDB::load_single_record_from_db(
            index,
            &self.right_mask_db_slices.code_gr,
            right_mask,
            self.device_manager.device_count(),
            MASK_CODE_LENGTH,
        );
    }
    fn increment_db_size(&mut self, index: usize) {
        self.current_db_sizes[index % self.device_manager.device_count()] += 1;
    }

    fn load_single_record_from_s3(
        &mut self,
        index: usize,
        _vector_id: VectorId,
        left_code_odd: &[u8],
        left_code_even: &[u8],
        right_code_odd: &[u8],
        right_code_even: &[u8],
        left_mask_odd: &[u8],
        left_mask_even: &[u8],
        right_mask_odd: &[u8],
        right_mask_even: &[u8],
    ) {
        ShareDB::load_single_record_from_s3(
            index,
            &self.left_code_db_slices.code_gr,
            left_code_odd,
            left_code_even,
            self.device_manager.device_count(),
            IRIS_CODE_LENGTH,
        );
        ShareDB::load_single_record_from_s3(
            index,
            &self.left_mask_db_slices.code_gr,
            left_mask_odd,
            left_mask_even,
            self.device_manager.device_count(),
            MASK_CODE_LENGTH,
        );
        ShareDB::load_single_record_from_s3(
            index,
            &self.right_code_db_slices.code_gr,
            right_code_odd,
            right_code_even,
            self.device_manager.device_count(),
            IRIS_CODE_LENGTH,
        );
        ShareDB::load_single_record_from_s3(
            index,
            &self.right_mask_db_slices.code_gr,
            right_mask_odd,
            right_mask_even,
            self.device_manager.device_count(),
            MASK_CODE_LENGTH,
        );
    }

    fn preprocess_db(&mut self) {
        // we also register the memory allocated, page-locking it for more performance
        self.register_host_memory();

        self.codes_engine
            .preprocess_db(&mut self.left_code_db_slices, &self.current_db_sizes);
        self.masks_engine
            .preprocess_db(&mut self.left_mask_db_slices, &self.current_db_sizes);
        self.codes_engine
            .preprocess_db(&mut self.right_code_db_slices, &self.current_db_sizes);
        self.masks_engine
            .preprocess_db(&mut self.right_mask_db_slices, &self.current_db_sizes);
    }

    fn current_db_sizes(&self) -> impl std::fmt::Debug {
        &self.current_db_sizes
    }

    fn fake_db(&mut self, fake_db_size: usize) {
        tracing::warn!(
            "Faking db with {} entries, returned results will be random.",
            fake_db_size
        );
        self.current_db_sizes =
            vec![fake_db_size / self.current_db_sizes.len(); self.current_db_sizes.len()];
    }
}
