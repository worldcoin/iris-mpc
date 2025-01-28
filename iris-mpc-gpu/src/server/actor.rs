use super::{BatchQuery, Eye, ServerJob, ServerJobResult};
use crate::{
    dot::{
        distance_comparator::DistanceComparator,
        share_db::{preprocess_query, DBChunkBuffers, ShareDB, SlicedProcessedDatabase},
        IRIS_CODE_LENGTH, MASK_CODE_LENGTH, ROTATIONS,
    },
    helpers::{
        self,
        comm::NcclComm,
        device_manager::DeviceManager,
        dtoh_on_stream_sync, htod_on_stream_sync,
        query_processor::{
            CompactQuery, CudaVec2DSlicerRawPointer, DeviceCompactQuery, DeviceCompactSums,
        },
    },
    threshold_ring::protocol::{ChunkShare, ChunkShareView, Circuits},
};
use cudarc::{
    cublas::CudaBlas,
    driver::{
        result::{self, event::elapsed, mem_get_info},
        sys::CUevent,
        CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice,
    },
};
use eyre::eyre;
use futures::{Future, FutureExt};
use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    helpers::sha256::sha256_bytes,
    iris_db::iris::{IrisCode, MATCH_THRESHOLD_RATIO},
    IrisCodeDbSlice,
};
use itertools::Itertools;
use rand::{rngs::StdRng, SeedableRng};
use ring::hkdf::{Algorithm, Okm, Salt, HKDF_SHA256};
use std::{collections::HashMap, mem, sync::Arc, time::Instant};
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

#[derive(Debug, Clone)]
pub struct ServerActorHandle {
    job_queue: mpsc::Sender<ServerJob>,
}

impl ServerActorHandle {
    pub async fn submit_batch_query(
        &mut self,
        batch: BatchQuery,
    ) -> impl Future<Output = ServerJobResult> {
        let (tx, rx) = oneshot::channel();
        let job = ServerJob {
            batch,
            return_channel: tx,
        };
        self.job_queue.send(job).await.unwrap();
        rx.map(|x| x.unwrap())
    }
}

const DB_CHUNK_SIZE: usize = 1 << 15;
const KDF_SALT: &str = "111a1a93518f670e9bb0c2c68888e2beb9406d4c4ed571dc77b801e676ae3091"; // Random 32 byte salt
const SUPERMATCH_THRESHOLD: usize = 4_000;

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
    phase2_buckets: Circuits,
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
    n_buckets: usize,
    return_partial_results: bool,
    disable_persistence: bool,
    enable_debug_timing: bool,
    code_chunk_buffers: Vec<DBChunkBuffers>,
    mask_chunk_buffers: Vec<DBChunkBuffers>,
    phase1_events: Vec<Vec<CUevent>>,
    phase2_events: Vec<Vec<CUevent>>,
    match_distances_buffer_codes_left: Vec<ChunkShare<u16>>,
    match_distances_buffer_codes_right: Vec<ChunkShare<u16>>,
    match_distances_buffer_masks_left: Vec<ChunkShare<u16>>,
    match_distances_buffer_masks_right: Vec<ChunkShare<u16>>,
    match_distances_counter_left: Vec<CudaSlice<u32>>,
    match_distances_counter_right: Vec<CudaSlice<u32>>,
    match_distances_indices_left: Vec<CudaSlice<u32>>,
    match_distances_indices_right: Vec<CudaSlice<u32>>,
    buckets: ChunkShare<u32>,
}

const NON_MATCH_ID: u32 = u32::MAX;

impl ServerActor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        party_id: usize,
        chacha_seeds: ([u32; 8], [u32; 8]),
        job_queue_size: usize,
        max_db_size: usize,
        max_batch_size: usize,
        match_distances_buffer_size: usize,
        n_buckets: usize,
        return_partial_results: bool,
        disable_persistence: bool,
        enable_debug_timing: bool,
    ) -> eyre::Result<(Self, ServerActorHandle)> {
        let device_manager = Arc::new(DeviceManager::init());
        Self::new_with_device_manager(
            party_id,
            chacha_seeds,
            device_manager,
            job_queue_size,
            max_db_size,
            max_batch_size,
            match_distances_buffer_size,
            n_buckets,
            return_partial_results,
            disable_persistence,
            enable_debug_timing,
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
        n_buckets: usize,
        return_partial_results: bool,
        disable_persistence: bool,
        enable_debug_timing: bool,
    ) -> eyre::Result<(Self, ServerActorHandle)> {
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
            n_buckets,
            return_partial_results,
            disable_persistence,
            enable_debug_timing,
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
        n_buckets: usize,
        return_partial_results: bool,
        disable_persistence: bool,
        enable_debug_timing: bool,
    ) -> eyre::Result<(Self, ServerActorHandle)> {
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
            n_buckets,
            return_partial_results,
            disable_persistence,
            enable_debug_timing,
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
        n_buckets: usize,
        return_partial_results: bool,
        disable_persistence: bool,
        enable_debug_timing: bool,
    ) -> eyre::Result<Self> {
        assert_ne!(max_batch_size, 0);
        let mut kdf_nonce = 0;
        let kdf_salt: Salt = Salt::new(HKDF_SHA256, &hex::decode(KDF_SALT)?);
        let n_queries = max_batch_size * ROTATIONS;

        // helper closure to generate the next chacha seeds
        let mut next_chacha_seeds =
            |seeds: ([u32; 8], [u32; 8])| -> eyre::Result<([u32; 8], [u32; 8])> {
                let nonce = kdf_nonce;
                kdf_nonce += 1;
                Ok((
                    derive_seed(seeds.0, &kdf_salt, nonce)?,
                    derive_seed(seeds.1, &kdf_salt, nonce)?,
                ))
            };

        tracing::info!("Starting engines...");

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

        tracing::info!("Allocated db in {:?}", now.elapsed());

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

        let phase2_buckets = Circuits::new(
            party_id,
            match_distances_buffer_size,
            match_distances_buffer_size / 64,
            next_chacha_seeds(chacha_seeds)?,
            device_manager.clone(),
            comms.clone(),
        );

        let distance_comparator = DistanceComparator::init(n_queries, device_manager.clone());
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

        let code_chunk_buffers = vec![codes_engine.alloc_db_chunk_buffer(DB_CHUNK_SIZE); 2];
        let mask_chunk_buffers = vec![masks_engine.alloc_db_chunk_buffer(DB_CHUNK_SIZE); 2];

        // Create all needed events
        let phase1_events = vec![device_manager.create_events(); 2];
        let phase2_events = vec![device_manager.create_events(); 2];

        // Buffers and counters for match distribution
        let match_distances_buffer_codes_left =
            distance_comparator.prepare_match_distances_buffer(match_distances_buffer_size);
        let match_distances_buffer_codes_right =
            distance_comparator.prepare_match_distances_buffer(match_distances_buffer_size);
        let match_distances_buffer_masks_left =
            distance_comparator.prepare_match_distances_buffer(match_distances_buffer_size);
        let match_distances_buffer_masks_right =
            distance_comparator.prepare_match_distances_buffer(match_distances_buffer_size);
        let match_distances_counter_left = distance_comparator.prepare_match_distances_counter();
        let match_distances_counter_right = distance_comparator.prepare_match_distances_counter();
        let match_distances_indices_left =
            distance_comparator.prepare_match_distances_index(match_distances_buffer_size);
        let match_distances_indices_right =
            distance_comparator.prepare_match_distances_index(match_distances_buffer_size);
        let buckets = distance_comparator.prepare_match_distances_buckets(n_buckets);

        for dev in device_manager.devices() {
            dev.synchronize().unwrap();
        }

        Ok(Self {
            party_id,
            job_queue,
            device_manager,
            codes_engine,
            masks_engine,
            phase2,
            phase2_batch,
            phase2_buckets,
            buckets,
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
            n_buckets,
            return_partial_results,
            disable_persistence,
            enable_debug_timing,
            code_chunk_buffers,
            mask_chunk_buffers,
            phase1_events,
            phase2_events,
            match_distances_buffer_codes_left,
            match_distances_buffer_codes_right,
            match_distances_buffer_masks_left,
            match_distances_buffer_masks_right,
            match_distances_counter_left,
            match_distances_counter_right,
            match_distances_indices_left,
            match_distances_indices_right,
        })
    }

    pub fn run(mut self) {
        while let Some(job) = self.job_queue.blocking_recv() {
            let ServerJob {
                batch,
                return_channel,
            } = job;
            let _ = self.process_batch_query(batch, return_channel);
        }
        tracing::info!("Server Actor finished due to all job queues being closed");
    }

    pub fn current_db_sizes(&self) -> Vec<usize> {
        self.current_db_sizes.clone()
    }

    pub fn set_current_db_sizes(&mut self, sizes: Vec<usize>) {
        self.current_db_sizes = sizes;
    }

    pub fn load_full_db(
        &mut self,
        left: &IrisCodeDbSlice,
        right: &IrisCodeDbSlice,
        db_size: usize,
    ) {
        assert!(
            [left.0.len(), right.0.len(),]
                .iter()
                .all(|&x| x == db_size * IRIS_CODE_LENGTH),
            "Internal DB mismatch, left and right iris code db sizes differ, expected {}, left \
             has {}, while right has {}",
            db_size * IRIS_CODE_LENGTH,
            left.0.len(),
            right.0.len()
        );

        assert!(
            [left.1.len(), right.1.len()]
                .iter()
                .all(|&x| x == db_size * MASK_CODE_LENGTH),
            "Internal DB mismatch, left and right mask code db sizes differ, expected {}, left \
             has {}, while right has {}",
            db_size * MASK_CODE_LENGTH,
            left.1.len(),
            right.1.len()
        );

        let db_lens1 = self
            .codes_engine
            .load_full_db(&mut self.left_code_db_slices, left.0);
        let db_lens2 = self
            .masks_engine
            .load_full_db(&mut self.left_mask_db_slices, left.1);
        let db_lens3 = self
            .codes_engine
            .load_full_db(&mut self.right_code_db_slices, right.0);
        let db_lens4 = self
            .masks_engine
            .load_full_db(&mut self.right_mask_db_slices, right.1);

        assert_eq!(db_lens1, db_lens2);
        assert_eq!(db_lens1, db_lens3);
        assert_eq!(db_lens1, db_lens4);

        self.current_db_sizes = db_lens1;
    }

    pub fn load_single_record_from_db(
        &mut self,
        index: usize,
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

    #[allow(clippy::too_many_arguments)]
    pub fn load_single_record_from_s3(
        &mut self,
        index: usize,
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

    pub fn increment_db_size(&mut self, index: usize) {
        self.current_db_sizes[index % self.device_manager.device_count()] += 1;
    }

    pub fn preprocess_db(&mut self) {
        self.codes_engine
            .preprocess_db(&mut self.left_code_db_slices, &self.current_db_sizes);
        self.masks_engine
            .preprocess_db(&mut self.left_mask_db_slices, &self.current_db_sizes);
        self.codes_engine
            .preprocess_db(&mut self.right_code_db_slices, &self.current_db_sizes);
        self.masks_engine
            .preprocess_db(&mut self.right_mask_db_slices, &self.current_db_sizes);
    }

    pub fn register_host_memory(&self) {
        self.codes_engine
            .register_host_memory(&self.left_code_db_slices, self.max_db_size);
        self.masks_engine
            .register_host_memory(&self.left_mask_db_slices, self.max_db_size);
        self.codes_engine
            .register_host_memory(&self.right_code_db_slices, self.max_db_size);
        self.masks_engine
            .register_host_memory(&self.right_mask_db_slices, self.max_db_size);
    }

    fn process_batch_query(
        &mut self,
        batch: BatchQuery,
        return_channel: oneshot::Sender<ServerJobResult>,
    ) -> eyre::Result<()> {
        let now = Instant::now();
        let mut events: HashMap<&str, Vec<Vec<CUevent>>> = HashMap::new();

        tracing::info!("Started processing batch");

        let mut batch = batch;
        let mut batch_size = batch.store_left.code.len();
        assert!(batch_size > 0 && batch_size <= self.max_batch_size);
        assert!(
            batch_size == batch.store_left.mask.len()
                && batch_size == batch.request_ids.len()
                && batch_size == batch.metadata.len()
                && batch_size == batch.store_right.code.len()
                && batch_size == batch.store_right.mask.len()
                && batch_size * ROTATIONS == batch.query_left.code.len()
                && batch_size * ROTATIONS == batch.query_left.mask.len()
                && batch_size * ROTATIONS == batch.query_right.code.len()
                && batch_size * ROTATIONS == batch.query_right.mask.len()
                && batch_size * ROTATIONS == batch.db_left.code.len()
                && batch_size * ROTATIONS == batch.db_left.mask.len()
                && batch_size * ROTATIONS == batch.db_right.code.len()
                && batch_size * ROTATIONS == batch.db_right.mask.len()
                && batch_size * ROTATIONS == batch.query_left_preprocessed.len()
                && batch_size * ROTATIONS == batch.query_right_preprocessed.len()
                && batch_size * ROTATIONS == batch.db_left_preprocessed.len()
                && batch_size * ROTATIONS == batch.db_right_preprocessed.len(),
            "Query batch sizes mismatch"
        );

        ///////////////////////////////////////////////////////////////////
        // PERFORM DELETIONS (IF ANY)
        ///////////////////////////////////////////////////////////////////
        if !batch.deletion_requests_indices.is_empty() {
            tracing::info!("Performing deletions");
            // Prepare dummy deletion shares
            let (dummy_queries, dummy_sums) = self.prepare_deletion_shares()?;

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
        // SYNC BATCH CONTENTS AND FILTER OUT INVALID ENTRIES
        ///////////////////////////////////////////////////////////////////
        let tmp_now = Instant::now();
        tracing::info!("Syncing batch entries");

        // Compute hash of the request ids concatenated
        let batch_hash = sha256_bytes(batch.request_ids.join(""));
        tracing::info!("Current batch hash: {}", hex::encode(&batch_hash[0..4]));

        let valid_entries =
            self.sync_batch_entries(&batch.valid_entries, self.max_batch_size, &batch_hash)?;
        let valid_entry_idxs = valid_entries.iter().positions(|&x| x).collect::<Vec<_>>();
        batch_size = valid_entry_idxs.len();
        batch.retain(&valid_entry_idxs);
        tracing::info!("Sync and filter done in {:?}", tmp_now.elapsed());

        ///////////////////////////////////////////////////////////////////
        // COMPARE LEFT EYE QUERIES
        ///////////////////////////////////////////////////////////////////
        tracing::info!("Comparing left eye queries");
        // *Query* variant including Lagrange interpolation.
        let compact_query_left = CompactQuery {
            code_query:        batch.query_left_preprocessed.code.clone(),
            mask_query:        batch.query_left_preprocessed.mask.clone(),
            code_query_insert: batch.db_left_preprocessed.code.clone(),
            mask_query_insert: batch.db_left_preprocessed.mask.clone(),
        };
        let query_store_left = batch.store_left;

        let (compact_device_queries_left, compact_device_sums_left) = record_stream_time!(
            &self.device_manager,
            &self.streams[0],
            events,
            "query_preprocess",
            self.enable_debug_timing,
            {
                // This needs to be max_batch_size, even though the query can be shorter to have
                // enough padding for GEMM
                let compact_device_queries_left = compact_query_left.htod_transfer(
                    &self.device_manager,
                    &self.streams[0],
                    self.max_batch_size,
                )?;

                let compact_device_sums_left = compact_device_queries_left.query_sums(
                    &self.codes_engine,
                    &self.masks_engine,
                    &self.streams[0],
                    &self.cublas_handles[0],
                )?;

                (compact_device_queries_left, compact_device_sums_left)
            }
        );

        tracing::info!("Comparing left eye queries against DB and self");
        self.compare_query_against_db_and_self(
            &compact_device_queries_left,
            &compact_device_sums_left,
            &mut events,
            Eye::Left,
            batch_size,
        );

        ///////////////////////////////////////////////////////////////////
        // COMPARE RIGHT EYE QUERIES
        ///////////////////////////////////////////////////////////////////
        tracing::info!("Comparing right eye queries");
        // *Query* variant including Lagrange interpolation.
        let compact_query_right = CompactQuery {
            code_query:        batch.query_right_preprocessed.code.clone(),
            mask_query:        batch.query_right_preprocessed.mask.clone(),
            code_query_insert: batch.db_right_preprocessed.code.clone(),
            mask_query_insert: batch.db_right_preprocessed.mask.clone(),
        };
        let query_store_right = batch.store_right;

        let (compact_device_queries_right, compact_device_sums_right) = record_stream_time!(
            &self.device_manager,
            &self.streams[0],
            events,
            "query_preprocess",
            self.enable_debug_timing,
            {
                // This needs to be MAX_BATCH_SIZE, even though the query can be shorter to have
                // enough padding for GEMM
                let compact_device_queries_right = compact_query_right.htod_transfer(
                    &self.device_manager,
                    &self.streams[0],
                    self.max_batch_size,
                )?;

                let compact_device_sums_right = compact_device_queries_right.query_sums(
                    &self.codes_engine,
                    &self.masks_engine,
                    &self.streams[0],
                    &self.cublas_handles[0],
                )?;

                (compact_device_queries_right, compact_device_sums_right)
            }
        );

        tracing::info!("Comparing right eye queries against DB and self");
        self.compare_query_against_db_and_self(
            &compact_device_queries_right,
            &compact_device_sums_right,
            &mut events,
            Eye::Right,
            batch_size,
        );

        ///////////////////////////////////////////////////////////////////
        // MERGE LEFT & RIGHT results
        ///////////////////////////////////////////////////////////////////
        tracing::info!("Joining both sides");
        // Merge results and fetch matching indices
        // Format: host_results[device_index][query_index]
        self.distance_comparator.join_db_matches(
            &self.db_match_list_left,
            &self.db_match_list_right,
            &self.final_results,
            &self.current_db_sizes,
            &self.streams[0],
        );

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

        // Evaluate the results across devices
        // Format: merged_results[query_index]
        let mut merged_results =
            get_merged_results(&host_results, self.device_manager.device_count());

        // List the indices of the queries that did not match.
        let insertion_list = merged_results
            .iter()
            .enumerate()
            .filter(|&(idx, &num)| {
                num == NON_MATCH_ID
                    // Filter-out supermatchers on both sides (TODO: remove this in the future)
                    && partial_match_counters_left[idx] <= SUPERMATCH_THRESHOLD
                    && partial_match_counters_right[idx] <= SUPERMATCH_THRESHOLD
            })
            .map(|(idx, _num)| idx)
            .collect::<Vec<_>>();

        // Spread the insertions across devices.
        let insertion_list = distribute_insertions(&insertion_list, &self.current_db_sizes);

        // Calculate the new indices for the inserted queries
        let matches = calculate_insertion_indices(
            &mut merged_results,
            &insertion_list,
            &self.current_db_sizes,
            batch_size,
        );

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

        // Write back to in-memory db
        let previous_total_db_size = self.current_db_sizes.iter().sum::<usize>();
        let n_insertions = insertion_list.iter().map(|x| x.len()).sum::<usize>();

        // Check if we actually have space left to write the new entries
        if previous_total_db_size + n_insertions > self.max_db_size {
            tracing::error!(
                "Cannot write new entries, since DB size would be exceeded, current: {}, batch \
                 insertions: {}, max: {}",
                previous_total_db_size,
                n_insertions,
                self.max_db_size
            );
            eyre::bail!("DB size exceeded");
        }

        if self.disable_persistence {
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
                        for insertion_idx in insertion_list[i].clone() {
                            write_db_at_index(
                                &self.left_code_db_slices,
                                &self.left_mask_db_slices,
                                &self.right_code_db_slices,
                                &self.right_mask_db_slices,
                                &compact_device_queries_left,
                                &compact_device_sums_left,
                                &compact_device_queries_right,
                                &compact_device_sums_right,
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
                    }
                }
            );
        }

        // Pass to internal sender thread
        return_channel
            .send(ServerJobResult {
                merged_results,
                request_ids: batch.request_ids,
                metadata: batch.metadata,
                matches,
                match_ids: match_ids_filtered,
                partial_match_ids_left,
                partial_match_ids_right,
                store_left: query_store_left,
                store_right: query_store_right,
                deleted_ids: batch.deletion_requests_indices,
                matched_batch_request_ids,
            })
            .unwrap();

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
            "Batch took {:?} [{:.2} Melems/s]",
            now.elapsed(),
            processed_mil_elements_per_second
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

        Ok(())
    }

    fn compare_query_against_db_and_self(
        &mut self,
        compact_device_queries: &DeviceCompactQuery,
        compact_device_sums: &DeviceCompactSums,
        events: &mut HashMap<&str, Vec<Vec<CUevent>>>,
        eye_db: Eye,
        batch_size: usize,
    ) {
        let batch_streams = &self.streams[0];
        let batch_cublas = &self.cublas_handles[0];

        // which database are we querying against
        let (code_db_slices, mask_db_slices) = match eye_db {
            Eye::Left => (&self.left_code_db_slices, &self.left_mask_db_slices),
            Eye::Right => (&self.right_code_db_slices, &self.right_mask_db_slices),
        };

        let (db_match_bitmap, batch_match_bitmap) = match eye_db {
            Eye::Left => (&self.db_match_list_left, &self.batch_match_list_left),
            Eye::Right => (&self.db_match_list_right, &self.batch_match_list_right),
        };

        let (
            match_distances_buffers_codes,
            match_distances_buffers_masks,
            match_distances_counters,
            match_distances_indices,
        ) = match eye_db {
            Eye::Left => (
                &self.match_distances_buffer_codes_left,
                &self.match_distances_buffer_masks_left,
                &self.match_distances_counter_left,
                &self.match_distances_indices_left,
            ),
            Eye::Right => (
                &self.match_distances_buffer_codes_right,
                &self.match_distances_buffer_masks_right,
                &self.match_distances_counter_right,
                &self.match_distances_indices_right,
            ),
        };

        let mut total_distance_counter = 0;
        for (i, device) in self.device_manager.devices().iter().enumerate() {
            let counter = device.dtoh_sync_copy(&match_distances_counters[i]).unwrap();
            total_distance_counter += counter[0];
        }

        tracing::info!("Matching distances collected: {}", total_distance_counter);

        if total_distance_counter >= self.match_distances_buffer_size as u32 {
            let now = std::time::Instant::now();
            tracing::info!("Collected enough match distances, starting bucket calculation");

            self.device_manager.await_streams(&batch_streams);

            let indices = match_distances_indices
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    dtoh_on_stream_sync(x, &self.device_manager.device(i), &batch_streams[i])
                        .unwrap()
                })
                .collect::<Vec<_>>();

            let resort_indices = (0..indices.len())
                .map(|i| {
                    let mut resort_indices = (0..indices[i].len()).collect::<Vec<_>>();
                    resort_indices.sort_by_key(|&j| indices[i][j]);
                    resort_indices
                })
                .collect::<Vec<_>>();

            let shares = sort_shares_by_indices(
                &self.device_manager,
                &resort_indices,
                match_distances_buffers_codes,
                batch_streams,
            );

            let match_distances_buffers_codes_view =
                shares.iter().map(|x| x.as_view()).collect::<Vec<_>>();

            let shares = sort_shares_by_indices(
                &self.device_manager,
                &resort_indices,
                match_distances_buffers_masks,
                batch_streams,
            );

            let match_distances_buffers_masks_view =
                shares.iter().map(|x| x.as_view()).collect::<Vec<_>>();

            self.phase2_buckets.compare_multiple_thresholds(
                &match_distances_buffers_codes_view,
                &match_distances_buffers_masks_view,
                batch_streams,
                &(1..=self.n_buckets)
                    .map(|x: usize| {
                        Circuits::translate_threshold_a(
                            MATCH_THRESHOLD_RATIO / (self.n_buckets as f64) * (x as f64),
                        ) as u16
                    })
                    .collect::<Vec<_>>(),
                &mut self.buckets,
            );

            let buckets = self
                .phase2_buckets
                .open_buckets(&self.buckets, batch_streams);

            let mut results = String::new();
            for i in 0..buckets.len() {
                let step = MATCH_THRESHOLD_RATIO / (self.n_buckets as f64);
                let previous_threshold = step * (i as f64);
                let threshold = step * (i as f64 + 1.0);
                let previous_count = if i == 0 { 0 } else { buckets[i - 1] };
                let count = buckets[i] - previous_count;
                results.push_str(&format!(
                    "    {:.3}-{:.3}: {:?}\n",
                    previous_threshold, threshold, count
                ));
            }
            tracing::info!("Bucket results:\n{}", results);

            let reset_all_buffers =
                |counter: &[CudaSlice<u32>],
                 indices: &[CudaSlice<u32>],
                 codes: &[ChunkShare<u16>],
                 masks: &[ChunkShare<u16>],
                 buckets: &ChunkShare<u32>| {
                    reset_slice(self.device_manager.devices(), counter, 0, batch_streams);
                    reset_slice(self.device_manager.devices(), indices, 0xff, batch_streams);
                    reset_share(self.device_manager.devices(), masks, 0xff, batch_streams);
                    reset_share(self.device_manager.devices(), codes, 0xff, batch_streams);
                    reset_single_share(self.device_manager.devices(), buckets, 0, batch_streams, 0);
                };

            match eye_db {
                Eye::Left => {
                    reset_all_buffers(
                        &self.match_distances_counter_left,
                        &self.match_distances_indices_left,
                        &self.match_distances_buffer_codes_left,
                        &self.match_distances_buffer_masks_left,
                        &self.buckets,
                    );
                }
                Eye::Right => {
                    reset_all_buffers(
                        &self.match_distances_counter_right,
                        &self.match_distances_indices_right,
                        &self.match_distances_buffer_codes_right,
                        &self.match_distances_buffer_masks_right,
                        &self.buckets,
                    );
                }
            }

            self.device_manager.await_streams(&batch_streams);

            tracing::info!("Bucket calculation took {:?}", now.elapsed());
        }

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
        // ---- END BATCH DEDUP ----

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
                .map(|&s| (s.max(1).div_ceil(64) * 64))
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
                batch_streams,
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
                            &code_dots,
                            &mask_dots,
                            batch_size,
                            self.match_distances_buffer_size,
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

        // Reset the results buffers for reuse
        for dst in &[&self.results, &self.batch_results, &self.final_results] {
            reset_slice(self.device_manager.devices(), dst, 0xff, &self.streams[0]);
        }
    }

    fn sync_batch_entries(
        &mut self,
        valid_entries: &[bool],
        max_batch_size: usize,
        batch_hash: &[u8],
    ) -> eyre::Result<Vec<bool>> {
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
                    "Batch mismatch with node {}. Queues seem to be out of sync.",
                    i
                );
                metrics::counter!("batch.mismatch").increment(1);
                return Err(eyre!(
                    "Batch mismatch with node {}. Queues seem to be out of sync.",
                    i
                ));
            }
        }

        Ok(valid_merged)
    }

    fn prepare_deletion_shares(&self) -> eyre::Result<(DeviceCompactQuery, DeviceCompactSums)> {
        let (dummy_code_share, dummy_mask_share) = get_dummy_shares_for_deletion(self.party_id);
        let compact_query = {
            let code = preprocess_query(
                &dummy_code_share
                    .all_rotations()
                    .into_iter()
                    .flat_map(|e| e.coefs)
                    .collect::<Vec<_>>(),
            );
            let mask = preprocess_query(
                &dummy_mask_share
                    .all_rotations()
                    .into_iter()
                    .flat_map(|e| e.coefs)
                    .collect::<Vec<_>>(),
            );
            CompactQuery {
                code_query:        code.clone(),
                mask_query:        mask.clone(),
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
fn derive_seed(seed: [u32; 8], kdf_salt: &Salt, nonce: usize) -> eyre::Result<[u32; 8]> {
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
    match_distances_indices: &[CudaSlice<u32>],
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
        code_dots,
        mask_dots,
        batch_size,
        max_bucket_distances,
        streams,
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

fn reset_share<T>(
    devs: &[Arc<CudaDevice>],
    dst: &[ChunkShare<T>],
    value: u8,
    streams: &[CudaStream],
) {
    for i in 0..devs.len() {
        reset_single_share(devs, &dst[i], value, streams, i);
    }
}

fn reset_slice<T>(
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
                mem::size_of::<u32>() * 15 + src_index * mem::size_of::<u32>() * ROTATIONS,
                mem::size_of::<u32>(),
                streams[device_index].stream,
            );
        }
    }
}

pub fn get_dummy_shares_for_deletion(
    party_id: usize,
) -> (GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare) {
    let mut rng: StdRng = StdRng::seed_from_u64(0);
    let dummy: IrisCode = IrisCode::default();
    let iris_share: GaloisRingIrisCodeShare =
        GaloisRingIrisCodeShare::encode_iris_code(&dummy.code, &dummy.mask, &mut rng)[party_id]
            .clone();
    let mask_share: GaloisRingTrimmedMaskCodeShare =
        GaloisRingIrisCodeShare::encode_mask_code(&dummy.mask, &mut rng)[party_id]
            .clone()
            .into();
    (iris_share, mask_share)
}

fn sort_shares_by_indices(
    device_manager: &DeviceManager,
    resort_indices: &[Vec<usize>],
    shares: &[ChunkShare<u16>],
    streams: &[CudaStream],
) -> Vec<ChunkShare<u16>> {
    let a = shares
        .iter()
        .enumerate()
        .map(|(i, x)| dtoh_on_stream_sync(&x.a, &device_manager.device(i), &streams[i]).unwrap())
        .collect::<Vec<_>>();

    let b = shares
        .iter()
        .enumerate()
        .map(|(i, x)| dtoh_on_stream_sync(&x.b, &device_manager.device(i), &streams[i]).unwrap())
        .collect::<Vec<_>>();

    (0..a.len())
        .map(|i| {
            let new_a = resort_indices[i]
                .iter()
                .map(|&j| a[i][j])
                .collect::<Vec<_>>();
            let a = htod_on_stream_sync(&new_a, &device_manager.device(i), &streams[i]).unwrap();

            let new_b = resort_indices[i]
                .iter()
                .map(|&j| b[i][j])
                .collect::<Vec<_>>();
            let b = htod_on_stream_sync(&new_b, &device_manager.device(i), &streams[i]).unwrap();

            ChunkShare::new(a, b)
        })
        .collect::<Vec<_>>()
}
