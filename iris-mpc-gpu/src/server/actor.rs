use super::{BatchQuery, Eye, ServerJob, ServerJobResult};
use crate::{
    dot::{
        distance_comparator::DistanceComparator,
        share_db::{preprocess_query, ShareDB, SlicedProcessedDatabase},
        IRIS_CODE_LENGTH, MASK_CODE_LENGTH, ROTATIONS,
    },
    helpers::{
        self,
        comm::NcclComm,
        device_manager::DeviceManager,
        query_processor::{CompactQuery, DeviceCompactQuery, DeviceCompactSums},
    },
    threshold_ring::protocol::{ChunkShare, Circuits},
};
use cudarc::{
    cublas::CudaBlas,
    driver::{
        result::{self, event::elapsed},
        sys::CUevent,
        CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice,
    },
};
use eyre::eyre;
use futures::{Future, FutureExt};
use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    iris_db::iris::IrisCode,
    IrisCodeDbSlice,
};
use itertools::Itertools;
use rand::{rngs::StdRng, SeedableRng};
use ring::hkdf::{Algorithm, Okm, Salt, HKDF_SHA256};
use std::{collections::HashMap, mem, sync::Arc, time::Instant};
use tokio::sync::{mpsc, oneshot};

macro_rules! record_stream_time {
    ($manager:expr, $streams:expr, $map:expr, $label:expr, $block:block) => {
        let evt0 = $manager.create_events();
        let evt1 = $manager.create_events();
        $manager.record_event($streams, &evt0);
        $block
        $manager.record_event($streams, &evt1);
        $map.entry($label).or_default().extend(vec![evt0, evt1])
    };
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

const DB_CHUNK_SIZE: usize = 512;
pub struct ServerActor {
    job_queue:              mpsc::Receiver<ServerJob>,
    device_manager:         Arc<DeviceManager>,
    party_id:               usize,
    // engines
    codes_engine:           ShareDB,
    masks_engine:           ShareDB,
    batch_codes_engine:     ShareDB,
    batch_masks_engine:     ShareDB,
    phase2:                 Circuits,
    phase2_batch:           Circuits,
    distance_comparator:    DistanceComparator,
    comms:                  Vec<Arc<NcclComm>>,
    // DB slices
    left_code_db_slices:    SlicedProcessedDatabase,
    left_mask_db_slices:    SlicedProcessedDatabase,
    right_code_db_slices:   SlicedProcessedDatabase,
    right_mask_db_slices:   SlicedProcessedDatabase,
    streams:                Vec<Vec<CudaStream>>,
    cublas_handles:         Vec<Vec<CudaBlas>>,
    results:                Vec<CudaSlice<u32>>,
    batch_results:          Vec<CudaSlice<u32>>,
    final_results:          Vec<CudaSlice<u32>>,
    db_match_list_left:     Vec<CudaSlice<u64>>,
    db_match_list_right:    Vec<CudaSlice<u64>>,
    batch_match_list_left:  Vec<CudaSlice<u64>>,
    batch_match_list_right: Vec<CudaSlice<u64>>,
    current_db_sizes:       Vec<usize>,
    query_db_size:          Vec<usize>,
    max_batch_size:         usize,
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
    ) -> eyre::Result<(Self, ServerActorHandle)> {
        let device_manager = Arc::new(DeviceManager::init());
        Self::new_with_device_manager(
            party_id,
            chacha_seeds,
            device_manager,
            job_queue_size,
            max_db_size,
            max_batch_size,
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
    ) -> eyre::Result<Self> {
        let mut kdf_nonce = 0;
        let kdf_salt: Salt = Salt::new(HKDF_SHA256, b"IRIS_MPC");
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

        let left_code_db_slices = codes_engine.alloc_db(max_db_size);
        let left_mask_db_slices = masks_engine.alloc_db(max_db_size);
        let right_code_db_slices = codes_engine.alloc_db(max_db_size);
        let right_mask_db_slices = masks_engine.alloc_db(max_db_size);

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

    pub fn load_single_record(
        &mut self,
        index: usize,
        left_code: &[u16],
        left_mask: &[u16],
        right_code: &[u16],
        right_mask: &[u16],
    ) {
        ShareDB::load_single_record(
            index,
            &self.left_code_db_slices.code_gr,
            left_code,
            self.device_manager.device_count(),
            IRIS_CODE_LENGTH,
        );
        ShareDB::load_single_record(
            index,
            &self.left_mask_db_slices.code_gr,
            left_mask,
            self.device_manager.device_count(),
            MASK_CODE_LENGTH,
        );
        ShareDB::load_single_record(
            index,
            &self.right_code_db_slices.code_gr,
            right_code,
            self.device_manager.device_count(),
            IRIS_CODE_LENGTH,
        );
        ShareDB::load_single_record(
            index,
            &self.right_mask_db_slices.code_gr,
            right_mask,
            self.device_manager.device_count(),
            MASK_CODE_LENGTH,
        );
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

    fn process_batch_query(
        &mut self,
        batch: BatchQuery,
        return_channel: oneshot::Sender<ServerJobResult>,
    ) -> eyre::Result<()> {
        let now = Instant::now();
        let mut events: HashMap<&str, Vec<Vec<CUevent>>> = HashMap::new();

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
                && batch_size * ROTATIONS == batch.db_right.mask.len(),
            "Query batch sizes mismatch"
        );

        ///////////////////////////////////////////////////////////////////
        // PERFORM DELETIONS (IF ANY)
        ///////////////////////////////////////////////////////////////////

        if !batch.deletion_requests_indices.is_empty() {
            // Prepare dummy deletion shares
            let (dummy_queries, dummy_sums) = self.prepare_deletion_shares()?;

            // Overwrite the in-memory db
            for deletion_index in batch.deletion_requests_indices.clone() {
                let device_index = deletion_index % self.device_manager.device_count() as u32;
                let device_db_index = deletion_index / self.device_manager.device_count() as u32;
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

        let valid_entries = self.sync_batch_entries(&batch.valid_entries)?;
        let valid_entry_idxs = valid_entries.iter().positions(|&x| x).collect::<Vec<_>>();
        batch_size = valid_entry_idxs.len();
        batch.retain(&valid_entry_idxs);

        ///////////////////////////////////////////////////////////////////
        // COMPARE LEFT EYE QUERIES
        ///////////////////////////////////////////////////////////////////

        // *Query* variant including Lagrange interpolation.
        let compact_query_left = {
            let code_query = preprocess_query(
                &batch
                    .query_left
                    .code
                    .into_iter()
                    .flat_map(|e| e.coefs)
                    .collect::<Vec<_>>(),
            );

            let mask_query = preprocess_query(
                &batch
                    .query_left
                    .mask
                    .into_iter()
                    .flat_map(|e| e.coefs)
                    .collect::<Vec<_>>(),
            );
            // *Storage* variant (no interpolation).
            let code_query_insert = preprocess_query(
                &batch
                    .db_left
                    .code
                    .into_iter()
                    .flat_map(|e| e.coefs)
                    .collect::<Vec<_>>(),
            );
            let mask_query_insert = preprocess_query(
                &batch
                    .db_left
                    .mask
                    .into_iter()
                    .flat_map(|e| e.coefs)
                    .collect::<Vec<_>>(),
            );
            CompactQuery {
                code_query,
                mask_query,
                code_query_insert,
                mask_query_insert,
            }
        };
        let query_store_left = batch.store_left;

        // THIS needs to be max_batch_size, even though the query can be shorter to have
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

        self.compare_query_against_db_and_self(
            &compact_device_queries_left,
            &compact_device_sums_left,
            &mut events,
            Eye::Left,
        );

        ///////////////////////////////////////////////////////////////////
        // COMPARE RIGHT EYE QUERIES
        ///////////////////////////////////////////////////////////////////

        // *Query* variant including Lagrange interpolation.
        let compact_query_right = {
            let code_query = preprocess_query(
                &batch
                    .query_right
                    .code
                    .into_iter()
                    .flat_map(|e| e.coefs)
                    .collect::<Vec<_>>(),
            );

            let mask_query = preprocess_query(
                &batch
                    .query_right
                    .mask
                    .into_iter()
                    .flat_map(|e| e.coefs)
                    .collect::<Vec<_>>(),
            );
            // *Storage* variant (no interpolation).
            let code_query_insert = preprocess_query(
                &batch
                    .db_right
                    .code
                    .into_iter()
                    .flat_map(|e| e.coefs)
                    .collect::<Vec<_>>(),
            );
            let mask_query_insert = preprocess_query(
                &batch
                    .db_right
                    .mask
                    .into_iter()
                    .flat_map(|e| e.coefs)
                    .collect::<Vec<_>>(),
            );
            CompactQuery {
                code_query,
                mask_query,
                code_query_insert,
                mask_query_insert,
            }
        };
        let query_store_right = batch.store_right;

        // THIS needs to be MAX_BATCH_SIZE, even though the query can be shorter to have
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

        self.compare_query_against_db_and_self(
            &compact_device_queries_right,
            &compact_device_sums_right,
            &mut events,
            Eye::Right,
        );

        ///////////////////////////////////////////////////////////////////
        // MERGE LEFT & RIGHT results
        ///////////////////////////////////////////////////////////////////

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

        // Evaluate the results across devices
        // Format: merged_results[query_index]
        let mut merged_results =
            get_merged_results(&host_results, self.device_manager.device_count());

        // List the indices of the queries that did not match.
        let insertion_list = merged_results
            .iter()
            .enumerate()
            .filter(|&(_idx, &num)| num == NON_MATCH_ID)
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

        // Fetch and truncate the match counters
        let match_counters_devices = self
            .distance_comparator
            .fetch_match_counters()
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
        let match_ids = self
            .distance_comparator
            .fetch_all_match_ids(match_counters_devices);

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

        // Write back to in-memory db
        let previous_total_db_size = self.current_db_sizes.iter().sum::<usize>();

        record_stream_time!(
            &self.device_manager,
            &self.streams[0],
            events,
            "db_write",
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

                    // DEBUG
                    tracing::debug!(
                        "Updating DB size on device {}: {:?}",
                        i,
                        self.current_db_sizes[i]
                    );
                }
            }
        );

        // Pass to internal sender thread
        return_channel
            .send(ServerJobResult {
                merged_results,
                request_ids: batch.request_ids,
                matches,
                match_ids,
                store_left: query_store_left,
                store_right: query_store_right,
                deleted_ids: batch.deletion_requests_indices,
            })
            .unwrap();

        // Wait for all streams before get timings
        self.device_manager.await_streams(&self.streams[0]);
        self.device_manager.await_streams(&self.streams[1]);

        // Reset the results buffers for reuse
        for dst in &[
            &self.db_match_list_left,
            &self.db_match_list_right,
            &self.batch_match_list_left,
            &self.batch_match_list_right,
        ] {
            reset_slice(self.device_manager.devices(), dst, 0, &self.streams[0]);
        }

        reset_slice(
            self.device_manager.devices(),
            &self.distance_comparator.match_counters,
            0,
            &self.streams[0],
        );

        // ---- END RESULT PROCESSING ----
        log_timers(events);

        tracing::info!(
            "Batch took {:?} [{:.2} Melems/s]",
            now.elapsed(),
            (self.max_batch_size * previous_total_db_size) as f64
                / now.elapsed().as_secs_f64()
                / 1e6
        );
        Ok(())
    }

    fn compare_query_against_db_and_self(
        &mut self,
        compact_device_queries: &DeviceCompactQuery,
        compact_device_sums: &DeviceCompactSums,
        events: &mut HashMap<&str, Vec<Vec<CUevent>>>,
        eye_db: Eye,
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

        // ---- START BATCH DEDUP ----
        tracing::debug!(party_id = self.party_id, "Starting batch deduplication");

        record_stream_time!(&self.device_manager, batch_streams, events, "batch_dot", {
            compact_device_queries.compute_dot_products(
                &mut self.batch_codes_engine,
                &mut self.batch_masks_engine,
                &self.query_db_size,
                0,
                batch_streams,
                batch_cublas,
            );

            compact_device_sums.compute_dot_reducers(
                &mut self.batch_codes_engine,
                &mut self.batch_masks_engine,
                &self.query_db_size,
                0,
                batch_streams,
            );
        });

        record_stream_time!(
            &self.device_manager,
            batch_streams,
            events,
            "batch_reshare",
            {
                self.batch_codes_engine
                    .reshare_results(&self.query_db_size, batch_streams);
                self.batch_masks_engine
                    .reshare_results(&self.query_db_size, batch_streams);
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
            {
                self.phase2_batch.compare_threshold_masked_many(
                    &code_dots_batch,
                    &mask_dots_batch,
                    batch_streams,
                );
            }
        );

        let res = self.phase2_batch.take_result_buffer();
        let chunk_size = self.phase2_batch.chunk_size();
        open(
            &mut self.phase2_batch,
            &res,
            &self.distance_comparator,
            batch_match_bitmap,
            chunk_size,
            &db_sizes_batch,
            &db_sizes_batch,
            0,
            &db_sizes_batch,
            batch_streams,
        );
        self.phase2_batch.return_result_buffer(res);

        tracing::debug!(party_id = self.party_id, "Finished batch deduplication");
        // ---- END BATCH DEDUP ----

        // Create new initial events
        let mut current_dot_event = self.device_manager.create_events();
        let mut next_dot_event = self.device_manager.create_events();
        let mut current_exchange_event = self.device_manager.create_events();
        let mut next_exchange_event = self.device_manager.create_events();
        let mut current_phase2_event = self.device_manager.create_events();
        let mut next_phase2_event = self.device_manager.create_events();

        // ---- START DATABASE DEDUP ----
        tracing::debug!(party_id = self.party_id, "Start DB deduplication");
        let mut db_chunk_idx = 0;
        loop {
            tracing::debug!(
                party_id = self.party_id,
                chunk = db_chunk_idx,
                "starting chunk"
            );

            let request_streams = &self.streams[db_chunk_idx % 2];
            let request_cublas_handles = &self.cublas_handles[db_chunk_idx % 2];

            let offset = db_chunk_idx * DB_CHUNK_SIZE;
            let chunk_size = self
                .current_db_sizes
                .iter()
                .map(|s| (s - DB_CHUNK_SIZE * db_chunk_idx).clamp(0, DB_CHUNK_SIZE))
                .collect::<Vec<_>>();

            // We need to pad the chunk size to be a multiple of 4, because the underlying
            // `gemm_ex` expects this. We filter out potential "phantom matches"
            // for the padded data in the `open` later.
            let dot_chunk_size = chunk_size
                .iter()
                .map(|s| s.div_ceil(4) * 4)
                .collect::<Vec<_>>();

            // First stream doesn't need to wait
            if db_chunk_idx == 0 {
                self.device_manager
                    .record_event(request_streams, &current_dot_event);
                self.device_manager
                    .record_event(request_streams, &current_exchange_event);
                self.device_manager
                    .record_event(request_streams, &current_phase2_event);
            }

            tracing::debug!(
                party_id = self.party_id,
                chunk = db_chunk_idx,
                "waiting for dot-event"
            );
            self.device_manager
                .await_event(request_streams, &current_dot_event);

            // ---- START PHASE 1 ----
            record_stream_time!(&self.device_manager, batch_streams, events, "db_dot", {
                compact_device_queries.dot_products_against_db(
                    &mut self.codes_engine,
                    &mut self.masks_engine,
                    code_db_slices,
                    mask_db_slices,
                    &dot_chunk_size,
                    offset,
                    request_streams,
                    request_cublas_handles,
                );
            });

            // wait for the exchange result buffers to be ready
            tracing::debug!(
                party_id = self.party_id,
                chunk = db_chunk_idx,
                "waiting for exchange-event"
            );
            self.device_manager
                .await_event(request_streams, &current_exchange_event);

            record_stream_time!(&self.device_manager, batch_streams, events, "db_reduce", {
                compact_device_sums.compute_dot_reducer_against_db(
                    &mut self.codes_engine,
                    &mut self.masks_engine,
                    code_db_slices,
                    mask_db_slices,
                    &dot_chunk_size,
                    offset,
                    request_streams,
                );
            });

            tracing::debug!(
                party_id = self.party_id,
                chunk = db_chunk_idx,
                "recording dot-event"
            );
            self.device_manager
                .record_event(request_streams, &next_dot_event);

            record_stream_time!(&self.device_manager, batch_streams, events, "db_reshare", {
                self.codes_engine
                    .reshare_results(&dot_chunk_size, request_streams);
                self.masks_engine
                    .reshare_results(&dot_chunk_size, request_streams);
            });

            // ---- END PHASE 1 ----

            tracing::debug!(
                party_id = self.party_id,
                chunk = db_chunk_idx,
                "waiting for phase2-event"
            );
            self.device_manager
                .await_event(request_streams, &current_phase2_event);

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
                    batch_streams,
                    events,
                    "db_threshold",
                    {
                        self.phase2.compare_threshold_masked_many(
                            &code_dots,
                            &mask_dots,
                            request_streams,
                        );
                    }
                );
                // we can now record the exchange event since the phase 2 is no longer using the
                // code_dots/mask_dots which are just reinterpretations of the exchange result
                // buffers
                tracing::debug!(
                    party_id = self.party_id,
                    chunk = db_chunk_idx,
                    "recording exchange-event"
                );
                self.device_manager
                    .record_event(request_streams, &next_exchange_event);

                let res = self.phase2.take_result_buffer();
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
                    request_streams,
                );
                self.phase2.return_result_buffer(res);
            }
            tracing::debug!(
                party_id = self.party_id,
                chunk = db_chunk_idx,
                "recording phase2-event"
            );
            self.device_manager
                .record_event(request_streams, &next_phase2_event);

            // ---- END PHASE 2 ----

            // Update events for synchronization
            current_dot_event = next_dot_event;
            current_exchange_event = next_exchange_event;
            current_phase2_event = next_phase2_event;
            next_dot_event = self.device_manager.create_events();
            next_exchange_event = self.device_manager.create_events();
            next_phase2_event = self.device_manager.create_events();

            // Increment chunk index
            db_chunk_idx += 1;

            tracing::debug!(
                party_id = self.party_id,
                chunk = db_chunk_idx,
                "finished chunk"
            );

            // Break if we reached the end of the database
            if db_chunk_idx * DB_CHUNK_SIZE >= *self.current_db_sizes.iter().max().unwrap() {
                break;
            }
        }
        // ---- END DATABASE DEDUP ----

        // Wait for protocol to finish
        tracing::debug!(party_id = self.party_id, "waiting for db search to finish");
        self.device_manager.await_streams(&self.streams[0]);
        self.device_manager.await_streams(&self.streams[1]);
        tracing::debug!(party_id = self.party_id, "db search finished");

        // Reset the results buffers for reuse
        for dst in &[&self.results, &self.batch_results, &self.final_results] {
            reset_slice(self.device_manager.devices(), dst, 0xff, &self.streams[0]);
        }
    }

    fn sync_batch_entries(&mut self, valid_entries: &[bool]) -> eyre::Result<Vec<bool>> {
        let mut buffer = self
            .device_manager
            .device(0)
            .alloc_zeros(valid_entries.len() * self.comms[0].world_size())
            .unwrap();

        let buffer_self = self
            .device_manager
            .device(0)
            .htod_copy(valid_entries.iter().map(|&x| x as u8).collect::<Vec<_>>())?;

        self.comms[0]
            .all_gather(&buffer_self, &mut buffer)
            .map_err(|e| eyre!(format!("{:?}", e)))?;

        let results = self.device_manager.device(0).dtoh_sync_copy(&buffer)?;
        let results: Vec<_> = results
            .chunks_exact(results.len() / self.comms[0].world_size())
            .collect();

        let mut valid_merged = vec![];
        for i in 0..results[0].len() {
            valid_merged.push(
                [results[0][i], results[1][i], results[2][i]]
                    .iter()
                    .all(|&x| x == 1),
            );
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
        // TODO: send to metrics
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

        tracing::debug!(
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
            helpers::dtod_at_offset(
                db.code_gr.limb_0[device_index],
                dst_index * code_length,
                *query.limb_0[device_index].device_ptr(),
                code_length * 15 + src_index * code_length * ROTATIONS,
                code_length,
                streams[device_index].stream,
            );

            helpers::dtod_at_offset(
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
