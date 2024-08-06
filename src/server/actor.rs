use super::{BatchQuery, ServerJob, ServerJobResult};
use crate::{
    dot::{
        distance_comparator::DistanceComparator,
        share_db::{preprocess_query, ShareDB, SlicedProcessedDatabase},
        IRIS_CODE_LENGTH, ROTATIONS,
    },
    helpers::{self, device_manager::DeviceManager, query_processor::CompactQuery},
    setup::galois_engine::degree4::GaloisRingIrisCodeShare,
    threshold_ring::protocol::{ChunkShare, Circuits},
};
use cudarc::{
    cublas::CudaBlas,
    driver::{
        result::{self, event::elapsed, mem_prefetch_async},
        sys::{CUmemLocationType, CUmemLocation_st},
        CudaDevice, CudaSlice, CudaStream, DevicePtr,
    },
    nccl::Comm,
};
use futures::{Future, FutureExt};
use ring::hkdf::{Algorithm, Okm, Salt, HKDF_SHA256};
use std::{mem, sync::Arc, time::Instant};
use tokio::sync::{mpsc, oneshot};

#[allow(unused)]
macro_rules! debug_record_event {
    ($manager:expr, $streams:expr, $timers:expr) => {
        let evts = $manager.create_events();
        $manager.record_event($streams, &evts);
        $timers.push(evts);
    };
}

macro_rules! forget_vec {
    ($vec:expr) => {
        while let Some(item) = $vec.pop() {
            std::mem::forget(item);
        }
    };
}

fn prepare_query_shares(shares: Vec<GaloisRingIrisCodeShare>) -> Vec<Vec<u8>> {
    preprocess_query(&shares.into_iter().flat_map(|e| e.coefs).collect::<Vec<_>>())
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

const DB_SIZE: usize = 1 << 24;
const DB_BUFFER: usize = 8 * (1 << 16);
const DB_CHUNK_SIZE: usize = 1 << 15;
const N_QUERIES: usize = 64 * 8;
const QUERIES: usize = ROTATIONS * N_QUERIES;

pub struct ServerActor {
    job_queue:           mpsc::Receiver<ServerJob>,
    device_manager:      Arc<DeviceManager>,
    party_id:            usize,
    // engines
    codes_engine:        ShareDB,
    masks_engine:        ShareDB,
    batch_codes_engine:  ShareDB,
    batch_masks_engine:  ShareDB,
    phase2:              Circuits,
    phase2_batch:        Circuits,
    distance_comparator: DistanceComparator,
    // DB slices
    code_db_slices:      SlicedProcessedDatabase,
    mask_db_slices:      SlicedProcessedDatabase,
    streams:             Vec<Vec<CudaStream>>,
    memory_streams:      Vec<CudaStream>,
    cublas_handles:      Vec<Vec<CudaBlas>>,
    results:             Vec<CudaSlice<u32>>,
    batch_results:       Vec<CudaSlice<u32>>,
    final_results:       Vec<CudaSlice<u32>>,
    current_db_sizes:    Vec<usize>,
    query_db_size:       Vec<usize>,
}

const RESULTS_INIT_HOST: [u32; N_QUERIES * ROTATIONS] = [u32::MAX; N_QUERIES * ROTATIONS];
const FINAL_RESULTS_INIT_HOST: [u32; N_QUERIES] = [u32::MAX; N_QUERIES];

impl ServerActor {
    pub fn new(
        party_id: usize,
        chacha_seeds: ([u32; 8], [u32; 8]),
        codes_db: &[u16],
        masks_db: &[u16],
        job_queue_size: usize,
    ) -> eyre::Result<(Self, ServerActorHandle)> {
        let device_manager = Arc::new(DeviceManager::init());
        Self::new_with_device_manager(
            party_id,
            chacha_seeds,
            codes_db,
            masks_db,
            device_manager,
            job_queue_size,
        )
    }
    pub fn new_with_device_manager(
        party_id: usize,
        chacha_seeds: ([u32; 8], [u32; 8]),
        codes_db: &[u16],
        masks_db: &[u16],
        device_manager: Arc<DeviceManager>,
        job_queue_size: usize,
    ) -> eyre::Result<(Self, ServerActorHandle)> {
        let ids = device_manager.get_ids_from_magic(0);
        let comms = device_manager.instantiate_network_from_ids(party_id, ids);
        Self::new_with_device_manager_and_comms(
            party_id,
            chacha_seeds,
            codes_db,
            masks_db,
            device_manager,
            comms,
            job_queue_size,
        )
    }

    pub fn new_with_device_manager_and_comms(
        party_id: usize,
        chacha_seeds: ([u32; 8], [u32; 8]),
        codes_db: &[u16],
        masks_db: &[u16],
        device_manager: Arc<DeviceManager>,
        comms: Vec<Arc<Comm>>,
        job_queue_size: usize,
    ) -> eyre::Result<(Self, ServerActorHandle)> {
        assert_eq!(
            codes_db.len(),
            masks_db.len(),
            "Internal DB mismatch, codes and masks sizes differ"
        );

        let (tx, rx) = mpsc::channel(job_queue_size);
        let actor = Self::init(
            party_id,
            chacha_seeds,
            codes_db,
            masks_db,
            device_manager,
            comms,
            rx,
        )?;
        Ok((actor, ServerActorHandle { job_queue: tx }))
    }

    fn init(
        party_id: usize,
        chacha_seeds: ([u32; 8], [u32; 8]),
        codes_db: &[u16],
        masks_db: &[u16],
        device_manager: Arc<DeviceManager>,
        comms: Vec<Arc<Comm>>,
        job_queue: mpsc::Receiver<ServerJob>,
    ) -> eyre::Result<Self> {
        let mut kdf_nonce = 0;
        let kdf_salt: Salt = Salt::new(HKDF_SHA256, b"IRIS_MPC");

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

        println!("Creating engines 1");
        // Phase 1 Setup
        let codes_engine = ShareDB::init(
            party_id,
            device_manager.clone(),
            DB_CHUNK_SIZE,
            QUERIES,
            next_chacha_seeds(chacha_seeds)?,
            comms.clone(),
        );

        println!("Creating engines 2");

        let masks_engine = ShareDB::init(
            party_id,
            device_manager.clone(),
            DB_CHUNK_SIZE,
            QUERIES,
            next_chacha_seeds(chacha_seeds)?,
            comms.clone(),
        );

        let code_db_slices = codes_engine.load_db(codes_db, DB_SIZE, DB_SIZE + DB_BUFFER, true);
        let mask_db_slices = masks_engine.load_db(masks_db, DB_SIZE, DB_SIZE + DB_BUFFER, true);

        // Engines for inflight queries
        let batch_codes_engine = ShareDB::init(
            party_id,
            device_manager.clone(),
            QUERIES,
            QUERIES,
            next_chacha_seeds(chacha_seeds)?,
            comms.clone(),
        );

        let batch_masks_engine = ShareDB::init(
            party_id,
            device_manager.clone(),
            QUERIES,
            QUERIES,
            next_chacha_seeds(chacha_seeds)?,
            comms.clone(),
        );

        // Phase 2 Setup
        let phase2_chunk_size = QUERIES * DB_CHUNK_SIZE;

        // Not divided by GPU_COUNT since we do the work on all GPUs for simplicity,
        // also not padded to 2048 since we only require it to be a multiple of 64
        let phase2_batch_chunk_size = QUERIES * QUERIES;
        assert!(
            phase2_batch_chunk_size % 64 == 0,
            "Phase2 batch chunk size must be a multiple of 64"
        );
        assert!(
            phase2_chunk_size % 64 == 0,
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
            comms,
        );

        let distance_comparator = DistanceComparator::init(QUERIES, device_manager.clone());
        // Prepare streams etc.
        let mut streams = vec![];
        let mut cublas_handles = vec![];
        for _ in 0..2 {
            let tmp_streams = device_manager.fork_streams();
            cublas_handles.push(device_manager.create_cublas(&tmp_streams));
            streams.push(tmp_streams);
        }
        let memory_streams = device_manager.fork_streams();

        let final_results = distance_comparator.prepare_final_results();
        let results = distance_comparator.prepare_results();
        let batch_results = distance_comparator.prepare_results();

        let current_db_sizes: Vec<usize> =
            vec![DB_SIZE / device_manager.device_count(); device_manager.device_count()];
        let query_db_size = vec![QUERIES; device_manager.device_count()];

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
            code_db_slices,
            mask_db_slices,
            streams,
            memory_streams,
            cublas_handles,
            results,
            batch_results,
            final_results,
            current_db_sizes,
            query_db_size,
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

    fn process_batch_query(
        &mut self,
        batch: BatchQuery,
        return_channel: oneshot::Sender<ServerJobResult>,
    ) -> eyre::Result<()> {
        let now = Instant::now();
        // *Query* variant including Lagrange interpolation.
        let compact_query = {
            let code_query = prepare_query_shares(batch.query.code);
            let mask_query = prepare_query_shares(batch.query.mask);
            // *Storage* variant (no interpolation).
            let code_query_insert = prepare_query_shares(batch.db.code);
            let mask_query_insert = prepare_query_shares(batch.db.mask);
            CompactQuery {
                code_query,
                mask_query,
                code_query_insert,
                mask_query_insert,
            }
        };
        println!("Time for preprocess: {:?}", now.elapsed());
        let now = Instant::now();
        let query_store = batch.store;

        let batch_streams = &self.streams[0];
        let batch_cublas = &self.cublas_handles[0];

        // Transfer queries to device
        let compact_device_queries =
            compact_query.htod_transfer(&self.device_manager, batch_streams)?;

        let compact_device_sums = compact_device_queries.query_sums(
            &self.codes_engine,
            &self.masks_engine,
            batch_streams,
            batch_cublas,
        )?;

        self.device_manager.await_streams(batch_streams);
        println!("Time for query copy: {:?}", now.elapsed());
        let now = Instant::now();

        // ---- START BATCH DEDUP ----
        tracing::debug!(party_id = self.party_id, "Starting batch deduplication");
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

        self.batch_codes_engine
            .reshare_results(&self.query_db_size, batch_streams);
        self.batch_masks_engine
            .reshare_results(&self.query_db_size, batch_streams);

        let db_sizes_batch = vec![QUERIES; self.device_manager.device_count()];
        // TODO: remove
        let mut code_dots_batch = self.batch_codes_engine.result_chunk_shares(&db_sizes_batch);
        let mut mask_dots_batch = self.batch_masks_engine.result_chunk_shares(&db_sizes_batch);
        self.phase2_batch.compare_threshold_masked_many(
            &code_dots_batch,
            &mask_dots_batch,
            batch_streams,
        );
        let res = self.phase2_batch.take_result_buffer();
        let chunk_size = self.phase2_batch.chunk_size();
        open(
            &mut self.phase2_batch,
            &res,
            &self.distance_comparator,
            &self.batch_results,
            chunk_size,
            &db_sizes_batch,
            &db_sizes_batch,
            0,
            batch_streams,
        );
        self.phase2_batch.return_result_buffer(res);

        forget_vec!(code_dots_batch);
        forget_vec!(mask_dots_batch);
        tracing::debug!(party_id = self.party_id, "Finished batch deduplication");
        // ---- END BATCH DEDUP ----

        self.device_manager.await_streams(batch_streams);
        println!("Time for batch dedup: {:?}", now.elapsed());

        // Create new initial events
        let mut start_event = self.device_manager.create_events(false);
        let mut current_dot_event = self.device_manager.create_events(false);
        let mut next_dot_event = self.device_manager.create_events(false);
        let mut current_exchange_event = self.device_manager.create_events(false);
        let mut next_exchange_event = self.device_manager.create_events(false);
        let mut current_phase2_event = self.device_manager.create_events(false);
        let mut next_phase2_event = self.device_manager.create_events(false);

        // ---- START DATABASE DEDUP ----
        let now = Instant::now();
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

            // // Prefetch the next chunk
            // if chunk_size[0] == DB_CHUNK_SIZE {
            //     for i in 0..self.device_manager.device_count() {
            //         self.device_manager.device(i).bind_to_thread().unwrap();
            //         for ptr in &[
            //             self.code_db_slices.code_gr.limb_0[i],
            //             self.code_db_slices.code_gr.limb_1[i],
            //             self.mask_db_slices.code_gr.limb_0[i],
            //             self.mask_db_slices.code_gr.limb_1[i],
            //         ] {
            //             unsafe {
            //                 mem_prefetch_async(
            //                     ptr + (DB_CHUNK_SIZE * (db_chunk_idx + 1) *
            // IRIS_CODE_LENGTH)                         as u64,
            //                     DB_CHUNK_SIZE * IRIS_CODE_LENGTH,
            //                     CUmemLocation_st {
            //                         type_:
            // CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            // id:    i as i32,                     },
            //                     self.memory_streams[i].stream,
            //                 )
            //                 .unwrap();
            //             }
            //         }
            //     }
            // }

            tracing::debug!(
                party_id = self.party_id,
                chunk = db_chunk_idx,
                "waiting for dot-event"
            );
            self.device_manager
                .await_event(request_streams, &current_dot_event);

            self.device_manager
                .record_event(request_streams, &start_event);

            // ---- START PHASE 1 ----
            compact_device_queries.dot_products_against_db(
                &mut self.codes_engine,
                &mut self.masks_engine,
                &self.code_db_slices,
                &self.mask_db_slices,
                &dot_chunk_size,
                offset,
                request_streams,
                request_cublas_handles,
            );

            // wait for the exchange result buffers to be ready
            tracing::debug!(
                party_id = self.party_id,
                chunk = db_chunk_idx,
                "waiting for exchange-event"
            );
            self.device_manager
                .await_event(request_streams, &current_exchange_event);

            compact_device_sums.compute_dot_reducer_against_db(
                &mut self.codes_engine,
                &mut self.masks_engine,
                &self.code_db_slices,
                &self.mask_db_slices,
                &dot_chunk_size,
                offset,
                request_streams,
            );

            tracing::debug!(
                party_id = self.party_id,
                chunk = db_chunk_idx,
                "recording dot-event"
            );
            self.device_manager
                .record_event(request_streams, &next_dot_event);

            self.codes_engine
                .reshare_results(&dot_chunk_size, request_streams);
            self.masks_engine
                .reshare_results(&dot_chunk_size, request_streams);

            // ---- END PHASE 1 ----

            tracing::debug!(
                party_id = self.party_id,
                chunk = db_chunk_idx,
                "waiting for phase2-event"
            );
            self.device_manager
                .await_event(request_streams, &current_phase2_event);

            // ---- START PHASE 2 ----
            // TODO: remove
            let max_chunk_size = dot_chunk_size.iter().max().copied().unwrap();
            let phase_2_chunk_sizes = vec![max_chunk_size; self.device_manager.device_count()];
            let mut code_dots = self.codes_engine.result_chunk_shares(&phase_2_chunk_sizes);
            let mut mask_dots = self.masks_engine.result_chunk_shares(&phase_2_chunk_sizes);
            {
                assert_eq!(
                    (max_chunk_size * QUERIES) % 64,
                    0,
                    "Phase 2 input size must be a multiple of 64"
                );
                self.phase2.set_chunk_size(max_chunk_size * QUERIES / 64);
                self.phase2
                    .compare_threshold_masked_many(&code_dots, &mask_dots, request_streams);
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
                    &self.results,
                    max_chunk_size * QUERIES / 64,
                    &dot_chunk_size,
                    &chunk_size,
                    offset,
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

            forget_vec!(code_dots);
            forget_vec!(mask_dots);
            // ---- END PHASE 2 ----

            let elapsed =
                unsafe { elapsed(start_event[0] as *mut _, next_dot_event[0] as *mut _).unwrap() };
            println!("Time for dot {}: {:?}", db_chunk_idx, elapsed);

            // Update events for synchronization
            current_dot_event = next_dot_event;
            current_exchange_event = next_exchange_event;
            current_phase2_event = next_phase2_event;
            next_dot_event = self.device_manager.create_events(false);
            next_exchange_event = self.device_manager.create_events(false);
            next_phase2_event = self.device_manager.create_events(false);
            start_event = self.device_manager.create_events(false);

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
        tracing::debug!(party_id = self.party_id, "waiting for batch work to finish");
        self.device_manager.await_streams(&self.streams[0]);
        self.device_manager.await_streams(&self.streams[1]);
        tracing::debug!(party_id = self.party_id, "batch work finished");
        println!("Time for DB dedup: {:?}", now.elapsed());

        let now = Instant::now();

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

        // ---- START RESULT PROCESSING ----

        // Merge results and fetch matching indices
        // Format: host_results[device_index][query_index]
        self.distance_comparator.merge_results(
            &self.batch_results,
            &self.results,
            &self.final_results,
            &self.streams[0],
        );

        self.device_manager.await_streams(&self.streams[0]);

        // Fetch the final results (blocking)
        let host_results = self
            .distance_comparator
            .fetch_final_results(&self.final_results);

        // Evaluate the results across devices
        // Format: merged_results[query_index]
        let mut merged_results =
            get_merged_results(&host_results, self.device_manager.device_count());

        // List the indices of the queries that did not match.
        let insertion_list = merged_results
            .iter()
            .enumerate()
            .filter(|&(_idx, &num)| num == u32::MAX)
            .map(|(idx, _num)| idx)
            .collect::<Vec<_>>();

        // Spread the insertions across devices.
        let insertion_list = distribute_insertions(&insertion_list, &self.current_db_sizes);

        // Calculate the new indices for the inserted queries
        let matches = calculate_insertion_indices(
            &mut merged_results,
            &insertion_list,
            &self.current_db_sizes,
        );

        println!("Time for result processing: {:?}", now.elapsed());

        // for i in 0..self.device_manager.device_count() {
        //     self.device_manager.device(i).bind_to_thread().unwrap();
        //     for insertion_idx in insertion_list[i].clone() {
        //         // Append to codes and masks db
        //         for (db, query, sums) in [
        //             (
        //                 &self.code_db_slices,
        //                 &compact_device_queries.code_query_insert,
        //                 &compact_device_sums.code_query_insert,
        //             ),
        //             (
        //                 &self.mask_db_slices,
        //                 &compact_device_queries.mask_query_insert,
        //                 &compact_device_sums.mask_query_insert,
        //             ),
        //         ] {
        //             unsafe {
        //                 helpers::dtod_at_offset(
        //                     *db.code_gr.limb_0[i].device_ptr(),
        //                     self.current_db_sizes[i] * IRIS_CODE_LENGTH,
        //                     *query.limb_0[i].device_ptr(),
        //                     IRIS_CODE_LENGTH * 15 + insertion_idx * IRIS_CODE_LENGTH
        // * ROTATIONS,                     IRIS_CODE_LENGTH, self.streams[0][i].stream,
        //   );

        //                 helpers::dtod_at_offset(
        //                     *db.code_gr.limb_1[i].device_ptr(),
        //                     self.current_db_sizes[i] * IRIS_CODE_LENGTH,
        //                     *query.limb_1[i].device_ptr(),
        //                     IRIS_CODE_LENGTH * 15 + insertion_idx * IRIS_CODE_LENGTH
        // * ROTATIONS,                     IRIS_CODE_LENGTH, self.streams[0][i].stream,
        //   );

        //                 helpers::dtod_at_offset(
        //                     *db.code_sums_gr.limb_0[i].device_ptr(),
        //                     self.current_db_sizes[i] * mem::size_of::<u32>(),
        //                     *sums.limb_0[i].device_ptr(),
        //                     mem::size_of::<u32>() * 15
        //                         + insertion_idx * mem::size_of::<u32>() * ROTATIONS,
        //                     mem::size_of::<u32>(),
        //                     self.streams[0][i].stream,
        //                 );

        //                 helpers::dtod_at_offset(
        //                     *db.code_sums_gr.limb_1[i].device_ptr(),
        //                     self.current_db_sizes[i] * mem::size_of::<u32>(),
        //                     *sums.limb_1[i].device_ptr(),
        //                     mem::size_of::<u32>() * 15
        //                         + insertion_idx * mem::size_of::<u32>() * ROTATIONS,
        //                     mem::size_of::<u32>(),
        //                     self.streams[0][i].stream,
        //                 );
        //             }
        //         }
        //         self.current_db_sizes[i] += 1;
        //     }

        //     // DEBUG
        //     tracing::info!(
        //         "Updating DB size on device {}: {:?}",
        //         i,
        //         self.current_db_sizes[i]
        //     );
        // }

        // Pass to internal sender thread
        return_channel
            .send(ServerJobResult {
                merged_results: vec![],
                request_ids:    vec![],
                matches:        vec![],
                store:          query_store,
            })
            .unwrap();

        // Reset the results buffers for reuse
        reset_results(
            self.device_manager.devices(),
            &self.results,
            &RESULTS_INIT_HOST,
            &self.streams[0],
        );
        reset_results(
            self.device_manager.devices(),
            &self.batch_results,
            &RESULTS_INIT_HOST,
            &self.streams[0],
        );
        reset_results(
            self.device_manager.devices(),
            &self.final_results,
            &FINAL_RESULTS_INIT_HOST,
            &self.streams[0],
        );

        tracing::info!("CPU time of one iteration {:?}", now.elapsed());
        Ok(())
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

// fn prepare_query_shares(shares: Vec<GaloisRingIrisCodeShare>) -> ([u8;
// 25395200], [u8; 25395200]) {     let mut res0: [u8; 25395200] = [0u8; QUERIES
// * IRIS_CODE_LENGTH];     let mut res1 = [0u8; QUERIES * IRIS_CODE_LENGTH];
//   for (i, s) in shares.into_iter().flat_map(|e| e.coefs).enumerate() {
//   res0[i] = (s as i8 - 127 - 1) as u8; res1[i] = ((s >> 8) as i8 - 127 - 1)
//   as u8; } (res0, res1)
// }

#[allow(clippy::too_many_arguments)]
fn open(
    party: &mut Circuits,
    x: &[ChunkShare<u64>],
    distance_comparator: &DistanceComparator,
    results_ptrs: &[CudaSlice<u32>],
    chunk_size: usize,
    db_sizes: &[usize],
    real_db_sizes: &[usize],
    offset: usize,
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
        party
            .send_view(&res.b, party.next_id(), idx, streams)
            .unwrap();
        a.push(res.a);
        b.push(res.b);
    }
    for (idx, res) in x.iter().enumerate() {
        let mut res = res.get_offset(1, chunk_size);
        party
            .receive_view(&mut res.a, party.prev_id(), idx, streams)
            .unwrap();
        c.push(res.a);
    }
    cudarc::nccl::result::group_end().unwrap();

    distance_comparator.open_results(
        &a,
        &b,
        &c,
        results_ptrs,
        db_sizes,
        real_db_sizes,
        offset,
        streams,
    );
}

fn get_merged_results(host_results: &[Vec<u32>], n_devices: usize) -> Vec<u32> {
    let mut results = vec![];
    for j in 0..host_results[0].len() {
        let mut match_entry = u32::MAX;
        for i in 0..host_results.len() {
            let match_idx = host_results[i][j] * n_devices as u32 + i as u32;
            if host_results[i][j] != u32::MAX && match_idx < match_entry {
                match_entry = match_idx;
            }
        }

        results.push(match_entry);

        // DEBUG
        tracing::debug!(
            "Query {}: match={} [index: {}]",
            j,
            match_entry != u32::MAX,
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

fn reset_results(
    devs: &[Arc<CudaDevice>],
    dst: &[CudaSlice<u32>],
    src: &[u32],
    streams: &[CudaStream],
) {
    for i in 0..devs.len() {
        devs[i].bind_to_thread().unwrap();
        unsafe { result::memcpy_htod_async(*dst[i].device_ptr(), src, streams[i].stream) }.unwrap();
    }
}

fn calculate_insertion_indices(
    merged_results: &mut [u32],
    insertion_list: &[Vec<usize>],
    db_sizes: &[usize],
) -> Vec<bool> {
    let mut matches = vec![true; N_QUERIES];
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
