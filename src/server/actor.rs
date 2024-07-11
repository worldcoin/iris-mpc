use super::{BatchQuery, ServerJob, ServerJobResult};
use crate::{
    config::{Config, ServersConfig},
    dot::{
        distance_comparator::DistanceComparator,
        share_db::{preprocess_query, ShareDB},
        IRIS_CODE_LENGTH, ROTATIONS,
    },
    helpers::{self, device_manager::DeviceManager, task_monitor::TaskMonitor},
    setup::galois_engine::degree4::GaloisRingIrisCodeShare,
    threshold_ring::protocol::{ChunkShare, Circuits},
};
use cudarc::{
    cublas::CudaBlas,
    driver::{
        result::{
            self,
            event::{self, elapsed},
        },
        sys::{CUdeviceptr, CUevent_st},
        CudaDevice, CudaSlice, CudaStream,
    },
};
use futures::{Future, FutureExt};
use ring::hkdf::{Algorithm, Okm, Salt, HKDF_SHA256};
use static_assertions::const_assert;
use std::{
    mem,
    ops::IndexMut,
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, oneshot};

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

const DB_SIZE: usize = 8 * 1_000;
const DB_BUFFER: usize = 8 * 1_000;
const N_QUERIES: usize = 32;
/// The number of batches before a stream is re-used.
const MAX_BATCHES_BEFORE_REUSE: usize = 5;

/// The number of batches that are launched concurrently.
///
/// Code can run concurrently in:
/// - requests `N` and `N - 1`,
/// - requests `N` and `N - MAX_CONCURRENT_BATCHES`, because the next request
///   phase 1 awaits completion of the request `MAX_CONCURRENT_BATCHES` behind
///   it, or
/// - request `N` only, because phase 2 limits the critical section to one
///   batch.
/// - requests `N` and `N - 1`,
/// - requests `N` and `N - MAX_CONCURRENT_BATCHES`, because the next request
///   phase 1 awaits completion of the request `MAX_CONCURRENT_BATCHES` behind
///   it, or
/// - request `N` only, because phase 2 limits the critical section to one
///   batch.
const MAX_CONCURRENT_BATCHES: usize = 2;

const QUERIES: usize = ROTATIONS * N_QUERIES;

pub struct ServerActor {
    job_queue: mpsc::Receiver<ServerJob>,
    device_manager: Arc<DeviceManager>,
    server_tasks: TaskMonitor,
    // engines
    codes_engine: ShareDB,
    masks_engine: ShareDB,
    batch_codes_engine: ShareDB,
    batch_masks_engine: ShareDB,
    phase2: Arc<Mutex<Circuits>>,
    phase2_batch: Arc<Mutex<Circuits>>,
    distance_comparator: Arc<Mutex<DistanceComparator>>,
    // DB slices
    code_db_slices: (
        (Vec<CudaSlice<i8>>, Vec<CudaSlice<i8>>),
        (Vec<CudaSlice<u32>>, Vec<CudaSlice<u32>>),
    ),
    mask_db_slices: (
        (Vec<CudaSlice<i8>>, Vec<CudaSlice<i8>>),
        (Vec<CudaSlice<u32>>, Vec<CudaSlice<u32>>),
    ),
    streams: Vec<Vec<CudaStream>>,
    cublas_handles: Vec<Vec<CudaBlas>>,
    results: Vec<Vec<CudaSlice<u32>>>,
    batch_results: Vec<Vec<CudaSlice<u32>>>,
    final_results: Vec<Vec<CudaSlice<u32>>>,
    current_dot_event: Vec<*mut CUevent_st>,
    next_dot_event: Vec<*mut CUevent_st>,
    current_exchange_event: Vec<*mut CUevent_st>,
    next_exchange_event: Vec<*mut CUevent_st>,
    start_timer: Vec<*mut CUevent_st>,
    end_timer: Vec<*mut CUevent_st>,
    timer_events: Vec<Vec<Vec<*mut CUevent_st>>>,
    previous_stream_event: Vec<*mut CUevent_st>,
    previous_previous_stream_event: Vec<*mut CUevent_st>,
    current_stream_event: Vec<*mut CUevent_st>,
    current_db_size_mutex: Vec<Arc<Mutex<usize>>>,
    query_db_size: Vec<usize>,
    previous_thread_handle: Option<JoinHandle<()>>,
    phase2_chunk_size_max: usize,
    request_counter: usize,
}

const RESULTS_INIT_HOST: [u32; N_QUERIES * ROTATIONS] = [u32::MAX; N_QUERIES * ROTATIONS];
const FINAL_RESULTS_INIT_HOST: [u32; N_QUERIES] = [u32::MAX; N_QUERIES];

impl ServerActor {
    pub fn new(
        config: Config,
        chacha_seeds: ([u32; 8], [u32; 8]),
        codes_db: &[u16],
        masks_db: &[u16],
        job_queue_size: usize,
    ) -> eyre::Result<(Self, ServerActorHandle)> {
        let (tx, rx) = mpsc::channel(job_queue_size);
        let actor = Self::init(config, chacha_seeds, codes_db, masks_db, rx)?;
        Ok((actor, ServerActorHandle { job_queue: tx }))
    }

    fn init(
        config: Config,
        chacha_seeds: ([u32; 8], [u32; 8]),
        codes_db: &[u16],
        masks_db: &[u16],
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

        let Config {
            party_id,
            bootstrap_url,
            servers,
            ..
        } = config;

        println!("Starting engines...");

        let device_manager = Arc::new(DeviceManager::init());
        let mut server_tasks = TaskMonitor::new();
        let ServersConfig {
            codes_engine_port,
            masks_engine_port,
            batch_codes_engine_port,
            batch_masks_engine_port,
            phase_2_port,
            phase_2_batch_port,
        } = servers;

        // Phase 1 Setup
        let codes_engine = ShareDB::init(
            party_id,
            device_manager.clone(),
            DB_SIZE + DB_BUFFER,
            QUERIES,
            next_chacha_seeds(chacha_seeds)?,
            bootstrap_url.clone(),
            Some(true),
            Some(codes_engine_port),
            Some(&mut server_tasks),
        );
        server_tasks.check_tasks();

        let masks_engine = ShareDB::init(
            party_id,
            device_manager.clone(),
            DB_SIZE + DB_BUFFER,
            QUERIES,
            next_chacha_seeds(chacha_seeds)?,
            bootstrap_url.clone(),
            Some(true),
            Some(masks_engine_port),
            Some(&mut server_tasks),
        );
        server_tasks.check_tasks();

        let code_db_slices = codes_engine.load_db(&codes_db, DB_SIZE, DB_SIZE + DB_BUFFER, true);
        let mask_db_slices = masks_engine.load_db(&masks_db, DB_SIZE, DB_SIZE + DB_BUFFER, true);

        // Engines for inflight queries
        let batch_codes_engine = ShareDB::init(
            party_id,
            device_manager.clone(),
            QUERIES * device_manager.device_count(),
            QUERIES,
            next_chacha_seeds(chacha_seeds)?,
            bootstrap_url.clone(),
            Some(true),
            Some(batch_codes_engine_port),
            Some(&mut server_tasks),
        );
        server_tasks.check_tasks();

        let batch_masks_engine = ShareDB::init(
            party_id,
            device_manager.clone(),
            QUERIES * device_manager.device_count(),
            QUERIES,
            next_chacha_seeds(chacha_seeds)?,
            bootstrap_url.clone(),
            Some(true),
            Some(batch_masks_engine_port),
            Some(&mut server_tasks),
        );
        server_tasks.check_tasks();

        // Phase 2 Setup
        let phase2_chunk_size = QUERIES * DB_SIZE / device_manager.device_count();
        let phase2_chunk_size_max = QUERIES * (DB_SIZE + DB_BUFFER) / device_manager.device_count();

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
        assert!(
            phase2_chunk_size_max % 64 == 0,
            "Phase2 MAX chunk size must be a multiple of 64"
        );

        let phase2_batch = Arc::new(Mutex::new(Circuits::new(
            party_id,
            phase2_batch_chunk_size,
            phase2_batch_chunk_size / 64,
            next_chacha_seeds(chacha_seeds)?,
            bootstrap_url.clone(),
            Some(phase_2_batch_port),
            Some(&mut server_tasks),
            device_manager.clone(),
        )));
        server_tasks.check_tasks();

        let phase2 = Arc::new(Mutex::new(Circuits::new(
            party_id,
            phase2_chunk_size,
            phase2_chunk_size_max / 64,
            next_chacha_seeds(chacha_seeds)?,
            bootstrap_url.clone(),
            Some(phase_2_port),
            Some(&mut server_tasks),
            device_manager.clone(),
        )));
        server_tasks.check_tasks();

        let distance_comparator = Arc::new(Mutex::new(DistanceComparator::init(
            QUERIES,
            device_manager.clone(),
        )));
        // Prepare streams etc.
        let mut streams = vec![];
        let mut cublas_handles = vec![];
        let mut results = vec![];
        let mut batch_results = vec![];
        let mut final_results = vec![];
        for _ in 0..MAX_BATCHES_BEFORE_REUSE {
            let tmp_streams = device_manager.fork_streams();
            cublas_handles.push(device_manager.create_cublas(&tmp_streams));
            streams.push(tmp_streams);
            results.push(distance_comparator.lock().unwrap().prepare_results());
            batch_results.push(distance_comparator.lock().unwrap().prepare_results());
            final_results.push(distance_comparator.lock().unwrap().prepare_final_results());
        }

        let previous_previous_stream_event = device_manager.create_events();
        let previous_stream_event = device_manager.create_events();
        let current_stream_event = device_manager.create_events();

        let previous_thread_handle: Option<JoinHandle<()>> = None;
        let current_dot_event = device_manager.create_events();
        let next_dot_event = device_manager.create_events();
        let current_exchange_event = device_manager.create_events();
        let next_exchange_event = device_manager.create_events();
        let timer_events = vec![];
        let start_timer = device_manager.create_events();
        let end_timer = device_manager.create_events();

        let current_db_size: Vec<usize> =
            vec![DB_SIZE / device_manager.device_count(); device_manager.device_count()];
        let query_db_size = vec![QUERIES; device_manager.device_count()];
        let current_db_size_mutex = current_db_size
            .iter()
            .map(|&s| Arc::new(Mutex::new(s)))
            .collect::<Vec<_>>();

        Ok(Self {
            job_queue,
            device_manager,
            server_tasks,
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
            cublas_handles,
            results,
            batch_results,
            final_results,
            current_dot_event,
            next_dot_event,
            current_exchange_event,
            next_exchange_event,
            start_timer,
            end_timer,
            timer_events,
            previous_stream_event,
            previous_previous_stream_event,
            current_stream_event,
            current_db_size_mutex,
            query_db_size,
            previous_thread_handle,
            phase2_chunk_size_max,
            request_counter: 0,
        })
    }

    pub fn run(mut self) {
        while let Some(job) = self.job_queue.blocking_recv() {
            let ServerJob {
                batch,
                return_channel,
            } = job;
            self.process_batch_query(batch, return_channel);
        }

        // await the last thread for phase 2
        if let Some(handle) = self.previous_thread_handle {
            handle.join().unwrap();
        }
        // Clean up server tasks, then wait for them to finish
        self.server_tasks.abort_all();

        std::thread::sleep(Duration::from_secs(5));
        for timers in self.timer_events {
            unsafe {
                self.device_manager.device(0).bind_to_thread().unwrap();
                let dot_time = elapsed(timers[0][0], timers[1][0]).unwrap();
                let exchange_time = elapsed(timers[2][0], timers[3][0]).unwrap();
                println!(
                    "Dot time: {:?}, Exchange time: {:?}",
                    dot_time, exchange_time
                );
            }
        }

        for i in 0..self.device_manager.device_count() {
            unsafe {
                self.device_manager.device(i).bind_to_thread().unwrap();
                let total_time = elapsed(self.start_timer[i], self.end_timer[i]).unwrap();
                println!("Total time: {:?}", total_time);
            }
        }

        self.server_tasks.check_tasks_finished();
    }

    fn process_batch_query(
        &mut self,
        batch: BatchQuery,
        return_channel: oneshot::Sender<ServerJobResult>,
    ) {
        let now = Instant::now();
        // *Query* variant including Lagrange interpolation.
        let code_query = prepare_query_shares(batch.query.code);
        let mask_query = prepare_query_shares(batch.query.mask);
        // *Storage* variant (no interpolation).
        let code_query_insert = prepare_query_shares(batch.db.code);
        let mask_query_insert = prepare_query_shares(batch.db.mask);
        let query_store = batch.store;

        let mut timers = vec![];

        // SAFETY: these streams can only safely be re-used after more than
        // MAX_CONCURRENT_BATCHES.
        let request_streams = &self.streams[self.request_counter % MAX_BATCHES_BEFORE_REUSE];
        let request_cublas_handles =
            &self.cublas_handles[self.request_counter % MAX_BATCHES_BEFORE_REUSE];
        let request_results = &self.results[self.request_counter % MAX_BATCHES_BEFORE_REUSE];
        let request_results_batch =
            &self.batch_results[self.request_counter % MAX_BATCHES_BEFORE_REUSE];
        let request_final_results =
            &self.final_results[self.request_counter % MAX_BATCHES_BEFORE_REUSE];

        // First stream doesn't need to wait on anyone
        if self.request_counter == 0 {
            self.device_manager
                .record_event(request_streams, &self.current_dot_event);
            self.device_manager
                .record_event(request_streams, &self.current_exchange_event);
            self.device_manager
                .record_event(request_streams, &self.start_timer);
        }
        // Transfer queries to device
        // TODO: free all of this!
        let code_query = self
            .device_manager
            .htod_transfer_query(&code_query, request_streams);
        let mask_query = self
            .device_manager
            .htod_transfer_query(&mask_query, request_streams);
        let code_query_insert = self
            .device_manager
            .htod_transfer_query(&code_query_insert, request_streams);
        let mask_query_insert = self
            .device_manager
            .htod_transfer_query(&mask_query_insert, request_streams);
        let code_query_sums =
            self.codes_engine
                .query_sums(&code_query, request_streams, request_cublas_handles);
        let mask_query_sums =
            self.masks_engine
                .query_sums(&mask_query, request_streams, request_cublas_handles);
        let code_query_insert_sums = self.codes_engine.query_sums(
            &code_query_insert,
            request_streams,
            request_cublas_handles,
        );
        let mask_query_insert_sums = self.masks_engine.query_sums(
            &mask_query_insert,
            request_streams,
            request_cublas_handles,
        );

        // update the db size, skip this for the first two
        if self.request_counter > MAX_CONCURRENT_BATCHES {
            // We have two streams working concurrently, we'll await the stream before
            // previous one.
            // SAFETY: these streams can only safely be re-used after more than
            // MAX_CONCURRENT_BATCHES.
            let previous_previous_streams = &self.streams
                [(self.request_counter - MAX_CONCURRENT_BATCHES) % MAX_BATCHES_BEFORE_REUSE];
            self.device_manager.await_event(
                previous_previous_streams,
                &self.previous_previous_stream_event,
            );
            self.device_manager.await_streams(previous_previous_streams);
        }

        let current_db_size_stream = self
            .current_db_size_mutex
            .iter()
            .map(|e| *e.lock().unwrap())
            .collect::<Vec<_>>();

        // BLOCK 1: calculate individual dot products
        self.device_manager
            .await_event(request_streams, &self.current_dot_event);

        // ---- START BATCH DEDUP ----

        self.batch_codes_engine.dot(
            &code_query,
            &code_query_insert,
            &self.query_db_size,
            request_streams,
            request_cublas_handles,
        );

        self.batch_masks_engine.dot(
            &mask_query,
            &mask_query_insert,
            &self.query_db_size,
            request_streams,
            request_cublas_handles,
        );

        self.batch_codes_engine.dot_reduce(
            &code_query_sums,
            &code_query_insert_sums,
            &self.query_db_size,
            request_streams,
        );

        self.batch_masks_engine.dot_reduce(
            &mask_query_sums,
            &mask_query_insert_sums,
            &self.query_db_size,
            request_streams,
        );

        self.batch_codes_engine
            .reshare_results(&self.query_db_size, request_streams);
        self.batch_masks_engine
            .reshare_results(&self.query_db_size, request_streams);

        // ---- END BATCH DEDUP ----
        debug_record_event!(self.device_manager, request_streams, timers);

        self.codes_engine.dot(
            &code_query,
            &(
                helpers::device_ptrs(&self.code_db_slices.0 .0),
                helpers::device_ptrs(&self.code_db_slices.0 .1),
            ),
            &current_db_size_stream,
            request_streams,
            request_cublas_handles,
        );

        self.masks_engine.dot(
            &mask_query,
            &(
                helpers::device_ptrs(&self.mask_db_slices.0 .0),
                helpers::device_ptrs(&self.mask_db_slices.0 .1),
            ),
            &current_db_size_stream,
            request_streams,
            request_cublas_handles,
        );

        debug_record_event!(self.device_manager, request_streams, timers);

        // BLOCK 2: calculate final dot product result, exchange and compare
        self.device_manager
            .await_event(request_streams, &self.current_exchange_event);

        self.codes_engine.dot_reduce(
            &code_query_sums,
            &(
                helpers::device_ptrs(&self.code_db_slices.1 .0),
                helpers::device_ptrs(&self.code_db_slices.1 .1),
            ),
            &current_db_size_stream,
            request_streams,
        );
        self.masks_engine.dot_reduce(
            &mask_query_sums,
            &(
                helpers::device_ptrs(&self.mask_db_slices.1 .0),
                helpers::device_ptrs(&self.mask_db_slices.1 .1),
            ),
            &current_db_size_stream,
            request_streams,
        );

        self.device_manager
            .record_event(request_streams, &self.next_dot_event);

        debug_record_event!(self.device_manager, request_streams, timers);

        self.codes_engine
            .reshare_results(&current_db_size_stream, request_streams);
        self.masks_engine
            .reshare_results(&current_db_size_stream, request_streams);

        debug_record_event!(self.device_manager, request_streams, timers);

        self.device_manager
            .record_event(request_streams, &self.next_exchange_event);

        println!("phase 1 done");

        // SAFETY:
        // - We are sending these streams and events to a single thread (without cloning
        //   them), then dropping our references to them (without destroying them).
        // - These pointers are aligned, dereferencable, and initialized.
        // Unique usage:
        // - Streams are re-used after MAX_BATCHES_BEFORE_REUSE threads, but we only
        //   launch MAX_CONCURRENT_BATCHES threads at a time. So this reference performs
        //   the only accesses to its memory across both C and Rust.
        // - New current stream events are created for each batch. They are only re-used
        //   after MAX_CONCURRENT_BATCHES, but we wait for the previous batch to finish
        //   before running that code.
        // - End events are re-used in each thread, but we only end one thread at a
        //   time.
        const_assert!(MAX_BATCHES_BEFORE_REUSE > MAX_CONCURRENT_BATCHES);

        // into_iter() makes the Rust compiler check that the streams are not re-used.
        let mut thread_streams = request_streams
            .iter()
            .map(|s| unsafe { s.stream.as_mut().unwrap() })
            .collect::<Vec<_>>();
        // The compiler can't tell that we wait for the previous batch before re-using
        // these events.
        let mut thread_current_stream_event = self
            .current_stream_event
            .iter()
            .map(|e| unsafe { e.as_mut().unwrap() })
            .collect::<Vec<_>>();
        let mut thread_end_timer = self
            .end_timer
            .iter()
            .map(|e| unsafe { e.as_mut().unwrap() })
            .collect::<Vec<_>>();

        let thread_device_manager = self.device_manager.clone();
        let thread_current_db_size_mutex = self
            .current_db_size_mutex
            .iter()
            .map(Arc::clone)
            .collect::<Vec<_>>();
        let db_sizes_batch = self.query_db_size.clone();
        let thread_request_results_batch = helpers::device_ptrs(request_results_batch);
        let thread_request_results = helpers::device_ptrs(request_results);
        let thread_request_final_results = helpers::device_ptrs(request_final_results);

        // Batch phase 1 results
        let thread_code_results_batch = helpers::device_ptrs(&self.batch_codes_engine.results);
        let thread_code_results_peer_batch =
            helpers::device_ptrs(&self.batch_codes_engine.results_peer);
        let thread_mask_results_batch = helpers::device_ptrs(&self.batch_masks_engine.results);
        let thread_mask_results_peer_batch =
            helpers::device_ptrs(&self.batch_masks_engine.results_peer);

        // DB phase 1 results
        let thread_code_results = helpers::device_ptrs(&self.codes_engine.results);
        let thread_code_results_peer = helpers::device_ptrs(&self.codes_engine.results_peer);
        let thread_mask_results = helpers::device_ptrs(&self.masks_engine.results);
        let thread_mask_results_peer = helpers::device_ptrs(&self.masks_engine.results_peer);

        // SAFETY: phase2 and phase2_batch are only used in one spawned threat at a
        // time.
        let thread_phase2 = self.phase2.clone();
        let thread_phase2_batch = self.phase2_batch.clone();
        let thread_distance_comparator = self.distance_comparator.clone();
        let thread_code_db_slices = helpers::slice_tuples_to_ptrs(&self.code_db_slices);
        let thread_mask_db_slices = helpers::slice_tuples_to_ptrs(&self.mask_db_slices);
        let thread_request_ids = batch.request_ids.clone();
        let thread_sender = return_channel;
        let thread_prev_handle = self.previous_thread_handle.take();
        let phase2_chunk_size_max = self.phase2_chunk_size_max;

        self.previous_thread_handle = Some(thread::spawn(move || {
            // Wait for Phase 1 to finish
            helpers::await_streams(&mut thread_streams);

            // Iterate over a list of tracing payloads, and create logs with mappings to
            // payloads Log at least a "start" event using a log with trace.id
            // and parent.trace.id
            for tracing_payload in batch.metadata.iter() {
                tracing::info!(
                    node_id = tracing_payload.node_id,
                    dd.trace_id = tracing_payload.trace_id,
                    dd.span_id = tracing_payload.span_id,
                    "Phase 1 finished",
                );
            }
            // Wait for Phase 2 of previous round to finish in order to not have them
            // overlapping. SAFETY: waiting here makes sure we don't access
            // these mutable streams or events concurrently:
            // - CUstream: thread_streams (only re-used after MAX_BATCHES_BEFORE_REUSE
            //   batches),
            // - CUevent: thread_current_stream_event, thread_end_timer,
            // - Comm: phase2, phase2_batch.
            if let Some(phandle) = thread_prev_handle {
                phandle.join().unwrap();
            }

            let thread_devs = thread_device_manager.devices();
            let mut thread_phase2_batch = thread_phase2_batch.lock().unwrap();
            let mut thread_phase2 = thread_phase2.lock().unwrap();
            let tmp_distance_comparator = thread_distance_comparator.lock().unwrap();
            let (result_sizes, db_sizes): (Vec<_>, Vec<_>) = thread_current_db_size_mutex
                .iter()
                .map(|e| {
                    let db_size = *e.lock().unwrap();
                    ((db_size * QUERIES).div_ceil(64) * 64, db_size)
                })
                .unzip();

            // Iterate over a list of tracing payloads, and create logs with mappings to
            // payloads Log at least a "start" event using a log with trace.id
            // and parent.trace.id
            for tracing_payload in batch.metadata.iter() {
                tracing::info!(
                    node_id = tracing_payload.node_id,
                    dd.trace_id = tracing_payload.trace_id,
                    dd.span_id = tracing_payload.span_id,
                    "Phase 2 finished",
                );
            }

            let result_sizes_batch = db_sizes_batch
                .iter()
                .map(|&e| e * QUERIES)
                .collect::<Vec<_>>();

            let mut code_dots_batch: Vec<ChunkShare<u16>> = helpers::device_ptrs_to_shares(
                &thread_code_results_batch,
                &thread_code_results_peer_batch,
                &result_sizes_batch,
                thread_devs,
            );
            let mut mask_dots_batch: Vec<ChunkShare<u16>> = helpers::device_ptrs_to_shares(
                &thread_mask_results_batch,
                &thread_mask_results_peer_batch,
                &result_sizes_batch,
                thread_devs,
            );

            let mut code_dots: Vec<ChunkShare<u16>> = helpers::device_ptrs_to_shares(
                &thread_code_results,
                &thread_code_results_peer,
                &result_sizes,
                thread_devs,
            );
            let mut mask_dots: Vec<ChunkShare<u16>> = helpers::device_ptrs_to_shares(
                &thread_mask_results,
                &thread_mask_results_peer,
                &result_sizes,
                thread_devs,
            );

            // TODO: use phase 1 streams here
            let phase2_streams = thread_phase2
                .get_devices()
                .iter()
                .map(|d| d.fork_default_stream().unwrap())
                .collect::<Vec<_>>();

            // Phase 2 [Batch]: compare each result against threshold
            thread_phase2_batch.compare_threshold_masked_many(
                &code_dots_batch,
                &mask_dots_batch,
                &phase2_streams,
            );

            // Phase 2 [Batch]: Reveal the binary results
            let res = thread_phase2_batch.take_result_buffer();
            let mut thread_request_results_slice_batch: Vec<CudaSlice<u32>> =
                helpers::device_ptrs_to_slices(
                    &thread_request_results_batch,
                    &vec![QUERIES; thread_devs.len()],
                    thread_devs,
                );

            // Iterate over a list of tracing payloads, and create logs with mappings to
            // payloads Log at least a "start" event using a log with trace.id
            // and parent.trace.id
            for tracing_payload in batch.metadata.iter() {
                tracing::info!(
                    node_id = tracing_payload.node_id,
                    dd.trace_id = tracing_payload.trace_id,
                    dd.span_id = tracing_payload.span_id,
                    "Phase 2 finished",
                );
            }

            let chunk_size_batch = thread_phase2_batch.chunk_size();
            open(
                &mut thread_phase2_batch,
                &res,
                &tmp_distance_comparator,
                &thread_request_results_slice_batch,
                chunk_size_batch,
                &db_sizes_batch,
                &phase2_streams,
            );
            thread_phase2_batch.return_result_buffer(res);

            // Phase 2 [DB]: compare each result against threshold
            thread_phase2.compare_threshold_masked_many(&code_dots, &mask_dots, &phase2_streams);

            // Phase 2 [DB]: Reveal the binary results
            let res = thread_phase2.take_result_buffer();
            let mut thread_request_results_slice: Vec<CudaSlice<u32>> =
                helpers::device_ptrs_to_slices(
                    &thread_request_results,
                    &vec![QUERIES; thread_devs.len()],
                    thread_devs,
                );

            let chunk_size = thread_phase2.chunk_size();
            open(
                &mut thread_phase2,
                &res,
                &tmp_distance_comparator,
                &thread_request_results_slice,
                chunk_size,
                &db_sizes,
                &phase2_streams,
            );
            thread_phase2.return_result_buffer(res);

            // Merge results and fetch matching indices
            // Format: host_results[device_index][query_index]
            tmp_distance_comparator.merge_results(
                &thread_request_results_batch,
                &thread_request_results,
                &thread_request_final_results,
                &phase2_streams,
            );

            thread_device_manager.await_streams(&phase2_streams);

            let host_results =
                tmp_distance_comparator.fetch_final_results(&thread_request_final_results);

            // Evaluate the results across devices
            // Format: merged_results[query_index]
            let mut merged_results = get_merged_results(&host_results, thread_devs.len());

            // List the indices of the queries that did not match.
            let insertion_list = merged_results
                .iter()
                .enumerate()
                .filter(|&(_idx, &num)| num == u32::MAX)
                .map(|(idx, _num)| idx)
                .collect::<Vec<_>>();

            // Spread the insertions across devices.
            let insertion_list = distribute_insertions(&insertion_list, &db_sizes);

            // Calculate the new indices for the inserted queries
            let matches =
                calculate_insertion_indices(&mut merged_results, &insertion_list, &db_sizes);

            for i in 0..thread_devs.len() {
                thread_devs[i].bind_to_thread().unwrap();
                let mut old_db_size = *thread_current_db_size_mutex[i].lock().unwrap();
                for insertion_idx in insertion_list[i].clone() {
                    // Append to codes and masks db
                    for (db, query, sums) in [
                        (
                            &thread_code_db_slices,
                            &code_query_insert,
                            &code_query_insert_sums,
                        ),
                        (
                            &thread_mask_db_slices,
                            &mask_query_insert,
                            &mask_query_insert_sums,
                        ),
                    ] {
                        helpers::dtod_at_offset(
                            db.0 .0[i],
                            old_db_size * IRIS_CODE_LENGTH,
                            query.0[i],
                            IRIS_CODE_LENGTH * 15 + insertion_idx * IRIS_CODE_LENGTH * ROTATIONS,
                            IRIS_CODE_LENGTH,
                            phase2_streams[i].stream,
                        );

                        helpers::dtod_at_offset(
                            db.0 .1[i],
                            old_db_size * IRIS_CODE_LENGTH,
                            query.1[i],
                            IRIS_CODE_LENGTH * 15 + insertion_idx * IRIS_CODE_LENGTH * ROTATIONS,
                            IRIS_CODE_LENGTH,
                            phase2_streams[i].stream,
                        );

                        helpers::dtod_at_offset(
                            db.1 .0[i],
                            old_db_size * mem::size_of::<u32>(),
                            sums.0[i],
                            mem::size_of::<u32>() * 15
                                + insertion_idx * mem::size_of::<u32>() * ROTATIONS,
                            mem::size_of::<u32>(),
                            phase2_streams[i].stream,
                        );

                        helpers::dtod_at_offset(
                            db.1 .1[i],
                            old_db_size * mem::size_of::<u32>(),
                            sums.1[i],
                            mem::size_of::<u32>() * 15
                                + insertion_idx * mem::size_of::<u32>() * ROTATIONS,
                            mem::size_of::<u32>(),
                            phase2_streams[i].stream,
                        );
                    }
                    old_db_size += 1;
                }

                // Write new db sizes to device
                *thread_current_db_size_mutex[i].lock().unwrap() +=
                    insertion_list[i].len() as usize;

                // DEBUG
                println!(
                    "Updating DB size on device {}: {:?}",
                    i,
                    *thread_current_db_size_mutex[i].lock().unwrap()
                );

                // Update Phase 2 chunk size to max all db sizes on all devices
                let max_db_size = thread_current_db_size_mutex
                    .iter()
                    .map(|e| *e.lock().unwrap())
                    .max()
                    .unwrap();
                let new_chunk_size = (QUERIES * max_db_size).div_ceil(64) * 64;
                assert!(new_chunk_size <= phase2_chunk_size_max);
                thread_phase2.set_chunk_size(new_chunk_size / 64);

                // Emit stream finished event to unblock the stream after the following stream.
                // Since previous timers are overwritten, only the final end timers are used to
                // calculate the total time.
                //
                // SAFETY:
                // - the events are created before launching the thread, so they are never null.
                // - the streams have already been created, and we hold a reference to their
                //   CudaDevice, which makes sure they aren't dropped.
                unsafe {
                    event::record(
                        *thread_current_stream_event.index_mut(i),
                        *thread_streams.index_mut(i),
                    )
                    .unwrap();

                    // DEBUG: emit event to measure time for e2e process
                    event::record(*thread_end_timer.index_mut(i), *thread_streams.index_mut(i))
                        .unwrap();
                }
            }
            // Pass to internal sender thread
            // we have allocated enough space such that this should not block
            thread_sender
                .send(ServerJobResult {
                    merged_results,
                    thread_request_ids,
                    matches,
                    store: query_store,
                })
                .unwrap();

            // Reset the results buffers for reuse
            reset_results(
                thread_device_manager.devices(),
                &thread_request_results,
                &RESULTS_INIT_HOST,
                &phase2_streams,
            );
            reset_results(
                thread_device_manager.devices(),
                &thread_request_results_batch,
                &RESULTS_INIT_HOST,
                &phase2_streams,
            );
            reset_results(
                thread_device_manager.devices(),
                &thread_request_final_results,
                &FINAL_RESULTS_INIT_HOST,
                &phase2_streams,
            );

            // Make sure to not call `Drop` on those
            forget_vec!(code_dots);
            forget_vec!(mask_dots);
            forget_vec!(code_dots_batch);
            forget_vec!(mask_dots_batch);
            forget_vec!(thread_request_results_slice);
            forget_vec!(thread_request_results_slice_batch);
        }));
        // Prepare for next batch
        self.server_tasks.check_tasks();

        self.timer_events.push(timers);

        self.previous_previous_stream_event = self.previous_stream_event.clone();
        self.previous_stream_event = self.current_stream_event.clone();
        self.current_stream_event = self.device_manager.create_events();
        self.current_dot_event = self.next_dot_event.clone();
        self.current_exchange_event = self.next_exchange_event.clone();
        self.next_dot_event = self.device_manager.create_events();
        self.next_exchange_event = self.device_manager.create_events();

        println!("CPU time of one iteration {:?}", now.elapsed());
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

fn prepare_query_shares(shares: Vec<GaloisRingIrisCodeShare>) -> Vec<Vec<u8>> {
    preprocess_query(&shares.into_iter().flat_map(|e| e.coefs).collect::<Vec<_>>())
}

fn open(
    party: &mut Circuits,
    x: &[ChunkShare<u64>],
    distance_comparator: &DistanceComparator,
    results_ptrs: &[CudaSlice<u32>],
    chunk_size: usize,
    db_sizes: &[usize],
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

    distance_comparator.open_results(&a, &b, &c, results_ptrs, db_sizes, streams);
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
        println!(
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
    dst: &[CUdeviceptr],
    src: &[u32],
    streams: &[CudaStream],
) {
    for i in 0..devs.len() {
        devs[i].bind_to_thread().unwrap();
        unsafe { result::memcpy_htod_async(dst[i], src, streams[i].stream) }.unwrap();
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
