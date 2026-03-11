//! Phase timeline tracer for visualizing search pipeline concurrency.
//!
//! Records enter/exit timestamps for each phase per session, outputting
//! Chrome Trace Event Format JSON that can be loaded into Perfetto UI
//! (https://ui.perfetto.dev/) or `chrome://tracing`.
//!
//! # Usage
//!
//! Enable at runtime via environment variable:
//! - `HAWK_PHASE_TRACE=1` — record all phases
//! - `HAWK_PHASE_TRACE=dot_product,mpc_lift,oblivious_min` — record only listed phases
//!
//! Traces are flushed to `/tmp/hawk_phase_trace_{party}_{batch}.json` after
//! each batch, or uploaded to S3 if configured.

use serde::Serialize;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use crossbeam::channel::{Receiver, Sender};

// ── Session Context (task-local) ────────────────────────────────────────────

tokio::task_local! {
    pub static SESSION_CTX: SessionContext;
}

#[derive(Clone, Debug)]
pub struct SessionContext {
    pub i_eye: usize,
    pub i_session: usize,
    pub orient: char,
}

impl SessionContext {
    pub fn tid(&self) -> String {
        let batch = current_batch();
        format!(
            "b{batch}_{}_eye{}_sess{}",
            self.orient, self.i_eye, self.i_session
        )
    }
}

// ── Chrome Trace Event ──────────────────────────────────────────────────────

#[derive(Serialize)]
struct TraceEvent {
    name: &'static str,
    cat: &'static str,
    ph: char,
    ts: f64,
    pid: u32,
    tid: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    args: Option<serde_json::Value>,
}

// ── Global Tracer ───────────────────────────────────────────────────────────

static TRACER: OnceLock<PhaseTracer> = OnceLock::new();
static ENABLED: AtomicBool = AtomicBool::new(false);

struct PhaseTracer {
    tx: Sender<TraceEvent>,
    rx: Receiver<TraceEvent>,
    start_time: Instant,
    party_id: u32,
    /// None = all phases enabled, Some(set) = only these phase names
    filter: Option<Vec<&'static str>>,
}

/// Initialize the global phase tracer. Call once at startup.
/// Reads `HAWK_PHASE_TRACE` env var to determine which phases to trace.
pub fn init(party_id: u32) {
    let env_val = std::env::var("HAWK_PHASE_TRACE").unwrap_or_default();
    if env_val.is_empty() || env_val == "0" {
        return;
    }

    let filter = if env_val == "1" {
        None // all phases
    } else {
        // Parse comma-separated phase names. We leak the string to get
        // 'static lifetimes — this is a one-time init cost.
        let leaked: &'static str = Box::leak(env_val.into_boxed_str());
        Some(leaked.split(',').map(|s| s.trim()).collect::<Vec<_>>())
    };

    let (tx, rx) = crossbeam::channel::unbounded();
    let tracer = PhaseTracer {
        tx,
        rx,
        start_time: Instant::now(),
        party_id,
        filter,
    };

    let _ = TRACER.set(tracer);
    ENABLED.store(true, Ordering::Release);

    let filter_desc = TRACER
        .get()
        .unwrap()
        .filter
        .as_ref()
        .map(|f| f.join(","))
        .unwrap_or_else(|| "all".to_string());
    tracing::info!("Phase tracer enabled (filter: {filter_desc})");
}

#[inline(always)]
pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

fn tracer() -> &'static PhaseTracer {
    TRACER.get().expect("phase tracer not initialized")
}

fn is_phase_enabled(name: &str) -> bool {
    match &tracer().filter {
        None => true,
        Some(filter) => filter.contains(&name),
    }
}

static CURRENT_BATCH: AtomicU32 = AtomicU32::new(0);

/// Advance to the next batch. Call before each batch's search phase so
/// that tids include the batch number and don't collide across batches.
pub fn advance_batch() {
    CURRENT_BATCH.fetch_add(1, Ordering::Release);
}

fn current_batch() -> u32 {
    CURRENT_BATCH.load(Ordering::Relaxed)
}

/// Flush all collected events to a JSON file. Each server batch gets
/// its own file. Tids include the batch number so loading multiple
/// files into Perfetto works correctly.
pub fn flush() {
    if !is_enabled() {
        return;
    }
    let tracer = tracer();
    let mut events = Vec::new();
    while let Ok(ev) = tracer.rx.try_recv() {
        events.push(ev);
    }
    if events.is_empty() {
        return;
    }

    let batch = current_batch();
    let path = format!(
        "/tmp/hawk_phase_trace_party{}_batch{}.json",
        tracer.party_id, batch
    );

    match serde_json::to_string(&events) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                tracing::warn!("Failed to write phase trace to {path}: {e}");
            } else {
                tracing::info!("Phase trace written to {path} ({} events)", events.len());
            }
        }
        Err(e) => {
            tracing::warn!("Failed to serialize phase trace: {e}");
        }
    }
}

// ── Phase Guard (RAII) ──────────────────────────────────────────────────────

/// RAII guard that emits a 'B' (begin) event on creation and 'E' (end) on drop.
pub struct PhaseGuard {
    name: &'static str,
    cat: &'static str,
    tid: String,
    tx: &'static Sender<TraceEvent>,
    pid: u32,
    start_time: Instant,
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        let ts = self.start_time.elapsed().as_secs_f64() * 1e6;
        let _ = self.tx.send(TraceEvent {
            name: self.name,
            cat: self.cat,
            ph: 'E',
            ts,
            pid: self.pid,
            tid: self.tid.clone(),
            args: None,
        });
    }
}

/// Begin a traced phase. Returns a guard that emits the end event on drop.
/// Returns `None` if tracing is disabled or the phase is filtered out.
pub fn phase_begin(
    name: &'static str,
    cat: &'static str,
    args: Option<serde_json::Value>,
) -> Option<PhaseGuard> {
    if !is_enabled() || !is_phase_enabled(name) {
        return None;
    }

    let tracer = tracer();
    let tid = SESSION_CTX
        .try_with(|ctx| ctx.tid())
        .unwrap_or_else(|_| "no_session".to_string());

    let ts = tracer.start_time.elapsed().as_secs_f64() * 1e6;
    let _ = tracer.tx.send(TraceEvent {
        name,
        cat,
        ph: 'B',
        ts,
        pid: tracer.party_id,
        tid: tid.clone(),
        args,
    });

    Some(PhaseGuard {
        name,
        cat,
        tid,
        tx: &tracer.tx,
        pid: tracer.party_id,
        start_time: tracer.start_time,
    })
}

