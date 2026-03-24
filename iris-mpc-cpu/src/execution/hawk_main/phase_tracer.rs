//! Phase timeline tracer for visualizing search pipeline concurrency.
//!
//! Records enter/exit timestamps for each phase per session, outputting
//! Chrome Trace Event Format JSON that can be loaded into Perfetto UI
//! (https://ui.perfetto.dev/) or `chrome://tracing`.
//!
//! Enabled at compile time via the `phase_trace` Cargo feature.
//! Traces are flushed to `/tmp/hawk_phase_trace_party{N}_batch{B}.json`
//! after each batch.

use crossbeam::channel::{Receiver, Sender};
use serde::Serialize;
use std::sync::OnceLock;
use std::time::Instant;

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
        format!(
            "{}_eye{}_sess{}",
            self.orient, self.i_eye, self.i_session
        )
    }
}

// ── Chrome Trace Event ──────────────────────────────────────────────────────

#[derive(Serialize)]
struct TraceEvent {
    name: &'static str,
    ph: char,
    ts: f64,
    pid: u32,
    tid: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cname: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    args: Option<serde_json::Value>,
}

/// Map phase names to Perfetto/chrome:tracing predefined colors.
fn phase_color(name: &str) -> Option<&'static str> {
    match name {
        "dot_product" => Some("good"),             // green
        "mpc_lift" => Some("rail_animation"),      // orange
        "oblivious_min" => Some("bad"),            // blue
        "open_nodes" => Some("olive"),             // olive
        "prune_candidates" => Some("rail_load"),   // indigo
        "insert_and_trim" => Some("generic_work"), // purple
        _ => None,
    }
}

// ── Global Tracer ───────────────────────────────────────────────────────────

static TRACER: OnceLock<PhaseTracer> = OnceLock::new();

struct PhaseTracer {
    tx: Sender<TraceEvent>,
    rx: Receiver<TraceEvent>,
    start_time: Instant,
    party_id: u32,
}

/// Initialize the global phase tracer. Call once at startup.
pub fn init(party_id: u32) {
    let (tx, rx) = crossbeam::channel::unbounded();
    let tracer = PhaseTracer {
        tx,
        rx,
        start_time: Instant::now(),
        party_id,
    };

    let _ = TRACER.set(tracer);
    tracing::info!("Phase tracer enabled");
}

fn tracer() -> &'static PhaseTracer {
    TRACER.get().expect("phase tracer not initialized")
}

/// Flush all collected events to a JSON file for the given batch.
pub fn flush(batch: u32) {
    let tracer = tracer();
    let mut events = Vec::new();
    while let Ok(ev) = tracer.rx.try_recv() {
        events.push(ev);
    }
    if events.is_empty() {
        return;
    }

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
            ph: 'E',
            ts,
            pid: self.pid,
            tid: self.tid.clone(),
            cname: phase_color(self.name),
            args: None,
        });
    }
}

/// Begin a traced phase. Returns a guard that emits the end event on drop.
pub fn phase_begin(
    name: &'static str,
    args: Option<serde_json::Value>,
) -> PhaseGuard {
    let tracer = tracer();
    let tid = SESSION_CTX
        .try_with(|ctx| ctx.tid())
        .unwrap_or_else(|_| "no_session".to_string());

    let ts = tracer.start_time.elapsed().as_secs_f64() * 1e6;
    let _ = tracer.tx.send(TraceEvent {
        name,
        ph: 'B',
        ts,
        pid: tracer.party_id,
        tid: tid.clone(),
        cname: phase_color(name),
        args,
    });

    PhaseGuard {
        name,
        tid,
        tx: &tracer.tx,
        pid: tracer.party_id,
        start_time: tracer.start_time,
    }
}

