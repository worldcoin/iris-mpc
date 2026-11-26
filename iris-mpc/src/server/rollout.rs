use super::drain::{self, Drain};
use ampc_server_utils::shutdown_handler::ShutdownHandler;
use iris_mpc_common::config::{Config, CpuCellSlot, StartupMode};
use metrics::Gauge;
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const REFRESH: Duration = Duration::from_secs(10);

// These values are part of the dashboard's query contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Phase {
    Starting = 1,
    Preloading = 2,
    Loaded = 3,
    Draining = 4,
    Paused = 5,
    Reconciling = 6,
    WaitingForAgreement = 7,
    Serving = 8,
    BlockedServing = 9,
    Failed = 10,
    Stopping = 11,
    Reconciled = 12,
}

#[derive(Clone, Default)]
pub(super) struct Rollout(Option<Arc<Mutex<Observation>>>);

struct Observation {
    phase: Phase,
    last_phase: Phase,
    changed_at: Instant,
    drain: Option<Arc<Drain>>,
    blocked: bool,
    phase_gauge: Gauge,
    admitting: Gauge,
    phase_age: Gauge,
    observed_at: Gauge,
    shutdown: Arc<ShutdownHandler>,
}

impl Rollout {
    pub(super) fn new(config: &Config, shutdown: Arc<ShutdownHandler>) -> Self {
        let Some(peers) = config
            .cpu_cell
            .as_ref()
            .and_then(|cell| cell.peers.as_ref())
        else {
            return Self::default();
        };
        let party = match config.party_id {
            0 => "0",
            1 => "1",
            2 => "2",
            _ => return Self::default(),
        };
        let slot = match peers.slot {
            CpuCellSlot::A => "a",
            CpuCellSlot::B => "b",
        };
        let phase = match config.startup_mode {
            StartupMode::Active => Phase::Starting,
            StartupMode::Candidate => Phase::Preloading,
        };
        tracing::info!(
            party_id = config.party_id,
            logical_slot = slot,
            "CPU cell rollout identity"
        );
        Self(Some(Arc::new(Mutex::new(Observation {
            phase,
            last_phase: phase,
            changed_at: Instant::now(),
            drain: None,
            blocked: false,
            phase_gauge: metrics::gauge!("cpu_cell_rollout_phase", "party" => party, "slot" => slot),
            admitting: metrics::gauge!("cpu_cell_rollout_admitting", "party" => party, "slot" => slot),
            phase_age: metrics::gauge!("cpu_cell_rollout_phase_age_seconds", "party" => party, "slot" => slot),
            observed_at: metrics::gauge!("cpu_cell_rollout_observed_at", "party" => party, "slot" => slot),
            shutdown,
        }))))
    }

    pub(super) async fn refresh(&self) -> std::convert::Infallible {
        if self.0.is_none() {
            return std::future::pending().await;
        }
        let mut interval = tokio::time::interval(REFRESH);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            self.update(|_| {});
        }
    }

    pub(super) fn set_phase(&self, phase: Phase) {
        self.update(|state| {
            if state.phase != Phase::Failed {
                state.phase = phase;
            }
        });
    }

    pub(super) fn start_serving(&self, drain: Option<Arc<Drain>>) {
        self.update(|state| {
            state.drain = drain;
            state.phase = Phase::Serving;
        });
    }

    pub(super) fn set_blocked(&self, blocked: bool) {
        self.update(|state| state.blocked = blocked);
    }

    fn update(&self, change: impl FnOnce(&mut Observation)) {
        let Some(state) = &self.0 else { return };
        let mut state = state.lock().unwrap();
        change(&mut state);
        let phase = if state.phase == Phase::Serving {
            let drain = state.drain.as_ref().map(|drain| drain.status());
            let blocked =
                state.blocked || drain.as_ref().is_some_and(|s| s.blocked_reason.is_some());
            let phase = serving_phase(drain.map(|s| s.phase), blocked);
            if matches!(phase, Phase::Serving | Phase::BlockedServing)
                && state.shutdown.is_shutting_down()
            {
                Phase::Stopping
            } else {
                phase
            }
        } else {
            state.phase
        };
        let now = Instant::now();
        if phase != state.last_phase {
            state.last_phase = phase;
            state.changed_at = now;
        }
        state.phase_gauge.set(phase as u8 as f64);
        state
            .admitting
            .set(if matches!(phase, Phase::Serving | Phase::BlockedServing) {
                1.0
            } else {
                0.0
            });
        state.phase_age.set((now - state.changed_at).as_secs_f64());
        state.observed_at.set(chrono::Utc::now().timestamp() as f64);
    }
}

fn serving_phase(drain: Option<drain::Phase>, blocked: bool) -> Phase {
    match drain {
        Some(drain::Phase::Freezing | drain::Phase::Draining | drain::Phase::Quiescent) => {
            Phase::Draining
        }
        Some(drain::Phase::Paused) => Phase::Paused,
        Some(drain::Phase::Failed) => Phase::Failed,
        Some(drain::Phase::Active) | None if blocked => Phase::BlockedServing,
        Some(drain::Phase::Active) | None => Phase::Serving,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heartbeat_reflects_local_blockers_and_preserves_processing_failure_on_exit() {
        let urls = r#"["http://one","http://two","http://three"]"#;
        let config = serde_json::from_value(serde_json::json!({
            "cpu_cell": {"deployment_id": "test", "peers": {"slot": "a", "a": urls, "b": urls}}
        }))
        .unwrap();
        let rollout = Rollout::new(&config, Arc::new(ShutdownHandler::new(0)));
        let phase = || rollout.0.as_ref().unwrap().lock().unwrap().last_phase;
        let drain = Drain::new(0, Duration::from_secs(5), true);
        rollout.start_serving(Some(drain.clone()));
        assert_eq!(phase(), Phase::Serving);
        drain.block("SNS replay failed");
        rollout.update(|_| {});
        assert_eq!(phase(), Phase::BlockedServing);
        rollout.set_phase(Phase::Failed);
        rollout.set_phase(Phase::Stopping);
        assert_eq!(phase(), Phase::Failed);
    }

    #[test]
    fn a_blocked_preflight_can_still_admit_but_a_closed_drain_takes_precedence() {
        assert_eq!(
            serving_phase(Some(drain::Phase::Active), false),
            Phase::Serving
        );
        assert_eq!(
            serving_phase(Some(drain::Phase::Active), true),
            Phase::BlockedServing
        );
        for phase in [
            drain::Phase::Freezing,
            drain::Phase::Draining,
            drain::Phase::Quiescent,
        ] {
            assert_eq!(serving_phase(Some(phase), true), Phase::Draining);
        }
        assert_eq!(
            serving_phase(Some(drain::Phase::Paused), true),
            Phase::Paused
        );
        assert_eq!(
            serving_phase(Some(drain::Phase::Failed), false),
            Phase::Failed
        );
    }
}
