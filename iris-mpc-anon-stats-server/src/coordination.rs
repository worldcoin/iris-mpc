use crate::{
    config::AnonStatsServerConfig,
    health::{spawn_healthcheck_server_with_state, HealthServerState},
};
use eyre::Result;
use iris_mpc_common::{
    helpers::task_monitor::TaskMonitor,
    server_coordination::{self, CoordinationSettings},
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tracing::warn;
use uuid::Uuid;

#[derive(Clone)]
pub struct CoordinationHandles {
    ready_flag: Arc<AtomicBool>,
    shutting_down_flag: Arc<AtomicBool>,
}

impl CoordinationHandles {
    pub fn ready_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.ready_flag)
    }

    pub fn set_ready(&self) {
        self.ready_flag.store(true, Ordering::SeqCst);
    }

    pub fn mark_shutting_down(&self) {
        self.shutting_down_flag.store(true, Ordering::SeqCst);
    }

    pub fn shutting_down_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.shutting_down_flag)
    }
}

pub fn start_coordination_server(
    config: &AnonStatsServerConfig,
    task_monitor: &mut TaskMonitor,
) -> CoordinationHandles {
    let ready_flag = Arc::new(AtomicBool::new(false));
    let shutting_down_flag = Arc::new(AtomicBool::new(false));
    let state = HealthServerState::new(
        Arc::clone(&ready_flag),
        Arc::clone(&shutting_down_flag),
        Arc::new(config.image_name.clone()),
        Arc::new(Uuid::new_v4().to_string()),
    );

    let port = config.healthcheck_ports[config.party_id].clone();
    task_monitor.spawn(async move {
        if let Err(err) = spawn_healthcheck_server_with_state(port, state).await {
            warn!(error = ?err, "Healthcheck server terminated with error");
        }
        Ok::<(), eyre::Report>(())
    });

    CoordinationHandles {
        ready_flag,
        shutting_down_flag,
    }
}

pub async fn wait_for_others_unready(config: &AnonStatsServerConfig) -> Result<()> {
    let settings = build_coordination_settings(config);
    server_coordination::wait_for_others_unready_with_settings(&settings).await
}

pub async fn wait_for_others_ready(config: &AnonStatsServerConfig) -> Result<()> {
    let settings = build_coordination_settings(config);
    server_coordination::wait_for_others_ready_with_settings(&settings).await
}

fn build_coordination_settings(config: &AnonStatsServerConfig) -> CoordinationSettings<'_> {
    CoordinationSettings {
        party_id: config.party_id,
        node_hostnames: &config.node_hostnames,
        healthcheck_ports: &config.healthcheck_ports,
        image_name: &config.image_name,
        http_query_retry_delay_ms: config.http_query_retry_delay_ms,
        startup_sync_timeout_secs: config.startup_sync_timeout_secs,
    }
}
