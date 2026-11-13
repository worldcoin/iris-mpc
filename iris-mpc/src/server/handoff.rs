//! Planned A/B handoff over party-local records and the existing peer status network.
use super::{
    drain,
    reconciliation::{CandidateReconciliation, ReconciliationContext, ReconciliationReport},
};
use ampc_server_utils::shutdown_handler::ShutdownHandler;
use axum::{routing::get, Json, Router};
use eyre::{ensure, eyre, Result, WrapErr};
use iris_mpc_common::{
    config::{CommonConfig, Config, StartupMode},
    helpers::sha256::sha256_bytes,
};
use iris_mpc_store::{
    cpu_cell::{CpuCellHandoffs, HandoffPhase},
    Store,
};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};
use tokio::{
    sync::RwLock,
    time::{sleep, timeout_at, Instant},
};
use uuid::Uuid;

// This version requires claimed-sequence batch hashes and durable-C drain reports.
const PROTOCOL: u32 = 1;
const POLL_INTERVAL: Duration = Duration::from_secs(1);
const PREFLIGHT_TIMEOUT: Duration = Duration::from_secs(60 * 60);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Intent {
    id: Uuid,
    source: Uuid,
    destination: Uuid,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Completion {
    source_process: Uuid,
    candidate_processes: [Uuid; 3],
    report: drain::Report,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Status {
    protocol: u32,
    party_id: usize,
    deployment: Uuid,
    process: Uuid,
    mode: StartupMode,
    common_config: [u8; 32],
    ready: bool,
    intent: Option<Intent>,
    source_processes: Option<[Uuid; 3]>,
    candidate_processes: Option<[Uuid; 3]>,
    drain: Option<drain::Status>,
    completion: Option<Completion>,
    reconciliation: Option<ReconciliationReport>,
    error: Option<String>,
}

#[derive(Clone)]
pub(super) struct Coordinator {
    status: Arc<RwLock<Status>>,
    source_urls: [reqwest::Url; 3],
    candidate_urls: [reqwest::Url; 3],
    client: reqwest::Client,
    drain: Option<Arc<drain::Drain>>,
}

fn urls(values: &[String; 3]) -> Result<[reqwest::Url; 3]> {
    let parsed = values
        .iter()
        .map(|value| {
            let url = reqwest::Url::parse(value)?;
            ensure!(
                matches!(url.scheme(), "http" | "https")
                    && url.host_str().is_some()
                    && url.username().is_empty()
                    && url.password().is_none()
                    && url.query().is_none()
                    && url.fragment().is_none(),
                "invalid CPU handoff peer URL"
            );
            Ok(url)
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        parsed[0] != parsed[1] && parsed[0] != parsed[2] && parsed[1] != parsed[2],
        "handoff requires three distinct peer addresses"
    );
    parsed
        .try_into()
        .map_err(|_| eyre!("handoff requires three peer addresses"))
}

impl Coordinator {
    pub(super) fn new(
        config: &Config,
        process: Uuid,
        drain: Option<Arc<drain::Drain>>,
    ) -> Result<Option<Self>> {
        let Some(cell) = &config.cpu_cell else {
            return Ok(None);
        };
        let Some(candidate_urls) = &cell.candidate_status_urls else {
            ensure!(
                cell.handoff.is_none(),
                "automatic handoff requires candidate_status_urls"
            );
            return Ok(None);
        };
        ensure!(config.party_id < 3, "invalid handoff party");
        ensure!(
            config.db_backed_ingest && config.cpu_drain_timeout_secs > 0,
            "runtime handoff requires DB-backed intake and a positive drain budget"
        );
        let deployment = Uuid::parse_str(&cell.deployment_id)?;
        ensure!(!deployment.is_nil(), "CPU deployment must not be nil");
        let (source_urls, intent) = if config.startup_mode == StartupMode::Candidate {
            let requested = cell
                .handoff
                .as_ref()
                .ok_or_else(|| eyre!("candidate coordination requires handoff intent"))?;
            let intent = Intent {
                id: Uuid::parse_str(&requested.id)?,
                source: Uuid::parse_str(&requested.source_deployment_id)?,
                destination: deployment,
            };
            ensure!(
                !intent.id.is_nil() && !intent.source.is_nil() && intent.source != deployment,
                "invalid handoff intent"
            );
            (urls(&requested.source_status_urls.0)?, Some(intent))
        } else {
            ensure!(
                cell.handoff.is_none(),
                "handoff intent belongs to the destination candidate"
            );
            let peers = config
                .server_coordination
                .as_ref()
                .ok_or_else(|| eyre!("source handoff requires server coordination"))?;
            ensure!(
                peers.node_hostnames.len() == 3 && peers.healthcheck_ports.len() == 3,
                "handoff requires three source addresses"
            );
            let addresses = std::array::from_fn(|i| {
                format!(
                    "http://{}:{}/handoff-status",
                    peers.node_hostnames[i], peers.healthcheck_ports[i]
                )
            });
            (urls(&addresses)?, None)
        };
        Ok(Some(Self {
            status: Arc::new(RwLock::new(Status {
                protocol: PROTOCOL,
                party_id: config.party_id,
                deployment,
                process,
                mode: config.startup_mode,
                common_config: sha256_bytes(serde_json::to_vec(&CommonConfig::from(
                    config.clone(),
                ))?),
                ready: false,
                intent,
                source_processes: None,
                candidate_processes: None,
                drain: None,
                completion: None,
                reconciliation: None,
                error: None,
            })),
            source_urls,
            candidate_urls: urls(&candidate_urls.0)?,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(3))
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            drain,
        }))
    }

    pub(super) fn routes(&self) -> Router {
        let coordinator = self.clone();
        Router::new().route(
            "/handoff-status",
            get(move || {
                let coordinator = coordinator.clone();
                async move {
                    let mut status = coordinator.status.read().await.clone();
                    status.drain = coordinator.drain.as_ref().map(|drain| drain.status());
                    Json(status)
                }
            }),
        )
    }

    pub(super) async fn mark_ready(&self, process: Uuid) {
        let mut status = self.status.write().await;
        status.process = process;
        status.ready = true;
    }

    async fn peers(&self, addresses: &[reqwest::Url; 3], mode: StartupMode) -> Result<[Status; 3]> {
        let mut reports = Vec::with_capacity(3);
        for (party, url) in addresses.iter().enumerate() {
            let mut response = self
                .client
                .get(url.clone())
                .send()
                .await?
                .error_for_status()?;
            let mut body = Vec::new();
            while let Some(chunk) = response.chunk().await? {
                ensure!(
                    body.len() + chunk.len() <= 64 * 1024,
                    "handoff status exceeds 64 KiB"
                );
                body.extend_from_slice(&chunk);
            }
            let status: Status = serde_json::from_slice(&body)?;
            ensure!(
                status.protocol == PROTOCOL
                    && status.party_id == party
                    && status.mode == mode
                    && !status.process.is_nil(),
                "incompatible handoff participant at party {party}"
            );
            reports.push(status);
        }
        let reports: [Status; 3] = reports
            .try_into()
            .map_err(|_| eyre!("missing handoff party"))?;
        ensure!(
            reports
                .iter()
                .all(|p| p.common_config == reports[0].common_config
                    && p.deployment == reports[0].deployment),
            "handoff cell configuration or deployment differs across parties"
        );
        ensure!(
            reports[0].process != reports[1].process
                && reports[0].process != reports[2].process
                && reports[1].process != reports[2].process,
            "handoff process identities are not distinct"
        );
        Ok(reports)
    }

    async fn wait_peers(
        &self,
        addresses: &[reqwest::Url; 3],
        mode: StartupMode,
        deadline: Instant,
    ) -> Result<[Status; 3]> {
        loop {
            match timeout_at(deadline, self.peers(addresses, mode)).await {
                Ok(Ok(peers)) => return Ok(peers),
                Ok(Err(error)) if Instant::now() < deadline => {
                    tracing::debug!(?error, "Waiting for CPU handoff participants");
                    timeout_at(deadline, sleep(POLL_INTERVAL))
                        .await
                        .wrap_err_with(|| format!("handoff peer deadline exceeded: {error:#}"))?;
                }
                Ok(Err(error)) => return Err(error),
                Err(error) => return Err(error).wrap_err("handoff peer deadline exceeded"),
            }
        }
    }

    // Called after migrations and before the candidate's ordinary startup guard.
    pub(super) async fn prepare_candidate(&self, handoffs: &CpuCellHandoffs) -> Result<()> {
        let own = self.status.read().await.clone();
        let intent = own
            .intent
            .as_ref()
            .ok_or_else(|| eyre!("missing candidate intent"))?;
        let deadline = Instant::now() + PREFLIGHT_TIMEOUT;
        let peers = loop {
            let peers = self
                .wait_peers(&self.source_urls, StartupMode::Active, deadline)
                .await?;
            ensure!(
                peers.iter().all(|peer| peer.deployment == intent.source
                    && peer.drain.as_ref().is_some_and(|d| !d.requested
                        && d.error.is_none()
                        && d.blocked_reason.is_none())),
                "source is not participating and eligible for this handoff"
            );
            if peers.iter().all(|peer| peer.ready) {
                break peers;
            }
            timeout_at(deadline, sleep(POLL_INTERVAL))
                .await
                .wrap_err("source readiness deadline exceeded")?;
        };
        self.status.write().await.source_processes =
            Some(std::array::from_fn(|i| peers[i].process));
        handoffs
            .request(intent.id, intent.source, intent.destination)
            .await
    }

    async fn source_step(&self, handoffs: &CpuCellHandoffs) -> Result<bool> {
        let Some(record) = handoffs.latest().await? else {
            return Ok(false);
        };
        let own = self.status.read().await.clone();
        if record.phase == HandoffPhase::Activated && record.destination == own.deployment {
            return Ok(false);
        }
        ensure!(
            record.source == own.deployment && record.phase == HandoffPhase::Requested,
            "source has an incompatible handoff record"
        );
        let intent = Intent {
            id: record.handoff_id,
            source: record.source,
            destination: record.destination,
        };
        self.status.write().await.intent = Some(intent.clone());
        let sources = self.peers(&self.source_urls, StartupMode::Active).await?;
        let candidates = self
            .peers(&self.candidate_urls, StartupMode::Candidate)
            .await?;
        let source_processes = std::array::from_fn(|i| sources[i].process);
        let candidate_processes = std::array::from_fn(|i| candidates[i].process);
        ensure!(
            sources.iter().all(|p| p.deployment == intent.source)
                && source_processes[own.party_id] == own.process,
            "source deployment or process changed"
        );
        ensure!(
            candidates.iter().all(|p| p.deployment == intent.destination
                && p.intent.as_ref() == Some(&intent)
                && p.source_processes == Some(source_processes)
                && p.error.is_none()),
            "candidate intent or source processes changed"
        );
        if !sources.iter().all(|p| p.ready) || !candidates.iter().all(|p| p.ready) {
            return Ok(false);
        }
        ensure!(
            source_processes
                .iter()
                .all(|id| !candidate_processes.contains(id)),
            "source and candidate processes overlap"
        );
        for source in &sources {
            let status = source
                .drain
                .as_ref()
                .ok_or_else(|| eyre!("missing source drain eligibility"))?;
            ensure!(
                status.error.is_none() && status.blocked_reason.is_none(),
                "source is not drain-eligible"
            );
            if status.requested {
                ensure!(
                    source.intent.as_ref() == Some(&intent)
                        && source.source_processes == Some(source_processes)
                        && source.candidate_processes == Some(candidate_processes),
                    "source is draining another handoff"
                );
            }
        }
        {
            let mut status = self.status.write().await;
            status.source_processes = Some(source_processes);
            status.candidate_processes = Some(candidate_processes);
        }
        if !sources.iter().all(|p| {
            p.intent.as_ref() == Some(&intent)
                && p.source_processes == Some(source_processes)
                && p.candidate_processes == Some(candidate_processes)
                && p.error.is_none()
        }) {
            return Ok(false);
        }
        self.drain
            .as_ref()
            .ok_or_else(|| eyre!("missing source drain"))?
            .request_managed(intent.id, &candidate_processes)
            .await?;
        Ok(true)
    }

    pub(super) async fn run_source(
        self,
        handoffs: CpuCellHandoffs,
        shutdown: Arc<ShutdownHandler>,
    ) {
        loop {
            let result = tokio::select! {
                _ = shutdown.wait_for_shutdown() => return,
                result = self.source_step(&handoffs) => result,
            };
            match result {
                Ok(true) => {
                    self.status.write().await.error = None;
                    break;
                }
                Ok(false) => {
                    self.status.write().await.error = None;
                }
                Err(error) => {
                    let reason = format!("{error:#}");
                    let mut status = self.status.write().await;
                    if status.error.as_ref() != Some(&reason) {
                        tracing::warn!(party_id = status.party_id, error = %reason, "CPU handoff coordination blocked");
                        status.error = Some(reason);
                    }
                    if self.drain.as_ref().is_some_and(|drain| drain.requested()) {
                        break;
                    }
                }
            }
            tokio::select! { _ = shutdown.wait_for_shutdown() => return, _ = sleep(POLL_INTERVAL) => {} }
        }
        // TaskMonitor treats a completed task as a failure, including successful drain initiation.
        shutdown.wait_for_shutdown().await;
    }

    async fn completion(&self, handoffs: &CpuCellHandoffs) -> Result<Option<Completion>> {
        let own = self.status.read().await.clone();
        let intent = own
            .intent
            .as_ref()
            .ok_or_else(|| eyre!("missing candidate intent"))?;
        let record = handoffs.get(intent.id).await?;
        ensure!(
            record.source == intent.source && record.destination == intent.destination,
            "durable handoff intent changed"
        );
        if record.phase != HandoffPhase::Drained {
            return Ok(None);
        }
        let source_processes = own
            .source_processes
            .ok_or_else(|| eyre!("source cohort was not verified"))?;
        let candidates: [Uuid; 3] = record
            .candidate_processes
            .ok_or_else(|| eyre!("missing durable candidate cohort"))?
            .try_into()
            .map_err(|_| eyre!("invalid durable candidate cohort"))?;
        ensure!(
            record.source_process == Some(source_processes[own.party_id])
                && candidates[own.party_id] == own.process,
            "durable completion belongs to replaced processes"
        );
        Ok(Some(Completion {
            source_process: source_processes[own.party_id],
            candidate_processes: candidates,
            report: serde_json::from_value(
                record
                    .drain_report
                    .ok_or_else(|| eyre!("missing durable drain report"))?,
            )?,
        }))
    }

    fn candidate_cohort(own: &Status, peers: &[Status; 3], completion: &Completion) -> Result<()> {
        ensure!(
            peers.iter().enumerate().all(|(party, p)| p.process
                == completion.candidate_processes[party]
                && p.intent == own.intent
                && p.source_processes == own.source_processes
                && p.error.is_none()),
            "candidate processes or handoff changed"
        );
        Ok(())
    }

    fn completed_reports(own: &Status, peers: &[Status; 3], completion: &Completion) -> Result<()> {
        Self::candidate_cohort(own, peers, completion)?;
        let sources = own
            .source_processes
            .ok_or_else(|| eyre!("missing verified source cohort"))?;
        ensure!(
            peers
                .iter()
                .enumerate()
                .all(|(party, p)| p
                    .completion
                    .as_ref()
                    .is_some_and(|c| c.source_process == sources[party]
                        && c.candidate_processes == completion.candidate_processes
                        && c.report == completion.report)),
            "durable completed watermark or publication differs across parties"
        );
        Ok(())
    }

    async fn wait_candidates_loaded(&self) -> Result<()> {
        let own = self.status.read().await.clone();
        let preflight_deadline = Instant::now() + PREFLIGHT_TIMEOUT;
        loop {
            let peers = self
                .wait_peers(
                    &self.candidate_urls,
                    StartupMode::Candidate,
                    preflight_deadline,
                )
                .await?;
            ensure!(
                peers.iter().all(|p| p.intent == own.intent
                    && (!p.ready || p.source_processes == own.source_processes)
                    && p.error.is_none()),
                "candidate preflight differs across parties"
            );
            if peers.iter().all(|p| p.ready) {
                break;
            }
            timeout_at(preflight_deadline, sleep(POLL_INTERVAL))
                .await
                .wrap_err("candidate preload deadline exceeded")?;
        }
        Ok(())
    }

    pub(super) async fn reconcile_candidate(
        &self,
        handoffs: &CpuCellHandoffs,
        store: &Store,
        candidate: &mut CandidateReconciliation,
        budget: Duration,
    ) -> Result<ReconciliationContext> {
        self.wait_candidates_loaded().await?;
        let deadline = Instant::now() + budget;
        let completion = loop {
            if let Some(completion) = timeout_at(deadline, self.completion(handoffs)).await?? {
                break completion;
            }
            timeout_at(deadline, sleep(POLL_INTERVAL))
                .await
                .wrap_err("durable drain completion deadline exceeded")?;
        };
        self.status.write().await.completion = Some(completion.clone());
        let own = self.status.read().await.clone();
        let sources = own
            .source_processes
            .ok_or_else(|| eyre!("missing verified source cohort"))?;
        loop {
            let peers = self
                .wait_peers(&self.candidate_urls, StartupMode::Candidate, deadline)
                .await?;
            if peers.iter().all(|p| p.completion.is_some()) {
                Self::completed_reports(&own, &peers, &completion)?;
                break;
            }
            Self::candidate_cohort(&own, &peers, &completion)?;
            timeout_at(deadline, sleep(POLL_INTERVAL)).await?;
        }
        let context = ReconciliationContext::new(
            own.intent
                .ok_or_else(|| eyre!("missing handoff intent"))?
                .id,
            sources,
            completion.candidate_processes,
            completion.report.watermark.clone(),
        )?;
        let deadline = Instant::now() + budget;
        let report = candidate
            .reconcile(store, context.clone(), deadline)
            .await?
            .clone();
        self.status.write().await.reconciliation = Some(report);
        loop {
            let peers = self
                .wait_peers(&self.candidate_urls, StartupMode::Candidate, deadline)
                .await?;
            Self::candidate_cohort(&*self.status.read().await, &peers, &completion)?;
            if let [Some(a), Some(b), Some(c)] = peers.map(|p| p.reconciliation) {
                ReconciliationReport::verify_agreement(&context, &[a, b, c])?;
                return Ok(context);
            }
            timeout_at(deadline, sleep(POLL_INTERVAL)).await?;
        }
    }

    pub(super) async fn fail(&self, error: &eyre::Report) {
        let mut status = self.status.write().await;
        status.error = Some(format!("{error:#}"));
        tracing::error!(party_id = status.party_id, error = ?error, "CPU candidate handoff failed; remains passive");
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    #[cfg(feature = "db_dependent")]
    pub(in crate::server) async fn assert_sources_require_error_free_acknowledgements(
        sources: &[Coordinator],
        handoffs: &[CpuCellHandoffs],
        loading_candidate: &Coordinator,
        candidate_process: Uuid,
    ) -> Result<()> {
        for (source, handoff) in sources.iter().zip(handoffs) {
            assert!(!source.source_step(handoff).await?);
            assert!(!source.drain.as_ref().unwrap().requested());
        }
        loading_candidate.mark_ready(candidate_process).await;
        // Publish all three acknowledgements before testing the final initiation gate.
        for (source, handoff) in sources.iter().zip(handoffs) {
            assert!(!source.source_step(handoff).await?);
        }
        sources[2].status.write().await.error = Some("candidate peer unavailable".into());
        assert!(!sources[0].source_step(&handoffs[0]).await?);
        assert!(!sources[0].drain.as_ref().unwrap().requested());
        sources[2].status.write().await.error = None;
        assert!(sources[0].source_step(&handoffs[0]).await?);
        assert!(sources[0].drain.as_ref().unwrap().requested());
        Ok(())
    }

    #[tokio::test]
    async fn staggered_preload_waits_then_checks_current_durable_reports() -> Result<()> {
        let listeners = futures::future::try_join_all(
            (0..3).map(|_| tokio::net::TcpListener::bind("127.0.0.1:0")),
        )
        .await?;
        let addresses: [String; 3] = std::array::from_fn(|i| {
            format!(
                "http://{}/handoff-status",
                listeners[i].local_addr().unwrap()
            )
        });
        let sources = std::array::from_fn(|_| Uuid::new_v4());
        let candidates = std::array::from_fn(|_| Uuid::new_v4());
        let destination = Uuid::new_v4();
        let intent = Uuid::new_v4();
        let mut coordinators = Vec::new();
        let mut tasks = Vec::new();
        for (party, listener) in listeners.into_iter().enumerate() {
            let config: Config = serde_json::from_value(serde_json::json!({
                "party_id": party, "startup_mode": "candidate", "db_backed_ingest": true,
                "cpu_cell": {"deployment_id": destination, "candidate_status_urls": serde_json::to_string(&addresses)?,
                    "handoff": {"id": intent, "source_deployment_id": sources[0], "source_status_urls": serde_json::to_string(&addresses)?}}
            }))?;
            let coordinator = Coordinator::new(&config, candidates[party], None)?.unwrap();
            if party < 2 {
                coordinator.status.write().await.source_processes = Some(sources);
                coordinator.mark_ready(candidates[party]).await;
            }
            let router = coordinator.routes();
            tasks.push(tokio::spawn(async move {
                axum::serve(listener, router).await.unwrap()
            }));
            coordinators.push(coordinator);
        }
        let wait = coordinators[0].wait_candidates_loaded();
        tokio::pin!(wait);
        // A peer still preparing has no source cohort yet; it must not fail fast B.
        assert!(tokio::time::timeout(Duration::from_millis(50), &mut wait)
            .await
            .is_err());
        coordinators[2].status.write().await.source_processes = Some(sources);
        coordinators[2].mark_ready(candidates[2]).await;
        tokio::time::timeout(Duration::from_secs(3), wait).await??;
        let mut peers: [Status; 3] = futures::future::join_all(
            coordinators
                .iter()
                .map(|c| async { c.status.read().await.clone() }),
        )
        .await
        .try_into()
        .unwrap();
        for (party, peer) in peers.iter_mut().enumerate() {
            peer.completion = Some(Completion {
                source_process: sources[party],
                candidate_processes: candidates,
                report: drain::Report {
                    watermark: Some("42".into()),
                    last_publication: None,
                },
            });
        }
        let own = peers[0].clone();
        let completion = own.completion.as_ref().unwrap();
        Coordinator::completed_reports(&own, &peers, completion)?;
        let original = peers[2].clone();
        peers[2].completion = None;
        assert!(Coordinator::completed_reports(&own, &peers, completion).is_err());
        peers[2] = original.clone();
        peers[2].completion.as_mut().unwrap().report.watermark = Some("43".into());
        assert!(Coordinator::completed_reports(&own, &peers, completion).is_err());
        peers[2] = original;
        peers[2].process = Uuid::new_v4();
        assert!(Coordinator::completed_reports(&own, &peers, completion).is_err());
        for task in tasks {
            task.abort();
        }
        Ok(())
    }
}
