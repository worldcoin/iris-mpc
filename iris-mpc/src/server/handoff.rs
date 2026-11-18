//! Planned A/B handoff over party-local records and the existing peer status network.
use super::{
    drain,
    reconciliation::{CandidateReconciliation, ReconciliationContext, ReconciliationReport},
};
use ampc_server_utils::shutdown_handler::ShutdownHandler;
use axum::{routing::get, Json, Router};
use eyre::{ensure, eyre, Result, WrapErr};
use iris_mpc_common::{
    config::{CommonConfig, Config, CpuCellSlot, StartupMode},
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

// Durable-C reports with deployment-scoped handoff progress.
const PROTOCOL: u32 = 2;
const POLL_INTERVAL: Duration = Duration::from_secs(1);
const PREFLIGHT_TIMEOUT: Duration = Duration::from_secs(60 * 60);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Intent {
    id: Uuid,
    source: Uuid,
    destination: Uuid,
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
    drain: Option<drain::Status>,
    completion: Option<drain::Report>,
    reconciliation: Option<ReconciliationReport>,
    error: Option<String>,
}

#[derive(Clone)]
pub(super) struct Coordinator {
    status: Arc<RwLock<Status>>,
    source_urls: [reqwest::Url; 3],
    own_urls: [reqwest::Url; 3],
    other_urls: [reqwest::Url; 3],
    client: reqwest::Client,
    drain: Arc<std::sync::RwLock<Option<Arc<drain::Drain>>>>,
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
        let Some(peers) = &cell.peers else {
            ensure!(cell.handoff.is_none(), "automatic handoff requires peers");
            return Ok(None);
        };
        let (own, other) = match peers.slot {
            CpuCellSlot::A => (&peers.a, &peers.b),
            CpuCellSlot::B => (&peers.b, &peers.a),
        };
        let own_urls = urls(&own.0)?;
        let other_urls = urls(&other.0)?;
        ensure!(config.party_id < 3, "invalid handoff party");
        ensure!(
            config.db_backed_ingest && config.cpu_drain_timeout_secs > 0,
            "runtime handoff requires DB-backed intake and a positive drain budget"
        );
        let peers = config
            .server_coordination
            .as_ref()
            .ok_or_else(|| eyre!("handoff requires server coordination"))?;
        ensure!(
            peers.node_hostnames.len() == 3 && peers.healthcheck_ports.len() == 3,
            "handoff requires three serving addresses"
        );
        let serving_urls = urls(&std::array::from_fn(|i| {
            format!(
                "http://{}:{}/handoff-status",
                peers.node_hostnames[i], peers.healthcheck_ports[i]
            )
        }))?;
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
            (serving_urls, None)
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
                drain: None,
                completion: None,
                reconciliation: None,
                error: None,
            })),
            source_urls,
            own_urls,
            other_urls,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(3))
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            drain: Arc::new(std::sync::RwLock::new(drain)),
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
                    status.drain = coordinator.drain().map(|drain| drain.status());
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

    fn drain(&self) -> Option<Arc<drain::Drain>> {
        self.drain.read().unwrap().clone()
    }

    pub(super) async fn agreed_completion(&self) -> Result<drain::Report> {
        self.status
            .read()
            .await
            .completion
            .clone()
            .ok_or_else(|| eyre!("missing agreed durable completion"))
    }

    /// Keep both status listeners bound to the same state while adopting the
    /// ordinary serving listener's fresh UUID and drain handle.
    pub(super) async fn bind_serving(
        &self,
        config: &Config,
        process: Uuid,
        drain: Arc<drain::Drain>,
    ) -> Result<Self> {
        let peers = config
            .server_coordination
            .as_ref()
            .ok_or_else(|| eyre!("missing serving coordination"))?;
        let addresses = std::array::from_fn(|party| {
            format!(
                "http://{}:{}/handoff-status",
                peers.node_hostnames[party], peers.healthcheck_ports[party]
            )
        });
        let mut serving = self.clone();
        serving.source_urls = urls(&addresses)?;
        *self.drain.write().unwrap() = Some(drain);
        self.status.write().await.process = process;
        Ok(serving)
    }

    pub(super) async fn activate_candidate(&self, handoffs: &CpuCellHandoffs) -> Result<()> {
        let own = self.status.read().await.clone();
        let peers = self.peers(&self.own_urls, StartupMode::Candidate).await?;
        let intent = own
            .intent
            .as_ref()
            .ok_or_else(|| eyre!("missing activation intent"))?;
        let completion = own
            .completion
            .as_ref()
            .ok_or_else(|| eyre!("missing activation completion"))?;
        let report = own
            .reconciliation
            .as_ref()
            .ok_or_else(|| eyre!("missing activation reconciliation"))?;
        Self::completed_reports(&own, &peers, completion)?;
        let [Some(a), Some(b), Some(c)] = peers.map(|peer| peer.reconciliation) else {
            return Err(eyre!("missing peer activation reconciliation"));
        };
        ReconciliationReport::verify_agreement(&report.context, &[a, b, c])?;
        handoffs
            .activate(
                intent.id,
                own.deployment,
                &serde_json::to_value(completion)?,
            )
            .await?;
        let mut status = self.status.write().await;
        status.mode = StartupMode::Active;
        status.ready = false;
        Ok(())
    }

    pub(super) async fn serving_barrier(
        &self,
        expected: &[Uuid; 3],
        context: &ReconciliationContext,
        deadline: Instant,
    ) -> Result<()> {
        let own = self.status.read().await.clone();
        let peers = self
            .wait_peers(&self.own_urls, StartupMode::Active, deadline)
            .await?;
        Self::candidate_peers(&own, &peers)?;
        ensure!(
            peers
                .iter()
                .zip(expected)
                .all(|(peer, process)| peer.process == *process),
            "serving cohort process changed"
        );
        ensure!(
            peers.iter().all(|peer| peer.completion == own.completion),
            "serving completion or final publication differs"
        );
        let [Some(a), Some(b), Some(c)] = peers.map(|peer| peer.reconciliation) else {
            return Err(eyre!("missing serving reconciliation"));
        };
        ReconciliationReport::verify_agreement(context, &[a, b, c])
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
                    && (status.mode == mode
                        || (mode == StartupMode::Candidate
                            && status.mode == StartupMode::Active
                            && status.completion.is_some()
                            && status.reconciliation.is_some()))
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
    pub(super) async fn prepare_candidate(
        &self,
        handoffs: &CpuCellHandoffs,
    ) -> Result<StartupMode> {
        let own = self.status.read().await.clone();
        let intent = own
            .intent
            .as_ref()
            .ok_or_else(|| eyre!("missing candidate intent"))?;
        if let Some(record) = handoffs.latest().await? {
            if record.handoff_id == intent.id {
                ensure!(
                    record.source == intent.source && record.destination == intent.destination,
                    "durable handoff intent changed"
                );
                if record.phase == HandoffPhase::Activated {
                    self.status.write().await.mode = StartupMode::Active;
                    return Ok(StartupMode::Active);
                }
                return Ok(StartupMode::Candidate);
            }
        }
        let deadline = Instant::now() + PREFLIGHT_TIMEOUT;
        loop {
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
                break;
            }
            timeout_at(deadline, sleep(POLL_INTERVAL))
                .await
                .wrap_err("source readiness deadline exceeded")?;
        }
        handoffs
            .request(intent.id, intent.source, intent.destination)
            .await?;
        Ok(StartupMode::Candidate)
    }

    pub(super) async fn wait_cold_cohort(&self, deadline: Instant) -> Result<()> {
        let own = self.status.read().await.clone();
        loop {
            let peers = self
                .wait_peers(&self.own_urls, StartupMode::Active, deadline)
                .await?;
            Self::candidate_peers(&own, &peers)?;
            // Just-promoted peers still have warm evidence; already-serving peers are ready.
            if peers
                .iter()
                .all(|p| !p.ready && p.completion.is_none() && p.reconciliation.is_none())
            {
                return Ok(());
            }
            timeout_at(deadline, sleep(POLL_INTERVAL))
                .await
                .wrap_err("waiting for all three activated parties to cold start")?;
        }
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
        ensure!(
            sources.iter().all(|p| p.deployment == intent.source)
                && sources[own.party_id].process == own.process,
            "source deployment or process changed"
        );
        let mut started = false;
        for source in &sources {
            let drain = source
                .drain
                .as_ref()
                .ok_or_else(|| eyre!("missing source drain eligibility"))?;
            ensure!(
                drain.error.is_none() && drain.blocked_reason.is_none(),
                "source is not drain-eligible"
            );
            if drain.requested {
                ensure!(
                    source.intent.as_ref() == Some(&intent),
                    "source is draining another handoff"
                );
                started = true;
            }
        }
        // Once a peer seals this handoff, finish draining even if B is reloading.
        if !started {
            let candidates = self.peers(&self.other_urls, StartupMode::Candidate).await?;
            ensure!(
                candidates.iter().all(|p| p.deployment == intent.destination
                    && p.intent.as_ref() == Some(&intent)
                    && p.error.is_none()),
                "candidate intent differs or candidate failed"
            );
            if !sources.iter().all(|p| p.ready) || !candidates.iter().all(|p| p.ready) {
                return Ok(false);
            }
        }
        self.drain()
            .ok_or_else(|| eyre!("missing source drain"))?
            .request_managed(intent.id)
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
                    if self.drain().is_some_and(|drain| drain.requested()) {
                        break;
                    }
                }
            }
            tokio::select! { _ = shutdown.wait_for_shutdown() => return, _ = sleep(POLL_INTERVAL) => {} }
        }
        // TaskMonitor treats a completed task as a failure, including successful drain initiation.
        shutdown.wait_for_shutdown().await;
    }

    async fn completion(&self, handoffs: &CpuCellHandoffs) -> Result<Option<drain::Report>> {
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
        Ok(Some(serde_json::from_value(
            record
                .drain_report
                .ok_or_else(|| eyre!("missing durable drain report"))?,
        )?))
    }

    fn candidate_peers(own: &Status, peers: &[Status; 3]) -> Result<()> {
        ensure!(
            peers.iter().all(|p| p.deployment == own.deployment
                && p.intent == own.intent
                && p.error.is_none()),
            "candidate handoff differs or candidate failed"
        );
        ensure!(
            own.mode != StartupMode::Candidate
                || peers
                    .iter()
                    .all(|peer| peer.mode != StartupMode::Active || !peer.ready),
            "destination already serving; fresh reconciliation requires coordinated recovery"
        );
        Ok(())
    }

    fn completed_reports(
        own: &Status,
        peers: &[Status; 3],
        completion: &drain::Report,
    ) -> Result<()> {
        Self::candidate_peers(own, peers)?;
        ensure!(
            peers
                .iter()
                .all(|p| (p.ready || p.mode == StartupMode::Active)
                    && p.completion.as_ref() == Some(completion)),
            "durable completed watermark or publication differs across parties"
        );
        Ok(())
    }

    async fn wait_candidates_loaded(&self) -> Result<()> {
        let own = self.status.read().await.clone();
        let preflight_deadline = Instant::now() + PREFLIGHT_TIMEOUT;
        loop {
            let peers = self
                .wait_peers(&self.own_urls, StartupMode::Candidate, preflight_deadline)
                .await?;
            Self::candidate_peers(&own, &peers)?;
            if peers
                .iter()
                .all(|p| p.ready || p.mode == StartupMode::Active)
            {
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

        loop {
            let peers = self
                .wait_peers(&self.own_urls, StartupMode::Candidate, deadline)
                .await?;
            if peers
                .iter()
                .all(|p| (p.ready || p.mode == StartupMode::Active) && p.completion.is_some())
            {
                Self::completed_reports(&own, &peers, &completion)?;
                break;
            }
            Self::candidate_peers(&own, &peers)?;
            timeout_at(deadline, sleep(POLL_INTERVAL)).await?;
        }
        let context = ReconciliationContext::new(
            own.intent
                .ok_or_else(|| eyre!("missing handoff intent"))?
                .id,
            completion.watermark.clone(),
        )?;
        let deadline = Instant::now() + budget;
        let report = candidate
            .reconcile(store, context.clone(), deadline)
            .await?
            .clone();
        self.status.write().await.reconciliation = Some(report);
        loop {
            let peers = self
                .wait_peers(&self.own_urls, StartupMode::Candidate, deadline)
                .await?;
            Self::candidate_peers(&*self.status.read().await, &peers)?;
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
    pub(in crate::server) async fn reject_changed_activation_evidence(
        coordinators: &[Coordinator],
        handoffs: &[CpuCellHandoffs],
    ) -> Result<()> {
        let original = coordinators[2].status.read().await.clone();
        for missing_completion in [true, false] {
            let mut peer = original.clone();
            if missing_completion {
                peer.completion = None;
            } else {
                peer.reconciliation = None;
            }
            *coordinators[2].status.write().await = peer;
            assert!(coordinators[0]
                .activate_candidate(&handoffs[0])
                .await
                .is_err());
        }
        let mut changed = original.clone();
        changed.completion.as_mut().unwrap().watermark =
            Some(iris_mpc_store::normalize_sns_sequence_number("43")?);
        *coordinators[2].status.write().await = changed;
        assert!(coordinators[0]
            .activate_candidate(&handoffs[0])
            .await
            .is_err());
        let mut changed = original.clone();
        changed.reconciliation.as_mut().unwrap().context =
            ReconciliationContext::new(Uuid::new_v4(), None)?;
        *coordinators[2].status.write().await = changed;
        assert!(coordinators[0]
            .activate_candidate(&handoffs[0])
            .await
            .is_err());
        let mut replacement = original.clone();
        replacement.process = Uuid::new_v4();
        replacement.ready = false;
        replacement.completion = None;
        replacement.reconciliation = None;
        *coordinators[2].status.write().await = replacement;
        assert!(coordinators[0]
            .activate_candidate(&handoffs[0])
            .await
            .is_err());
        let mut serving = original.clone();
        serving.mode = StartupMode::Active;
        serving.ready = true;
        *coordinators[2].status.write().await = serving;
        assert!(coordinators[0]
            .activate_candidate(&handoffs[0])
            .await
            .is_err());
        *coordinators[2].status.write().await = original;
        for handoff in handoffs {
            let record = handoff.latest().await?.unwrap();
            assert_eq!(record.phase, HandoffPhase::Drained);
            assert!(handoff.check_startup(record.source, false).await.is_err());
        }
        Ok(())
    }

    #[cfg(feature = "db_dependent")]
    pub(in crate::server) async fn reject_changed_serving_evidence(
        coordinators: &[Coordinator],
        context: &ReconciliationContext,
    ) -> Result<()> {
        let mut expected = [Uuid::nil(); 3];
        for (party, coordinator) in coordinators.iter().enumerate() {
            expected[party] = coordinator.status.read().await.process;
        }
        let original = coordinators[2].status.read().await.clone();
        let mut changed = original.clone();
        changed.process = Uuid::new_v4();
        *coordinators[2].status.write().await = changed;
        assert!(coordinators[0]
            .serving_barrier(
                &expected,
                context,
                Instant::now() + Duration::from_millis(100)
            )
            .await
            .is_err());
        let mut changed = original.clone();
        changed.reconciliation = None;
        *coordinators[2].status.write().await = changed;
        let error = coordinators[0]
            .serving_barrier(
                &expected,
                context,
                Instant::now() + Duration::from_millis(100),
            )
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "missing serving reconciliation");
        let mut changed = original.clone();
        changed.completion.as_mut().unwrap().last_publication = Some(drain::Publication {
            batch_id: 1,
            batch_sha256: "ab".repeat(32),
        });
        *coordinators[2].status.write().await = changed;
        assert!(coordinators[0]
            .serving_barrier(
                &expected,
                context,
                Instant::now() + Duration::from_millis(100)
            )
            .await
            .is_err());
        *coordinators[2].status.write().await = original;
        Ok(())
    }

    #[cfg(feature = "db_dependent")]
    pub(in crate::server) async fn reject_incompatible_next_candidate(
        sources: &[Coordinator],
        handoffs: &[CpuCellHandoffs],
        candidate: &Coordinator,
    ) -> Result<()> {
        let original = candidate.status.read().await.clone();
        for wrong_deployment in [true, false] {
            let mut incompatible = original.clone();
            if wrong_deployment {
                incompatible.deployment = Uuid::new_v4();
            } else {
                incompatible.intent.as_mut().unwrap().id = Uuid::new_v4();
            }
            *candidate.status.write().await = incompatible;
            for (source, handoff) in sources.iter().zip(handoffs) {
                assert!(source.source_step(handoff).await.is_err());
                assert!(!source.drain().unwrap().requested());
                assert_eq!(
                    handoff.latest().await?.unwrap().phase,
                    HandoffPhase::Requested
                );
            }
        }
        *candidate.status.write().await = original;
        for (source, handoff) in sources.iter().zip(handoffs) {
            assert!(!source.source_step(handoff).await?);
            assert!(!source.drain().unwrap().requested());
            assert!(source.status.read().await.ready);
            assert_eq!(
                handoff.latest().await?.unwrap().phase,
                HandoffPhase::Requested
            );
        }
        Ok(())
    }

    #[cfg(feature = "db_dependent")]
    pub(in crate::server) async fn assert_sources_wait_for_preload_then_finish_drain(
        sources: &[Coordinator],
        handoffs: &[CpuCellHandoffs],
        loading_candidate: &Coordinator,
        candidate_process: Uuid,
    ) -> Result<()> {
        for (source, handoff) in sources.iter().zip(handoffs) {
            assert!(!source.source_step(handoff).await?);
            assert!(!source.drain().unwrap().requested());
        }
        loading_candidate.mark_ready(candidate_process).await;
        assert!(sources[0].source_step(&handoffs[0]).await?);
        assert!(sources[0].drain().unwrap().requested());
        loading_candidate.status.write().await.ready = false;
        for (source, handoff) in sources.iter().zip(handoffs).skip(1) {
            assert!(source.source_step(handoff).await?);
        }
        loading_candidate.mark_ready(candidate_process).await;
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
        let source = Uuid::new_v4();
        let candidates: [Uuid; 3] = std::array::from_fn(|_| Uuid::new_v4());
        let destination = Uuid::new_v4();
        let intent = Uuid::new_v4();
        let mut coordinators = Vec::new();
        let mut tasks = Vec::new();
        for (party, listener) in listeners.into_iter().enumerate() {
            let config: Config = serde_json::from_value(serde_json::json!({
                "party_id": party, "startup_mode": "candidate", "db_backed_ingest": true,
                "server_coordination": {"healthcheck_ports": "[\"3000\",\"3001\",\"3002\"]"},
                "cpu_cell": {"deployment_id": destination,
                    "peers": {"slot": "b", "a": serde_json::to_string(&addresses)?, "b": serde_json::to_string(&addresses)?},
                    "handoff": {"id": intent, "source_deployment_id": source, "source_status_urls": serde_json::to_string(&addresses)?}}
            }))?;
            if party == 0 {
                let mut missing = config.clone();
                missing.cpu_cell.as_mut().unwrap().peers = None;
                assert!(Coordinator::new(&missing, candidates[party], None)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("automatic handoff requires peers"));
                for invalid in ["not a URL", &addresses[1]] {
                    let mut malformed = config.clone();
                    malformed
                        .cpu_cell
                        .as_mut()
                        .unwrap()
                        .peers
                        .as_mut()
                        .unwrap()
                        .a
                        .0[0] = invalid.into();
                    assert!(Coordinator::new(&malformed, candidates[party], None).is_err());
                }
                for (hosts, ports) in [(2, 3), (3, 2), (4, 3), (3, 4)] {
                    let mut malformed = config.clone();
                    let peers = malformed.server_coordination.as_mut().unwrap();
                    peers.node_hostnames.resize(hosts, "localhost".into());
                    peers.healthcheck_ports.resize(ports, "3000".into());
                    assert_eq!(
                        Coordinator::new(&malformed, candidates[party], None)
                            .err()
                            .unwrap()
                            .to_string(),
                        "handoff requires three serving addresses"
                    );
                }
                for (host, port) in [
                    ("localhost", "3000"),
                    ("localhost", "invalid"),
                    ("bad host", "3000"),
                ] {
                    let mut malformed = config.clone();
                    let peers = malformed.server_coordination.as_mut().unwrap();
                    peers.node_hostnames = vec![host.into(); 3];
                    peers.healthcheck_ports = vec![port.into(); 3];
                    assert!(Coordinator::new(&malformed, candidates[party], None).is_err());
                }
            }
            let coordinator = Coordinator::new(&config, candidates[party], None)?.unwrap();
            if party < 2 {
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
        // A still-loading peer must not fail the other candidates.
        assert!(tokio::time::timeout(Duration::from_millis(50), &mut wait)
            .await
            .is_err());
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
        for peer in &mut peers {
            peer.completion = Some(drain::Report {
                watermark: Some("42".into()),
                last_publication: None,
            });
        }
        let own = peers[0].clone();
        let completion = own.completion.as_ref().unwrap();
        Coordinator::completed_reports(&own, &peers, completion)?;
        let original = peers[2].clone();
        peers[2].completion = None;
        assert!(Coordinator::completed_reports(&own, &peers, completion).is_err());
        peers[2] = original.clone();
        peers[2].completion.as_mut().unwrap().watermark = Some("43".into());
        assert!(Coordinator::completed_reports(&own, &peers, completion).is_err());
        peers[2] = original;
        peers[2].process = Uuid::new_v4();
        peers[2].ready = false;
        peers[2].completion = None;
        assert!(Coordinator::completed_reports(&own, &peers, completion).is_err());
        peers[2].ready = true;
        peers[2].completion = Some(completion.clone());
        Coordinator::completed_reports(&own, &peers, completion)?;
        for task in tasks {
            task.abort();
        }
        Ok(())
    }
}
