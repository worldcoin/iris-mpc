//! In-process, one-way drain with read-only status reporting.
use ampc_server_utils::config::ServerCoordinationConfig;
use axum::{routing::get, Json, Router};
use eyre::Result;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use tokio::sync::Notify;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Publication {
    pub batch_id: u64,
    pub batch_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    pub watermark: Option<String>,
    pub persisted_count: u64,
    pub persisted_sha256: String,
    pub last_publication: Option<Publication>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Status {
    pub party_id: usize,
    pub process_uuid: String,
    pub requested: bool,
    pub offered_through: u64,
    pub cut: Option<u64>,
    pub phase: String,
    pub report: Option<Report>,
    pub error: Option<String>,
    pub blocked_reason: Option<String>,
}

async fn read_status(mut response: reqwest::Response) -> Result<Status> {
    const MAX_STATUS_BYTES: usize = 64 * 1024;
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        eyre::ensure!(
            body.len() + chunk.len() <= MAX_STATUS_BYTES,
            "drain status exceeds 64 KiB"
        );
        body.extend_from_slice(&chunk);
    }
    let status: Status = serde_json::from_slice(&body)?;
    eyre::ensure!(
        uuid::Uuid::parse_str(&status.process_uuid).is_ok() && status.party_id < 3,
        "invalid drain identity"
    );
    eyre::ensure!(
        matches!(
            status.phase.as_str(),
            "active" | "freezing" | "draining" | "quiescent" | "paused" | "failed"
        ),
        "invalid drain phase"
    );
    if let Some(report) = &status.report {
        let valid_sha =
            |sha: &str| sha.len() == 64 && sha.bytes().all(|byte| byte.is_ascii_hexdigit());
        eyre::ensure!(
            valid_sha(&report.persisted_sha256),
            "invalid persisted SHA256"
        );
        eyre::ensure!(
            report.persisted_count == 0 || report.watermark.is_some(),
            "missing persisted watermark"
        );
        if let Some(watermark) = &report.watermark {
            eyre::ensure!(
                iris_mpc_store::normalize_sns_sequence_number(watermark)? == *watermark,
                "invalid persisted watermark"
            );
        }
        if let Some(publication) = &report.last_publication {
            eyre::ensure!(
                valid_sha(&publication.batch_sha256),
                "invalid publication SHA256"
            );
        }
    }
    Ok(status)
}

struct Inner {
    status: Status,
    deadline: Option<Instant>,
    ingest_stopped: bool,
    receiver_stopped: bool,
    pending_results: usize,
    last_publication: Option<Publication>,
}

pub struct Drain {
    inner: Mutex<Inner>,
    changed: Notify,
    ready: Mutex<Option<Arc<AtomicBool>>>,
    timeout: Duration,
}

impl Drain {
    pub fn new(party_id: usize, timeout: Duration, supported: bool) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(Inner {
                status: Status {
                    party_id,
                    process_uuid: String::new(),
                    requested: false,
                    offered_through: 0,
                    cut: None,
                    phase: "active".into(),
                    report: None,
                    error: None,
                    blocked_reason: if supported {
                        None
                    } else {
                        Some("drain requires CPU linear scan with durable DB-backed ingest".into())
                    },
                },
                deadline: None,
                ingest_stopped: false,
                receiver_stopped: false,
                pending_results: 0,
                last_publication: None,
            }),
            changed: Notify::new(),
            ready: Mutex::new(None),
            timeout,
        })
    }

    pub fn routes(self: &Arc<Self>) -> Router {
        let this = self.clone();
        Router::new().route(
            "/drain-status",
            get(move || {
                let this = this.clone();
                async move { Json(this.status()) }
            }),
        )
    }

    pub fn initialize(&self, uuid: String, ready: Arc<AtomicBool>) {
        self.inner.lock().unwrap().status.process_uuid = uuid;
        *self.ready.lock().unwrap() = Some(ready);
    }

    pub fn request(&self) {
        let mut inner = self.inner.lock().unwrap();
        if inner.status.requested {
            return;
        }
        inner.status.requested = true;
        inner.deadline = Some(Instant::now() + self.timeout);
        if inner.status.error.is_none() {
            inner.status.phase = "freezing".into();
        }
        if let Some(ready) = &*self.ready.lock().unwrap() {
            ready.store(false, Ordering::SeqCst);
        }
        if let Some(reason) = &inner.status.blocked_reason.clone() {
            inner.status.phase = "failed".into();
            inner.status.error = Some(reason.clone());
        }
        metrics::counter!("cpu_cell_drain_transitions", "state" => "requested").increment(1);
        tracing::info!(
            party_id = inner.status.party_id,
            offered_through = inner.status.offered_through,
            "CPU cell drain requested"
        );
        self.changed.notify_waiters();
    }

    pub fn mark_ready(&self) {
        let inner = self.inner.lock().unwrap();
        if !inner.status.requested {
            if let Some(ready) = &*self.ready.lock().unwrap() {
                ready.store(true, Ordering::SeqCst);
            }
        }
    }

    pub fn status(&self) -> Status {
        self.inner.lock().unwrap().status.clone()
    }
    pub fn block(&self, reason: &str) {
        let mut inner = self.inner.lock().unwrap();
        inner.status.blocked_reason = Some(reason.into());
        if inner.status.requested {
            self.fail_locked(&mut inner, reason);
        }
    }

    pub fn requested(&self) -> bool {
        self.inner.lock().unwrap().status.requested
    }
    pub fn failed(&self) -> bool {
        self.inner.lock().unwrap().status.error.is_some()
    }
    pub fn fail(&self, operation: &str) {
        let mut inner = self.inner.lock().unwrap();
        self.fail_locked(&mut inner, operation);
    }

    fn fail_locked(&self, inner: &mut Inner, operation: &str) {
        inner.status.requested = true;
        inner.status.report = None;
        if let Some(ready) = &*self.ready.lock().unwrap() {
            ready.store(false, Ordering::SeqCst);
        }
        if inner.status.error.is_none() {
            inner.status.phase = "failed".into();
            inner.status.error = Some(operation.into());
            metrics::counter!("cpu_cell_drain_transitions", "state" => "failed").increment(1);
            tracing::error!(
                party_id = inner.status.party_id,
                operation,
                "CPU cell drain failed; process remains paused"
            );
        }
        self.changed.notify_waiters();
    }

    /// Publish a positive offer while holding the same mutex used by `request`.
    /// An offer cannot be retracted: a peer may already have admitted that batch.
    pub fn publish_offer(&self, batch_id: u64, messages: u32, publish: impl FnOnce()) -> bool {
        let mut inner = self.inner.lock().unwrap();
        if inner.status.requested {
            return false;
        }
        if messages > 0 {
            inner.status.offered_through = inner.status.offered_through.max(batch_id);
        }
        publish();
        true
    }

    pub async fn may_finish(&self, batch_id: u64) -> bool {
        loop {
            let notified = self.changed.notified();
            let status = self.status();
            if status.error.is_some() {
                return false;
            }
            if !status.requested {
                return true;
            }
            if let Some(cut) = status.cut {
                return batch_id <= cut;
            }
            tokio::select! { _ = notified => {}, _ = tokio::time::sleep(Duration::from_millis(100)) => {} }
        }
    }

    /// Only a read-only batch-agreement wait can be interrupted. Once claims,
    /// parsing or publication start, their futures must run to settlement.
    pub async fn wait_for_exclusion(&self, batch_id: u64) {
        loop {
            let notified = self.changed.notified();
            let status = self.status();
            if status.error.is_some() || status.cut.is_some_and(|cut| batch_id > cut) {
                return;
            }
            tokio::select! { _ = notified => {}, _ = tokio::time::sleep(Duration::from_millis(100)) => {} }
        }
    }

    pub fn ingest_stopped(&self) {
        self.inner.lock().unwrap().ingest_stopped = true;
        self.changed.notify_waiters();
    }
    pub fn receiver_stopped(&self) {
        self.inner.lock().unwrap().receiver_stopped = true;
        self.changed.notify_waiters();
    }
    pub fn result_started(&self) {
        self.inner.lock().unwrap().pending_results += 1;
    }
    pub fn result_sent(&self, publication: Publication) {
        let mut inner = self.inner.lock().unwrap();
        inner.pending_results = inner
            .pending_results
            .checked_sub(1)
            .expect("result must have been admitted");
        inner.last_publication = Some(publication);
        self.changed.notify_waiters();
    }
    pub fn result_send_failed(&self) {
        self.inner.lock().unwrap().pending_results -= 1;
        if self.requested() {
            self.fail("result queue closed before publication");
        } else {
            self.block("result queue closed before publication");
        }
    }
    pub fn settled(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.ingest_stopped && inner.receiver_stopped && inner.pending_results == 0
    }
    pub fn remaining_budget(&self) -> Duration {
        self.inner
            .lock()
            .unwrap()
            .deadline
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
            .unwrap_or_default()
    }
    pub fn last_publication(&self) -> Option<Publication> {
        self.inner.lock().unwrap().last_publication.clone()
    }
    pub fn publish_report(&self, report: Report) {
        let mut inner = self.inner.lock().unwrap();
        if inner.status.error.is_none() {
            inner.status.report = Some(report);
            inner.status.phase = "quiescent".into();
            metrics::counter!("cpu_cell_drain_transitions", "state" => "quiescent").increment(1);
            tracing::info!(
                party_id = inner.status.party_id,
                cut = inner.status.cut,
                "CPU cell workers settled; comparing persisted prefix"
            );
        }
    }

    pub async fn coordinate(
        self: Arc<Self>,
        config: ServerCoordinationConfig,
        expected_uuids: HashSet<String>,
    ) -> Result<()> {
        if expected_uuids.len() != 3 {
            self.block("verified drain cohort must contain three distinct process UUIDs");
        }
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(3))
            .build()?;
        let mut last_peer_failure: Option<String> = None;
        loop {
            tokio::time::sleep(if self.status().phase == "paused" {
                Duration::from_secs(10)
            } else {
                Duration::from_millis(500)
            })
            .await;
            if !self.requested() || self.failed() {
                continue;
            }
            if self
                .inner
                .lock()
                .unwrap()
                .deadline
                .is_some_and(|deadline| Instant::now() >= deadline)
            {
                self.fail(&format!(
                    "drain deadline exceeded; {}; outstanding operations were not cancelled",
                    last_peer_failure
                        .as_deref()
                        .unwrap_or("workers or peer reports are not settled")
                ));
                continue;
            }
            let mut statuses = vec![self.status()];
            let own_party = config.party_id;
            for party in 0..3 {
                if party == own_party {
                    continue;
                }
                let url = format!(
                    "http://{}:{}/drain-status",
                    config.node_hostnames[party], config.healthcheck_ports[party]
                );
                let remaining = self
                    .inner
                    .lock()
                    .unwrap()
                    .deadline
                    .map(|deadline| deadline.saturating_duration_since(Instant::now()))
                    .unwrap_or(Duration::from_secs(3));
                match client
                    .get(url)
                    .timeout(
                        remaining
                            .min(Duration::from_secs(3))
                            .max(Duration::from_millis(1)),
                    )
                    .send()
                    .await
                {
                    Ok(response) if response.status().is_success() => match read_status(response)
                        .await
                    {
                        Ok(status)
                            if status.party_id == party
                                && expected_uuids.contains(&status.process_uuid) =>
                        {
                            statuses.push(status)
                        }
                        _ => {
                            self.fail("invalid party or current process identity in drain report");
                            break;
                        }
                    },
                    Ok(response) => {
                        let status = response.status();
                        last_peer_failure = Some(format!("party {party} returned HTTP {status}"));
                        tracing::debug!(party, %status, "Waiting for peer drain status");
                        break;
                    }
                    Err(error) => {
                        last_peer_failure =
                            Some(format!("party {party} drain-status request failed"));
                        tracing::debug!(party, error = %error, "Waiting for peer drain status");
                        break;
                    }
                }
            }
            if statuses.len() != 3 {
                continue;
            }
            let uuids: HashSet<_> = statuses
                .iter()
                .map(|status| status.process_uuid.clone())
                .collect();
            if uuids != expected_uuids {
                self.fail("drain cohort process identities changed");
                continue;
            }
            if statuses.iter().any(|status| status.error.is_some()) {
                self.fail("a peer reported drain failure");
                continue;
            }
            if !statuses.iter().all(|status| status.requested) {
                let waiting: Vec<_> = statuses
                    .iter()
                    .filter(|status| !status.requested)
                    .map(|status| status.party_id)
                    .collect();
                last_peer_failure =
                    Some(format!("waiting for drain request on parties {waiting:?}"));
                continue;
            }
            let cut = statuses
                .iter()
                .map(|status| status.offered_through)
                .min()
                .unwrap();
            {
                let mut inner = self.inner.lock().unwrap();
                if inner.status.error.is_some() {
                    continue;
                }
                if inner
                    .deadline
                    .is_some_and(|deadline| Instant::now() >= deadline)
                {
                    drop(inner);
                    self.fail("drain deadline exceeded during peer exchange");
                    continue;
                }
                if inner.status.cut.is_none() {
                    inner.status.cut = Some(cut);
                    inner.status.phase = "draining".into();
                } else if inner.status.cut != Some(cut) {
                    drop(inner);
                    self.fail("frozen drain admission cut changed");
                    continue;
                }
            }
            self.changed.notify_waiters();
            if statuses.iter().all(|status| status.report.is_some()) {
                if statuses.iter().any(|status| {
                    status.cut != Some(cut)
                        || !matches!(status.phase.as_str(), "quiescent" | "paused")
                }) {
                    self.fail("peer report was produced outside the frozen quiescent cut");
                    continue;
                }
                let reports: Vec<_> = statuses
                    .iter()
                    .map(|status| status.report.as_ref().unwrap())
                    .collect();
                if reports.iter().any(|report| *report != reports[0]) {
                    self.fail("persisted prefix or final publication differs across parties");
                    continue;
                }
                let valid_publication = match &reports[0].last_publication {
                    None => cut == 0,
                    Some(publication) => {
                        publication.batch_id == cut
                            && publication.batch_sha256.len() == 64
                            && publication
                                .batch_sha256
                                .bytes()
                                .all(|byte| byte.is_ascii_hexdigit())
                    }
                };
                if !valid_publication {
                    self.fail("final publication does not acknowledge the admitted cut");
                    continue;
                }
                let mut inner = self.inner.lock().unwrap();
                if inner.status.error.is_some() {
                    continue;
                }
                if inner
                    .deadline
                    .is_some_and(|deadline| Instant::now() >= deadline)
                {
                    drop(inner);
                    self.fail("drain deadline exceeded before final agreement");
                    continue;
                }
                if inner.status.phase != "paused" {
                    metrics::counter!("cpu_cell_drain_transitions", "state" => "paused")
                        .increment(1);
                    tracing::info!(
                        party_id = inner.status.party_id,
                        cut,
                        "CPU cell drain agreed; process remains paused"
                    );
                }
                inner.status.phase = "paused".into();
                inner.deadline = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn freeze_rejects_new_offers_and_waits_for_the_frozen_cut() {
        let drain = Drain::new(0, Duration::from_secs(1), true);
        let published = AtomicBool::new(false);
        assert!(drain.publish_offer(4, 1, || published.store(true, Ordering::SeqCst)));
        drain.request();
        assert!(!drain.publish_offer(5, 1, || panic!("new offer published after freeze")));
        assert!(published.load(Ordering::SeqCst));
        drain.inner.lock().unwrap().status.cut = Some(3);
        assert!(drain.may_finish(3).await);
        assert!(!drain.may_finish(4).await);
    }
    #[test]
    fn pending_receipt_and_publication_prevent_settlement() {
        let drain = Drain::new(0, Duration::from_secs(1), true);
        drain.request();
        drain.receiver_stopped();
        drain.result_started();
        assert!(!drain.settled());
        drain.ingest_stopped();
        assert!(!drain.settled());
        drain.result_sent(Publication {
            batch_id: 1,
            batch_sha256: "ab".repeat(32),
        });
        assert!(drain.settled());
    }
    #[test]
    fn failure_cannot_be_overwritten_by_a_report() {
        let drain = Drain::new(0, Duration::from_secs(1), true);
        drain.request();
        drain.fail("ambiguous SNS publication");
        drain.publish_report(Report {
            watermark: None,
            persisted_count: 0,
            persisted_sha256: "ab".repeat(32),
            last_publication: None,
        });
        assert!(drain.status().report.is_none());
        assert_eq!(drain.status().phase, "failed");
    }
    async fn cohort(timeout: Duration) -> (Vec<Arc<Drain>>, Vec<tokio::task::JoinHandle<()>>) {
        let drains: Vec<_> = (0..3)
            .map(|party| Drain::new(party, timeout, true))
            .collect();
        let mut ports = vec![];
        let mut tasks = vec![];
        for (party, drain) in drains.iter().enumerate() {
            drain.initialize(
                uuid::Uuid::from_u128(party as u128 + 1).to_string(),
                Arc::new(AtomicBool::new(true)),
            );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            ports.push(listener.local_addr().unwrap().port().to_string());
            let routes = drain.routes();
            tasks.push(tokio::spawn(async move {
                axum::serve(listener, routes).await.unwrap();
            }));
        }
        let uuids: HashSet<_> = drains
            .iter()
            .map(|drain| drain.status().process_uuid)
            .collect();
        for (party, drain) in drains.iter().enumerate() {
            let mut config: ServerCoordinationConfig =
                serde_json::from_value(serde_json::json!({})).unwrap();
            config.party_id = party;
            config.healthcheck_ports = ports.clone();
            config.node_hostnames = vec!["127.0.0.1".into(); 3];
            let drain = drain.clone();
            let uuids = uuids.clone();
            tasks.push(tokio::spawn(async move {
                drain.coordinate(config, uuids).await.unwrap();
            }));
        }
        (drains, tasks)
    }
    async fn until(mut predicate: impl FnMut() -> bool) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while !predicate() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
    }
    fn proof(watermark: &str, cut: u64) -> Report {
        Report {
            watermark: Some(iris_mpc_store::normalize_sns_sequence_number(watermark).unwrap()),
            persisted_count: 2,
            persisted_sha256: "cd".repeat(32),
            last_publication: Some(Publication {
                batch_id: cut,
                batch_sha256: "ab".repeat(32),
            }),
        }
    }
    #[tokio::test]
    async fn three_party_drain_excludes_unilateral_prefetch_and_waits_for_publication() {
        let (drains, tasks) = cohort(Duration::from_secs(4)).await;
        for drain in &drains {
            assert!(drain.publish_offer(4, 2, || {}));
            drain.request();
        }
        // A positive offer that only one party made before freeze must be excluded.
        drains[0].inner.lock().unwrap().status.offered_through = 5;
        until(|| drains.iter().all(|drain| drain.status().cut == Some(4))).await;
        for drain in &drains {
            drain.publish_report(proof("100", 4));
        }
        until(|| drains.iter().all(|drain| drain.status().phase == "paused")).await;
        for task in tasks {
            task.abort();
        }
    }
    #[tokio::test]
    async fn unequal_fresh_watermarks_fail_the_whole_cohort() {
        let (drains, tasks) = cohort(Duration::from_secs(4)).await;
        for drain in &drains {
            drain.publish_offer(1, 1, || {});
            drain.request();
        }
        until(|| drains.iter().all(|drain| drain.status().cut == Some(1))).await;
        drains[0].publish_report(proof("99", 1));
        drains[1].publish_report(proof("100", 1));
        drains[2].publish_report(proof("100", 1));
        until(|| drains.iter().all(|drain| drain.failed())).await;
        assert!(drains.iter().all(|drain| drain.status().phase == "failed"));
        for task in tasks {
            task.abort();
        }
    }
    #[tokio::test]
    async fn missing_peer_authorization_times_out_without_cancelling_inflight_work() {
        let (drains, tasks) = cohort(Duration::from_millis(200)).await;
        drains[0].result_started();
        drains[0].request();
        until(|| drains[0].failed()).await;
        assert_eq!(drains[0].inner.lock().unwrap().pending_results, 1);
        drains[0].result_sent(Publication {
            batch_id: 1,
            batch_sha256: "ab".repeat(32),
        });
        assert!(drains[0].status().report.is_none());
        for task in tasks {
            task.abort();
        }
    }
    #[tokio::test]
    async fn matching_maxima_do_not_hide_different_prefixes_or_stale_publication() {
        for stale_publication in [false, true] {
            let (drains, tasks) = cohort(Duration::from_secs(4)).await;
            for drain in &drains {
                drain.publish_offer(4, 1, || {});
                drain.request();
            }
            until(|| drains.iter().all(|drain| drain.status().cut == Some(4))).await;
            let mut mismatched = proof("100", 4);
            if stale_publication {
                // All three can share the same stale publication; equality alone is insufficient.
                mismatched.last_publication.as_mut().unwrap().batch_id = 3;
                for drain in &drains {
                    drain.publish_report(mismatched.clone());
                }
            } else {
                mismatched.persisted_sha256 = "ef".repeat(32);
                drains[0].publish_report(mismatched);
                drains[1].publish_report(proof("100", 4));
                drains[2].publish_report(proof("100", 4));
            }
            until(|| drains.iter().all(|drain| drain.failed())).await;
            assert!(drains.iter().all(|drain| drain.status().phase == "failed"));
            for task in tasks {
                task.abort();
            }
        }
    }
    #[test]
    fn failure_freezes_admission_and_readiness_even_before_request() {
        let drain = Drain::new(0, Duration::from_secs(1), true);
        let ready = Arc::new(AtomicBool::new(true));
        drain.initialize("process".into(), ready.clone());
        drain.fail("publication outcome unknown");
        drain.request();
        drain.mark_ready();
        assert!(!ready.load(Ordering::SeqCst));
        assert!(drain.requested());
        assert_eq!(drain.status().phase, "failed");
        assert!(!drain.publish_offer(1, 1, || panic!("failure admitted work")));
    }
    #[test]
    fn ordinary_enqueue_failure_preserves_restart_recovery_and_blocks_later_drain() {
        let drain = Drain::new(0, Duration::from_secs(1), true);
        drain.result_started();
        drain.result_send_failed();
        assert!(!drain.requested());
        assert_eq!(drain.status().phase, "active");
        drain.request();
        assert!(drain.failed());
    }
    #[test]
    fn late_publication_completion_cannot_recover_an_ambiguous_failure() {
        let drain = Drain::new(0, Duration::from_secs(1), true);
        drain.request();
        drain.result_started();
        drain.fail("SNS publication timed out");
        drain.result_sent(Publication {
            batch_id: 1,
            batch_sha256: "ab".repeat(32),
        });
        drain.publish_report(proof("100", 1));
        drain.mark_ready();
        assert!(drain.failed());
        assert!(drain.status().report.is_none());
        assert_eq!(drain.status().phase, "failed");
    }
    #[test]
    fn eligibility_failure_after_request_blocks_a_late_report() {
        let drain = Drain::new(0, Duration::from_secs(1), true);
        drain.request();
        drain.block("startup publication was unresolved");
        drain.publish_report(proof("100", 1));
        assert_eq!(drain.status().phase, "failed");
        assert_eq!(
            drain.status().error.as_deref(),
            Some("startup publication was unresolved")
        );
        assert!(drain.status().report.is_none());
    }
    #[tokio::test]
    async fn duplicate_cohort_identities_block_drain_before_request() {
        let drain = Drain::new(0, Duration::from_secs(1), true);
        let config: ServerCoordinationConfig =
            serde_json::from_value(serde_json::json!({})).unwrap();
        let duplicate_cohort =
            HashSet::from(["same-process".to_owned(), "another-process".to_owned()]);
        let task = tokio::spawn(drain.clone().coordinate(config, duplicate_cohort));
        until(|| drain.status().blocked_reason.is_some()).await;
        assert!(!drain.requested());
        drain.request();
        assert!(drain.failed());
        assert_eq!(drain.status().phase, "failed");
        assert!(drain.status().report.is_none());
        task.abort();
    }
}
