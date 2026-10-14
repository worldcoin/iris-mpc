use super::*;
use ampc_anon_stats::start_coordination_server;
use ampc_server_utils::{
    wait_for_others_ready, wait_for_startup_barriers, ServerCoordinationConfig,
};
use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use tokio::{
    sync::{Mutex, Notify},
    time::{sleep, timeout},
};

fn configs() -> [ServerCoordinationConfig; 3] {
    let listeners: Vec<_> = (0..3)
        .map(|_| std::net::TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    let ports: Vec<_> = listeners
        .iter()
        .map(|l| l.local_addr().unwrap().port().to_string())
        .collect();
    std::array::from_fn(|party| {
        let mut config: ServerCoordinationConfig = serde_json::from_str("{}").unwrap();
        config.party_id = party;
        config.node_hostnames = vec!["127.0.0.1".to_string(); 3];
        config.healthcheck_ports = ports.clone();
        config.http_query_retry_delay_ms = 5;
        config.http_query_timeout_ms = 30;
        config.startup_sync_timeout_secs = 2;
        config
    })
}

fn start(
    config: &ServerCoordinationConfig,
    tasks: &mut TaskMonitor,
) -> (CoordinationHandles, Arc<Mutex<HashSet<String>>>, String) {
    let peers = Arc::new(Mutex::new(HashSet::new()));
    let uuid = format!("party-{}", config.party_id);
    (
        start_coordination_server(config, tasks, peers.clone(), uuid.clone()),
        peers,
        uuid,
    )
}

async fn probe(config: &ServerCoordinationConfig, path: &str) -> reqwest::Response {
    let url = format!(
        "http://127.0.0.1:{}/{}",
        config.healthcheck_ports[config.party_id], path
    );
    timeout(Duration::from_secs(2), async {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(100))
            .build()
            .unwrap();
        loop {
            if let Ok(response) = client.get(&url).send().await {
                return response;
            }
            sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap()
}

async fn assert_stopping(config: &ServerCoordinationConfig) {
    assert_eq!(
        probe(config, "ready").await.status(),
        reqwest::StatusCode::SERVICE_UNAVAILABLE
    );
    let health = probe(config, "health").await;
    assert_eq!(health.status(), reqwest::StatusCode::OK);
    let health: serde_json::Value = health.json().await.unwrap();
    assert_eq!(health["is_ready"], false);
    assert_eq!(health["shutting_down"], true);
}

struct Dropped(Arc<AtomicBool>);
impl Drop for Dropped {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn missing_peer_startup_cancels_and_reports_shutdown() {
    let config = &configs()[0];
    let shutdown = ShutdownHandler::new(1);
    let mut tasks = TaskMonitor::new();
    let (coordination, peers, uuid) = start(config, &mut tasks);
    probe(config, "health").await;
    // The original bare barrier ignores the actual shutdown notification.
    shutdown.trigger_manual_shutdown();
    assert!(timeout(
        Duration::from_millis(30),
        wait_for_startup_barriers(config, &peers, &uuid)
    )
    .await
    .is_err());
    let result = timeout(
        Duration::from_millis(100),
        startup(
            &shutdown,
            &coordination,
            wait_for_startup_barriers(config, &peers, &uuid),
        ),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(result.is_none());
    set_ready(&shutdown, &coordination);
    assert_stopping(config).await;
    cleanup(&shutdown, &coordination, &mut tasks, Ok(()))
        .await
        .unwrap();
    assert!(tasks.is_empty());
    assert!(shutdown.get_network_cancellation_token().is_cancelled());
}

#[tokio::test]
async fn active_iteration_drains_publication_then_mark_without_network_cancel() {
    let config = &configs()[0];
    let shutdown = ShutdownHandler::new(1);
    let mut tasks = TaskMonitor::new();
    let (coordination, _, _) = start(config, &mut tasks);
    set_ready(&shutdown, &coordination);
    let published = Notify::new();
    let finish_mark = Notify::new();
    let marked = AtomicBool::new(false);
    let dropped = Arc::new(AtomicBool::new(false));
    let guard = Dropped(dropped.clone());
    let work = async {
        let _guard = guard;
        published.notify_one();
        finish_mark.notified().await;
        assert!(!shutdown.get_network_cancellation_token().is_cancelled());
        marked.store(true, Ordering::SeqCst);
        Ok(())
    };
    let control = async {
        published.notified().await;
        shutdown.trigger_manual_shutdown();
        timeout(Duration::from_secs(1), async {
            while !coordination.shutting_down_flag().load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_stopping(config).await;
        assert!(!dropped.load(Ordering::SeqCst));
        assert!(!marked.load(Ordering::SeqCst));
        finish_mark.notify_one();
    };
    let (result, _) = tokio::join!(
        iteration(&shutdown, &coordination, Duration::from_secs(1), work),
        control
    );
    assert!(marked.load(Ordering::SeqCst));
    cleanup(&shutdown, &coordination, &mut tasks, result)
        .await
        .unwrap();
    assert!(dropped.load(Ordering::SeqCst));
    assert!(tasks.is_empty());
}

#[tokio::test]
async fn first_stalled_iteration_times_out_then_cleans_up() {
    let config = &configs()[0];
    let shutdown = ShutdownHandler::new(1);
    let mut tasks = TaskMonitor::new();
    let (coordination, _, _) = start(config, &mut tasks);
    let began = Notify::new();
    let dropped = Arc::new(AtomicBool::new(false));
    let guard = Dropped(dropped.clone());
    let work = async {
        let _guard = guard;
        began.notify_one();
        std::future::pending::<Result<()>>().await
    };
    let control = async {
        began.notified().await;
        shutdown.trigger_manual_shutdown();
    };
    let (result, _) = timeout(Duration::from_secs(1), async {
        tokio::join!(
            iteration(&shutdown, &coordination, Duration::from_millis(20), work),
            control
        )
    })
    .await
    .unwrap();
    assert!(result
        .as_ref()
        .unwrap_err()
        .to_string()
        .contains("drain timed out"));
    assert!(dropped.load(Ordering::SeqCst));
    assert_stopping(config).await;
    let error = cleanup(&shutdown, &coordination, &mut tasks, result)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("drain timed out"));
    assert!(shutdown.get_network_cancellation_token().is_cancelled());
    assert!(tasks.is_empty());
}

#[tokio::test]
async fn shutdown_wins_simultaneous_startup_and_iteration_readiness() {
    let config = &configs()[0];
    let shutdown = ShutdownHandler::new(1);
    let mut tasks = TaskMonitor::new();
    let (coordination, _, _) = start(config, &mut tasks);
    shutdown.trigger_manual_shutdown();
    let started = AtomicBool::new(false);
    let initialized = startup(&shutdown, &coordination, async {
        started.store(true, Ordering::SeqCst);
        Ok(())
    })
    .await
    .unwrap();
    assert!(initialized.is_none());
    iteration(&shutdown, &coordination, Duration::ZERO, async {
        started.store(true, Ordering::SeqCst);
        Ok(())
    })
    .await
    .unwrap();
    set_ready(&shutdown, &coordination);
    assert!(!started.load(Ordering::SeqCst));
    assert_stopping(config).await;
    cleanup(&shutdown, &coordination, &mut tasks, Ok(()))
        .await
        .unwrap();
}

#[tokio::test]
async fn startup_and_iteration_errors_survive_cleanup() {
    for startup_failure in [true, false] {
        let config = &configs()[0];
        let shutdown = ShutdownHandler::new(1);
        let mut tasks = TaskMonitor::new();
        let (coordination, _, _) = start(config, &mut tasks);
        let dropped = Arc::new(AtomicBool::new(false));
        let guard = Dropped(dropped.clone());
        tasks.spawn(async move {
            let _guard = guard;
            std::future::pending::<Result<()>>().await
        });
        let result = if startup_failure {
            startup(&shutdown, &coordination, async {
                Err::<(), _>(eyre!("startup sentinel"))
            })
            .await
            .map(|_| ())
        } else {
            iteration(&shutdown, &coordination, Duration::from_secs(1), async {
                Err(eyre!("iteration sentinel"))
            })
            .await
        };
        let expected = if startup_failure {
            "startup sentinel"
        } else {
            "iteration sentinel"
        };
        let error = cleanup(&shutdown, &coordination, &mut tasks, result)
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), expected);
        assert!(dropped.load(Ordering::SeqCst));
        assert!(tasks.is_empty());
        assert!(shutdown.get_network_cancellation_token().is_cancelled());
    }
}

#[tokio::test]
async fn failing_drain_and_background_task_errors_are_not_hidden() {
    let config = &configs()[0];
    let shutdown = ShutdownHandler::new(1);
    let mut tasks = TaskMonitor::new();
    let (coordination, _, _) = start(config, &mut tasks);
    let began = Notify::new();
    let work = async {
        began.notify_one();
        shutdown.wait_for_shutdown().await;
        Err(eyre!("publish sentinel"))
    };
    let control = async {
        began.notified().await;
        shutdown.trigger_manual_shutdown();
    };
    let (result, _) = tokio::join!(
        iteration(&shutdown, &coordination, Duration::from_secs(1), work),
        control
    );
    assert!(format!("{:?}", result.as_ref().unwrap_err()).contains("publish sentinel"));
    tasks.spawn(async { Err(eyre!("background sentinel")) });
    tokio::task::yield_now().await;
    let error = cleanup(&shutdown, &coordination, &mut tasks, result)
        .await
        .unwrap_err();
    assert!(format!("{error:?}").contains("publish sentinel"));
    assert!(tasks.is_empty());
}

#[tokio::test]
async fn staggered_three_party_startup_converges() {
    let configs = configs();
    let parties = configs.into_iter().map(|config| async move {
        sleep(Duration::from_millis(config.party_id as u64 * 20)).await;
        let shutdown = ShutdownHandler::new(1);
        let mut tasks = TaskMonitor::new();
        let (coordination, peers, uuid) = start(&config, &mut tasks);
        let result = startup(&shutdown, &coordination, async {
            wait_for_startup_barriers(&config, &peers, &uuid).await?;
            set_ready(&shutdown, &coordination);
            let verified = peers.lock().await.clone();
            wait_for_others_ready(&config, verified).await?;
            assert_eq!(
                probe(&config, "ready").await.status(),
                reqwest::StatusCode::OK
            );
            Ok(())
        })
        .await
        .map(|_| ());
        // Keep every peer available until all parties passed peer readiness.
        (shutdown, coordination, tasks, result)
    });
    let initialized = timeout(Duration::from_secs(5), futures::future::join_all(parties))
        .await
        .unwrap();
    for (shutdown, coordination, mut tasks, result) in initialized {
        cleanup(&shutdown, &coordination, &mut tasks, result)
            .await
            .unwrap();
    }
}

#[tokio::test]
#[ignore = "launched in an isolated process by sigterm_interrupts_startup"]
async fn sigterm_child() {
    assert_eq!(std::env::var("POP4440_SIGNAL_CHILD").as_deref(), Ok("1"));
    let config = &configs()[0];
    let shutdown = ShutdownHandler::new(1);
    shutdown.register_signal_handler().await;
    let mut tasks = TaskMonitor::new();
    let (coordination, peers, uuid) = start(config, &mut tasks);
    probe(config, "health").await;
    println!("SIGTERM_CHILD_READY");
    let result = timeout(
        Duration::from_secs(5),
        startup(
            &shutdown,
            &coordination,
            wait_for_startup_barriers(config, &peers, &uuid),
        ),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(result.is_none());
    assert_stopping(config).await;
    cleanup(&shutdown, &coordination, &mut tasks, Ok(()))
        .await
        .unwrap();
    assert!(tasks.is_empty());
}

#[tokio::test]
async fn sigterm_interrupts_startup() {
    use tokio::io::{AsyncBufReadExt, BufReader};
    let mut child = tokio::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "lifecycle::tests::sigterm_child",
            "--ignored",
            "--nocapture",
        ])
        .env("POP4440_SIGNAL_CHILD", "1")
        .stdout(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    timeout(Duration::from_secs(5), async {
        loop {
            let line = lines
                .next_line()
                .await
                .unwrap()
                .expect("signal child exited before readiness");
            if line.contains("SIGTERM_CHILD_READY") {
                break;
            }
        }
    })
    .await
    .unwrap();
    let status = tokio::process::Command::new("kill")
        .args(["-TERM", &child.id().unwrap().to_string()])
        .status()
        .await
        .unwrap();
    assert!(status.success());
    assert!(timeout(Duration::from_secs(5), child.wait())
        .await
        .unwrap()
        .unwrap()
        .success());
}

#[tokio::test]
async fn cleanup_propagates_background_error_without_primary_failure() {
    let config = &configs()[0];
    let shutdown = ShutdownHandler::new(1);
    let mut tasks = TaskMonitor::new();
    let (coordination, _, _) = start(config, &mut tasks);
    tasks.spawn(async { Err(eyre!("background sentinel")) });
    tokio::task::yield_now().await;
    let error = cleanup(&shutdown, &coordination, &mut tasks, Ok(()))
        .await
        .unwrap_err();
    assert!(format!("{error:?}").contains("background sentinel"));
    assert!(tasks.is_empty());
    assert!(shutdown.get_network_cancellation_token().is_cancelled());
}
