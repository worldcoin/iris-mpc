#![allow(unused)]
#![recursion_limit = "256"]
use std::sync::Arc;

use ampc_server_utils::wait_for_others_ready;
use eyre::{bail, eyre, Result};
use futures::future::try_join_all;
use iris_mpc::server::server_main;
use iris_mpc_common::{
    config::Config,
    helpers::sync::{Modification, MOD_STATUS_COMPLETED},
    postgres::{AccessMode, PostgresClient},
    IrisVectorId,
};
use iris_mpc_cpu::{
    execution::hawk_main::BothEyes,
    hawkers::plaintext_store::PlaintextStore,
    hnsw::graph::{graph_store::GraphPg, GraphMutation, UpdateEntryPoint},
};
use iris_mpc_utils::{
    aws::{AwsClient, AwsClientConfig},
    constants::AWS_PUBLIC_KEY_BASE_URL,
};
use rand::{thread_rng, Rng};
use serial_test::serial;
use tokio::time::{timeout, Duration};
use tokio::{sync::Notify, task::JoinSet};
use tracing::{info_span, Instrument};
use uuid::Uuid;

const RUST_LOG: &str = "info";

mod utils;
mod workflows;

use crate::utils::{
    genesis_runner, irises,
    modifications::{ModificationInput, ModificationType},
    mpc_node::{db_ops, MpcNodes},
};

#[test]
#[serial]
#[ignore = "requires external setup"]
fn test_hawk_init() -> Result<()> {
    std::thread::Builder::new()
        .name("test_hawk_init()".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || run_test_hawk_init())
        .expect("failed to spawn test thread")
        .join()
        .expect("test thread panicked")
}

fn run_test_hawk_init() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(format!("iris_mpc={RUST_LOG},iris_mpc_cpu={RUST_LOG},iris_mpc_common={RUST_LOG},iris_mpc_upgrade_hawk={RUST_LOG},ampc_server_utils={RUST_LOG},{}={RUST_LOG}", env!("CARGO_CRATE_NAME")))
        .try_init()
        .ok();

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let mut configs = genesis_runner::get_node_configs();

        // the MpcNode init code writes iris codes to the GPU database but we want
        // to initialize the CPU database. Also align the gpu_schema_name_suffix
        // so that gpu_stores uses the same schema (SMPC_hnsw_dev_X) as the
        // server's iris_store, ensuring modifications land where server_main reads them.
        for config in configs.iter_mut() {
            config.database = config.cpu_database.clone();
            config.gpu_schema_name_suffix = config.hnsw_schema_name_suffix.clone();
        }

        let hawk_prf0 = configs[0].hawk_prf_key;
        assert!(
            configs.iter().all(|c| c.hawk_prf_key == hawk_prf0),
            "All hawk_prf_key values in configs must be equal"
        );

        // Allocate free OS ports for healthcheck, MPC service listener, and
        // MPC outbound service ports so there are no conflicts with any
        // already-running services on the host.  We bind port 0 to let the OS
        // pick a free port, record it, then drop the listener so the server
        // can bind the same port when it starts.
        let alloc_ports = |n: usize| -> Vec<String> {
            let listeners: Vec<std::net::TcpListener> = (0..n)
                .map(|_| {
                    std::net::TcpListener::bind("127.0.0.1:0").expect("failed to bind free port")
                })
                .collect();
            listeners
                .iter()
                .map(|l| l.local_addr().unwrap().port().to_string())
                .collect()
            // listeners dropped here, ports released for the servers to claim
        };
        let n = configs.len();
        let healthcheck_ports = alloc_ports(n);
        let service_ports = alloc_ports(n);
        let service_outbound_ports = alloc_ports(n);
        for config in configs.iter_mut() {
            if let Some(ref mut sc) = config.server_coordination {
                sc.healthcheck_ports = healthcheck_ports.clone();
            }
            config.service_ports = service_ports.clone();
            config.service_outbound_ports = service_outbound_ports.clone();
            config.enable_modifications_sync = true;
        }

        let plaintext_irises = genesis_runner::get_irises();
        let shares_rng_seed: u64 = thread_rng().gen();
        let secret_shared_irises =
            irises::share_irises_locally(&plaintext_irises, shares_rng_seed)?;

        // Generate shares for upload (full-size mask shares for S3)
        let upload_shares =
            irises::share_irises_for_upload_locally(&plaintext_irises, shares_rng_seed)?;

        let uniqueness_modifications: Vec<ModificationInput> = (1..=10)
            .map(|id| ModificationInput::new(id, id, ModificationType::Uniqueness, true, true))
            .collect();

        // Create AwsClients and upload shares to S3 for each party
        let mut aws_clients: Vec<AwsClient> = Vec::with_capacity(configs.len());
        for config in configs.iter() {
            let aws_config = AwsClientConfig::new(
                config.environment.clone(),
                AWS_PUBLIC_KEY_BASE_URL.into(),
                config.shares_bucket_name.clone(),
                String::new(), // sns_request_topic_arn - not needed for upload
                0,             // sqs_long_poll_wait_time - not needed for upload
                vec![],        // sqs_response_queue_urls - not needed for upload
            )
            .await;
            println!("aws config: {:#?}", aws_config);
            let mut client = AwsClient::new(aws_config);
            client.set_public_keyset().await?;
            aws_clients.push(client);
        }

        // Generate deterministic UUIDs for each modification
        let mod_uuids: Vec<Uuid> = uniqueness_modifications
            .iter()
            .map(|m| Uuid::from_u128(m.mod_id as u128))
            .collect();

        // Upload shares to S3 for each party
        for (party_idx, aws_client) in aws_clients.iter().enumerate() {
            for (mod_idx, m) in uniqueness_modifications.iter().enumerate() {
                let iris_idx = (m.mod_id - 1) as usize;
                let uuid = &mod_uuids[mod_idx];
                let shares = &upload_shares[iris_idx];
                aws_client.s3_upload_iris_shares(uuid, shares).await?;
            }
        }

        let per_party_modifications: Vec<Vec<Modification>> = (0..configs.len())
            .map(|party_idx| {
                uniqueness_modifications
                    .iter()
                    .enumerate()
                    .map(|(mod_idx, m)| {
                        let uuid = &mod_uuids[mod_idx];
                        Modification {
                            id: m.mod_id,
                            serial_id: Some(m.serial_id),
                            request_type: ModificationType::Uniqueness.to_string(),
                            s3_url: Some(uuid.to_string()),
                            status: MOD_STATUS_COMPLETED.to_string(),
                            persisted: party_idx != 0,
                            result_message_body: Some(format!(r#"{{"node_id":{party_idx}}}"#)),
                        }
                    })
                    .collect()
            })
            .collect();

        let uniqueness_mutations: Vec<(i64, BothEyes<Vec<GraphMutation<IrisVectorId>>>)> =
            uniqueness_modifications
                .iter()
                .map(|m| {
                    let vector_id = IrisVectorId::new(m.serial_id as u32, 0);
                    let mutation = GraphMutation::AddNode {
                        id: vector_id,
                        height: 1,
                        update_ep: UpdateEntryPoint::False,
                    };
                    (m.mod_id, [vec![mutation.clone()], vec![mutation]])
                })
                .collect();

        let nodes = MpcNodes::new(&configs).await;
        let mut join_set = JoinSet::new();
        for (party_idx, node) in nodes.into_iter().enumerate() {
            let party_shares = secret_shared_irises[party_idx].clone();
            let modifications = per_party_modifications[party_idx].clone();
            let mutations = uniqueness_mutations.clone();
            join_set.spawn(async move {
                node.init_tables(&party_shares).await?;
                let mut tx = node.gpu_stores.iris.tx().await?;
                for m in modifications.iter() {
                    db_ops::write_modification(&mut tx, m).await?;
                }
                tx.commit().await?;
                if party_idx != 0 {
                    let mut graph_tx = node.cpu_stores.graph.tx().await?;
                    for (mod_id, mutation) in mutations.iter() {
                        let serialized = bincode::serialize(mutation)?;
                        graph_tx
                            .insert_hawk_graph_mutations(*mod_id, &serialized)
                            .await?;
                    }
                    graph_tx.tx.commit().await?;
                }
                Ok::<_, eyre::Report>(())
            });
        }
        join_runners!(join_set);

        // assert_hawk_mutations_len(0, &configs, line!()).await?;

        // server_main holds !Send pprof state, so each server runs on its own
        // OS thread with its own Tokio runtime to avoid the Send requirement.
        let (exit_tx, mut exit_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<(), eyre::Report>>();
        let notify = Arc::new(Notify::new());
        let _server_threads: Vec<_> = configs
            .iter()
            .cloned()
            .enumerate()
            .map(|(party_idx, config)| {
                let exit_tx = exit_tx.clone();
                let notify = notify.clone();
                std::thread::spawn(move || {
                    let rt = tokio::runtime::Builder::new_multi_thread()
                        .enable_all()
                        .build()
                        .expect("failed to build server runtime");
                    let span = info_span!("mpc_node", idx = party_idx);
                    let result = rt.block_on(async {
                        tokio::select! {
                            res = server_main(config).instrument(span) => res,
                            _ = notify.notified() => Ok(()),
                        }
                    });
                    let _ = exit_tx.send(result);
                })
            })
            .collect();
        drop(exit_tx); // close sender so channel signals when all servers exit

        let ready_futures = configs.iter().map(|config| {
            let server_coord_config = config
                .server_coordination
                .clone()
                .ok_or_else(|| eyre!("server_coordination is required"));
            async move {
                let server_coord_config = server_coord_config?;
                wait_for_others_ready(&server_coord_config).await
            }
        });
        let mut ready_all = async { try_join_all(ready_futures).await.map(|_| ()) };
        tokio::pin!(ready_all);

        tokio::select! {
            ready_res = timeout(Duration::from_secs(60), &mut ready_all) => {
                ready_res??;
            }
            exit_res = exit_rx.recv() => {
                bail!("A server task exited unexpectedly: {:?}", exit_res);
            }
        }

        // stop zombie threads
        notify.notify_waiters();
        assert_hawk_mutations_len(uniqueness_mutations.len(), &configs, line!()).await?;
        Ok(())
    })
}

async fn assert_hawk_mutations_len(
    expected_len: usize,
    configs: &[Config],
    line: u32,
) -> Result<()> {
    for config in configs {
        let url = config
            .get_cpu_db_url()
            .ok_or_else(|| eyre!("cpu_database url is required for party {}", config.party_id))?;
        let schema_name = config.get_cpu_db_schema();
        let client = PostgresClient::new(&url, &schema_name, AccessMode::ReadOnly).await?;
        let graph = GraphPg::<PlaintextStore>::new(&client).await?;
        let mutations = graph.get_hawk_graph_mutations(None).await?;
        assert_eq!(
            mutations.len(),
            expected_len,
            "failure on line {}: party {} has unexpected number of hawk graph mutations: got {}, expected {}",
            line,
            config.party_id,
            mutations.len(),
            expected_len,
        );
    }

    Ok(())
}
