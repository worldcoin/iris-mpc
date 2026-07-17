use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use ampc_actor_utils::network::mpc::handle::control_channel::ControlChannel;
use ampc_actor_utils::network::mpc::{
    build_network_handle, NetworkHandle, NetworkHandleArgs, NetworkValue,
};
use ampc_server_utils::shutdown_handler::ShutdownHandler;
use ampc_server_utils::TaskMonitor;
use aws_sdk_sqs::Client as SqsClient;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::Utc;
use eyre::{bail, eyre, Result, WrapErr};
use iris_mpc_common::config::Config;
use iris_mpc_common::helpers::aws::{
    SPAN_ID_MESSAGE_ATTRIBUTE_NAME, TRACE_ID_MESSAGE_ATTRIBUTE_NAME,
};
use iris_mpc_common::helpers::sha256::sha256_bytes;
use iris_mpc_common::helpers::smpc_request::{
    SQSMessage, IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RECOVERY_CHECK_MESSAGE_TYPE,
    RECOVERY_UPDATE_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE,
    UNIQUENESS_MESSAGE_TYPE,
};
use iris_mpc_common::helpers::smpc_response::SMPC_MESSAGE_TYPE_ATTRIBUTE;
use iris_mpc_store::{CoordinatorRequest as StoredCoordinatorRequest, Store};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

pub const COORDINATOR_PARTY_ID: usize = 0;
const PARTY_COUNT: usize = 3;
const EMPTY_POLL_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatedRequest {
    pub sequence_number: i64,
    pub request_id: String,
    pub message_body: String,
}

#[derive(Debug)]
pub struct CoordinatorBatch {
    pub batch_id: u64,
    pub requests: Vec<CoordinatedRequest>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProtocolState {
    Idle,
    Preparing(u64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum CoordinatorFrame {
    Prepare {
        batch_id: u64,
        requests: Vec<CoordinatedRequest>,
    },
    Prepared {
        batch_id: u64,
        digest: [u8; 32],
        rejected_request_ids: Vec<String>,
    },
    Rejected {
        batch_id: u64,
        error: String,
    },
    Commit {
        batch_id: u64,
        rejected_request_ids: Vec<String>,
    },
    Abort {
        batch_id: u64,
        error: String,
    },
}

#[derive(Debug)]
struct CurrentBatch {
    batch_id: u64,
    request_ids: Vec<String>,
}

/// Ordered request receiver backed by a dedicated connection from the existing
/// MPC TCP/mTLS stack. The network handle stays alive for the lifetime of the
/// control channel.
pub struct CoordinatorBatchReceiver {
    party_id: usize,
    store: Store,
    shutdown: CancellationToken,
    _networking: Box<dyn NetworkHandle>,
    channel: Box<dyn ControlChannel>,
    state: ProtocolState,
    current: Option<CurrentBatch>,
}

impl CoordinatorBatchReceiver {
    pub async fn connect(
        config: &Config,
        store: Store,
        shutdown: CancellationToken,
    ) -> Result<Self> {
        let coordination = config
            .server_coordination
            .as_ref()
            .ok_or_else(|| eyre!("server coordination config is required"))?;

        if config.party_id >= PARTY_COUNT
            || coordination.node_hostnames.len() != PARTY_COUNT
            || config.coordinator_ports.len() != PARTY_COUNT
            || config.coordinator_outbound_ports.len() != PARTY_COUNT
        {
            bail!(
                "coordinator networking requires exactly {PARTY_COUNT} hostnames, ports, and outbound ports"
            );
        }

        let mut addresses = coordination
            .node_hostnames
            .iter()
            .zip(&config.coordinator_ports)
            .map(|(host, port)| format!("{host}:{port}"))
            .collect::<Vec<_>>();
        addresses[config.party_id] =
            format!("0.0.0.0:{}", config.coordinator_ports[config.party_id]);
        let outbound_addresses = coordination
            .node_hostnames
            .iter()
            .zip(&config.coordinator_outbound_ports)
            .map(|(host, port)| format!("{host}:{port}"))
            .collect::<Vec<_>>();

        let mut networking = build_network_handle(
            NetworkHandleArgs {
                party_index: config.party_id,
                addresses,
                outbound_addresses,
                connection_parallelism: 1,
                request_parallelism: 1,
                sessions_per_request: 1,
                tls: config.tls.clone(),
            },
            shutdown.clone(),
        )
        .await
        .wrap_err("failed to build coordinator network handle")?;
        let channel = networking
            .control_channel()
            .await
            .wrap_err("failed to establish coordinator control channel")?;

        Ok(Self {
            party_id: config.party_id,
            store,
            shutdown,
            _networking: networking,
            channel,
            state: ProtocolState::Idle,
            current: None,
        })
    }

    pub async fn next_batch(&mut self, max_batch_size: usize) -> Result<Option<CoordinatorBatch>> {
        if self.state != ProtocolState::Idle {
            bail!("coordinator protocol requested a batch while not idle");
        }

        if self.party_id == COORDINATOR_PARTY_ID {
            self.next_coordinator_batch(max_batch_size).await
        } else {
            self.next_party_batch().await
        }
    }

    async fn next_coordinator_batch(
        &mut self,
        max_batch_size: usize,
    ) -> Result<Option<CoordinatorBatch>> {
        loop {
            if self.shutdown.is_cancelled() {
                return Ok(None);
            }

            let rows = self
                .store
                .claim_coordinator_requests(max_batch_size)
                .await
                .wrap_err("failed to claim coordinator requests")?;
            if rows.is_empty() {
                tokio::select! {
                    _ = tokio::time::sleep(EMPTY_POLL_INTERVAL) => {}
                    _ = self.shutdown.cancelled() => return Ok(None),
                }
                continue;
            }

            let batch_id = u64::try_from(rows[0].sequence_number)
                .wrap_err("coordinator sequence number does not fit u64")?;
            let requests = rows
                .into_iter()
                .map(|row| CoordinatedRequest {
                    sequence_number: row.sequence_number,
                    request_id: row.request_id,
                    message_body: row.message_body,
                })
                .collect::<Vec<_>>();
            let frame = CoordinatorFrame::Prepare {
                batch_id,
                requests: requests.clone(),
            };

            self.send_both(frame).await?;
            self.begin_batch(batch_id, &requests)?;
            return Ok(Some(CoordinatorBatch { batch_id, requests }));
        }
    }

    async fn next_party_batch(&mut self) -> Result<Option<CoordinatorBatch>> {
        let shutdown = self.shutdown.clone();
        let frame = tokio::select! {
            frame = self.recv_from_coordinator() => frame?,
            _ = shutdown.cancelled() => return Ok(None),
        };

        match frame {
            CoordinatorFrame::Prepare { batch_id, requests } => {
                self.begin_batch(batch_id, &requests)?;
                Ok(Some(CoordinatorBatch { batch_id, requests }))
            }
            other => bail!("expected coordinator Prepare frame, got {other:?}"),
        }
    }

    fn begin_batch(&mut self, batch_id: u64, requests: &[CoordinatedRequest]) -> Result<()> {
        if self.state != ProtocolState::Idle {
            bail!("coordinator protocol cannot begin batch {batch_id} while busy");
        }
        self.state = ProtocolState::Preparing(batch_id);
        self.current = Some(CurrentBatch {
            batch_id,
            request_ids: requests
                .iter()
                .map(|request| request.request_id.clone())
                .collect(),
        });
        Ok(())
    }

    pub async fn prepared(
        &mut self,
        digest: [u8; 32],
        rejected_request_ids: Vec<String>,
    ) -> Result<Vec<String>> {
        let (batch_id, request_ids) = self
            .current
            .as_ref()
            .map(|current| (current.batch_id, current.request_ids.clone()))
            .ok_or_else(|| eyre!("coordinator protocol has no current batch"))?;
        if self.state != ProtocolState::Preparing(batch_id) {
            bail!("coordinator protocol is not preparing the current batch");
        }

        let mut rejected = BTreeSet::new();
        merge_rejected_request_ids(&request_ids, &mut rejected, &rejected_request_ids)?;

        if self.party_id == COORDINATOR_PARTY_ID {
            let next = self.channel.recv_next().await?;
            let prev = self.channel.recv_prev().await?;
            for frame in [decode_frame(next)?, decode_frame(prev)?] {
                match frame {
                    CoordinatorFrame::Prepared {
                        batch_id: peer_batch_id,
                        digest: peer_digest,
                        rejected_request_ids: peer_rejected,
                    } if peer_batch_id == batch_id && peer_digest == digest => {
                        merge_rejected_request_ids(&request_ids, &mut rejected, &peer_rejected)?;
                    }
                    CoordinatorFrame::Rejected {
                        batch_id: peer_batch_id,
                        error,
                    } if peer_batch_id == batch_id => {
                        self.send_both(CoordinatorFrame::Abort {
                            batch_id,
                            error: error.clone(),
                        })
                        .await?;
                        bail!("party rejected coordinator batch {batch_id}: {error}");
                    }
                    other => {
                        let error =
                            format!("invalid acknowledgement for coordinator batch: {other:?}");
                        self.send_both(CoordinatorFrame::Abort {
                            batch_id,
                            error: error.clone(),
                        })
                        .await?;
                        bail!(error);
                    }
                }
            }

            let rejected_request_ids = rejected.iter().cloned().collect::<Vec<_>>();
            let updated = match self
                .store
                .commit_coordinator_batch(
                    &request_ids,
                    &rejected_request_ids,
                    "request was rejected by at least one MPC party",
                )
                .await
            {
                Ok(updated) => updated,
                Err(error) => {
                    let message =
                        format!("failed to persist coordinator batch {batch_id}: {error}");
                    self.send_both(CoordinatorFrame::Abort {
                        batch_id,
                        error: message.clone(),
                    })
                    .await?;
                    return Err(error.wrap_err(message));
                }
            };
            let expected = u64::try_from(request_ids.len() - rejected.len())?;
            if updated != expected {
                let error = format!(
                    "marked {updated} coordinator requests processing, expected {expected}"
                );
                self.send_both(CoordinatorFrame::Abort {
                    batch_id,
                    error: error.clone(),
                })
                .await?;
                bail!(error);
            }
            tracing::info!(
                batch_id,
                requests = request_ids.len(),
                updated,
                "Coordinator committed ordered batch"
            );
            self.send_both(CoordinatorFrame::Commit {
                batch_id,
                rejected_request_ids,
            })
            .await?;
        } else {
            self.send_to_coordinator(CoordinatorFrame::Prepared {
                batch_id,
                digest,
                rejected_request_ids,
            })
            .await?;
            match self.recv_from_coordinator().await? {
                CoordinatorFrame::Commit {
                    batch_id: committed,
                    rejected_request_ids,
                } if committed == batch_id => {
                    rejected.clear();
                    merge_rejected_request_ids(&request_ids, &mut rejected, &rejected_request_ids)?;
                }
                CoordinatorFrame::Abort {
                    batch_id: aborted,
                    error,
                } if aborted == batch_id => {
                    bail!("coordinator aborted batch {batch_id}: {error}");
                }
                other => bail!("expected Commit for batch {batch_id}, got {other:?}"),
            }
        }

        self.state = ProtocolState::Idle;
        self.current = None;
        Ok(rejected.into_iter().collect())
    }

    pub async fn reject(&mut self, error: impl Into<String>) -> Result<()> {
        let error = error.into();
        let batch_id = self
            .current
            .as_ref()
            .map(|current| current.batch_id)
            .ok_or_else(|| eyre!("coordinator protocol has no current batch"))?;

        if self.party_id == COORDINATOR_PARTY_ID {
            self.send_both(CoordinatorFrame::Abort {
                batch_id,
                error: error.clone(),
            })
            .await?;
        } else {
            self.send_to_coordinator(CoordinatorFrame::Rejected {
                batch_id,
                error: error.clone(),
            })
            .await?;
        }
        bail!("rejected coordinator batch {batch_id}: {error}")
    }

    async fn send_both(&mut self, frame: CoordinatorFrame) -> Result<()> {
        let bytes = bincode::serialize(&frame)?;
        self.channel
            .send_next(NetworkValue::Bytes(bytes.clone()))
            .await?;
        self.channel.send_prev(NetworkValue::Bytes(bytes)).await?;
        Ok(())
    }

    async fn send_to_coordinator(&mut self, frame: CoordinatorFrame) -> Result<()> {
        let value = NetworkValue::Bytes(bincode::serialize(&frame)?);
        match self.party_id {
            1 => self.channel.send_prev(value).await?,
            2 => self.channel.send_next(value).await?,
            _ => bail!("invalid non-coordinator party id {}", self.party_id),
        }
        Ok(())
    }

    async fn recv_from_coordinator(&mut self) -> Result<CoordinatorFrame> {
        let value = match self.party_id {
            1 => self.channel.recv_prev().await?,
            2 => self.channel.recv_next().await?,
            _ => bail!("invalid non-coordinator party id {}", self.party_id),
        };
        decode_frame(value)
    }
}

fn decode_frame(value: NetworkValue) -> Result<CoordinatorFrame> {
    let NetworkValue::Bytes(bytes) = value else {
        bail!("coordinator channel received a non-bytes frame");
    };
    Ok(bincode::deserialize(&bytes)?)
}

fn merge_rejected_request_ids(
    request_ids: &[String],
    rejected: &mut BTreeSet<String>,
    additions: &[String],
) -> Result<()> {
    for request_id in additions {
        if !request_ids.contains(request_id) {
            bail!("rejected request {request_id} is not in the current coordinator batch");
        }
        rejected.insert(request_id.clone());
    }
    Ok(())
}

pub fn batch_digest(requests: &[CoordinatedRequest]) -> [u8; 32] {
    let bytes = bincode::serialize(requests).expect("coordinator requests serialize");
    sha256_bytes(bytes)
}

pub fn coordinator_routes(store: Store) -> Router {
    Router::new()
        .route("/coordinator/requests", post(submit_request))
        .route("/coordinator/requests/:request_id", get(get_request))
        .with_state(ApiState { store })
}

#[derive(Clone)]
struct ApiState {
    store: Store,
}

type ApiError = (StatusCode, String);

#[derive(Debug, Deserialize)]
pub struct SubmitRequest {
    #[serde(default)]
    pub request_id: Option<String>,
    pub message_type: String,
    pub payload: Value,
    #[serde(default)]
    pub trace_id: Option<String>,
    #[serde(default)]
    pub span_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RequestStatus {
    pub request_id: String,
    pub sequence_number: i64,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl From<StoredCoordinatorRequest> for RequestStatus {
    fn from(value: StoredCoordinatorRequest) -> Self {
        Self {
            request_id: value.request_id,
            sequence_number: value.sequence_number,
            status: value.status,
            result: value
                .result_body
                .map(|body| serde_json::from_str(&body).unwrap_or(Value::String(body))),
            error: value.error_message,
        }
    }
}

async fn submit_request(
    State(state): State<ApiState>,
    Json(request): Json<SubmitRequest>,
) -> Result<(StatusCode, Json<RequestStatus>), ApiError> {
    if !is_supported_message_type(&request.message_type) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("unsupported message_type {}", request.message_type),
        ));
    }

    let request_id = request
        .request_id
        .clone()
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    if request_id.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "request_id is empty".to_string()));
    }
    let body = build_api_envelope(&request_id, &request)
        .map_err(|error| (StatusCode::BAD_REQUEST, format!("invalid payload: {error}")))?;

    let inserted = state
        .store
        .insert_coordinator_request(&request_id, &body)
        .await
        .map_err(|error| {
            tracing::error!(?error, "Failed to enqueue coordinator API request");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to enqueue request".to_string(),
            )
        })?;

    let record = state
        .store
        .get_coordinator_request(&request_id)
        .await
        .map_err(|error| {
            tracing::error!(?error, "Failed to read enqueued coordinator request");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to read request".to_string(),
            )
        })?
        .ok_or((
            StatusCode::INTERNAL_SERVER_ERROR,
            "enqueued request was not found".to_string(),
        ))?;

    Ok((
        if inserted {
            StatusCode::ACCEPTED
        } else {
            StatusCode::OK
        },
        Json(RequestStatus::from(record)),
    ))
}

async fn get_request(
    State(state): State<ApiState>,
    Path(request_id): Path<String>,
) -> Result<Json<RequestStatus>, ApiError> {
    match state.store.get_coordinator_request(&request_id).await {
        Ok(Some(record)) => Ok(Json(RequestStatus::from(record))),
        Ok(None) => Err((StatusCode::NOT_FOUND, "request not found".to_string())),
        Err(error) => {
            tracing::error!(?error, request_id, "Failed to poll coordinator request");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to poll request".to_string(),
            ))
        }
    }
}

fn build_api_envelope(request_id: &str, request: &SubmitRequest) -> Result<String> {
    let mut attributes = serde_json::Map::new();
    attributes.insert(
        SMPC_MESSAGE_TYPE_ATTRIBUTE.to_string(),
        json!({"Type": "String", "Value": request.message_type.clone()}),
    );
    if let Some(trace_id) = &request.trace_id {
        attributes.insert(
            TRACE_ID_MESSAGE_ATTRIBUTE_NAME.to_string(),
            json!({"Type": "String", "Value": trace_id}),
        );
    }
    if let Some(span_id) = &request.span_id {
        attributes.insert(
            SPAN_ID_MESSAGE_ATTRIBUTE_NAME.to_string(),
            json!({"Type": "String", "Value": span_id}),
        );
    }

    let now = Utc::now();
    Ok(serde_json::to_string(&json!({
        "Type": "Notification",
        "MessageId": request_id,
        "SequenceNumber": now.timestamp_millis().to_string(),
        "TopicArn": "coordinator-api",
        "Message": serde_json::to_string(&request.payload)?,
        "Timestamp": now.to_rfc3339(),
        "UnsubscribeURL": "",
        "MessageAttributes": attributes,
    }))?)
}

fn is_supported_message_type(message_type: &str) -> bool {
    matches!(
        message_type,
        IDENTITY_DELETION_MESSAGE_TYPE
            | UNIQUENESS_MESSAGE_TYPE
            | REAUTH_MESSAGE_TYPE
            | RECOVERY_CHECK_MESSAGE_TYPE
            | RESET_CHECK_MESSAGE_TYPE
            | RESET_UPDATE_MESSAGE_TYPE
            | RECOVERY_UPDATE_MESSAGE_TYPE
    )
}

pub fn spawn_coordinator_sqs_ingest(
    task_monitor: &mut TaskMonitor,
    client: SqsClient,
    config: Config,
    store: Store,
    shutdown_handler: Arc<ShutdownHandler>,
) {
    if config.party_id != COORDINATOR_PARTY_ID || config.requests_queue_url.is_empty() {
        return;
    }

    task_monitor.spawn(async move {
        tracing::info!("Starting coordinator-only SQS ingest");
        while !shutdown_handler.is_shutting_down() {
            if let Err(error) = ingest_sqs_batch(&client, &config, &store).await {
                tracing::warn!(?error, "Coordinator SQS ingest failed; retrying");
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }

        std::future::pending::<Result<()>>().await
    });
}

async fn ingest_sqs_batch(client: &SqsClient, config: &Config, store: &Store) -> Result<()> {
    let output = client
        .receive_message()
        .wait_time_seconds(config.sqs_long_poll_wait_time.min(20) as i32)
        .max_number_of_messages(10)
        .queue_url(&config.requests_queue_url)
        .send()
        .await?;

    for message in output.messages.unwrap_or_default() {
        let body = message
            .body()
            .ok_or_else(|| eyre!("SQS message has no body"))?;
        let request_id = serde_json::from_str::<SQSMessage>(body)
            .map(|message| message.message_id)
            .unwrap_or_else(|_| {
                message
                    .message_id()
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| Uuid::new_v4().to_string())
            });

        store
            .insert_coordinator_request(&request_id, body)
            .await
            .wrap_err("failed to durably enqueue SQS request")?;

        let receipt_handle = message
            .receipt_handle()
            .ok_or_else(|| eyre!("SQS message has no receipt handle"))?;
        client
            .delete_message()
            .queue_url(&config.requests_queue_url)
            .receipt_handle(receipt_handle)
            .send()
            .await?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_envelope_round_trips_as_sqs_message() {
        let request = SubmitRequest {
            request_id: None,
            message_type: UNIQUENESS_MESSAGE_TYPE.to_string(),
            payload: json!({"signup_id": "signup", "s3_key": "shares"}),
            trace_id: Some("trace".to_string()),
            span_id: Some("span".to_string()),
        };
        let body = build_api_envelope("request", &request).unwrap();
        let envelope: SQSMessage = serde_json::from_str(&body).unwrap();

        assert_eq!(envelope.message_id, "request");
        assert_eq!(
            envelope
                .message_attributes
                .get(SMPC_MESSAGE_TYPE_ATTRIBUTE)
                .and_then(|value| value.string_value()),
            Some(UNIQUENESS_MESSAGE_TYPE)
        );
    }

    #[test]
    fn digest_changes_with_request_order() {
        let a = CoordinatedRequest {
            sequence_number: 1,
            request_id: "a".to_string(),
            message_body: "body-a".to_string(),
        };
        let b = CoordinatedRequest {
            sequence_number: 2,
            request_id: "b".to_string(),
            message_body: "body-b".to_string(),
        };

        assert_ne!(batch_digest(&[a.clone(), b.clone()]), batch_digest(&[b, a]));
    }

    #[test]
    fn digest_includes_ledger_sequence() {
        let request = CoordinatedRequest {
            sequence_number: 1,
            request_id: "a".to_string(),
            message_body: "body-a".to_string(),
        };
        let mut moved_request = request.clone();
        moved_request.sequence_number = 2;

        assert_ne!(batch_digest(&[request]), batch_digest(&[moved_request]));
    }

    #[test]
    fn rejected_request_ids_must_belong_to_current_batch() {
        let mut rejected = BTreeSet::new();
        merge_rejected_request_ids(
            &["a".to_string(), "b".to_string()],
            &mut rejected,
            &["b".to_string()],
        )
        .unwrap();
        assert_eq!(rejected, BTreeSet::from(["b".to_string()]));

        assert!(merge_rejected_request_ids(
            &["a".to_string(), "b".to_string()],
            &mut rejected,
            &["c".to_string()],
        )
        .is_err());
    }
}
