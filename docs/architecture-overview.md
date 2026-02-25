# Architecture Overview

## Crate Map

```
iris-mpc-cpu/        MPC computation engine
iris-mpc-common/     Shared types used by all crates
iris-mpc-utils/      Service client (testing tool)
iris-mpc/            Server / service layer
iris-mpc-bins/       Binaries, scripts, deployment assets
iris-mpc-store/      Postgres persistence layer
```

## Crate Responsibilities

### iris-mpc-cpu
The core MPC computation engine. Runs the ABY3 protocol over HNSW graphs.

| Module | Path | Role |
|--------|------|------|
| HawkActor | `src/execution/hawk_main.rs` | Central state machine: sessions, graphs, iris stores, batch processing |
| hawk_main submodules | `src/execution/hawk_main/` | Search, insert, matching, reset, scheduling, intra-batch |
| ABY3 store | `src/hawkers/aby3/aby3_store.rs` | Secret-shared distance computation |
| SharedIrises | `src/hawkers/shared_irises.rs` | In-memory versioned iris store |
| HNSW graph | `src/hnsw/` | Graph structure, searcher, layer distribution |
| Network | `src/network/tcp.rs` | TCP connections between MPC parties |
| Session | `src/execution/session.rs` | NetworkSession + PRF = cryptographic context |

### iris-mpc-common
Shared types with no business logic.

| Module | Path | Key Types |
|--------|------|-----------|
| Job | `src/job.rs` | `BatchQuery`, `RequestIndex`, `ServerJobResult`, `BatchMetadata` |
| VectorId | `src/vector_id.rs` | `VectorId`, `SerialId (u32)`, `VersionId (i16)` |
| SMPC request | `src/helpers/smpc_request.rs` | `UniquenessRequest`, `IdentityDeletionRequest`, `ReAuthRequest`, etc. |
| SMPC response | `src/helpers/smpc_response.rs` | `UniquenessResult`, `IdentityDeletionResult`, `ReAuthResult`, etc. |
| Sync | `src/helpers/sync.rs` | `Modification`, `ModificationKey` |
| InMemoryStore | `src/helpers/inmemory_store.rs` | Trait for loading iris data |

### iris-mpc-utils
Testing tool that exercises the full system via AWS (SNS/SQS/S3).

| Module | Path | Role |
|--------|------|------|
| Client entry | `src/client/mod.rs` | `exec()` loop: upload shares, enqueue, dequeue, correlate |
| SharesUploader | `src/client/components/shares_uploader.rs` | Uploads iris shares to S3 |
| RequestEnqueuer | `src/client/components/request_enqueuer.rs` | Publishes requests to SNS |
| ResponseDequeuer | `src/client/components/response_dequeuer.rs` | Polls SQS, correlates responses |
| RequestGenerator | `src/client/components/request_generator.rs` | Generates batches from TOML config |
| SharesGenerator | `src/client/components/shares_generator.rs` | Generates or loads iris shares |
| Type system | `src/client/typeset/data/` | Request, RequestBatch, ResponsePayload, descriptors |
| Options | `src/client/options/` | TOML config parsing and validation |
| TOML configs | `assets/service-client/` | `simple-1.toml`, `complex-1.toml`, etc. |

### iris-mpc
The server process that bridges AWS infrastructure with the MPC engine.

| Module | Path | Role |
|--------|------|------|
| Batch processor | `src/services/processors/batch.rs` | Collects SQS messages into `BatchQuery` |
| Job processor | `src/services/processors/job.rs` | Converts `ServerJobResult` to SNS responses, persists |
| Modifications sync | `src/services/processors/modifications_sync.rs` | Rollforward/rollback of modifications |

### iris-mpc-bins
Binaries and operational scripts.

| Item | Path | Role |
|------|------|------|
| Server binary | `bin/iris-mpc/server.rs` | Main MPC server entry point |
| Service client | `bin/service-client/` | CLI binary wrapping iris-mpc-utils |
| Run script | `scripts/run-service-client.sh` | Shell wrapper with file resolution and AWS profile setup |

## Data Flow (Simplified)

```
Service Client                    Server                         MPC Engine
─────────────                    ──────                         ──────────
S3 upload shares  ──────►
SNS publish request ────►  SQS receive ──► BatchQuery ────►  HawkActor.handle_job()
                           collect batch     build batch        ├─ apply_deletions
                                                                ├─ do_search (Normal)
                                                                ├─ do_search (Mirror)
                                                                ├─ handle_mutations
                                                                └─ HawkResult
                          SNS publish  ◄── ServerJobResult ◄── HawkResult.job_result()
SQS receive response ◄──
correlate with request
```
