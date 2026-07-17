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
| Coordinator | `src/coordinator.rs` | Party-0 FIFO ingress, API, and prepare/commit protocol |
| Batch processor | `src/services/processors/batch.rs` | Converts coordinated envelopes into `BatchQuery` |
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
Client                       Party 0 coordinator                 MPC parties
──────                       ───────────────────                 ───────────
POST request ──────────────► durable FIFO in Postgres
legacy SQS (optional) ─────►
                              Prepare over dedicated mTLS ─────► build BatchQuery
                              ◄──────────── Prepared/digest
                              Commit ──────────────────────────► execute GPU or HNSW job
GET result ◄─────────────── completed/rejected/failed result
legacy SNS (optional) ◄──── party-0 publish only
```

The coordinator API is mounted on party 0's existing coordination HTTP server:

- `POST /coordinator/requests` accepts `message_type`, `payload`, and an optional idempotency `request_id`.
- `GET /coordinator/requests/{request_id}` returns `pending`, `preparing`, `processing`, `completed`, `rejected`, or `failed`.

The request inbox and result are stored in Postgres rather than a local WAL, because server disks may be ephemeral. Inter-party coordinator traffic uses a separate connection from the execution layer while reusing the existing TCP/mTLS networking stack.

During prepare, each party reports requests whose shares could not be fetched, parsed, or decrypted. Party 0 atomically persists the union as `rejected`, and the commit instructs every party to discard those requests before GPU or HNSW execution.
