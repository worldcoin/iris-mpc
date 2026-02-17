# Service Client

**Crate:** `iris-mpc-utils/src/client/`
**Binary:** `iris-mpc-bins/bin/service-client/`
**Script:** `scripts/run-service-client.sh`

## Running

```bash
# Default (simple-1.toml, dev-dkr environment)
scripts/run-service-client.sh

# Complex config with iris shares
scripts/run-service-client.sh -i 20250710-1k.ndjson complex-1.toml

# Staging environment
scripts/run-service-client.sh -e dev-stg -i 20250710-1k.ndjson complex-1.toml
```

The script resolves bare filenames by searching:
1. `iris-mpc-bins/data/`
2. `iris-mpc-utils/assets/iris-codes-plaintext/`
3. `iris-mpc-utils/assets/service-client/`
4. `iris-mpc-utils/assets/aws-config/`

Environments: `dev-dkr` (LocalStack at localhost:4566) or `dev-stg` (AWS staging).

## TOML Config Format

**Directory:** `iris-mpc-utils/assets/service-client/`

### Simple Config

```toml
[shares_generator.Compute]
rng_seed = 42

[request_batch.Simple]
batch_size = 3
```

### Complex Config

```toml
[shares_generator.FromFile]
rng_seed = 42
selection_strategy = "All"

[request_batch.Complex]
batches = [
  [
    { label = "40-Uniqueness", payload = { Uniqueness = { iris_pair = [
      { index = 1 }, { index = 2 },
    ] } } },
  ],
  [
    { label = "00-Deletion", payload = { IdentityDeletion = { parent = "40-Uniqueness" } } },
    { label = "10-Reauth", payload = { Reauthorisation = { iris_pair = [...], parent = "41-Uniqueness" } } },
    { label = "20-ResetCheck", payload = { ResetCheck = { iris_pair = [...] } } },
    { label = "30-ResetUpdate", payload = { ResetUpdate = { iris_pair = [...], parent = "42-Uniqueness" } } },
  ],
]
```

- `parent` references a label from any batch. Resolved via `set_child_parent_descriptors_from_labels` at init.
- `iris_pair` indexes refer to positions in the iris shares file.
- Batches are processed sequentially; requests within a batch are processed together.

## Components

### ServiceClient (`src/client/mod.rs`)

Orchestrates the pipeline:

```rust
pub struct ServiceClient {
    request_generator: RequestGenerator,
    shares_uploader: SharesUploader,
    request_enqueuer: RequestEnqueuer,
    response_dequeuer: ResponseDequeuer,
}
```

`init()` sets public keyset and purges response queue.
`exec()` runs the batch loop (see request-lifecycle.md).

### RequestGenerator (`src/client/components/request_generator.rs`)

- `from_options(opts)` — parses TOML config into `RequestBatchSet`
- Calls `set_child_parent_descriptors_from_labels()` to resolve `Label` → `SignupId`
- `next()` — yields one `RequestBatch` at a time

### SharesUploader (`src/client/components/shares_uploader.rs`)

- Iterates requests, calls `get_shares_info()` (returns `None` for IdentityDeletion)
- Uploads iris shares to S3 in parallel
- Sets ALL requests to `SharesUploaded` status (even those with no shares)

### RequestEnqueuer (`src/client/components/request_enqueuer.rs`)

- Filters by `is_enqueueable()`
- Converts `Request` → `RequestPayload` → `SnsMessageInfo`
- For IdentityDeletion: payload = `IdentityDeletionRequest { serial_id }` from parent descriptor
- Publishes to SNS, sets status to `Enqueued`

### ResponseDequeuer (`src/client/components/response_dequeuer.rs`)

- Polls SQS with `sqs_receive_messages(Some(3))` (one per MPC node)
- Parses message kind → `ResponsePayload` variant
- Validates response (checks success/error fields)
- Attempts correlation via `batch.get_idx_of_correlated(&response)`
- If correlated: `set_correlation()` logs + updates state, purges SQS message
- If NOT correlated: **silently ignored**, message left in queue
- Loop continues until `batch.has_enqueued_items()` is false
- **No timeout** — blocks indefinitely if a response never correlates

## Type System

### Request (`src/client/typeset/data/request.rs`)

```rust
pub enum Request {
    IdentityDeletion { info, parent: UniquenessRequestDescriptor },
    Reauthorization  { info, iris_pair, parent, reauth_id: Uuid },
    ResetCheck       { info, iris_pair, reset_id: Uuid },
    ResetUpdate      { info, iris_pair, parent, reset_id: Uuid },
    Uniqueness       { info, iris_pair, signup_id: Uuid },
}
```

Key methods:
- `is_enqueueable()` — `SharesUploaded` + parent resolved (for child types)
- `is_correlation(response)` — type-specific key matching
- `set_correlation(response)` — logs correlation, updates info, checks if fully correlated
- `uniqueness_resolution()` — extracts `(signup_id, serial_id)` from correlated Uniqueness response

### RequestInfo (`src/client/typeset/data/request_info.rs`)

```rust
pub struct RequestInfo {
    uid: Uuid,                                    // Unique request ID
    label: Option<String>,                        // User-assigned label
    label_of_parent: Option<String>,             // Parent's label
    batch_idx: usize,                            // Batch ordinal
    batch_item_idx: usize,                       // Position within batch
    status: RequestStatus,
    correlation_set: [Option<ResponsePayload>; N_PARTIES],  // Responses from 3 nodes
}
```

Display format: `"{label}.{RequestType}"` if label present, else `"{batch_idx}.{item_idx}.{RequestType}"`.

### UniquenessRequestDescriptor (`src/client/typeset/data/descriptors.rs`)

```rust
pub enum UniquenessRequestDescriptor {
    IrisSerialId(IrisSerialId),   // Resolved — ready to enqueue
    Label(String),                 // From TOML config — needs resolution
    SignupId(uuid::Uuid),          // Intermediate — needs cross-batch resolution
}
```

Resolution: `Label` → `SignupId` (at init) → `IrisSerialId` (at batch processing time).

### RequestBatch (`src/client/typeset/data/request_batch.rs`)

```rust
pub struct RequestBatch {
    batch_idx: usize,
    requests: Vec<Request>,
}
```

Key methods:
- `get_idx_of_correlated(response)` — `find()` first request where `is_correlation` matches
- `get_idx_of_child(idx)` — find intra-batch child of a correlated request
- `has_enqueued_items()` — any request with status `Enqueued`
- `is_enqueueable()` — any request where `is_enqueueable()` is true
- `resolve_cross_batch_parents(resolutions)` — `SignupId` → `IrisSerialId`

### RequestBatchSet (`src/client/typeset/data/request_batch.rs`)

Holds all batches. Created from TOML options via `from_options()`.

- `set_child_parent_descriptors_from_labels()` — searches across ALL batches for matching parent labels, converts `Label` → `SignupId`

### ResponsePayload (`src/client/typeset/data/smpc_payloads.rs`)

```rust
pub enum ResponsePayload {
    IdentityDeletion(IdentityDeletionResult),
    Reauthorization(ReAuthResult),
    ResetCheck(ResetCheckResult),
    ResetUpdate(ResetUpdateAckResult),
    Uniqueness(UniquenessResult),
}
```

Parsed from SQS message body based on `kind` attribute. Panics on unknown kind.

## Correlation Logging

When a response correlates with a request:
```
INFO {label}.{RequestType} :: Correlated -> Node-{node_id}
```

When fully correlated (all 3 nodes):
```
INFO {label}.{RequestType} :: Correlated
```

If a response does NOT correlate: **no log output**.
