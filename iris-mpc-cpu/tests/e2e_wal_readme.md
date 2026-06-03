# iris-mpc-cpu WAL Integration Tests

Integration tests for the Write-Ahead Log (WAL) pipeline: startup roll-forward via
`hawk_main` and sidecar checkpoint cycles via `sidecar_main`.  Architecture mirrors
`iris-mpc-upgrade-hawk/tests/workflows` but targets a different set of services and
observability surfaces.

---

## Scope

These tests exercise the WAL end-to-end without requiring a service client or real
iris request traffic.  The graph itself is opaque at test time (MPC-encrypted shares in
production, or simply unverified in plaintext test mode), so assertions are made
exclusively against:

- The `hawk_graph_mutations` WAL table (row counts, modification IDs)
- The `genesis_graph_checkpoint` table (new rows, WAL anchor, BLAKE3 hash)
- S3 objects (existence of uploaded checkpoint blobs, hash agreement across parties)

The two services under test:

| Service | Entry point | Role in WAL pipeline |
|---|---|---|
| `hawk_main` | `hawk_main::exec(HawkArgs, CancellationToken)` | On startup: load latest checkpoint from S3, replay WAL mutations since that checkpoint, signal ready |
| `sidecar_main` | `checkpoint_protocol::sidecar_main(cfg, graph, s3, networking, shutdown)` | Daemon: freeze WAL height → 3-party hash consensus → upload checkpoint to S3 → insert checkpoint row in DB |

---

## Termination Conditions

### TC-1 — Ready signal (roll-forward / startup)

Used for tests that exercise `hawk_main` startup and WAL replay.

`hawk_main` signals readiness via the coordination server after all three parties have
loaded their checkpoint, applied the WAL delta, and exchanged ready signals.  The test
calls `wait_for_others_ready(&server_coord_config)` (from `ampc_server_utils`) for each
party, combined with `try_join_all`, inside a `tokio::select!` that also monitors the
`JoinSet` for unexpected early exit.

```rust
let ready_futures = configs.iter().map(|config| {
    async move { wait_for_others_ready(&config.server_coord_config).await }
});
tokio::select! {
    res = timeout(dur, try_join_all(ready_futures)) => { res?? }
    Some(exit) = join_set.join_next() => bail!("unexpected task exit: {:?}", exit)
}
```

Signal: "the graph has been materialized from checkpoint + WAL delta and the service
is accepting work."

**hawk_main always requires the full 3-party MPC network to be up**, even for pure
roll-forward startup.  TC-1 tests therefore need the same loopback network setup as
TC-2 sidecar tests.

### TC-2 — DB checkpoint appearance (sidecar cycle)

Used for tests that exercise the sidecar's full checkpoint cycle.

After the sidecar completes a cycle it inserts a new row into `genesis_graph_checkpoint`
(only after the S3 upload succeeds and peer hashes have been agreed upon).  The test
records a baseline checkpoint count before starting the sidecar, then polls
`GraphPg::get_latest_genesis_graph_checkpoint()` until the count advances past the
baseline.  The S3 object is then verified to exist.

```
baseline = count_checkpoints(db)
start sidecar...
poll until count_checkpoints(db) > baseline  (with timeout)
verify S3 object at checkpoint.s3_key exists
```

Signal: "the sidecar materialised the WAL, reached 3-party hash consensus, and durably
stored the checkpoint."

---

## Module Structure

```
iris-mpc-cpu/tests/
├── e2e_wal.rs                   # Test binary: #[test]+#[serial] functions, run_test! macro
├── e2e_wal_readme.md            # This file
├── workflows/
│   ├── mod.rs                   # run_hawk!, run_sidecar!, stop_and_join! macros
│   ├── wal_100.rs               # Startup: empty WAL → ready (baseline smoke test)
│   ├── wal_101.rs               # Startup: checkpoint at M, WAL M+1..N → roll-forward
│   ├── wal_102.rs               # Sidecar: WAL present → checkpoint uploaded to S3
│   ├── wal_103.rs               # Combined: hawk startup roll-forward + sidecar cycle
│   └── wal_104.rs               # Sidecar pruning modes
└── utils/
    ├── mod.rs                   # Type aliases, port constants, CpuNodeConfig
    ├── runner.rs                # TestRun trait (with run() orchestrator) + CpuTestContext
    ├── cpu_node.rs              # CpuNode, CpuNodes, DbStores, WalAssertions
    ├── wal_builder.rs           # Synthetic WAL mutation factory (AddNode only)
    ├── checkpoint_seeder.rs     # Pre-seed a checkpoint into DB + S3 via GraphMem → GraphV4
    └── wait_conditions.rs       # wait_for_all_ready() [TC-1], wait_for_new_checkpoint() [TC-2]
```

---

## Components

### `e2e_wal.rs` — Test binary

Follows the `e2e_genesis.rs` pattern exactly:

- `TEST_FAILED: LazyLock<AtomicBool>` — once any test fails, subsequent tests bail early
- `run_test!(kind, idx, constructor)` macro — creates a `tokio::Runtime`, instantiates the test
  struct, calls `test.run(&ctx)` inside a `tokio::select!` with `ctrl_c()` for interruption
- Individual `#[test]` + `#[serial]` functions, one per `wal_NNN` scenario

```rust
#[test]
#[serial]
fn test_wal_100() -> eyre::Result<()> {
    run_test!(100, 1, Wal100::new())
}
```

---

### `utils/runner.rs` — `TestRun` trait

Lifecycle hook trait implemented by each `wal_NNN` struct.  The `run()` method
orchestrates all phases in order; implementing types only need to override the phases
they care about.

```
run()              orchestrator — calls all phases, propagates first error
setup()            prepare DB state: seed WAL mutations, seed checkpoint
setup_assert()     verify preconditions (e.g. WAL row count = N before exec)
exec()             REQUIRED — spawn services, wait for termination condition
exec_assert()      REQUIRED — verify post-conditions (checkpoint row, S3, WAL HWM)
teardown()         cancel services, truncate tables, clean S3
teardown_assert()  optional final invariant check
```

`CpuTestContext` carries: `configs: [CpuNodeConfig; 3]`, `env: TestEnvironment`
(local vs docker), `kind: usize` (test number, 100–104).

---

### `utils/mod.rs` — Config types and port constants

`CpuNodeConfig` is a test-local struct (not the production `iris_mpc_common::Config`)
that carries just what each test needs.  Key fields:

- `db_url`, `db_schema` — PostgreSQL connection
- `checkpoint_bucket` — S3 bucket for checkpoint objects
- `party_id` — 0, 1, or 2
- `coordination_port` — port for the `ampc_server_utils` coordination server
- `sidecar: SidecarTestConfig` — overridable sidecar settings

Hardcoded loopback port arrays (must not conflict with each other):

```rust
// hawk_main MPC network
pub const HAWK_ADDRS: [&str; 3] = ["127.0.0.1:16000", "127.0.0.1:16100", "127.0.0.1:16200"];

// sidecar_main MPC network (different ports)
pub const SIDECAR_ADDRS: [&str; 3] = ["127.0.0.1:16010", "127.0.0.1:16110", "127.0.0.1:16210"];
```

`SidecarTestConfig::min_mutations_per_cycle` defaults to **5** — tests seed at least
that many WAL rows before starting the sidecar.

---

### `utils/cpu_node.rs` — `DbStores`, `CpuNode`, `CpuNodes`, `WalAssertions`

**`DbStores`** — per-party database handles for test setup and assertion:

```rust
pub struct DbStores {
    pub graph: GraphPg<PlaintextStore>,
    // No iris store needed: WAL is seeded synthetically
}
```

The checkpoint format (`GraphV4` / bincode) is store-agnostic — the same blob is
readable by `hawk_main` regardless of whether it uses `PlaintextStore` or `Aby3Store`.

**`CpuNodes`** — array of 3, mirrors `MpcNodes` from genesis tests:

```rust
impl CpuNodes {
    pub async fn new(configs: &[CpuNodeConfig; 3]) -> Result<Self>;
    pub async fn apply_assertions(&self, assertions: &[WalAssertions; 3]) -> Result<()>;
    pub async fn assert_checkpoint_count(&self, expected: usize) -> Result<()>;
    pub async fn assert_checkpoint_hashes_agree(&self) -> Result<()>;
    pub async fn cleanup_s3_checkpoints(&self, configs: &[CpuNodeConfig; 3]) -> Result<()>;
    pub async fn truncate_checkpoint_tables(&self) -> Result<()>;
}
```

`truncate_checkpoint_tables()` truncates three tables per party:
`hawk_graph_mutations`, `genesis_graph_checkpoint`, and the persistent state table.

**`WalAssertions`** — builder for per-party post-conditions:

```rust
pub struct WalAssertions {
    pub wal_row_count: Option<usize>,           // rows in hawk_graph_mutations
    pub max_modification_id: Option<i64>,       // WAL high-water mark
    pub checkpoint_count: Option<usize>,        // rows in genesis_graph_checkpoint
    pub latest_checkpoint_mod_id: Option<i64>,  // WAL anchor of newest checkpoint
    pub s3_object_exists: Option<bool>,         // whether newest checkpoint is in S3
}
```

---

### `utils/wal_builder.rs` — `WalMutationBuilder`

Constructs and inserts synthetic `hawk_graph_mutations` rows without requiring a live
MPC request pipeline or real iris data.  Each row is a bincode-serialized
`BothEyes<Vec<GraphMutation<IrisVectorId>>>`.

Both `AddNode` and `AddEdges` mutations are supported — both are required to form a
real graph.  Reset-update, recovery-update, and other modification types are not
supported here as they assume a node already exists.  Keep neighbor lists under 100
entries per `add_edges` call (enforced by an assertion).

```rust
impl WalMutationBuilder {
    pub fn new() -> Self;
    // AddNode: use sequential node_ids (0, 1, 2, ...) starting from 0.
    pub fn add_node(self, mod_id: i64, node_id: u32, height: usize) -> Self;
    // AddEdges: neighbors.len() must be ≤ 100.
    pub fn add_edges(self, mod_id: i64, base: u32, neighbors: Vec<u32>, layer: usize) -> Self;
    pub async fn seed(&self, graph: &GraphPg<PlaintextStore>) -> Result<()>;
    pub async fn seed_all(&self, nodes: &CpuNodes) -> Result<()>;
}
```

Each call appends a `WalEntry` enum variant.  `seed()` converts each entry to a
`GraphMutation { seq_no, ops: [AddNode{…} | AddEdges{…}] }` for **both** eyes,
bincode-serializes the `BothEyes<Vec<GraphMutation>>`, and upserts it at the
given `modification_id`.

---

### `utils/checkpoint_seeder.rs` — `CheckpointSeeder`

Builds a minimal serialized graph blob and uploads it to S3, establishing a base
checkpoint so that WAL roll-forward tests have a realistic `(checkpoint, delta)` state.

The checkpoint format is `bincode([GraphMem<IrisVectorId>; 2])` — a **pair** of empty
graphs (left + right eye), matching exactly what `sidecar_main` (`terminal.rs`) writes
and `RebuildFromCheckpoint` (`materializer.rs`) reads.  (An earlier draft used a single
`GraphV4` blob, which was the wrong format — the materializer deserializes to
`BothEyes<GraphMem<VectorId>>`, not a single-eye `GraphV4`.)

Pipeline per party:
1. Create empty `[GraphMem<IrisVectorId>; 2]` (no nodes, zero entry point, both eyes)
2. Serialize with `bincode::serialize`
3. Compute BLAKE3 hash of the bytes
4. Upload to S3 at key `"{party_id}/checkpoints/seed_{last_mod_id}"`
5. Insert `genesis_graph_checkpoint` DB row via `GraphPg::insert_genesis_graph_checkpoint`

---

### `utils/wait_conditions.rs` — TC-1 and TC-2

```rust
/// TC-1: Call wait_for_others_ready for all 3 parties in parallel via try_join_all.
/// Also monitors the JoinSet for unexpected task exit (early error detection).
pub async fn wait_for_all_ready(
    configs: &[CpuNodeConfig; 3],
    join_set: &mut JoinSet<eyre::Result<()>>,
    timeout: Duration,
) -> eyre::Result<()>;

/// TC-2: Poll genesis_graph_checkpoint table until each party exceeds baseline_count,
/// then verify S3 objects. Returns the new checkpoint rows.
pub async fn wait_for_new_checkpoint(
    nodes: &CpuNodes,
    configs: &[CpuNodeConfig; 3],
    baseline_count: usize,
    timeout: Duration,
) -> eyre::Result<[GraphCheckpointRow; 3]>;
```

---

### `workflows/mod.rs` — Macros

Both services are daemon loops; tests start them, wait for a termination condition, then
cancel via a shared `CancellationToken`.

Services can be called inline (no subprocess needed).  No global side-effects prevent
this.  A `JoinSet` is used rather than bare `tokio::spawn` so task exits can be
monitored.

```rust
/// Spawn hawk_main for all 3 parties. Returns a JoinSet.
/// All parties share the provided CancellationToken.
/// Each party gets its own loopback network handle from HAWK_ADDRS.
run_hawk!(configs, shutdown_ct)  ->  JoinSet<eyre::Result<()>>

/// Spawn sidecar_main for all 3 parties. Returns a JoinSet.
/// Each party gets its own loopback network handle from SIDECAR_ADDRS
/// (different ports from HAWK_ADDRS to avoid bind conflicts when both run).
run_sidecar!(configs, shutdown_ct)  ->  JoinSet<eyre::Result<()>>

/// Cancel the token, drain the JoinSet, propagate any errors.
stop_and_join!(shutdown_ct, join_set)
```

---

## Test Scenarios

### `wal_100` — Baseline startup, empty WAL

- **Setup:** clean DB, no checkpoint, no WAL mutations
- **Exec (TC-1):** `run_hawk!` → `wait_for_all_ready()`
- **Assert:** all 3 parties reach ready, WAL row count = 0, checkpoint count = 0
- **Purpose:** smoke test that startup works without any WAL state

### `wal_101` — Roll-forward: checkpoint at M, WAL mutations M+1..N

- **Setup:** `CheckpointSeeder::seed_all(last_mod_id=50)`, then
  `WalMutationBuilder` seeds 50 mutations (mod_ids 51..=100): `add_node` for each new
  node plus `add_edges` to wire up neighbors
- **Exec (TC-1):** `run_hawk!` → `wait_for_all_ready()`
- **Assert:** WAL rows unchanged (roll-forward does not consume them); checkpoint count = 1;
  confirms startup loaded the checkpoint and applied the 50-mutation delta

### `wal_102` — Sidecar: WAL present → checkpoint uploaded

- **Setup:** `WalMutationBuilder` seeds mutations (mod_ids 1..=10): `add_node` for each
  node and `add_edges` for neighbors;
  `min_mutations_per_cycle = 5`
- **Exec (TC-2):** `run_sidecar!` → `wait_for_new_checkpoint(baseline=0)`
- **Assert:** 1 new checkpoint row per party, `latest_checkpoint_mod_id = 10`, S3 object
  present; test materializes its own graph from the seeded WAL, hashes it, and verifies
  all 3 parties' stored BLAKE3 hashes match

### `wal_103` — Combined: startup roll-forward then sidecar cycle

- **Setup:** `CheckpointSeeder::seed_all(last_mod_id=50)`, WAL mutations 51..=100
- **Phase 1 (TC-1):** `run_hawk!` → `wait_for_all_ready()` → `stop_and_join!`
- **Phase 2 (TC-2):** `run_sidecar!` → `wait_for_new_checkpoint(baseline=1)`
- **Assert:** new checkpoint anchored at `mod_id = 100`; BLAKE3 hashes agree across
  parties and match the test's own materialization

### `wal_104` — Sidecar pruning modes

- **Setup:** seed a base checkpoint and 10 WAL mutations
- **Run 1:** `pruning_mode = None` — checkpoint rows accumulate
- **Run 2:** `pruning_mode = OlderNonArchival` — non-archival rows older than latest deleted
- **Run 3:** `pruning_mode = AllOlder` — all but latest deleted
- **Assert per run:** `genesis_graph_checkpoint` row count matches expected post-pruning state;
  pruned S3 objects no longer exist (head_object returns 404)

---

## Infrastructure Assumptions

- **PostgreSQL** running locally (via docker-compose) with a separate schema per party
- **LocalStack** at `http://localhost:4566` for S3 (and potentially SQS/SNS/SecretsManager
  depending on what `server_main` requires — see Q10); Docker variant at
  `http://localstack:4566`
- **No GPU required** — `PlaintextStore` / plaintext graph format used for all DB setup
  and WAL seeding; the checkpoint blob format is `bincode([GraphMem; 2])`, store-agnostic
- **MPC network required for all tests** — both hawk_main and sidecar_main require the
  3-party loopback TCP network; loopback handles are built inline using hardcoded
  localhost ports (`HAWK_ADDRS`, `SIDECAR_ADDRS`)
- **Both services are called inline** — no subprocess spawning; both accept a
  `CancellationToken` and can be driven from a `JoinSet`
- **TC-2 (sidecar) tests are fully implemented** — `run_sidecar!` is live; wal_102, 103, 104
  can run as soon as config loading is unblocked
- **TC-1 (hawk_main) tests need `server_main` wiring** — blocked on Q10/Q11
- Config files per party at `tests/resources/node-config/{local,docker}/`

---

## Open Questions

1. ~~**`hawk_main` callable signature**~~ **Partially resolved:** TC-1 tests call
   `iris_mpc::server::server_main(config: Config) -> Result<()>` (from
   `iris-mpc/src/server/mod.rs`).  The `iris-mpc` crate must be added as a
   `[dev-dependency]` in `iris-mpc-cpu/Cargo.toml`.  The `Config` struct is
   `iris_mpc_common::config::Config` (loaded from TOML).  In `run_hawk!`, wrap
   the call in `tokio::select!` to honour the `CancellationToken`:
   ```rust
   tokio::select! {
       res = server_main(config) => res,
       _ = shutdown.cancelled() => Ok(()),
   }
   ```
   **Still blocked on Q10 and Q11.**

2. ~~**`build_hawk_network_handle` signature**~~ **Resolved:**
   `iris_mpc_cpu::execution::hawk_main::build_hawk_network_handle`.  Full signature:
   ```rust
   pub async fn build_hawk_network_handle(
       args: &HawkArgs,
       shutdown_ct: CancellationToken,
   ) -> Result<Box<dyn NetworkHandle>>
   ```
   For test use, construct `HawkArgs` with only the networking fields populated and set
   `disable_persistence = true` / `hnsw_disable_memory_persistence = true` (all HNSW
   fields can be zero).  `run_sidecar!` is fully implemented using this pattern.

3. ~~**`GraphPg::insert_genesis_graph_checkpoint` signature**~~ **Resolved:** already
   implemented in `CpuNode::seed_party`.

4. ~~**Persistent state table name**~~ **Resolved:** table is `persistent_state`;
   `DbStores::truncate_checkpoint_tables` truncates all three tables.

5. ~~**`ampc_server_utils` as a test dependency**~~ **Resolved:** `ampc-server-utils` is
   already a main dependency in `iris-mpc-cpu/Cargo.toml`.
   `use ampc_server_utils::{wait_for_others_ready, ServerCoordinationConfig}` is now
   imported directly in `wait_conditions.rs`.

6. ~~**Constructing `ServerCoordinationConfig` from test config**~~ **Resolved:**
   `wait_for_all_ready` builds a per-party `ServerCoordinationConfig` inline using
   `configs.iter().map(|c| c.healthcheck_port)` for `healthcheck_ports` and
   `"127.0.0.1"` for `node_hostnames`.  The `coordination_port` field on `CpuNodeConfig`
   is effectively the same as `healthcheck_port` — the coordination server HTTP endpoint
   IS the healthcheck port.

7. ~~**WAL materialization API**~~ **Resolved:**
   `GraphMem::insert_apply_all(&[GraphMutation<V>]) -> Result<()>` (on
   `iris_mpc_cpu::hnsw::graph::layered_graph::GraphMem`).  The checkpoint format is
   `bincode([GraphMem<VectorId>; 2])` — a raw bincode of both-eyes `GraphMem` pair, NOT
   a `GraphV4`.  `DbStores::compute_reference_hash()` now implements the full pipeline:
   fetch WAL rows → apply `insert_apply_all` to `[GraphMem; 2]` → `bincode::serialize` →
   `blake3::hash`.  `CpuNodes::assert_checkpoint_hashes_match_reference` compares the
   result against each party's stored `blake3_hash`.

8. ~~**`BothEyes` type path**~~ **Resolved:** `use iris_mpc_cpu::execution::hawk_main::BothEyes`
   — `pub type BothEyes<T> = [T; 2]`.

9. ~~**`IrisVectorId` type path**~~ **Resolved:** `use iris_mpc_common::IrisVectorId`
   (re-export of `iris_mpc_common::vector_id::VectorId`).  Construct via
   `IrisVectorId::from_serial_id(u32)`.

10. ~~**`server_main` AWS service requirements for TC-1**~~ **Resolved.**
    All required AWS resources are provisioned by `init-localstack.sh` (SQS queues,
    SNS topics, S3 buckets, Secrets Manager secrets, KMS keys) and ECDH keys are
    rotated by `global_setup` before any test runs.  `server_main` can reach all
    resources it needs via LocalStack.  See implementation plan below.

11. ~~**TOML config files and `CpuTestContext::load_configs()`**~~ **Resolved.**
    `utils/configs.rs` provides `hardcoded_configs(env)` which builds all three
    `CpuNodeConfig` values inline.  No TOML files are needed.  For TC-1, a parallel
    `make_hawk_config` function builds the full `iris_mpc_common::config::Config`
    from the same hardcoded values.  See implementation plan below.

---

