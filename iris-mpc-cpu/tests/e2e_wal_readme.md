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
- The `hawk_graph_checkpoints` table (new rows, WAL anchor, BLAKE3 hash)
- S3 objects (existence of uploaded checkpoint blobs, hash agreement across parties)

The two services under test:

| Service | Entry point | Role in WAL pipeline |
|---|---|---|
| `hawk_main` | `hawk_main::exec(HawkArgs)` | On startup: load latest checkpoint from S3, replay WAL mutations since that checkpoint, signal ready |
| `sidecar_main` | `checkpoint_protocol::sidecar_main(cfg, graph, s3, networking, shutdown)` | Daemon: freeze WAL height → 3-party hash consensus → upload checkpoint to S3 → insert checkpoint row in DB |

---

## Termination Conditions

### TC-1 — Ready endpoint (roll-forward / startup)

Used for tests that exercise `hawk_main` startup and WAL replay.

`hawk_main` finishes startup when all three parties have loaded their checkpoint,
applied the WAL delta, and signalled readiness through the coordination server.  The
test polls the coordination server's ready endpoint (one per party) with a timeout.

```
for each party:
    GET http://localhost:{coordination_port}/ready  →  200 OK
```

Signal: "the graph has been materialized from checkpoint + WAL delta and the service
is accepting work."

### TC-2 — S3 checkpoint appearance (sidecar cycle)

Used for tests that exercise the sidecar's full checkpoint cycle.

After the sidecar completes a cycle it inserts a new row into `hawk_graph_checkpoints`
(only after the S3 upload succeeds and peer hashes have been agreed upon).  The test
records a baseline checkpoint count before starting the sidecar, then polls
`GraphPg::recent_checkpoints()` until the count increases.  The S3 object is then
verified to exist.

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
├── e2e_wal.rs                   # Test binary: registers all wal_NNN test cases
├── e2e_wal_readme.md            # This file
├── workflows/
│   ├── mod.rs                   # TestRun trait re-export + run_hawk!, run_sidecar!, stop_and_join! macros
│   ├── wal_100.rs               # Startup: empty WAL → ready (baseline smoke test)
│   ├── wal_101.rs               # Startup: checkpoint at M, WAL M+1..N → roll-forward
│   ├── wal_102.rs               # Sidecar: WAL present → checkpoint uploaded to S3
│   ├── wal_103.rs               # Combined: hawk startup roll-forward + sidecar cycle
│   └── wal_104.rs               # Sidecar pruning modes
└── utils/
    ├── mod.rs                   # Type aliases, sub-module declarations
    ├── runner.rs                # TestRun trait + CpuTestContext
    ├── cpu_node.rs              # CpuNode, CpuNodes, DbStores, WalAssertions
    ├── wal_builder.rs           # Synthetic WAL mutation factory (no real iris data needed)
    ├── checkpoint_seeder.rs     # Pre-seed a checkpoint into DB + S3
    └── wait_conditions.rs       # wait_for_all_ready() [TC-1], wait_for_new_checkpoint() [TC-2]
```

---

## Components

### `utils/runner.rs` — `TestRun` trait

Lifecycle hook trait implemented by each `wal_NNN` struct.  Mirrors the genesis test
pattern.

```
setup()          prepare DB state, seed WAL mutations, seed checkpoint
setup_assert()   verify preconditions (e.g. WAL row count = N before exec)
exec()           spawn services, wait for termination condition
exec_assert()    verify post-conditions (checkpoint row, S3 object, WAL state)
teardown()       cancel services, clean S3, truncate WAL table
```

`CpuTestContext` carries: `party_configs: [CpuNodeConfig; 3]`, `env: TestEnvironment`
(local vs docker), `kind: usize` (test number).

---

### `utils/cpu_node.rs` — `DbStores`, `CpuNode`, `CpuNodes`, `WalAssertions`

**`DbStores`** — per-party database handles for test setup and assertion:

```rust
pub struct DbStores {
    pub graph: GraphPg<PlaintextStore>,
    // No iris store needed: WAL is seeded synthetically
}
```

**`CpuNode`** — one party's handles:

```rust
pub struct CpuNode {
    pub stores: DbStores,
}
```

**`CpuNodes`** — array of 3, mirrors `MpcNodes` from genesis tests:

```rust
impl CpuNodes {
    pub async fn new(configs: &[CpuNodeConfig; 3]) -> Result<Self>;

    // Run WalAssertions against each party
    pub async fn apply_assertions(&self, assertions: &[WalAssertions; 3]) -> Result<()>;

    // Verify each party has expected checkpoint count (DB rows)
    pub async fn assert_checkpoint_count(&self, expected: usize) -> Result<()>;

    // Verify all 3 parties' latest checkpoint BLAKE3 hashes agree
    pub async fn assert_checkpoint_hashes_agree(&self) -> Result<()>;

    // Delete all checkpoint S3 objects + DB rows (teardown)
    pub async fn cleanup_s3_checkpoints(&self, configs: &[CpuNodeConfig; 3]) -> Result<()>;

    // Truncate hawk_graph_mutations for clean state
    pub async fn truncate_wal(&self) -> Result<()>;
}
```

**`WalAssertions`** — builder for per-party post-conditions:

```rust
pub struct WalAssertions {
    pub wal_row_count: Option<usize>,           // rows in hawk_graph_mutations
    pub max_modification_id: Option<i64>,       // WAL high-water mark
    pub checkpoint_count: Option<usize>,        // rows in hawk_graph_checkpoints
    pub latest_checkpoint_mod_id: Option<i64>,  // WAL anchor of newest checkpoint
    pub s3_object_exists: Option<bool>,         // whether newest checkpoint is in S3
}
```

---

### `utils/wal_builder.rs` — `WalMutationBuilder`

Constructs and inserts synthetic `hawk_graph_mutations` rows without requiring a live
MPC request pipeline or real iris data.  Each row is a bincode-serialized
`BothEyes<Vec<GraphMutation<IrisVectorId>>>`.

```rust
pub struct WalMutationBuilder { ... }

impl WalMutationBuilder {
    pub fn new() -> Self;
    pub fn add_node(self, mod_id: i64, node_id: u32, height: usize) -> Self;
    pub fn add_edges(self, mod_id: i64, base: u32, neighbors: Vec<u32>, layer: usize) -> Self;
    // Persist all mutations to one party's graph store
    pub async fn seed(&self, graph: &GraphPg<PlaintextStore>) -> Result<()>;
    // Convenience: seed same mutations to all 3 parties
    pub async fn seed_all(&self, nodes: &CpuNodes) -> Result<()>;
}
```

---

### `utils/checkpoint_seeder.rs` — `CheckpointSeeder`

Builds a minimal serialized graph blob, uploads it to S3, and inserts the corresponding
`hawk_graph_checkpoints` DB row.  This establishes a base checkpoint so that WAL
roll-forward tests have a realistic starting state.

```rust
pub struct CheckpointSeeder {
    pub last_iris_id: i64,
    pub last_modification_id: i64,  // WAL anchor: mutations after this will be rolled forward
}

impl CheckpointSeeder {
    pub async fn seed_party(
        &self,
        graph: &GraphPg<PlaintextStore>,
        s3: &S3Client,
        bucket: &str,
        party_id: usize,
    ) -> Result<GraphCheckpointRow>;

    pub async fn seed_all(
        &self,
        nodes: &CpuNodes,
        configs: &[CpuNodeConfig; 3],
    ) -> Result<[GraphCheckpointRow; 3]>;
}
```

---

### `utils/wait_conditions.rs` — TC-1 and TC-2

```rust
/// TC-1: Poll each party's coordination server ready endpoint until all 3 respond 200.
/// Returns Err if timeout is exceeded.
pub async fn wait_for_all_ready(
    configs: &[CpuNodeConfig; 3],
    timeout: Duration,
) -> Result<()>;

/// TC-2: Poll the DB checkpoint table until the row count exceeds `baseline_count`,
/// then verify the S3 object at each party's latest checkpoint key exists.
/// Returns the new checkpoint rows for each party.
pub async fn wait_for_new_checkpoint(
    nodes: &CpuNodes,
    configs: &[CpuNodeConfig; 3],
    baseline_count: usize,
    timeout: Duration,
) -> Result<[GraphCheckpointRow; 3]>;
```

---

### `workflows/mod.rs` — Macros

Unlike genesis tests (which run to completion), both `hawk_main` and `sidecar_main` are
daemon loops.  Tests start them, wait for a termination condition, then cancel them.

```rust
/// Spawn hawk_main for all 3 parties concurrently.
/// Returns a CancellationToken (shared) and Vec<JoinHandle>.
macro_rules! run_hawk { ($configs:expr) => { ... } }

/// Spawn sidecar_main for all 3 parties concurrently.
macro_rules! run_sidecar { ($configs:expr, $nodes:expr) => { ... } }

/// Cancel the shared token and await all handles.
macro_rules! stop_and_join { ($token:expr, $handles:expr) => { ... } }
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
  `WalMutationBuilder` seeds mutations 51..100
- **Exec (TC-1):** `run_hawk!` → `wait_for_all_ready()`
- **Assert:** WAL rows still present (not consumed/deleted by roll-forward), service ready;
  confirms the startup path loaded the checkpoint and applied the 50-mutation delta

### `wal_102` — Sidecar: WAL present → checkpoint uploaded

- **Setup:** `WalMutationBuilder` seeds mutations 1..30; `min_mutations_per_cycle=10`
- **Exec (TC-2):** `run_sidecar!` → `wait_for_new_checkpoint(baseline=0)`
- **Assert:** 1 new checkpoint row per party, `latest_checkpoint_mod_id = 30`,
  S3 object present, all 3 BLAKE3 hashes agree

### `wal_103` — Combined: startup roll-forward then sidecar cycle

- **Setup:** `CheckpointSeeder::seed_all(last_mod_id=50)`, WAL mutations 51..100
- **Phase 1 (TC-1):** `run_hawk!` → `wait_for_all_ready()` → `stop_and_join!`
  (confirms roll-forward applied)
- **Phase 2 (TC-2):** `run_sidecar!` → `wait_for_new_checkpoint(baseline=1)`
  (sidecar snapshots the rolled-forward state)
- **Assert:** new checkpoint anchored at `mod_id = 100`, hashes agree

### `wal_104` — Sidecar pruning modes

- **Setup:** pre-existing checkpoint, WAL mutations
- **Exec:** multiple sidecar cycles with varying `pruning_mode`
  (`None`, `OlderNonArchival`, `AllOlder`)
- **Assert:** `hawk_graph_checkpoints` row count matches expected post-pruning state per mode

---

## Infrastructure Assumptions

- **PostgreSQL** running locally (via docker-compose) with a separate schema per party
- **LocalStack** at `http://localhost:4566` for S3; Docker variant at
  `http://localstack:4566`
- **No GPU required** — `PlaintextStore` used for all DB setup and WAL seeding
- **MPC network required for sidecar tests** — the 3-party hash consensus step requires
  a TCP network among the three sidecar processes; the loopback network setup already
  used in `tests/e2e.rs` can be reused
- **hawk_main tests (TC-1) may not need a live MPC network** if roll-forward is
  performed locally per-party (open question — see below)
- Config files per party at `tests/resources/node-config/{local,docker}/`

---

## Open Questions

1. **Ready endpoint URL/path:** The coordination server's ready signal is managed via
   `start_coordination_server_with_extra_routes()` / `set_node_ready()` /
   `wait_for_others_ready()`.  What HTTP path does the ready endpoint expose, and on
   which port relative to the party config?  Can we poll it from the test process, or
   do we need to use the coordination client type directly?

2. **hawk_main networking for TC-1 tests:** Does `hawk_main` require the full 3-party
   MPC TCP network to be up before it will signal ready, even if it is only doing
   startup roll-forward?  If so, the TC-1 tests need the same network setup as TC-2.
   If not, each party can be started independently and polled separately.

3. **Calling hawk_main/sidecar_main inline vs subprocess:** The genesis tests call
   `genesis::exec()` as an inline async function.  Can `hawk_main` and `sidecar_main`
   be called the same way from test code, or do they have global side-effects
   (signal handlers, process-level tracing) that require spawning subprocesses?

4. **PlaintextStore vs Aby3Store for checkpoint seeding:** `CheckpointSeeder` needs to
   serialize a graph blob that `hawk_main` (which uses `GraphPg<Aby3Store>`) will accept
   on startup.  Is the serialized format store-agnostic (i.e. is it the raw HNSW
   adjacency structure, independent of the vector store type), or does the checkpoint
   blob embed store-specific vector data that would make a `PlaintextStore`-generated
   blob unreadable by `Aby3Store`-backed startup?

5. **Sidecar networking setup:** `sidecar_main` takes a `networking` handle built by
   `build_hawk_network_handle()`.  How should the test set up 3 loopback network handles
   that can talk to each other?  Is the pattern from `tests/e2e.rs` directly reusable,
   or does the sidecar use a different transport layer?

6. **WAL seeding for TC-1 vs real service writes:** For `wal_101` (roll-forward test),
   the WAL mutations are seeded synthetically.  The `hawk_main` startup path reads these
   mutations and applies them to the in-memory graph, but never validates that the
   mutations form a coherent graph — only that the WAL replay mechanism works.  Is this
   sufficient, or do we need the mutations to form a valid HNSW graph for startup to
   succeed?

7. **Minimum mutation threshold for sidecar:** `sidecar_main` has a
   `min_mutations_per_cycle` guard — it will not produce a checkpoint if fewer than N
   new mutations exist since the last checkpoint.  For TC-2 tests, we need to seed at
   least that many WAL rows.  Should `min_mutations_per_cycle` be overridden to 1 in
   the test config to simplify seeding?

8. **Deterministic BLAKE3 hashes:** `wal_102` and `wal_103` assert that all 3 parties'
   checkpoint hashes agree, but do not assert a specific hash value (since the graph
   blob is not deterministically ordered for synthetically seeded WAL mutations).  If
   deterministic hashing is needed for stronger assertions, the WAL seeding approach
   would need to produce a fully-ordered, reproducible graph.

9. **Test isolation / schema cleanup:** Each test should run against a fresh DB schema.
   What is the right cleanup strategy — drop/recreate schema per test, or truncate all
   relevant tables?  The genesis tests truncate tables in teardown; the same approach
   should work here but needs to cover `hawk_graph_mutations`,
   `hawk_graph_checkpoints`, and the persistent state table.

10. **`e2e_wal.rs` test runner pattern:** The genesis tests use a serial runner with a
    global `tokio::test` that iterates over test cases.  Should `e2e_wal.rs` follow the
    same pattern (one `#[tokio::test]` that runs all `wal_NNN` cases in sequence), or
    use separate `#[tokio::test]` functions per scenario to get independent test
    reporting?
