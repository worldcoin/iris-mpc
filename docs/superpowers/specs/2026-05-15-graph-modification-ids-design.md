# Per-graph modification ids for HNSW mutations

## Goal

Make in-memory application of HNSW graph mutations deterministically verifiable by attaching a per-graph modification id to each atomic mutation and to the graph itself. The id is a monotonic, per-graph `u64`. Apply-time enforcement of strict-increase order is the immediate value; the same id will later be reused as a timestamp embedded in graph edges for stale-edge identification.

## Non-goals

- Refactoring `insert::insert` or `HnswSearcher::insert_prepare_batch`. Per-slot splits (separating `RemoveNode` for reauth, separating compaction into its own group) are deferred until that refactor.
- The eventual stale-edge / per-edge-timestamp logic.
- Forward/backward-compatible deserialization of on-disk artifacts written before this change. Hard cutover.
- Defining `GraphV4` or wiring the upgrade path through it. `GraphV3` is left alone except for one-line touch-ups to the `From` conversions so the codebase compiles.
- Cross-eye id synchronization. Left and right `GraphMem` instances are independent — their ids may diverge.

## Background

Today an in-memory HNSW graph mutation is represented by a list of atom operations grouped into a single `GroupedMutations` newtype, recently renamed to `GraphMutation` (struct) and `MutationOp` (enum). `SingleHawkMutation.plans: BothEyes<Option<ConnectPlan>>` carries one such group per side per request slot. `GraphMem::insert_apply(plan: Vec<MutationOp<V>>)` is infallible and accepts a flat op list; callers (notably `insert.rs:155`) flatten the per-slot groups before applying.

There is no per-graph ordering invariant today. Replay (`test_utils.rs:200-232`, `db_sanity_check.rs:324`) relies on Postgres `modification_id` ordering, which is correct in practice but is not enforced by the graph itself. The eventual stale-edge work needs a per-graph timestamp anyway, so this change is both an immediate safeguard and the foundation for that future work.

Two minor pre-existing issues are noted, with disposition:

- **Write/read shape asymmetry** in `hawk_graph_mutations`: writes serialize `BothEyes<Option<GroupedMutations>>` (`hawk_main.rs:1654`), reads expect `BothEyes<Vec<GraphMutation_old>>` (`graph_store.rs:38`). These bincode encodings differ by the Option tag byte. This change fixes the asymmetry as a side effect — both sides will use `BothEyes<Vec<GraphMutation>>` (where `GraphMutation` is the post-rename struct).
- **`GraphV3` round-trip resets the counter.** S3 checkpoint write/read currently goes through `GraphV3` (`graph.rs:146`, `:276`). Until `GraphV4` is introduced, any checkpoint→restart cycle resets `last_modification_id` to 0. Within a single process lifetime the safeguard works fully; across restarts it is a no-op until `GraphV4` lands. Acceptable because the WAL has not run in production yet.

## Design

### Types

**`GraphMutation`** in `iris-mpc-cpu/src/hnsw/graph/mutation.rs` becomes a full struct with two public fields:

```rust
pub struct GraphMutation<V: Ord> {
    pub id: u64,
    pub ops: Vec<MutationOp<V>>,
}
```

The `id` is the per-graph modification id; `ops` is the existing list of atoms. The `ConnectPlan` / `ConnectPlanV` type aliases continue to refer to `GraphMutation`, unchanged in meaning.

**`SingleHawkMutation.plans`** in `iris-mpc-cpu/src/execution/hawk_main.rs` becomes `BothEyes<Vec<GraphMutation<VectorId>>>`. Empty `Vec` represents "no change on that side." In this changeset each side's `Vec` contains 0 or 1 group; the shape is forward-compatible with the future per-slot-splits refactor.

**`GraphMem`** in `iris-mpc-cpu/src/hnsw/graph/layered_graph.rs` gains a public field:

```rust
pub struct GraphMem<V> {
    pub entry_points: Vec<EntryPoint<V>>,
    pub layers: Vec<Layer<V>>,
    pub last_modification_id: u64,
}
```

The field holds the id of the most recently applied `GraphMutation`. `0` means no mutation has been applied. It is serialized with the rest of `GraphMem` via the existing serde derivation (no `#[serde(default)]`; hard cutover).

`GraphMem` exposes:

```rust
pub fn next_modification_id(&self) -> u64 { self.last_modification_id + 1 }
```

This is a pure peek and never mutates `self`. Callers that need to mint several ids in one batch seed a local running counter from this and increment locally.

### Apply-time semantics

`insert_apply` becomes fallible and takes a single `GraphMutation`:

```rust
pub fn insert_apply(&mut self, mutation: &GraphMutation<V>) -> Result<()>;
```

- Check at entry: `mutation.id > self.last_modification_id`. If not, return `Err` with a descriptive message; `self` is unchanged.
- On success: run the existing two-pass node/edge logic over `mutation.ops` (unchanged internals), then set `self.last_modification_id = mutation.id`.

A convenience for bulk application:

```rust
pub fn insert_apply_all(&mut self, mutations: &[GraphMutation<V>]) -> Result<()>;
```

Iterates and short-circuits on the first violation. The graph state at the failure point reflects the successfully applied prefix.

The flatten-then-apply at `insert.rs:155` (`grouped_mutations` → `Vec<MutationOp>` → `graph.insert_apply`) is removed. `insert.rs` instead iterates the per-slot groups returned by `insert_prepare_batch` and calls `insert_apply` on each, propagating errors.

### Id assignment in `insert::insert`

A small local helper in `insert.rs`:

```rust
struct MutationIdAllocator { next: u64 }
impl MutationIdAllocator {
    fn new(start: u64) -> Self { Self { next: start } }
    fn next(&mut self) -> u64 { let id = self.next; self.next += 1; id }
}
```

`insert::insert` seeds `MutationIdAllocator::new(graph.next_modification_id())` once at the top of the call. Each slot's group is stamped with `alloc.next()` as it is finalized. Across slots the ids are strictly increasing.

`HnswSearcher::insert_prepare_batch` keeps its current return shape (`Vec<Option<GraphMutation>>`) and its current behavior of appending compaction edges into the triggering slot's group. The only change is that the groups it works with carry the `id` field set by `insert.rs` before the call. Per-slot splits — separating compaction into its own group and separating `RemoveNode` for replacements — are explicitly out of scope here and will be addressed alongside the broader `insert.rs` / `insert_prepare_batch` refactor.

`insert::insert` lifts the per-slot `Vec<Option<GraphMutation>>` returned by `insert_prepare_batch` into the `BothEyes<Vec<GraphMutation>>` shape on `SingleHawkMutation`: `None` becomes an empty `Vec`, `Some(g)` becomes `vec![g]`.

### Persistence and replay

Both write and read paths use one canonical shape: `BothEyes<Vec<GraphMutation<VectorId>>>`.

- `HawkMutation::persist` (`hawk_main.rs:1654`): `bincode::serialize(&mutation.plans)` where `mutation.plans: BothEyes<Vec<GraphMutation>>`. Stored in `hawk_graph_mutations.serialized_mutations`.
- `GraphMutationRow::deserialize_mutations` (`graph_store.rs:38`): returns `BothEyes<Vec<GraphMutation<IrisVectorId>>>`. The write/read asymmetry described in Background is resolved.

Hard cutover: existing `hawk_graph_mutations` rows are dropped at deploy. No version byte, no migration code. Existing graph checkpoints will not deserialize (the new field has no `#[serde(default)]`) and are also rebuilt at cutover.

Replay updates in `test_utils.rs:200-232` and `db_sanity_check.rs:324`:

```rust
for row in mutation_rows {
    let both_eyes = row.deserialize_mutations()?;
    for m in &both_eyes[LEFT]  { graph[LEFT].insert_apply(m)?;  }
    for m in &both_eyes[RIGHT] { graph[RIGHT].insert_apply(m)?; }
}
```

Strict-increase violations surface as `Err` from `insert_apply` and halt replay.

### `GraphV3` interop

Compile-only touch-ups, no upgrade-logic design:

- `From<graph_v3::GraphV3> for GraphMem<IrisVectorId>` (`graph.rs:551`) sets `last_modification_id: 0` in the constructed `GraphMem`. Semantically equivalent to "fresh graph after import": the next mutation must have id ≥ 1.
- `From<GraphMem<IrisVectorId>> for graph_v3::GraphV3` (`graph.rs:601`) drops the field (no field on `GraphV3`).

A `GraphMem → GraphV3 → GraphMem` round-trip silently resets the counter; this is a known transition-window limitation, surfaced in Background. The proper resolution (a `GraphV4` carrying `last_modification_id`) is a follow-up.

### Genesis

`iris-mpc-cpu/src/genesis/hawk_handle.rs:231` and `:358` (the two paths that construct `SingleHawkMutation` directly) get the new `BothEyes<Vec<GraphMutation>>` shape and seed a `MutationIdAllocator` from the genesis graph's `next_modification_id()`. No structural change.

## Touchpoints

Files whose call sites or types need updating:

- `iris-mpc-cpu/src/hnsw/graph/mutation.rs` — `GraphMutation` struct gains `id` field and constructor.
- `iris-mpc-cpu/src/hnsw/graph/layered_graph.rs` — `GraphMem` gains `last_modification_id`, `next_modification_id`. `insert_apply` becomes fallible and per-group. `insert_apply_all` added.
- `iris-mpc-cpu/src/hnsw/graph/mod.rs` — re-export updates if needed (likely none beyond what already exists).
- `iris-mpc-cpu/src/hnsw/graph/test_utils.rs` — replay loop; test fixtures that build mutations.
- `iris-mpc-cpu/src/hnsw/graph/graph_store.rs` — `GraphMutationRow::deserialize_mutations` and tests.
- `iris-mpc-cpu/src/hnsw/searcher.rs` — `insert_prepare`, `insert_from_search_results`, and `insert_prepare_batch` thread the allocator / id; the `insert_apply` call sites become per-group + `?`-propagating.
- `iris-mpc-cpu/src/execution/hawk_main.rs` — `SingleHawkMutation.plans` type; `HawkMutation::persist`; existing serde and persistence tests.
- `iris-mpc-cpu/src/execution/hawk_main/insert.rs` — introduce `MutationIdAllocator`; stamp groups; iterate per-group apply; lift `Vec<Option<…>>` into `Vec<…>`.
- `iris-mpc-cpu/src/execution/hawk_main/test_utils.rs` — update mutation-builder helpers.
- `iris-mpc-cpu/src/genesis/hawk_handle.rs` — id stamping at the two `SingleHawkMutation` build sites.
- `iris-mpc-cpu/src/utils/serialization/graph.rs` — `From` conversions to/from `GraphV3` (one-line touch-ups).
- `iris-mpc-bins/bin/iris-mpc-cpu/db_sanity_check.rs` — replay loop matches `test_utils.rs`.

## Testing

Add or update:

- **`mutation.rs` op-level tests**: unaffected — they target `MutationOp` mechanics via `Layer`.
- **`layered_graph.rs` apply tests**:
  - Apply succeeds and advances `last_modification_id` to the applied id.
  - Apply with `id == last_modification_id` returns `Err`; state unchanged.
  - Apply with `id < last_modification_id` returns `Err`; state unchanged.
  - `insert_apply_all` short-circuits on the first violating group; the preceding groups have been applied and the counter reflects the last successful one.
- **`insert.rs` tests** (combined-replace, pure-deletion, none-slot, plus a new batch test):
  - Each per-side `Vec<GraphMutation>` contains 0 or 1 group in this changeset.
  - Across a batch the assigned ids are strictly increasing starting at `graph.next_modification_id()`.
  - Combined-replace still produces a single group containing AddNode + AddEdges + RemoveNode (per current behavior; split is deferred).
- **`graph_store.rs` round-trip test**: update to the new `BothEyes<Vec<GraphMutation>>` shape; assert a multi-group payload round-trips correctly.
- **`hawk_main.rs` serde tests for `SingleHawkMutation`**: update payload shape; cover the empty-Vec and one-group cases.

## Verification

After the implementation lands:

- `cargo build --release --all-features --tests` succeeds.
- `just lint` clean.
- `cargo test --release -- --test-threads=1` and `cargo test --release --features db_dependent -- --test-threads=1` pass.

## Known transition-window limitations (documented, not designed around)

- `GraphMem → GraphV3 → GraphMem` round-trips silently reset `last_modification_id`. Until `GraphV4` exists, S3-stored checkpoints lose the counter on restart. The strict-increase safeguard remains fully effective within a single process lifetime; across restarts it is a no-op. WAL has not run in production yet, so this is acceptable for the immediate rollout.

## Follow-ups (out of scope for this spec)

- `GraphV4` introduction and the proper upgrade path for `GraphMem`.
- Refactor of `insert::insert` and `HnswSearcher::insert_prepare_batch`, after which per-slot splits land: separate `RemoveNode` for replacements, separate compaction occurrences.
- "Compaction at end of batch" rework.
- Per-edge timestamps and stale-edge identification, using the same id concept.
