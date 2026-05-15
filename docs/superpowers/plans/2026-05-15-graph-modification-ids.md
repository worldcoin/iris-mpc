# Per-graph modification ids for HNSW mutations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-graph monotonic `u64` modification id to each `GraphMutation` group and to `GraphMem` itself, with strict-increase validation in `insert_apply`, as a deterministic-replay safeguard and a foundation for upcoming stale-edge logic.

**Architecture:** `GraphMutation<V>` changes from a newtype `(Vec<MutationOp<V>>)` to a record struct `{ id: u64, ops: Vec<MutationOp<V>> }`. `GraphMem` gains a public `last_modification_id: u64` and a non-mutating `next_modification_id()` peek. `insert_apply` becomes fallible per-group with a strict-increase check; the flattening at `insert.rs:155` is replaced with per-group iteration. `insert.rs` mints sequential ids via a local allocator seeded from `graph.next_modification_id()`. `SingleHawkMutation.plans` becomes `BothEyes<Vec<GraphMutation>>` (each side's Vec carries 0 or 1 group in this changeset; the multi-group capability is forward-compatible for the deferred per-slot-splits refactor). Hard cutover for persisted state.

**Tech Stack:** Rust workspace, bincode/serde for persistence, eyre for errors, sqlx for Postgres, tokio for async, cargo for build/test.

**Spec reference:** `docs/superpowers/specs/2026-05-15-graph-modification-ids-design.md`.

---

## File Structure

Files modified, with their responsibility:

- `iris-mpc-cpu/src/hnsw/graph/mutation.rs` — `GraphMutation` struct shape; `MutationOp` enum unchanged.
- `iris-mpc-cpu/src/hnsw/graph/layered_graph.rs` — `GraphMem` field + peek + per-group fallible `insert_apply` + `insert_apply_all`; in-file tests updated to the new apply signature.
- `iris-mpc-cpu/src/hnsw/graph/graph_store.rs` — `GraphMutationRow::deserialize_mutations` return shape; round-trip test updated.
- `iris-mpc-cpu/src/hnsw/graph/test_utils.rs` — WAL replay loop converts each row's groups via per-group `insert_apply`; in-file fixtures construct `GraphMutation` with explicit ids.
- `iris-mpc-cpu/src/hnsw/graph/mod.rs` — no shape change expected, but re-exports may need confirming.
- `iris-mpc-cpu/src/hnsw/searcher.rs` — `insert_prepare` / `insert_from_search_results` / `insert_prepare_batch` thread the new field through; the `graph.insert_apply(plan.0)` site becomes `graph.insert_apply(&plan)?`.
- `iris-mpc-cpu/src/execution/hawk_main.rs` — `SingleHawkMutation.plans` shape; `HawkMutation::persist` and its callers; serde tests; the no-memory-persistence fast path in `HawkActor::insert`.
- `iris-mpc-cpu/src/execution/hawk_main/insert.rs` — `MutationIdAllocator`; per-slot id stamping; per-group `insert_apply` iteration replacing the flatten at line 155.
- `iris-mpc-cpu/src/execution/hawk_main/test_utils.rs` — test fixtures constructing `GraphMutation` with ids.
- `iris-mpc-cpu/src/genesis/hawk_handle.rs` — calls into `insert::insert` which already handles id-stamping; the genesis paths get the new `BothEyes<Vec<GraphMutation>>` shape passed through.
- `iris-mpc-cpu/src/utils/serialization/graph.rs` — one-line touch-ups to `From<GraphV3>` (set `last_modification_id: 0`) and `From<GraphMem>` (drop the field).
- `iris-mpc-bins/bin/iris-mpc-cpu/db_sanity_check.rs` — replay loop mirrors `test_utils.rs`.

---

## Verification Commands (used throughout)

```bash
# Quick check, parses only:
cargo check --release -p iris-mpc-cpu --all-features

# Compile tests:
cargo build --release --all-features --tests

# Run a single test:
cargo test --release --package iris-mpc-cpu <test_name> -- --test-threads=1

# Run all (long):
cargo test --release -- --test-threads=1

# Full lint:
just lint
```

---

## Task 1: Convert `GraphMutation` to a record struct with `id` and `ops`

Mechanical change. `GraphMutation<V>` stops being a newtype around `Vec<MutationOp<V>>` and becomes `{ id: u64, ops: Vec<MutationOp<V>> }`. All `.0` accesses become `.ops`. All constructor sites get an `id: 0` placeholder; later tasks will mint real ids. No behavior change yet (no validation, no counter).

**Files:**
- Modify: `iris-mpc-cpu/src/hnsw/graph/mutation.rs:3-4` (struct definition)
- Modify: `iris-mpc-cpu/src/hnsw/graph/layered_graph.rs` (test fixtures, no production accesses to `.0` since `insert_apply` takes flat `Vec<MutationOp>` today)
- Modify: `iris-mpc-cpu/src/hnsw/searcher.rs` (lines 1566, 1763, 1784, 1824, 2097, and the `.0` access in `insert_apply(plan.0)`)
- Modify: `iris-mpc-cpu/src/execution/hawk_main.rs` (lines 796, 2475, plus serde test fixtures)
- Modify: `iris-mpc-cpu/src/execution/hawk_main/insert.rs` (lines around 141 — building `GroupedMutations(request_mutations)` becomes building `GraphMutation { id: 0, ops: request_mutations }`)
- Modify: `iris-mpc-cpu/src/execution/hawk_main/test_utils.rs`

- [ ] **Step 1: Write the failing test for the new struct shape**

Add this test to the end of `iris-mpc-cpu/src/hnsw/graph/mutation.rs` (just before the existing `#[cfg(test)] mod tests` opens, or at the end of that module). Add it inside the `mod tests {}` block (the closing brace is at line 259), just above the closing `}`:

```rust
    #[test]
    fn graph_mutation_has_id_and_ops_fields() {
        use crate::hnsw::graph::mutation::{GraphMutation, MutationOp, UpdateEntryPoint};
        let m: GraphMutation<i32> = GraphMutation {
            id: 7,
            ops: vec![MutationOp::AddNode {
                id: 42,
                height: 1,
                update_ep: UpdateEntryPoint::False,
            }],
        };
        assert_eq!(m.id, 7);
        assert_eq!(m.ops.len(), 1);

        // bincode round-trip preserves both fields
        let bytes = bincode::serialize(&m).unwrap();
        let back: GraphMutation<i32> = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back.id, 7);
        assert_eq!(back.ops.len(), 1);
    }
```

- [ ] **Step 2: Run the test, confirm it fails to compile**

Run:
```bash
cargo test --release -p iris-mpc-cpu --lib hnsw::graph::mutation::tests::graph_mutation_has_id_and_ops_fields -- --test-threads=1
```

Expected: compile error — the struct is currently a tuple struct, so `GraphMutation { id: …, ops: … }` won't parse.

- [ ] **Step 3: Change the struct definition**

In `iris-mpc-cpu/src/hnsw/graph/mutation.rs:3-4`, replace:

```rust
#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphMutation<V: Ord>(pub Vec<MutationOp<V>>);
```

with:

```rust
#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphMutation<V: Ord> {
    pub id: u64,
    pub ops: Vec<MutationOp<V>>,
}
```

`#[derive(Default)]` continues to work because `u64` and `Vec<T>` both implement `Default`. The serialized layout changes: bincode for the new struct serializes `id` (8 bytes LE) followed by the length-prefixed `Vec<MutationOp>`, where previously it was just the length-prefixed Vec.

- [ ] **Step 4: Compile-check; fix all `.0` accesses and tuple-style constructors**

Run:
```bash
cargo check --release -p iris-mpc-cpu --all-features --tests
```

Expected: compile errors at the call sites that use `.0` (field access) or `GraphMutation(vec)` (tuple constructor). Fix each as follows:

**`iris-mpc-cpu/src/hnsw/searcher.rs`:**

- Line 1566: `let mutations = vec![Some(GraphMutation(group_mutations))];` becomes `let mutations = vec![Some(GraphMutation { id: 0, ops: group_mutations })];`
- Line 1763: `group.0.push(MutationOp::RemoveEdges { ... });` becomes `group.ops.push(MutationOp::RemoveEdges { ... });`
- Line 1784: same `.0` → `.ops` change.
- Line 1824: `graph.insert_apply(plan.0);` becomes `graph.insert_apply(plan.ops);` (the signature still takes `Vec<MutationOp>` until Task 5; we are only fixing the access here).
- Line 2097: `let mutations = vec![Some(GraphMutation(vec![...]))];` becomes `let mutations = vec![Some(GraphMutation { id: 0, ops: vec![...] })];`. The closing `]))]` becomes `]) })]`.

**`iris-mpc-cpu/src/execution/hawk_main.rs`:**

- Line ~796: `Some(GraphMutation(mutations))` becomes `Some(GraphMutation { id: 0, ops: mutations })`.
- Line ~2475 (inside `create_test_connect_plan`): the constructor builds `GraphMutation(vec![ MutationOp::AddNode { ... }, MutationOp::AddEdges { ... }, ])`. Replace with `GraphMutation { id: 0, ops: vec![ MutationOp::AddNode { ... }, MutationOp::AddEdges { ... }, ] }`.

**`iris-mpc-cpu/src/execution/hawk_main/insert.rs`:**

- Around line 141, `mutations[idx] = Some(GroupedMutations(request_mutations));` — note this currently uses the old name `GroupedMutations` if the rename didn't reach it; after the rename it should already say `GraphMutation(request_mutations)`. Replace with `mutations[idx] = Some(GraphMutation { id: 0, ops: request_mutations });`.
- Around line 150-155, the flatten:

```rust
let all_mutations: Vec<MutationOp<V::VectorRef>> = grouped_mutations
    .iter()
    .filter_map(|opt| opt.as_ref())
    .flat_map(|group| group.0.iter().cloned())
    .collect();
```

becomes:

```rust
let all_mutations: Vec<MutationOp<V::VectorRef>> = grouped_mutations
    .iter()
    .filter_map(|opt| opt.as_ref())
    .flat_map(|group| group.ops.iter().cloned())
    .collect();
```

(Task 5 will replace this entire flatten with per-group iteration.)

**`iris-mpc-cpu/src/execution/hawk_main/test_utils.rs`:**

Find any `GraphMutation(...)` tuple-style constructors and convert to `GraphMutation { id: 0, ops: ... }`. Run the compile-check after these to find any remaining.

**`iris-mpc-cpu/src/hnsw/graph/layered_graph.rs` tests** (lines 1003-1190): these are pure `Vec<MutationOp>` passes to `graph.insert_apply(vec![...])`. No `GraphMutation` constructions in this file, so no change needed here for Task 1. Confirm with the compile-check.

- [ ] **Step 5: Re-run the new unit test; confirm it passes**

```bash
cargo test --release -p iris-mpc-cpu --lib hnsw::graph::mutation::tests::graph_mutation_has_id_and_ops_fields -- --test-threads=1
```

Expected: PASS.

- [ ] **Step 6: Full compile, no warnings about unused fields**

```bash
cargo build --release --all-features --tests
```

Expected: clean build.

- [ ] **Step 7: Commit**

```bash
git add iris-mpc-cpu/src/hnsw/graph/mutation.rs \
        iris-mpc-cpu/src/hnsw/searcher.rs \
        iris-mpc-cpu/src/execution/hawk_main.rs \
        iris-mpc-cpu/src/execution/hawk_main/insert.rs \
        iris-mpc-cpu/src/execution/hawk_main/test_utils.rs
git commit -m "$(cat <<'EOF'
refactor(hnsw): make GraphMutation a record struct with id and ops

GraphMutation<V> changes from a newtype wrapper around Vec<MutationOp<V>>
into a record struct with explicit `id: u64` and `ops: Vec<MutationOp<V>>`
fields. All current construction sites set `id: 0` as a placeholder;
later commits introduce id assignment, the GraphMem counter, and the
strict-increase validation.

No behavioral change in this commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `last_modification_id` field and `next_modification_id()` peek to `GraphMem`

**Files:**
- Modify: `iris-mpc-cpu/src/hnsw/graph/layered_graph.rs:51-67` (struct definition), `:111-117` (`new`), `:102-109` (`Clone` impl)
- Modify: `iris-mpc-cpu/src/utils/serialization/graph.rs:551-559` (`From<GraphV3> for GraphMem`), `:601-610` (`From<GraphMem> for GraphV3`)
- Modify: `iris-mpc-cpu/src/hnsw/graph/layered_graph.rs::migrate` (around line 741) — pass the field through.

- [ ] **Step 1: Write failing test for the peek**

In `iris-mpc-cpu/src/hnsw/graph/layered_graph.rs`, inside the existing `#[cfg(test)] mod tests { ... }` block (around line 774-1191), add this test at the very end of the module (just before the closing `}` at line 1191):

```rust
    #[test]
    fn next_modification_id_is_one_past_last_and_does_not_mutate() {
        use crate::hnsw::GraphMem;
        use iris_mpc_common::IrisVectorId;
        let mut graph = GraphMem::<IrisVectorId>::new();
        assert_eq!(graph.last_modification_id, 0);
        assert_eq!(graph.next_modification_id(), 1);
        assert_eq!(graph.next_modification_id(), 1, "peek must not mutate");
        graph.last_modification_id = 42;
        assert_eq!(graph.next_modification_id(), 43);
        assert_eq!(graph.last_modification_id, 42, "peek must not mutate");
    }
```

- [ ] **Step 2: Run; confirm it fails to compile**

```bash
cargo test --release -p iris-mpc-cpu --lib hnsw::graph::layered_graph::tests::next_modification_id_is_one_past_last_and_does_not_mutate -- --test-threads=1
```

Expected: compile error — the field and method don't exist.

- [ ] **Step 3: Add the field**

In `iris-mpc-cpu/src/hnsw/graph/layered_graph.rs:51-67`, replace:

```rust
#[derive(Default, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(bound = "V: Ref + Display + FromStr")]
pub struct GraphMem<V: Ref + Display + FromStr + Ord> {
    /// Entry points for HNSW search.
    ///
    /// If the graph is built by a searcher in `LinearScan` mode, this list will contain all nodes assigned
    /// to an `insertion_level >= max_graph_layer`. The searcher uses `get_temporary_entry_point`
    /// while no such node exists.
    ///
    /// If the graph is built by a searcher in `Standard` or `Bounded` mode this list
    /// will contain a single entry point at any given time, which corresponds to a node
    /// in the highest layer of the graph.
    pub entry_points: Vec<EntryPoint<V>>,

    /// The layers of the hierarchical graph. The nodes of each layer are a
    /// subset of the nodes of the previous layer, and graph neighborhoods in
    /// each layer represent approximate nearest neighbors within that layer.
    pub layers: Vec<Layer<V>>,
}
```

with:

```rust
#[derive(Default, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(bound = "V: Ref + Display + FromStr")]
pub struct GraphMem<V: Ref + Display + FromStr + Ord> {
    /// Entry points for HNSW search.
    ///
    /// If the graph is built by a searcher in `LinearScan` mode, this list will contain all nodes assigned
    /// to an `insertion_level >= max_graph_layer`. The searcher uses `get_temporary_entry_point`
    /// while no such node exists.
    ///
    /// If the graph is built by a searcher in `Standard` or `Bounded` mode this list
    /// will contain a single entry point at any given time, which corresponds to a node
    /// in the highest layer of the graph.
    pub entry_points: Vec<EntryPoint<V>>,

    /// The layers of the hierarchical graph. The nodes of each layer are a
    /// subset of the nodes of the previous layer, and graph neighborhoods in
    /// each layer represent approximate nearest neighbors within that layer.
    pub layers: Vec<Layer<V>>,

    /// The id of the most recently applied `GraphMutation`. `0` means no
    /// mutation has been applied. Advanced by `insert_apply` on success.
    pub last_modification_id: u64,
}
```

- [ ] **Step 4: Update `GraphMem::new`**

In `iris-mpc-cpu/src/hnsw/graph/layered_graph.rs:112-117`:

```rust
    pub fn new() -> Self {
        GraphMem {
            entry_points: vec![],
            layers: vec![],
        }
    }
```

becomes:

```rust
    pub fn new() -> Self {
        GraphMem {
            entry_points: vec![],
            layers: vec![],
            last_modification_id: 0,
        }
    }
```

- [ ] **Step 5: Update `GraphMem::from_precomputed`**

In `iris-mpc-cpu/src/hnsw/graph/layered_graph.rs:123-134`:

```rust
    pub fn from_precomputed(entry_points: Vec<(V, usize)>, layers: Vec<Layer<V>>) -> Self {
        GraphMem {
            entry_points: entry_points
                .into_iter()
                .map(|ep| EntryPoint {
                    point: ep.0,
                    layer: ep.1,
                })
                .collect::<Vec<_>>(),
            layers,
        }
    }
```

becomes (add the field at the end of the struct literal):

```rust
    pub fn from_precomputed(entry_points: Vec<(V, usize)>, layers: Vec<Layer<V>>) -> Self {
        GraphMem {
            entry_points: entry_points
                .into_iter()
                .map(|ep| EntryPoint {
                    point: ep.0,
                    layer: ep.1,
                })
                .collect::<Vec<_>>(),
            layers,
            last_modification_id: 0,
        }
    }
```

- [ ] **Step 6: Update `Clone` impl**

In `iris-mpc-cpu/src/hnsw/graph/layered_graph.rs:102-109`:

```rust
impl<V: Ref + Display + FromStr + Ord> Clone for GraphMem<V> {
    fn clone(&self) -> Self {
        GraphMem {
            entry_points: self.entry_points.clone(),
            layers: self.layers.clone(),
        }
    }
}
```

becomes:

```rust
impl<V: Ref + Display + FromStr + Ord> Clone for GraphMem<V> {
    fn clone(&self) -> Self {
        GraphMem {
            entry_points: self.entry_points.clone(),
            layers: self.layers.clone(),
            last_modification_id: self.last_modification_id,
        }
    }
}
```

- [ ] **Step 7: Add `next_modification_id` method**

Inside `impl<V: Ref + Display + FromStr + Ord> GraphMem<V>` (around line 111+), add after `pub fn new()`:

```rust
    /// Returns the id that the next applied `GraphMutation` must equal or
    /// exceed. This is a pure peek — it does not modify the graph. Callers
    /// minting several ids in one batch seed a local running counter from
    /// this value and increment locally.
    pub fn next_modification_id(&self) -> u64 {
        self.last_modification_id + 1
    }
```

- [ ] **Step 8: Update `migrate` to carry the field across**

In `iris-mpc-cpu/src/hnsw/graph/layered_graph.rs` around line 741-772 (the `migrate` function), the last expression is:

```rust
    GraphMem::<V> {
        entry_points: new_entry_point,
        layers: new_layers,
    }
```

becomes:

```rust
    GraphMem::<V> {
        entry_points: new_entry_point,
        layers: new_layers,
        last_modification_id: graph.last_modification_id,
    }
```

(Note `graph` is the input parameter — it's moved into `migrate`, so its `last_modification_id` field is accessible before this point. Re-bind if it's been partly moved. Inspect the function body — the `entry_points` iter clone implies `graph` is not fully moved at the bottom; the field access should work. If a move-after-use error occurs, capture `last_modification_id` into a local before the iter/loop bodies.)

- [ ] **Step 9: Update `From<GraphV3> for GraphMem`**

In `iris-mpc-cpu/src/utils/serialization/graph.rs:551-559`:

```rust
impl From<graph_v3::GraphV3> for GraphMem<IrisVectorId> {
    fn from(value: graph_v3::GraphV3) -> Self {
```

Wherever this impl builds the resulting `GraphMem { entry_points: ..., layers: ... }` struct literal, add `last_modification_id: 0,` as a trailing field. (Open the file to find the exact form; the conversion just imports old fields.)

- [ ] **Step 10: Update `From<GraphMem> for GraphV3`**

In `iris-mpc-cpu/src/utils/serialization/graph.rs:601-610`, the impl that destructures or accesses `GraphMem` fields needs to drop `last_modification_id` (GraphV3 has no such field). If the impl uses field-access (`value.entry_points`, etc.), no change is needed — the new field is simply not referenced. If it uses pattern-destructuring, add `last_modification_id: _,` to acknowledge the field. Verify by compile-check.

- [ ] **Step 11: Run the unit test; confirm PASS**

```bash
cargo test --release -p iris-mpc-cpu --lib hnsw::graph::layered_graph::tests::next_modification_id_is_one_past_last_and_does_not_mutate -- --test-threads=1
```

Expected: PASS.

- [ ] **Step 12: Full compile**

```bash
cargo build --release --all-features --tests
```

Expected: clean build.

- [ ] **Step 13: Commit**

```bash
git add iris-mpc-cpu/src/hnsw/graph/layered_graph.rs \
        iris-mpc-cpu/src/utils/serialization/graph.rs
git commit -m "$(cat <<'EOF'
feat(hnsw): add last_modification_id and next_modification_id to GraphMem

`GraphMem` now carries a public `last_modification_id: u64` that tracks
the id of the most recently applied mutation. `next_modification_id()`
peeks at `last + 1` without mutating the graph.

`From<GraphV3> for GraphMem` defaults the new field to 0 (semantically
"fresh graph after import"). Round-trips through `GraphV3` silently drop
the counter; the proper `GraphV4` upgrade path is a follow-up.

Validation in `insert_apply` is added in a subsequent commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Make `insert_apply` fallible and per-group, add `insert_apply_all`, update call sites

This is the validation-enabling task. `insert_apply` changes from `fn(&mut self, plan: Vec<MutationOp<V>>)` to `fn(&mut self, mutation: &GraphMutation<V>) -> Result<()>` with a strict-increase check. `insert_apply_all` is added for slice convenience. All existing callers (production and tests) wrap their `Vec<MutationOp>` into a `GraphMutation { id: <next sequential>, ops }` and propagate `?`.

**Files:**
- Modify: `iris-mpc-cpu/src/hnsw/graph/layered_graph.rs:178-339` (the function itself + tests around line 996-1190)
- Modify: `iris-mpc-cpu/src/hnsw/searcher.rs:1824, 2092`
- Modify: `iris-mpc-cpu/src/execution/hawk_main/insert.rs:150-155`
- Modify: `iris-mpc-cpu/src/hnsw/graph/test_utils.rs:228-229, 381, 405`
- Modify: `iris-mpc-cpu/src/execution/hawk_main/test_utils.rs:116`
- Modify: `iris-mpc-bins/bin/iris-mpc-cpu/db_sanity_check.rs:324`

- [ ] **Step 1: Write failing tests for the new signature and validation**

In `iris-mpc-cpu/src/hnsw/graph/layered_graph.rs`, inside the existing `#[cfg(test)] mod tests` block, add these tests (near the existing `add_edges_*` tests):

```rust
    #[test]
    fn insert_apply_advances_last_modification_id_on_success() {
        use crate::hnsw::graph::mutation::{GraphMutation, MutationOp, UpdateEntryPoint};
        let mut graph = GraphMem::<IrisVectorId>::new();
        let a = IrisVectorId::from_serial_id(1);
        let mutation = GraphMutation::<IrisVectorId> {
            id: 1,
            ops: vec![MutationOp::AddNode {
                id: a,
                height: 1,
                update_ep: UpdateEntryPoint::SetUnique { layer: 0 },
            }],
        };
        graph.insert_apply(&mutation).expect("strict-increase should hold");
        assert_eq!(graph.last_modification_id, 1);
    }

    #[test]
    fn insert_apply_rejects_id_equal_to_last_modification_id() {
        use crate::hnsw::graph::mutation::{GraphMutation, MutationOp, UpdateEntryPoint};
        let mut graph = GraphMem::<IrisVectorId>::new();
        graph.last_modification_id = 5;
        let mutation = GraphMutation::<IrisVectorId> {
            id: 5,
            ops: vec![MutationOp::AddNode {
                id: IrisVectorId::from_serial_id(1),
                height: 1,
                update_ep: UpdateEntryPoint::SetUnique { layer: 0 },
            }],
        };
        let res = graph.insert_apply(&mutation);
        assert!(res.is_err(), "equal id must be rejected");
        assert_eq!(graph.last_modification_id, 5, "state must be unchanged on Err");
        assert_eq!(graph.layers.len(), 0, "no ops should have been applied");
    }

    #[test]
    fn insert_apply_rejects_id_below_last_modification_id() {
        use crate::hnsw::graph::mutation::{GraphMutation, MutationOp, UpdateEntryPoint};
        let mut graph = GraphMem::<IrisVectorId>::new();
        graph.last_modification_id = 10;
        let mutation = GraphMutation::<IrisVectorId> {
            id: 9,
            ops: vec![MutationOp::AddNode {
                id: IrisVectorId::from_serial_id(1),
                height: 1,
                update_ep: UpdateEntryPoint::SetUnique { layer: 0 },
            }],
        };
        let res = graph.insert_apply(&mutation);
        assert!(res.is_err());
        assert_eq!(graph.last_modification_id, 10);
    }

    #[test]
    fn insert_apply_all_short_circuits_on_first_violation() {
        use crate::hnsw::graph::mutation::{GraphMutation, MutationOp, UpdateEntryPoint};
        let mut graph = GraphMem::<IrisVectorId>::new();
        let a = IrisVectorId::from_serial_id(1);
        let b = IrisVectorId::from_serial_id(2);
        let mutations = vec![
            GraphMutation::<IrisVectorId> {
                id: 1,
                ops: vec![MutationOp::AddNode {
                    id: a,
                    height: 1,
                    update_ep: UpdateEntryPoint::SetUnique { layer: 0 },
                }],
            },
            // Equal id — should fail.
            GraphMutation::<IrisVectorId> {
                id: 1,
                ops: vec![MutationOp::AddNode {
                    id: b,
                    height: 1,
                    update_ep: UpdateEntryPoint::False,
                }],
            },
        ];
        let res = graph.insert_apply_all(&mutations);
        assert!(res.is_err(), "second mutation must be rejected");
        assert_eq!(graph.last_modification_id, 1, "first applied; counter at 1");
        // First mutation's AddNode took effect, second did not.
        assert!(graph.layers[0].get_links(&a).is_some());
        assert!(graph.layers[0].get_links(&b).is_none());
    }
```

- [ ] **Step 2: Run; confirm tests fail to compile**

```bash
cargo test --release -p iris-mpc-cpu --lib hnsw::graph::layered_graph::tests::insert_apply -- --test-threads=1
```

Expected: compile errors — `insert_apply` doesn't have the new signature, `insert_apply_all` doesn't exist.

- [ ] **Step 3: Change the `insert_apply` signature and add validation**

In `iris-mpc-cpu/src/hnsw/graph/layered_graph.rs:178`, replace the signature line:

```rust
    pub fn insert_apply(&mut self, plan: Vec<MutationOp<V>>) {
```

with:

```rust
    pub fn insert_apply(&mut self, mutation: &GraphMutation<V>) -> Result<()> {
        if mutation.id <= self.last_modification_id {
            return Err(eyre::eyre!(
                "GraphMem::insert_apply: mutation id {} is not strictly greater than \
                 last_modification_id {}",
                mutation.id,
                self.last_modification_id,
            ));
        }
```

Then inside the function body, the two passes iterate `plan` today (lines 180, 224). Change them to iterate `mutation.ops`:

Find:
```rust
        // Pass 1: apply node-level mutations.
        for mutation in plan.iter() {
```

Replace with:
```rust
        // Pass 1: apply node-level mutations.
        for op in mutation.ops.iter() {
```

Then inside Pass 1, every `match mutation { ... }` becomes `match op { ... }`. Likewise:

Find:
```rust
        // Pass 2: apply edge-level mutations.
        for mutation in plan.into_iter() {
```

Replace with:
```rust
        // Pass 2: apply edge-level mutations.
        for op in mutation.ops.iter().cloned() {
```

(The `cloned()` is because we now take `mutation` by reference, so we can no longer move out of `mutation.ops`.) Then inside Pass 2, `match mutation { ... }` becomes `match op { ... }`.

At the very end of the function, just before the closing `}` of `insert_apply`, advance the counter and return Ok:

```rust
        self.last_modification_id = mutation.id;
        Ok(())
    }
```

Make sure the `use eyre::Result` import is already present at the top of the file (the file uses `eyre::Result` elsewhere — line 17 `use eyre::Result;` already does it). Confirm.

- [ ] **Step 4: Add `insert_apply_all`**

Immediately after the closing `}` of `insert_apply` in `iris-mpc-cpu/src/hnsw/graph/layered_graph.rs`, add:

```rust
    /// Apply a slice of `GraphMutation` groups in order. Short-circuits on
    /// the first strict-increase violation. The graph state at the failure
    /// point reflects the successfully applied prefix.
    pub fn insert_apply_all(&mut self, mutations: &[GraphMutation<V>]) -> Result<()> {
        for m in mutations {
            self.insert_apply(m)?;
        }
        Ok(())
    }
```

- [ ] **Step 5: Fix the in-file tests in `layered_graph.rs`**

The tests at `:996-1190` call `graph.insert_apply(vec![...])` with flat `Vec<MutationOp>`. Each call needs to become `graph.insert_apply(&GraphMutation { id: <unique sequentially-increasing>, ops: vec![...] }).unwrap();`.

Strategy: pick monotonic ids within each test, starting at 1. For tests that make multiple `insert_apply` calls, increment.

For example, in `add_edges_outgoing_writes_only_to_id_list` (around line 997):

```rust
        graph.insert_apply(vec![
            MutationOp::AddNode { ... },
            MutationOp::AddNode { ... },
            MutationOp::AddNode { ... },
        ]);
        graph.insert_apply(vec![MutationOp::AddEdges { ... }]);
```

becomes:

```rust
        graph.insert_apply(&GraphMutation {
            id: 1,
            ops: vec![
                MutationOp::AddNode { ... },
                MutationOp::AddNode { ... },
                MutationOp::AddNode { ... },
            ],
        }).unwrap();
        graph.insert_apply(&GraphMutation {
            id: 2,
            ops: vec![MutationOp::AddEdges { ... }],
        }).unwrap();
```

Apply the same pattern to every other `graph.insert_apply(vec![…])` site in this file. Use locally increasing ids per test. Use the import `use crate::hnsw::graph::mutation::{EdgeType, GraphMutation, MutationOp, UpdateEntryPoint};` (the `use` at line 994 needs `GraphMutation` added).

- [ ] **Step 6: Fix `insert_from_search_results` in `searcher.rs`**

In `iris-mpc-cpu/src/hnsw/searcher.rs:1824`:

```rust
        let plan = self
            .insert_prepare(store, graph, inserted_vector, links_unstructured, update_ep)
            .await?;
        graph.insert_apply(plan.ops);
```

The id from `insert_prepare` is currently 0 (set in Task 1). For this changeset, `insert_from_search_results` is used in tests; we want strict-increase to hold. Assign the id from `graph.next_modification_id()` before constructing the plan. But the plan is constructed by `insert_prepare` and `insert_prepare_batch` deeper — we need to thread it. Simplest: mint the id here and overwrite `plan.id`:

```rust
        let mut plan = self
            .insert_prepare(store, graph, inserted_vector, links_unstructured, update_ep)
            .await?;
        plan.id = graph.next_modification_id();
        graph.insert_apply(&plan)?;
        Ok(())
```

- [ ] **Step 7: Fix the test call site in `searcher.rs:2092`**

In `iris-mpc-cpu/src/hnsw/searcher.rs`, around line 2090 inside `test_insert_prepare_batch`:

```rust
            // Apply the mutations to the graph
            graph_store.insert_apply(mutations);
```

`mutations` here is `Vec<MutationOp>`. Replace with:

```rust
            // Apply the mutations to the graph
            graph_store.insert_apply(&GraphMutation {
                id: (i as u64) + 1,
                ops: mutations,
            }).unwrap();
```

(`i` is the loop variable in the surrounding `for (i, ...)` loop at the test; if `mutations` came from outside a loop, pick `1` for the first call and increment.)

There's a second `insert_apply` (or `insert_prepare_batch`) site nearby for inserting the 6th vector; trace through and ensure ids are unique-and-increasing within that test.

- [ ] **Step 8: Fix `insert::insert` per-group apply at `insert.rs:150-155`**

In `iris-mpc-cpu/src/execution/hawk_main/insert.rs`, replace the flatten:

```rust
    // Flatten all mutations for in-memory graph application.
    let all_mutations: Vec<MutationOp<V::VectorRef>> = grouped_mutations
        .iter()
        .filter_map(|opt| opt.as_ref())
        .flat_map(|group| group.ops.iter().cloned())
        .collect();
    graph.insert_apply(all_mutations);
```

with per-group iteration:

```rust
    // Apply each finalized group to the in-memory graph; strict-increase
    // ordering is enforced by `insert_apply`.
    for group in grouped_mutations.iter().flatten() {
        graph.insert_apply(group)?;
    }
```

(Each `group` is currently `&GraphMutation<V::VectorRef>` with `id: 0` — Task 4 makes ids meaningful. With all-zero ids and a fresh graph, the first apply succeeds (0 > 0 is false, so the strict-increase check FAILS). To keep this commit passing tests, also stamp ids here before applying. Insert before the loop:

```rust
    // Temporary in-place id stamping; Task 4 introduces MutationIdAllocator
    // and moves this to mutation construction time.
    {
        let mut next_id = graph.next_modification_id();
        for slot in grouped_mutations.iter_mut().flatten() {
            slot.id = next_id;
            next_id += 1;
        }
    }
    for group in grouped_mutations.iter().flatten() {
        graph.insert_apply(group)?;
    }
```

For `grouped_mutations.iter_mut().flatten()` to work, `grouped_mutations` must be mutable. If the variable is currently `let grouped_mutations`, change to `let mut grouped_mutations`.)

- [ ] **Step 9: Fix the replay loops**

In `iris-mpc-cpu/src/hnsw/graph/test_utils.rs:228-229`:

```rust
            graph[LEFT].insert_apply(both_eyes[LEFT].clone());
            graph[RIGHT].insert_apply(both_eyes[RIGHT].clone());
```

Replace with:

```rust
            for m in &both_eyes[LEFT]  { graph[LEFT].insert_apply(m)?;  }
            for m in &both_eyes[RIGHT] { graph[RIGHT].insert_apply(m)?; }
```

Note `both_eyes[…]` is currently `Vec<MutationOp<IrisVectorId>>` per the deserialize_mutations signature (`graph_store.rs:38`). After Task 5 changes the shape to `Vec<GraphMutation<IrisVectorId>>`, this loop iterates groups directly. For this task, we are at the point where deserialize_mutations still returns the old shape — so this loop won't compile yet. Hold this change until Task 5 changes the persistence shape, or:

**Alternative for Task 3**: Skip the replay-loop edit in this task. Leave the lines failing-to-compile until Task 5 introduces the matching read shape. To make Task 3 compile in isolation, replace lines 228-229 with a `todo!()` temporarily:

```rust
            // TODO(graph-mod-ids Task 5): once deserialize_mutations returns
            // BothEyes<Vec<GraphMutation<…>>>, replay per-group with insert_apply.
            todo!("replay loop updated in Task 5 when persistence shape changes");
```

This keeps the build clean until Task 5 fixes the shape and the loop. Apply the same `todo!` placeholder at `iris-mpc-bins/bin/iris-mpc-cpu/db_sanity_check.rs:324` (where `row.deserialize_mutations()?` is also called).

(If you prefer not to use `todo!`, keep the existing flat-Vec apply with `mutation.ops` field access; you'll need a temporary helper `fn apply_flat(&mut self, ops: Vec<MutationOp<V>>)` that wraps in a transient `GraphMutation` with `id = self.next_modification_id()`. The `todo!` route is simpler and explicit.)

- [ ] **Step 10: Fix the fixture replay in `test_utils.rs:381, 405`**

In `iris-mpc-cpu/src/hnsw/graph/test_utils.rs`, around line 381:

```rust
        left_graph.insert_apply(vec![ep_mutation]);
```

becomes:

```rust
        left_graph.insert_apply(&GraphMutation { id: 1, ops: vec![ep_mutation] }).unwrap();
```

And around line 405:

```rust
            left_graph.insert_apply(mutations);
```

becomes:

```rust
            left_graph.insert_apply(&GraphMutation { id: (i as u64) + 2, ops: mutations }).unwrap();
```

(The `i` is the loop variable; `+2` shifts past the `id: 1` used above for the entry-point mutation.)

Add the import `use crate::hnsw::graph::mutation::GraphMutation;` if not already present.

- [ ] **Step 11: Fix the fixture in `hawk_main/test_utils.rs:116`**

In `iris-mpc-cpu/src/execution/hawk_main/test_utils.rs`, around line 116:

```rust
                graph.insert_apply(mutations);
```

becomes:

```rust
                graph.insert_apply(&GraphMutation { id: 1, ops: mutations }).unwrap();
```

(Inspect the surrounding loop. If this is called in a loop and `mutations` is freshly built each iteration, mint a per-iteration id: `id: (iter_index as u64) + 1`.)

Add the import if not already present.

- [ ] **Step 12: Run the new tests; confirm PASS**

```bash
cargo test --release -p iris-mpc-cpu --lib hnsw::graph::layered_graph::tests::insert_apply -- --test-threads=1
cargo test --release -p iris-mpc-cpu --lib hnsw::graph::layered_graph::tests::insert_apply_all_short_circuits_on_first_violation -- --test-threads=1
```

Expected: PASS.

- [ ] **Step 13: Run all unit tests in `iris-mpc-cpu` to catch regressions**

```bash
cargo test --release -p iris-mpc-cpu --lib -- --test-threads=1
```

Expected: PASS for everything except tests that depend on the replay loops (those are `todo!`-stubbed; they're feature-gated `db_dependent` and won't run in this command).

- [ ] **Step 14: Commit**

```bash
git add iris-mpc-cpu/src/hnsw/graph/layered_graph.rs \
        iris-mpc-cpu/src/hnsw/searcher.rs \
        iris-mpc-cpu/src/execution/hawk_main/insert.rs \
        iris-mpc-cpu/src/hnsw/graph/test_utils.rs \
        iris-mpc-cpu/src/execution/hawk_main/test_utils.rs \
        iris-mpc-bins/bin/iris-mpc-cpu/db_sanity_check.rs
git commit -m "$(cat <<'EOF'
feat(hnsw): make insert_apply fallible with strict-increase id check

GraphMem::insert_apply now takes &GraphMutation<V> and returns Result<()>.
Mutations whose id is not strictly greater than the graph's
last_modification_id are rejected without side effects. On success the
counter advances to the applied id. insert_apply_all is a slice
convenience that short-circuits on the first violation.

insert.rs stamps groups with sequential ids from
graph.next_modification_id() immediately before per-group apply; Task 4
moves this to the natural construction point via MutationIdAllocator.

WAL replay sites (test_utils.rs and db_sanity_check.rs) are stubbed with
todo!() until Task 5 changes the persistence shape and reintroduces the
per-group apply path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Introduce `MutationIdAllocator` and stamp ids at construction time in `insert::insert`

The temporary in-place stamping added at the end of `insert::insert` in Task 3 is replaced by stamping each group at its natural construction point. The `HawkActor::insert` no-memory-persistence fast path is also updated to set `id: 0` explicitly (no graph available; not applied).

**Files:**
- Modify: `iris-mpc-cpu/src/execution/hawk_main/insert.rs`
- Modify: `iris-mpc-cpu/src/execution/hawk_main.rs:770-799`

- [ ] **Step 1: Write the structural test for sequential id assignment**

In `iris-mpc-cpu/src/execution/hawk_main/insert.rs`, the existing tests (`test_insert_with_pure_deletion_preserves_slot_order`, `test_insert_with_combined_replace_emits_addnode_then_removenode`, `test_insert_with_none_slot_yields_none`) already exercise the function. Add a new test asserting that the returned groups carry strictly-increasing ids starting from the graph's `next_modification_id`:

```rust
    #[tokio::test]
    async fn test_insert_stamps_strictly_increasing_ids_per_slot() {
        let mut store = PlaintextStore::default();
        let mut graph: GraphMem<<PlaintextStore as VectorStore>::VectorRef> = GraphMem::new();
        let searcher = HnswSearcher::new_with_test_parameters();

        let expected_start = graph.next_modification_id();

        let plans = vec![
            Some(dummy_insert_plan(UpdateEntryPoint::SetUnique { layer: 0 })),
            None,
            Some(dummy_insert_plan(UpdateEntryPoint::False)),
        ];
        let insert_ids: VecRequests<Option<<PlaintextStore as VectorStore>::VectorRef>> =
            vec![None, None, None];
        let replace_ids: VecRequests<Option<<PlaintextStore as VectorStore>::VectorRef>> =
            vec![None, None, None];

        let grouped = insert(
            &mut store,
            &mut graph,
            &searcher,
            plans,
            &insert_ids,
            &replace_ids,
        )
        .await
        .expect("insert should succeed");

        let ids: Vec<u64> = grouped
            .iter()
            .filter_map(|opt| opt.as_ref().map(|g| g.id))
            .collect();
        assert_eq!(ids.len(), 2, "two non-None slots produce two groups");
        assert_eq!(ids[0], expected_start, "first id is next_modification_id");
        assert_eq!(ids[1], expected_start + 1, "ids are sequential");
        assert_eq!(graph.last_modification_id, expected_start + 1, "counter advanced");
    }
```

- [ ] **Step 2: Run; confirm FAIL**

```bash
cargo test --release -p iris-mpc-cpu --lib execution::hawk_main::insert::tests::test_insert_stamps_strictly_increasing_ids_per_slot -- --test-threads=1
```

Expected: FAIL — Task 3's in-place stamping happens after `insert_prepare_batch` but the test checks that the returned groups carry the expected ids; this may already pass. Confirm. If it passes, jump to Step 5 (refactor) since the behavior is already correct from Task 3's in-place stamping.

- [ ] **Step 3: Add `MutationIdAllocator` to `insert.rs`**

At the top of `iris-mpc-cpu/src/execution/hawk_main/insert.rs`, just above `pub async fn insert(...)`, add:

```rust
/// Mints sequential `u64` ids for `GraphMutation` groups built during a single
/// `insert` call. Seeded from `GraphMem::next_modification_id()`; never touches
/// the graph itself. The graph's `last_modification_id` only advances when each
/// stamped group is later applied via `GraphMem::insert_apply`.
struct MutationIdAllocator {
    next: u64,
}

impl MutationIdAllocator {
    fn new(start: u64) -> Self {
        Self { next: start }
    }

    fn next(&mut self) -> u64 {
        let id = self.next;
        self.next += 1;
        id
    }
}
```

- [ ] **Step 4: Stamp ids at construction time, remove Task-3 in-place stamping**

In `iris-mpc-cpu/src/execution/hawk_main/insert.rs::insert`, near the top of the function (right after the `assert_eq!`s, around line 80), seed the allocator:

```rust
    let mut id_allocator = MutationIdAllocator::new(graph.next_modification_id());
```

Within the slot-building loop (around line 93-143), where the current code is:

```rust
        if !request_mutations.is_empty() {
            mutations[idx] = Some(GraphMutation { id: 0, ops: request_mutations });
        }
```

replace `id: 0` with `id: id_allocator.next()`:

```rust
        if !request_mutations.is_empty() {
            mutations[idx] = Some(GraphMutation { id: id_allocator.next(), ops: request_mutations });
        }
```

Then remove the in-place stamping block added in Task 3 (around line 150) — the one that begins `// Temporary in-place id stamping`. The block is no longer needed because the groups already carry their ids. Drop the `let mut grouped_mutations` qualifier reverting it to `let grouped_mutations` if appropriate.

The per-group apply loop stays:

```rust
    for group in grouped_mutations.iter().flatten() {
        graph.insert_apply(group)?;
    }
```

- [ ] **Step 5: Update `HawkActor::insert` no-memory-persistence fast path**

In `iris-mpc-cpu/src/execution/hawk_main.rs`, around line 770-799, the path enters when `self.args.hnsw_disable_memory_persistence` is true. Today after Task 1 it builds `Some(GraphMutation { id: 0, ops: mutations })`. Leave `id: 0` here with a clarifying comment, since this branch never applies the mutation to a graph:

```rust
                    if mutations.is_empty() {
                        None
                    } else {
                        // No in-memory graph to mint from in this branch
                        // (hnsw_disable_memory_persistence). The mutation is
                        // returned for downstream serialization but never
                        // applied here, so a placeholder id of 0 is fine.
                        Some(GraphMutation { id: 0, ops: mutations })
                    }
```

- [ ] **Step 6: Run the new test and the existing insert tests; confirm PASS**

```bash
cargo test --release -p iris-mpc-cpu --lib execution::hawk_main::insert -- --test-threads=1
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add iris-mpc-cpu/src/execution/hawk_main/insert.rs \
        iris-mpc-cpu/src/execution/hawk_main.rs
git commit -m "$(cat <<'EOF'
feat(hawk): mint modification ids in insert::insert at construction

MutationIdAllocator is seeded from graph.next_modification_id() at the
top of insert::insert and stamps each per-slot group at the natural
construction point. The temporary in-place stamping added in the
previous commit is removed.

HawkActor::insert's no-memory-persistence fast path keeps id: 0 with a
comment — that branch returns the plan but never applies it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Change `SingleHawkMutation.plans` to `BothEyes<Vec<GraphMutation>>`, fix persistence and replay

This task picks up the WAL replay paths stubbed with `todo!()` in Task 3, switches the on-disk shape to `BothEyes<Vec<GraphMutation>>` end-to-end (write and read), and updates all serde tests.

**Files:**
- Modify: `iris-mpc-cpu/src/execution/hawk_main.rs:1593-1602` (`SingleHawkMutation` definition), `:1652-1668` (`HawkMutation::persist`), `:1980-2100` (`handle_mutations` lifting), serde tests around `:2475-2645`.
- Modify: `iris-mpc-cpu/src/hnsw/graph/graph_store.rs:24-43` (`GraphMutationRow::deserialize_mutations`), round-trip test around `:559-577`.
- Modify: `iris-mpc-cpu/src/hnsw/graph/test_utils.rs:226-229` (replay loop — replace `todo!`).
- Modify: `iris-mpc-bins/bin/iris-mpc-cpu/db_sanity_check.rs:324` (replay loop — replace `todo!`).

- [ ] **Step 1: Write a failing serde round-trip test for the new SingleHawkMutation shape**

Find the existing `SingleHawkMutation` serde test in `iris-mpc-cpu/src/execution/hawk_main.rs` (around line 2487-2515, the test that constructs a mutation with `plans: [Some(plan), None]` style). Below it (or replace the closest existing variant), add:

```rust
    #[test]
    fn single_hawk_mutation_per_side_vec_round_trips() {
        let plan_a = GraphMutation {
            id: 5,
            ops: vec![MutationOp::AddNode {
                id: VectorId::from_serial_id(1),
                height: 1,
                update_ep: UpdateEntryPoint::False,
            }],
        };
        let plan_b = GraphMutation {
            id: 6,
            ops: vec![MutationOp::RemoveNode {
                id: VectorId::from_serial_id(2),
            }],
        };
        let mutation = SingleHawkMutation {
            plans: [vec![plan_a.clone()], vec![plan_b.clone()]],
            modification_key: None,
            request_index: None,
        };
        let bytes = mutation.serialize().expect("serialize");
        let back: SingleHawkMutation = bincode::deserialize(&bytes).expect("deserialize");
        assert_eq!(back.plans[0].len(), 1);
        assert_eq!(back.plans[0][0].id, 5);
        assert_eq!(back.plans[1].len(), 1);
        assert_eq!(back.plans[1][0].id, 6);
    }
```

- [ ] **Step 2: Run; confirm FAIL to compile**

```bash
cargo test --release -p iris-mpc-cpu --lib execution::hawk_main::tests::single_hawk_mutation_per_side_vec_round_trips -- --test-threads=1
```

Expected: compile error — `plans` field shape is `BothEyes<Option<ConnectPlan>>`, can't take `[vec![...], vec![...]]`.

- [ ] **Step 3: Change `SingleHawkMutation.plans` shape**

In `iris-mpc-cpu/src/execution/hawk_main.rs:1593-1602`:

```rust
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SingleHawkMutation {
    pub plans: BothEyes<Option<ConnectPlan>>,
    ...
}
```

becomes:

```rust
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SingleHawkMutation {
    pub plans: BothEyes<Vec<GraphMutation<VectorId>>>,

    #[serde(skip)]
    pub modification_key: Option<ModificationKey>,

    #[serde(skip)]
    pub request_index: Option<RequestIndex>,
}
```

(`ConnectPlan` is already aliased to `GraphMutation<VectorId>` at `:472`. We could keep `BothEyes<Vec<ConnectPlan>>` for brevity — pick that:)

```rust
    pub plans: BothEyes<Vec<ConnectPlan>>,
```

- [ ] **Step 4: Update `HawkMutation::handle_mutations` to lift from `Option` to `Vec`**

In `iris-mpc-cpu/src/execution/hawk_main.rs`, find the construction site around line 1979:

```rust
        let mut plans_both_sides: Vec<BothEyes<Option<ConnectPlan>>> =
```

becomes:

```rust
        let mut plans_both_sides: Vec<BothEyes<Vec<ConnectPlan>>> =
```

Wherever the `plans_both_sides` element is initialized to `[None, None]` (default), change to `[Vec::new(), Vec::new()]`. Find the loop around line 2042-2044:

```rust
            // Store plans for this side
            for (plan, both_sides) in izip!(plans, &mut plans_both_sides) {
                both_sides[*side as usize] = plan;
            }
```

The `plan` here is `Option<ConnectPlan>` (per-slot return from `hawk_actor.insert`). Lift to a Vec:

```rust
            // Store plans for this side
            for (plan, both_sides) in izip!(plans, &mut plans_both_sides) {
                if let Some(p) = plan {
                    both_sides[*side as usize].push(p);
                }
            }
```

(The initialization needs to produce per-slot `[Vec::new(), Vec::new()]`. Confirm by reading the construction site; it likely uses `vec![[None, None]; n]` or similar. Change to `vec![[Vec::new(), Vec::new()]; n]`. If that doesn't compile due to non-Clone Vec elements within the array, build it explicitly: `(0..n).map(|_| [Vec::new(), Vec::new()]).collect()`.)

- [ ] **Step 5: Update `HawkMutation::persist` to serialize the new shape**

In `iris-mpc-cpu/src/execution/hawk_main.rs:1651-1665`, the existing body serializes `&mutation.plans` (which is now `BothEyes<Vec<ConnectPlan>>` — different bytes from before; that's intentional). The persist body otherwise reads as today; no logic change needed, just confirm the comment is still accurate:

```rust
                    // Serialize the plans (BothEyes<Vec<ConnectPlan>>)
                    let serialized = bincode::serialize(&mutation.plans)
```

Update the comment to say `BothEyes<Vec<ConnectPlan>>` instead of the previous `BothEyes<Option<ConnectPlan>>`.

- [ ] **Step 6: Update `GraphMutationRow::deserialize_mutations`**

In `iris-mpc-cpu/src/hnsw/graph/graph_store.rs:37-43`:

```rust
impl GraphMutationRow {
    pub fn deserialize_mutations(&self) -> Result<BothEyes<Vec<GraphMutation<IrisVectorId>>>> {
        let both_eyes: BothEyes<Vec<GraphMutation<IrisVectorId>>> =
            bincode::deserialize(&self.serialized_mutations)?;
        Ok(both_eyes)
    }
}
```

The return type is already `BothEyes<Vec<GraphMutation<IrisVectorId>>>` — only the *meaning* of `GraphMutation` has changed (it's now a record struct). Since Task 1 changed the struct shape, this code stays as-is but the on-disk bytes it expects are now: each side serialized as `Vec<{id, Vec<MutationOp>}>` rather than the previous mismatched `Option<Vec<MutationOp>>` shape. No code change required here beyond confirming compile-clean.

Update the docstring at line 27 of the same file to reflect the new shape:

```rust
    /// Bincode-serialized `BothEyes<Vec<GraphMutation<VectorId>>>` (mutations for both eyes)
    pub serialized_mutations: Vec<u8>,
```

(That's already the comment — keep it.)

- [ ] **Step 7: Replace the `todo!()` placeholders in the replay loops**

In `iris-mpc-cpu/src/hnsw/graph/test_utils.rs:226-229`, replace the Task-3 `todo!()` with:

```rust
        for row in mutation_rows {
            let both_eyes = row.deserialize_mutations()?;
            for m in &both_eyes[LEFT]  { graph[LEFT].insert_apply(m)?;  }
            for m in &both_eyes[RIGHT] { graph[RIGHT].insert_apply(m)?; }
        }
```

In `iris-mpc-bins/bin/iris-mpc-cpu/db_sanity_check.rs:324`, similarly:

```rust
        let both_eyes = row.deserialize_mutations()?;
        for m in &both_eyes[LEFT]  { graph[LEFT].insert_apply(m)?;  }
        for m in &both_eyes[RIGHT] { graph[RIGHT].insert_apply(m)?; }
```

(Inspect surrounding code to align variable names and error propagation. The pattern is per-group apply with `?` error propagation.)

- [ ] **Step 8: Update the `hawk_graph_mutations` round-trip test in `graph_store.rs`**

The existing test at `graph_store.rs:559-577` (`test_insert_hawk_graph_mutations_round_trip`) inserts a fixed byte payload and reads it back. Add a new test that exercises the full per-row deserialize:

```rust
    #[tokio::test]
    #[cfg(feature = "db_dependent")]
    async fn test_hawk_graph_mutations_full_round_trip() -> Result<()> {
        use crate::hnsw::graph::mutation::{GraphMutation, MutationOp, UpdateEntryPoint};
        let store = TestGraphPg::<PlaintextStore>::new().await?;

        let plan_left = GraphMutation::<IrisVectorId> {
            id: 1,
            ops: vec![MutationOp::AddNode {
                id: IrisVectorId::from_serial_id(1),
                height: 1,
                update_ep: UpdateEntryPoint::SetUnique { layer: 0 },
            }],
        };
        let plan_right = GraphMutation::<IrisVectorId> {
            id: 1,
            ops: vec![MutationOp::AddNode {
                id: IrisVectorId::from_serial_id(2),
                height: 1,
                update_ep: UpdateEntryPoint::SetUnique { layer: 0 },
            }],
        };
        let both: BothEyes<Vec<GraphMutation<IrisVectorId>>> =
            [vec![plan_left.clone()], vec![plan_right.clone()]];
        let payload = bincode::serialize(&both)?;

        let mut graph_tx = store.tx().await?;
        store
            .insert_hawk_graph_mutations(&mut graph_tx.tx, 42, &payload)
            .await?;
        graph_tx.tx.commit().await?;

        let rows = store.get_hawk_graph_mutations_after(None).await?;
        assert_eq!(rows.len(), 1);
        let back = rows[0].deserialize_mutations()?;
        assert_eq!(back[0].len(), 1);
        assert_eq!(back[0][0].id, plan_left.id);
        assert_eq!(back[1].len(), 1);
        assert_eq!(back[1][0].id, plan_right.id);

        store.cleanup().await?;
        Ok(())
    }
```

(Import `BothEyes` and `IrisVectorId` and the mutation types as needed.)

- [ ] **Step 9: Fix the existing serde tests around `:2475-2645`**

The tests in `iris-mpc-cpu/src/execution/hawk_main.rs` construct `SingleHawkMutation` literals like:

```rust
        let mutation = SingleHawkMutation {
            plans: [Some(plan.clone()), None],
            modification_key: Some(key),
            request_index: None,
        };
```

Convert each to:

```rust
        let mutation = SingleHawkMutation {
            plans: [vec![plan.clone()], Vec::new()],
            modification_key: Some(key),
            request_index: None,
        };
```

`[None, None]` → `[Vec::new(), Vec::new()]`. `[Some(p), Some(q)]` → `[vec![p], vec![q]]`. Apply this systematically to all sites in the test module.

Also update any assertions that pattern-match on `Some(...)` / `None` to check Vec length / index access. For example:

- `mutation.plans[LEFT].is_some()` → `!mutation.plans[LEFT].is_empty()`
- `mutation.plans[LEFT].as_ref().unwrap()` → `&mutation.plans[LEFT][0]`
- `mutation.plans[LEFT].is_none()` → `mutation.plans[LEFT].is_empty()`

- [ ] **Step 10: Fix any production assertions on `plans[…].is_some()`**

Search the codebase for any non-test pattern-match on `plans[…]`:

```bash
grep -rn "\.plans\[" /home/bgillespie/Workspaces/iris-mpc/iris-mpc/iris-mpc-cpu/src --include='*.rs'
```

Update each access from Option-style to Vec-style as in Step 9.

- [ ] **Step 11: Compile**

```bash
cargo build --release --all-features --tests
```

Expected: clean build.

- [ ] **Step 12: Run the in-module serde test and the new round-trip test**

```bash
cargo test --release -p iris-mpc-cpu --lib execution::hawk_main::tests::single_hawk_mutation_per_side_vec_round_trips -- --test-threads=1
cargo test --release -p iris-mpc-cpu --lib --features db_dependent hnsw::graph::graph_store::tests::test_hawk_graph_mutations_full_round_trip -- --test-threads=1
```

Expected: PASS.

(The second test requires the dev Postgres running: `just dev-pg-up`.)

- [ ] **Step 13: Commit**

```bash
git add iris-mpc-cpu/src/execution/hawk_main.rs \
        iris-mpc-cpu/src/hnsw/graph/graph_store.rs \
        iris-mpc-cpu/src/hnsw/graph/test_utils.rs \
        iris-mpc-bins/bin/iris-mpc-cpu/db_sanity_check.rs
git commit -m "$(cat <<'EOF'
feat(hawk): per-side Vec<GraphMutation> in SingleHawkMutation; fix wire shape

SingleHawkMutation.plans changes from BothEyes<Option<ConnectPlan>> to
BothEyes<Vec<ConnectPlan>>. Empty Vec means "no change on that side";
the multi-group capability is forward-compatible with the deferred
per-slot-splits refactor (today each Vec carries 0 or 1 group).

This also resolves the pre-existing write/read encoding asymmetry in
hawk_graph_mutations: writes used to serialize Option<GroupedMutations>
while reads expected Vec<GraphMutation_old>. The new canonical shape is
BothEyes<Vec<GraphMutation>> on both sides.

WAL replay loops in test_utils.rs and db_sanity_check.rs are updated to
apply each row's groups per-eye, per-group, propagating insert_apply
errors. Strict-increase replay safety is now active end-to-end within a
single process lifetime; cross-restart coverage awaits GraphV4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Full-suite verification

**Files:** none modified — just verify everything together.

- [ ] **Step 1: Bring up the dev Postgres**

```bash
just dev-pg-up
```

- [ ] **Step 2: Format check**

```bash
cargo fmt --all -- --check
```

Expected: clean.

- [ ] **Step 3: Clippy**

```bash
cargo clippy --workspace --all-targets --all-features -q -- -D warnings
```

Expected: clean.

- [ ] **Step 4: All tests, single-threaded**

```bash
cargo test --release -- --test-threads=1
```

Expected: PASS.

- [ ] **Step 5: db_dependent tests**

```bash
cargo test --release --features db_dependent -- --test-threads=1
```

Expected: PASS.

- [ ] **Step 6: Spot-check existing tests that exercise the changed paths**

```bash
cargo test --release -p iris-mpc-cpu --lib execution::hawk_main -- --test-threads=1
cargo test --release -p iris-mpc-cpu --lib hnsw -- --test-threads=1
```

Expected: PASS, including the new tests added in Tasks 1-5.

- [ ] **Step 7: Optional — full `just lint`**

```bash
just lint
```

Expected: clean.

- [ ] **Step 8: Tear down the dev Postgres**

```bash
just dev-pg-down
```

- [ ] **Step 9: No commit needed (verification only)**

If anything failed, return to the relevant task and fix; this task is purely a final gate.

---

## Self-Review (already completed by plan author)

**Spec coverage:** Each requirement in the spec maps to a task —

- "`GraphMutation` becomes a full struct with `id` and `ops`" → Task 1.
- "`SingleHawkMutation.plans` becomes `BothEyes<Vec<GraphMutation>>`" → Task 5.
- "`GraphMem` gains `pub last_modification_id: u64` and `next_modification_id()` peek" → Task 2.
- "`insert_apply` fallible per-group with strict-increase check; `insert_apply_all`" → Task 3.
- "MutationIdAllocator in `insert.rs`, seeded from `next_modification_id`" → Task 4.
- "Persistence canonical shape `BothEyes<Vec<GraphMutation>>`, replay updated" → Task 5.
- "GraphV3 From conversions touched up" → Task 2 (steps 9-10).
- "Genesis path" → handled implicitly by Task 5 (the per-side Vec shape change flows through `insert::insert` which genesis calls).
- "Tests" → distributed across Tasks 1, 2, 3, 4, 5 with explicit test additions.

**Placeholder scan:** no `TBD`, `TODO` (other than the explicit Task-3-to-Task-5 `todo!()` bridge that is removed in Task 5 Step 7), no "implement later". The Task-3 bridge is an intentional checkpointing device — flagged in the commit message and resolved in the immediately following task.

**Type consistency:** `GraphMutation<V> { id: u64, ops: Vec<MutationOp<V>> }`, `last_modification_id: u64`, `next_modification_id(&self) -> u64`, `insert_apply(&mut self, &GraphMutation<V>) -> Result<()>`, `insert_apply_all(&mut self, &[GraphMutation<V>]) -> Result<()>`, `plans: BothEyes<Vec<ConnectPlan>>` (= `BothEyes<Vec<GraphMutation<VectorId>>>`) — used consistently across all tasks.
