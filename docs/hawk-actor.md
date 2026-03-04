# HawkActor — MPC Batch Processing Engine

**Primary file:** `iris-mpc-cpu/src/execution/hawk_main.rs`
**Submodules:** `iris-mpc-cpu/src/execution/hawk_main/`

## HawkActor Struct (line ~274)

Holds the full state of one MPC party node:

| Field | Type | Purpose |
|-------|------|---------|
| `args` | `HawkArgs` | CLI config (party_index, addresses, HNSW params, etc.) |
| `searcher` | `Arc<HnswSearcher>` | HNSW search/insert logic and parameters |
| `prf_key` | `Option<Arc<[u8; 16]>>` | Shared PRF key for HNSW layer selection |
| `iris_store` | `BothEyes<Aby3SharedIrisesRef>` | In-memory secret-shared iris codes [Left, Right] |
| `graph_store` | `BothEyes<GraphRef>` | In-memory HNSW graphs [Left, Right] |
| `workers_handle` | `BothEyes<IrisPoolHandle>` | NUMA-aware worker pools [Left, Right] |
| `networking` | `Box<dyn NetworkHandle>` | Manages TCP connections to other MPC parties |
| `party_id` | `usize` | This node's index (0, 1, or 2) |
| `error_ct` | `CancellationToken` | Signals networking errors |
| `anon_stats_store` | `Option<AnonStatsStore>` | Optional anonymized distance statistics |

## Key Type Aliases (lines ~360-375)

```rust
pub type BothEyes<T> = [T; 2];           // [Left, Right]
pub type BothOrient<T> = [T; 2];         // [Normal, Mirror]
pub type VecRequests<T> = Vec<T>;        // Per-request in a batch
type VecEdges<T> = Vec<T>;              // Per-neighbor (graph edges)
type MapEdges<T> = HashMap<VectorId, T>;
type GraphRef = Arc<RwLock<GraphMem<Aby3VectorRef>>>;
type Aby3Ref = Arc<RwLock<Aby3Store>>;
```

## Constants (lines ~164-209)

```rust
HAWK_DISTANCE_FN = DistanceFn::MinFhd      // Distance function choice
HAWK_MINFHD_ROTATIONS = 11                 // Rotations for MinFhd
HAWK_BASE_ROTATIONS_MASK = CENTER_AND_10_MASK  // Base rotations for HNSW search
NEIGHBORHOOD_MODE = NeighborhoodMode::Sorted   // Candidate list strategy
LINEAR_SCAN_MAX_GRAPH_LAYER = 1
```

Compile-time checks (lines 173-201) enforce valid combinations of these constants.

## HawkHandle — Concurrency Wrapper (line ~1479)

`HawkHandle` wraps `HawkActor` in an `mpsc::channel(1)` for backpressure:

```
submit_batch_query(BatchQuery)
  → HawkRequest::from(BatchQuery)
  → send to mpsc channel
  → handle_job processes it
  → oneshot channel returns HawkResult
  → HawkResult.job_result() → ServerJobResult
```

## handle_job Flow (line ~1551)

```
handle_job(hawk_actor, sessions, request)
│
├── 1. numa_realloc(request)           — Move iris data to local NUMA node
│
├── 2. apply_deletions(hawk_actor, &request)  — Replace deleted irises with dummies
│      (reset.rs:62-79, runs BEFORE searches)
│
├── 3. do_search(Normal) ─┐            — HNSW search + intra-batch matching
│   do_search(Mirror)  ───┤            — Both orientations run concurrently via try_join!
│                          │
│   Each do_search:        │
│   ├── intra_batch_is_match (spawned)
│   ├── search::search (HNSW nearest neighbors)
│   ├── BatchStep1::new (organize search results)
│   ├── is_match_batch (fill missing matches)
│   └── step2 (combine with intra-batch)
│                          │
│   matches_normal.step3(matches_mirror)  — Final matching decision
│
├── 4. update_anon_stats(&search_results) — Persist distance statistics
│
├── 5. search_to_reset(...)            — Find insertion points for reset updates
│
├── 6. handle_mutations(...)           — Insert into HNSW graph, produce ConnectPlans
│      ├── Collect update_ids (reauth targets, reset targets)
│      ├── For each eye: build insert_plans from decisions
│      ├── hawk_actor.insert(sessions, plans, update_ids)
│      └── Build SingleHawkMutation with modification_key + request_index
│
└── 7. HawkResult::new(batch, match_result, mutations)
```

## HawkRequest (line ~887)

Constructed from `BatchQuery` via `From<BatchQuery>`:

| Field | Type | Source |
|-------|------|--------|
| `batch` | `BatchQuery` | Original batch data |
| `queries` | `SearchQueries<MASK>` | Normal orientation: left-vs-left, right-vs-right |
| `queries_mirror` | `SearchQueries<MASK>` | Mirror orientation: left-vs-right, right-vs-left |
| `ids` | `SearchIds` | Request identifiers |

Mirror orientation (lines 922-938) swaps left/right for full-face mirror attack detection.

## HawkResult → ServerJobResult (line ~1264)

`job_result()` maps internal results to the external `ServerJobResult`:
- `merged_results` — per-request: inserted_id or first match_id or `u32::MAX`
- `matches` — `true` if not a `UniqueInsert`
- `match_ids` — Normal orientation, both eyes, no intra-batch
- `partial_match_ids_{left,right}` — per-eye matches
- `full_face_mirror_*` — Mirror orientation matches
- `deleted_ids` — echo of `batch.deletion_requests_indices`
- `actor_data` — `HawkMutation` for graph persistence

## HawkMutation (line ~1387)

```rust
struct HawkMutation(Vec<SingleHawkMutation>);

struct SingleHawkMutation {
    plans: BothEyes<Option<ConnectPlan>>,    // Graph updates per eye
    modification_key: Option<ModificationKey>, // For persistence correlation
    request_index: Option<RequestIndex>,       // Back-reference to batch position
}
```

`persist()` batches graph link updates by side and executes them.

## Submodule Index

| File | Role |
|------|------|
| `reset.rs` | `apply_deletions()`, `search_to_reset()` |
| `search.rs` | HNSW search orchestration, `SearchParams`, `SearchQueries` |
| `insert.rs` | HNSW insertion, `InsertPlanV` |
| `matching.rs` | Multi-step matching pipeline: `BatchStep1` → `BatchStep2` → `BatchStep3`, `Decision` enum |
| `intra_batch.rs` | Within-batch iris comparison |
| `is_match_batch.rs` | Batch distance comparison for missing vector IDs |
| `rot.rs` | Rotation support: `VecRotationSupport`, mask constants (`ALL_ROTATIONS_MASK`, `CENTER_AND_10_MASK`, `CENTER_ONLY_MASK`) |
| `scheduler.rs` | `parallelize()` — runs futures concurrently |
| `session_groups.rs` | `SessionGroups` — organizes sessions by orientation and purpose |
| `state_check.rs` | Post-batch state validation across MPC parties |
| `iris_worker.rs` | NUMA-aware worker pool for iris operations |
| `test_utils.rs` | Test helpers (cfg(test) only) |

## Deletion Flow Detail

1. `apply_deletions` (reset.rs:62-79): Acquires write locks on both iris stores, calls `from_0_indices` to map 0-based indices to current VectorIds, then `store.update(del_id, dummy)` for each.
2. No HNSW graph mutation — the dummy iris has max distance and will never match.
3. `ServerJobResult.deleted_ids` just echoes `batch.deletion_requests_indices`.
4. Duplicate deletions within same batch: second `update` is idempotent (same version, same dummy).
5. Duplicate deletions across batches: `from_0_indices` picks up new version, bumps version again (dummy replaces dummy).
