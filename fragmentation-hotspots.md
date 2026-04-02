# Fragmentation Hotspots Investigation

Follow-up to: [Genesis Batch Duration Degradation Post-Startup: Memory Fragmentation Investigation](https://www.notion.so/330a3b540bfe810f8334fa8d79859e2f)

## 1. `Layer::set_links` — neighborhood replacement alloc/free churn

**File:** `iris-mpc-cpu/src/hnsw/graph/layered_graph.rs:395`

```rust
pub fn set_links(&mut self, from: V, links: Vec<V>) {
    self.set_hash.add_unordered((&from, &links));
    let previous = self.links.insert(from.clone(), links);
    if let Some(previous) = previous {
        self.set_hash.remove((&from, &previous))
    }
}
```

**Problem:** Every neighborhood update allocates a fresh `Vec<VectorId>` (~2.5 KB at M=320) and drops the old one. No reuse of the existing allocation.

**Scale:** Each HNSW insert updates the neighbor lists of many existing nodes (not just the new node). With batch size ~75, that's potentially thousands of 2.5 KB alloc+free cycles per batch. Over hours, this creates heap fragmentation — many 2.5 KB holes interspersed with live allocations of different sizes.

**Fix idea:** Use `HashMap::entry()` to reuse the existing Vec's backing allocation:
```rust
match self.links.entry(from) {
    Entry::Occupied(mut e) => {
        self.set_hash.remove((e.key(), e.get()));
        let existing = e.get_mut();
        existing.clear();
        existing.extend_from_slice(&links);
        self.set_hash.add_unordered((e.key(), e.get()));
    }
    Entry::Vacant(e) => {
        self.set_hash.add_unordered((e.key(), &links));
        e.insert(links);
    }
}
```
Zero frees/allocs for updated neighborhoods — the old Vec keeps its capacity.

## 2. `layer_search_batched_v2` — per-search HashSet + Vec churn

**File:** `iris-mpc-cpu/src/hnsw/searcher.rs:1016`

Three fresh allocations per search, dropped on return:

1. **`visited: HashSet`** (line 1032) — grows to ~200K+ entries per search. At 8 bytes per VectorId + HashMap overhead (~50 bytes/entry with bucket metadata), that's ~10 MB per search. Freshly allocated every search, freed on return.

2. **`opened: HashSet`** (line 1035) — at most ~500 entries, so ~25 KB. Small individually but still churn.

3. **`cur_unopened: Vec`** (line 1106, 1226) — rebuilt from scratch every loop iteration via `Vec::from_iter` + filter + clone. Not just allocated once — reallocated every iteration of the main while loop.

**Scale:** Genesis searches every node in the batch (batch_size ~75, x2 orientations = ~150 searches). Each search allocates and frees ~10 MB for `visited` alone. Over hours: ~150 × (batches/hour) × 10 MB of alloc/free churn per hour. The `visited` HashSet is by far the dominant allocation here.

**Fix idea:** Pool `visited` and `opened` HashSets across searches. `clear()` retains the backing allocation. Could be a `SearchBuffers` struct held on the session or passed through the search call:
```rust
struct SearchBuffers<V> {
    visited: HashSet<V>,
    opened: HashSet<V>,
    cur_unopened: Vec<V>,
}

impl<V> SearchBuffers<V> {
    fn clear(&mut self) {
        self.visited.clear();  // retains capacity
        self.opened.clear();
        self.cur_unopened.clear();
    }
}
```

## 3. `get_links` clones on every read

**File:** `iris-mpc-cpu/src/hnsw/graph/layered_graph.rs:391`

```rust
pub fn get_links(&self, from: &V) -> Option<Vec<V>> {
    self.links.get(from).cloned()
}
```

**Problem:** Every `get_links` call clones the full neighbor Vec (~2.5 KB). Called hundreds of times per search in `open_node` / `open_nodes_batch` (searcher.rs:1327, 1365). The clone is immediately consumed — iterated, filtered, and dropped.

**Scale:** ~300 calls per search × ~150 searches per batch = ~45K clones per batch. Each is a 2.5 KB alloc+free cycle = ~110 MB of alloc churn per batch.

**Fix idea:** Return `&[V]` instead of cloning:
```rust
pub fn get_links(&self, from: &V) -> Option<&[V]> {
    self.links.get(from).map(|v| v.as_slice())
}
```
Requires adjusting callers to borrow instead of own, but the data is only read — no caller mutates the returned Vec.

## 4. MPC protocol intermediates — `lift`, `extract_msb`, `binary_add_3`

**Files:** `ampc-common/ampc-actor-utils/src/protocol/binary.rs`, `fhd_ops.rs`

The MPC call chain per `eval_distance_batch` (MinRotation path) is:

```
eval_distance_batch
  → dot_product (CPU workers, see #5)
  → gr_to_lifted_distances
      → galois_ring_to_rep3 (reshare_products: Vec alloc per batch)
      → lift_distances → lift()
          → transpose_pack_u64 (creates Vec<VecShare<u64>>)
          → a2b_pre per element (3 VecShares built element-by-element)
          → binary_add_3_get_two_carries
              → transposed_pack_xor/and (new Vecs per call)
              → and_many loop (one MPC round + Vec alloc per bit position)
          → bit_inject (3 rounds, Vec alloc per party role)
  → oblivious_min_distance_batch
      → oblivious_cross_compare per round
          → cross_mul → reshare_products (Vec alloc)
          → extract_msb_batch
              → x.to_vec() (copies input, line 1244)
              → two_way_split (reshare_products: Vec alloc)
              → transpose_pack_u64 (×2, Vec alloc)
              → binary_add_2_get_msb (Vecs per bit level)
      → min_of_pair_batch / min_round_robin_batch (more of the same)
```

**Problem:** Every call in this chain creates and drops intermediate `Vec`s and `VecShare`s. The sizes are proportional to the batch of vectors being distance-compared. In `layer_search_batched_v2`, the search opens ~200K nodes, meaning `eval_distance_batch` is called many times with varying batch sizes. Each call produces a full `lift` + `extract_msb` chain.

Key allocation patterns:
- `reshare_products` (ops.rs:39): allocates `round_a: Vec<RingElement<T>>`, clones it for network send, receives `round_b`, zips into output Vec. That's 3 Vecs per call × n elements.
- `lift` (binary.rs:788): builds `x_a` (VecShare, padded to 64-aligned), plus 3 Vec<VecShare> (`x1`, `x2`, `x3`) from element-by-element push loops. Then `bit_inject` at the end creates more Vecs.
- `extract_msb_batch` (binary.rs:1244): `x.to_vec()` copies the entire input slice before processing.
- `binary_add_3_get_two_carries`: iterates over bit positions calling `and_many` in a loop — each iteration allocates intermediate Vecs for the carry propagation.

**Scale:** The dot product produces 11 rotation distances per vector (MinRotation). So a distance eval against N vectors produces 11×N distances that flow through `lift` → `extract_msb` → `oblivious_min`. The MPC intermediates are O(11 × batch_size × bit_width) allocations per distance evaluation. With 16-bit distances lifted to 32-bit, that's ~32 levels of binary operations on the 11×-expanded batch. Unlike the searcher allocations (#2) which are large but few, these are many small-to-medium allocations (hundreds to thousands of bytes) in rapid succession — and the 11× rotation multiplier makes them 11× worse than a naive count would suggest.

**Fix idea:** This is deeper and harder to fix incrementally. Possible approaches:
- Pre-allocate a `MpcBuffers` scratch space sized for the maximum expected batch, reuse across calls.
- Convert `VecShare` operations to work on pre-allocated slices rather than producing new Vecs.
- The `x.to_vec()` in `extract_msb_batch` is a quick win — accept owned `VecShare` or use a `Cow`.

**Note:** These are in `ampc-common`, not `iris-mpc`. Changes here affect all consumers.

## 5. Dot product dispatch — per-chunk `Vec` and `oneshot` churn

**File:** `iris-mpc-cpu/src/execution/hawk_main/iris_worker.rs:181`

```rust
fn dispatch_rotation_dot_product_batch(
    &mut self,
    query: ArcIris,
    vector_ids: Vec<VectorId>,
    responses: &mut Vec<oneshot::Receiver<Vec<RingElement<u16>>>>,
) -> Result<()> {
    for chunk in vector_ids.chunks(Self::ROT_AWARE_BATCH_CHUNK_SIZE) {
        let (tx, rx) = oneshot::channel();
        let task = IrisTask::RotationAwareDotProductBatch {
            query: query.clone(),
            vector_ids: chunk.to_vec(),  // <-- alloc per chunk
            rsp: tx,                      // <-- oneshot channel per chunk
        };
        ...
    }
}
```

**Problem:** Each chunk (128 VectorIds) creates a new `Vec` via `chunk.to_vec()` (~1 KB) plus a `oneshot::channel()` (tokio oneshot is ~100 bytes). With ~200K vectors opened per search, that's ~1,500 chunks per eval_distance_batch call.

**Scale (REVISED):** Much larger than initially estimated. Per batch (batch_size=48):
- ~1,500 chunks per dot_product call
- ~25 dot_product calls per search (open_nodes iterations)
- 96 searches per batch (48 queries × 2 orientations)
- Total: **~3.6 million alloc/free cycles per batch**, each ~1 KB
- That's **~3.6 GB of alloc churn per batch** just from chunk dispatch

All on tokio threads, all in the arena. Despite being uniform-sized (which ptmalloc2 bins well), the sheer volume makes this potentially the **largest single contributor to arena churn by volume**.

**Fix idea:** Pre-allocate a pool of chunk buffers on `IrisPoolHandle`. Workers return buffers alongside results for recycling. Alternatively, send `(offset, len)` into a shared slab instead of owned Vecs.
