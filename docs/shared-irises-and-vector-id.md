# SharedIrises and VectorId

## VectorId

**File:** `iris-mpc-common/src/vector_id.rs`

```rust
pub type SerialId = u32;    // re-exported as IrisSerialId
pub type VersionId = i16;

pub struct VectorId {
    id: u32,        // serial_id (1-based)
    version: i16,   // version_id (starts at 0)
}
```

Key methods:
- `from_serial_id(id: u32)` — version = 0
- `from_0_index(index: u32)` — serial_id = index + 1, version = 0
- `next_version()` — same serial_id, version + 1
- `serial_id()` → u32, `version_id()` → i16, `index()` → u32 (serial_id)

**Important:** `IrisSerialId = SerialId = u32` (re-exported in `iris-mpc-common/src/lib.rs:33`).

The 0-based index used in `BatchQuery.deletion_requests_indices` maps to serial_id via `index + 1`.

## SharedIrises<I>

**File:** `iris-mpc-cpu/src/hawkers/shared_irises.rs`

In-memory store for secret-shared iris codes, keyed by serial_id.

```rust
pub struct SharedIrises<I> {
    points: Vec<Option<(VersionId, I)>>,  // indexed by serial_id
    size: usize,                           // count of occupied slots
    next_id: u32,                          // next unused serial_id
    set_hash: SetHash,                     // XOR-based hash for state checks
    empty_iris: I,                         // default for missing entries
}
```

### insert (line ~64)

```rust
pub fn insert(&mut self, vector_id: VectorId, iris: I) -> VectorId
```

- Extends `points` if needed to fit serial_id
- If slot occupied: removes old VectorId from `set_hash`, decrements `size`
- Sets `points[serial_id] = Some((version, iris))`
- Adds new VectorId to `set_hash`, increments `size`
- Updates `next_id` high-water mark

### update (line ~97)

```rust
pub fn update(&mut self, original_id: VectorId, iris: I) -> VectorId
```

Calls `insert(original_id.next_version(), iris)`. Bumps version by 1.

### from_0_indices (line ~175)

```rust
pub fn from_0_indices(&self, indices: &[u32]) -> Vec<VectorId>
```

Converts 0-based indices to VectorIds:
- `VectorId::from_0_index(idx)` → serial_id = idx + 1, version = 0
- Looks up `get_current_version(serial_id)` — returns stored version if exists
- Falls back to version 0 if not found

### get_current_version (line ~117)

```rust
pub fn get_current_version(&self, serial_id: SerialId) -> Option<VersionId>
```

Returns the currently stored version for a serial_id, or None if not present.

## SharedIrisesRef

**Same file, line ~190+**

```rust
pub struct SharedIrisesRef<I> {
    data: Arc<RwLock<SharedIrises<I>>>,
}
```

Async wrapper around `SharedIrises`. All methods acquire read/write locks:
- `insert()`, `append()`, `update()` — write lock
- `db_size()`, `checksum()` — read lock

## Deletion Semantics

When deleting an iris:
1. `apply_deletions` calls `from_0_indices` to get current VectorIds
2. `store.update(del_id, dummy)` replaces iris with dummy, bumps version
3. The HNSW graph is NOT modified — the dummy iris simply never matches

Duplicate deletion (same batch): Both calls use the same stale VectorId, second `insert` overwrites the first with identical data. Idempotent.

Duplicate deletion (cross-batch): `from_0_indices` picks up the bumped version from the first deletion. Second `update` bumps version again (dummy replaces dummy). No error, no meaningful change.

## SetHash

Used for cross-party state validation. XOR-based — `add_unordered(id)` and `remove(id)` are inverses. The `checksum()` is compared across MPC parties in `state_check` after each batch.
