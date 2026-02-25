# BatchQuery and ServerJobResult

**File:** `iris-mpc-common/src/job.rs`

## BatchQuery

The central data structure for a batch of requests sent to the MPC engine.

### Core Fields

```rust
pub struct BatchQuery {
    // Request ordering
    pub requests_order: Vec<RequestIndex>,     // Original submission order

    // Uniqueness + Reauth + ResetCheck fields
    pub request_ids: Vec<String>,              // Operation IDs (signup_id, reauth_id, reset_id)
    pub request_types: Vec<String>,            // Message type strings
    pub metadata: Vec<BatchMetadata>,
    pub skip_persistence: Vec<bool>,
    pub or_rule_indices: Vec<Vec<u32>>,        // OR-rule serial IDs per request

    // Iris data (concatenated across rotations)
    pub left_iris_rotated_requests: IrisQueryBatchEntries,
    pub right_iris_rotated_requests: IrisQueryBatchEntries,
    pub left_iris_interpolated_requests: IrisQueryBatchEntries,
    pub right_iris_interpolated_requests: IrisQueryBatchEntries,
    // ... mirrored variants for full-face mirror detection

    // Reauth-specific
    pub reauth_target_indices: HashMap<String, u32>,  // request_id → 0-based target index
    pub reauth_use_or_rule: HashMap<String, bool>,

    // Deletion-specific
    pub deletion_requests_indices: Vec<u32>,           // 0-based indices to delete
    pub deletion_requests_metadata: Vec<BatchMetadata>,

    // Reset-specific
    pub reset_update_indices: Vec<u32>,
    pub reset_update_request_ids: Vec<String>,
    pub reset_update_shares: Vec<GaloisSharesBothSides>,

    // Persistence tracking
    pub modifications: HashMap<ModificationKey, Modification>,

    // Lookback
    pub luc_lookback_records: usize,
}
```

### RequestIndex (line ~332)

```rust
pub enum RequestIndex {
    UniqueReauthResetCheck(usize),  // Index into request_ids, request_types, iris data, etc.
    Deletion(usize),                 // Index into deletion_requests_indices
    ResetUpdate(usize),              // Index into reset_update_*
}
```

`requests_order` preserves the original submission order across all types. Each variant's inner `usize` indexes into that type's specific arrays.

### Push Methods

```rust
// Adds uniqueness/reauth/reset_check request
pub fn push_matching_request(
    sns_message_id, request_id, request_type, metadata, or_rule_serial_ids, skip_persistence
)

// Adds deletion request
pub fn push_deletion_request(
    sns_message_id, deletion_0_index: u32, metadata   // deletion_0_index is 0-based
)

// Adds reset update request
pub fn push_reset_update_request(
    sns_message_id, reset_update_0_index: u32, request_id, metadata, shares
)
```

## ModificationKey (iris-mpc-common/src/helpers/sync.rs)

```rust
pub enum ModificationKey {
    RequestId(String),        // For uniqueness insertions (keyed by signup_id)
    RequestSerialId(u32),     // For reauth, reset, deletion (keyed by 1-based serial_id)
}
```

## ServerJobResult

```rust
pub type ServerJobResult = iris_mpc_common::job::ServerJobResult<HawkMutation>;
```

Generic struct parameterized by `actor_data` type. Key fields:

### Per-Request Arrays (indexed by request position)

| Field | Type | Populated By |
|-------|------|-------------|
| `merged_results` | `Vec<u32>` | Inserted index, or first match index, or `u32::MAX` |
| `request_ids` | `Vec<String>` | Echo of input |
| `request_types` | `Vec<String>` | Echo of input |
| `metadata` | `Vec<BatchMetadata>` | Echo of input |
| `matches` | `Vec<bool>` | `true` if NOT a `UniqueInsert` |
| `matches_with_skip_persistence` | `Vec<bool>` | `true` if NOT `UniqueInsert` or `UniqueInsertSkipped` |
| `skip_persistence` | `Vec<bool>` | Echo of input |
| `match_ids` | `Vec<Vec<u32>>` | Normal orient, both eyes, no intra-batch |
| `partial_match_ids_left` | `Vec<Vec<u32>>` | Left eye only, normal orient |
| `partial_match_ids_right` | `Vec<Vec<u32>>` | Right eye only, normal orient |
| `partial_match_counters_left` | `Vec<usize>` | Count of left-eye matches |
| `partial_match_counters_right` | `Vec<usize>` | Count of right-eye matches |
| `successful_reauths` | `Vec<bool>` | `true` for `ReauthUpdate` decisions |
| `matched_batch_request_ids` | `Vec<Vec<String>>` | Intra-batch match request IDs |

### Mirror Detection Fields

| Field | Type |
|-------|------|
| `full_face_mirror_match_ids` | `Vec<Vec<u32>>` |
| `full_face_mirror_partial_match_ids_{left,right}` | `Vec<Vec<u32>>` |
| `full_face_mirror_partial_match_counters_{left,right}` | `Vec<usize>` |
| `full_face_mirror_attack_detected` | `Vec<bool>` |

### Deletion / Reset / Reauth Echo Fields

| Field | Type | Source |
|-------|------|--------|
| `deleted_ids` | `Vec<u32>` | `batch.deletion_requests_indices` (0-based) |
| `reset_update_indices` | `Vec<u32>` | Echo of input |
| `reset_update_request_ids` | `Vec<String>` | Echo of input |
| `reset_update_shares` | `Vec<GaloisSharesBothSides>` | Echo of input |
| `reauth_target_indices` | `HashMap<String, u32>` | Echo of input |
| `reauth_or_rule_used` | `HashMap<String, bool>` | Echo of input |
| `modifications` | `HashMap<ModificationKey, Modification>` | Echo of input |

### Actor Data

| Field | Type |
|-------|------|
| `actor_data` | `HawkMutation` |

Contains `Vec<SingleHawkMutation>`, one per entry in `requests_order`. Each has `plans: BothEyes<Option<ConnectPlan>>` for graph persistence.
