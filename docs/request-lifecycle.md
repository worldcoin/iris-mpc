# Request Lifecycle

End-to-end flow from service client through server to MPC engine and back.

## 1. Client Side

tbd

## 2. Server Side

### Batch Collection

**File:** `iris-mpc/src/services/processors/batch.rs`

The server collects SQS messages into a `BatchQuery`:

| Message Type | Handler | Key Logic |
|-------------|---------|-----------|
| `identity_deletion` | `process_identity_deletion` (line 419) | Dedup check on `modifications[RequestSerialId]`, then `push_deletion_request(serial_id - 1)` |
| `uniqueness` | `process_uniqueness_request` (line 475) | Parse iris shares from S3, `push_matching_request` |
| `reauth` | `process_reauth_request` | Similar to uniqueness with reauth target |
| `reset_check` | `process_reset_check_request` | Query-only, no persistence |
| `reset_update` | `process_reset_update_request` | Iris replacement at target serial_id |

**Deletion dedup** (lines 436-447): If the same serial_id already has a modification in the CURRENT batch, the second request is silently skipped (`return Ok(())`). The SQS message is already deleted (line 427), so no response is ever generated for the skipped request.

This dedup only applies WITHIN a single server batch. Across batches, the `modifications` map is fresh.

### Job Result Processing

**File:** `iris-mpc/src/services/processors/job.rs`

After MPC computation, `ServerJobResult` is converted to per-request SNS responses:

```
identity_deletion_results = deleted_ids.iter().map(|idx| {
    serial_id = idx + 1;
    IdentityDeletionResult::new(party_id, serial_id, true)  // always success=true
})
```

Results sent to SNS per type (lines 389-440):
1. Uniqueness results
2. Reauth results
3. Identity deletion results
4. Reset check results
5. Reset update results

## 3. MPC Side

### Deletion Processing

**File:** `iris-mpc-cpu/src/execution/hawk_main/reset.rs` — `apply_deletions()` (line 62)

```rust
let dummy = GaloisRingSharedIris::dummy_for_party(hawk_actor.party_id);
let del_ids = request.deletion_ids(&stores[LEFT]);  // from_0_indices
for del_id in del_ids {
    for store in &mut stores {
        store.update(del_id, dummy.clone());  // Both eyes
    }
}
```

- Runs BEFORE any search operations in `handle_job`
- No HNSW graph modification — dummy iris has max distance
- `ServerJobResult.deleted_ids` = echo of input indices

### Uniqueness Processing

1. HNSW search for nearest neighbors (both orientations)
2. Distance comparison to determine matches
3. Decision: `UniqueInsert`, `UniqueInsertSkipped`, `ReauthUpdate`, or `NoMutation`
4. Insert into HNSW graph if mutation decision
5. `merged_results[i]` = inserted serial_id, or first match serial_id, or `u32::MAX`

## Serial ID Mapping Summary

```
Client IrisSerialId (1-based)
  ↓ request_enqueuer: IdentityDeletionRequest { serial_id }
Server process_identity_deletion:
  ↓ push_deletion_request(serial_id - 1)   →  0-based index
MPC apply_deletions:
  ↓ from_0_indices  →  VectorId(serial_id, current_version)
  ↓ store.update(del_id, dummy)
ServerJobResult.deleted_ids = [0-based indices]
  ↓ job.rs: serial_id = idx + 1
IdentityDeletionResult { serial_id }   →  back to 1-based
  ↓ SNS → SQS
Client is_correlation: parent.IrisSerialId == result.serial_id
```
