# Continuous Rerandomization Plan

## Overview

Replaces the existing, one-off rerandomization protocol by a continuous, online process that rerandomizes shares while the system is running. No downtime or restart required.

Key design decision: in-memory shares are less likely to be exfiltrated, so only the DB (at-rest persistence) is rerandomized. The actor is completely unmodified. The rerand server handles everything, writing to a staging schema and then copying to live once all parties confirm.

## Architecture

1. **Rerand Server** (modified `iris-mpc-bins/bin/iris-mpc-upgrade/rerandomize_db.rs`, separate process, one per party) — rerandomizes shares, writes to staging, coordinates with peers via S3 markers, copies confirmed chunks to live DB. Replaces the existing one-off `RerandomizeDb` subcommand with a new `RerandomizeContinuous` subcommand. Core rerandomization logic in `iris-mpc-upgrade/src/rerandomization.rs` is reused; the new subcommand adds the continuous loop, S3 coordination, and staging management.
2. **Main Server** (existing, minimal changes) — at startup, syncs rerand progress with peers and catches up any missing chunks from staging before loading the DB into memory. Acquires `RERAND_MODIFY_LOCK` during modification writes to serialize with rerand applies.

The GPU actor, batch processing, and result processor are completely untouched.

## Seed & Randomness

One epoch is active at a time. At the start of each epoch:

1. Each rerand server generates a fresh BLS12-381 keypair
2. Private key is saved to Secrets Manager at `{env}/iris-mpc-db-rerandomization/epoch-{E}/private-key-party-{P}`
3. Public key is uploaded to S3 at `s3://bucket/rerand/epoch-{E}/party-{P}/public-key`
4. Each rerand server downloads the other two parties' public keys from S3 (polling until all present)
5. Each derives the same 32-byte `shared_secret` via the BLS12-381 pairing

Only the rerand server needs access to the key. The main server never touches it.

### Keygen is idempotent on restart

When starting an epoch, the rerand server:

1. Best-effort cleanup: attempts to delete the previous epoch's key from Secrets Manager (covers crash during epoch transition where deletion was skipped)
2. Checks if an epoch-scoped private key already exists in Secrets Manager at `{env}/iris-mpc-db-rerandomization/epoch-{E}/private-key-party-{P}`
3. If yes: loads it, derives the public key, and uploads the public key to S3 if not already present (covers crash-after-SM-write-before-S3-upload)
4. If no: generates a new keypair, saves the private key to Secrets Manager first, then uploads the public key to S3

Secrets Manager is checked first because the private key is written to SM before the public key is uploaded to S3. If we crash between the two writes, on restart we find the key in SM and re-upload to S3.

### Epoch transition

One epoch at a time, no overlap:

1. All three rerand servers finish processing all chunks for epoch E
2. Each server uploads a completion marker: `s3://bucket/rerand/epoch-{E}/party-{P}/complete`
3. Each server polls until all three completion markers exist
4. Keys for epoch E are deleted from Secrets Manager — old secret is destroyed, old shares (overwritten in live DB) are unrecoverable
5. Epoch E+1 begins: create/publish `manifest.json`, keygen, derive new `shared_secret`, start processing

Old S3 markers under `epoch-{E}/` are left in place (no active cleanup). Use S3 lifecycle policies to reap old epoch prefixes after a retention period.

On restart mid-epoch: private key is still in SM, public keys and markers are still in S3, `rerand_progress` table tells you the current epoch and which chunk to resume from. Re-derive `shared_secret`, continue.

## S3 Coordination Bus

All cross-party coordination uses S3 markers in a shared bucket. Each party writes to its own prefixed paths. Marker layout:

```
s3://bucket/rerand/epoch-{E}/party-{P}/public-key              # public key for DH
s3://bucket/rerand/epoch-{E}/party-{P}/max-id                  # party P watermark for manifest (MAX(id))
s3://bucket/rerand/epoch-{E}/party-{P}/manifest.json           # epoch chunking manifest (party 0 writes, others read)
s3://bucket/rerand/epoch-{E}/party-{P}/chunk-{K}/staged        # chunk K staging committed
s3://bucket/rerand/epoch-{E}/party-{P}/chunk-{K}/version-hash  # 32-byte blake3 hash of version map (fast-path comparison)
s3://bucket/rerand/epoch-{E}/party-{P}/chunk-{K}/version-map   # chunk K [(id, version_id)] pairs (downloaded only on hash mismatch)
s3://bucket/rerand/epoch-{E}/party-{P}/complete                # epoch E fully done
```

Coordination is polling-based: a rerand server checks for peer markers by listing the S3 prefix. A few seconds of polling latency is fine for background work. All polling loops have a 30-minute timeout to surface permanently stuck peers.

Authentication: the shared bucket uses IAM prefix policies to scope write access per party. Each party can only write to `s3://bucket/rerand/epoch-*/party-{P}/*`. All parties can read/list the full `s3://bucket/rerand/epoch-{E}/` prefix to observe peer markers. The manifest is written by the designated writer (party 0) under its own prefix (`party-0/manifest.json`) and is read-only for others.

## Schema Changes

### New column on `irises`

```sql
ALTER TABLE irises ADD COLUMN rerand_epoch INTEGER NOT NULL DEFAULT 0;
```

### Modified `increment_version_id` trigger

```sql
CREATE OR REPLACE FUNCTION increment_version_id()
RETURNS TRIGGER AS $$
BEGIN
    IF (OLD.left_code IS DISTINCT FROM NEW.left_code OR
        OLD.left_mask IS DISTINCT FROM NEW.left_mask OR
        OLD.right_code IS DISTINCT FROM NEW.right_code OR
        OLD.right_mask IS DISTINCT FROM NEW.right_mask)
       AND NEW.rerand_epoch IS NOT DISTINCT FROM OLD.rerand_epoch THEN
        NEW.version_id = COALESCE(OLD.version_id, 0) + 1;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

When `rerand_epoch` changes (rerandomization), share data changes but `version_id` stays the same. When `rerand_epoch` stays the same (user-facing modification), `version_id` bumps as before.

### Staging schema

Each party has a staging schema (`{live_schema}_rerand_staging`), created automatically by a migration that derives the name from `current_schema()`:

```sql
CREATE TABLE irises (
    epoch                INTEGER NOT NULL,
    id                   BIGINT NOT NULL,
    chunk_id             INTEGER NOT NULL,
    left_code            BYTEA,
    left_mask            BYTEA,
    right_code           BYTEA,
    right_mask           BYTEA,
    original_version_id  SMALLINT,
    rerand_epoch         INTEGER,
    PRIMARY KEY (epoch, id)
);
CREATE INDEX idx_staging_irises_epoch_chunk ON irises (epoch, chunk_id);
```

### Coordination table

A `rerand_progress` table in each party's DB:

```sql
CREATE TABLE rerand_progress (
    epoch           INTEGER NOT NULL,
    chunk_id        INTEGER NOT NULL,
    staging_written BOOLEAN NOT NULL DEFAULT FALSE,
    all_confirmed   BOOLEAN NOT NULL DEFAULT FALSE,
    live_applied    BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (epoch, chunk_id)
);
```

Chunk ranges are derived from the manifest (`chunk_size`, `max_id_inclusive`) and `chunk_id`, so they are not stored here.

Lifecycle: `staging_written` → `all_confirmed` → `live_applied`.

## Flow

### Step 1: Rerand Server (per party, separate process)

Runs continuously:

1. Determine the active epoch E (uses local `rerand_progress` as start hint, then scans S3 for the highest epoch with a manifest but without all three `complete` markers).
2. Derive `shared_secret` for epoch E (keygen or resume — see above)
3. Pick next chunk range `[start, end)` for chunk K from the manifest
4. **Stage**: delete any partial staging data for this chunk (crash recovery clean slate), read entries from live schema recording each entry's `version_id`, rerandomize shares using `BLAKE3(shared_secret || iris_id)` XOF, write to staging schema with `epoch = E`, `original_version_id`, `chunk_id = K`, and `rerand_epoch = E + 1`
5. Set `staging_written = TRUE` in local `rerand_progress` for `(epoch = E, chunk_id = K)`
6. Upload version map `[(id, original_version_id)]` for the chunk to S3: `s3://bucket/rerand/epoch-{E}/party-{P}/chunk-{K}/version-map`
7. Upload S3 staged marker: `s3://bucket/rerand/epoch-{E}/party-{P}/chunk-{K}/staged`
8. Poll S3 until all 3 party staged markers exist for chunk K
9. Set `all_confirmed = TRUE` in local `rerand_progress` for `(epoch = E, chunk_id = K)`
10. **Modification fence and Apply**:
    a. Download all 3 parties' version maps for chunk K. Compute cross-party disagreement set: IDs where any party captured a different `original_version_id`.
    b. **Fenced Apply Transaction**: open a single transaction to check local divergence and apply safely:
       ```sql
       BEGIN;
       -- Block the main server from writing modifications
       SELECT pg_advisory_xact_lock(RERAND_MODIFY_LOCK);

       -- Query local divergences (modifications that landed after staging)
       -- ... compute skip set = cross-party disagreements ∪ local divergences ...
       -- Delete skip set from staging schema

       -- Proceed to apply
       SELECT pg_advisory_xact_lock(RERAND_APPLY_LOCK);
       
       UPDATE irises SET
         left_code = staging.left_code,
         left_mask = staging.left_mask,
         right_code = staging.right_code,
         right_mask = staging.right_mask,
         rerand_epoch = staging.rerand_epoch
       FROM staging_schema.irises AS staging
       WHERE irises.id = staging.id
         AND staging.epoch = E
         AND staging.chunk_id = K
         AND irises.version_id = staging.original_version_id;
         
       DELETE FROM staging_schema.irises WHERE epoch = E AND chunk_id = K;
       UPDATE rerand_progress SET live_applied = TRUE WHERE epoch = E AND chunk_id = K;
       COMMIT;  -- Both RERAND_MODIFY_LOCK and RERAND_APPLY_LOCK released here
       ```
11. Proceed to next chunk (or start epoch transition if all chunks done).

**Crash recovery for staging**: if the process crashes mid-staging, `staging_written` is still `FALSE`. On restart, the code re-enters the staging block and deletes any partial rows before re-reading. This ensures all staging rows come from one read pass (no mixed-snapshot version_ids). Inserts use `ON CONFLICT (epoch, id) DO NOTHING` as a safety net.

**Crash recovery for S3 upload**: the S3 staged marker upload is outside the `if !staging_written` block. If the process crashes after `set_staging_written` but before the S3 upload, the marker is re-uploaded on restart (idempotent PUT).

### Step 2: Main Server Startup (minimal changes)

At startup, before `load_iris_db`:

1. **Existing**: modification sync (`sync_modifications`) — all parties catch up on modifications, producing identical `version_id` values. This transaction acquires `pg_advisory_xact_lock(RERAND_MODIFY_LOCK)` to serialize with rerand applies.
2. **New**: rerand sync validation — parties exchange a compact rerand watermark during the existing startup sync (`SyncState` exchange):
   - Each party computes `(epoch, max_confirmed_chunk)` from its local `rerand_progress` table: the active epoch E and the highest `chunk_id` where `all_confirmed = TRUE`. Returns `None` if the `rerand_progress` table doesn't exist yet (rolling deploy before migration); real DB errors propagate as `Err`.
   - Each party sends this single `(epoch, max_confirmed_chunk)` pair as part of `SyncState`.
   - The startup validator checks two invariants: epoch gap ≤ 1 and chunk gap ≤ 1 (within the same epoch). If either is violated, startup fails (indicates DB corruption).
   - If a peer is at most 1 chunk or 1 epoch ahead, this is within protocol tolerance. **Startup does NOT apply any chunks.** The rerand worker is responsible for catch-up through the full modification-fence path. If the node is behind, the rerand worker must run and complete the pending chunk before the next startup succeeds.
3. **New**: lock-first startup readiness loop:
   - Acquire `pg_advisory_lock(RERAND_APPLY_LOCK)` on a dedicated connection.
   - While the lock is held (applies frozen), verify local DB readiness: no `all_confirmed=TRUE AND live_applied=FALSE` rows remain, and local applied watermark has reached the startup target from step 2.
   - Special case: if startup target is `(epoch = E, max_confirmed_chunk = -1)`, no chunk is confirmed in epoch E yet, so startup does not wait for an apply in epoch E.
   - If behind, release the lock and retry after a short sleep (lets rerand worker apply the pending chunk).
   - If local watermark has already advanced past the startup target, fail startup (snapshot stale; restart and resync).
   - Once ready, keep the lock held through step 4. This prevents the rerand loop from applying new chunks while the DB snapshot is being read.
4. **Existing**: `load_iris_db` — loads from live DB into GPU memory. The advisory lock is still held, so the rerand server cannot apply new chunks while the DB is being read into memory.
5. Release the advisory lock: `SELECT pg_advisory_unlock(RERAND_APPLY_LOCK)` on the dedicated connection, then drop the connection.

**Why startup does not apply chunks**: every chunk apply must go through the modification fence (cross-party version-map exchange + local divergence check under `RERAND_MODIFY_LOCK`). The startup path has no access to the S3 coordination bus and cannot perform the fence. Applying unfenced chunks at startup would create the same cross-party share divergence the fence was designed to prevent.

**Rollout note**: if the rerand tables haven't been migrated yet, `build_rerand_sync_state` returns `Ok(None)` for missing table. The validation is skipped and startup proceeds without error.

### Epoch and chunk desync safety checks

The startup sync validates two invariants derived from the protocol's synchronization barriers:

- **Epoch gap ≤ 1**: epochs transition via a 3-party S3 barrier (`complete` markers), so no peer can be more than 1 epoch ahead. A gap > 1 is fatal.
- **Chunk gap ≤ 1** (within the same epoch): the per-chunk S3 barrier (`staged` markers) prevents any peer from confirming more than 1 chunk ahead. A gap > 1 is fatal.

If either check fails, the main server refuses to start. This catches DB corruption, manual interference, or bugs in the rerand server early, before any data is loaded into memory.

### Advisory locks

Three advisory lock keys are used:

- **`RERAND_APPLY_LOCK`** — serializes chunk applies with `load_iris_db`. Used as `pg_advisory_xact_lock` inside `apply_staging_chunk`'s transaction (auto-released on commit/rollback/drop), and as session-level `pg_advisory_lock` during startup to hold through `load_iris_db`.
- **`RERAND_MODIFY_LOCK`** — serializes modification writes with the rerand modification fence. The rerand server acquires it (`pg_advisory_xact_lock`) at the start of its unified fence+apply transaction (Step 1.10) to hold through the fence check and apply window. The main server acquires it (`pg_advisory_xact_lock`) inside its modification transaction to prevent writes during the fence window.

**Why `pg_advisory_xact_lock` for applies and modifications**: session-level locks are tied to a connection. If a process is killed while holding a session-level lock on a pooled connection, the connection may be returned to the pool with the lock still held, blocking future acquirers indefinitely. Transaction-level locks avoid this: when the connection is dropped, the transaction rolls back and the lock is released automatically. This is why the entire modification fence and apply step was combined into a single transaction.

## Conflict Resolution: Rerandomization vs Modifications

### The problem

Modifications (reauthentications, deletions) propagate asynchronously to each party via independent SQS queues. During continuous rerandomization, a modification can land on some parties but not others between the time different parties stage a chunk. Without protection, this causes cross-party share divergence: different parties apply the rerand to different underlying shares, breaking the MPC invariant that all 3 parties' shares reconstruct to the same plaintext.

### The modification fence

The modification fence ensures all parties agree on which rows to skip before applying a chunk. It has two components:

1. **Cross-party version-map exchange** (Steps 1.6–1.8): after staging, each party uploads its `[(id, original_version_id)]` map for the chunk to S3, along with a 32-byte blake3 hash of the map. After the S3 barrier, each party first downloads only the 3 hashes (96 bytes total). If all hashes match, the maps are identical and the cross-party disagreement set is empty (fast path — no full map download needed). If any hash differs, the full maps are downloaded and diffed to compute the exact set of IDs where any party captured a different `original_version_id` (slow path). This catches modifications that arrived on some parties before staging but not others. In practice, disagreements are rare (only when a modification races with the staging window), so the fast path runs ~100% of the time.

2. **Local divergence check under lock**: as part of the single apply transaction, the rerand server acquires `pg_advisory_xact_lock(RERAND_MODIFY_LOCK)` (blocking the main server from writing modifications), then queries for IDs where `staging.original_version_id ≠ irises.version_id`. This catches modifications that arrived after staging but before apply. The lock ensures no new modifications can land during the check + apply window.

The union of both sets is deleted from staging before apply. All parties compute the same skip set (the cross-party exchange is deterministic, and the local check is under lock), so the apply produces consistent results across all parties.

### Why the optimistic lock is still needed

The skip-set deletion removes divergent rows from staging before apply. The apply SQL still includes `WHERE irises.version_id = staging.original_version_id` as a final safety net — if a row somehow slipped through the fence (e.g., a modification landed in the narrow window between the local check and the apply within the same lock), the optimistic lock catches it. On its own the optimistic lock does NOT guarantee cross-party consistency (different parties can have different live `version_id` values), but combined with the fence it serves as defense-in-depth.

### Why `rerand_epoch` and the trigger are kept

Without the trigger change, the rerand apply would bump `version_id` (because share data changes). This is not a safety issue — the optimistic lock works correctly either way — but it inflates `version_id` by 1 per epoch per row. Since `version_id` is `SMALLINT` (max 32767), this limits the total number of rerandomizations + modifications before overflow. The trigger keeps `version_id` as a pure user-modification counter, preserving the full range for actual reauthentications.

## Chunking

Chunk boundaries must be identical across parties for chunk K to be meaningful. Define them via an epoch manifest object in S3:

- `s3://bucket/rerand/epoch-{E}/party-0/manifest.json`: `{ epoch: E, chunk_size: N, max_id_inclusive: M }`
- Party 0 writes the manifest once at epoch start under its own prefix (IAM-compliant); other parties poll until it exists and treat it as immutable.
- **Watermark sync**: before the manifest is written, each party P uploads its local watermark `max_id_party_P = SELECT MAX(id) FROM irises` to `s3://bucket/rerand/epoch-{E}/party-{P}/max-id`.
- The manifest writer waits until all three `max-id` markers exist, then sets `max_id_inclusive` as:
  - `M = min(max_id_party_0, max_id_party_1, max_id_party_2) - safety_buffer_ids`
  - `safety_buffer_ids` is configurable (default 0 or one chunk) to avoid rerandomizing the "tip" where replication/ingest lag could differ across parties.
- New inserts with `id > M` are left for a future epoch.
- Chunk K corresponds to `[start, end)` where `start = 1 + K * N` and `end = min(start + N, M + 1)`.

A configurable delay (`--chunk-delay`, default e.g. 5s) is inserted between chunks to avoid sustained DB load. The rerand server should not stress the live DB with continuous writes — the delay spreads the I/O over time. The delay, chunk size, and number of parallel DB connections should all be configurable via CLI flags or environment variables.
