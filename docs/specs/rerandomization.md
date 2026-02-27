# Continuous Rerandomization Plan

## Overview

Replaces the existing, one-off rerandomization protocol by a continuous, online process that rerandomizes shares while the system is running. No downtime or restart required.

Key design decision: in-memory shares are less likely to be exfiltrated, so only the DB (at-rest persistence) is rerandomized. The actor is completely unmodified. The rerand server handles everything, writing to a staging schema and then copying to live once all parties confirm.

## Architecture

1. **Rerand Server** (modified `iris-mpc-bins/bin/iris-mpc-upgrade/rerandomize_db.rs`, separate process, one per party) — rerandomizes shares, writes to staging, coordinates with peers via S3 markers, copies confirmed chunks to live DB. Replaces the existing one-off `RerandomizeDb` subcommand with a new `RerandomizeContinuous` subcommand. Core rerandomization logic in `iris-mpc-upgrade/src/rerandomization.rs` is reused; the new subcommand adds the continuous loop, S3 coordination, and staging management.
2. **Main Server** (existing, minimal changes) — at startup, syncs rerand progress with peers and catches up any missing chunks from staging before loading the DB into memory.

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

1. Checks if an epoch-scoped private key already exists in Secrets Manager at `{env}/iris-mpc-db-rerandomization/epoch-{E}/private-key-party-{P}`
2. If yes: loads it, derives the public key, and uploads the public key to S3 if not already present (covers crash-after-SM-write-before-S3-upload)
3. If no: generates a new keypair, saves the private key to Secrets Manager first, then uploads the public key to S3

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
s3://bucket/rerand/epoch-{E}/party-{P}/public-key            # public key for DH
s3://bucket/rerand/epoch-{E}/party-{P}/max-id                # party P watermark for manifest (MAX(id))
s3://bucket/rerand/epoch-{E}/party-{P}/manifest.json         # epoch chunking manifest (party 0 writes, others read)
s3://bucket/rerand/epoch-{E}/party-{P}/chunk-{K}/staged      # chunk K staging committed
s3://bucket/rerand/epoch-{E}/party-{P}/complete              # epoch E fully done
```

Coordination is polling-based: a rerand server checks for peer markers by listing the S3 prefix. A few seconds of polling latency is fine for background work.

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

Each party has a staging schema (e.g. `SMPC_rerand_staging`) with:

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

1. Determine the active epoch E and load its manifest (the highest epoch with a manifest at `s3://bucket/rerand/epoch-{E}/party-0/manifest.json` but without all three completion markers). If no manifest exists for the next epoch, create it (party 0 only): collect watermarks, compute `max_id_inclusive`, write `manifest.json`.
2. Derive `shared_secret` for epoch E (keygen or resume — see above)
3. Pick next chunk range `[start, end)` for chunk K from the manifest
4. Read entries from live schema, recording each entry's `version_id`
5. Rerandomize shares using `BLAKE3(shared_secret || iris_id)` XOF
6. Write rerandomized shares to staging schema with `epoch = E`, `original_version_id`, `chunk_id = K`, and `rerand_epoch = E + 1`
7. Set `staging_written = TRUE` in local `rerand_progress` for `(epoch = E, chunk_id = K)`
8. Upload S3 marker after staging commit: `s3://bucket/rerand/epoch-{E}/party-{P}/chunk-{K}/staged`
9. Poll S3 until all 3 party markers exist for chunk K
10. Set `all_confirmed = TRUE` in local `rerand_progress` for `(epoch = E, chunk_id = K)`
11. Acquire `pg_advisory_lock(RERAND_APPLY_LOCK)` on a dedicated connection, then copy from staging to live DB, delete staging, and mark applied — all in one transaction (scoped to epoch and chunk):
    ```sql
    SELECT pg_advisory_lock(RERAND_APPLY_LOCK);   -- on dedicated connection
    BEGIN;
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
    COMMIT;
    SELECT pg_advisory_unlock(RERAND_APPLY_LOCK);  -- release after commit
    ```
12. Proceed to next chunk (or start epoch transition if all chunks done)

### Step 2: Main Server Startup (minimal changes)

At startup, before `load_iris_db`:

1. **Existing**: modification sync (`sync_modifications`) — all parties catch up on modifications, producing identical `version_id` values
2. **New**: rerand sync — parties exchange a compact rerand watermark during the existing startup sync (`SyncState` exchange):
   - Each party computes `(epoch, max_confirmed_chunk)` from its local `rerand_progress` table: the active epoch E and the highest `chunk_id` where `all_confirmed = TRUE`.
   - Each party sends this single `(epoch, max_confirmed_chunk)` pair as part of `SyncState`.
   - Each party checks whether any peer is exactly 1 confirmed chunk ahead (within the same epoch, or has moved to the next epoch). If so, it applies that single chunk (`my_max_confirmed + 1`) from staging to the live DB.
   - **Why at most 1 chunk**: the rerand loop has a strict per-chunk synchronization barrier — a node cannot stage chunk K+1 until all three parties have confirmed chunk K via S3 markers. Therefore it is impossible for any peer to be more than 1 confirmed chunk ahead. The implementation enforces this with a fatal bail if the gap exceeds 1 (indicates DB corruption).
   - **Why `max` across peers**: `all_confirmed = TRUE` at any party means that party observed all three S3 `staged` markers, which means all three parties successfully committed the chunk to their staging schemas. A slower party may not have polled S3 yet, but its staging data is already there.
   - Edge case: if all parties report the same `max_confirmed_chunk`, there is nothing to catch up and the step is skipped.
3. **New (DB-only catch-up)**: acquire `pg_advisory_lock(RERAND_APPLY_LOCK)` on a dedicated connection. If step 2 identified a chunk to apply, run the same apply transaction as Step 1.11. **Keep the lock held** through step 4.
4. **Existing**: `load_iris_db` — loads from live DB into GPU memory. The advisory lock is still held, so the rerand server cannot apply new chunks while the DB is being read into memory.
5. Release the advisory lock: `SELECT pg_advisory_unlock(RERAND_APPLY_LOCK)` on the dedicated connection, then drop the connection.

### Epoch and chunk desync safety checks

The startup sync validates two invariants derived from the protocol's synchronization barriers:

- **Epoch gap ≤ 1**: epochs transition via a 3-party S3 barrier (`complete` markers), so no peer can be more than 1 epoch ahead. A gap > 1 is fatal.
- **Chunk gap ≤ 1** (within the same epoch): the per-chunk S3 barrier (`staged` markers) prevents any peer from confirming more than 1 chunk ahead. A gap > 1 is fatal.

If either check fails, the main server refuses to start. This catches DB corruption, manual interference, or bugs in the rerand server early, before any data is loaded into memory.

### Advisory lock: startup vs rerand server concurrency

Both the rerand server (Step 1.11) and the main server startup (Steps 2.3–2.4) acquire `pg_advisory_lock(RERAND_APPLY_LOCK)` before applying chunks. This ensures:

- Only one process applies chunks at a time (no interleaving).
- The main server holds the lock from catch-up through `load_iris_db`, so the rerand server cannot sneak in applies between catch-up and memory load.
- If either process crashes, the connection drops and Postgres automatically releases the session-level lock. No stale locks.

**Implementation with connection pools (sqlx)**: session-level advisory locks are tied to a specific Postgres connection. When using a connection pool, acquire a **dedicated connection** (`pool.acquire()`) and hold it (do not drop/return it) for the entire lock window. The catch-up queries and `load_iris_db` can use the pool normally — the dedicated connection just sits idle holding the lock. Release with `pg_advisory_unlock(...)` on the same connection after `load_iris_db` completes, then drop the connection.

```rust
let mut lock_conn = pool.acquire().await?;
sqlx::query("SELECT pg_advisory_lock($1)")
    .bind(RERAND_APPLY_LOCK)
    .execute(&mut *lock_conn).await?;

if let Some((epoch, chunk_id)) = catchup_chunk {
    apply_staging_chunk(&pool, epoch, chunk_id).await?;
}
load_iris_db(&pool).await?;

sqlx::query("SELECT pg_advisory_unlock($1)")
    .bind(RERAND_APPLY_LOCK)
    .execute(&mut *lock_conn).await?;
drop(lock_conn);
```

### Why modification sync before rerand sync matters

Modification sync ensures all parties have the same `version_id` values before the rerand staging copy runs. This guarantees the optimistic lock (`WHERE version_id = original_version_id`) produces the same skip set on all parties — the same entries are updated, the same entries are skipped.

## Conflict Resolution: Rerandomization vs Modifications

### Why the optimistic lock is needed

The rerand server reads entry X at time T with `version_id = V`. A modification (reauth/deletion) may happen later, bumping `version_id` to V+1. The staging still has `original_version_id = V`. The optimistic lock prevents overwriting the modification:

```sql
UPDATE irises SET ... WHERE version_id = original_version_id;
-- V ≠ V+1 → entry X skipped
```

### Why `rerand_epoch` and the trigger are needed

Without the trigger change, the staging copy would bump `version_id` (because share data changed). The trigger change keeps `version_id` as a pure "user-facing modification counter," separate from rerandomization.

## Chunking

Chunk boundaries must be identical across parties for chunk K to be meaningful. Define them via an epoch manifest object in S3:

- `s3://bucket/rerand/epoch-{E}/party-0/manifest.json`: `{ epoch: E, chunk_size: N, max_id_inclusive: M }`
- Party 0 writes the manifest once at epoch start under its own prefix (IAM-compliant); other parties poll until it exists and treat it as immutable.
- **Watermark sync**: before the manifest is written, each party P uploads its local watermark `max_id_party_P = SELECT MAX(id) FROM irises` to `s3://bucket/rerand/epoch-{E}/party-{P}/max-id`.
- The manifest writer waits until all three `max-id` markers exist, then sets `max_id_inclusive` as:
  - `M = min(max_id_party_0, max_id_party_1, max_id_party_2) - safety_buffer_ids`
  - `safety_buffer_ids` is configurable (default 0 or one chunk) to avoid rerandomizing the “tip” where replication/ingest lag could differ across parties.
- New inserts with `id > M` are left for a future epoch.
- Chunk K corresponds to `[start, end)` where `start = 1 + K * N` and `end = min(start + N, M + 1)`.

A configurable delay (`--chunk-delay`, default e.g. 5s) is inserted between chunks to avoid sustained DB load. The rerand server should not stress the live DB with continuous writes — the delay spreads the I/O over time. The delay, chunk size, and number of parallel DB connections should all be configurable via CLI flags or environment variables.