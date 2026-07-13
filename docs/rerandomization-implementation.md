# Continuous rerandomization implementation

This is the canonical specification for continuous rerandomization. The design
optimizes for a small set of database invariants that remain true under worker
crashes, concurrent semantic writes, different party progress, and stale S3
snapshots.

## 1. Safety model

The system has three MPC parties and two authoritative stores per party (GPU
and HNSW). A database row contains its normal share bytes, semantic
`version_id`, collision-resistant `semantic_id`, and `rerand_epoch`.

The implementation maintains these invariants:

1. Epoch `0` means an ordinary application-produced share.
2. Every semantic change atomically increments `version_id`, replaces
   `semantic_id` with a fresh UUID, and resets `rerand_epoch` to `0`.
3. A rerandomization changes only share bytes and `rerand_epoch`, except that
   the bootstrap pass fills a missing `semantic_id`. It is accepted only
   through the restricted CAS path matching
   `(id, version_id, semantic_id, old_epoch)`.
4. Servers normalize every loaded row to epoch `0` before using it. Raw rows
   from different epochs never enter MPC.
5. S3 is a cache. Aurora metadata and the database identity remain authoritative.
6. At most one new epoch is globally active. No store begins epoch `e + 1`
   before all six stores have completed epoch `e`.
7. A referenced seed is commitment-verified before use. Missing live data
   seeds refuse startup; missing cache-only seeds cause Aurora fallback.
8. Epoch seeds are retained. This implementation has no automatic deletion
   path, so incomplete or unavailable coordination evidence cannot make a
   persisted or cached row undecodable.
9. A private three-party consistency check passes before readiness and at
   each configured recurring boundary.

The design does not require three databases to update a row atomically. A
party may safely be at a different persisted epoch because load-time
normalization removes the local offset before MPC.

This is representation churn for persisted shares, not proactive security
against a mobile adversary. All parties derive the same retained epoch seed,
so a party that knows that seed can compute every party's public-point offset.
Secrets Manager and deletion-free seed retention are therefore inside this
feature's threat boundary; proactive refresh would require private per-party
zero-share contributions and is a separate protocol.

## 2. Row annotation and write authority

`irises.rerand_epoch` is a non-negative integer with default `0`.
`irises.semantic_id` is a random UUID assigned by the database, not accepted
from normal callers. The migration initially leaves this column nullable so
adding it does not rewrite the existing iris table. Inserts and semantic writes
always assign it. The first rerandomization pass backfills legacy NULLs, and
the database refuses to complete that pass while any NULL remains.

The database trigger classifies a write from the old and new share bytes:

- If semantic bytes changed through a normal writer, it increments
  `version_id`, rotates `semantic_id`, updates semantic modification time, and
  forces epoch `0`.
- If the epoch changes through the restricted rerandomization function, it
  requires the expected old version and epoch, moves strictly forward, and
  leaves semantic metadata unchanged, except that the initial pass assigns a
  UUID to a legacy NULL `semantic_id`.
- Direct or backward epoch changes are rejected.

The control row binds a schema to an immutable environment, coordination ID,
party, store kind, globally unique store ID, and dedicated rerandomization
login. Serving and sweep processes must match every identity field; the S3
exporter matches the globally unique store ID. The dedicated login receives
only `SELECT` on iris rows and execution of the schema-bound pass/CAS
functions; initialization rejects inherited mutation,
owner, or schema-creation privileges.

After the migration, the table owner initializes each store exactly once:

```sql
SELECT initialize_rerand_store(
  store_id, environment, coordination_id, party_id::smallint,
  'gpu' /* or 'hnsw' */, writer_role::name
);
```

The sweeper uses that dedicated login. Its `RERAND_EXPECTED_STORE_REGISTRY`
is a JSON array of exactly six `{party_id, store_kind, store_id, writer_role}`
bindings.

## 3. Offset construction

Each positive epoch has one verified 32-byte seed. Domain-separated XOF input
binds the protocol version, epoch, iris ID, component, and coefficient. The
three party evaluations form a sharing of zero, so adding the complete epoch
offset preserves the underlying iris.

Epoch offsets are absolute. Retargeting a row from epoch `a` to `b` subtracts
the offset for `a` and adds the offset for `b`; epoch `0` has the zero offset.
Therefore retries are deterministic and a cached older rerandomization remains
semantically usable when its version is current and its seed is still present.

## 4. Serialized epochs and the sweeper

Completion records are immutable and keyed by environment, coordination ID,
epoch, party, and store kind. Their content binds the store ID, writer role,
coordination ID, offset generation, and seed commitment.
The sweeper is configured with the exact six-entry store registry; every marker
must match its expected registry entry and the locally verified canonical epoch
commitment. Store IDs must be globally unique and must not be reused for a new
deployment. `RERAND_COORDINATION_ID` is also globally unique and never reused;
a replacement deployment in the same environment must use a fresh value. It
is persisted in every store identity and namespaces all S3 coordination
objects and Secrets Manager seed/scalar IDs.

For one store, a pass:

1. Validates the store identity and the exact global predecessor completion set.
2. Acquires a schema-scoped session advisory lock; a second worker fails.
3. Selects the single allowed target epoch and durably binds it to the verified
   seed commitment before changing a row. Resume requires the same binding.
4. Loads and commitment-verifies the source and target seeds needed by the
   current row inventory.
5. Streams rows in primary-key order from the durable cursor.
6. Computes retargeted bytes outside the write transaction.
7. Applies bounded batches through
   `(id, version_id, semantic_id, old_epoch)` CAS.
8. Advances the cursor only after the batch commits.
9. Performs a final inventory check, advances the monotonic completion state,
   clears the cursor, and publishes the immutable store completion record.

A CAS miss means a concurrent semantic write won. That write is truthfully at
epoch `0`; it is not overwritten and a later pass rerandomizes it. A crash may
repeat computation, but cannot double-apply an epoch.

Legacy exports use the same `(1381126734, schema OID)` advisory lock as store
initialization and the sweeper. A complete legacy export, or an incremental
export including its complete fallback, takes a session lock on a dedicated
PostgreSQL connection before its first rerandomization-state check and holds it
until marker publication or return. Lock acquisition is a fail-fast try-lock:
an active pass or another legacy export makes the export fail closed.
Initialization takes the blocking transaction-level form of the same lock, so
its identity transition happens only before an export starts or after the
export has safely published and released its session. This closes both races
in which initialization or a pass could otherwise occur between the
exporter's initial check and legacy marker. Safe content-addressed complete
snapshots do not take this lock; their per-record metadata and authoritative
inventory reconciliation make mutable legacy keys unnecessary.

## 5. Seed creation and retention

Within one immutable coordination ID, parties publish immutable public material
and commitments for one epoch. Before any store targets that epoch, all parties
must derive the same commitment.
Objects are create-only: an existing object with different bytes is an error.

Seeds and their private DH material are create-only and are not automatically
deleted. This deliberately trades storage growth for a simple hard guarantee:
no partial confirmation, unavailable party, malformed marker, permission
failure, or AWS failure can delete a key. A future retention protocol would be
a separate change with its own proof and rollout.

If a required seed is nevertheless unavailable, a cached row is replaced from
Aurora during reconciliation. If authoritative Aurora itself references the
missing seed, startup refuses and requires offline recovery.

Serving only reads coordination commitments; it never creates or repairs them.
Aurora positive epochs must be the control row's last-completed or active
epoch, and each loaded seed must match both all three immutable S3 commitments
and the corresponding commitment persisted in the control row. Any mismatch
refuses startup before normalization.

## 6. Fast S3 loading

S3 accelerates transfer of large share blobs but never decides current data.
Only complete immutable snapshots may publish the safe completion pointer.
Incremental or partial exports cannot replace it.

The content-addressed manifest binds:

- format and protocol version;
- source store identity;
- immutable randomized generation in each chunk key;
- exact row count and ID boundary; and
- ordered chunk/block ranges, sizes, and cryptographic hashes.

Every record carries `id`, `version_id`, `rerand_epoch`, and the 16 raw bytes of
`semantic_id`. The loader rejects malformed manifests, wrong stores,
overlapping ranges, missing or duplicate IDs, truncated blocks, and hash
failures. It then opens a stable Aurora snapshot and streams the compact
authoritative `(id, version_id, semantic_id, rerand_epoch)` inventory. The
snapshot's own count/min/max must describe the exact contiguous database; an
earlier caller count cannot authorize loading only a prefix.

For each authoritative ID:

- A cached row with the same semantic version and `semantic_id`, plus an
  available cached-epoch seed, is normalized to epoch `0` and used.
- A version or semantic-ID mismatch, missing authoritative semantic ID, cache
  miss, or unavailable cached-epoch seed fetches the complete row from the
  same Aurora snapshot.
- Cache IDs beyond Aurora's authoritative contiguous boundary are ignored;
  non-contiguous authoritative inventory is an error.

The Aurora row's seed must be available. Invalid cache state falls back to a
full Aurora load rather than partially trusting the cache. This scans only
small metadata for cache hits; it does not transfer Aurora share blobs for the
whole database. Large S3 chunks are fetched and hashed in ordered ranges of at
most 200 records, preserving the legacy loader's bounded memory profile.

## 7. Full consistency check

The check covers every loaded code and mask coefficient plus a public constant
inventory sentinel keyed by the row ID. Fresh common challenges generate an
independent Galois-ring weight for every `(row, lane, repetition)`. Each party
compresses the complete database directly to one Galois-ring element per
repetition: 48 `u16` coefficients for twelve repetitions. Under the
BLAKE3-XOF-as-PRF assumption, a non-zero error survives one compression with
probability at least `15/16`; twelve independent repetitions therefore bound
the false-negative probability by `2^-48`.

The compressed syndrome is never opened. The parties first add fresh
pairwise-cancelling arithmetic masks, making each local `u16` contribution
uniform to the one peer which receives it. For each coefficient, they then
treat its three masked words as replicated Boolean shares and evaluate a
three-party ABY3 circuit:

1. add the three 16-bit words modulo `2^16` with Boolean full adders;
2. OR-reduce every bit of all 48 results plus one private local-scan-failure
   word; and
3. reconstruct only the final one-bit `nonzero` result.

Replicated AND gates use fresh pre-exchanged pairwise random bits. No iris
syndrome, masked aggregate, carry, or failed-check value is reconstructed, so
the transcript remains private for a one-row database and when the check
fails, under the same honest-but-curious one-corruption model as the serving
protocol. Strict domain-separated frames bind every private exchange to the
fresh challenge and circuit step. The readiness barrier compares inventories
and agrees the challenge; a later local scan failure enters the same private
zero-test instead of leaving peers blocked.

When rerandomization is enabled, the startup check is mandatory and runs after
normalization but before readiness.

For recurring checks:

1. Pre-intake synchronization exchanges a latched `due` bit and uses the
   global OR. With no messages, parties enter an idle maintenance rendezvous;
   with a nonzero agreed count, they all form the same boundary batch.
2. A due boundary withholds the receive permit, so no following batch is
   prefetched.
3. An already-received boundary batch is processed normally and its result is
   made durable.
4. The actor performs the same full scan and private one-bit zero-test.
5. Successful idle checks replace their active rendezvous marker with a
   distinct inactive marker, so a later check at the same idle batch ID cannot
   accept stale peer state.
6. Idle success advances the synchronized batch generation before receive is
   released, so an early finisher waits for every peer to leave the old
   maintenance boundary.
7. Success resets the deadline and resumes receive; failure stops serving.

This ordering needs no SQS acknowledgement redesign and cannot strand an
acknowledged but unprocessed boundary batch. If an approximate SQS count
overestimates available messages, exact batch formation can delay the check;
no actor work is submitted while that boundary is still forming.

## 8. Recovery and restore

Live, in-place, crash-resumable resharing is outside this implementation.
Recovery is an offline staging operation:

1. Quiesce the sources and read stable snapshots.
2. Normalize source rows with verified seeds.
3. Build a fresh unpublished target, preserving exact versions and inventory.
4. Run the full private cross-party check.
5. Promote only after success; otherwise discard the target.

A restored database is never allowed to inherit unrelated S3 progress. Its
immutable store identity, coordination ID, and live epoch inventory are
revalidated. A fresh replacement store set must receive a new coordination ID;
the old value is never recycled even when the environment name is unchanged.
Even if an in-place rollback reuses a numeric `version_id`, its historic
`semantic_id` cannot match a later divergent semantic replay, so the cache
falls back to Aurora. Missing seeds refuse startup rather than guessing or
loading annotated bytes raw.

## 9. Verification and rollout

Focused verification must cover:

- offset cancellation and deterministic retargeting;
- semantic-write versus rerandomization trigger behavior;
- stale-version and stale-epoch CAS misses;
- worker crash/resume and singleton locking;
- immutable/mismatched coordination objects;
- exact six-store epoch gating and permanent seed retention;
- S3 block corruption, wrong identity, inventory gaps, version mismatch, and
  unavailable-cache-seed Aurora fallback;
- Aurora and S3 normalization producing identical epoch-zero rows;
- private one-bit zero-testing for valid, missing, and corrupt one-row states,
  scan failures, and inventory mismatch;
- startup gating and recurring boundary durability.

Deployment ordering is strict because only the rerandomization-aware exporter
and loader enforce the fail-closed guards. Deploy them while rerandomization is
disabled, apply the database migration, initialize the store identities, then
enable rerandomization-aware serving and only after that start the sweepers.
Do not leave an old exporter or server available to restart across store
initialization. Across that transition, rerandomization-aware legacy exporters,
initialization, and newly started sweepers are mutually exclusive through their
shared database lock; an export that cannot acquire it must be retried, not
bypassed. Enable
complete safe exports only after the first pass has completed. A safe export
refuses rows whose `semantic_id` is still NULL. Once a store identity is
initialized, the exporter refuses legacy complete and incremental publication,
and a server with rerandomization disabled refuses to load that store raw.

Rollback must never expose annotated shares to old code. Quiesce serving,
exporters, and sweepers first; verify that no pass is active; normalize every
row to epoch `0`; apply the down migration; deploy the old exporter and servers;
then resume service. Do not revert servers before the down migration. This
implementation does not provide the required durable database normalizer, so
after any positive rerandomization pass rollback is intentionally unsupported
until such a normalizer exists. Before the first positive pass, the normalization
step is only verification that every row is already at epoch `0`.
