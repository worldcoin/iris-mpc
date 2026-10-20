# CPU cell drain primitive

Drain is an in-process, one-way transition for the CPU linear-scan service with
`SMPC__DB_BACKED_INGEST=true` and persistence enabled. `Drain::request()` is the
internal entry point; there is currently no external trigger. The future
approved, authenticated database/peer handoff protocol will call this entry
point. GPU and HNSW binaries do not implement this operation.

The read-only `/drain-status` route uses the existing coordination listener
and configured peer addresses. The current three process UUIDs must match the
cohort verified at startup; a restart or stale process blocks a successful
drain. A non-null `blocked_reason` means the process cannot prove a safe drain.
These HTTP status and UUID checks are not authenticated retirement evidence.

Success requires all three reports to have `phase: paused`, no error, the
expected party IDs and original process UUIDs, the same `cut`, and identical
`report` values. `/ready` becomes 503 while `/health` remains live; this is an
intentional paused process, not a completed shutdown.

The paused state is memory-only. It does not durably retire the cell, fence
restarted processes, reconcile state, activate a candidate, or authorize a
handoff. Durable ownership and startup/recovery guards must be implemented
before production handoff is enabled. Use only one active writer per
party/database.

Drain proceeds in three steps:

1. **Agree on the last batch to finish.** `Drain::request()` stops each party
   from advertising new batches with work. These advertisements are called
   *offers*. The last batch offered by all three parties is the stopping point,
   or *cut*: the minimum of their frozen highest positive offers. For example,
   offers through batches 42, 42 and 41 give a cut of 41. Existing offers stay
   visible so slower parties can finish admitted work; a batch prefetched by
   only one party cannot extend the cut.

2. **Settle work already in flight.** Stop the batch receiver and let any
   started SQS receive finish: save the request in Postgres before deleting
   its receipt. Finish processing every admitted batch, committing its results
   and completing its SNS publications. Only after the workers and publications
   settle can unused database claims be released for later processing.
   Existing malformed-envelope/invalid-sequence discard policy is unchanged.

3. **Verify that all parties completed the same requests.** Each party reads
   a fresh persisted request watermark, W, and rejects unresolved valid requests
   at or below it. W is a request sequence number; the cut is a batch number.
   Matching W alone is insufficient: sets `{10, 20, 30}` and `{10, 30}` both
   end at 30. Each party therefore streams its ordered persisted sequence IDs
   into a SHA-256 hash and counts them; all three reports must agree. Sequence
   numbers need not be consecutive. For a nonzero cut, the final publication
   must be for that batch, with the same full batch digest on every party.
   Cut zero requires no new publication.

`SMPC__CPU_DRAIN_TIMEOUT_SECS` bounds the whole operation (default 300 seconds).
Tune it for the expected in-flight workload before starting the process. A
missing drain request, peer identity change, timeout, DB or publication
failure, asymmetric persisted prefix, or failed startup SNS replay blocks
success. An exhausted ingest error makes this process ineligible for a successful
drain even if an unrelated later receive succeeds: this version does not track
recovery of individual ambiguous receipt operations. Normal serving retries
continue until drain is requested.

Timeout does not cancel a DB/SNS side-effect future or release its
claims: the process stays failed/paused and must not be used for handoff.

There is no resume operation. Recovery currently uses the existing coordinated
restart and startup recovery path after diagnosing the failure. A restart
does not preserve drain state; a successful drain report alone must never
authorize candidate activation.

Unit and component tests invoke `Drain::request()` directly, including
three-party status exchange. Local and stage end-to-end handoff verification
will exercise the approved database/peer protocol once it exists.
