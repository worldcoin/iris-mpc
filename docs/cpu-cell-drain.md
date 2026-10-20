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

On `Drain::request()`, each process freezes new positive batch offers. A batch
offered by all three parties is admitted; the common cut is the minimum of
the three frozen highest positive offers. This completes already-admitted work without
admitting a unilateral prefetched next batch. Existing batch offers remain
available until lagging parties finish. The ingest worker stops after settling
its already-started SQS receive, durable insert and receipt deletion. Existing
malformed-envelope/invalid-sequence discard policy is unchanged.

The receiver must stop and every admitted result must commit and finish all
SNS publications before unused claims can be released. Only then does the
process query a fresh persisted watermark, reject unresolved valid rows below
it, and stream a SHA-256 digest/count of the ordered persisted sequence set.
SNS sequence numbers are not dense. Comparing maxima alone is insufficient.
The final full batch digest and batch ID must acknowledge the common cut;
cut zero has no new publication. All three current reports must agree.

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
