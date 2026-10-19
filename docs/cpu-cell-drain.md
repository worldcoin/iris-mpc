# Drain an active CPU cell

Drain is a one-way, operator-authorized transition for the CPU linear-scan
service with `SMPC__DB_BACKED_INGEST=true` and persistence enabled. It does not
activate a candidate, reconcile state, resume processing, or change traffic.
The CPU binary handles SIGUSR1; GPU and HNSW binaries do not implement this
operator operation. Use only one active writer per party/database.

The operator must have Kubernetes RBAC permission to exec into all three active
party pods. There is no mutable HTTP control endpoint. The read-only
`/drain-status` route uses the existing coordination listener and configured
peer addresses. The current three process UUIDs must match the cohort verified
at startup; a restart or stale process blocks a successful drain.

1. Identify the three active CPU linear-scan pods and record their pod UIDs and
   `/health` process UUIDs. Check all are ready, DB-backed ingest is enabled,
   and no rollout or competing operator is changing these pods. Read
   `/drain-status` first: `blocked_reason` must be null on every party. A
   non-null reason means this process cannot prove a safe drain; diagnose and
   recover it before signaling, because SIGUSR1 freezes it permanently.
2. Signal each of those exact pods. The linear-scan container runs its binary
   directly as PID 1. Set the context, namespace, pod and container explicitly:

   ```sh
   kubectl --context "$CONTEXT" --namespace "$NAMESPACE" exec "$POD" \
     -c "$CONTAINER" -- kill -USR1 1
   ```

3. Read all three status reports through the existing coordination port:

   ```sh
   kubectl --context "$CONTEXT" --namespace "$NAMESPACE" exec "$POD" \
     -c "$CONTAINER" -- curl --fail --silent --max-time 5 \
     "http://127.0.0.1:$COORDINATION_PORT/drain-status"
   ```

Success requires all three reports to have `phase: paused`, no error, the
expected party IDs and original process UUIDs, the same `cut`, and identical
`report` values. `/ready` becomes 503 while `/health` remains live; this is an
intentional paused process, not a completed shutdown. Keep the pods alive.

On SIGUSR1, each process freezes new positive batch offers. A batch offered by
all three parties is admitted; the common cut is the minimum of the three
frozen highest positive offers. This completes already-admitted work without
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
missing operator signal, peer identity change, timeout, DB or publication
failure, asymmetric persisted prefix, or failed startup SNS replay blocks
success. An exhausted ingest error makes this process ineligible for a successful
drain even if an unrelated later receive succeeds: this version does not track
recovery of individual ambiguous receipt operations. Normal serving retries
continue until the operator requests drain.

Timeout does not cancel a DB/SNS side-effect future or release its
claims: the process stays failed/paused and must not be used for handoff.

There is no resume command. Recovery uses the existing coordinated restart and
startup recovery path after diagnosing the failure; do not signal SIGTERM as a
substitute for a successful drain or activate a candidate from a failed report.
