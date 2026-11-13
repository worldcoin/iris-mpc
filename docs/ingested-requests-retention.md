# DB-backed ingestion and replay

The durable completion watermark C records processing progress independently of
retained journal rows. It does not record successful result publication.

Admission normalizes the SNS envelope sequence number and locks C with
`FOR SHARE` in the same transaction as the journal check or insertion.
Completion and startup recovery take `FOR UPDATE` on C before journal writes,
so admission cannot insert using a stale checkpoint. SQS receipt deletion
follows successful commit.
Admission has a 30-second overall deadline and a transaction-local statement
timeout that preserves any shorter database limit. Timeout leaves the receipt
unacknowledged and follows the existing ingest error path.

- Above C, admission preserves primary-key deduplication.
- At or below C, a retained key confirms an existing journal delivery and is
  acknowledged as a duplicate.
- At or below C with no retained key, admission fails without inserting or
  deleting the SQS receipt. This includes duplicates whose keys were reaped:
  C cannot distinguish them from delayed first deliveries.

Existing malformed-envelope and invalid-sequence discard handling is unchanged.

The existing batch agreement and recovery design assumes a complete ordered
prefix of SNS input. Every input stream covered by one C must preserve that
order; independently ordered message groups cannot establish a single prefix.
[SNS FIFO ordering is per message group](https://docs.aws.amazon.com/sns/latest/dg/fifo-topic-message-ordering.html).
This guard does not validate deployed queue/group configuration; verify that
configuration satisfies the prefix contract before enabling retention.

Ordinary source-queue retries remain supported while their journal keys exist.
Blind DLQ redrive, archive replay, manual insertion behind C, and switching SNS
sources without a recovery plan are unsupported. In particular,
[DLQ redrive can interleave old messages with new traffic](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-configure-dead-letter-queue-redrive.html).
A missing old key produces an ingest error, follows the existing retry/backoff
path, and may eventually move to the configured DLQ. The rejection can block its
FIFO group until resolution or DLQ movement. Inspect
`db_ingest_errors` and the ingress error logs, then reconcile delivery and
processing across all parties before replay. Do not lower C or insert old
requests into ordinary pending work to bypass the rejection.
An exhausted ingress error also blocks this process from successful drain,
even if the message later moves to the DLQ; see [CPU drain recovery](cpu-cell-drain.md).

Deploy guarded admission to every writer before enabling journal deletion.
After keys have been deleted, rolling back to unguarded admission can reprocess
old requests; disabling further cleanup does not restore those keys.
This change does not enable retention jobs.

## Prepared steady-state jobs

`iris-ampc-ingested-requests-retention` has separate stage/prod common values
and per-party values on the `ampc-hnsw` clusters. Both environments start
suspended and with `RETENTION__DRY_RUN=true`; the existing reset jobs stay
suspended. Registration belongs in the corresponding cluster-apps catalogs,
using `common-cronjob` 1.11.0 in namespace `ampc-hnsw`.

The provisional grace is **30 days since `persisted_at`**, not receipt time.
Only persisted rows with `is_valid IS TRUE`, a normalized 40-digit sequence,
and sequence ≤ the local durable C qualify. Invalid or NULL validity and
malformed/NULL sequences are retained for investigation. C must be a JSON
string containing a normalized sequence within the u128 range; absent, NULL,
or malformed C retains every row. Comparisons use fixed-width text with C
collation and no numeric casts. The only target is `ingested_requests`;
`cpu_cell_handoffs` is preserved.

The daily 04:30 UTC schedule forbids overlap, uses 2,000-row batches with 250 ms
pauses, and caps SQL statements at 30 seconds. The chart has no Job deadline
field: verified image-provided GNU `timeout` sends TERM after 1,800 seconds and
KILL after another 30 seconds. `restartPolicy: Never` and `backoffLimit: 0`
prevent retries. Scheduling/image-pull delays are not covered by this process
bound; a Pending job can block the next run under `Forbid` and needs operator
attention. Disabling future runs does not cancel an existing Job.

This is configuration preparation, not approval to activate. POP-4496 must
approve grace and invalid-row policy; POP-4534 must validate guarded admission,
queue ordering/replay constraints, all-party dry-run eligibility, resource
impact, and monitoring before any unsuspend or destructive-mode change.
