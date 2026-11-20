# CPU handoff record

A and B share one party's Postgres. The handoff table records local progress;
existing peer coordination establishes agreement across the three parties.
This is a planned handoff between cooperating binaries using existing database
and network access controls.

The runtime coordinates drain, candidate reconciliation and serving activation
when configured. Deployment wiring and
cold recovery are [POP-4377](https://linear.app/worldcoin/issue/POP-4377) and
[POP-4378](https://linear.app/worldcoin/issue/POP-4378).

## Record and transitions

`cpu_cell_handoffs` has one row per handoff: its ID, source/destination deployment
IDs, phase, the draining source process ID and the completed drain report.
Deployment IDs are unique releases, not reusable A/B slot labels. Retain history.

| Local phase | Meaning |
| --- | --- |
| `requested` | Deployment selected the destination; A may still serve. |
| `draining` | A recorded drain-start. Its current process can finish admitted work; a restarted A cannot start serving. |
| `drained` | A persisted the existing agreed report after all admitted effects settled. A remains stopped. |
| `activated` | B recorded its local transition. Peer agreement and the current-process barrier are still required before serving. |

The small store API uses ordinary SQL transactions. Exact retries preserve the
record; conflicting identities, evidence or transitions fail. Only one handoff
can be unfinished. A later handoff must start from the previous destination and
use a fresh destination deployment ID.

Drain initiation persists `draining` before allowing the drain operation. A
canceled or failed database operation leaves admission closed because commit
may have occurred. Completion is stored after the drain reaches the all-party
agreed `Paused` state, not merely after local work finishes.

The database report remains readable after A exits. The drain compares durable
completed watermarks and final publications, independent of retained request
history. Each B reads its local record, exchanges it with the current B peers,
and reconciles its retained workers at the agreed watermark. A need not remain
alive for this exchange. Matching canonical iris metadata reports allow B to
prepare its serving actor from the reconciled resident workers. The current B
processes must still agree on activation before processing requests.

## Migrations and startup

The [forward-only SQLx migration](../migrations/20261108000000_cpu_cell_handoffs.sql)
creates the handoff table through the existing migration path. With no CPU cell
configuration, an empty table preserves ordinary startup. Configured A binds its
managed drain before the first request exists. Migrations alone do not start a
handoff or require an operator to access production Postgres.

The [follow-up migration](../migrations/20261113000000_cpu_cell_remove_candidate_processes.sql)
removes the original candidate-process binding while preserving progress and
completion. Handoff protocol 2 accompanies this schema change. Stop old
participating binaries and use a coordinated cold upgrade with no handoff in
flight before applying it: protocol 1 binaries still reference the removed
column, and startup migrations run before peer compatibility checks. Enable
warm handoff only after every party runs the compatible release. Downgrading
to the old handoff implementation after this migration is unsupported.

The runtime coordinator records the intended handoff from
deployment configuration, using the service's existing database connection.
POP-4377 supplies that configuration through the existing party-owned deployment
workflow. There is no admin
example or separate SQL installation step.

IDs are UUIDs. Set `SMPC__CPU_CELL__DEPLOYMENT_ID` on each participating CPU
linear-scan deployment. The candidate still uses `SMPC__STARTUP_MODE=candidate`.

Run one live process per party per slot and stop it before replacing it, as in
the existing single-replica `Recreate` deployment. A and B may overlap; two
processes for the same slot may not. If a node is unreachable, fence the old
process before starting its replacement: force-deleting its Pod does not prove
it stopped. Supporting same-slot overlap would require a different contract.

Any recorded handoff history requires deployment-aware startup. Missing
configuration or an invalid phase blocks active startup. The current requested
source and the latest activated destination may enter active startup. After
drain-start, the source is blocked after migrations and before claim recovery,
replay or intake. Existing source connections can settle admitted work.
Candidates remain passive until promotion completes; a restarted process must
rebuild resident state.

Both active and candidate startup run the ordinary SQLx migrator before reading
handoff history. Already-applied migrations are not reapplied. A newer release
may contain pending migrations; they must be compatible with A continuing to
serve. Migration errors prevent startup. B then opens read-only connections for
preload. Freeze further schema changes until handoff finishes; incompatible
changes require a coordinated cold deployment. The standalone migrator remains
available through the existing deployment workflow.

Before recording a new request, B verifies all three current A processes expose the
compatible handoff protocol and are serving without a drain already requested.
For an existing matching request, B resumes from the database without contacting A.
An A initiating the first seal checks the intended handoff, source readiness and
drain eligibility, and loaded candidates. Once an A has sealed the same handoff,
the remaining A processes finish draining even if B is reloading. Existing
three-party drain agreement must complete before B can reconcile. A request
alone does not stop intake.
Deployment must also stop ordinary alternate writers: a row cannot revoke an
already-open legacy connection. Keep automatic handoff disabled until deployment
and cold recovery are tested.

## Runtime configuration

Omitting all `SMPC__CPU_CELL__PEERS__*` settings keeps automatic handoff disabled. All
participating processes use the same fixed A/B candidate-status URL lists,
indexed by party. Set `SLOT` to the process's own slot (`a` or `b`):

```sh
SMPC__CPU_CELL__PEERS__SLOT=b
SMPC__CPU_CELL__PEERS__A='["http://a0:3090/handoff-status","http://a1:3090/handoff-status","http://a2:3090/handoff-status"]'
SMPC__CPU_CELL__PEERS__B='["http://b0:3090/handoff-status","http://b1:3090/handoff-status","http://b2:3090/handoff-status"]'
```

The slot selects its own cohort for reconciliation/activation and the other
cohort for its later source role. These settings stay fixed through promotion
and across releases in the same slot. Both lists must contain three distinct,
valid URLs; incomplete peer configuration rejects startup before preload.

On candidate B additionally set:

```sh
SMPC__CPU_CELL__HANDOFF__SOURCE_DEPLOYMENT_ID=<current-A-deployment-uuid>
SMPC__CPU_CELL__HANDOFF__SOURCE_STATUS_URLS='["http://a0:3000/handoff-status","http://a1:3000/handoff-status","http://a2:3000/handoff-status"]'
```

Source URLs address the current serving cohort's healthcheck listeners; the
A/B lists address candidate listeners. Those ports can differ. Active source
coordination uses its existing `server_coordination` hostnames and ports.

For a new rollout, generate one fresh destination deployment UUID and commit
it across all three parties. Candidates that support the destination-UUID default
use it when `SMPC__CPU_CELL__HANDOFF__ID` is omitted. For an older warm-handoff
image, set that field explicitly to the same fresh destination UUID. An explicit
ID takes precedence. A retry or restart must keep the original source, destination
and effective handoff ID, including an older deployment’s distinct explicit ID.

Deploy the next A with the same A/B lists, `SLOT=a`, a new deployment UUID,
and B as its source. Retire the old A processes before reusing
the slot. B keeps serving until the intended next A cohort is loaded and
compatible; URLs alone do not authorize a seal.

The destination is B's `SMPC__CPU_CELL__DEPLOYMENT_ID`. Configure DB-backed intake
and a positive `SMPC__CPU_DRAIN_TIMEOUT_SECS`. B's first request precedes its
startup history check, including after a previous handoff was activated.
Preload/preflight can wait up to an hour. After all B processes finish preload,
each B has one drain-timeout budget to observe and agree on completion, then a
fresh budget of the same duration for reconciliation and metadata agreement.
An A whose preflight fails keeps serving until it seals. There is no extra
all-A preflight acknowledgement round: asymmetric connectivity can let one A
seal while another cannot proceed. Such a partial pause fails drain agreement;
it never authorizes B or automatically resumes A. Uncertain seal attempts also
leave admission closed.

The opt-in adds a read-only `/handoff-status` route on A's existing coordination
listener and binds B's candidate status listener to all interfaces. Fixed URLs,
bounded responses and process UUIDs do **not** authenticate peers. Production
must restrict these listeners to the intended party processes using its existing
network/access controls; do not publish them as serving endpoints. Common
configuration must match within each cell. A and B may have different serving
settings, but their protocol and forward-only schema changes must be compatible
and reviewed together. Configuring automatic handoff also enables B promotion
after agreement; use the passive stage configuration when testing preload alone.

The deployment migrator applies migrations forward; this migration has no down
operation. Retain handoff history because startup uses it to keep retired
deployments stopped.

## Interrupted handoff

Failed preload leaves A serving before the first seal. A replacement B starts
without readiness or reports, reads the existing request, loads canonical state
and reconciles again. It can agree with already-reconciled B peers while writers
remain paused, even after A exits. Reconciliation context is the handoff and C;
each process publishes its report through its live status after reconciliation.
A replacement starts with no readiness or report. No original B process list
is stored in the handoff record.

Existing preload and agreement deadlines remain bounded. A reload exceeding
another peer's remaining agreement budget can require restarting that failed
candidate too. Fresh preload retains the normal loader's dense-ID requirement.
Missing/mismatched reports, an A restart during drain or uncertain effects still
stop progress. No timeout resumes A. A fresh B cannot inherit reconciliation or
in-memory activation readiness. Once serving startup establishes the peer cohort
and MPC sessions, replacing a B requires coordinated recovery in POP-4378,
including when activation is only partially recorded and intake remains closed.

After promotion, losing one serving process shuts down its serving peers through
the existing heartbeat. Restart all three with the deployed slot configuration.
Each replacement reads its latest handoff row. An exact match for its activated
handoff and destination selects coordinated cold startup. The three fresh
processes must agree before rebuilding from canonical state and passing the
normal startup barriers. Readiness stays closed until serving resumes.
If the latest row is a successor’s `requested` handoff and names this deployment
as its source, the replacement uses the handoff ID in its manifest to look up
its original `activated` row. Once the successor starts sealing, the source
cannot restart this way. Partial activation still requires POP-4378 recovery.

Before enabling handoff, test this party-owned cold-restart procedure, including
settling outstanding effects and reloading canonical state. Do not clear
handoff records or remove configuration while an old process may still act.
Database restore and slot reuse need the same operating discipline. This table
cannot cancel an SNS/SQS request already in flight or detect restored history.

## Resident promotion

Normal promotion consumes B's reconciled workers once, initializes deferred
serving networking and actor state, records local activation, and passes the
existing coordination barriers before opening intake. It does not repeat
cold-start migrations, modification replay or the database preload. A local
`activated` row alone does not permit serving.

Serving coordination can allocate a fresh process UUID. The handoff status
must publish that serving identity consistently, and activation must check the
exact current three-party cohort against matching handoff, completion and
reconciliation reports. Replacing B requires fresh preload and reconciliation;
recorded progress cannot supply the replacement's in-memory readiness.

Promotion and peer waits have a bounded budget. A failure leaves B closed and
A stopped; it does not roll back the durable seal. Include deferred serving
initialization and the first completed B request when measuring handoff pause.
The `candidate_handoff_to_first_publication` log measures from B's post-preload
completion wait through reconciliation, promotion and its first nonempty result
publication. Correlate the A admission seal separately for end-to-end pause.
Deployment wiring for the next destination remains POP-4377.
