# CPU handoff record

A and B share one party's Postgres. The handoff table records local progress;
existing peer coordination establishes agreement across the three parties.
This is a planned handoff between cooperating binaries using existing database
and network access controls.

The runtime coordinates drain and candidate reconciliation automatically when
configured. Serving activation remains [POP-4376](https://linear.app/worldcoin/issue/POP-4376); deployment wiring and
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
alive for this exchange. Matching canonical iris metadata reports leave B
reconciled and passive.

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
Candidates remain passive; a restarted process must rebuild resident state.

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

Omitting `SMPC__CPU_CELL__CANDIDATE_STATUS_URLS` keeps automatic handoff disabled.
On all participating A and B processes set that variable to a JSON array of
three fixed B `/handoff-status` URLs, indexed by party. A uses its existing
`server_coordination` hostnames and healthcheck ports for A status exchange.
On B additionally set:

```sh
SMPC__CPU_CELL__HANDOFF__ID=<fresh-handoff-uuid>
SMPC__CPU_CELL__HANDOFF__SOURCE_DEPLOYMENT_ID=<current-A-deployment-uuid>
SMPC__CPU_CELL__HANDOFF__SOURCE_STATUS_URLS='["http://a0:8080/handoff-status","http://a1:8080/handoff-status","http://a2:8080/handoff-status"]'
```

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
and reviewed together. This configuration enables no serving activation.

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
activation; recovery after partial activation remains POP-4378.

After all three parties record activation, the same destination deployment can
use existing coordinated cold startup: stop/fence all old processes, configure
all three parties in active mode with the candidate handoff intent removed, and
restart them together. Startup barriers precede recovery and a full canonical
reload; recorded activation does not restore warm readiness. Single-party and
rolling restarts remain unsupported. An incomplete handoff still blocks active
startup and needs the coordinated recovery procedure in POP-4378.

Before enabling handoff, test this party-owned cold-restart procedure, including
settling outstanding effects and reloading canonical state. Do not clear
handoff records or remove configuration while an old process may still act.
Database restore and slot reuse need the same operating discipline. This table
cannot cancel an SNS/SQS request already in flight or detect restored history.
