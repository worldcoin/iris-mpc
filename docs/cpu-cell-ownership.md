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
IDs, phase, current source/candidate process IDs and the completed drain report.
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
configuration or an invalid phase blocks active startup. Active source startup
is allowed only for the current requested handoff; after drain-start it is
blocked after migrations and before claim recovery, replay or intake. Existing source
connections can settle admitted work. Candidates remain passive; a database row
cannot restore lost resident readiness after a process restart.

Both active and candidate startup run the ordinary SQLx migrator before reading
handoff history. Already-applied migrations are not reapplied. A newer release
may contain pending migrations; they must be compatible with A continuing to
serve. Migration errors prevent startup. B then opens read-only connections for
preload. Freeze further schema changes until handoff finishes; incompatible
changes require a coordinated cold deployment. The standalone migrator remains
available through the existing deployment workflow.

Before recording a request, B verifies all three current A processes expose the
compatible handoff protocol and are serving without a drain already requested.
Each A then waits for all B preloads and the same six-process cohort before
initiating its managed drain once. A request alone does not stop intake.
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
Preload/preflight can wait up to an hour; the drain/reconciliation budget starts
after all B processes finish preload. Preflight failures leave A serving;
uncertain seal attempts and failures after drain-start never resume it.

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

Failed preload leaves A serving. After drain-start, mismatched/missing reports,
process replacement or uncertain effects stop progress. No timeout resumes A.
A fresh B cannot inherit its predecessor's reconciliation or activation.

The local foundation deliberately blocks active restarts after activation too;
normal serving-crash handling is part of the cold-recovery integration. Before
enabling handoff, test a party-owned cold-restart procedure that stops/fences old
processes, settles outstanding effects and reloads canonical state. Do not clear
handoff records or remove configuration while an old process may still act.
Database restore and slot reuse need the same operating discipline. This table
cannot cancel an SNS/SQS request already in flight or detect restored history.
