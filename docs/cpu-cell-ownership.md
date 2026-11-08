# CPU handoff record

A and B share one party's Postgres. The handoff table records local progress;
existing peer coordination establishes agreement across the three parties.
This is a planned handoff between cooperating binaries using existing database
and network access controls.

The local foundation does not enable automatic promotion. Coordination and
serving activation are [POP-4454](https://linear.app/worldcoin/issue/POP-4454) and
[POP-4376](https://linear.app/worldcoin/issue/POP-4376); deployment wiring and
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

The database report remains readable after A exits. Future coordination reuses
the existing full drain and reconciliation comparisons; equality of maximum
request sequence numbers alone does not establish matching work.

## Migrations and startup

The [versioned SQLx migration](../migrations/20261108000000_cpu_cell_handoffs.up.sql)
creates the handoff table through the existing migration path. An empty table leaves ordinary
startup, migrations and exporter writes unchanged. Deploying this foundation
does not start a handoff or require an operator to access production Postgres.

The runtime coordinator in POP-4454 will record the intended handoff from
deployment configuration, using the service's existing database connection.
POP-4377 supplies that configuration through the existing party-owned deployment
workflow. This PR provides the store API and startup/drain guards; it does not
yet call the request API automatically or enable promotion. There is no admin
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
blocked before migrations, claim recovery, replay or intake. Existing source
connections can settle admitted work. Candidates remain passive; a database row
cannot restore lost resident readiness after a process restart.

Once history exists, serving processes validate existing migrations without
running DDL; alternate runtime writers are rejected. The standalone migrator
keeps its established deployment role. Compatible schema changes must finish
before candidate preload; schema validation does not serialize a concurrent
deployment migration.

Before recording the first handoff, the coordinator/deployment integration must
ensure all participating processes use the handoff-aware runtime and ordinary
alternate writers are stopped. A new row cannot revoke an already-open legacy
connection. Keep automatic handoff disabled until that integration and recovery
are tested. No role provisioning or process secrets are needed.

The down migration can remove an unused table, but refuses to delete handoff
history because that would remove the restart guards. All handoff tests use
`CPU_CELL_TEST_DATABASE_URL` pointing to disposable PostgreSQL.

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
