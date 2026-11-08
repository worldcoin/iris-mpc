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

## Installation and startup

Install explicitly in the canonical schema with the existing deployment tools.
This table is not installed by ordinary server startup. The example reads
`CPU_CELL_DATABASE_URL` from the environment:

```sh
cargo run -p iris-mpc-store --example cpu_cell_admin -- install SCHEMA
cargo run -p iris-mpc-store --example cpu_cell_admin -- request SCHEMA HANDOFF_ID SOURCE_DEPLOYMENT_ID DESTINATION_DEPLOYMENT_ID
cargo run -p iris-mpc-store --example cpu_cell_admin -- status SCHEMA HANDOFF_ID
```

IDs are UUIDs. Set `SMPC__CPU_CELL__DEPLOYMENT_ID` on each participating CPU
linear-scan deployment. The candidate still uses `SMPC__STARTUP_MODE=candidate`.

The table's presence requires deployment-aware startup. Missing configuration,
an empty table or an invalid phase blocks active startup. Active source startup
is allowed only for the current requested handoff; after drain-start it is
blocked before migrations, claim recovery, replay or intake. Existing source
connections can settle admitted work. Candidates remain passive; a database row
cannot restore lost resident readiness after a process restart.

Managed schemas validate existing migrations without running DDL. Alternate
writers and the standalone migrator cannot bypass the managed-schema guard.
Schemas without the handoff table retain existing startup behavior.

Install only with legacy writers stopped and compatible schema changes already
applied. Installing the table changes restart behavior immediately. Do not
install it in a serving environment until the complete handoff and recovery
paths have been verified. No role provisioning, process secrets or dynamic
write-permission changes are part of this feature.

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
