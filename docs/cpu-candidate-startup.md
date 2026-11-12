# Passive CPU candidate startup

Use the `iris-mpc-linear-scan` binary with the active cell's database/schema and CPU scan settings, plus:

```sh
SMPC__STARTUP_MODE=candidate
SMPC__ENABLE_S3_IMPORTER=false
SMPC__CLEAR_DB_BEFORE_INIT=false
SMPC__CANDIDATE_STATUS_PORT=3090
```

`startup_mode` defaults to `active`, preserving coordinated startup. The HNSW entrypoint rejects candidate mode. Candidate mode rejects seeding/clearing, fake data, invalid party/loading settings, and unsupported CPU scan settings.

The candidate runs the ordinary SQLx migrator, checks handoff history, then opens Postgres with read-only transactions on every preload connection. Already-applied migrations are not reapplied; pending migrations must be compatible with A serving. It preloads the configured resident eye using the exact-scan worker layout and retains row IDs and observed versions. With [runtime handoff configuration](cpu-cell-ownership.md#runtime-configuration), it verifies A participation and records the intended handoff before the history check, then coordinates durable completion and reconciliation after preload. It runs no MPC network, queue reads, claim resets, modification/result replay, or live execution. S3 loading is outside this implementation.

## Operator status

Status binds to **127.0.0.1 by default**. Explicit runtime handoff configuration binds it to all interfaces for fixed peer status exchange. Protect that listener with existing party network/access controls; do not publish it through the serving Service.

| Path | Meaning |
| --- | --- |
| `/candidate-status` | Process UUID, party, `loading`/`loaded`/`reconciled`, loaded row count, `serving: false` |
| `/health` | Process responds; independent of remote dependencies |
| `/startup` | 200 after preload, 503 while loading |
| `/ready` | Always 503: candidate must never receive live traffic |

Preload errors emit a failed transition and exit nonzero; use container logs/exit status after failure. The complete preload has a one-hour deadline. SIGTERM/SIGINT releases retained memory and closes the status listener. A candidate needs its own workload/probes: a serving-readiness gate must not prevent an operator from observing successful preload.

## Handoff boundary

`loaded` means **preloaded and unreconciled**. The active cell can continue writing during parallel reads; there is no common snapshot or committed frontier yet. Resident shares and metadata are kept in a private `CandidatePreload` containing `InitializedWorkers`, not a running actor.

Cold-eye caches are deliberately deferred during preload: rereading their versions while A writes could disagree with the resident preload. After all durable drain reports agree, reconciliation updates retained shares and versions and initializes the cold-eye pool/cache at the agreed frontier. Matching metadata reports leave the workers resident and passive. Future activation must retain this process UUID before constructing the actor and joining the request path. There is no activation endpoint. An interrupted handoff follows the [cold-recovery contract](cpu-cell-ownership.md#interrupted-handoff); restarting an active process does not undo a recorded drain.

## Focused regression check

Against a disposable Postgres instance:

```sh
CPU_CELL_TEST_DATABASE_URL=postgres://postgres:postgres@127.0.0.1:55437/postgres \
  cargo test -p iris-mpc --lib candidate -- --include-ignored
```

The database test uses the actual linear-scan entrypoint, normal migrations and worker loader without AWS/peer configuration. It checks retained versions, read-only preload connections, status semantics, shutdown, and startup failure.

The three-party reconciliation regression uses three isolated schemas on one disposable Postgres instance:

```sh
CPU_CELL_TEST_DATABASE_URL=postgres://postgres:postgres@127.0.0.1:55437/postgres \
  cargo test -p iris-mpc --lib --features db_dependent three_party_reconciliation -- --test-threads=1
```

It generates valid per-party shares, preloads three candidates, and simulates A's committed insert/update/removal writes. Runtime coordinators verify participation, record requests and wait for every candidate before starting three drains over local HTTP. They persist completion, reopen PostgreSQL connections, stop A's endpoints, and let B exchange the durable reports and reconcile. Completed request retention differs across parties, including no retained completed rows, while pending rows above the watermark survive. The test checks exact retries, old A restart rejection, each party's resident and cold-eye payloads, removed/stale IDs, and retained resident pools for either resident eye. A separate case changes one party's database version and rejects agreement despite equal row counts. The cold wrapper is replaced during reconciliation; its private worker handle is not asserted. This is an idle drain with no newly admitted batch; real SQS/SNS workers, authenticated peer exchange and serving activation are outside the fixture. Passive entrypoint behavior and admitted-batch publication conditions retain separate tests.
