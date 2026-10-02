# Passive CPU candidate startup

Use the `iris-mpc-linear-scan` binary with the active cell's database/schema and CPU scan settings, plus:

```sh
SMPC__CPU_STARTUP_MODE=candidate
SMPC__ENABLE_S3_IMPORTER=false
SMPC__CANDIDATE_STATUS_PORT=3090
```

`cpu_startup_mode` defaults to `active`, preserving coordinated startup. HNSW and GPU entrypoints reject candidate mode. Candidate mode rejects seeding/clearing, fake data, invalid party/loading settings, and unsupported CPU scan settings.

The candidate opens existing Postgres tables with read-only transactions on every pooled connection. It preloads the configured resident eye using the exact-scan worker layout and retains row IDs and observed versions. It runs no migrations, coordination, MPC network, queue reads, claim resets, modification/result replay, or live execution. S3 loading is outside this first implementation.

## Operator status

Status binds to **127.0.0.1 only**. Access it with deployment-authorized pod exec or port forwarding; do not publish it through the serving Service.

| Path | Meaning |
| --- | --- |
| `/candidate-status` | Process UUID, party, `loading`/`loaded`, loaded row count, `serving: false` |
| `/health` | Process responds; independent of remote dependencies |
| `/startup` | 200 after preload, 503 while loading |
| `/ready` | Always 503: candidate must never receive live traffic |

Preload errors emit a failed transition and exit nonzero; use container logs/exit status after failure. The complete preload has a one-hour deadline. SIGTERM/SIGINT releases retained memory and closes the status listener. A candidate needs its own workload/probes: a serving-readiness gate must not prevent an operator from observing successful preload.

## Handoff boundary

`loaded` means **preloaded and unreconciled**. The active cell can continue writing during parallel reads; there is no common snapshot or committed frontier yet. Resident shares and metadata are kept in a private `CandidatePreload` containing `InitializedWorkers`, not a running actor.

Cold-eye caches are deliberately deferred: rereading their versions while A writes could disagree with the resident preload. A subsequent reconciliation/activation implementation must reconcile retained shares and versions, initialize the cold-eye pool/cache against the agreed frontier, and only then construct the actor and join the request path. Activation must retain this process UUID. This change provides no activation endpoint. Stop the candidate to abandon it; ordinary active startup remains the rollback path.

## Focused regression check

Against a disposable Postgres instance:

```sh
POP4370_TEST_DATABASE_URL=postgres://postgres:postgres@127.0.0.1:55437/postgres \
  cargo test -p iris-mpc --lib candidate -- --include-ignored
```

The database test uses the actual linear-scan entrypoint and worker loader with only an iris table present and no AWS/peer configuration. It checks retained versions, read-only enforcement across multiple connections, status semantics, shutdown, and startup failure.
