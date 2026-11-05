# Party-local CPU-cell ownership

Each party keeps its own authority in its existing canonical PostgreSQL schema.
The `cpu_cell_*` tables record deployment intent, serving generations, exact
process cohorts, seals, retirements, and recovery. An attempt is a handoff;
a generation is a deployment of slot `a` or `b`. Reusing a slot requires a new
generation and a new database login.

This implements the local authority and startup guards for POP-4453. Keep
handoff disabled until authenticated cross-party coordination (POP-4454),
activation (POP-4376), and party-owned provisioning (POP-4377) are integrated.
The administrative intent operation alone does not trigger a handoff.

## Permissions and process identity

| Principal | Responsibility |
| --- | --- |
| Provisioner | Install schema/migrations and bootstrap control objects before runtime enablement |
| NOLOGIN authority owner | Own control objects and constrained `SECURITY DEFINER` functions |
| Intent login | Register immutable deployments and approve source/destination pairs |
| Recovery login | Supersede old authority and establish a fresh owner after external fencing |
| Generation login | Bind one process lifetime and invoke only permitted runtime transitions |

The provisioner creates dedicated roles. Runtime and administrative logins must
have no role memberships, ownership, elevated PostgreSQL attributes, or canonical
write/DDL grants. Candidates receive canonical `SELECT` permissions, with no
canonical writes even if they disable the session's read-only setting.

Every boot generates a fresh UUID and random 32-byte secret in memory. The
database stores only the secret's SHA-256 digest. Binding checks the authenticated
generation login, UUID, secret, and approved deployment identity. The coordination
health endpoint uses the same UUID. A restart cannot reuse a generation's binding;
it requires explicit recovery with a fresh generation/login. Never persist or log
the process secret.

These secrets protect control operations. Canonical DML grants belong to the
generation login, so they do not fence malicious code holding that credential.
The database also cannot attest that a running image matches its configured
digest. Deployment integrity and credential isolation remain party-owned.

## Handoff ordering

1. The intent login approves immutable source/destination identities and whether
   automatic handoff is permitted. Schema/protocol/configuration compatibility
   labels must match exactly; the database does not infer version compatibility.
2. B binds its process and preloads with read-only permissions. Only successful
   preload records B as loaded. The source freezes the exact three A and three B
   process UUIDs into one attempt.
3. A persists its admission seal before requesting the in-process drain. The
   original process retains write access to settle admitted work, including on
   replacement database connections. A fresh startup rejects the seal.
4. B records the reconciled frontier and proof digest. A independently submits
   the matching settled proof to retire durably. Retirement revokes its canonical
   DML privileges; the migration ledger remains read-only even for an owner.
5. The bound source records a retirement certificate only after the future
   coordinator authenticates matching durable retirement evidence from all three
   parties. The local SQL operation cannot authenticate remote peers itself.
6. Only the frozen B process can acquire ownership with the matching proof.
   Actual serving must also pass the future exact-process activation barrier.

The authority row serializes transitions. Conflicting attempts, stale process
bindings, changed proofs, and superseded attempts fail. An empty persisted
frontier is represented by `None`, independently of the required proof digest.
Completed attempts and retired generations remain recorded across later slot
reuse. There is no timeout takeover, warm resume, or separate COMMIT/ABORT vote.
There is no cancellation endpoint in this scope; a future cancellation protocol
must invalidate delayed seal requests and account for a seal at another party.

## Installation and recovery

`iris-mpc-store/examples/cpu_cell_admin.rs` is the administrative entrypoint.
It takes the connection from `CPU_CELL_DATABASE_URL`, never a command-line
credential. Use the separate principal appropriate to each operation:

```sh
cargo run --locked -p iris-mpc-store --example cpu_cell_admin -- COMMAND ARGS...
```

The supported argument shapes are:

```text
bootstrap SCHEMA OWNER INCARNATION PARTY INTENT_LOGIN RECOVERY_LOGIN
register SCHEMA INCARNATION GENERATION LOGIN SLOT IMAGE_DIGEST SCHEMA_VERSION PROTOCOL CONFIGURATION
intent SCHEMA INCARNATION INTENT SOURCE DESTINATION AUTOMATIC_BOOL
status SCHEMA
begin-recovery SCHEMA INCARNATION RECOVERY EXPECTED_REVISION
finish-recovery SCHEMA INCARNATION RECOVERY FRESH_GENERATION EXTERNAL_FENCING_ACK NEW_INCARNATION
```

Use `status` under the intent/recovery login to read the current incarnation,
revision, owner, attempt, and recovery state without exposing process secrets.
`begin-recovery` compares the observed revision atomically; a stale command
cannot seal a later owner. If completion is uncertain, read status before retrying.

Before first installation, stop or externally fence all legacy processes and
alternate writers, provision the canonical schema and application migrations,
then create the dedicated NOLOGIN owner and intent/recovery/runtime logins.
The provisioner must be able to grant canonical privileges and `SET ROLE` to
the authority owner. `bootstrap` installs atomically and cannot reset an existing
authority. It starts blocked, with no owner.

Register the initial generation, begin recovery using current authority state,
and finish only after verifying external fencing. Then start the generation
with matching `cpu_cell` configuration: `incarnation`, `generation`, `slot`,
`image` (`sha256:` plus 64 lowercase hex characters), `schema_version`,
`protocol`, and `configuration`. These identity labels must come from trusted
deployment configuration. Managed startup requires DB-backed intake and
pre-provisioned migrations; seed/clear operations are rejected.

For an ordinary crash, begin recovery first to supersede any pending activation
and revoke old generations' DML. Settle or externally fence every possible old
A/B owner, including in-flight external effects. Register a fresh generation and
credential, then finish recovery with the external fencing acknowledgment. Keep
the old seals and retirement history. Recovery has no automatic lease expiry.

After database restore, quarantine the restored database before starting any
runtime. Old rows cannot detect that their own history was rolled back. Fence
old processes and credentials externally, and finish recovery with a fresh
externally tracked incarnation; update all participating deployment identities.
An acknowledgment records that an operator verified fencing; it does not perform
or prove fencing. An already-issued SNS/SQS call cannot be canceled by a database
transaction.

The permanent authority marker makes legacy constructors and alternate write
paths reject a managed schema before migrations, seeding, claim reset, replay,
or intake. Removing `cpu_cell` configuration does not opt out. Database lookup
errors fail startup. Previously running or older binaries still require external
fencing before bootstrap; catalog checks do not revoke their existing sessions.

## Validation and cost

The existing PostgreSQL CI service runs the role-backed ownership tests and an
explicit startup-guard step, including the tests normally ignored without a
database. They cover credential permissions, process replacement, concurrent
acquisition, sealed settlement, retirement, repeated slot reuse, and recovery.

Authority checks occur at startup, connection creation, and handoff/recovery
transitions, not on each iris comparison. Control pools are bounded and use
statement/lock deadlines. Transitions serialize on one party-local row; grant
changes scale with schema objects rather than gallery size. Read-only startup
validates migration history without running DDL. Exporter write helpers perform
a bounded authority lookup before writing; read/export paths remain unchanged.
