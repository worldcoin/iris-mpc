# ADR-003: Retention for append-only tables (retention-reaper)

## Context
- Several PoP Postgres tables are append-only and grow without bound: the `anon_stats_*`
  tables (POP-3905) and `modifications` (POP-3931), with more likely to follow.
- POP-3926 researched this and initially chose **native range-partitioning + `DROP
  PARTITION`** (Option B) for `anon_stats`, justified by a projected 400 GB/wk/party
  *if* POP-3908 widens the upstream filter — and a guarded `DELETE` for `modifications`.
- Building Option B (attach-as-partition migration + a bespoke partition-lifecycle binary)
  produced ~1.4k lines of catalog-manipulating Rust plus a fragile migration (PK-collision,
  populated-DEFAULT scan, partition-overlap, sequence-ownership were all real bugs found in
  review/validation). It was near-unreviewable, and — being a different mechanism from the
  `modifications` DELETE — guaranteed a second bespoke implementation per table class.

## Decision
Use one generic mechanism for all append-only retention:
1. **Migration:** add `created_at TIMESTAMPTZ NOT NULL DEFAULT now()` + a btree index on it.
   `ADD COLUMN ... DEFAULT now()` is metadata-only on PG11+ (verified on PG16.14:
   relfilenode unchanged — `now()` is STABLE, not VOLATILE, so the fast-default path applies).
2. **`retention-reaper`** (`iris-mpc-bins/bin/retention-reaper`): a single config-driven
   CronJob binary that runs a **bounded, batched, guarded `DELETE`** per configured job
   (`{table, ts_column, retention, guard, batch_size}` via `RETENTION_JOBS`), deleting in
   `ctid` batches older than the window, and emitting metrics (rows deleted, oldest-retained
   age, **dead-tuple ratio**, last success). One binary serves every table/party/cluster.
3. Deployed as a k8s `CronJob` (helm `common-cronjob`), one per DB; config lists that DB's
   tables. `anon_stats` guard = `processed = TRUE`; `modifications` guard = its terminal-state
   + indexed-watermark predicate.

## Rationale
- **Simplicity / reviewability:** ~300 lines of straightforward SQL vs ~1.4k lines of catalog
  manipulation + a fragile migration. Reviewable in an afternoon.
- **Reusability:** the maintainability win — one binary + N config rows across `anon_stats`,
  `modifications`, and future tables, instead of one bespoke binary per class.
- **Correctness fit:** a guarded DELETE honors POP-3905's literal ask ("clean up
  `processed=TRUE` rows") and never deletes uncollected rows — which `DROP PARTITION`
  (age-only, whole-partition) cannot.
- **The 400 GB/wk driver is speculative and gated:** POP-3908 is gated on retention existing;
  current volume is ~4 GB/wk. Building partition machinery for a hypothetical future load is a
  YAGNI inversion. Partitioning offers no read benefit here (reads are by
  `(processed, origin, operation)`, not by time).

## Consequences
- **Pros:** one small, testable, reusable tool; trivial migrations; matches the ticket's
  semantics; unifies with POP-3931.
- **Cons / watch:** batched `DELETE` leaves dead tuples; at sustained extreme churn autovacuum
  could fall behind → bloat. Mitigated by per-table autovacuum tuning + the
  `retention.dead_tuple_ratio` monitor. **Escalation (documented, measurement-gated):** if
  that monitor shows sustained bloat post-POP-3908 for a specific table, partition *that*
  table via **pg_partman** (the extension) — not a hand-rolled binary. This ADR supersedes
  POP-3926's Option-B choice for `anon_stats`; pg_partman remains the escalation, not the
  default.
