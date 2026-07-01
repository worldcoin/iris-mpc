# retention-reaper — Datadog monitors (observability gate, POP-3905)

The `retention-reaper` CronJob emits these StatsD metrics (prefix per deploy, tagged
`service` + `party` + `table`):

- `retention.last_success` — gauge `1` on a fully successful run.
- `retention.rows_deleted` — count per run, per table.
- `retention.oldest_retained_seconds` — gauge: age of the oldest surviving row (min `created_at`) per table. This is the "is retention actually working" signal.
- `retention.dead_tuple_ratio` — gauge: `n_dead_tup / (n_dead_tup + n_live_tup)` from `pg_stat_user_tables`. **The key health signal for DELETE-based retention** — if autovacuum can't keep pace with the delete churn, this climbs.
- `retention.run.duration` — histogram.

Validate all of these in **staging** before enabling prod (prod ships `suspend: true`). Two clusters run independently (smpcv2 + ampc-hnsw), so group by `service` (encodes cluster+party); the cross-party monitor evaluates within each cluster.

---

## 1. Missing-run / job-failure
The CronJob runs daily; no success in >26h means it failed, didn't schedule, or the pod died — silent unbounded growth resumes.
- **Metric monitor**, no-data alerting: `sum(last_26h):default_zero(sum:retention.last_success{env:stage} by {service}) < 1`, evaluated per `service`. Page.

## 2. Oldest-retained exceeds the window  (retention falling behind)
If the reaper stops deleting (bug, lock contention, guard mistake), the oldest row ages past the window — the direct symptom.
- **Metric monitor**: `max(last_6h):max:retention.oldest_retained_seconds{env:stage} by {service,table} > 1900800` (= 22 days = the 14-day window + a generous margin for backfilled legacy rows + slack). Page. Tune the margin after observing the first weeks (legacy rows are stamped at migration time and drain over the first window).

## 3. Bloat — dead-tuple ratio climbing  (the DELETE-approach risk)
The one real failure mode of batched DELETE: autovacuum not keeping up with delete churn → table/index bloat.
- **Metric monitor**: `avg(last_2h):avg:retention.dead_tuple_ratio{env:stage} by {service,table} > 0.4` warn, `> 0.6` alert. If this fires sustained at scale, the escalation is (a) tune per-table autovacuum (`autovacuum_vacuum_scale_factor` down, cost limit up), then (b) if still losing, partition that table via **pg_partman** (the documented Option-B escalation) — not a bespoke binary.

## 4. Rows-deleted anomaly  (over-deletion guard)
Catches a mis-set retention/guard deleting far more than a normal day.
- **Anomaly monitor**: `anomalies(sum:retention.rows_deleted{env:stage} by {service}.as_count(), 'agile', 3, direction='above', ...)`, plus a hard-ceiling backstop `sum(last_1d) > <2× steady-state daily volume>` (set from staging baseline). Page.

## 5. Cross-party divergence
3 independent DBs per cluster; if one party's reaper stalls, its retained window diverges (the INC-30/44/79 drift class).
- **Metric monitor** (within a cluster): `max(last_6h):( max:retention.oldest_retained_seconds{env:stage,cluster:X} - min:retention.oldest_retained_seconds{env:stage,cluster:X} ) > 172800` (2-day tolerance), per `table`. Page.

---

## Staging validation checklist (gate — before prod un-suspend)
- [ ] All 5 monitors created against `env:stage`, grouped by `service`.
- [ ] #1 verified: suspend one party's CronJob, confirm it pages within 26h.
- [ ] #3 baseline: observe steady-state `dead_tuple_ratio` per table; confirm it stays flat (autovacuum keeping up) at current volume; set the alert margin above observed baseline.
- [ ] #4 hard-ceiling set from observed staging `rows_deleted`.
- [ ] Then clone to `env:prod` and flip `suspend: false` **per party, canary one first**.
- [ ] Re-evaluate #3 thresholds if/when POP-3908 raises volume — that's the trigger to consider pg_partman for the hottest table.
