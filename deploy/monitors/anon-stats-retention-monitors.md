# anon-stats-retention — Datadog monitors (BLOCKING observability gate, POP-3905)

Per the POP-3926 decision: **all four monitors must fire correctly in staging before the prod CronJob is un-suspended for any party.** This repo has no monitors-as-code; these are the implementable specs (apply via Datadog UI / terraform / MCP). The binary emits the metrics below via StatsD (`SMPC__SERVICE__METRICS__*`, hostIP:8125), tagged `service`, `env`, `party`, `table`.

Metrics emitted by the binary (per run, tagged by `party` + `table` where applicable):
- `anon_stats_retention.last_success` — gauge `1` on a fully successful run (else not emitted / `0`).
- `anon_stats_retention.partitions_ahead_days` — gauge: newest forward daily partition date − today, in days.
- `anon_stats_retention.partition_count` — gauge: live partitions per table.
- `anon_stats_retention.rows_dropped` — count per run.
- `anon_stats_retention.bytes_dropped` — count per run (`pg_total_relation_size` before drop).
- `anon_stats_retention.oldest_retained_created_at` — gauge: epoch seconds of the oldest live partition's lower bound.
- `anon_stats_retention.run.duration` — histogram (run wall-clock).

> **Two clusters.** Retention runs as independent CronJobs for both the **smpcv2** and **ampc-hnsw** clusters (separate anon-stats DBs), 3 parties each = 6 jobs per env. The StatsD prefix / `service` tag distinguishes them (`anon-stats-retention-smpcv2-<P>` vs `anon-stats-retention-hnsw-<P>`). All monitors below must therefore group by `service` (which encodes cluster+party), not by `party` alone, and the cross-party divergence monitor (#4) must be evaluated **within each cluster**. Exact metric/tag resolution (StatsD prefix vs tag) MUST be confirmed against live staging output as part of this gate before relying on the queries verbatim.

---

## 1. Missing-run / job-failure  (the silent-killer case)

The CronJob runs daily; a missing success in >26h means the run failed, never scheduled, or the pod silently died — growth resumes invisibly.

- **Type:** Metric monitor (no-data alerting).
- **Query:** `sum(last_26h):default_zero(sum:anon_stats_retention.last_success{env:stage} by {party}) < 1`
- **No-data:** alert after 26h of no data, evaluated per `party`.
- **Severity:** page. Each of the 3 parties evaluated independently (multi-alert by `party`).
- **Why 26h:** daily cadence + a 2h buffer for run duration + jitter.

## 2. Partitions-ahead-below-premake  (watch the buffer, not the job)

Fires *days before* an insert could ever hit a missing partition. This is the leading indicator — it trips while there's still runway, unlike monitor #1 which trips after a failure.

- **Type:** Metric monitor.
- **Query:** `min(last_2h):min:anon_stats_retention.partitions_ahead_days{env:stage} by {party,table} < 3`
- **Threshold:** warn `< 4`, alert `< 3` (premake target is 7; below 3 days ahead = act now).
- **Severity:** page. Multi-alert by `party` + `table`.

## 3. Rows/bytes dropped > Nσ from baseline  (catch an over-aggressive scope)

Catches an inverted/over-wide drop *before* it finishes eating data — e.g. a window misconfiguration dropping far more than a normal day.

- **Type:** Anomaly monitor.
- **Query:** `avg(last_4h):anomalies(sum:anon_stats_retention.rows_dropped{env:stage} by {party}.as_count(), 'agile', 3, direction='above', alert_window='last_1h', interval=3600, count_default_zero='true')`
- **Threshold:** anomaly bound 3σ, direction above. Also a hard ceiling backstop: `sum(last_1d):sum:anon_stats_retention.rows_dropped{env:stage} by {party} > <CEILING>` where CEILING ≈ 2× the steady-state daily insert volume (set after observing staging baseline).
- **Severity:** page. The hard ceiling is the safety net if anomaly training data is thin.

## 4. Cross-party divergence  (the INC-30/44/79 drift class)

The 3 parties run independent CronJobs against independent DBs. If one stalls, retained-window sizes diverge — the same drift class that caused prior incidents.

- **Type:** Metric monitor (formula across party tags).
- **Query:** divergence of the oldest-retained boundary across parties exceeds 1 day:
  `max(last_2h):( max:anon_stats_retention.oldest_retained_created_at{env:stage} by {} - min:anon_stats_retention.oldest_retained_created_at{env:stage} by {} ) > 86400`
  (evaluate per `table`; 86400s = 1 day tolerance.)
- **Severity:** page.
- **Note:** a healthy steady state keeps all 3 parties within the same daily-partition boundary; >1 day divergence means one party's reap or premake is lagging.

## 5. DEFAULT partition is silently accumulating live data  (premake-failure sink)

Because a DEFAULT (legacy catch-all) partition exists, if premake ever fails to create a forward daily partition, new inserts **don't error — they silently land in the DEFAULT partition**. Two consequences: (a) the DEFAULT's `max(created_at)` advances, so the legacy wholesale-drop (`max(created_at) < now−14d`) never fires → retention silently stops for everything in DEFAULT; (b) that day's partition can no longer be created without moving rows. Monitor #2 (partitions-ahead) is the leading guard, but this is the explicit detector of the failure itself.

- **Type:** Metric monitor on a binary-emitted gauge. The retention binary should emit `anon_stats_retention.default_partition_rows{table,party}` (row count of the `*_legacy` DEFAULT partition) and/or `anon_stats_retention.default_partition_max_created_at_epoch`. Alert when the DEFAULT row count is rising run-over-run, or `default_partition_max_created_at` advances past `migration_timestamp + 1 day`.
- **Query (rows rising):** `change(avg(last_2d):avg:anon_stats_retention.default_partition_rows{env:stage} by {service,table}) > 0` (sustained increase = inserts sinking into DEFAULT).
- **Severity:** page. **NOTE:** this requires adding `default_partition_rows` to the binary — a small follow-up to the current build (the binary already enumerates partitions; it can size the DEFAULT the same way).

---

## Staging validation checklist (gate — must pass before prod un-suspend)
- [ ] All 4 monitors created against `env:stage`, multi-alerted by `party` (+ `table` where noted).
- [ ] Monitor #1 verified by suspending one party's CronJob and confirming it pages within 26h.
- [ ] Monitor #2 verified by setting `PREMAKE_DAYS=2` on one party and confirming it warns.
- [ ] Monitor #3 baseline ceiling set from observed staging `rows_dropped`.
- [ ] Monitor #4 verified by pausing one party's reap and confirming divergence pages.
- [ ] Only after all four: clone monitors to `env:prod` and flip `suspend: false` per party (canary one party first).
