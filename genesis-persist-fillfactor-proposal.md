# FILLFACTOR Change Proposal for Genesis Links Table

## The problem

Genesis batch persistence degrades from ~8s to ~50s per batch as the graph grows to 4M nodes. Each batch UPSERTs ~30K rows (x2 sides) into the links table via `INSERT ... ON CONFLICT DO UPDATE SET links = EXCLUDED.links`. Only the `links` bytea column (~1.9KB) changes; the indexed columns `(graph_id, serial_id, version_id, layer)` are never modified.

## What the metrics show

Aurora CloudWatch data from the weekend (March 28-30) for `hnsw-mpc-cluster-prod`:

**buffer_cache_hit_ratio tracks persist duration exactly:**

| Period | Cache Hit % | Persist (s) |
|---|---|---|
| Sat 00:00-08:00 UTC | 99.94-99.98 | ~8 |
| Sat 09:00 (transition) | 98.77 | ~25 |
| Sat 10:00-23:00 | 98.1-99.2 | 19-36 |
| Sun 01:00-04:00 (recovery) | 99.12-99.91 | ~10.4 |
| Sun 07:00 onward | 97.7-98.4 | 40-50 |

The 2% drop in hit ratio (99.95% to 98%) sounds negligible but isn't: with ~300K page accesses per persist call (30K rows x ~5 pages each x 2 sides), a 2% miss rate means ~6,000 storage round-trips vs ~150 at 99.95%. At 2-3ms per round-trip, that's ~15s of additional latency per persist.

**volume_read_iops confirms the cache miss explosion:**

| Period | Read IOPs/hr (cluster) | Persist (s) |
|---|---|---|
| Sat 00:00-07:00 | 2.2-3.4M | ~8 |
| Sat 08:00 (transition) | 8.0M | ~25 |
| Mon 00:00-09:00 | 40-55M | 46-50 |

Storage reads increased **25x** over the weekend. This is not a storage latency problem -- `read_latency` actually *decreased* from 7-8ms to 2.4ms, and `write_latency` decreased from 0.9ms to 0.6ms. The storage layer is fine. The problem is the sheer number of I/Os caused by cache misses.

**The degradation is self-reinforcing:** more cache misses mean more pages fetched from storage, which evict more useful pages from `shared_buffers`, which cause more cache misses on the next persist. The btree index pages (needed by every UPSERT) get evicted by one-shot heap pages (each touched once during a random UPSERT, then never again).

## Why FILLFACTOR helps

With the default FILLFACTOR=100, every `DO UPDATE SET links = EXCLUDED.links` is a **non-HOT update**:

1. Old tuple marked dead (stays on page until VACUUM)
2. New tuple written to a **different** page (no room on the original)
3. Btree index updated to point to the new page location
4. Dead tuples accumulate, bloating both heap and index

Per UPSERT row, this is ~5 page accesses: btree descent (3-4 pages) + old heap page read + new heap page write + index page updates. For 30K rows x 2 sides = **~300K page accesses per persist**.

With FILLFACTOR=70, each page retains 30% free space. Updates become **HOT (Heap-Only Tuple)** because:
- The new tuple fits on the **same** page as the old one
- The btree index is **not touched** (it still points to the same page; only the non-indexed `links` column changed)
- Dead tuples are pruned in-place during normal page access, reducing bloat

Per UPSERT row, this drops to ~2 page accesses: heap page read + heap page write (same page). **No btree I/O at all.** For 30K rows x 2 sides = **~120K page accesses per persist** -- a 60% reduction.

The reduced I/O has a compounding effect on the cache: fewer page accesses means fewer evictions, which means the btree index stays cached, which means the few non-HOT updates (new row inserts) are also faster.

Additionally, HOT avoids dead tuple bloat in the index. With the current pattern, each persist creates ~60K dead heap tuples. Without HOT, each dead tuple also leaves a stale index entry that only VACUUM can clean. With HOT, index entries are unaffected and dead tuples are pruned opportunistically during page access. This directly addresses the self-reinforcing degradation seen in the metrics.

## The change

```sql
ALTER TABLE {links_table} SET (fillfactor = 70);
```

Applied to the links tables on all 3 party databases. No `VACUUM FULL` needed -- new page writes from our UPSERTs will adopt the new fillfactor naturally. Within a few hours, most pages will have been rewritten with free space and HOT updates will start taking effect.

No code changes, no redeployment, no downtime. Genesis continues running throughout.

## Tradeoff

~43% more disk space for the heap (fewer rows per page due to reserved space). For a ~60GB table, this means ~86GB. The btree index size is unchanged.

## Verification

Before the change, capture a baseline:
```sql
SELECT relname, n_tup_upd, n_tup_hot_upd,
       CASE WHEN n_tup_upd > 0
            THEN round(100.0 * n_tup_hot_upd / n_tup_upd) ELSE 0
       END AS hot_update_pct
FROM pg_stat_user_tables WHERE relname LIKE '%links%';
```

After a few hours, re-run. `hot_update_pct` should rise from ~0% to 90%+. Correlate with `genesis_batch_persist_duration` and `buffer_cache_hit_ratio` -- both should improve as HOT kicks in.
