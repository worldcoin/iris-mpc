# Genesis Persistence Slowdown -- Aurora Investigation Brief

## The operation

Each genesis batch persists ~30K rows (x2, once per eye) via:

```sql
INSERT INTO {links_table} (graph_id, serial_id, version_id, layer, links)
SELECT $1, * FROM UNNEST($2::int8[], $3::int4[], $4::int2[], $5::bytea[])
ON CONFLICT (graph_id, serial_id, version_id, layer)
DO UPDATE SET links = EXCLUDED.links
```

- `links` is ~1.9KB bytea (serialized neighbor list). Only column that changes -- indexed columns are never modified.
- Rows are random-access across a ~32M-row table (~60GB).
- Observed: ~8s/batch early in genesis, degrading to ~50s/batch at 4M nodes.

## DB-side changes to evaluate

### 1. FILLFACTOR (enable HOT updates)

Default FILLFACTOR=100 means no free page space. Every UPDATE writes a new tuple on a different page and **updates the btree index** -- doubling I/O per row. Since only `links` changes (not indexed), HOT updates would skip the index write entirely, but need free space on the page.

```sql
ALTER TABLE {links_table} SET (fillfactor = 70);
VACUUM FULL {links_table};
```

**Verify with:**
```sql
SELECT relname, n_tup_upd, n_tup_hot_upd,
       CASE WHEN n_tup_upd > 0
            THEN round(100.0 * n_tup_hot_upd / n_tup_upd) ELSE 0
       END AS hot_update_pct
FROM pg_stat_user_tables WHERE relname LIKE '%links%';
```
Currently `hot_update_pct` is likely ~0%. After change should be 90%+.

Tradeoff: ~43% more disk space for the table.

### 2. TOAST tuning (reduce random heap I/O)

Rows are ~1,970 bytes total -- just under the ~2,048-byte TOAST threshold. Values stay **inline**, so each 8KB heap page holds only ~4 rows. 30K random UPSERTs = ~7,500 random page reads/writes.

Forcing `links` out-of-line:
```sql
ALTER TABLE {links_table} SET (toast_tuple_target = 128);
VACUUM FULL {links_table};
```

Heap rows shrink to ~35 bytes (~230 rows/page). Same 30K UPSERTs touch ~130 heap pages instead of ~7,500.

**Tradeoff -- TOAST OID consumption:** `ON CONFLICT DO UPDATE SET links = EXCLUDED.links` allocates a **new TOAST OID for every row**, even on the UPDATE path (PostgreSQL doesn't compare old vs new values). That's ~30K new OIDs per call x 2 sides x ~41K batches = **~2.5B OIDs** over a full genesis (60% of the 4B 32-bit OID space). If autovacuum keeps up and reclaims dead TOAST chunks promptly, live OID density stays low (~32M) and this is fine. If VACUUM falls behind, density climbs and PostgreSQL burns CPU scanning for free OIDs under `OidGenLock`.

**Risk level:** Moderate. Requires monitoring but not a showstopper. FILLFACTOR (option 1) is the safer first step since it has no TOAST interaction.

**Monitor:** `LWLock:OidGenLock` and `LWLock:buffer_io` wait events in Performance Insights. Also check autovacuum is keeping up on the TOAST tables (`pg_stat_user_tables.n_dead_tup` for the `pg_toast_*` relations).

### 3. shared_buffers sizing

Aurora has **no OS filesystem page cache** -- cache misses go directly to network storage. AWS recommends `shared_buffers` at **75% of instance memory** (not the standard PG 25%). If the btree index + hot heap pages don't fit, every row update is a storage round-trip.

## Datadog metrics to check

Correlate these with `avg:ampc_hnsw_sync_protocol_0.genesis_batch_persist_duration{*}` over the same time range. Filter by `dbinstanceidentifier` tag.

| Metric | Why |
|--------|-----|
| `aws.rds.buffer_cache_hit_ratio` | Cache miss rate. If low/declining as graph grows, working set exceeds buffer pool. |
| `aws.rds.volume_read_iops` | Storage reads. High = cache misses driving random I/O to storage. |
| `aws.rds.volume_write_iops` | Storage writes. Quantifies write amplification per batch. |
| `aws.rds.read_latency` | Per-I/O storage read latency. |
| `aws.rds.write_latency` | Per-I/O storage write latency. |
| `aws.rds.commit_latency` | Time per COMMIT. High = redo log pressure. |
| `aws.rds.disk_queue_depth` | Outstanding I/O requests. High = I/O backlog. |
| `aws.rds.dbload_non_cpu` | Sessions waiting on I/O or locks (not CPU). |
| `aws.rds.freeable_memory` | Available memory. Declining = buffer pool pressure. |
| `aws.rds.transaction_logs_generation` | WAL generation rate. Quantifies redo volume from our UPSERTs. |
