# db-sanity-check

Read-only validation of a single MPC party's Postgres database state. Checks HNSW graph structure, persistent state consistency, cross-schema (HNSW vs GPU) alignment, and the modifications table.

## Usage

```bash
db-sanity-check \
  --db-url <DATABASE_URL> \
  --hnsw-schema <SCHEMA> \
  --gpu-schema <SCHEMA> \
  [--m <M>] \
  [--exclusions-file <PATH>] \
  [--output-dir <DIR>]
```

### Arguments

| Argument | Env var | Required | Default | Description |
|----------|---------|----------|---------|-------------|
| `--db-url` | `DATABASE_URL` | yes | | Postgres connection string |
| `--hnsw-schema` | | yes | | HNSW (CPU) schema name (e.g. `SMPC_hnsw_dev_0`) |
| `--gpu-schema` | | yes | | GPU schema name (e.g. `SMPC_gpu_dev_0`) |
| `--m` | | no | `256` | HNSW M parameter for degree bound checks |
| `--exclusions-file` | | no | | Path to JSON file with `{"deleted_serial_ids": [...]}` |
| `--output-dir` | | no | `.` | Directory for JSON output files |

### Exit code

- `0` — all checks passed
- `1` — one or more checks failed

## Examples

### Against local Docker (single schema per party)

```bash
# Start infra
docker compose -f docker-compose.dev.yaml up -d

# Seed DBs (requires store.ndjson — generate with generate-benchmark-data first)
cargo run --release -p iris-mpc-bins --bin generate-benchmark-data -- --size 1000
cargo run --release -p iris-mpc-bins --bin init-test-dbs -- \
  --source iris-mpc-bins/data/store.ndjson \
  --db-url-party1 "postgres://postgres:postgres@localhost:5432/SMPC_dev_0" \
  --db-url-party2 "postgres://postgres:postgres@localhost:5432/SMPC_dev_1" \
  --db-url-party3 "postgres://postgres:postgres@localhost:5432/SMPC_dev_2" \
  --db-schema-party1 SMPC_dev_0 \
  --db-schema-party2 SMPC_dev_1 \
  --db-schema-party3 SMPC_dev_2 \
  --target-db-size 1000

# Run sanity check (locally, same schema for both hnsw/gpu)
cargo run --release -p iris-mpc-bins --bin db-sanity-check -- \
  --db-url "postgres://postgres:postgres@localhost:5432/SMPC_dev_0" \
  --hnsw-schema SMPC_dev_0 \
  --gpu-schema SMPC_dev_0 \
  --output-dir sanity-check/party0
```

### Against a remote DB (separate HNSW and GPU schemas)

```bash
db-sanity-check \
  --db-url "postgres://user:pass@rds-host:5432/mydb" \
  --hnsw-schema SMPC_hnsw_prod_0 \
  --gpu-schema SMPC_gpu_prod_0 \
  --m 256 \
  --exclusions-file deleted_serial_ids.json \
  --output-dir sanity-check/party0
```

## Checks

| ID | Check | Category |
|----|-------|----------|
| 1a | No orphan graph nodes (per eye) | HNSW graph |
| 1b | Node coverage vs exclusions list (per eye) | HNSW graph |
| 1c | Layer hierarchy (higher-layer nodes in all lower layers) | HNSW graph |
| 1d | Neighbor validity (all neighbors exist at same layer) | HNSW graph |
| 1e | No self-loops | HNSW graph |
| 1f | Degree bounds (M_limit per layer) | HNSW graph |
| 1g | Entry point validity | HNSW graph |
| 1h | Left/Right graph sync (same layer-0 serial IDs) | HNSW graph |
| 2a | last_indexed_iris_id consistency | Persistent state |
| 2b | Graph max serial_id alignment (left == right) | Persistent state |
| 3a | Same row count (HNSW vs GPU) | Cross-schema |
| 3b | Same max serial ID | Cross-schema |
| 3c | Byte-identical shares (SQL JOIN) | Cross-schema |
| 4a | All modifications COMPLETED & persisted | Modifications |

## Output files

Written to `--output-dir`:

- **`checks.json`** — array of `{id, name, status, detail}`
- **`stats.json`** — object of key-value statistics
- **`degree_histogram.json`** — array of `{eye, layer, degree, node_count}`
