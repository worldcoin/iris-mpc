# db-sanity-check

Read-only validation of a single MPC party's Postgres database state. Checks HNSW graph structure, persistent state consistency, and cross-schema (HNSW vs GPU) alignment.

The graph can be loaded either from the Postgres `hawk_graph_links` table (default) or from an S3 genesis checkpoint (pass `--checkpoint-s3-bucket`). S3 mode adds an extra check (0a) that validates checkpoint metadata against the `persistent_state` watermarks.

## Usage

```bash
db-sanity-check \
  --hnsw-db-url <HNSW_DATABASE_URL> \
  --gpu-db-url <GPU_DATABASE_URL> \
  --hnsw-schema <SCHEMA> \
  --gpu-schema <SCHEMA> \
  --seed <SEED> \
  [--m <M>] \
  [--exclusions-s3-uri <S3_URI>] \
  [--checkpoint-s3-bucket <BUCKET>] \
  [--checkpoint-s3-key <KEY>] \
  [--checkpoint-s3-region <REGION>] \
  [--force-path-style] \
  [--output-dir <DIR>] \
  [--s3-output <S3_URI>]
```

### Arguments

| Argument | Env var | Required | Default | Description |
|----------|---------|----------|---------|-------------|
| `--hnsw-db-url` | `HNSW_DATABASE_URL` | yes | | Postgres connection string for the HNSW (CPU) database |
| `--gpu-db-url` | `GPU_DATABASE_URL` | yes | | Postgres connection string for the GPU database |
| `--hnsw-schema` | | yes | | HNSW (CPU) schema name (e.g. `SMPC_hnsw_dev_0`) |
| `--gpu-schema` | | yes | | GPU schema name (e.g. `SMPC_gpu_dev_0`) |
| `--seed` | | yes | | RNG seed for reproducible cross-schema sampling (check 3c) |
| `--m` | | no | `256` | HNSW M parameter for degree bound checks |
| `--layer-probability` | | no | `1/M` | Layer probability q for geometric distribution check |
| `--exclusions-s3-uri` | | no | | S3 URI to JSON exclusions file with `{"deleted_serial_ids": [...]}` (e.g. `s3://bucket/path/deleted_serial_ids.json`) |
| `--checkpoint-s3-bucket` | | no | | Enable S3 checkpoint mode. Bucket holding genesis graph checkpoints. When set, the graph is downloaded from S3 instead of loaded from `hawk_graph_links`. |
| `--checkpoint-s3-key` | | no | | Specific checkpoint key. Requires `--checkpoint-s3-bucket`. If omitted, the latest checkpoint is auto-discovered via the `genesis_graph_checkpoint` DB table. |
| `--checkpoint-s3-region` | | no | ambient | Region override for the checkpoint bucket. Required when the checkpoint bucket lives in a different region from iris/exclusions buckets. Requires `--checkpoint-s3-bucket`. |
| `--force-path-style` | | no | `false` | Force path-style S3 addressing (required for LocalStack; must stay `false` in production — prod S3 uses virtual-hosted style). |
| `--output-dir` | | no | `.` | Directory for JSON output files |
| `--s3-output` | | no | | S3 URI to upload output files to (e.g. `s3://bucket/prefix/`) |

### Exit code

- `0` — all checks passed
- `1` — one or more checks failed

## Examples

### Against local Docker (single schema per party)

```bash
# Start infra
docker compose -f docker-compose.dev.yaml up -d

# Seed DBs (requires store.ndjson — generate with construct_graph_ptxt first)
cd iris-mpc-bins && cargo run --release --bin construct-graph-ptxt -- --job-spec resources/iris-mpc-cpu/construct_graph_ptxt_benchmark.toml
cargo run --release -p iris-mpc-bins --bin init-test-dbs -- \
  --source iris-mpc-bins/data/store.ndjson \
  --db-url-party1 "postgres://postgres:postgres@localhost:5432/SMPC_dev_0" \
  --db-url-party2 "postgres://postgres:postgres@localhost:5432/SMPC_dev_1" \
  --db-url-party3 "postgres://postgres:postgres@localhost:5432/SMPC_dev_2" \
  --db-schema-party1 SMPC_dev_0 \
  --db-schema-party2 SMPC_dev_1 \
  --db-schema-party3 SMPC_dev_2 \
  --target-db-size 1000

# Run sanity check (locally, same DB instance for both hnsw/gpu)
cargo run --release -p iris-mpc-bins --bin db-sanity-check -- \
  --hnsw-db-url "postgres://postgres:postgres@localhost:5432/SMPC_dev_0" \
  --gpu-db-url "postgres://postgres:postgres@localhost:5432/SMPC_dev_0" \
  --hnsw-schema SMPC_dev_0 \
  --gpu-schema SMPC_dev_0 \
  --seed 42 \
  --output-dir sanity-check/party0
```

### Against remote DBs (separate HNSW and GPU instances)

```bash
db-sanity-check \
  --hnsw-db-url "postgres://user:pass@hnsw-rds-host:5432/mydb" \
  --gpu-db-url "postgres://user:pass@gpu-rds-host:5432/mydb" \
  --hnsw-schema SMPC_hnsw_prod_0 \
  --gpu-schema SMPC_gpu_prod_0 \
  --seed 42 \
  --m 256 \
  --exclusions-s3-uri s3://my-bucket/path/deleted_serial_ids.json \
  --output-dir sanity-check/party0
```

### With S3 upload

```bash
db-sanity-check \
  --hnsw-db-url "postgres://user:pass@hnsw-rds-host:5432/mydb" \
  --gpu-db-url "postgres://user:pass@gpu-rds-host:5432/mydb" \
  --hnsw-schema SMPC_hnsw_prod_0 \
  --gpu-schema SMPC_gpu_prod_0 \
  --seed 42 \
  --output-dir /tmp/sanity-check \
  --s3-output s3://my-bucket/sanity-check/party0/
```

### With S3 checkpoint (post-genesis validation)

```bash
db-sanity-check \
  --hnsw-db-url "postgres://user:pass@hnsw-rds-host:5432/mydb" \
  --gpu-db-url "postgres://user:pass@gpu-rds-host:5432/mydb" \
  --hnsw-schema SMPC_hnsw_prod_0 \
  --gpu-schema SMPC_gpu_prod_0 \
  --seed 42 \
  --checkpoint-s3-bucket wf-smpcv2-prod-hnsw-checkpoint \
  --checkpoint-s3-region us-east-1 \
  --output-dir /tmp/sanity-check
```

Omit `--checkpoint-s3-key` to auto-discover the latest checkpoint from the `genesis_graph_checkpoint` table. For LocalStack e2e tests, add `--force-path-style`.

## Checks

| ID | Check | Category |
|----|-------|----------|
| 0a | Checkpoint metadata vs persistent_state (S3 mode only) | Checkpoint |
| 1a | No orphan graph nodes (per eye) | HNSW graph |
| 1b | Node coverage vs exclusions list (per eye) | HNSW graph |
| 1c | Layer hierarchy (higher-layer nodes in all lower layers) | HNSW graph |
| 1d | Neighbor validity (all neighbors exist at same layer) | HNSW graph |
| 1e | No self-loops | HNSW graph |
| 1f | No duplicate neighbors | HNSW graph |
| 1g | Degree bounds (M_limit per layer) | HNSW graph |
| 1h | Entry point validity | HNSW graph |
| 1i | Left/Right graph sync (same layer-0 serial IDs) | HNSW graph |
| 1j | Layer density near geometric (within 3σ of Binomial(N, q^L)) | HNSW graph |
| 2a | last_indexed_iris_id matches irises table max serial ID | Persistent state |
| 2b | Graph max serial_id alignment (left == right) | Persistent state |
| 3c | Byte-identical shares for sampled IDs (random sample of ~1k + up to 100 recent modification serial IDs; pending iris-code-updating modifications excluded) | Cross-schema |

### Check behavior under S3 checkpoint mode
- **0a** validates that the checkpoint's `last_indexed_iris_id` and `last_indexed_modification_id` match the party's `persistent_state` watermarks. Runs only when `--checkpoint-s3-bucket` is set.
- **1a–1j, 1i** consume the S3 in-memory graph. Check 1b (node coverage) still compares against the *current* `irises` table, so irises added after the checkpoint will appear as uncovered — expected, and surfaced more directly by check 0a.
- **2b** reads the max serial ID from the in-memory `GraphMem` layer 0 instead of `hawk_graph_links`, since the links table is not the source of truth when the graph was loaded from S3.
- **2a, 3c** are source-independent (backed by `persistent_state` and `irises` tables).

Additionally, the following are logged as informational (not PASS/FAIL checks):

| ID | Log | Category |
|----|-----|----------|
| 3a | Row count comparison (HNSW vs GPU, id ≤ last_indexed) | Cross-schema |
| 3b | Max serial ID comparison (id ≤ last_indexed) | Cross-schema |

## Scaling notes

The tool currently loads the full HNSW graph into memory (one eye at a time). For production-scale databases, consider delegating checks to Postgres directly: checks on `hawk_graph_links` metadata (1a, 1b, 1c, 1h, 1i, 1j) can be pure SQL queries needing no application memory, and checks that inspect the serialized `links` blob (1d, 1e, 1f, 1g) can stream rows instead of loading the full graph.

## Output files

Written to `--output-dir`:

- **`checks.json`** — array of `{id, name, status, detail}` (with optional `warnings` array)
- **`stats.json`** — object of key-value statistics
- **`degree_histogram.csv`** — CSV with columns `eye, layer, degree, node_count`
- **`report.txt`** — human-readable report (same content as stdout)

In S3 checkpoint mode, `stats.json` additionally includes:

- `checkpoint_s3_key` — the S3 object that was loaded
- `checkpoint_last_indexed_iris_id`
- `checkpoint_last_indexed_modification_id`
- `checkpoint_blake3_hash` — BLAKE3 hash of the checkpoint bytes. All 3 parties must produce the same value; `scripts/run-sanity-checks-genesis-e2e.sh` asserts this cross-party equality after each genesis run.

When `--s3-output` is set, all four files are uploaded to the specified S3 prefix after being written locally.
