# Graph & Iris Tooling

How to generate synthetic iris codes, build/inspect HNSW graphs, and deploy an
iris DB + graph checkpoint to the dev cluster.

---

## 1. The counting rule: one row per eye, one graph per eye

This trips everyone up, so it comes first.

**Iris NDJSON files hold one row per *eye*, not per identity.**
Each line is a `Base64IrisCode` (`{"iris_codes": "...", "mask_codes": "..."}`),
written newline-delimited by
`iris-mpc-cpu/src/utils/serialization/types/iris_base64.rs::write_to_iris_ndjson`.

`init-single-db` reads the stream and calls `.tuples()` on it
(`iris-mpc-bins/bin/iris-mpc-cpu/init_single_db.rs:97`), so consecutive rows are
paired:

| NDJSON line (0-indexed) | Meaning                     |
| ----------------------- | --------------------------- |
| 0                       | serial id 1, **left** eye   |
| 1                       | serial id 1, **right** eye  |
| 2                       | serial id 2, left eye       |
| 3                       | serial id 2, right eye       |

So **even lines are the left-eye database, odd lines are the right-eye database**.
That is exactly what `IrisSelection::{Even,Odd}` selects
(`iris-mpc-cpu/src/utils/serialization/iris_ndjson.rs::irises_from_ndjson_iter`).

**HNSW graphs are per-eye.** There is one graph over the left-eye store and one
over the right-eye store. The on-disk "graph pair" file is just the two graphs
serialized back-to-back (`bincode` of `[GraphV5; 2]`, see
`write_graph_pair_current`), which is why `cat g.dat g.dat > pair.dat` produces a
valid pair file — the trick used in `scripts/tools/init-single-db.sh:32`.

**Therefore:**

- A "100k graph" is **two** graphs of 100k nodes each.
- Feeding it requires **200k NDJSON rows** (100k identities × 2 eyes).
- `TARGET_DB_SIZE` in the deploy script counts **iris pairs** (identities), i.e.
  serial ids — so `TARGET_DB_SIZE=100000` consumes 200k rows.

---

## 2. Building the binaries

```bash
# note: bare `cargo` may fail on a missing sccache in this environment
RUSTC_WRAPPER="" cargo build -p iris-mpc-bins --release \
  --bin construct-graph-ptxt \
  --bin graph-utils \
  --bin init-single-db \
  --bin graph-mem-cli
```

| Binary                | Source                                                | Purpose                                       |
| --------------------- | ----------------------------------------------------- | --------------------------------------------- |
| `construct-graph-ptxt`| `iris-mpc-bins/bin/iris-mpc-cpu/construct_graph_ptxt.rs` | generate synthetic irises + build a graph  |
| `graph-utils`         | `iris-mpc-bins/bin/iris-mpc-cpu/graph_utils.rs`       | stat / split / pair / upgrade / diff graphs   |
| `init-single-db`      | `iris-mpc-bins/bin/iris-mpc-cpu/init_single_db.rs`    | secret-share an NDJSON into one party's DB    |
| `graph-mem-cli`       | `iris-mpc-bins/bin/iris-mpc-cpu/graph_mem_cli.rs`     | load/backup graph checkpoints to/from S3 + DB |

---

## 3. Generating synthetic iris codes

`construct-graph-ptxt` is driven by a TOML job spec (`--job-spec`). Its
`[irises]` section is an `IrisesConfig` (`iris-mpc-cpu/src/utils/cli.rs`) with two
variants: `Random` (generate) and `NdjsonFile` (read).

### Step 3a — generate the NDJSON (both eyes)

To end up with a 100k-identity DB, generate **200 000** random codes.
`iris-random.toml`:

```toml
distance_fn = "MinRotation"        # "Simple" | "MinRotation"
distance_ops = "Fhd"               # "Fhd" | "Nhd"
hnsw_prf_seed = 42

[irises]
option = "Random"
number = 200000                    # rows written = 200k = 100k identities
seed = 42                          # omit for a nondeterministic run
output_path = "data/synthetic-irises-100k-pairs.ndjson"

[searcher]
max_graph_layer = 1

[searcher.params]
option = "Standard"
ef_constr = 320
ef_search = 320
M = 256

[output]
path = "data/throwaway-graph.dat"  # ignore; see step 3b
```

```bash
mkdir -p data
./target/release/construct-graph-ptxt --job-spec iris-random.toml
```

The `Random` variant writes the codes to `output_path` *before* graph
construction. The graph it then builds treats **all 200k rows as one store**,
which is not what you want — hence the throwaway output and the per-eye passes
below. (A model config lives at
`iris-mpc-bins/resources/iris-mpc-cpu/construct_graph_ptxt_benchmark.toml`.)

### Step 3b — build one graph per eye

Run `construct-graph-ptxt` twice against the NDJSON, once per eye, using
`selection`:

`graph-left.toml`:

```toml
distance_fn = "MinRotation"
distance_ops = "Fhd"
hnsw_prf_seed = 42

[irises]
option = "NdjsonFile"
path = "data/synthetic-irises-100k-pairs.ndjson"
limit = 100000                     # nodes in THIS eye's graph
selection = "Even"                 # "Odd" for the right eye

# [graph]                          # optional: extend an existing graph
# path = "data/starting-graph.dat"
# format = "v5"

[searcher]
max_graph_layer = 1

[searcher.params]
option = "Standard"
ef_constr = 320
ef_search = 320
M = 256

[output]
path = "data/graph-left-100k.dat"
```

```bash
./target/release/construct-graph-ptxt --job-spec graph-left.toml
sed 's/Even/Odd/; s/graph-left/graph-right/' graph-left.toml > graph-right.toml
./target/release/construct-graph-ptxt --job-spec graph-right.toml
```

`limit` is strict: `irises_from_ndjson` errors if the file yields fewer codes
than requested — a built-in guard against the 100k/200k mistake.

Instead of `[output] path`, you can emit intermediate sizes:

```toml
[output]
base_directory = "data/"
filename_stem = "graph-320-256-minfhd"
checkpoints = { Regular = 25000 }        # or { Values = [1000, 10000, 100000] }
```

which writes `graph-320-256-minfhd_25000.dat`, `..._50000.dat`, … Each file is a
complete graph for that prefix of the store.

### Shortcut used by the dev deploy

For **random** iris codes there are no genuine matches, so a single-eye graph is
reused for both eyes: `scripts/tools/init-single-db.sh` concatenates the one
graph file with itself. That is sufficient for load/perf testing but is *not* a
semantically correct right-eye graph — don't do it when match behaviour matters.

---

## 4. Inspecting a graph with `graph-utils`

```
graph-utils <COMMAND>
  stat            <src_file> [--src-format <fmt>]
  split-pair      <src_file> <dst_left> <dst_right> [--src-format <fmt>]
  make-pair       <src_left> <src_right> <dst_file>
                  [--src-format-left <fmt>] [--src-format-right <fmt>]
  upgrade-format  <src_file> <dst_file> [--src-format <fmt>] [--pair]
  diff            <file_1> <file_2> [--src-format-1 <fmt>] [--src-format-2 <fmt>]
                  [jaccard [-n N] | links [--sort-by <index|...>]]
```

`<fmt>` is a `GraphFormat` (`iris-mpc-cpu/src/utils/serialization/graph.rs`):
`current`, `v5`, `v4`, `v3`, `v2`, `v1`, `v0`, `raw`. Default is `current`
(= V5). `current`/`v5` add Rice-coded neighbourhoods and an edge-invalidation
map; `v4` adds sequence numbers over `v3`.

### Stat a single-eye graph

```bash
./target/release/graph-utils stat data/graph-left-100k.dat
```

```
Reading graph from file: data/graph-left-100k.dat
Succesfully read graph from file.

=== Graph Statistics ===
File format: Current
Checksum: 12345678...

Total nodes: 100000
Layer 0 nodes: 100000
Layer 1 nodes: 3106

Total entry points: 1
Layer 1 entry points: 1
```

`Total nodes` is the **sum over layers**, so the per-eye DB size is
`Layer 0 nodes` — that's the number to compare against `wc -l / 2`.
A `WARNING: entry points present in multiple layers` line means the graph is
malformed for our loader.

### Stat a pair file

`stat` reads a **single** graph, not a pair. Split first:

```bash
./target/release/graph-utils split-pair data/pair.dat \
  data/left.dat data/right.dat --src-format v3
./target/release/graph-utils stat data/left.dat --src-format current
./target/release/graph-utils stat data/right.dat --src-format current
```

`split-pair`/`make-pair` always **write** in the current (V5) format regardless
of the input format — so after a split, stat with `--src-format current`.

### Make a proper pair file

```bash
./target/release/graph-utils make-pair \
  data/graph-left-100k.dat data/graph-right-100k.dat data/graph-pair-100k.dat
```

### Sanity checks

```bash
# unknown/legacy format: let it trial all formats and rewrite as current
./target/release/graph-utils upgrade-format old.dat new.dat            # single
./target/release/graph-utils upgrade-format old-pair.dat new-pair.dat --pair

# compare two graphs
./target/release/graph-utils diff a.dat b.dat jaccard -n 15
./target/release/graph-utils diff a.dat b.dat links --sort-by index
```

---

## 5. Verifying the iris file with `wc -l`

The NDJSON is one JSON object per line, so line count is authoritative:

```bash
wc -l data/synthetic-irises-100k-pairs.ndjson
# 200000  -> 100000 identities (serial ids), 100000 nodes per eye
```

The invariant to check before deploying:

```bash
rows=$(wc -l < data/synthetic-irises-100k-pairs.ndjson)
nodes=$(./target/release/graph-utils stat data/graph-left-100k.dat \
        | awk '/^Layer 0 nodes:/ {print $4}')
echo "rows=$rows  identities=$((rows / 2))  graph_layer0=$nodes"
# require: rows / 2 >= nodes   and   rows % 2 == 0
```

Rules of thumb:

- `rows` must be **even** — an odd count means a truncated/half-written file and
  `.tuples()` will silently drop the last row.
- `rows / 2` must be **≥** the per-eye graph node count and **≥**
  `TARGET_DB_SIZE`, otherwise the graph references serial ids that don't exist in
  the DB.
- On a gzipped file: `gzip -dc file.ndjson.gz | wc -l`.

---

## 6. Deploying to the dev cluster with `scripts/tools/init-single-db.sh`

`scripts/tools/init-single-db.sh` is the **entrypoint of the init-db image**
(`Dockerfile.arm64.init-db`, which bakes in `/bin/init-single-db` and
`/bin/graph-mem-cli`). It runs **once per party** — three pods, three
`PARTY_ID`s, three databases.

### What it does

1. `aws s3 cp` the gzipped irises + graph from
   `s3://wf-smpcv2-dev-hnsw-performance-reports/` and gunzip to `/tmp`.
2. `cat "$GRAPH_FILE" "$GRAPH_FILE" > pair` — turns the **single-eye** graph into
   a both-eyes pair file.
3. `init-single-db` — secret-shares the NDJSON and inserts this party's shares up
   to `TARGET_DB_SIZE` pairs.
4. `psql` — sets `persistent_state (genesis, last_indexed_iris_id) =
   TARGET_DB_SIZE`, *before* the checkpoint, because `load-checkpoint` embeds
   that value into the checkpoint row.
5. `graph-mem-cli … load-checkpoint --graph-format $GRAPH_FORMAT` — loads the
   pair file, uploads it to the checkpoint S3 bucket and inserts the checkpoint
   row. On a fresh DB this leaves exactly one, most-recent checkpoint.
6. Sleeps forever (the deployment restarts on its own; no need to re-run).

### Step 6a — upload the artifacts

The script downloads `${IRISES_FILE}.gz` and `${GRAPH_FILE}.gz`, so upload
gzipped, with the `.gz` suffix added on top of the name you'll pass in:

```bash
gzip -k data/synthetic-irises-100k-pairs.ndjson
gzip -k data/graph-left-100k.dat

aws s3 cp data/synthetic-irises-100k-pairs.ndjson.gz \
  s3://wf-smpcv2-dev-hnsw-performance-reports/
aws s3 cp data/graph-left-100k.dat.gz \
  s3://wf-smpcv2-dev-hnsw-performance-reports/
```

### Step 6b — build/push the image

The image is manual-only: run the **"Init-DB Build and push Arm64 docker image"**
workflow (`.github/workflows/build-and-push-init-db-arm64.yaml`,
`workflow_dispatch`) from the Actions tab. It pushes
`ghcr.io/<repo>-init-db:<sha>-arm64`. Point the dev init-db deployment at that
tag.

### Step 6c — set the environment

| Variable                              | Default                                | Notes                                                     |
| ------------------------------------- | -------------------------------------- | --------------------------------------------------------- |
| `IRISES_FILE`                         | `synthetic-irises-1M.ndjson`           | S3 key **without** `.gz`                                  |
| `GRAPH_FILE`                          | `graph-synthetic-minfhd5-1M.dat`       | **single-eye** graph; script duplicates it                |
| `GRAPH_FORMAT`                        | `v3`                                   | must match how the graph was serialized                   |
| `TARGET_DB_SIZE`                      | `1048576`                              | iris **pairs** = serial ids; needs `2 ×` rows in the NDJSON |
| `SMPC__SERVER_COORDINATION__PARTY_ID` | —                                      | 0/1/2, per pod                                            |
| `SMPC__CPU_DATABASE__URL`             | —                                      | postgres URL for that party                               |
| `SMPC__HNSW_SCHEMA_NAME_SUFFIX`       | —                                      | schema is `SMPC${suffix}_dev_${PARTY_ID}`                  |
| `GRAPH_CHECKPOINT_S3_BUCKET`          | `wf-smpcv2-dev-hnsw-checkpoint`        | where iris-mpc-cpu later reads the checkpoint from        |
| `GRAPH_CHECKPOINT_S3_REGION`          | `eu-central-1`                         |                                                           |

For the 100k example:

```yaml
env:
  IRISES_FILE: "synthetic-irises-100k-pairs.ndjson"
  GRAPH_FILE:  "graph-left-100k.dat"
  GRAPH_FORMAT: "current"      # graph-utils / construct-graph-ptxt write V5
  TARGET_DB_SIZE: "100000"     # 100k pairs <- 200k NDJSON rows
```

`GRAPH_FORMAT` is the most common failure: the default `v3` is for the legacy
`graph-synthetic-minfhd5-1M.dat` artifact. Anything you build today with
`construct-graph-ptxt` or `graph-utils` is `current` (V5). If the deserializer
errors, confirm the format locally with
`graph-utils stat <file> --src-format <fmt>` (or let `upgrade-format` trial the
formats for you) before redeploying.

### Local dry run

Everything the script does works outside the cluster, given AWS creds and a
reachable postgres:

```bash
PARTY_ID=0
DB_URL=postgres://postgres:postgres@localhost:5432
DB_SCHEMA=SMPC_dev_0

./target/release/init-single-db \
  --party-id "$PARTY_ID" --source data/synthetic-irises-100k-pairs.ndjson \
  --db-url "$DB_URL" --db-schema "$DB_SCHEMA" --target-db-size 100000

cat data/graph-left-100k.dat data/graph-left-100k.dat > /tmp/pair.dat

./target/release/graph-mem-cli \
  --db-url "$DB_URL" --schema "$DB_SCHEMA" --file /tmp/pair.dat \
  --s3-bucket wf-smpcv2-dev-hnsw-checkpoint --party-id "$PARTY_ID" \
  --aws-region eu-central-1 \
  load-checkpoint --graph-format current
```

Other `graph-mem-cli` subcommands that are useful here:

- `backup-graph` — DB checkpoint + mutations → pair file (then `split-pair` + `stat`).
- `compare-to-db --diff-method detailed-jaccard` — file vs. DB graph.
- `verify-backup` — round-trip check against a known file.

`init-single-db` is resumable: it reads `get_max_serial_id()` and skips
`2 × n_existing` rows, so re-running with a larger `TARGET_DB_SIZE` tops up the
DB rather than reinserting. It never shrinks a DB.
