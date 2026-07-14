#!/bin/sh

set -e

# ---------------------------------------------------------------------------
# Configuration (overridable via env). Defaults reproduce the stage setup.
# ---------------------------------------------------------------------------
# S3 location of the plaintext source data (iris gallery + graph).
SOURCE_S3_BUCKET="${SOURCE_S3_BUCKET:-wf-smpcv2-stage-hnsw-performance-reports}"
GALLERY_OBJECT_KEY="${GALLERY_OBJECT_KEY:-gallery_left.ndjson}"
# Single bincode file holding BothEyes<GraphMem> (both left AND right graphs).
GRAPH_OBJECT_KEY="${GRAPH_OBJECT_KEY:-graph.dat}"

# Checkpoint bucket that iris-mpc-cpu will later read the restored graph from.
GRAPH_CHECKPOINT_S3_BUCKET="${GRAPH_CHECKPOINT_S3_BUCKET:-wf-smpcv2-stage-hnsw-checkpoint}"
GRAPH_CHECKPOINT_S3_REGION="${GRAPH_CHECKPOINT_S3_REGION:-eu-central-1}"

TARGET_DB_SIZE="${TARGET_DB_SIZE:-287895}"

PARTY_ID="$SMPC__SERVER_COORDINATION__PARTY_ID"
DB_URL="$SMPC__CPU_DATABASE__URL"
DB_SCHEMA="SMPC_correctness_test_stage_$PARTY_ID"

GALLERY_FILE="/tmp/gallery_left_right_interleaved.ndjson"
GRAPH_FILE="/tmp/graph.dat"

# ---------------------------------------------------------------------------
# 1. Download plaintext irises and the (both-eyes) plaintext graph.
# ---------------------------------------------------------------------------
echo "downloading plaintext irises and graph from s3://$SOURCE_S3_BUCKET"
aws s3 cp "s3://$SOURCE_S3_BUCKET/$GALLERY_OBJECT_KEY" "$GALLERY_FILE"
aws s3 cp "s3://$SOURCE_S3_BUCKET/$GRAPH_OBJECT_KEY" "$GRAPH_FILE"

# ---------------------------------------------------------------------------
# 2. Initialize the iris DB from the plaintext gallery.
# ---------------------------------------------------------------------------
echo "starting iris init"
/bin/init-single-db \
  --party-id "$PARTY_ID" \
  --source "$GALLERY_FILE" \
  --db-url "$DB_URL" \
  --db-schema "$DB_SCHEMA" \
  --target-db-size "$TARGET_DB_SIZE"
echo "iris init done"

# ---------------------------------------------------------------------------
# 3. Record last_indexed_iris_id BEFORE the checkpoint: load-checkpoint embeds
#    this value into the checkpoint row it creates.
# ---------------------------------------------------------------------------
psql "$DB_URL" -c "
INSERT INTO persistent_state (domain, \"key\", \"value\")
VALUES ('genesis', 'last_indexed_iris_id', '$TARGET_DB_SIZE')
ON CONFLICT (domain, \"key\")
DO UPDATE SET \"value\" = EXCLUDED.\"value\";
"
echo "persistent_state updated"

# ---------------------------------------------------------------------------
# 4. Restore the graph (both eyes from one file) and upload exactly one
#    checkpoint. On a fresh DB this leaves a single, most-recent checkpoint.
# ---------------------------------------------------------------------------
echo "starting graph restore + checkpoint"
/bin/graph-mem-cli \
  --db-url "$DB_URL" \
  --schema "$DB_SCHEMA" \
  --file "$GRAPH_FILE" \
  --s3-bucket "$GRAPH_CHECKPOINT_S3_BUCKET" \
  --party-id "$PARTY_ID" \
  --aws-region "$GRAPH_CHECKPOINT_S3_REGION" \
  load-checkpoint
echo "graph restore + checkpoint done"

echo "init complete"
exit 0
