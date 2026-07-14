#!/bin/sh

set -e

IRISES_FILE="/tmp/gallery_left_right_interleaved.ndjson"
GRAPH_FILE="/tmp/graph.dat"
GRAPH_FORMAT="v3"

TARGET_DB_SIZE="1048576"
PARTY_ID="$SMPC__SERVER_COORDINATION__PARTY_ID"
DB_URL="$SMPC__CPU_DATABASE__URL"
DB_SCHEMA="SMPC_1M_dev_$PARTY_ID"

# Checkpoint bucket that iris-mpc-cpu will later read the restored graph from.
GRAPH_CHECKPOINT_S3_BUCKET="${GRAPH_CHECKPOINT_S3_BUCKET:-wf-smpcv2-dev-hnsw-checkpoint}"
GRAPH_CHECKPOINT_S3_REGION="${GRAPH_CHECKPOINT_S3_REGION:-eu-central-1}"

echo "downloading plaintext irises and per-eye graphs from google drive"

# 1M plaintext irises that match the graph.dat
curl "https://drive.usercontent.google.com/download?id=1je1stRXfVrHy2LRVcfg-SrRiz_yw33S4&export=download&confirm=t" -o ${IRISES_FILE}

# an additional 1M irises
# curl "https://drive.usercontent.google.com/download?id=1whFu0GIezDA2_YD9eMr9cY60oU0BF_Ao&export=download&confirm=t" -o /tmp/irises2M.ndjson

# serialized BothEyes<GraphMem> in format V3
curl "https://drive.usercontent.google.com/download?id=1vjswOMB7Yn-f7TDOqfzQr1_ZS-ubSg4E&export=download&confirm=t" -o ${GRAPH_FILE}

echo "starting iris init"
/bin/init-single-db \
  --party-id "$PARTY_ID" \
  --source "$IRISES_FILE" \
  --db-url "$DB_URL" \
  --db-schema "$DB_SCHEMA" \
  --target-db-size "$TARGET_DB_SIZE"
echo "iris init done"

# ---------------------------------------------------------------------------
# Record last_indexed_iris_id BEFORE the checkpoint: load-checkpoint embeds
# this value into the checkpoint row it creates.
# ---------------------------------------------------------------------------
psql "$DB_URL" -c "
INSERT INTO persistent_state (domain, \"key\", \"value\")
VALUES ('genesis', 'last_indexed_iris_id', '$TARGET_DB_SIZE')
ON CONFLICT (domain, \"key\")
DO UPDATE SET \"value\" = EXCLUDED.\"value\";
"
echo "persistent_state updated"

# ---------------------------------------------------------------------------
# Restore the graph (both eyes from one file) and upload exactly one
# checkpoint. On a fresh DB this leaves a single, most-recent checkpoint.
# ---------------------------------------------------------------------------
echo "starting graph restore + checkpoint"
/bin/graph-mem-cli \
  --db-url "$DB_URL" \
  --schema "$DB_SCHEMA" \
  --file "$GRAPH_FILE" \
  --s3-bucket "$GRAPH_CHECKPOINT_S3_BUCKET" \
  --party-id "$PARTY_ID" \
  --aws-region "$GRAPH_CHECKPOINT_S3_REGION" \
  load-checkpoint --graph-format "$GRAPH_FORMAT"
echo "graph restore + checkpoint done"
echo "init complete"


# the deployment tends to be auto-restart. no need to do that here. wait for the next deployment.
shutdown_handler() {
    echo "Shutdown requested"
    exit 0
}

trap shutdown_handler SIGTERM SIGINT
while true; do     
    sleep 1        
done 
