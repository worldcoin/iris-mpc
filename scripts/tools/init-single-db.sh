#!/bin/sh

set -e

IRISES_FILE="${IRISES_FILE:-synthetic-irises-1M.ndjson}"
GRAPH_FILE="${GRAPH_FILE:-graph-synthetic-minfhd5-1M.dat}"
GRAPH_FORMAT="${GRAPH_FORMAT:-v3}"

TARGET_DB_SIZE="${TARGET_DB_SIZE:-1048576}"
PARTY_ID="${SMPC__SERVER_COORDINATION__PARTY_ID}"
DB_URL="${SMPC__CPU_DATABASE__URL}"
# matches behavior of iris-mpc
DB_SCHEMA="SMPC${SMPC__HNSW_SCHEMA_NAME_SUFFIX}_dev_${PARTY_ID}"

# Checkpoint bucket that iris-mpc-cpu will later read the restored graph from.
GRAPH_CHECKPOINT_S3_BUCKET="${GRAPH_CHECKPOINT_S3_BUCKET:-wf-smpcv2-dev-hnsw-checkpoint}"
GRAPH_CHECKPOINT_S3_REGION="${GRAPH_CHECKPOINT_S3_REGION:-eu-central-1}"

echo "downloading plaintext irises and graph from s3"

aws s3 cp "s3://wf-smpcv2-dev-hnsw-performance-reports/${GRAPH_FILE}.gz" "/tmp/${GRAPH_FILE}.gz" --only-show-errors
aws s3 cp "s3://wf-smpcv2-dev-hnsw-performance-reports/${IRISES_FILE}.gz" "/tmp/${IRISES_FILE}.gz" --only-show-errors

echo "download complete. unzipping"

gzip -dc "/tmp/${GRAPH_FILE}.gz" > "/tmp/${GRAPH_FILE}"
gzip -dc "/tmp/${IRISES_FILE}.gz" > "/tmp/${IRISES_FILE}"

echo "starting iris init"
/bin/init-single-db \
  --party-id "$PARTY_ID" \
  --source "/tmp/${IRISES_FILE}" \
  --db-url "$DB_URL" \
  --db-schema "$DB_SCHEMA" \
  --target-db-size "$TARGET_DB_SIZE"
echo "iris init done"

# ---------------------------------------------------------------------------
# Record last_indexed_iris_id BEFORE the checkpoint: load-checkpoint embeds
# this value into the checkpoint row it creates.
# ---------------------------------------------------------------------------
psql "${DB_URL}" -v ON_ERROR_STOP=1 -c "
INSERT INTO \"${DB_SCHEMA}\".persistent_state (domain, \"key\", \"value\")
VALUES ('genesis', 'last_indexed_iris_id', '${TARGET_DB_SIZE}')
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
  --file "/tmp/${GRAPH_FILE}" \
  --s3-bucket "$GRAPH_CHECKPOINT_S3_BUCKET" \
  --party-id "$PARTY_ID" \
  --aws-region "$GRAPH_CHECKPOINT_S3_REGION" \
  --graph-format "$GRAPH_FORMAT" \
  load-checkpoint 
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
