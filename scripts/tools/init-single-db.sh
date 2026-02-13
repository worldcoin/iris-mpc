#!/bin/sh

# opt-in codes left, right
aws s3 cp s3://wf-smpcv2-stage-hnsw-performance-reports/graph_right.dat /tmp/graph_right.dat
aws s3 cp s3://wf-smpcv2-stage-hnsw-performance-reports/graph_left.dat /tmp/graph_left.dat

aws s3 cp s3://wf-smpcv2-stage-hnsw-performance-reports/gallery_left.ndjson /tmp/gallery_left_right_interleaved.ndjson


echo "starting left init"
/bin/init-single-db \
  --party-id $SMPC__SERVER_COORDINATION__PARTY_ID \
  --source "/tmp/gallery_left_right_interleaved.ndjson" \
  --db-url "$SMPC__CPU_DATABASE__URL" \
  --db-schema "SMPC_correctness_test_stage_$SMPC__SERVER_COORDINATION__PARTY_ID" \
  --target-db-size 287895

echo "starting restore graph left"
/bin/graph-mem-cli --db-url $SMPC__CPU_DATABASE__URL --schema SMPC_correctness_test_stage_$SMPC__SERVER_COORDINATION__PARTY_ID --file /tmp/graph_left.dat restore-side --side "left"
echo "restore graph left done"

echo "starting restore graph right"
/bin/graph-mem-cli --db-url $SMPC__CPU_DATABASE__URL --schema SMPC_correctness_test_stage_$SMPC__SERVER_COORDINATION__PARTY_ID --file /tmp/graph_right.dat restore-side --side "right"
echo "restore graph right done"

psql "$SMPC__CPU_DATABASE__URL" -c "
INSERT INTO persistent_state (domain, \"key\", \"value\")
VALUES ('genesis', 'last_indexed_iris_id', '287895')
ON CONFLICT (domain, \"key\")
DO UPDATE SET \"value\" = EXCLUDED.\"value\";
"
echo "persistent_state updated"

shutdown_handler() {
    echo "Shutdown requested"
    exit 0
}

trap shutdown_handler SIGTERM SIGINT

while true; do
    sleep 1
done
