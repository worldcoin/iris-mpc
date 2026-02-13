#!/bin/sh

# 1M irises that match the graph.dat
aws s3 cp s3://wf-smpcv2-stage-sync-protocol/output-graph-all-data.dat /tmp/graph.dat
aws s3 cp s3://wf-smpcv2-stage-sync-protocol/all_data_iris_wsubject_ids.ndjson /tmp/irises.ndjson


echo "starting"
/bin/init-single-db \
  --party-id $SMPC__SERVER_COORDINATION__PARTY_ID \
  --source "/tmp/irises.ndjson" \
  --db-url "$SMPC__CPU_DATABASE__URL" \
  --db-schema "genesis_cpu1M_dev_$SMPC__SERVER_COORDINATION__PARTY_ID" \
  --target-db-size 577316
echo "cpu1M init done"
/bin/init-single-db \
    --party-id $SMPC__SERVER_COORDINATION__PARTY_ID \
    --source "/tmp/irises.ndjson" \
    --db-url "$SMPC__CPU_DATABASE__URL" \
    --db-schema "genesis_gpu1M_dev_$SMPC__SERVER_COORDINATION__PARTY_ID" \
    --target-db-size 577316
echo "gpu1M init done"
/bin/graph-mem-cli --db-url $SMPC__CPU_DATABASE__URL --schema genesis_cpu1M_dev_$SMPC__SERVER_COORDINATION__PARTY_ID --file /tmp/graph.dat restore-db
echo "restore graph done"

psql "$SMPC__CPU_DATABASE__URL" -c "
INSERT INTO persistent_state (domain, \"key\", \"value\")
VALUES ('genesis', 'last_indexed_iris_id', '577316')
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
