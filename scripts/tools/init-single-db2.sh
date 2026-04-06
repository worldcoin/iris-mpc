#!/bin/sh

# 1M irises that match the graph.dat
curl "https://drive.usercontent.google.com/download?id=1je1stRXfVrHy2LRVcfg-SrRiz_yw33S4&export=download&confirm=t" -o /tmp/irises.ndjson

curl "https://drive.usercontent.google.com/download?id=1whFu0GIezDA2_YD9eMr9cY60oU0BF_Ao&export=download&confirm=t" -o /tmp/irises2M.ndjson

curl "https://drive.usercontent.google.com/download?id=1vjswOMB7Yn-f7TDOqfzQr1_ZS-ubSg4E&export=download&confirm=t" -o /tmp/graph.dat



echo "starting"
/bin/init-single-db \
  --party-id $SMPC__SERVER_COORDINATION__PARTY_ID \
  --source "/tmp/irises.ndjson" \
  --db-url "$SMPC__CPU_DATABASE__URL" \
  --db-schema "genesis_cpu1M_dev_$SMPC__SERVER_COORDINATION__PARTY_ID" \
  --target-db-size 1048576
echo "cpu1M init done"
/bin/init-single-db \
    --party-id $SMPC__SERVER_COORDINATION__PARTY_ID \
    --source "/tmp/irises.ndjson" \
    --db-url "$SMPC__CPU_DATABASE__URL" \
    --db-schema "genesis_gpu1M_dev_$SMPC__SERVER_COORDINATION__PARTY_ID" \
    --target-db-size 1048576
echo "gpu1M init done"
/bin/init-single-db \
    --party-id $SMPC__SERVER_COORDINATION__PARTY_ID \
    --source "/tmp/irises2M.ndjson" \
    --db-url "$SMPC__CPU_DATABASE__URL" \
    --db-schema "genesis_gpu1M_dev_$SMPC__SERVER_COORDINATION__PARTY_ID" \
    --target-db-size 2097152 \
    --skip 0
echo "1M irises added to gpu"
/bin/graph-mem-cli --db-url $SMPC__CPU_DATABASE__URL --schema genesis_cpu1M_dev_$SMPC__SERVER_COORDINATION__PARTY_ID --file /tmp/graph.dat restore-db
echo "restore graph done"

psql "$SMPC__CPU_DATABASE__URL" -c "
INSERT INTO persistent_state (domain, \"key\", \"value\")
VALUES ('genesis', 'last_indexed_iris_id', '1048576')
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
