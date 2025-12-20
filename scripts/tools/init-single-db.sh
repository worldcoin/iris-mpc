#!/bin/sh

echo "starting"
/bin/init-single-db \
  --party-id $SMPC__SERVER_COORDINATION__PARTY_ID \
  --source "/opt/irises.ndjson" \
  --db-url "$SMPC__CPU_DATABASE__URL" \
  --db-schema "genesis_cpu1M_dev_$SMPC__SERVER_COORDINATION__PARTY_ID" \
  --target-db-size 1048576
echo "cpu1M init done"
/bin/init-single-db \
    --party-id $SMPC__SERVER_COORDINATION__PARTY_ID \
    --source "/opt/irises.ndjson" \
    --db-url "$SMPC__CPU_DATABASE__URL" \
    --db-schema "genesis_gpu1M_dev_$SMPC__SERVER_COORDINATION__PARTY_ID" \
    --target-db-size 1048576
echo "gpu1M init done"
/bin/init-single-db \
    --party-id $SMPC__SERVER_COORDINATION__PARTY_ID \
    --source "/opt/irises-1k.ndjson" \
    --db-url "$SMPC__CPU_DATABASE__URL" \
    --db-schema "genesis_gpu1M_dev_$SMPC__SERVER_COORDINATION__PARTY_ID" \
    --target-db-size 1049576
    --skip 0
echo "100K irises added to gpu"
/bin/graph-mem-cli --db-url $SMPC__CPU_DATABASE__URL --schema genesis_gpu1M_dev_$SMPC__SERVER_COORDINATION__PARTY_ID --file /opt/graph.dat restore-db
echo "restore graph done"

shutdown_requested=false

shutdown_handler() {
    echo "Shutdown requested"
    shutdown_requested=true
}

trap shutdown_handler SIGTERM SIGINT SIGHUP SIGQUIT

while [ "$shutdown_requested" = false ]; do
    sleep infinity &  # Run in background
    wait $!           # Wait for the background process
done
