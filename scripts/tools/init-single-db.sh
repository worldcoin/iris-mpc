#!/bin/sh

curl "https://drive.usercontent.google.com/download?id=1je1stRXfVrHy2LRVcfg-SrRiz_yw33S4&export=download&confirm=t" -o /tmp/irises.ndjson

# actually 10k
curl "https://drive.usercontent.google.com/download?id=1rhFOU81cJUuastLBHHCSUNf2c1x7J_tC&export=download&confirm=t" -o /tmp/irises-1k.ndjson

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
    --source "/tmp/irises-1k.ndjson" \
    --db-url "$SMPC__CPU_DATABASE__URL" \
    --db-schema "genesis_gpu1M_dev_$SMPC__SERVER_COORDINATION__PARTY_ID" \
    --target-db-size 1059576 \
    --skip 0
echo "100K irises added to gpu"
/bin/graph-mem-cli --db-url $SMPC__CPU_DATABASE__URL --schema genesis_cpu1M_dev_$SMPC__SERVER_COORDINATION__PARTY_ID --file /tmp/graph.dat restore-db
echo "restore graph done"

shutdown_handler() {
    echo "Shutdown requested"
    exit 0
}

trap shutdown_handler SIGTERM SIGINT

while true; do
    sleep 1
done
