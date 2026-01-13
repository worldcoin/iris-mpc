#!/bin/sh


curl "https://drive.usercontent.google.com/download?id=1whFu0GIezDA2_YD9eMr9cY60oU0BF_Ao&export=download&confirm=t" -o /tmp/irises.ndjson

echo "starting"
/bin/init-single-db \
    --party-id $SMPC__SERVER_COORDINATION__PARTY_ID \
    --source "/tmp/irises.ndjson" \
    --db-url "$SMPC__CPU_DATABASE__URL" \
    --db-schema "genesis_gpu1M_dev_$SMPC__SERVER_COORDINATION__PARTY_ID" \
    --target-db-size 2107152 \
    --skip 0
echo "1M irises added to gpu"

shutdown_handler() {
    echo "Shutdown requested"
    exit 0
}

trap shutdown_handler SIGTERM SIGINT

while true; do
    sleep 1
done
