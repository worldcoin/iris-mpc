#!/bin/sh
/bin/init-single-db \
  --party-id $SMPC__SERVER_COORDINATION__PARTY_ID \
  --source "/opt/irises.ndjson" \
  --db-url "$SMPC__CPU_DATABASE__URL" \
  --db-schema "genesis_cpu1M_dev_$SMPC__SERVER_COORDINATION__PARTY_ID" \
  --target-db-size 1048576 && \

/bin/init-single-db \
    --party-id $SMPC__SERVER_COORDINATION__PARTY_ID \
    --source "/opt/irises.ndjson" \
    --db-url "$SMPC__CPU_DATABASE__URL" \
    --db-schema "genesis_gpu1M_dev_$SMPC__SERVER_COORDINATION__PARTY_ID" \
    --target-db-size 1048576
