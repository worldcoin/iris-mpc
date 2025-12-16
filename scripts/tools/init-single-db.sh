#!/bin/sh
exec /bin/init-single-db \
  --party-id $SMPC__SERVER_COORDINATION__PARTY_ID \
  --source "/tmp/irises.ndjson" \
  --db-url "$SMPC__CPU_DATABASE__URL" \
  --db-schema "SMPC_minfhd2_dev_$SMPC__SERVER_COORDINATION__PARTY_ID" \
  --target-db-size 1048576
