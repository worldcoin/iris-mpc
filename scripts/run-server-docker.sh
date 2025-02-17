#!/usr/bin/env bash
set -e



NODE_ID="$1"
if [ -z "$NODE_ID" ]; then
  echo "Usage: run-server.sh <node_id>"
  exit 1
fi

INIT_SERVERS=false
if [ "$2" == "--init-servers" ]; then
  INIT_SERVERS=true
fi


if [ "$INIT_SERVERS" = true ]; then
  /bin/key-manager --region "$AWS_REGION" --endpoint-url "$AWS_ENDPOINT_URL" --node-id "$NODE_ID" --env dev rotate --public-key-bucket-name wf-dev-public-keys
  /bin/key-manager --region "$AWS_REGION" --endpoint-url "$AWS_ENDPOINT_URL" --node-id "$NODE_ID" --env dev rotate --public-key-bucket-name wf-dev-public-keys

fi

/bin/server-hawk