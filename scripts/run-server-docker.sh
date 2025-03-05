#!/usr/bin/env bash
set -e

NODE_ID="$1"
if [ -z "$NODE_ID" ]; then
  echo "Usage: run-server.sh <node_id>"
  exit 1
fi

# needs to run twice to create the keys with both AWSCURRENT and AWSPREVIOUS states
/bin/key-manager --region "$AWS_REGION" --endpoint-url "$AWS_ENDPOINT_URL" --node-id "$NODE_ID" --env dev rotate --public-key-bucket-name wf-dev-public-keys
/bin/key-manager --region "$AWS_REGION" --endpoint-url "$AWS_ENDPOINT_URL" --node-id "$NODE_ID" --env dev rotate --public-key-bucket-name wf-dev-public-keys

# Set the stack size to 100MB to receive large messages.
export RUST_MIN_STACK=104857600

/bin/iris-mpc-gpu