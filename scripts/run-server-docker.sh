#!/usr/bin/env bash
set -e

# Arg :: Node ID :: MPC node ordinal identifier.
NODE_ID="$1"
if [ -z "$NODE_ID" ]; then
  echo "Usage: run-server-docker.sh <node_id> <mode>"
  exit 1
fi

# Arg :: Binary :: Binary to run [standard | genesis].
BINARY="$2"
if [ -z "$BINARY" ]; then
  echo "Usage: run-server-docker.sh <node_id> <binary>"
  exit 1
fi

# needs to run twice to create the keys with both AWSCURRENT and AWSPREVIOUS states
/bin/key-manager \
    --region "$AWS_REGION" \
    --endpoint-url "$AWS_ENDPOINT_URL" \
    --node-id "$NODE_ID" \
    --env dev rotate \
    --public-key-bucket-name wf-dev-public-keys
/bin/key-manager \
    --region "$AWS_REGION" \
    --endpoint-url "$AWS_ENDPOINT_URL" \
    --node-id "$NODE_ID" \
    --env dev rotate \
    --public-key-bucket-name wf-dev-public-keys

# Set the stack size to 100MB to receive large messages.
export RUST_MIN_STACK=104857600

if [ "$BINARY" == "genesis" ]; then
    /bin/iris-mpc-hawk-genesis
else
    /bin/iris-mpc-hawk
fi
