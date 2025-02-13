#!/usr/bin/env bash
set -e

# needs to run twice to create the keys with both AWSCURRENT and AWSPREVIOUS states
/bin/key-manager --region us-east-1 --node-id "$NODE_ID" --env dev rotate --public-key-bucket-name wf-dev-public-keys
/bin/key-manager --region us-east-1 --node-id "$NODE_ID" --env dev rotate --public-key-bucket-name wf-dev-public-keys

/bin/server-hawk