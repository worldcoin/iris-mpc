#!/bin/bash
set -e

# needs to run twice to create the keys with both AWSCURRENT and AWSPREVIOUS states
AWS_ENDPOINT_URL=http://127.0.0.1:4566 cargo run --release --bin key-manager --  --region us-east-1 --node-id 0 --env dev rotate --public-key-bucket-name wf-dev-public-keys
AWS_ENDPOINT_URL=http://127.0.0.1:4566 cargo run --release --bin key-manager --  --region us-east-1 --node-id 1 --env dev rotate --public-key-bucket-name wf-dev-public-keys
AWS_ENDPOINT_URL=http://127.0.0.1:4566 cargo run --release --bin key-manager --  --region us-east-1 --node-id 2 --env dev rotate --public-key-bucket-name wf-dev-public-keys

AWS_ENDPOINT_URL=http://127.0.0.1:4566 cargo run --release --bin key-manager --  --region us-east-1 --node-id 0 --env dev rotate --public-key-bucket-name wf-dev-public-keys
AWS_ENDPOINT_URL=http://127.0.0.1:4566 cargo run --release --bin key-manager --  --region us-east-1 --node-id 1 --env dev rotate --public-key-bucket-name wf-dev-public-keys
AWS_ENDPOINT_URL=http://127.0.0.1:4566 cargo run --release --bin key-manager --  --region us-east-1 --node-id 2 --env dev rotate --public-key-bucket-name wf-dev-public-keys
