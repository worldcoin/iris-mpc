#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rm -rf "*.log"

docker-compose -f "$SCRIPT_DIR/docker-compose.rand.yaml" down --remove-orphans -v
docker-compose -f "$SCRIPT_DIR/docker-compose.rand.yaml" up -d

sleep 10

aws_local() {
    AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1 \
    aws --endpoint-url=http://${LOCALSTACK_HOST:-localhost}:4566 "$@"
}

# Create S3 bucket for rerand coordination markers
BUCKET_NAME=wf-smpcv2-rerand-testing
aws_local s3api create-bucket --bucket $BUCKET_NAME --region us-east-1

# Build binaries
cargo build -p iris-mpc-bins --release --bin seed-v2-dbs --bin rerandomize-db

TARGET_DIR=$(cargo metadata --format-version 1 | jq ".target_directory" -r)

# Set AWS env vars for localstack
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1
export AWS_ENDPOINT_URL="http://127.0.0.1:4566"

export ENVIRONMENT="testing"

# Seed DBs with initial data (using first 3 new-db containers as live DBs)
echo "=== Seeding DBs ==="
$TARGET_DIR/release/seed-v2-dbs \
  --db-url-party-0 postgres://postgres:postgres@localhost:6200 \
  --db-url-party-1 postgres://postgres:postgres@localhost:6201 \
  --db-url-party-2 postgres://postgres:postgres@localhost:6202 \
  --schema-name-party-0 SMPC_testing_0 \
  --schema-name-party-1 SMPC_testing_1 \
  --schema-name-party-2 SMPC_testing_2 \
  --fill-to 1000 \
  --batch-size 100
echo "Seeding complete"

# Run continuous rerandomization for all 3 parties in parallel
echo "=== Starting continuous rerandomization ==="
COMMON_ARGS="--chunk-size 200 --chunk-delay-secs 1 --s3-poll-interval-ms 2000 --safety-buffer-ids 0"

$TARGET_DIR/release/rerandomize-db rerandomize-continuous \
  --party-id 0 \
  --db-url postgres://postgres:postgres@localhost:6200 \
  --schema-name SMPC_testing_0 \
  --s3-bucket $BUCKET_NAME \
  --healthcheck-port 3010 \
  $COMMON_ARGS &
PID_0=$!

$TARGET_DIR/release/rerandomize-db rerandomize-continuous \
  --party-id 1 \
  --db-url postgres://postgres:postgres@localhost:6201 \
  --schema-name SMPC_testing_1 \
  --s3-bucket $BUCKET_NAME \
  --healthcheck-port 3011 \
  $COMMON_ARGS &
PID_1=$!

$TARGET_DIR/release/rerandomize-db rerandomize-continuous \
  --party-id 2 \
  --db-url postgres://postgres:postgres@localhost:6202 \
  --schema-name SMPC_testing_2 \
  --s3-bucket $BUCKET_NAME \
  --healthcheck-port 3012 \
  $COMMON_ARGS &
PID_2=$!

echo "Rerand servers started: PIDs $PID_0, $PID_1, $PID_2"
echo "Waiting for one epoch to complete (watching for completion markers in S3)..."

# Poll until epoch 0 completion markers exist for all parties
MAX_WAIT=300
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    COMPLETE=true
    for P in 0 1 2; do
        KEY="rerand/epoch-0/party-${P}/complete"
        if ! aws_local s3api head-object --bucket $BUCKET_NAME --key "$KEY" >/dev/null 2>&1; then
            COMPLETE=false
            break
        fi
    done
    if [ "$COMPLETE" = true ]; then
        echo "=== Epoch 0 completed! ==="
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "Waiting... ($ELAPSED s)"
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: Epoch 0 did not complete within ${MAX_WAIT}s"
fi

# Stop the rerand servers
kill $PID_0 $PID_1 $PID_2 2>/dev/null || true
wait $PID_0 $PID_1 $PID_2 2>/dev/null || true

echo "=== Continuous rerandomization test finished ==="
