#!/usr/bin/env bash

set -euo pipefail

rm -rf "*.log"

docker-compose -f docker-compose.rand.yaml down --remove-orphans -v
docker-compose -f docker-compose.rand.yaml up -d

sleep 10

aws_local() {
    AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1 aws --endpoint-url=http://${LOCALSTACK_HOST:-localhost}:4566 "$@"
}

# Create a bucket for public keys
BUCKET_NAME=wf-smpcv2-pki-testing
aws_local s3api create-bucket --bucket $BUCKET_NAME --region us-east-1

# Build rust binaries and grab target dir
cargo build -p iris-mpc-bins --release --bin seed-v2-dbs --bin rerandomize-db

TARGET_DIR=$(cargo metadata --format-version 1 | jq ".target_directory" -r)


# Seed DBs with initial data
echo "Seeding DBs"
$TARGET_DIR/release/seed-v2-dbs --db-url-party1 postgres://postgres:postgres@localhost:6200 --db-url-party2 postgres://postgres:postgres@localhost:6201 --db-url-party3 postgres://postgres:postgres@localhost:6202 --schema-name-party1 SMPC_testing_0 --schema-name-party2 SMPC_testing_1 --schema-name-party3 SMPC_testing_2 --fill-to 10000 --batch-size 100
echo "Seeding complete"


# Set AWS env vars for localstack
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1
export AWS_ENDPOINT_URL="http://127.0.0.1:4566"


# Set some common env vars for the rerandomize-db tool that are the same for all parties
export ENVIRONMENT="testing"
export PUBLIC_KEY_BUCKET_NAME=$BUCKET_NAME

awslocal secretsmanager create-secret --name dev/iris-mpc-db-rerandomization/tripartite-ecdh-private-key-0 --description "Secret for 0" --secret-string "{\"private-key\":\"\"}" --region us-east-1
awslocal secretsmanager create-secret --name dev/iris-mpc-db-rerandomization/tripartite-ecdh-private-key-1 --description "Secret for 1" --secret-string "{\"private-key\":\"\"}" --region us-east-1
awslocal secretsmanager create-secret --name dev/iris-mpc-db-rerandomization/tripartite-ecdh-private-key-2 --description "Secret for 2" --secret-string "{\"private-key\":\"\"}" --region us-east-1


# Stage 1: Generate a keypair for each party and upload public keys to S3, for the tripartite DH
echo "Key generation"
$TARGET_DIR/release/rerandomize-db key-gen --party-id 0 || echo "Key generation for party 0 may have failed if the key already exists, continuing..."
$TARGET_DIR/release/rerandomize-db key-gen --party-id 1 || echo "Key generation for party 1 may have failed if the key already exists, continuing..."
$TARGET_DIR/release/rerandomize-db key-gen --party-id 2 || echo "Key generation for party 2 may have failed if the key already exists, continuing..."
echo "Key generation complete"


# Stage 2: Rerandomize DBs, in two parts each to simulate chunking
echo "Rerandomizing DB 0 part 1"
$TARGET_DIR/release/rerandomize-db rerandomize-db \
  --party-id 0 \
  --source-db-url postgres://postgres:postgres@localhost:6200 \
  --dest-db-url postgres://postgres:postgres@localhost:6203 \
  --source-schema-name SMPC_testing_0 \
  --dest-schema-name SMPC_testing_rerand_0 \
  --range-min 1 \
  --range-max-inclusive 5000

echo "Rerandomizing DB 1 part 1"
$TARGET_DIR/release/rerandomize-db rerandomize-db \
  --party-id 1 \
  --source-db-url postgres://postgres:postgres@localhost:6201 \
  --dest-db-url postgres://postgres:postgres@localhost:6204 \
  --source-schema-name SMPC_testing_1 \
  --dest-schema-name SMPC_testing_rerand_1 \
  --range-min 1 \
  --range-max-inclusive 5000

echo "Rerandomizing DB 2 part 1" --range-min 1 
$TARGET_DIR/release/rerandomize-db rerandomize-db \
  --party-id 2 \
  --source-db-url postgres://postgres:postgres@localhost:6202 \
  --dest-db-url postgres://postgres:postgres@localhost:6205 \
  --source-schema-name SMPC_testing_2 \
  --dest-schema-name SMPC_testing_rerand_2 \
  --range-min 1 \
  --range-max-inclusive 5000

# Stage 2b: Second half of rerandomization, indices 5001-10000
echo "Rerandomizing DB 0 part 2"
$TARGET_DIR/release/rerandomize-db rerandomize-db \
  --party-id 0 \
  --source-db-url postgres://postgres:postgres@localhost:6200 \
  --dest-db-url postgres://postgres:postgres@localhost:6203 \
  --source-schema-name SMPC_testing_0 \
  --dest-schema-name SMPC_testing_rerand_0 \
  --range-min 5001

echo "Rerandomizing DB 1 part 2"
$TARGET_DIR/release/rerandomize-db rerandomize-db \
  --party-id 1 \
  --source-db-url postgres://postgres:postgres@localhost:6201 \
  --dest-db-url postgres://postgres:postgres@localhost:6204 \
  --source-schema-name SMPC_testing_1 \
  --dest-schema-name SMPC_testing_rerand_1 \
  --range-min 5001

echo "Rerandomizing DB 2 part 2"
$TARGET_DIR/release/rerandomize-db rerandomize-db \
  --party-id 2 \
  --source-db-url postgres://postgres:postgres@localhost:6202 \
  --dest-db-url postgres://postgres:postgres@localhost:6205 \
  --source-schema-name SMPC_testing_2 \
  --dest-schema-name SMPC_testing_rerand_2 \
  --range-min 5001

# Stage 3: Delete the private keys from secrets manager to prevent key reuse
# ATM this only deletes the secret keys, not the public keys in the bucket
echo "Deleting private keys from secrets manager, before:"
aws_local secretsmanager list-secrets --no-cli-pager
$TARGET_DIR/release/rerandomize-db key-cleanup --party-id 0
$TARGET_DIR/release/rerandomize-db key-cleanup --party-id 1
$TARGET_DIR/release/rerandomize-db key-cleanup --party-id 2
echo "Deleting private keys from secrets manager, after:"
aws_local secretsmanager list-secrets --no-cli-pager

# TEST only: Check that the rerandomized DBs have the same data as the original DBs (just in a different encoding)
echo "Checking Rerandomized DBs"
$TARGET_DIR/release/rerandomize-db rerandomize-check \
  --old-db-url-party-0 postgres://postgres:postgres@localhost:6200 \
  --old-db-url-party-1 postgres://postgres:postgres@localhost:6201 \
  --old-db-url-party-2 postgres://postgres:postgres@localhost:6202 \
  --old-schema-name-party-0 SMPC_testing_0 \
  --old-schema-name-party-1 SMPC_testing_1 \
  --old-schema-name-party-2 SMPC_testing_2 \
  --new-db-url-party-0 postgres://postgres:postgres@localhost:6203 \
  --new-db-url-party-1 postgres://postgres:postgres@localhost:6204 \
  --new-db-url-party-2 postgres://postgres:postgres@localhost:6205 \
  --new-schema-name-party-0 SMPC_testing_rerand_0 \
  --new-schema-name-party-1 SMPC_testing_rerand_1 \
  --new-schema-name-party-2 SMPC_testing_rerand_2
