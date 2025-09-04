#!/usr/bin/env bash

set -euo pipefail

rm -rf "*.log"

docker-compose -f docker-compose.rand.yaml down --remove-orphans -v
docker-compose -f docker-compose.rand.yaml up -d

sleep 1

aws_local() {
    AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1 aws --endpoint-url=http://${LOCALSTACK_HOST:-localhost}:4566 "$@"
}

# key1_metadata=$(aws_local kms create-key --region us-east-1 --description "Key for Party1" --key-spec ECC_NIST_P256 --key-usage KEY_AGREEMENT)
# echo "Created key1: $key1_metadata"
# key1_arn=$(echo "$key1_metadata" | jq ".KeyMetadata.Arn" -r)
# echo "Key1 ARN: $key1_arn"
# key2_metadata=$(aws_local kms create-key --region us-east-1 --description "Key for Party2" --key-spec ECC_NIST_P256 --key-usage KEY_AGREEMENT)
# echo "Created key2: $key2_metadata"
# key2_arn=$(echo "$key2_metadata" | jq ".KeyMetadata.Arn" -r)
# echo "Key2 ARN: $key2_arn"

sleep 1

cargo build --release --bin seed-v2-dbs --bin rerandomize-db --bin randomize_check

TARGET_DIR=$(cargo metadata --format-version 1 | jq ".target_directory" -r)

echo "Seeding DBs"
$TARGET_DIR/release/seed-v2-dbs --db-url-party1 postgres://postgres:postgres@localhost:6200 --db-url-party2 postgres://postgres:postgres@localhost:6201 --db-url-party3 postgres://postgres:postgres@localhost:6202 --schema-name-party1 SMPC_testing_0 --schema-name-party2 SMPC_testing_1 --schema-name-party3 SMPC_testing_2 --fill-to 10000 --batch-size 100
echo "Seeding complete"


echo "Rerandomizing DB 0"
$TARGET_DIR/release/rerandomize-db --party-id 0 --source-db-url postgres://postgres:postgres@localhost:6200 --dest-db-url postgres://postgres:postgres@localhost:6203 --source-schema-name SMPC_testing_0 --dest-schema-name SMPC_testing_0 --master-seed "asdfasdfasdfasdfasdfasdfasdfasdf"

echo "Rerandomizing DB 1"
$TARGET_DIR/release/rerandomize-db --party-id 1 --source-db-url postgres://postgres:postgres@localhost:6201 --dest-db-url postgres://postgres:postgres@localhost:6204 --source-schema-name SMPC_testing_1 --dest-schema-name SMPC_testing_1 --master-seed "asdfasdfasdfasdfasdfasdfasdfasdf"

echo "Rerandomizing DB 2"
$TARGET_DIR/release/rerandomize-db --party-id 2 --source-db-url postgres://postgres:postgres@localhost:6202 --dest-db-url postgres://postgres:postgres@localhost:6205 --source-schema-name SMPC_testing_2 --dest-schema-name SMPC_testing_2 --master-seed "asdfasdfasdfasdfasdfasdfasdfasdf"

echo "Checking Rerandomized DBs"
$TARGET_DIR/release/randomize_check
