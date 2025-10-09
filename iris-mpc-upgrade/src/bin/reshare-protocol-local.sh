#!/usr/bin/env bash

rm -rf "*.log"

docker-compose down --remove-orphans
docker-compose up -d

sleep 1

aws_local() {
    AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1 aws --endpoint-url=http://${LOCALSTACK_HOST:-localhost}:4566 "$@"
}

key1_metadata=$(aws_local kms create-key --region us-east-1 --description "Key for Party1" --key-spec ECC_NIST_P256 --key-usage KEY_AGREEMENT)
echo "Created key1: $key1_metadata"
key1_arn=$(echo "$key1_metadata" | jq ".KeyMetadata.Arn" -r)
echo "Key1 ARN: $key1_arn"
key2_metadata=$(aws_local kms create-key --region us-east-1 --description "Key for Party2" --key-spec ECC_NIST_P256 --key-usage KEY_AGREEMENT)
echo "Created key2: $key2_metadata"
key2_arn=$(echo "$key2_metadata" | jq ".KeyMetadata.Arn" -r)
echo "Key2 ARN: $key2_arn"

sleep 1

cargo build --release --bin seed-v2-dbs --bin reshare-server --bin reshare-client



TARGET_DIR=$(cargo metadata --format-version 1 | jq ".target_directory" -r)

$TARGET_DIR/release/seed-v2-dbs --db-url-party1 postgres://postgres:postgres@localhost:6200 --db-url-party2 postgres://postgres:postgres@localhost:6201 --db-url-party3 postgres://postgres:postgres@localhost:6202 --schema-name-party1 SMPC_testing_0 --schema-name-party2 SMPC_testing_1 --schema-name-party3 SMPC_testing_2 --fill-to 10000 --batch-size 100

$TARGET_DIR/release/reshare-server --party-id 2 --sender1-party-id 0 --sender2-party-id 1 --bind-addr 0.0.0.0:7000 --environment testing --db-url postgres://postgres:postgres@localhost:6203 --batch-size 100 & > reshare-server.log

sleep 5

AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1 AWS_ENDPOINT_URL=http://${LOCALSTACK_HOST:-localhost}:4566 $TARGET_DIR/release/reshare-client --party-id 0 --other-party-id 1 --target-party-id 2 --server-url http://localhost:7000 --environment testing --db-url postgres://postgres:postgres@localhost:6200 --db-start 1 --db-end 10001 --batch-size 100 --my-kms-key-arn $key1_arn --other-kms-key-arn $key2_arn --reshare-run-session-id testrun1 & > reshare-client-0.log

AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1 AWS_ENDPOINT_URL=http://${LOCALSTACK_HOST:-localhost}:4566 $TARGET_DIR/release/reshare-client --party-id 1 --other-party-id 0 --target-party-id 2 --server-url http://localhost:7000 --environment testing --db-url postgres://postgres:postgres@localhost:6201 --db-start 1 --db-end 10001 --batch-size 100 --my-kms-key-arn $key2_arn --other-kms-key-arn $key1_arn --reshare-run-session-id testrun1 > reshare-client-1.log

sleep 5
killall reshare-server
