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
