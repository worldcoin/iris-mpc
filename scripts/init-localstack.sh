#!/bin/bash
set -e

echo "Creating S3 bucket..."
awslocal s3 mb s3://wf-dev-public-keys

awslocal secretsmanager create-secret --name dev/iris-mpc/ecdh-private-key-0 --description "Secret for 0" --secret-string "{\"private-key\":\"\"}" --region us-east-1
awslocal secretsmanager create-secret --name dev/iris-mpc/ecdh-private-key-1 --description "Secret for 1" --secret-string "{\"private-key\":\"\"}" --region us-east-1
awslocal secretsmanager create-secret --name dev/iris-mpc/ecdh-private-key-2 --description "Secret for 2" --secret-string "{\"private-key\":\"\"}" --region us-east-1

awslocal kms create-key --region us-east-1 --key-spec ECC_NIST_P256 --key-usage KEY_AGREEMENT --tags "[{\"TagKey\":\"_custom_id_\",\"TagValue\":\"00000000-0000-0000-0000-000000000001\"}]"
awslocal kms create-key --region us-east-1 --key-spec ECC_NIST_P256 --key-usage KEY_AGREEMENT --tags "[{\"TagKey\":\"_custom_id_\",\"TagValue\":\"00000000-0000-0000-0000-000000000002\"}]"
awslocal kms create-key --region us-east-1 --key-spec ECC_NIST_P256 --key-usage KEY_AGREEMENT --tags "[{\"TagKey\":\"_custom_id_\",\"TagValue\":\"00000000-0000-0000-0000-000000000003\"}]"

echo "LocalStack initialization complete"