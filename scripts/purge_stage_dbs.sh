#!/usr/bin/env bash

source accounts_checker.sh

get_aws_secret() {
  local SECRET_NAME=$1
  local REGION=$2
  local SECRET_KEY=$3
  local PROFILE=$4

  # Fetch the secret value from AWS Secrets Manager
  SECRET_VALUE=$(aws secretsmanager get-secret-value --profile "$PROFILE" --secret-id "$SECRET_NAME" --region "$REGION" --query SecretString --output text)

  # Check if the secret value was retrieved successfully
  if [ -z "$SECRET_VALUE" ]; then
    echo "Failed to retrieve secret: $SECRET_NAME"
    exit 1
  fi

  # Extract the specific key (e.g., DATABASE_URL) from the JSON structure
  SECRET_KEY_VALUE=$(echo "$SECRET_VALUE" | jq -r ".${SECRET_KEY}")

  if [ -z "$SECRET_KEY_VALUE" ]; then
    echo "Failed to retrieve key: $SECRET_KEY from secret: $SECRET_NAME"
    exit 1
  fi

  echo "$SECRET_KEY_VALUE"
}

SECRET_NAME="stage/iris-mpc/rds-master-password"
REGION="eu-north-1"

MPC_1_DATABASE_URL=$(get_aws_secret "$SECRET_NAME" "$REGION" "DATABASE_URL" "worldcoin-smpcv2-1")
MPC_2_DATABASE_URL=$(get_aws_secret "$SECRET_NAME" "$REGION" "DATABASE_URL" "worldcoin-smpcv2-2")
MPC_3_DATABASE_URL=$(get_aws_secret "$SECRET_NAME" "$REGION" "DATABASE_URL" "worldcoin-smpcv2-3")

kubectx arn:aws:eks:eu-north-1:767397983205:cluster/mpc1-stage || kubectx smpcv2-stage-1
kubens iris-mpc
kubectl apply -f db-cleaner-helper-pod.yaml
echo "Waiting 10s for db-cleaner pod to be ready..."
sleep 10
kubectl exec -it db-cleaner -- bash -c "psql -H $MPC_1_DATABASE_URL -c 'SET search_path TO \"SMPC_stage_0\"; TRUNCATE irises RESTART IDENTITY;'"
kubectl exec -it db-cleaner -- bash -c "psql -H $MPC_1_DATABASE_URL -c 'SET search_path TO \"SMPC_stage_0\"; TRUNCATE sync RESTART IDENTITY;'"
kubectl exec -it db-cleaner -- bash -c "psql -H $MPC_1_DATABASE_URL -c 'SET search_path TO \"SMPC_stage_0\"; TRUNCATE results;'"
kubectl delete pod --force db-cleaner

kubectx arn:aws:eks:eu-north-1:381492197851:cluster/mpc2-stage || kubectx smpcv2-stage-2
kubens iris-mpc
kubectl apply -f db-cleaner-helper-pod.yaml
echo "Waiting 10s for db-cleaner pod to be ready..."
sleep 10
kubectl exec -it db-cleaner -- bash -c "psql -H $MPC_2_DATABASE_URL -c 'SET search_path TO \"SMPC_stage_1\"; TRUNCATE irises RESTART IDENTITY;'"
kubectl exec -it db-cleaner -- bash -c "psql -H $MPC_2_DATABASE_URL -c 'SET search_path TO \"SMPC_stage_1\"; TRUNCATE sync RESTART IDENTITY;'"
kubectl exec -it db-cleaner -- bash -c "psql -H $MPC_2_DATABASE_URL -c 'SET search_path TO \"SMPC_stage_1\"; TRUNCATE results;'"
kubectl delete pod --force db-cleaner

kubectx arn:aws:eks:eu-north-1:590184084615:cluster/mpc3-stage || kubectx smpcv2-stage-3
kubens iris-mpc
kubectl apply -f db-cleaner-helper-pod.yaml
echo "Waiting 10s for db-cleaner pod to be ready..."
sleep 10
kubectl exec -it db-cleaner -- bash -c "psql -H $MPC_3_DATABASE_URL -c 'SET search_path TO \"SMPC_stage_2\"; TRUNCATE irises RESTART IDENTITY;'"
kubectl exec -it db-cleaner -- bash -c "psql -H $MPC_3_DATABASE_URL -c 'SET search_path TO \"SMPC_stage_2\"; TRUNCATE sync RESTART IDENTITY;'"
kubectl exec -it db-cleaner -- bash -c "psql -H $MPC_3_DATABASE_URL -c 'SET search_path TO \"SMPC_stage_2\"; TRUNCATE results;'"
kubectl delete pod --force db-cleaner
