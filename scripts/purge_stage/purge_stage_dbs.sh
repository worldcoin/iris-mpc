#!/usr/bin/env bash

source ./accounts_checker.sh

if [ -z "$1" ]; then
  printf "\nError: Cleanup type parameter is required\n"
  printf "Usage: %s <cleanup_type>\n" "$0"
  printf "Available cleanup types: gpu, cpu\n"
  exit 1
fi

CLEANUP_TYPE=$1

# Set the appropriate secret name based on cleanup type
if [ "$CLEANUP_TYPE" == "cpu" ]; then
  SECRET_NAME="stage/hnsw-mpc/rds-aurora-master-password"
  echo "Using GPU secret: $SECRET_NAME"
elif [ "$CLEANUP_TYPE" == "gpu" ]; then
  SECRET_NAME="stage/iris-mpc/rds-aurora-master-password"
  echo "Using CPU secret: $SECRET_NAME"
else
  printf "\nUnknown cleanup type: %s\n" "$CLEANUP_TYPE"
  printf "Available cleanup types: gpu, cpu\n"
  exit 1
fi

REGION="eu-north-1"
SECRET_KEY="DATABASE_URL"

get_aws_secret() {
  local SECRET_NAME=$1
  local PARTY_ID=$2
  local PROFILE="worldcoin-smpcv-io-$PARTY_ID"

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

clean_mpc_database() {
  local DATABASE_URL=$1
  local PARTY_ID=$2

  if [ -z "$DATABASE_URL" ] || [ -z "$PARTY_ID" ]; then
    echo "Database URL and party ID are required"
    exit 1
  fi

  echo "Cleaning database for $PARTY_ID..."
  
  # Switch to the appropriate Kubernetes context
  kubectx arn:aws:eks:eu-north-1:024848486749:cluster/smpcv2-${PARTY_ID}-stage || kubectx smpc-io-stage-$PARTY_ID
  kubens iris-mpc
  
  # Create and use a temporary pod for database operations
  kubectl apply -f db-cleaner-helper-pod.yaml
  echo "Waiting 10s for db-cleaner pod to be ready..."
  sleep 10
  
  # Execute database cleanup commands
  kubectl exec -it db-cleaner -- bash -c "psql -H $DATABASE_URL -c 'SET search_path TO \"SMPC_stage_$PARTY_ID\"; TRUNCATE irises RESTART IDENTITY;'"
  kubectl exec -it db-cleaner -- bash -c "psql -H $DATABASE_URL -c 'SET search_path TO \"SMPC_stage_$PARTY_ID\"; TRUNCATE sync RESTART IDENTITY;'"
  kubectl exec -it db-cleaner -- bash -c "psql -H $DATABASE_URL -c 'SET search_path TO \"SMPC_stage_$PARTY_ID\"; TRUNCATE results RESTART IDENTITY;'"
  kubectl exec -it db-cleaner -- bash -c "psql -H $DATABASE_URL -c 'SET search_path TO \"SMPC_stage_$PARTY_ID\"; TRUNCATE modifications RESTART IDENTITY;'"
  kubectl exec -it db-cleaner -- bash -c "psql -H $DATABASE_URL -c 'SET search_path TO \"SMPC_stage_$PARTY_ID\"; TRUNCATE hawk_graph_entry RESTART IDENTITY;'"
  kubectl exec -it db-cleaner -- bash -c "psql -H $DATABASE_URL -c 'SET search_path TO \"SMPC_stage_$PARTY_ID\"; TRUNCATE hawk_graph_links RESTART IDENTITY;'"
  
  # Clean up and restart deployment
  kubectl delete pod --force db-cleaner
  kubectl rollout restart deployment iris-mpc -n iris-mpc
  
  echo "Cleanup completed for $PARTY_ID"
}


MPC_0_DATABASE_URL=$(get_aws_secret "$SECRET_NAME"  "0")
MPC_0_DATABASE_URL=${MPC_0_DATABASE_URL%/iris_mpc}

MPC_1_DATABASE_URL=$(get_aws_secret "$SECRET_NAME"  "1")
MPC_1_DATABASE_URL=${MPC_1_DATABASE_URL%/iris_mpc}

MPC_2_DATABASE_URL=$(get_aws_secret "$SECRET_NAME" "2")
MPC_2_DATABASE_URL=${MPC_2_DATABASE_URL%/iris_mpc}


if [ -z "$MPC_0_DATABASE_URL" ] || [ -z "$MPC_1_DATABASE_URL" ] || [ -z "$MPC_2_DATABASE_URL" ]; then
  echo "One or more database URLs are empty, please check your AWS Secrets."
  exit 1
fi

clean_mpc_database "$MPC_0_DATABASE_URL" "0"
clean_mpc_database "$MPC_1_DATABASE_URL" "1"
clean_mpc_database "$MPC_2_DATABASE_URL" "2"

echo "All database cleanup operations completed successfully."
