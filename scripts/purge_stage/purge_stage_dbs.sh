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
  CLUSTER_NAME="ampc-hnsw"
  NAMESPACE="ampc-hnsw"
  echo "Using CPU secret: $SECRET_NAME"
  echo "Using CPU cluster name: $CLUSTER_NAME"
  echo "Using CPU namespace: $NAMESPACE"
elif [ "$CLEANUP_TYPE" == "gpu" ]; then
  SECRET_NAME="stage/iris-mpc/rds-aurora-master-password"
  CLUSTER_NAME="smpcv2"
  NAMESPACE="iris-mpc"
  echo "Using GPU secret: $SECRET_NAME" 
  echo "Using GPU cluster name: $CLUSTER_NAME"
  echo "Using GPU namespace: $NAMESPACE"
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
  local ACCOUNT_ID=$3
  local CLUSTER_NAME=$4
  local NAMESPACE=$5


  if [ -z "$DATABASE_URL" ] || [ -z "$PARTY_ID" ] || [ -z "$ACCOUNT_ID" ] || [ -z "$CLUSTER_NAME" ] || [ -z "$NAMESPACE" ]; then
    echo "Database URL, party ID, account ID, cluster name and namespace are required"
    exit 1
  fi

  CLUSTER="arn:aws:eks:eu-north-1:$ACCOUNT_ID:cluster/$CLUSTER_NAME-$PARTY_ID-stage"
  echo "Cleaning database for $PARTY_ID with cluster name $CLUSTER_NAME"
  
  # Switch to the appropriate Kubernetes context
  kubectx $CLUSTER
  kubens $NAMESPACE
  
  # Create and use a temporary pod for database operations
  kubectl apply -f db-cleaner-helper-pod-$NAMESPACE.yaml
  echo "Waiting 10s for db-cleaner pod to be ready..."
  sleep 10

  # Remove the last segment after the last slash
  DATABASE_URL_WITH_CORRECT_NAME=$(echo "$DATABASE_URL" | sed 's|/[^/]*$||')

  echo "Cleaning Database for URL: $DATABASE_URL_WITH_CORRECT_NAME"
  
  # Execute database cleanup commands
  kubectl exec -it db-cleaner -- bash -c "psql -H $DATABASE_URL_WITH_CORRECT_NAME -c 'SET search_path TO \"SMPC_stage_$PARTY_ID\"; TRUNCATE irises RESTART IDENTITY;'"
  kubectl exec -it db-cleaner -- bash -c "psql -H $DATABASE_URL_WITH_CORRECT_NAME -c 'SET search_path TO \"SMPC_stage_$PARTY_ID\"; TRUNCATE sync RESTART IDENTITY;'"
  kubectl exec -it db-cleaner -- bash -c "psql -H $DATABASE_URL_WITH_CORRECT_NAME -c 'SET search_path TO \"SMPC_stage_$PARTY_ID\"; TRUNCATE results RESTART IDENTITY;'"
  kubectl exec -it db-cleaner -- bash -c "psql -H $DATABASE_URL_WITH_CORRECT_NAME -c 'SET search_path TO \"SMPC_stage_$PARTY_ID\"; TRUNCATE modifications RESTART IDENTITY;'"
  kubectl exec -it db-cleaner -- bash -c "psql -H $DATABASE_URL_WITH_CORRECT_NAME -c 'SET search_path TO \"SMPC_stage_$PARTY_ID\"; TRUNCATE hawk_graph_entry RESTART IDENTITY;'"
  kubectl exec -it db-cleaner -- bash -c "psql -H $DATABASE_URL_WITH_CORRECT_NAME -c 'SET search_path TO \"SMPC_stage_$PARTY_ID\"; TRUNCATE hawk_graph_links RESTART IDENTITY;'"
  
  # Clean up and restart deployment
  kubectl delete pod --force db-cleaner
  kubectl rollout restart deployment $NAMESPACE -n $NAMESPACE
  
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

clean_mpc_database "$MPC_0_DATABASE_URL" "0" $MPC_0_STAGE_ACCOUNT_ID $CLUSTER_NAME $NAMESPACE
clean_mpc_database "$MPC_1_DATABASE_URL" "1" $MPC_1_STAGE_ACCOUNT_ID $CLUSTER_NAME $NAMESPACE
clean_mpc_database "$MPC_2_DATABASE_URL" "2" $MPC_2_STAGE_ACCOUNT_ID $CLUSTER_NAME $NAMESPACE

echo "All database cleanup operations completed successfully."
