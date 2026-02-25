#!/bin/bash

# This script provides a way to update the targetRevision of an ArgoCD application in all SMPC stage and dev clusters.
# The script will loop through each cluster, port-forward ArgoCD server, authenticate to it, update the targetRevision of the application, and close port-forward.
# The script can be configured to use either CPU or GPU clusters by passing the appropriate argument. The default is GPU
# Usage examples:
#   ./scripts/tools/argo_update_revision.sh $(git branch --show-current)
#   ./scripts/tools/argo_update_revision.sh main
#   ./scripts/tools/argo_update_revision.sh <another-branch>
#   ./scripts/tools/argo_update_revision.sh <another-branch> cpu-dev
#
# Pre-requisites:
# 1. Ensure that you have the ArgoCD CLI installed on your local machine. (brew install argocd)
# 2. Ensure that you have the kubectl and kubectx installed on your local machine.
# 3. Ensure that your kubeconfig is configured to access the target clusters.


NEW_BRANCH=$1

MPCV2_TYPE=$2
# if MPCV2_TYPE is not provided, set it to "gpu"
if [ -z "$MPCV2_TYPE" ]; then
    MPCV2_TYPE="gpu"
fi

ENVIRONMENT_NAME=$3
# if ENVIRONMENT_NAME is not provided, set it to "stage"
if [ -z "$ENVIRONMENT_NAME" ]; then
    ENVIRONMENT_NAME="stage"
fi

ARGOCD_APP=$4

# if ARGOCD_APP is not provided, set it to either "iris-mpc" or "ampc-hnsw" based on MPCV2_TYPE, or a custom value
if [ -z "$ARGOCD_APP" ]; then
    if [ "$MPCV2_TYPE" == "gpu" ]; then
        ARGOCD_APP="iris-mpc"
    else
        ARGOCD_APP="ampc-hnsw"
    fi
fi


if [ "$MPCV2_TYPE" == "cpu" ]; then
  CLUSTER_NAME="ampc-hnsw"
  if [ "$ENVIRONMENT_NAME" == "dev" ]; then
    CLUSTERS=("$CLUSTER_NAME-0-dev" "$CLUSTER_NAME-1-dev" "$CLUSTER_NAME-2-dev")
  else
    CLUSTERS=("arn:aws:eks:eu-central-1:024848486749:cluster/$CLUSTER_NAME-0-stage" "arn:aws:eks:eu-central-1:024848486818:cluster/$CLUSTER_NAME-1-stage" "arn:aws:eks:eu-central-1:024848486770:cluster/$CLUSTER_NAME-2-stage")
  fi
  echo "Using CPU cluster name: $CLUSTER_NAME"
  echo "Using CPU argo app: $ARGOCD_APP"
  echo "Operating on environment: $ENVIRONMENT_NAME"
elif [ "$MPCV2_TYPE" == "gpu" ]; then
  # GPU clusters are only available in stage environment
  if [ "$ENVIRONMENT_NAME" != "stage" ]; then
     printf "GPU clusters are only available in stage environment\n"
     exit 1
  fi
  CLUSTER_NAME="smpcv2"
  CLUSTERS=("arn:aws:eks:eu-north-1:024848486749:cluster/$CLUSTER_NAME-0-stage" "arn:aws:eks:eu-north-1:024848486818:cluster/$CLUSTER_NAME-1-stage" "arn:aws:eks:eu-north-1:024848486770:cluster/$CLUSTER_NAME-2-stage")
  echo "Using GPU cluster name: $CLUSTER_NAME"
  echo "Using GPU argo app: $ARGOCD_APP"
  echo "Operating on environment: $ENVIRONMENT_NAME"
else
  printf "\nUnknown mpcv2 type: %s\n" "$MPCV2_TYPE"
  printf "Available types: gpu, cpu\n"
  exit 1
fi


# Define ArgoCD application details
ARGOCD_PORTS=(8081 8082 8083)

# Define a green color
GREEN='\033[0;32m'
# Reset color
NC='\033[0m'

# Function to port-forward ArgoCD and retrieve the password
get_argocd_password() {
    local cluster=$1
    local port=$2

    echo "Retrieving ArgoCD password for $cluster..."
    ARGOCD_PASSWORD=$(kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d)
    if [ -z "$ARGOCD_PASSWORD" ]; then
        echo "Failed to retrieve ArgoCD password for $cluster. Skipping..."
        return 1
    fi

    echo "Password retrieved successfully."
    return 0
}

# Function to authenticate to ArgoCD
authenticate_argocd() {
    local port=$1

    echo "Authenticating to ArgoCD on localhost:$port..."
    argocd login localhost:$port --username admin --password $ARGOCD_PASSWORD --insecure
    if [ $? -ne 0 ]; then
        echo "Failed to authenticate to ArgoCD on localhost:$port. Skipping..."
        return 1
    fi
    return 0
}

# Loop through each cluster
for i in "${!CLUSTERS[@]}"; do
    CLUSTER=${CLUSTERS[$i]}
    PORT=${ARGOCD_PORTS[$i]}

    echo "Switching context to $CLUSTER..."
    kubectl ctx $CLUSTER
    if [ $? -ne 0 ]; then
        echo "Failed to switch context to $CLUSTER. Skipping..."
        continue
    fi

    kubectl -n argocd port-forward svc/argocd-server $PORT:443 1> /dev/null &
    PORT_FORWARD_PID=$!
    sleep 2

    # Start port-forwarding and retrieve the ArgoCD password
    get_argocd_password $CLUSTER $PORT
    if [ $? -ne 0 ]; then
        continue
    fi

    # Authenticate to ArgoCD
    authenticate_argocd $PORT
    if [ $? -ne 0 ]; then
        continue
    fi

    echo "Disabling apps-bootstrap auto-sync to allow custom deployments."
    # Disable auto-sync for the apps-bootstrap
    argocd app set apps-bootstrap --sync-policy manual
    if [ $? -ne 0 ]; then
        echo "Failed to disable auto-sync for apps-bootstrap. Exiting the script"
        exit 1
    fi

    echo "Updating targetRevision for application $ARGOCD_APP in $CLUSTER..."
    # Use ArgoCD CLI to update the application
    argocd app set $ARGOCD_APP --revision $NEW_BRANCH --source-position 2
    if [ $? -ne 0 ]; then
        echo "Failed to update targetRevision for $ARGOCD_APP in $CLUSTER. Skipping..."
        continue
    fi

    echo "Syncing application $ARGOCD_APP in $CLUSTER..."
    argocd app sync $ARGOCD_APP 1> /dev/null
    if [ $? -ne 0 ]; then
        echo "Failed to sync application $ARGOCD_APP in $CLUSTER. Skipping..."
        continue
    fi

    kill $PORT_FORWARD_PID
    echo -e "${GREEN}✔${NC} Successfully updated $ARGOCD_APP in $CLUSTER to targetRevision $NEW_BRANCH."
    echo ""
done

echo -e "${GREEN}✔${NC} Done!"
