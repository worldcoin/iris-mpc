#!/bin/bash

# This script provides a way to update the targetRevision of an ArgoCD application in all SMPC stage clusters.
# The script will loop through each cluster, port-forward ArgoCD server, authenticate to it, update the targetRevision of the application, and close port-forward.
#
# Usage: ./argo_stage_update_revision.sh <your-branch | main>
#
# Pre-requisites:
# 1. Ensure that you have the ArgoCD CLI installed on your local machine. (brew install argocd)
# 2. Ensure that you have the kubectl and kubectx installed on your local machine.
# 3. Ensure that your kubeconfig is configured to access the target clusters.


# Define clusters and ArgoCD application details
CLUSTERS=("arn:aws:eks:eu-north-1:024848486749:cluster/smpcv2-0-stage" "arn:aws:eks:eu-north-1:024848486818:cluster/smpcv2-1-stage" "arn:aws:eks:eu-north-1:024848486770:cluster/smpcv2-2-stage")
ARGOCD_PORTS=(8081 8082 8083)
ARGOCD_APP="iris-mpc"
NEW_BRANCH=$1

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
    kubectx $CLUSTER
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
