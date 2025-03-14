#!/usr/bin/env bash

source ./accounts_checker.sh

if [ -z "$1" ]; then
  printf "\nError: Cleanup type parameter is required\n"
  printf "Usage: %s <cleanup_type>\n" "$0"
  printf "Available cleanup types: gpu, cpu\n"
  exit 1
fi

CLEANUP_TYPE=$1

purge_queues() {
    local PROFILE_NAME=$1
    local REGION=$2
    shift
    shift
    local QUEUE_NAMES=("$@")

    if [ -z "$PROFILE_NAME" ]; then
        echo "Profile name is required"
        exit 1
    fi

    for QUEUE_NAME in "${QUEUE_NAMES[@]}"; do
        # Get the Queue URL from the queue name, using the profile if specified
        QUEUE_URL=$(aws sqs get-queue-url --region "$REGION" --queue-name "$QUEUE_NAME" --output text --query 'QueueUrl' --profile "$PROFILE_NAME")


        if [ $? -ne 0 ]; then
            echo "Failed to get URL for queue: $QUEUE_NAME"
            continue
        fi

        # Purge the queue
        echo "Purging queue: $QUEUE_NAME (URL: $QUEUE_URL)"
        aws sqs purge-queue --region "$REGION" --queue-url "$QUEUE_URL" --profile "$PROFILE_NAME"

        if [ $? -ne 0 ]; then
            echo "Failed to purge queue: $QUEUE_NAME"
        else
            echo "Successfully purged queue: $QUEUE_NAME"
        fi

        sleep 2
    done
}

CPU_ORB_QUEUE_NAMES=(
    "hnsw-smpc-identity-deletion-results-dlq-eu-central-1.fifo"
    "hnsw-smpc-identity-deletion-results-eu-central-1.fifo"
    "hnsw-smpc-reauth-results-dlq-eu-central-1.fifo"
    "hnsw-smpc-reauth-results-eu-central-1.fifo"
    "hnsw-smpc-results-dlq-eu-central-1.fifo"
    "hnsw-smpc-results-eu-central-1.fifo"
)

CPU_SMPC_VPC_QUEUE_NAMES=(
    "hnsw-mpc-request-0-stage-dlq.fifo"
    "hnsw-mpc-request-0-stage.fifo"
    "hnsw-mpc-request-1-stage-dlq.fifo"
    "hnsw-mpc-request-1-stage.fifo"
    "hnsw-mpc-request-2-stage-dlq.fifo"
    "hnsw-mpc-request-2-stage.fifo"
)

# Define queue names for CPU
GPU_ORB_QUEUE_NAMES=(
    "iris-mpc-identity-deletion-results-dlq-eu-central-1.fifo"
    "iris-mpc-identity-deletion-results-eu-central-1.fifo"
    "iris-mpc-results-dlq-eu-central-1.fifo"
    "iris-mpc-results-eu-central-1.fifo"
)

GPU_SMPC_VPC_QUEUE_NAMES=(
    "smpcv2-0-stage.fifo"
    "smpcv2-0-stage-dlq.fifo"
    "smpcv2-1-stage.fifo"
    "smpcv2-1-stage-dlq.fifo"
    "smpcv2-2-stage.fifo"
    "smpcv2-2-stage-dlq.fifo"
)

# Purge queues based on the cleanup type
case "$CLEANUP_TYPE" in
  "gpu")
    echo "Purging GPU queues..."
    purge_queues "worldcoin-stage" "eu-central-1" "${GPU_ORB_QUEUE_NAMES[@]}"
    purge_queues "worldcoin-smpcv-io-vpc" "eu-north-1" "${GPU_SMPC_VPC_QUEUE_NAMES[@]}"
    ;;
  "cpu")
    echo "Purging CPU queues..."
    # purge_queues "worldcoin-stage" "eu-central-1" "${CPU_ORB_QUEUE_NAMES[@]}"
    purge_queues "worldcoin-smpcv-io-vpc" "eu-north-1" "${CPU_SMPC_VPC_QUEUE_NAMES[@]}"
    ;;
  *)
    printf "\nUnknown cleanup type: %s\n" "$CLEANUP_TYPE"
    printf "Available cleanup types: gpu, cpu\n"
    exit 1
    ;;
esac

echo "Queue purging completed."
