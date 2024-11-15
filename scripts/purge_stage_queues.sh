#!/usr/bin/env bash

source accounts_checker.sh

purge_queues() {
    local PROFILE_NAME=$1
    local REGION=$2
    shift
    shift
    local QUEUE_NAMES=("$@")

    for QUEUE_NAME in "${QUEUE_NAMES[@]}"; do
        # Get the Queue URL from the queue name, using the profile if specified
        if [ -n "$PROFILE_NAME" ]; then
            QUEUE_URL=$(aws sqs get-queue-url --region "$REGION" --queue-name "$QUEUE_NAME" --output text --query 'QueueUrl' --profile "$PROFILE_NAME")
        else
            QUEUE_URL=$(aws sqs get-queue-url --region "$REGION" --queue-name "$QUEUE_NAME" --output text --query 'QueueUrl')
        fi

        if [ $? -ne 0 ]; then
            echo "Failed to get URL for queue: $QUEUE_NAME"
            continue
        fi

        # Purge the queue
        echo "Purging queue: $QUEUE_NAME (URL: $QUEUE_URL)"
        if [ -n "$PROFILE_NAME" ]; then
            aws sqs purge-queue --region "$REGION" --queue-url "$QUEUE_URL" --profile "$PROFILE_NAME"
        else
            aws sqs purge-queue --region "$REGION" --queue-url "$QUEUE_URL"
        fi

        if [ $? -ne 0 ]; then
            echo "Failed to purge queue: $QUEUE_NAME"
        else
            echo "Successfully purged queue: $QUEUE_NAME"
        fi

        sleep 2
    done
}

ORB_QUEUE_NAMES=(
"iris-mpc-identity-deletion-results-dlq-eu-central-1.fifo"
"iris-mpc-identity-deletion-results-eu-central-1.fifo"
"iris-mpc-results-dlq-eu-central-1.fifo"
"iris-mpc-results-eu-central-1.fifo"
)

SMPC_0_QUEUE_NAMES=(
"smpcv2-0-stage.fifo"
"smpcv2-0-stage-dlq.fifo"
)

SMPC_1_QUEUE_NAMES=(
"smpcv2-1-stage.fifo"
"smpcv2-1-stage-dlq.fifo"
)

SMPC_2_QUEUE_NAMES=(
"smpcv2-2-stage.fifo"
"smpcv2-2-stage-dlq.fifo"
)

purge_queues "worldcoin-stage" "eu-central-1" "${ORB_QUEUE_NAMES[@]}"
purge_queues "worldcoin-smpcv-io-0" "eu-north-1" "${SMPC_0_QUEUE_NAMES[@]}"
purge_queues "worldcoin-smpcv-io-1" "eu-north-1" "${SMPC_1_QUEUE_NAMES[@]}"
purge_queues "worldcoin-smpcv-io-2" "eu-north-1" "${SMPC_2_QUEUE_NAMES[@]}"
