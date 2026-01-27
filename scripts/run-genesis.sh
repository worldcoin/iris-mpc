#!/usr/bin/env bash

# this script is meant to be called by Dockerfile.genesis.hawk

set -e

# Validate required environment variables
if [ -z "${GENESIS_MAX_HEIGHT}" ]; then
    echo "Error: GENESIS_MAX_HEIGHT environment variable is not set" >&2
    exit 1
fi

if [ -z "${GENESIS_BATCH_SIZE}" ]; then
    echo "Error: GENESIS_BATCH_SIZE environment variable is not set" >&2
    exit 1
fi

echo "Starting genesis with max height: ${GENESIS_MAX_HEIGHT}, batch size: ${GENESIS_BATCH_SIZE}"
/bin/iris-mpc-hawk-genesis --max-height=${GENESIS_MAX_HEIGHT} --batch-size=${GENESIS_BATCH_SIZE} --perform-snapshot=false
genesis_exit_code=$?

# Check if genesis exited due to a shutdown signal
if [ $genesis_exit_code -eq 130 ]; then
    echo "Genesis was interrupted by SIGINT (Ctrl+C). Exiting"
    exit 0
elif [ $genesis_exit_code -eq 143 ]; then
    echo "Genesis was terminated by SIGTERM. Exiting"
    exit 0
elif [ $genesis_exit_code -ne 0 ]; then
    echo "Genesis failed with exit code: $genesis_exit_code" >&2
    exit $genesis_exit_code
fi

echo "genesis finished. waiting for shutdown"

RUNNING=true
shutdown_handler() {
    echo "Received shutdown signal."
    RUNNING=false
}
trap shutdown_handler SIGTERM SIGINT SIGHUP SIGQUIT

while [ "$RUNNING" = true ]; do
    sleep 1
done

echo "Shutdown complete. Exiting."
exit 0
