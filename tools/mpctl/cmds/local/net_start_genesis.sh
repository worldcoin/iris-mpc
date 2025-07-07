#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-local-net-start-genesis

    DESCRIPTION
    ----------------------------------------------------------------
    Starts a local bare metal MPC network at genesis.

    ARGS
    ----------------------------------------------------------------
    batchsize   Size of indexation batches. Optional.
    batchsize-error     Error rate to apply for dynamic batch size calculation. Optional.
    height              Last Iris serial identifier to be indexed in job.

    DEFAULTS
    ----------------------------------------------------------------
    batchsize   64
    batchsize-error     128
    height              100
    "
}

function _main()
{
    local batch_size=${1}
    local batch_size_error_rate=${2}
    local height_max=${3}
    local idx_of_node

    for idx_of_node in $(seq 0 "$((MPCTL_COUNT_OF_PARTIES - 1))")
    do
        source "${MPCTL}"/cmds/local/node_start_genesis.sh \
            batchsize="${batch_size}" \
            batchsize-error="${batch_size_error_rate}" \
            height="${height_max}" \
            node="${idx_of_node}" \
            mode="detached"
    done
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "${MPCTL}"/utils/main.sh

unset _BATCH_SIZE
unset _BATCH_SIZE_ERROR_RATE
unset _HELP
unset _MAX_HEIGHT

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)
    case "$KEY" in
        batchsize) _BATCH_SIZE=${VALUE} ;;
        batchsize-error) _BATCH_SIZE_ERROR_RATE=${VALUE} ;;
        height) _MAX_HEIGHT=${VALUE} ;;
        help) _HELP="show" ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main \
        "${_BATCH_SIZE:-0}" \
        "${_BATCH_SIZE_ERROR_RATE:-128}" \
        "${_MAX_HEIGHT:-100}"
fi
