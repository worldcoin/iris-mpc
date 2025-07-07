#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-local-node-start-genesis

    DESCRIPTION
    ----------------------------------------------------------------
    Starts an MPC genesis node on bare metal.

    ARGS
    ----------------------------------------------------------------
    batchsize           Size of indexation batches. Optional.
    batchsize-error     Error rate to apply for dynamic batch size calculation. Optional.
    height              Last Iris serial identifier to be indexed in job.
    mode                Compilation mode: terminal | detached. Optional.
    node                Ordinal identifier of node.

    DEFAULTS
    ----------------------------------------------------------------
    batchsize           64
    batchsize-error     128
    height              100
    mode                terminal
    "
}

function _main()
{
    local batch_size=${1}
    local batch_size_error_rate=${2}
    local idx_of_node=${3}
    local height_max=${4}
    local mode=${5}

    local path_to_binary
    local path_to_log

    # Set env.
    source "${MPCTL}"/cmds/local/node_activate_env.sh \
        node="${idx_of_node}" \
        batchsize="${batch_size}"

    # Set path.
    path_to_binary="$(get_path_to_assets_of_node "${idx_of_node}")/bin/iris-mpc-hawk-genesis"

    # Start process: terminal.
    if [ "$mode" == "terminal" ]; then
        "${path_to_binary}" \
            --batch-size="${batch_size}" \
            --batch-size-r="${batch_size_error_rate}" \
            --max-height="${height_max}" \
            --perform-snapshot="false"

    # Start process: detached.
    else
        path_to_log="$(get_path_to_assets_of_node "${idx_of_node}")/logs/output.log"
        if [ -f "${path_to_log}" ]
        then
            rm "${path_to_log}"
        fi
        nohup "${path_to_binary}" \
            --batch-size="${batch_size}" \
            --batch-size-r="${batch_size_error_rate}" \
            --max-height="${height_max}" \
            --perform-snapshot="false" \
            > "${path_to_log}" 2>&1 &
    fi
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "${MPCTL}"/utils/main.sh

unset _BATCH_SIZE
unset _BATCH_SIZE_ERROR_RATE
unset _HELP
unset _IDX_OF_NODE
unset _MAX_HEIGHT
unset _MODE

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)
    case "$KEY" in
        batchsize) _BATCH_SIZE=${VALUE} ;;
        batchsize-error) _BATCH_SIZE_ERROR_RATE=${VALUE} ;;
        help) _HELP="show" ;;
        height) _MAX_HEIGHT=${VALUE} ;;
        mode) _MODE=${VALUE} ;;
        node) _IDX_OF_NODE=${VALUE} ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main \
        "${_BATCH_SIZE:-0}" \
        "${_BATCH_SIZE_ERROR_RATE:-128}" \
        "${_IDX_OF_NODE:-0}" \
        "${_MAX_HEIGHT:-100}" \
        "${_MODE:-"terminal"}"
fi
