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
    local idx_of_node=${2}
    local mode=${3}

    local path_to_binary
    local path_to_log

    # Set env.
    source "${MPCTL}"/cmds/local/node_activate_env.sh node="${idx_of_node}" batchsize="${batch_size}"

    # Set path.
    path_to_binary="$(get_path_to_assets_of_node "${idx_of_node}")/bin/iris-mpc-hawk"

    # Start process: terminal.
    if [ "$mode" == "terminal" ]; then
        "${path_to_binary}"

    # Start process: detached.
    else
        path_to_log="$(get_path_to_assets_of_node "${idx_of_node}")/logs/output.log"
        if [ -f "${path_to_log}" ]
        then
            rm "${path_to_log}"
        fi
        nohup "${path_to_binary}" > "${path_to_log}" 2>&1 &
    fi
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "${MPCTL}"/utils/main.sh

unset _BATCH_SIZE
unset _HELP
unset _IDX_OF_NODE
unset _MODE

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)
    case "$KEY" in
        batchsize) _BATCH_SIZE=${VALUE} ;;
        help) _HELP="show" ;;
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
        "${_IDX_OF_NODE:-0}" \
        "${_MODE:-"terminal"}"
fi
