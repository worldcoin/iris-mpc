#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-local-node-view-env

    DESCRIPTION
    ----------------------------------------------------------------
    Renders an MPC node's environment variables.

    ARGS
    ----------------------------------------------------------------
    node                Ordinal identifier of node.
    "
}

function _main()
{
    local idx_of_node=${1}

    source "${MPCTL}"/cmds/local/node_activate_env.sh \
        node="${idx_of_node}" \
        batchsize="${batch_size}"
    printenv | grep "SMPC" | sort
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "${MPCTL}"/utils/main.sh

unset _HELP
unset _IDX_OF_NODE

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)
    case "$KEY" in
        help) _HELP="show" ;;
        node) _IDX_OF_NODE=${VALUE} ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main "${_IDX_OF_NODE:-0}"
fi
