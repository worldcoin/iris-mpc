#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-infra-node-activate-env

    DESCRIPTION
    ----------------------------------------------------------------
    Activates a node's environment.

    ARGS
    ----------------------------------------------------------------
    batchsize   Size of indexation batches.
    node        Ordinal identifier of node.

    DEFAULTS
    ----------------------------------------------------------------
    batchsize   64
    "
}

function _main()
{
    local idx_of_node=${1}
    local size_of_batch=${2}

    # Set automatic variable exportation.
    set -a

    # Standard evars.
    source "$(get_path_to_monorepo)/.test.env"
    source "$(get_path_to_monorepo)/.test.hawk${idx_of_node}.env"

    # Overridden evars (necessary to run locally).
    export AWS_ENDPOINT_URL="http://127.0.0.1:4566"
    export RUST_MIN_STACK=104857600
    export SMPC__AWS__ENDPOINT="http://127.0.0.1:4566"
    export SMPC__CPU_DATABASE__URL="postgres://postgres:postgres@localhost:5432/SMPC_dev_${idx_of_node}"
    export SMPC__DATABASE__URL=${SMPC__CPU_DATABASE__URL}
    export SMPC__HNSW_SCHEMA_NAME_SUFFIX=_hnsw
    export SMPC__MAX_BATCH_SIZE=${size_of_batch}
    export SMPC__NODE_HOSTNAMES='["127.0.0.1","127.0.0.1","127.0.0.1"]'

    # Unset automatic variable exportation.
    set +a
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "${MPCTL}"/utils/main.sh

unset _HELP
unset _IDX_OF_NODE
unset _SIZE_OF_BATCH

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)
    case "$KEY" in
        batchsize) _SIZE_OF_BATCH=${VALUE} ;;
        help) _HELP="show" ;;
        node) _IDX_OF_NODE=${VALUE} ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main \
        "${_IDX_OF_NODE}" \
        "${_SIZE_OF_BATCH:-64}"
fi
