#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-local-node-update-binaries

    DESCRIPTION
    ----------------------------------------------------------------
    Updates binary set of a local bare metal MPC node.

    ARGS
    ----------------------------------------------------------------
    mode        Compilation mode: debug | release. Optional.
    node        Ordinal identifier of node.

    DEFAULTS
    ----------------------------------------------------------------
    mode        release
    "
}

function _main()
{
    local build_mode=${2}
    local idx_of_node=${1}

    cp \
        "$(get_path_to_target_binary iris-mpc-hawk ${build_mode})" \
        "$(get_path_to_assets_of_node ${idx_of_node})/bin"
    cp \
        "$(get_path_to_target_binary iris-mpc-hawk-genesis ${build_mode})" \
        "$(get_path_to_assets_of_node ${idx_of_node})/bin"

    log "MPC network :: updated binaries of node ${idx_of_node}"
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "${MPCTL}"/utils/main.sh

unset _BUILD_MODE
unset _HELP
unset _IDX_OF_NODE

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)
    case "$KEY" in
        help) _HELP="show" ;;
        mode) _BUILD_MODE=${VALUE} ;;
        node) _IDX_OF_NODE=${VALUE} ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main "${_IDX_OF_NODE}" "${_BUILD_MODE:-"release"}"
fi
