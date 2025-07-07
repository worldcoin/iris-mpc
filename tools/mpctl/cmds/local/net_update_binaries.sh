#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-local-net-update-binaries

    DESCRIPTION
    ----------------------------------------------------------------
    Updates binary set of a local bare metal MPC network.

    ARGS
    ----------------------------------------------------------------
    mode        Compilation mode: debug | release. Optional.

    DEFAULTS
    ----------------------------------------------------------------
    mode        release
    "
}

function _main()
{
    local build_mode=${1}
    local idx_of_node

    for idx_of_node in $(seq 0 "$((MPCTL_COUNT_OF_PARTIES - 1))")
    do
        source "${MPCTL}/cmds/local/node_update_binaries.sh" mode="${build_mode}" node="${idx_of_node}"
    done
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "${MPCTL}"/utils/main.sh

unset _BUILD_MODE
unset _HELP

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)
    case "$KEY" in
        help) _HELP="show" ;;
        mode) _BUILD_MODE=${VALUE} ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main "${_BUILD_MODE:-"release"}"
fi
