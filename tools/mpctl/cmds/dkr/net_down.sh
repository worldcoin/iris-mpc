#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-dkr-net-down

    DESCRIPTION
    ----------------------------------------------------------------
    Brings down MPC network.

    ARGS
    ----------------------------------------------------------------
    binary      Binary to execute: standard | genesis. Optional.

    DEFAULTS
    ----------------------------------------------------------------
    binary      standard
    "
}

function _main()
{
    local binary=${1}

    for idx_of_node in $(seq 0 "$((MPCTL_COUNT_OF_PARTIES - 1))")
    do
        source "${MPCTL}"/cmds/dkr/node_down.sh \
            binary="${binary}" \
            node="${idx_of_node}"
    done
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "${MPCTL}"/utils/main.sh

unset _BINARY
unset _HELP

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)
    case "$KEY" in
        binary) _BINARY=${VALUE} ;;
        help) _HELP="show" ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main "${_BINARY:-"standard"}"
fi
