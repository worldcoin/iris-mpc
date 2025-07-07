#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-dkr-net-up

    DESCRIPTION
    ----------------------------------------------------------------
    Brings up MPC network.

    ARGS
    ----------------------------------------------------------------
    binary      Binary to execute: standard | genesis. Optional.
    mode        Mode: detached | other. Optional.

    DEFAULTS
    ----------------------------------------------------------------
    binary      standard
    mode        detached
    "
}

function _main()
{
    local binary=${1}
    local mode=${2}

    for idx_of_node in $(seq 0 "$((MPCTL_COUNT_OF_PARTIES - 1))")
    do
        source "${MPCTL}"/cmds/dkr/node_up.sh \
            binary="${binary}" \
            node="${idx_of_node}" \
            mode="${mode}"
    done
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "${MPCTL}"/utils/main.sh

unset _BINARY
unset _HELP
unset _MODE

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)
    case "$KEY" in
        binary) _BINARY=${VALUE} ;;
        help) _HELP="show" ;;
        mode) _MODE=${VALUE} ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main \
        "${_BINARY:-"standard"}" \
        "${_MODE:-"detached"}"
fi
