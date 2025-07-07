#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-infra-net-teardown

    DESCRIPTION
    ----------------------------------------------------------------
    Tears down assets for an MPC network.
    "
}

function _main()
{
    log_break
    log "Network teardown :: begins"

    _teardown_processes
    log "    processes stopped"

    _teardown_assets
    log "    assets deleted"

    log "Network teardown :: complete"
    log_break
}

function _teardown_assets()
{
    local path_to_assets

    path_to_assets="$(get_path_to_assets_of_net)"
    if [ -d "$path_to_assets" ]; then
        rm -rf "$path_to_assets"
    fi
}

function _teardown_processes()
{
    echo "TODO: _teardown_processes"
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "${MPCTL}"/utils/main.sh

unset _HELP

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    case "$KEY" in
        help) _HELP="show" ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main
fi
