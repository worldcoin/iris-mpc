#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-local-node-view-logs

    DESCRIPTION
    ----------------------------------------------------------------
    Renders MPC node logs to stdout.

    ARGS
    ----------------------------------------------------------------
    node        Ordinal identifier of node.
    filter      Grep filter to apply to logs.

    DEFAULTS
    ----------------------------------------------------------------
    filter      none
    "
}

function _main()
{
    local idx_of_node=${1}
    local log_filter=${2}
    local log_dir_of_node
    local log_fpath

    log_dir_of_node="$(get_path_to_assets_of_node "${idx_of_node}")/logs"
    log_fpath="${log_dir_of_node}/output.log"

    if [ -f "${log_fpath}" ]
    then
        if [ "${log_filter}" = "none" ]; then
            less "${log_fpath}"
        else
            less "${log_fpath}" | grep "${log_filter}"
        fi
    else
        log "No logs found for node ${idx_of_node}"
    fi
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "${MPCTL}"/utils/main.sh

unset _FILTER
unset _HELP
unset _IDX_OF_NODE

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)
    case "$KEY" in
        filter) _FILTER=${VALUE} ;;
        help) _HELP="show" ;;
        node) _IDX_OF_NODE=${VALUE} ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main "${_IDX_OF_NODE:-"0"}" "${_FILTER:-"none"}"
fi
