#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-infra-net-pgres-truncate

    DESCRIPTION
    ----------------------------------------------------------------
    Truncates a network's postgres database tables.

    ARGS
    ----------------------------------------------------------------
    group       Group of table to truncate: all | cpu | gpu.

    DEFAULTS
    ----------------------------------------------------------------
    group       cpu
    "
}

function _main()
{
    local idx_of_node
    local table_group=${1}

    log_break
    log "Network postgres dB tables truncation begins"
    log "TABLE GROUP=${table_group}"

    for idx_of_node in $(seq 0 "$((MPCTL_COUNT_OF_PARTIES - 1))")
    do
        _do_truncation "${idx_of_node}" "${table_group}"
    done

    log_break
    log "Network postgres dB tables truncation complete"
    log_break
}

function _do_truncation()
{
    local idx_of_node=${1}
    local table_group=${2}

    log_break
    log "Node ${idx_of_node}: postgres dB table truncation begins"

    if [ "${table_group:-""}" = "all" ]; then
        _do_truncation_of_group "${idx_of_node}" "cpu"
        _do_truncation_of_group "${idx_of_node}" "gpu"

    elif [ "${table_group:-""}" = "cpu" ]; then
        _do_truncation_of_group "${idx_of_node}" "cpu"

    elif [ "${table_group:-""}" = "gpu" ]; then
        _do_truncation_of_group "${idx_of_node}" "gpu"

    else
        log_error "Unrecognized table group: ${table_group}"
    fi

    log "Node ${idx_of_node}: postgres dB table truncation complete"
    log_break
}

function _do_truncation_of_group() {
    local idx_of_node=${1}
    local table_group=${2}
    local db_name
    local sql_fpath

    log "... truncating ${table_group} tables"

    db_name=$(get_pgres_app_db_name "$idx_of_node")
    sql_fpath="$MPCTL/resources/sql/pgres-truncate-${table_group}-node-${idx_of_node}.sql"

    exec_pgres_script "${db_name}" "${sql_fpath}"
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "${MPCTL}"/utils/main.sh

unset _HELP
unset _GROUP

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)
    case "$KEY" in
        help) _HELP="show" ;;
        group) _GROUP=${VALUE} ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main "${_GROUP:-"cpu"}"
fi
