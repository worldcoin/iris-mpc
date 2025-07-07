#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-infra-net-pgres-restore

    DESCRIPTION
    ----------------------------------------------------------------
    Restores a network's postgres databases.
    "
}

function _main()
{
    local idx_of_node

    for idx_of_node in $(seq 0 "$((MPCTL_COUNT_OF_PARTIES - 1))")
    do
        _do_restore "${idx_of_node}"
    done

    log_break
    log "Network postgres databases restore complete"
    log_break
}

function _do_restore()
{
    local backup_dir
    local db_name
    local idx_of_node=${1}

    backup_dir="$(get_path_to_assets)/data/db-backups"
    db_name=$(get_pgres_app_db_name "$idx_of_node")

    log_break
    log "Node ${idx_of_node}: postgres dB restore begins"
    log_break

    exec_pgres_script "${db_name}" "${backup_dir}/${db_name}.sql"

    log_break
    log "Node ${idx_of_node}: postgres dB restore complete"
    log_break
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
