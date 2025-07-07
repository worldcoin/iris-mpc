#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-infra-net-pgres-dump

    DESCRIPTION
    ----------------------------------------------------------------
    Backs up a network's postgres databases.
    "
}

function _main()
{
    local idx_of_node

    for idx_of_node in $(seq 0 "$((MPCTL_COUNT_OF_PARTIES - 1))")
    do
        _do_backup "${idx_of_node}"
    done

    log_break
    log "Network postgres databases dumps complete"
    log_break
}

function _do_backup()
{
    local backup_dir
    local db_name
    local idx_of_node=${1}
    local super_user_name
    local super_user_password

    backup_dir="$(get_path_to_assets)/data/db-backups"
    db_name=$(get_pgres_app_db_name "$idx_of_node")
    super_user_name=$(get_pgres_super_user_name)
    super_user_password=$(get_pgres_super_user_password)

    log_break
    log "Node $idx_of_node: postgres dB dump begins"
    log "    dB name=${db_name}"
    log "    dB super user=${super_user_name}"
    log "    dB dump path=${backup_dir}"

    mkdir -p "${backup_dir}"

    docker exec \
        -i "${MPCTL_DKR_CONTAINER_PGRES_DB}" /bin/bash \
        -c "PGPASSWORD=${super_user_password} pg_dump -a --inserts -U ${super_user_name} -d ${db_name}" \
        > "${backup_dir}/${db_name}.sql"

    log "Node $idx_of_node: postgres dB dump complete"
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
