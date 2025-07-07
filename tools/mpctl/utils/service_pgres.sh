#!/usr/bin/env bash

# Default credentials of application user.
export MPCTL_DEFAULT_PGRES_APP_USER_NAME="postgres"
export MPCTL_DEFAULT_PGRES_APP_USER_PASSWORD="postgres"

# Default credentials of super user.
export MPCTL_DEFAULT_PGRES_SUPER_USER_NAME="postgres"
export MPCTL_DEFAULT_PGRES_SUPER_USER_PASSWORD="postgres"

# Default server host/port.
export MPCTL_DEFAULT_PGRES_SERVER_HOST="localhost"
export MPCTL_DEFAULT_PGRES_SERVER_PORT=5432

##############################################################################
# Executes a sql script on postgres.
##############################################################################
function exec_pgres_script()
{
    local db_name=${1}
    local path_to_sql=${2}

    local server_host
    local server_port
    local super_user_name
    local super_user_password

    server_host=$(get_pgres_server_host)
    server_port=$(get_pgres_server_port)
    super_user_name=$(get_pgres_super_user_name)
    super_user_password=$(get_pgres_super_user_password)

    export PGPASSWORD=${super_user_password}

    psql \
        -d "$db_name" \
        -h "$server_host" \
        -p "$server_port" \
        -f "$path_to_sql" \
        -U "$super_user_name"

    unset PGPASSWORD
}

##############################################################################
# Returns postgres app database name.
##############################################################################
function get_pgres_app_db_name()
{
    local idx_of_node=${1}

    echo "SMPC_$(get_environment_name)_${idx_of_node}"
}

##############################################################################
# Returns postgres app user name.
##############################################################################
function get_pgres_app_user_name()
{
    if is_env_var_set "SMPC__PGRES_APP_USER_NAME"; then
        echo "${SMPC__PGRES_APP_USER_NAME}"
    else
        echo "${MPCTL_DEFAULT_PGRES_APP_USER_NAME}"
    fi
}

##############################################################################
# Returns postgres app user password.
##############################################################################
function get_pgres_app_user_password()
{
    if is_env_var_set "SMPC__PGRES_APP_USER_PASSWORD"; then
        echo "${SMPC__PGRES_APP_USER_PASSWORD}"
    else
        echo "${MPCTL_DEFAULT_PGRES_APP_USER_PASSWORD}"
    fi
}

##############################################################################
# Returns postgres database server connection string for the application user.
##############################################################################
function get_pgres_server_connection_url_for_app_user()
{
    local idx_of_node
    local db_name
    local server_host
    local server_port
    local user_name
    local user_password

    if is_env_var_set SMPC__DATABASE__URL; then
        echo "$SMPC__DATABASE__URL"
    else
        idx_of_node=${1}
        user_name=$(get_pgres_app_user_name)
        user_password=$(get_pgres_app_user_password)
        server_host=$(get_pgres_server_host)
        server_port=$(get_pgres_server_port)
        db_name=$(get_pgres_app_db_name "$idx_of_node")

        echo "postgres://${user_name}:${user_password}@${server_host}:${server_port}/${db_name}"
    fi
}

##############################################################################
# Returns postgres database server host.
##############################################################################
function get_pgres_server_host()
{
    if is_env_var_set "SMPC__PGRES_SERVER_HOST"; then
        echo "${SMPC__PGRES_SERVER_HOST}"
    else
        echo "${MPCTL_DEFAULT_PGRES_SERVER_HOST}"
    fi
}

##############################################################################
# Returns postgres database server port.
##############################################################################
function get_pgres_server_port()
{
    if is_env_var_set "SMPC__PGRES_SERVER_PORT"; then
        echo "${SMPC__PGRES_SERVER_PORT}"
    else
        echo "${MPCTL_DEFAULT_PGRES_SERVER_PORT}"
    fi
}

##############################################################################
# Returns postgres database super user name.
##############################################################################
function get_pgres_super_user_name()
{
    if is_env_var_set "SMPC__PGRES_SUPER_USER_NAME"; then
        echo "${SMPC__PGRES_SUPER_USER_NAME}"
    else
        echo "${MPCTL_DEFAULT_PGRES_SUPER_USER_NAME}"
    fi
}

##############################################################################
# Returns postgres database super user password.
##############################################################################
function get_pgres_super_user_password()
{
    if is_env_var_set "SMPC__PGRES_SUPER_USER_PASSWORD"; then
        echo "${SMPC__PGRES_SUPER_USER_PASSWORD}"
    else
        echo "${MPCTL_DEFAULT_PGRES_SUPER_USER_PASSWORD}"
    fi
}
