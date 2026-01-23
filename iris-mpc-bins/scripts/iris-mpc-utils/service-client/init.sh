#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    hnsw-service-client-init

    DESCRIPTION
    ----------------------------------------------------------------
    Initialises HNSW service client for subsequent usage.
    "
}

function _main()
{
    _init_fsys
    _init_exec_configs
    _init_envs

    echo "------------------------------------------------------------------------------------------"
    echo "HNSW service client has been initialised.  Please edit the following aws-credentials files as necessary:"
    echo "$(_get_path_to_aws_opts_env "dev-dkr")/aws-credentials"
    echo "$(_get_path_to_aws_opts_env "dev-stg")/aws-credentials"
    echo "------------------------------------------------------------------------------------------"
}

function _init_exec_configs()
{
    cp "$(_get_path_to_resources)/exec-options-1.toml" "$(_get_path_to_exec_opts)/example-1.toml"
    cp "$(_get_path_to_resources)/exec-options-2.toml" "$(_get_path_to_exec_opts)/example-2.toml"
}

function _init_envs()
{
    local envs=("dev-dkr" "dev-stg")
    for env in "${envs[@]}"; do
        mkdir "$(_get_path_to_aws_opts_env ${env})"
        local resources=("aws-config" "aws-config.toml" "aws-credentials" "aws_evars.sh")
        for resource in "${resources[@]}"; do
            cp "$(_get_path_to_resource_of_env ${env} "${resource}")" \
               "$(_get_path_to_aws_opts_env ${env})/${resource}"
        done
    done
}

function _init_fsys()
{
    local paths=($(_get_path_to_exec_opts) $(_get_path_to_aws_opts))
    for path in "${paths[@]}"; do
        if [ -d ${path} ]; then
            rm -rf "${path}"
        fi
        mkdir -p "${path}"
    done
}

function _get_path_to_here()
{
    echo $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "$(_get_path_to_here)/utils.sh"

unset _HELP

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)
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
