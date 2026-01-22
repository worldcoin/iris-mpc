#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    hnsw-service-client-exec

    DESCRIPTION
    ----------------------------------------------------------------
    Executes HNSW service client.

    ARGS
    ----------------------------------------------------------------
    env         Environment: lcl-dkr | dev-stg.
    config      Path to a service client config toml file.

    DEFAULTS
    ----------------------------------------------------------------
    env         lcl-dkr
    config      $(_get_path_to_template "exec-config-example.toml")
    "
}

function _main()
{
    local env=${1}
    local cfg=${2}

    pushd "$(_get_path_to_iris_mpc_bins)" || exit
    cargo run \
        --release --bin service-client -- \
        --path-to-config "${cfg:-$(_get_path_to_template "exec-config-example.toml")}" \
        --path-to-config-aws "$(_get_path_to_aws_config "${env}")"
    popd || exit
}

function _get_path_to_aws_config()
{
    local env=${1}

    echo "$(_get_path_to_env ${1})/config-aws.toml"
}

function _get_path_to_env()
{
    echo "$(_get_path_to_here)/envs/${1}"
}

function _get_path_to_here()
{
    echo $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
}

function _get_path_to_template()
{
    echo "$(_get_path_to_here)/templates/${1}"
}

function _get_path_to_iris_mpc_bins()
{
    echo "$(_get_path_to_ancestor "$(_get_path_to_here)" "1")"
}

function _get_path_to_ancestor()
{
    local path=${1}
    local steps=${2}

    for idx in $(seq 0 ${steps})
    do
        path=$( cd "$( dirname "${path}" )" && pwd )
    done

    echo ${path}
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

unset _ENV
unset _HELP
unset _PATH_TO_CONFIG

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)
    case "$KEY" in
        config) _PATH_TO_CONFIG=${VALUE} ;;
        env) _ENV=${VALUE} ;;
        help) _HELP="show" ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main \
        "${_ENV:-"lcl-dkr"}" \
        "${_PATH_TO_CONFIG}"
fi
