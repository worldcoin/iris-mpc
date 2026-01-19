#!/usr/bin/env bash

function help() {
    echo "
    DESCRIPTION
    ----------------------------------------------------------------
    Executes HNSW service client.

    ARGS
    ----------------------------------------------------------------
    env         Environment: lcl-dkr | dev-stg. Optional.
    config      Path to a service client config toml file.

    DEFAULTS
    ----------------------------------------------------------------
    env         lcl-dkr
    config      $(_get_path_to_template "simple-uniqueness")
    "
}

function _main()
{
    local path_to_config=${2:-$(_get_path_to_template "simple-uniqueness")}
    local path_to_config_aws=$(_get_path_to_config_aws "${1}")

    pushd "$(_get_path_to_iris_mpc_bins)" || exit
    cargo run \
        --release --bin service-client -- \
        --path-to-config "${path_to_config}" \
        --path-to-config-aws "${path_to_config_aws}"
    popd || exit
}

function _get_path_to_env()
{
    echo "$(_get_path_to_here)/envs/${1}"
}

function _get_path_to_template()
{
    echo "$(_get_path_to_here)/templates/${1}.toml"
}

function _get_path_to_config_aws()
{
    local env=${1}

    echo "$(_get_path_to_env ${1})/config-aws.toml"
}

function _get_path_to_here()
{
    echo $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
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
