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
    opts        Path to a service client execution options toml file.

    DEFAULTS
    ----------------------------------------------------------------
    opts        $(_get_path_to_exec_opts)/examples/example-simple-1.toml
    "
}

function _main()
{
    local aws_opts_env=$(_get_aws_opts_env)
    local path_to_aws_opts=$(_get_path_to_aws_opts_env_asset ${aws_opts_env} "aws-config.toml")
    local path_to_exec_opts=${1:-$(_get_path_to_exec_opts)/examples/example-simple-1.toml}

    pushd "$(_get_path_to_iris_mpc_bins)" || exit
    cargo run \
        --release --bin service-client -- \
        --path-to-opts "${path_to_exec_opts}" \
        --path-to-opts-aws "${path_to_aws_opts}"
    popd || exit
}

function _get_path_to_iris_mpc_bins()
{
    echo "$(_get_path_to_ancestor "$(_get_path_to_here)" "1")"
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
    case "$KEY" in
        help) _HELP="show" ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main "${1}"
fi
