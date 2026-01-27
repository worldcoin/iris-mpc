#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    hnsw-service-client-set-env

    DESCRIPTION
    ----------------------------------------------------------------
    Sets HNSW service client environment for immediate usage.

    ARGS
    ----------------------------------------------------------------
    env         Environment: dev-dkr | dev-stg.

    DEFAULTS
    ----------------------------------------------------------------
    env         dev-dkr
    "
}

function _main()
{
    local env=${1}

    if [ "${env}" = "dev-dkr" ] || [ "${env}" = "dev-stg" ]; then
        _set_env "${env}"
    else
        _log_error "Invalid env: ${env}"
    fi
}

function _set_env()
{
    local env=${1}

    # Copy aws config/credential files to ~/.aws folder.
    cp "$(_get_path_to_aws_opts_env_asset ${env} "aws-config")" "${HOME}/.aws/config"
    cp "$(_get_path_to_aws_opts_env_asset ${env} "aws-credentials")" "${HOME}/.aws/credentials"

    # Activate env vars.
    source "$(_get_path_to_aws_opts_env_asset ${env} "aws_evars.sh")"

    # Set memo.
    printf "${env}" > $(_get_path_to_aws_opts)/env
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
    _main "${1:-"dev-dkr"}"
fi
