#!/usr/bin/env bash

function _help() {
    echo "
    DESCRIPTION
    ----------------------------------------------------------------
    Copies HNSW service client AWS config files to local file system.

    ARGS
    ----------------------------------------------------------------
    env         Environment: lcl-dkr | dev-stg. Optional.

    DEFAULTS
    ----------------------------------------------------------------
    env         lcl-dkr
    "
}

function _main()
{
    local path_to_resources=$(_get_path_to_env_folder ${1})

    cp "${path_to_resources}/aws-config" "${HOME}/.aws/config"
    cp "${path_to_resources}/aws-credentials" "${HOME}/.aws/credentials"
}

function _get_path_to_env_folder()
{
    echo "$(_get_path_to_here)/env-${1}"
}

function _get_path_to_here()
{
    echo $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

unset _ENV
unset _HELP

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)
    case "$KEY" in
        env) _ENV=${VALUE} ;;
        help) _HELP="show" ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main "${_ENV:-"lcl-dkr"}"
fi
