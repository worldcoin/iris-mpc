#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-dkr-services-up

    DESCRIPTION
    ----------------------------------------------------------------
    Brings up base dockerised services, i.e. localstack & PostgreSQL.

    ARGS
    ----------------------------------------------------------------
    mode        Mode: detached | other. Optional.

    DEFAULTS
    ----------------------------------------------------------------
    mode        detached
    "
}

function _main()
{
    local mode=${1}

    pushd "$(get_path_to_monorepo)" || exit
    if [ "$mode" == "detached" ]; then
        docker-compose -f "${MPCTL_DKR_COMPOSE_SERVICES}" up --detach
    else
        docker-compose -f "${MPCTL_DKR_COMPOSE_SERVICES}" up
    fi
    popd || exit
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "${MPCTL}"/utils/main.sh

unset _HELP
unset _MODE

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)
    case "$KEY" in
        help) _HELP="show" ;;
        mode) _MODE=${VALUE} ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main "${_MODE:-"detached"}"
fi
