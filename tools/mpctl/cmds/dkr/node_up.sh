#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-dkr-node-up

    DESCRIPTION
    ----------------------------------------------------------------
    Spins up a dockerised MPC node.

    ARGS
    ----------------------------------------------------------------
    binary      Binary to execute: standard | genesis. Optional.
    mode        Mode: detached | other. Optional.
    node        Ordinal identifier of node.

    DEFAULTS
    ----------------------------------------------------------------
    binary      standard
    mode        detached
    "
}

function _main()
{
    local idx_of_node=${1}
    local binary=${2}
    local mode=${3}

    pushd "$(get_path_to_monorepo)" || exit
    if [ "$mode" == "detached" ]; then
        docker-compose \
            -f "$(get_path_to_docker_compose_file_of_node ${binary})" \
            up --detach "$(get_name_of_docker_container_of_node ${idx_of_node})"
    else
        docker-compose \
            -f "$(get_path_to_docker_compose_file_of_node ${binary})" \
            up "$(get_name_of_docker_container_of_node ${idx_of_node})"
    fi
    popd || exit
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "${MPCTL}"/utils/main.sh

unset _BINARY
unset _HELP
unset _IDX_OF_NODE
unset _MODE

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)
    case "$KEY" in
        binary) _BINARY=${VALUE} ;;
        help) _HELP="show" ;;
        mode) _MODE=${VALUE} ;;
        node) _IDX_OF_NODE=${VALUE} ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main \
        "$_IDX_OF_NODE" \
        "${_BINARY:-"standard"}" \
        "${_MODE:-"detached"}"
fi
