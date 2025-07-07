#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-dkr-net-start

    DESCRIPTION
    ----------------------------------------------------------------
    Starts an MPC node.

    ARGS
    ----------------------------------------------------------------
    binary      Binary to execute: standard | genesis. Optional.
    node        Ordinal identifier of node.

    DEFAULTS
    ----------------------------------------------------------------
    binary      standard
    "
}

function _main()
{
    local idx_of_node=${1}
    local binary=${2}

    docker-compose -f "$(get_path_to_docker_compose_file_of_node ${binary})" \
        start "$(get_name_of_docker_container_of_node ${idx_of_node})"
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "${MPCTL}"/utils/main.sh

unset _BINARY
unset _HELP
unset _IDX_OF_NODE

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)
    case "$KEY" in
        binary) _BINARY=${VALUE} ;;
        help) _HELP="show" ;;
        node) _IDX_OF_NODE=${VALUE} ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main "$_IDX_OF_NODE" "${_BINARY:-"standard"}"
fi
