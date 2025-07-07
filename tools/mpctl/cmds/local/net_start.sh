#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-local-net-start

    DESCRIPTION
    ----------------------------------------------------------------
    Starts a local bare metal MPC network.

    ARGS
    ----------------------------------------------------------------
    batchsize   Size of indexation batches. Optional.

    DEFAULTS
    ----------------------------------------------------------------
    batchsize   64
    "
}

function _main()
{
    local size_of_batch=${1}
    local idx_of_node

    for idx_of_node in $(seq 0 "$((MPCTL_COUNT_OF_PARTIES - 1))")
    do
        source "${MPCTL}"/cmds/local/node_start.sh \
            node="${idx_of_node}" \
            mode="detached" \
            batchsize="${size_of_batch}"
    done
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "${MPCTL}"/utils/main.sh

unset _HELP
unset _SIZE_OF_BATCH

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)
    case "$KEY" in
        batchsize) _SIZE_OF_BATCH=${VALUE} ;;
        help) _HELP="show" ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main "${_SIZE_OF_BATCH:-64}"
fi
