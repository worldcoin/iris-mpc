#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-jobs-init-db-from-plain-text-iris-file

    DESCRIPTION
    ----------------------------------------------------------------
    Initializes database from previously generated plain text iris files.
    "
}

function _main()
{
    local skip_graph_inserts=${1}

    # Set values to be passed to binary.
    SMPC_INIT_DB_URL_PARTY_1="postgres://postgres:postgres@localhost:5432/SMPC_dev_0"
    SMPC_INIT_DB_URL_PARTY_2="postgres://postgres:postgres@localhost:5432/SMPC_dev_1"
    SMPC_INIT_DB_URL_PARTY_3="postgres://postgres:postgres@localhost:5432/SMPC_dev_2"
    SMPC_INIT_PATH_TO_PRNG_STATE="$(get_path_to_assets)/data/tmp/prng_state"
    SMPC_INIT_PATH_TO_IRIS_PLAINTEXT="$(get_path_to_assets)/data/iris-plaintext/store.ndjson"
    SMPC_INIT_SKIP_HNSW_GRAPH="${skip_graph_inserts}"
    SMPC_INIT_TARGET_DB_SIZE=100

    # Export to external environment so that binary can pick them up.
    export SMPC_INIT_DB_URL_PARTY_1
    export SMPC_INIT_DB_URL_PARTY_2
    export SMPC_INIT_DB_URL_PARTY_3
    export SMPC_INIT_PATH_TO_PRNG_STATE
    export SMPC_INIT_PATH_TO_IRIS_PLAINTEXT
    export SMPC_INIT_SKIP_HNSW_GRAPH
    export SMPC_INIT_TARGET_DB_SIZE

    pushd "$(get_path_to_monorepo)" || exit
    source "$(get_path_to_monorepo)/scripts/tools/init_db_from_plaintext_iris_file.sh"
    popd || exit
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "${MPCTL}"/utils/main.sh

unset _HELP
unset _SKIP_GRAPH_INSERTS

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    case "$KEY" in
        help) _HELP="show" ;;
        skip-graph-inserts) _SKIP_GRAPH_INSERTS=${VALUE} ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main "${_SKIP_GRAPH_INSERTS:-"true"}"
fi
