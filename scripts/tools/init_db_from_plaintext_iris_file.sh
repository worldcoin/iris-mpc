#!/usr/bin/env bash
set -e

# Default hnsw parameter :: EF.
declare DEFAULT_HNSW_PARAM_EF=320

# Default hnsw parameter :: M.
declare DEFAULT_HNSW_PARAM_M=256

# Default hnsw parameter :: P.
declare DEFAULT_HNSW_PARAM_P=256

# Default number of iris pairs to read from file.
declare DEFAULT_TARGET_DB_SIZE=5000

# Returns default db schema for an MPC participant.
function get_default_db_schema() {
    local party_idx=$((${1} - 1))

    echo "SMPC_dev_$party_idx"
}

# Returns default db url for an MPC participant.
function get_default_db_url() {
    # Assumes standard postgres setup for all parties.
    local party_idx=$((${1} - 1))
    local party_db="SMPC_dev_$party_idx"

    echo "postgres://postgres:postgres@dev_db:5432/$party_db"
}

# Returns default db schema for an MPC participant.
function get_default_hnsw_param_m() {
    local hnsw_param_m=$((${1} - 1))

    echo "SMPC_dev_$party_idx"
}

# Returns default path to test iris data in plaintext.
function get_default_path_to_iris_plaintext() {
    local here
    local root

    here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    root=$( dirname "$( dirname "$here" )" )

    echo "$root/iris-mpc-bins/data/store.ndjson"
}

# Returns default path to prng state file utilised between runs.
function get_default_path_to_prng_state() {
    echo ".prng_state"
}

# Handle the skip-hnsw-graph flag properly
SKIP_HNSW_GRAPH_FLAG=""
if [[ "${SMPC_INIT_SKIP_HNSW_GRAPH}" == "true" ]]; then
    SKIP_HNSW_GRAPH_FLAG="--skip-hnsw-graph"
fi


cargo run --release -p iris-mpc-bins --bin init-test-dbs -- \
    --db-schema-party1 \
        "${SMPC_INIT_DB_SCHEMA_PARTY_1:-$(get_default_db_schema 1)}" \
    --db-schema-party2 \
        "${SMPC_INIT_DB_SCHEMA_PARTY_2:-$(get_default_db_schema 2)}" \
    --db-schema-party3 \
        "${SMPC_INIT_DB_SCHEMA_PARTY_3:-$(get_default_db_schema 3)}" \
    --db-url-party1 \
        "${SMPC_INIT_DB_URL_PARTY_1:-$(get_default_db_url 1)}" \
    --db-url-party2 \
        "${SMPC_INIT_DB_URL_PARTY_2:-$(get_default_db_url 2)}" \
    --db-url-party3 \
        "${SMPC_INIT_DB_URL_PARTY_3:-$(get_default_db_url 3)}" \
    --hnsw-ef \
        "${SMPC_INIT_HNSW_PARAM_EF:-"$DEFAULT_HNSW_PARAM_EF"}" \
    --hnsw-m \
        "${SMPC_INIT_DB_HNSW_PARAM_M:-"$DEFAULT_HNSW_PARAM_M"}" \
    --target-db-size \
        "${SMPC_INIT_TARGET_DB_SIZE:-"$DEFAULT_TARGET_DB_SIZE"}" \
    --prng-state-file \
        "${SMPC_INIT_PATH_TO_PRNG_STATE:-$(get_default_path_to_prng_state)}" \
    --source \
        "${SMPC_INIT_PATH_TO_IRIS_PLAINTEXT:-$(get_default_path_to_iris_plaintext)}" \
    ${SKIP_HNSW_GRAPH_FLAG}
