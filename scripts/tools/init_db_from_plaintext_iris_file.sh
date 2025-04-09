#!/usr/bin/env bash
# set -e

# Returns default db schema for an MPC participant.
function get_default_db_schema() {
    local party_idx=$((${1} - 1))

    echo "SMPC_dev_$party_idx"
}

# Returns default db url for an MPC participant.
function get_default_db_url() {
    echo "postgres://postgres:postgres@localhost:5432"
}

# Returns default path to test iris data in plaintext.
function get_default_path_to_iris_plaintext() {
    local here
    local root

    here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    root=$( dirname "$( dirname "$here" )" )

    echo "$root/iris-mpc-cpu/data/store.ndjson"
}

# Execute binary.
cargo run --bin init-test-dbs -- \
    --db-schema-party1 \
        "${SMPC_DB_SCHEMA_PARTY_1:-$(get_default_db_schema 1)}" \
    --db-schema-party2 \
        "${SMPC_DB_SCHEMA_PARTY_2:-$(get_default_db_schema 2)}" \
    --db-schema-party3 \
        "${SMPC_DB_SCHEMA_PARTY_3:-$(get_default_db_schema 3)}" \
    --db-url-party1 \
        "${SMPC_DB_URL_PARTY_1:-$(get_default_db_url 1)}" \
    --db-url-party2 \
        "${SMPC_DB_URL_PARTY_2:-$(get_default_db_url 2)}" \
    --db-url-party3 \
        "${SMPC_DB_URL_PARTY_3:-$(get_default_db_url 3)}" \
    --source \
        "${SMPC_PATH_TO_IRIS_PLAINTEXT:-$(get_default_path_to_iris_plaintext)}"
