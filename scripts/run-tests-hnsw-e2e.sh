#!/bin/bash
set -e

# ----------------------------------------------------------------
# Executes end to end CPU tests.
# ----------------------------------------------------------------
function _main()
{
    local path_to_monorepo

    path_to_monorepo="$(_get_path_to_monorepo)"

    echo "Initialising system state"
    _init_system_state "${path_to_monorepo}"

    echo "Executing end to end tests"
    _exec_tests "${path_to_monorepo}"
}

function _exec_tests()
{
    local path_to_monorepo="${1}"

    pushd "${path_to_monorepo}/iris-mpc-upgrade-hawk" || exit
    cargo test --test e2e_genesis
    popd || exit
}

function _get_path_to_monorepo()
{
    local here
    local root

    # Path -> here.
    here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

    # Path -> monorepo root.
    root=$( dirname "$here" )

    echo "${root}"
}

function _init_system_state()
{
    # NOTE: This function is temporary.  In next iteration of e2e testing framework
    # the initialisation of system state will also be performed by the e2e tests.

    local path_to_monorepo="${1}"

    echo "Initialising postgres dBs"
    _init_system_state_pgres "${path_to_monorepo}"

    echo "Initialising AWS services"
    _init_system_state_aws "${path_to_monorepo}"
}

function _init_system_state_pgres()
{
    local path_to_monorepo="${1}"
    local path_to_iris_shares

    path_to_iris_shares="${path_to_monorepo}/iris-mpc-upgrade-hawk/tests/resources/iris-shares-plaintext/20250710-synthetic-irises-1k.ndjson"

    set -a
    SMPC_INIT_SKIP_HNSW_GRAPH=true
    SMPC_INIT_TARGET_DB_SIZE=100
    SMPC_INIT_PATH_TO_IRIS_PLAINTEXT="${path_to_iris_shares}"
    export SMPC_INIT_SKIP_HNSW_GRAPH
    export SMPC_INIT_TARGET_DB_SIZE
    export SMPC_INIT_PATH_TO_IRIS_PLAINTEXT
    set +a

    source "${path_to_monorepo}/scripts/tools/init_db_from_plaintext_iris_file.sh"
}

function _init_system_state_aws()
{
    local path_to_monorepo="${1}"

    echo "TODO: initialise local stack S3 bucket for Iris deletions"
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

_main
