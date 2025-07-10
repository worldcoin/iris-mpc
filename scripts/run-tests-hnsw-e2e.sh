#!/usr/bin/env bash
set -e

# ----------------------------------------------------------------
# Executes end to end CPU tests.
# ----------------------------------------------------------------
function _main()
{
    echo "Initialising system state"
    _init_system_state

    echo "Executing end to end tests"
    _exec_tests
}

function _exec_tests()
{
    pushd "$(_get_path_to_monorepo_root)/iris-mpc-upgrade-hawk" || exit
    cargo test --test e2e_genesis
    popd || exit
}

function _get_path_to_iris_shares()
{
    echo "${_get_path_to_monorepo_root}/iris-mpc-upgrade-hawk/tests/resources/iris-shares-plaintext/store-1000.ndjson"
}

function _get_path_to_monorepo_root()
{
    local here

    # Path -> here.
    here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

    # Path -> monorepo root.
    echo $( dirname "${here}" )
}

function _init_system_state()
{
    # NOTE: This function is temporary.  In next iteration of e2e testing framework
    # the initialisation of system state will also be performed by the e2e tests.

    echo "Initialising postgres dBs"
    _init_system_state_pgres

    echo "Initialising AWS services"
    _init_system_state_aws
}

function _init_system_state_pgres()
{
    export SMPC_INIT_SKIP_HNSW_GRAPH=true
    export SMPC_INIT_TARGET_DB_SIZE=100
    export SMPC_INIT_PATH_TO_IRIS_PLAINTEXT="$(_get_path_to_iris_shares)"
    source $(_get_path_to_monorepo_root)/scripts/tools/init_db_from_plaintext_iris_file.sh
}

function _init_system_state_aws()
{
    echo "TODO: initialise local stack S3 bucket for Iris deletions"
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

_main
