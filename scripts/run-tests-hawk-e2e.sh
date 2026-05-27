#!/bin/bash
set -e

# ----------------------------------------------------------------
# Executes end to end CPU tests.
# ----------------------------------------------------------------
function _main()
{
    _log "Executing end to end tests"

    pushd "$(_get_path_to_monorepo)/iris-mpc-upgrade-hawk" || exit

    # needs to run twice to create the keys with both AWSCURRENT and AWSPREVIOUS states
    cargo run --release -p iris-mpc-bins --bin key-manager --  --region us-east-1 --endpoint-url http://localstack:4566 --node-id 0 --env dev rotate --public-key-bucket-name wf-dev-public-keys
    cargo run --release -p iris-mpc-bins --bin key-manager --  --region us-east-1 --endpoint-url http://localstack:4566 --node-id 1 --env dev rotate --public-key-bucket-name wf-dev-public-keys
    cargo run --release -p iris-mpc-bins --bin key-manager --  --region us-east-1 --endpoint-url http://localstack:4566 --node-id 2 --env dev rotate --public-key-bucket-name wf-dev-public-keys

    cargo run --release -p iris-mpc-bins --bin key-manager --  --region us-east-1 --endpoint-url http://localstack:4566 --node-id 0 --env dev rotate --public-key-bucket-name wf-dev-public-keys
    cargo run --release -p iris-mpc-bins --bin key-manager --  --region us-east-1 --endpoint-url http://localstack:4566 --node-id 1 --env dev rotate --public-key-bucket-name wf-dev-public-keys
    cargo run --release -p iris-mpc-bins --bin key-manager --  --region us-east-1 --endpoint-url http://localstack:4566 --node-id 2 --env dev rotate --public-key-bucket-name wf-dev-public-keys

    cargo test --release --test e2e_hawk -- --include-ignored
    popd || exit
}

function _get_now()
{
    echo $(date +%Y-%m-%dT%H:%M:%S.%6N)
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

function _log ()
{
    local MSG=${1}

    echo -e "$(_get_now) [INFO] [$$] HNSW-E2E :: ${MSG}"
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

_main
