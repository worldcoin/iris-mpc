#!/bin/bash
set -e

# ----------------------------------------------------------------
# Executes end to end CPU tests.
# ----------------------------------------------------------------
function _main()
{
    _log "Executing end to end tests"

    pushd "$(_get_path_to_monorepo)/iris-mpc-upgrade-hawk" || exit
    cargo test --release --test e2e_genesis -- --include-ignored
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
