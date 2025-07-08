#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-ls | mpctl-dev-view-commands

    DESCRIPTION
    ----------------------------------------------------------------
    Displays set of supported commands.
    "
}

function _main()
{
    echo "

    # Developer commands.
    mpctl-dev-apply-linter
    mpctl-dev-run-tests
    mpctl-ls

    # Local docker based infra commands.
    mpctl-dkr-build-images
    mpctl-dkr-init-db-for-genesis
    mpctl-dkr-net-down
    mpctl-dkr-net-down-genesis
    mpctl-dkr-net-start
    mpctl-dkr-net-start-genesis
    mpctl-dkr-net-stop
    mpctl-dkr-net-stop-genesis
    mpctl-dkr-net-up
    mpctl-dkr-net-up-genesis
    mpctl-dkr-node-start
    mpctl-dkr-node-start-genesis
    mpctl-dkr-node-stop
    mpctl-dkr-node-stop-genesis
    mpctl-dkr-node-view-logs
    mpctl-dkr-node-view-logs-genesis
    mpctl-dkr-services-down
    mpctl-dkr-services-up

    # Jobs.
    mpctl-job-aws-sm-rotate-keys
    mpctl-job-pgres-dump
    mpctl-job-pgres-init-from-plain-text-iris-file
    mpctl-job-pgres-restore
    mpctl-job-pgres-truncate-all
    mpctl-job-pgres-truncate-cpu-tables
    mpctl-job-pgres-truncate-gpu-tables
    mpctl-job-write-iris-deletions-file
    mpctl-job-write-plain-text-iris-file

    # Local baremetal based infra commands.
    mpctl-local-compile-binaries
    mpctl-local-net-reset
    mpctl-local-net-setup
    mpctl-local-net-start
    mpctl-local-net-start-genesis
    mpctl-local-net-teardown
    mpctl-local-net-update-binaries
    mpctl-local-node-activate-env
    mpctl-local-node-start
    mpctl-local-node-start-genesis
    mpctl-local-node-update-binaries
    mpctl-local-node-view-env
    mpctl-local-node-view-logs
    "
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "${MPCTL}"/utils/main.sh

unset _HELP

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    case "$KEY" in
        help) _HELP="show" ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main
fi
