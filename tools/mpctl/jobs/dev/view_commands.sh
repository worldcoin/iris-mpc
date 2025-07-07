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

    # Docker based infra commands.
    mpctl-dkr-build-image
    mpctl-dkr-net-down
    mpctl-dkr-net-start
    mpctl-dkr-net-stop
    mpctl-dkr-net-up
    mpctl-dkr-net-view-status
    mpctl-dkr-node-start
    mpctl-dkr-node-stop
    mpctl-dkr-services-down
    mpctl-dkr-services-up

    # Baremetal based infra commands.
    mpctl-local-compile-binaries
    mpctl-local-net-setup
    mpctl-local-net-start
    mpctl-local-net-start-genesis
    mpctl-local-net-teardown
    mpctl-local-node-activate-env
    mpctl-local-node-start
    mpctl-local-node-start-genesis
    mpctl-local-node-update-binaries
    mpctl-local-node-view-logs

    # Jobs.
    mpctl-job-aws-sm-rotate-keys
    mpctl-job-pgres-dump
    mpctl-job-pgres-init-from-plain-text-iris-file
    mpctl-job-pgres-restore
    mpctl-job-pgres-truncate-graph-tables
    mpctl-job-pgres-truncate-iris-tables
    mpctl-job-pgres-truncate-state-tables
    mpctl-job-write-iris-serial-ids-for-deletion
    mpctl-job-write-plain-text-iris-file
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
