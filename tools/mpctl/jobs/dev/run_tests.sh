#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-dev-run-tests

    DESCRIPTION
    ----------------------------------------------------------------
    Runs unit tests.

    ARGS
    ----------------------------------------------------------------
    package     Package over which to run tests.
    tests       Filter to determine which tests to run.

    DEFAULTS
    ----------------------------------------------------------------
    package     iris-mpc-cpu
    "
}

function _main()
{
    local package=${1}
    local tests_filter=${2}

    pushd "$(get_path_to_monorepo)" || exit
    if [ "${tests_filter}" = "none" ]; then
        cargo test --release --package="${package}" --lib
    else
        cargo test --release --package="${package}" --lib "${tests_filter}"
    fi
    popd || exit
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "${MPCTL}"/utils/main.sh

unset _HELP
unset _PACKAGE
unset _TEST_FILTER

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)
    case "$KEY" in
        help) _HELP="show" ;;
        package) _PACKAGE=${VALUE} ;;
        tests) _TEST_FILTER=${VALUE} ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main \
        "${_PACKAGE:-"iris-mpc-cpu"}" \
        "${_TEST_FILTER:-"none"}"
fi
