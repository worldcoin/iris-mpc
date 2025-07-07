#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-code-apply-linter

    DESCRIPTION
    ----------------------------------------------------------------
    Applies linting prior to a commit.
    "
}

function _main()
{
    pushd "$(get_path_to_monorepo)" || exit
    cargo fmt --check
    cargo clippy --all-targets --all-features -- -D warnings --no-deps
    popd || exit
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
