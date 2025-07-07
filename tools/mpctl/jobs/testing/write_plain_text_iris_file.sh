#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-job-init-plain-text-iris-file

    DESCRIPTION
    ----------------------------------------------------------------
    Initializes plain text iris files.
    "
}

function _main()
{
    local target_dir

    target_dir="$(get_path_to_assets)/data/iris-plaintext"

    if [ -d "${target_dir}" ]; then
        rm -rf "${target_dir}"
    fi

    pushd "$(get_path_to_monorepo)" || exit
    cargo run --bin generate_benchmark_data
    cp -r "$(pwd)/iris-mpc-cpu/data" "${target_dir}"
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
