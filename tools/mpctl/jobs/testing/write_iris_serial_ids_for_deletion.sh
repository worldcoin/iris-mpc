#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-jobs-generate-iris-serial-ids-for-deletion

    DESCRIPTION
    ----------------------------------------------------------------
    Generates a file with Iris serial identifiers marked for deletion.
    "
}

function _main()
{
    local path_to_output

    path_to_output="$(get_path_to_resources)/misc/deleted_serial_ids.json"

    mkdir -p "$(get_path_to_resources)/misc"

    pushd "$(get_path_to_jobs)" || exit
    cargo run \
        --bin generate-serial-ids-for-deletion -- \
        --output="${path_to_output}"
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
