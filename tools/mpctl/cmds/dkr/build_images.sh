#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-dkr-build-images

    DESCRIPTION
    ----------------------------------------------------------------
    Builds Hawk server docker images.
    "
}

function _main()
{
    # Hawk server: standard.
    _build_image "${MPCTL_DKR_FILE_STANDARD}" "${MPCTL_DKR_IMAGE_NAME_STANDARD}"

    # Hawk server: genesis.
    _build_image "${MPCTL_DKR_FILE_GENESIS}" "${MPCTL_DKR_IMAGE_NAME_GENESIS}"
}

function _build_image()
{
    local image_fname=${1}
    local image_tag=${2}

    pushd "$(get_path_to_monorepo)" || exit
    docker build \
        -f "$(get_path_to_monorepo)/${image_fname}" \
        -t "${image_tag}:latest" .
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
