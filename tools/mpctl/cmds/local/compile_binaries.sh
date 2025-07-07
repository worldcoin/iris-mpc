#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-infra-bin-compile

    DESCRIPTION
    ----------------------------------------------------------------
    Compiles complete set of system binaries.

    ARGS
    ----------------------------------------------------------------
    mode        Compilation mode: debug | release. Optional.

    DEFAULTS
    ----------------------------------------------------------------
    mode        release
    "
}

function _main()
{
    local build_mode=${1}

    _do_build "${build_mode}" "iris-mpc" "client"
    _do_build "${build_mode}" "iris-mpc" "iris-mpc-hawk"
    _do_build "${build_mode}" "iris-mpc-common" "key-manager"
    _do_build "${build_mode}" "iris-mpc-cpu" "graph-mem-cli"
    _do_build "${build_mode}" "iris-mpc-cpu" "init-test-dbs"
    _do_build "${build_mode}" "iris-mpc-cpu" "generate_benchmark_data"
    _do_build "${build_mode}" "iris-mpc-upgrade-hawk" "iris-mpc-hawk-genesis"
}

function _do_build()
{
    local build_mode=${1}
    local build_path
    local build_subdir=${2}
    local build_target=${3}

    build_path="$(get_path_to_monorepo)"
    if [ ! -d "${build_path}" ]; then
        log_error "Invalid build path: $build_path"
        return
    fi

    log "Compiling binary: ${build_subdir} :: ${build_target} :: ${build_mode}"

    pushd "${build_path}" || exit
    if [ "${build_mode}" == "debug" ]; then
        cargo build --bin "${build_target}"
    else
        cargo build --bin "${build_target}" --"${build_mode}"
    fi
    popd || exit
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "${MPCTL}"/utils/main.sh

unset _HELP
unset _BUILD_MODE

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)
    case "$KEY" in
        help) _HELP="show" ;;
        mode) _BUILD_MODE=${VALUE} ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main "${_BUILD_MODE:-"release"}"
fi
