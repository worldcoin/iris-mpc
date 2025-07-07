#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-infra-net-setup

    DESCRIPTION
    ----------------------------------------------------------------
    Sets up assets for an MPC network.

    ARGS
    ----------------------------------------------------------------
    buildmode        Compilation mode: debug | release. Optional.

    DEFAULTS
    ----------------------------------------------------------------
    buildmode        release
    "
}

function _main()
{
    local build_mode=${1}

    log_break
    log "MPC network setup :: begins"

    _setup_fs
    log "    file system initialised"

    _setup_config
    log "    configuration initialised"

    _setup_binaries "${build_mode}"
    log "    binaries compiled"
    log "    binaries assigned"

    _setup_keys
    log "    secret keys initialised"

    log "MPC network setup :: ends"
    log_break
}

##############################################################################
# Initialises a networks's binary files.
##############################################################################
function _setup_binaries()
{
    local build_mode=${1}
    local idx_of_node

    # Compile binary set.
    source "${MPCTL}"/cmds/local/compile_binaries.sh mode="${build_mode}"

    # Copy net binaries.
    cp \
        "$(get_path_to_target_binary client ${build_mode})" \
        "$(get_path_to_assets_of_net)/bin"
    cp \
        "$(get_path_to_target_binary generate_benchmark_data ${build_mode})" \
        "$(get_path_to_assets_of_net)/bin"
    cp \
        "$(get_path_to_target_binary graph-mem-cli ${build_mode})" \
        "$(get_path_to_assets_of_net)/bin"
    cp \
        "$(get_path_to_target_binary init-test-dbs ${build_mode})" \
        "$(get_path_to_assets_of_net)/bin"
    cp \
        "$(get_path_to_target_binary key-manager ${build_mode})" \
        "$(get_path_to_assets_of_net)/bin"

    # Copy node binaries.
    for idx_of_node in $(seq 0 "$((MPCTL_COUNT_OF_PARTIES - 1))")
    do
        cp \
            "$(get_path_to_target_binary iris-mpc-hawk ${build_mode})" \
            "$(get_path_to_assets_of_node ${idx_of_node})/bin"
        cp \
            "$(get_path_to_target_binary iris-mpc-hawk-genesis ${build_mode})" \
            "$(get_path_to_assets_of_node ${idx_of_node})/bin"
    done
}

##############################################################################
# Initialises a networks's configuration files.
##############################################################################
function _setup_config()
{
    local idx_of_node

    for idx_of_node in $(seq 0 "$((MPCTL_COUNT_OF_PARTIES - 1))")
    do
        _setup_config_of_node "${idx_of_node}"
    done
}

##############################################################################
# Initialises a node's configuration files.
##############################################################################
function _setup_config_of_node()
{
    local idx_of_node=${1}
    local path_to_assets_of_node

    path_to_assets_of_node="$(get_path_to_assets_of_node "${idx_of_node}")"

    # Env vars.
    cp \
        "$(get_path_to_resources)/envs/direnv.toml" \
        "${path_to_assets_of_node}/env"
    cp \
        "$(get_path_to_resources)/envs/.envrc" \
        "${path_to_assets_of_node}/env"
    cp \
        "$(get_path_to_resources)/envs/local.base.env" \
        "${path_to_assets_of_node}/env/base.env"
    cp \
        "$(get_path_to_resources)/envs/local.node.${idx_of_node}.env" \
        "${path_to_assets_of_node}/env/node.env"
}

##############################################################################
# Initialises filesystem network assets.
##############################################################################
function _setup_fs()
{
    local idx_of_node
    local path_to_assets_of_node

    mkdir -p "$(get_path_to_assets_of_net)/bin"
    for idx_of_node in $(seq 0 "$((MPCTL_COUNT_OF_PARTIES - 1))")
    do
        path_to_assets_of_node="$(get_path_to_assets_of_node "${idx_of_node}")"
        mkdir -p "${path_to_assets_of_node}/bin"
        mkdir "${path_to_assets_of_node}/env"
        mkdir "${path_to_assets_of_node}/logs"
    done
}

##############################################################################
# Executes key-manager in order to generate rotating keys for each node within AWS KMS.
##############################################################################
function _setup_keys()
{
    source "${MPCTL}"/jobs/services/aws_sm_rotate.sh
}

# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

source "${MPCTL}"/utils/main.sh

unset _BUILD_MODE
unset _HELP

for ARGUMENT in "$@"
do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    case "$KEY" in
        buildmode) _BUILD_MODE=${VALUE} ;;
        help) _HELP="show" ;;
        *)
    esac
done

if [ "${_HELP:-""}" = "show" ]; then
    _help
else
    _main "${_BUILD_MODE:-"release"}"
fi
