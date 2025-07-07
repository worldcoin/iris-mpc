#!/usr/bin/env bash

#######################################
# Returns path to primary assets folder.
# Globals:
#   MPCTL - path to mpctl home directory.
#######################################
function get_path_to_assets()
{
    echo "${MPCTL_ASSETS:-${MPCTL}/assets}"
}

#######################################
# Returns path to primary assets folder for a local test network.
# Globals:
#   MPCTL - path to mpctl home directory.
#######################################
function get_path_to_assets_of_net()
{
    echo "$(get_path_to_assets)/net"
}

#######################################
# Returns path to a local test node's assets.
# Arguments:
#   Node ordinal identifier.
#######################################
function get_path_to_assets_of_node()
{
    local idx_of_node=${1}

    echo "$(get_path_to_assets_of_net)"/nodes/node-"$idx_of_node"
}

#######################################
# Returns path to a docker compose file for managing network nodes.
#######################################
function get_path_to_docker_compose_file_of_node()
{
    local binary=${1}

    if [ "${binary}" == "genesis" ]; then
        echo "$(get_path_to_monorepo)/${MPCTL_DKR_COMPOSE_HAWK_GENESIS}"
    else
        echo "$(get_path_to_monorepo)/${MPCTL_DKR_COMPOSE_HAWK}"
    fi
}

#######################################
# Returns path to jobs crate.
# Globals:
#   MPCTL - path to mpctl home directory.
#######################################
function get_path_to_jobs()
{
    echo "${MPCTL}/jobs"
}

#######################################
# Returns path to the monorepo within which solution has been developed.
#######################################
function get_path_to_monorepo()
{
    echo "$(get_path_to_parent_dir $(get_path_to_parent_dir ${MPCTL}))"
}

#######################################
# Returns path to a node's env directory.
# Arguments:
#   Node ordinal identifier.
#######################################
function get_path_to_node_env()
{
    local idx_of_node=${1}

    echo "$(get_path_to_assets_of_node "${idx_of_node}")/env"
}

#######################################
# Returns path to a node's logs directory.
# Arguments:
#   Node ordinal identifier.
#######################################
function get_path_to_node_logs()
{
    local idx_of_node=${1}

    echo "$(get_path_to_assets_of_node "$idx_of_node")"/logs
}

#######################################
# Returns path to parent directory.
#######################################
function get_path_to_parent_dir()
{
    local child_dir=${1}

    echo $( cd "$( dirname "${child_dir[0]}" )" && pwd )
}

#######################################
# Returns path to primary resources folder.
# Globals:
#   MPCTL - path to mpctl home directory.
#######################################
function get_path_to_resources()
{
    echo "${MPCTL}/resources"
}

#######################################
# Returns path to a target binary.
#######################################
function get_path_to_target_binary()
{
    local name_of_binary=${1}
    local build_mode=${2}

    echo "$(get_path_to_monorepo)/target/$build_mode/$name_of_binary"
}
