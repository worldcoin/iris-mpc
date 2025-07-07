#!/usr/bin/env bash

# Docker compose file: Hawk main.
export MPCTL_DKR_COMPOSE_HAWK="docker-compose.test.yaml"

# Docker compose file: Hawk main.
export MPCTL_DKR_COMPOSE_HAWK_GENESIS="docker-compose.test.genesis.yaml"

# Docker compose file: Base services.
export MPCTL_DKR_COMPOSE_SERVICES="docker-compose.dev.yaml"

# Docker container id: Hawk node.
export MPCTL_DKR_CONTAINER_HAWK_NODE="hawk_participant_"

# Docker container id: PostgreSQL dB.
export MPCTL_DKR_CONTAINER_PGRES_DB="iris-mpc-dev_db-1"

# Docker file: Hawk node.
export MPCTL_DKR_FILE_STANDARD="Dockerfile.dev.hawk"

# Docker file: Hawk node at genesis.
export MPCTL_DKR_FILE_GENESIS="Dockerfile.dev.hawk"

# Docker image id: Hawk node.
export MPCTL_DKR_IMAGE_NAME_STANDARD="hawk-server-local-build"

# Docker image id: Hawk node at genesis.
export MPCTL_DKR_IMAGE_NAME_GENESIS="hawk-server-genesis"

##############################################################################
# Returns a node's docker container name.
##############################################################################
function get_name_of_docker_container_of_node()
{
    local idx_of_node=${1}

    echo "${MPCTL_DKR_CONTAINER_HAWK_NODE}${idx_of_node}"
}
