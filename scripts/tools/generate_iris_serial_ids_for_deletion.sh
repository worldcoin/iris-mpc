#!/bin/bash
set -e

#######################################
# Utility: returns path to parent directory.
#######################################
function get_path_to_parent()
{
    local path_to_child=${1}

    echo "$( cd "$( dirname "${path_to_child}" )" && pwd )"
}

declare _HERE
declare _RESOURCES
declare _ROOT
declare _OUTPUT

# Set path -> here.
_HERE="$( get_path_to_parent "${BASH_SOURCE[0]}" )"

# Set path -> resources.
_RESOURCES="$( get_path_to_parent "${_HERE}" )/resources"

# Set path -> root.
_ROOT="$( get_path_to_parent "$( get_path_to_parent "${_HERE}" )" )"

# Set path -> output.
_OUTPUT="${_RESOURCES}/dev_serial_ids_marked_as_deleted.json"

# Execute job.
pushd "${_ROOT}" || exit
cargo run \
    --bin generate-serial-ids-for-deletion -- \
    --output="${_OUTPUT}"
popd || exit
