#!/usr/bin/env bash

# Default number of parties participating in MPC protocol.
export MPCTL_COUNT_OF_PARTIES=3

# Name of application monorepo.
export MPCTL_NAME_OF_MONREPO="iris-mpc"

# Default: name of application environment.
export MPCTL_DEFAULT_ENVIRONMENT="dev"

##############################################################################
# Returns name of current environment.
##############################################################################
function get_environment_name()
{
    if is_env_var_set "SMPC__ENVIRONMENT"; then
        echo "${SMPC__ENVIRONMENT}"
    else
        echo "${MPCTL_DEFAULT_ENVIRONMENT}"
    fi
}

# Import other utils - order matters.
source "${MPCTL}"/utils/os.sh
source "${MPCTL}"/utils/paths.sh
source "${MPCTL}"/utils/service_aws.sh
source "${MPCTL}"/utils/service_docker.sh
source "${MPCTL}"/utils/service_pgres.sh
