#!/usr/bin/env bash

# OS types.
declare _OS_LINUX="linux"
declare _OS_LINUX_REDHAT="$_OS_LINUX-redhat"
declare _OS_LINUX_SUSE="$_OS_LINUX-suse"
declare _OS_LINUX_ARCH="$_OS_LINUX-arch"
declare _OS_LINUX_DEBIAN="$_OS_LINUX-debian"
declare _OS_MACOSX="macosx"
declare _OS_UNKNOWN="unknown"

#######################################
# Returns OS wall clock time.
#######################################
function get_now()
{
    echo $(date +%Y-%m-%dT%H:%M:%S.%6N)
}

#######################################
# Returns OS type.
# Globals:
#   OSTYPE: type of OS being run.
#######################################
function get_os()
{
	if [[ "$OSTYPE" == "linux-gnu" ]]; then
		if [ -f /etc/redhat-release ]; then
			echo $_OS_LINUX_REDHAT
		elif [ -f /etc/SuSE-release ]; then
			echo $_OS_LINUX_SUSE
		elif [ -f /etc/arch-release ]; then
			echo $_OS_LINUX_ARCH
		elif [ -f /etc/debian_version ]; then
			echo $_OS_LINUX_DEBIAN
		fi
	elif [[ "$OSTYPE" == "darwin"* ]]; then
		echo $_OS_MACOSX
	else
		echo $_OS_UNKNOWN
	fi
}

#######################################
# Returns true if passed environment variable is set.
#######################################
function is_env_var_set ()
{
    local var_name="$1"
    local expected_value="${2:-}"

    # Method 1: Check if variable is unset or empty
    if [[ -z "${!var_name+x}" ]]; then
        return 1
    fi

    # If an expected value is provided, check for exact match
    if [[ -n "$expected_value" ]]; then
        if [[ "${!var_name}" != "$expected_value" ]]; then
            return 1
        fi
    fi

    # Variable is set (and matches expected value if provided)
    return 0
}

#######################################
# Wraps standard echo by adding application prefix.
#######################################
function log ()
{
    local MSG=${1}

    echo -e "$(get_now) [INFO] [$$] MPCTL :: ${MSG}"
}

#######################################
# Line break logging helper.
#######################################
function log_break()
{
    log "---------------------------------------------------------------------------------"
}

#######################################
# Wraps standard echo by adding application error prefix.
#######################################
function log_error ()
{
    local MSG=${1}

    echo -e "$(get_now) [ERROR] [$$] MPCTL :: $MSG"
}

#######################################
# Step logging helper..
#######################################
function log_step_upgrades()
{
    local STEP_ID=${1}
    local MSG=${2}
    local PREFIX=${3:-""}

    log_break
    if [ "$PREFIX" == "" ]; then
        log "STEP $STEP_ID: $MSG"
    else
        log "$PREFIX STEP $STEP_ID: $MSG"
    fi
}

#######################################
# Logs a warning message.
#######################################
function log_warning()
{
    local MSG=${1}

    echo -e "$(get_now) [WARN] [$$] MPCTL :: ${MSG}"
}

#######################################
# Wraps pushd command to suppress stdout.
#######################################
function pushd ()
{
    command pushd "$@" > /dev/null
}

#######################################
# Wraps popd command to suppress stdout.
#######################################
function popd ()
{
    command popd "$@" > /dev/null
}
