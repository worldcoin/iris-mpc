function _get_aws_opts_env()
{
    echo "$(cat $(_get_path_to_aws_opts)/env)"
}

function _get_now()
{
    echo $(date +%Y-%m-%dT%H:%M:%S.%6N)
}

function _get_path_to_ancestor()
{
    local path=${1}
    local steps=${2}

    for idx in $(seq 0 ${steps})
    do
        path=$( cd "$( dirname "${path}" )" && pwd )
    done

    echo ${path}
}

function _get_path_to_aws_opts()
{
    echo "$(_get_path_to_base)/aws-opts"
}

function _get_path_to_aws_opts_env()
{
    echo "$(_get_path_to_aws_opts)/${1}"
}

function _get_path_to_aws_opts_env_asset()
{
    echo "$(_get_path_to_aws_opts_env ${1})/${2}"
}

function _get_path_to_base()
{
    echo ${HOME}/.hnsw/service-client
}

function _get_path_to_exec_opts()
{
    echo "$(_get_path_to_base)/exec-opts"
}

function _get_path_to_here()
{
    echo $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
}

function _get_path_to_resources()
{
    echo "$(_get_path_to_root)/resources/iris-mpc-utils/service-client"
}

function _get_path_to_resource_of_env()
{
    echo "$(_get_path_to_resources)/${1}-${2}"
}

function _get_path_to_root()
{
    echo "$(_get_path_to_ancestor "$(_get_path_to_here)" "2")"
}

function _log_error ()
{
    local MSG=${1}

    echo -e "$(_get_now) [ERROR] [$$] HNSW-SERVICE-CLIENT :: $MSG"
}
