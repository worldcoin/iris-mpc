#!/usr/bin/env bash

function _help() {
    echo "
    COMMAND
    ----------------------------------------------------------------
    mpctl-services-net-aws-sm-rotate

    DESCRIPTION
    ----------------------------------------------------------------
    Rotates asymmetric key-pairs for all nodes within an MPC network.
    "
}

function _main()
{
    local idx_of_node

    log_break
    log "Rotating secret key rotation"
    log_break

    # Ensure AWS credentials are set.
    AWS_ACCESS_KEY_ID=$(get_aws_access_key_id)
    AWS_SECRET_ACCESS_KEY="$(get_aws_secret_access_key)"

    # Export to wider env.
    export AWS_ACCESS_KEY_ID
    export AWS_SECRET_ACCESS_KEY

    for _ in $(seq 0 1)
    do
        for idx_of_node in $(seq 0 "$((MPCTL_COUNT_OF_PARTIES - 1))")
        do
            log " ... rotating keys for node $idx_of_node"
            _rotate_keys "${idx_of_node}"
        done
    done

    log_break
    log "Rotating secret key rotation completed"
    log_break
}

function _rotate_keys()
{
    local idx_of_node=${1}

    pushd "$(get_path_to_monorepo)" || exit
    cargo run --bin \
        key-manager -- \
            --endpoint-url "$(get_aws_endpoint_url)" \
            --env "dev" \
            --node-id "${idx_of_node}" \
            --region "$(get_aws_region)" \
            rotate \
                --public-key-bucket-name wf-dev-public-keys
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
