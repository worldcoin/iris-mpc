#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IRIS_MPC_BINS="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [-e ENV] [EXEC_OPTS_TOML]

Run the HNSW service client.

Options:
  -e ENV    Environment: dev-dkr (default) or dev-stg
  -h        Show this help

Arguments:
  EXEC_OPTS_TOML  Path to execution options TOML file
                  (default: requests/simple-1.toml)

Examples:
  $(basename "$0")
  $(basename "$0") -e dev-stg
  $(basename "$0") -e dev-stg path/to/custom.toml
EOF
}

env="dev-dkr"
exec_opts=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e)
            if [[ $# -lt 2 || -z "${2:-}" ]]; then
                echo "Error: -e requires a non-empty environment argument." >&2
                usage
                exit 1
            fi
            env="$2"
            shift 2
            ;;
        -h)
            usage
            exit 0
            ;;
        *)
            exec_opts="$1"
            shift
            ;;
    esac
done

aws_opts="${SCRIPT_DIR}/env/${env}.toml"
exec_opts="${exec_opts:-${SCRIPT_DIR}/requests/simple-1.toml}"

if [[ ! -f "${aws_opts}" ]]; then
    echo "Error: unknown environment '${env}'" >&2
    echo "Valid: dev-dkr, dev-stg" >&2
    exit 1
fi

if [[ ! -f "${exec_opts}" ]]; then
    echo "Error: file not found: ${exec_opts}" >&2
    exit 1
fi

# Clear any stale AWS env vars - let the profile handle auth.
unset AWS_ACCESS_KEY_ID  2>/dev/null || true
unset AWS_SECRET_ACCESS_KEY 2>/dev/null || true
unset AWS_REGION 2>/dev/null || true
unset AWS_ENDPOINT_URL 2>/dev/null || true

case "${env}" in
    dev-dkr)
        export AWS_PROFILE="worldcoin-smpcv-io-vpc-dev-dkr"
        export AWS_ENDPOINT_URL="http://localhost:4566"
        ;;
    dev-stg)
        export AWS_PROFILE="worldcoin-smpcv-io-vpc-dev"
        ;;
esac

cd "${IRIS_MPC_BINS}"
exec cargo run --release --bin service-client -- \
    --path-to-opts "${exec_opts}" \
    --path-to-opts-aws "${aws_opts}"
