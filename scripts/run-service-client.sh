#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Ordered list of directories to search when resolving bare filenames.
SEARCH_DIRS=(
    "${PROJECT_ROOT}/iris-mpc-bins/data"
    "${PROJECT_ROOT}/iris-mpc-utils/assets/iris-codes-plaintext"
    "${PROJECT_ROOT}/iris-mpc-utils/assets/service-client"
    "${PROJECT_ROOT}/iris-mpc-utils/assets/aws-config"
)

usage() {
    cat <<EOF
Usage: $(basename "$0") [-e ENV] [-i IRIS_SHARES] [TOML_FILE]

Run the HNSW service client.

Options:
  -e ENV           Environment: dev-dkr (default) or dev-stg
  -i IRIS_SHARES   Path or filename of iris shares NDJSON file
                   (required for FromFile configs)
  -h               Show this help

Arguments:
  TOML_FILE   Path or filename of execution options TOML file
              (default: simple-1.toml)

File resolution order (for bare filenames):
  0. absolute path
  1. iris-mpc-bins/data/
  2. iris-mpc-utils/assets/iris-codes-plaintext/
  3. iris-mpc-utils/assets/service-client/
  4. iris-mpc-utils/assets/aws-config/

Examples:
  $(basename "$0")
  $(basename "$0") complex-1.toml -i 20250710-1k.ndjson
  $(basename "$0") -e dev-stg complex-1.toml -i 20250710-1k.ndjson
  $(basename "$0") -e dev-stg /absolute/path/to/custom.toml
EOF
}

# Resolve a filename by searching known directories.
resolve_file() {
    local filename="$1"
    local label="$2"

    # Absolute or relative path that exists as-is.
    if [[ -f "$filename" ]]; then
        realpath "$filename"
        return 0
    fi

    # Search known directories.
    for dir in "${SEARCH_DIRS[@]}"; do
        if [[ -f "${dir}/${filename}" ]]; then
            realpath "${dir}/${filename}"
            return 0
        fi
    done

    echo "Error: ${label} not found: ${filename}" >&2
    echo "Searched: . ${SEARCH_DIRS[*]}" >&2
    return 1
}

env="dev-dkr"
exec_opts=""
iris_shares=""

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
        -i)
            if [[ $# -lt 2 || -z "${2:-}" ]]; then
                echo "Error: -i requires a non-empty iris shares path." >&2
                usage
                exit 1
            fi
            iris_shares="$2"
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

# Resolve the AWS env config.
aws_opts="$(resolve_file "${env:-dev-dkr}.toml" "AWS environment config")"

# Resolve the execution options TOML.
exec_opts="$(resolve_file "${exec_opts:-simple-1.toml}" "TOML file")"

# Build optional iris shares argument.
iris_shares_args=()
if [[ -n "${iris_shares}" ]]; then
    iris_shares="$(resolve_file "${iris_shares}" "iris shares NDJSON file")"
    iris_shares_args=(--path-to-iris-shares "${iris_shares}")
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

cd "${PROJECT_ROOT}"
exec cargo run --release -p iris-mpc-bins --bin service-client -- \
    --path-to-opts "${exec_opts}" \
    --path-to-opts-aws "${aws_opts}" \
    "${iris_shares_args[@]+"${iris_shares_args[@]}"}"
