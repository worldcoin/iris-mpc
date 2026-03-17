#!/usr/bin/env bash
#
# Set or clear fixed_batch_size on hawk server parties via /config endpoint.
#
# Usage:
#   set-batch-size.sh <size> <url1> [url2] [url3] ...
#   set-batch-size.sh clear  <url1> [url2] [url3] ...
#
# Examples:
#   ./scripts/set-batch-size.sh 5 http://localhost:3000 http://localhost:3001 http://localhost:3002
#   ./scripts/set-batch-size.sh clear http://localhost:3000 http://localhost:3001 http://localhost:3002
#
# Exits 0 on success, non-zero if any party fails to confirm.

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <size|clear> <url1> [url2] ..." >&2
    exit 1
fi

ACTION="$1"
shift

if [ "$ACTION" = "clear" ]; then
    BODY='{"fixed_batch_size": null}'
else
    BODY="{\"fixed_batch_size\": $ACTION}"
fi

PARTY_IDX=0
for BASE_URL in "$@"; do
    URL="${BASE_URL}/config"

    RESPONSE=$(curl -sf -X POST "$URL" \
        -H 'Content-Type: application/json' \
        -d "$BODY") || {
        echo "ERROR: Party $PARTY_IDX: POST $URL failed" >&2
        exit 1
    }

    # For set (not clear), verify the server confirmed the expected value.
    if [ "$ACTION" != "clear" ]; then
        # Extract fixed_batch_size from JSON response using python3 (widely available).
        CONFIRMED=$(echo "$RESPONSE" | python3 -c \
            "import sys,json; v=json.load(sys.stdin).get('fixed_batch_size'); print(v if v is not None else '')" 2>/dev/null) || true

        if [ "$CONFIRMED" != "$ACTION" ]; then
            echo "ERROR: Party $PARTY_IDX: expected fixed_batch_size=$ACTION, got '$CONFIRMED'" >&2
            exit 1
        fi
    fi

    echo "Party $PARTY_IDX: OK ($RESPONSE)"
    PARTY_IDX=$((PARTY_IDX + 1))
done
