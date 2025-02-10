#!/bin/bash

# Function to display help
usage() {
    echo "Usage: $0 <local-party-id> <target-party-id> <db-start> <db-end>"
    echo "Example: $0 0 2 1 10001"
    exit 1
}

# Check if correct number of arguments is provided
if [ $# -ne 4 ]; then
    usage
fi

# Check required environment variables
if [ -z "$DATABASE_URL" ]; then
    echo "Error: DATABASE_URL environment variable is not set"
    exit 1
fi

if [ -z "$ENVIRONMENT" ]; then
    echo "Error: ENVIRONMENT environment variable is not set"
    exit 1
fi

# Get arguments
LOCAL_PARTY_ID=$1
TARGET_PARTY_ID=$2
DB_START=$3
DB_END=$4

# Validate local party ID
if [[ ! "$LOCAL_PARTY_ID" =~ ^[0-2]$ ]]; then
    echo "Error: Local party ID must be 0, 1, or 2"
    exit 1
fi

# Validate target party ID
if [[ ! "$TARGET_PARTY_ID" =~ ^[0-2]$ ]]; then
    echo "Error: Target party ID must be 0, 1, or 2"
    exit 1
fi

# Prevent running on the same party
if [ "$LOCAL_PARTY_ID" -eq "$TARGET_PARTY_ID" ]; then
    echo "Error: Local party ID cannot be the same as target party ID"
    echo "You cannot run reshare on the same party you are currently on"
    exit 1
fi

# Dynamically generate server_url
SERVER_URL="https://reshare-server.${TARGET_PARTY_ID}.stage.smpcv2.worldcoin.dev:6443"

# Constant parameters
BATCH_SIZE=100
RESHARE_RUN_SESSION_ID="session-${TARGET_PARTY_ID}"
CLIENT_TLS_CERT_PATH="/usr/local/share/ca-certificates/aws_orb_prod_private_ca.crt"

# Validate DB start and end values
if ! [[ "$DB_START" =~ ^[0-9]+$ ]] || ! [[ "$DB_END" =~ ^[0-9]+$ ]]; then
    echo "Error: DB start and end must be numeric values"
    exit 1
fi

if [ "$DB_START" -ge "$DB_END" ]; then
    echo "Error: DB start must be less than DB end"
    exit 1
fi

case $LOCAL_PARTY_ID in
0)
    MY_KMS_KEY_ARN="${KMS_P0}"
    ;;
1)
    MY_KMS_KEY_ARN="${KMS_P1}"
    ;;
2)
    MY_KMS_KEY_ARN="${KMS_P2}"
    ;;
esac

# Dynamically set other party-id considering target-party-id and local party ID
case $TARGET_PARTY_ID in
0)
    OTHER_PARTY_ID=$([ "$LOCAL_PARTY_ID" -eq 1 ] && echo 2 || echo 1)
    OTHER_KMS_KEY_ARN=$([ "$LOCAL_PARTY_ID" -eq 1 ] && echo ${KMS_P2} || echo ${KMS_P1})
    ;;
1)
    OTHER_PARTY_ID=$([ "$LOCAL_PARTY_ID" -eq 0 ] && echo 2 || echo 0)
    OTHER_KMS_KEY_ARN=$([ "$LOCAL_PARTY_ID" -eq 0 ] && echo ${KMS_P2} || echo ${KMS_P0})
    ;;
2)
    OTHER_PARTY_ID=$([ "$LOCAL_PARTY_ID" -eq 0 ] && echo 1 || echo 0)
    OTHER_KMS_KEY_ARN=$([ "$LOCAL_PARTY_ID" -eq 0 ] && echo ${KMS_P1} || echo ${KMS_P0})
    ;;
esac

# Check if KMS environment variables are set
if [ -z "$MY_KMS_KEY_ARN" ] || [ -z "$OTHER_KMS_KEY_ARN" ]; then
    echo "Error: KMS environment variables are not set for the specified party."
    echo "Please set KMS_P0, KMS_P1, and KMS_P2 environment variables."
    exit 1
fi

# Build full command
COMMAND="reshare-client \
    --party-id $LOCAL_PARTY_ID \
    --other-party-id $OTHER_PARTY_ID \
    --target-party-id $TARGET_PARTY_ID \
    --server-url $SERVER_URL \
    --environment $ENVIRONMENT \
    --db-url $DATABASE_URL \
    --db-start $DB_START \
    --db-end $DB_END \
    --batch-size $BATCH_SIZE \
    --my-kms-key-arn $MY_KMS_KEY_ARN \
    --other-kms-key-arn $OTHER_KMS_KEY_ARN \
    --reshare-run-session-id $RESHARE_RUN_SESSION_ID \
    --client-tls-cert-path $CLIENT_TLS_CERT_PATH"

# Display or execute command
echo "Generated command:"
echo "$COMMAND"

exec $COMMAND
