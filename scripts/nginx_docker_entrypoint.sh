#!/usr/bin/env bash

set -e

# Paths to store the files (adjust as needed)
SSL_CERT_PATH=${SSL_CERT_PATH:-"/etc/ssl/cert/certificate.crt"}
PRIVATE_KEY_PATH=${PRIVATE_KEY_PATH:-"/etc/ssl/cert/key.pem"}

# Ensure that all secret IDs are provided
if [ -z "$SSL_CERT_SECRET_ID" ] || [ -z "$PRIVATE_KEY_SECRET_ID" ]; then
    echo "Error: One or more secret IDs are not set."
    exit 1
fi

# Fetch the SSL certificate from AWS Secrets Manager
echo "Fetching SSL certificate..."
aws secretsmanager get-secret-value \
    --secret-id "$SSL_CERT_SECRET_ID" \
    --query 'SecretString' \
    --output text > "$SSL_CERT_PATH"

# Fetch the private key from AWS Secrets Manager
echo "Fetching private key..."
aws secretsmanager get-secret-value \
    --secret-id "$PRIVATE_KEY_SECRET_ID" \
    --query 'SecretString' \
    --output text > "$PRIVATE_KEY_PATH"

# Set appropriate permissions for the private key and password files
chmod 600 "$PRIVATE_KEY_PATH"

# Execute the regular command
exec "$@"
