#!/usr/bin/env bash
#
# Run the continuous rerandomization e2e chaos tests.
# Starts Postgres + localstack via docker-compose, runs the Rust tests, then
# tears everything down.
#
# Usage:
#   ./run-rerand-e2e-tests.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.rand.yaml"

cleanup() {
    echo "=== Tearing down containers ==="
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans -v 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Starting Postgres + localstack ==="
docker-compose -f "$COMPOSE_FILE" down --remove-orphans -v 2>/dev/null || true
docker-compose -f "$COMPOSE_FILE" up -d

echo "Waiting for services to be ready..."
for i in $(seq 1 30); do
    if docker exec iris-mpc-upgrade-new-db-1-1 pg_isready -U postgres -q 2>/dev/null; then
        break
    fi
    sleep 1
done
docker exec iris-mpc-upgrade-new-db-1-1 pg_isready -U postgres || { echo "Postgres not ready"; exit 1; }

for i in $(seq 1 30); do
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' iris-mpc-upgrade-localstack-1 2>/dev/null || echo "unknown")
    if [ "$STATUS" = "healthy" ]; then
        break
    fi
    sleep 1
done
echo "Infrastructure ready."

echo "=== Running e2e chaos tests ==="
cd "$REPO_ROOT"
AWS_ACCESS_KEY_ID=test \
AWS_SECRET_ACCESS_KEY=test \
AWS_DEFAULT_REGION=us-east-1 \
AWS_ENDPOINT_URL=http://127.0.0.1:4566 \
ENVIRONMENT=testing \
    cargo test -p iris-mpc-upgrade --test continuous_rerand_e2e --features db_dependent -- --nocapture

echo "=== All tests passed ==="
