#!/bin/sh
set -e

# Start Postgres in the background. docker-entrypoint.sh is built into the postgres image
docker-entrypoint.sh postgres &

# Wait for readiness
until pg_isready -U "$POSTGRES_USER"; do
  sleep 1
done

# Cleanup schemas

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "SMPC0" <<EOF
DROP SCHEMA IF EXISTS "genesis_cpu_0" CASCADE;
EOF

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "SMPC1" <<EOF
DROP SCHEMA IF EXISTS "genesis_cpu_1" CASCADE;
EOF

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "SMPC2" <<EOF
DROP SCHEMA IF EXISTS "genesis_cpu_2" CASCADE;
EOF

# Wait on background postgres (PID 1 swap)
wait -n
