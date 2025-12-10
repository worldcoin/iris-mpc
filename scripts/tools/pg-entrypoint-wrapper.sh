#!/bin/sh
set -e


# Cleanup schemas

if [ ! -s "$PGDATA/PG_VERSION" ]; then
    echo "Fresh volume detected - running standard postgres initialization"
    # Standard postgres entrypoint will handle initialization including init-db-custom.sh
    exec docker-entrypoint.sh postgres "$@"
else
    # Start Postgres in the background. docker-entrypoint.sh is built into the postgres image
    docker-entrypoint.sh postgres &

    # Wait for readiness
    until pg_isready -U "$POSTGRES_USER"; do
      sleep 1
    done

    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "SMPC0" <<EOF
    DROP SCHEMA IF EXISTS "genesis_cpu_dev_0" CASCADE;
    EOF

    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "SMPC1" <<EOF
    DROP SCHEMA IF EXISTS "genesis_cpu_dev_1" CASCADE;
    EOF

    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "SMPC2" <<EOF
    DROP SCHEMA IF EXISTS "genesis_cpu_dev_2" CASCADE;
    EOF

    wait -n
fi
