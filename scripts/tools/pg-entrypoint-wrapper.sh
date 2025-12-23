#!/bin/sh
set -e

if [ ! -s "$PGDATA/PG_VERSION" ]; then
    echo "Fresh volume detected - running standard postgres initialization"
    exec docker-entrypoint.sh postgres "$@"
fi

# Existing volume => cleanup schemas after postgres starts
docker-entrypoint.sh postgres &

until pg_isready -U "$POSTGRES_USER"; do
  sleep 1
done

# Call a separate SQL cleanup script
/bin/cleanup-cpu-schemas.sh

wait -n
