#!/bin/sh
set -e

psql -U "$POSTGRES_USER" -d postgres <<EOF
DROP DATABASE IF EXISTS "SMPC_dev_0";
DROP DATABASE IF EXISTS "SMPC_dev_1";
DROP DATABASE IF EXISTS "SMPC_dev_2";
CREATE DATABASE "SMPC_dev_0";
CREATE DATABASE "SMPC_dev_1";
CREATE DATABASE "SMPC_dev_2";
EOF

wait_for_db() {
    DB="$1"
    echo "Waiting for database $DB to become ready..."
    until psql -U "$POSTGRES_USER" -d "$DB" -c "SELECT 1;" >/dev/null 2>&1; do
        sleep 1
    done
}

wait_for_db "SMPC_dev_0"
wait_for_db "SMPC_dev_1"
wait_for_db "SMPC_dev_2"

echo "All databases are ready."

URL_ONE="postgres://$POSTGRES_USER:$POSTGRES_PASSWORD@localhost:5432/SMPC_dev_0"
URL_TWO="postgres://$POSTGRES_USER:$POSTGRES_PASSWORD@localhost:5432/SMPC_dev_1"
URL_THREE="postgres://$POSTGRES_USER:$POSTGRES_PASSWORD@localhost:5432/SMPC_dev_2"

/bin/init-test-dbs --source /opt/irises.dat --skip-hnsw-graph --target-db-size=10000 --db-schema-party1 "SMPC_gpu_dev_0" --db-schema-party2 "SMPC_gpu_dev_1" --db-schema-party3 "SMPC_gpu_dev_2" --db-url-party1 $URL_ONE --db-url-party2 $URL_TWO --db-url-party3 $URL_THREE
