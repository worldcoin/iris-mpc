#!/bin/sh
set -e

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

pg_restore -U postgres -d SMPC_dev_0 --clean --if-exists /opt/smpc_dev_0.sql
pg_restore -U postgres -d SMPC_dev_1 --clean --if-exists /opt/smpc_dev_1.sql
pg_restore -U postgres -d SMPC_dev_2 --clean --if-exists /opt/smpc_dev_2.sql
