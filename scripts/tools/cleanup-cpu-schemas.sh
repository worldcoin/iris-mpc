#!/bin/sh
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "SMPC_dev_0" <<EOF
DROP SCHEMA IF EXISTS "SMPC_cpu_dev_0" CASCADE;
EOF

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "SMPC_dev_1" <<EOF
DROP SCHEMA IF EXISTS "SMPC_cpu_dev_1" CASCADE;
EOF

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "SMPC_dev_2" <<EOF
DROP SCHEMA IF EXISTS "SMPC_cpu_dev_2" CASCADE;
EOF
