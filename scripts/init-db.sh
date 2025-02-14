#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE "SMPC_dev_0";
    CREATE DATABASE "SMPC_dev_1";
    CREATE DATABASE "SMPC_dev_2";
EOSQL