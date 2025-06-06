CREATE DATABASE "SMPC_dev_0";
CREATE DATABASE "SMPC_dev_1";
CREATE DATABASE "SMPC_dev_2";

CREATE ROLE ro_user WITH LOGIN PASSWORD 'postgres';

GRANT CONNECT ON DATABASE "SMPC_dev_0" TO ro_user;
GRANT CONNECT ON DATABASE "SMPC_dev_1" TO ro_user;
GRANT CONNECT ON DATABASE "SMPC_dev_2" TO ro_user;

\connect "SMPC_dev_0"
CREATE SCHEMA "SMPC_dev_0";
SET search_path TO "SMPC_dev_0";
GRANT USAGE ON SCHEMA "SMPC_dev_0" TO ro_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA "SMPC_dev_0" GRANT SELECT, INSERT, UPDATE, DELETE ON tables TO ro_user;

\connect "SMPC_dev_1"
CREATE SCHEMA "SMPC_dev_1";
SET search_path TO "SMPC_dev_1";
GRANT USAGE ON SCHEMA "SMPC_dev_1" TO ro_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA "SMPC_dev_1" GRANT SELECT, INSERT, UPDATE, DELETE ON tables TO ro_user;

\connect "SMPC_dev_2"
CREATE SCHEMA "SMPC_dev_2";
SET search_path TO "SMPC_dev_2";
GRANT USAGE ON SCHEMA "SMPC_dev_2" TO ro_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA "SMPC_dev_2" GRANT SELECT, INSERT, UPDATE, DELETE ON tables TO ro_user;
