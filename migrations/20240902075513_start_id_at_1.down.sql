-- Add down migration script here
ALTER TABLE irises ALTER COLUMN id SET MINVALUE 0;
SELECT setval(pg_get_serial_sequence('irises', 'id'), coalesce(max(id),0), false) FROM irises;
ALTER TABLE irises ALTER COLUMN id SET START WITH 0;
