-- Add up migration script here
-- we get the current maxixum id from the table and 1 otherwise and set the sequence to that value
SELECT setval(pg_get_serial_sequence('irises', 'id'), coalesce(max(id),1), false) FROM irises;
ALTER TABLE irises ALTER COLUMN id SET START WITH 1;
ALTER TABLE irises ALTER COLUMN id SET MINVALUE 1;
