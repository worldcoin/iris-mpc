CREATE TABLE IF NOT EXISTS results (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY (START WITH 0 MINVALUE 0) PRIMARY KEY,
    result_event TEXT NOT NULL
);
ALTER TABLE modifications ALTER COLUMN serial_id SET NOT NULL;
