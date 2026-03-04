CREATE TABLE IF NOT EXISTS rerand_control (
    id              INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    freeze_requested    BOOLEAN NOT NULL DEFAULT FALSE,
    freeze_generation   TEXT,
    frozen_generation   TEXT
);

INSERT INTO rerand_control (id) VALUES (1) ON CONFLICT DO NOTHING;
