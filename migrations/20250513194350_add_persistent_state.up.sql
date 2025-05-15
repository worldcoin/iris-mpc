-- Add persistent server state table --

CREATE TABLE IF NOT EXISTS persistent_state
(
    domain text NOT NULL,
    "key" text NOT NULL,
    "value" jsonb NOT NULL,
    CONSTRAINT persistent_state_pkey PRIMARY KEY (domain, "key")
);
