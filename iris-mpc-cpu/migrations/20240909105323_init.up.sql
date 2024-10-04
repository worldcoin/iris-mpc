CREATE TABLE IF NOT EXISTS hawk_vectors (
    id integer NOT NULL,
    point jsonb NOT NULL,
    CONSTRAINT hawk_vectors_pkey PRIMARY KEY (id)
);
