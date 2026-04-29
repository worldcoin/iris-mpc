-- Create graph_mutations table to serve as a diff based write ahead log.
-- modification_id may point to a row that has been applied and is deleted.
CREATE TABLE IF NOT EXISTS hawk_graph_mutations (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    modification_id BIGINT NOT NULL,
    -- BothEyes<Vec<GraphMutation>>
    serialized_mutations BYTEA NOT NULL
);
