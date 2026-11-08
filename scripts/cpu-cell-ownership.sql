-- Install in the canonical schema before enabling CPU handoff. The table's
-- presence makes startup require an explicit deployment identity.
CREATE TABLE cpu_cell_handoffs (
    sequence bigint GENERATED ALWAYS AS IDENTITY UNIQUE,
    handoff_id uuid PRIMARY KEY,
    source uuid UNIQUE NOT NULL,
    destination uuid UNIQUE NOT NULL,
    phase text NOT NULL DEFAULT 'requested'
        CHECK (phase IN ('requested', 'draining', 'drained', 'activated')),
    source_process uuid,
    candidate_processes uuid[],
    drain_report jsonb,
    CHECK (source <> destination),
    CHECK ((phase = 'requested') = (source_process IS NULL)),
    CHECK ((source_process IS NULL) = (candidate_processes IS NULL)),
    CHECK (candidate_processes IS NULL OR cardinality(candidate_processes) = 3),
    CHECK ((phase IN ('drained', 'activated')) = (drain_report IS NOT NULL))
);
CREATE UNIQUE INDEX cpu_cell_one_pending_handoff
    ON cpu_cell_handoffs ((true)) WHERE phase <> 'activated';
