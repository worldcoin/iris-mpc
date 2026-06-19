-- Rollback: remove mutation_format_version column
ALTER TABLE hawk_graph_mutations
  DROP COLUMN IF EXISTS mutation_format_version;
