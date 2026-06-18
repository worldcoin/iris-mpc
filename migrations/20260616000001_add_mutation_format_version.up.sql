-- Add mutation_format_version column to track serialization format of serialized_mutations blob.
-- This enables backward compatibility when GraphMutation struct changes incompatibly.
--
-- Default of 1 back-fills all existing rows as V1 (the format that existed before versioning).
-- New writes set this column explicitly via serialize_mutations_current().

ALTER TABLE hawk_graph_mutations
  ADD COLUMN mutation_format_version SMALLINT NOT NULL DEFAULT 1;
