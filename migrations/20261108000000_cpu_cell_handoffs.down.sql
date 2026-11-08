-- Handoff history is a restart fence and cannot be discarded by a rollback.
LOCK TABLE cpu_cell_handoffs IN ACCESS EXCLUSIVE MODE;
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM cpu_cell_handoffs) THEN
        RAISE EXCEPTION 'Cannot remove recorded CPU handoff history';
    END IF;
END $$;
DROP TABLE cpu_cell_handoffs;
