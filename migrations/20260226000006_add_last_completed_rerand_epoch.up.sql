ALTER TABLE rerand_control
    ADD COLUMN IF NOT EXISTS last_completed_epoch INTEGER;
