-- Step 1: Rename left_code back to code and left_mask back to mask
ALTER TABLE irises RENAME COLUMN left_code TO code;
ALTER TABLE irises RENAME COLUMN left_mask TO mask;

-- Step 2: Drop the right_code and right_mask columns
ALTER TABLE irises DROP COLUMN right_code;
ALTER TABLE irises DROP COLUMN right_mask;
