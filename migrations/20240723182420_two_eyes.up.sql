-- Step 1: Add new columns for right_code and right_mask with type BYTEA
ALTER TABLE irises ADD COLUMN right_code BYTEA;
ALTER TABLE irises ADD COLUMN right_mask BYTEA;

-- Step 2: Rename existing columns code to left_code and mask to left_mask
ALTER TABLE irises RENAME COLUMN code TO left_code;
ALTER TABLE irises RENAME COLUMN mask TO left_mask;

-- Step 3: Populate the new columns with an empty binary string instead of NULL
UPDATE irises SET right_code = '\x', right_mask = '\x';
