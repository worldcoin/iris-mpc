-- Step 1: Create a new table with the desired schema with BIGSERIAL for id
CREATE TABLE irises_old (
    id BIGSERIAL PRIMARY KEY,
    code BYTEA,
    mask BYTEA
);

-- Step 2: Copy the data from the current table to the new table (using fresh sequence generator for id)
INSERT INTO irises_old (code, mask)
SELECT code, mask FROM irises
ORDER BY id;

-- Step 3: Drop the current table
DROP TABLE irises;

-- Step 4: Rename the new table to the original table's name
ALTER TABLE irises_old RENAME TO irises;
