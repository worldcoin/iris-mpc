# Test cases

## 100

Preconditions:
GPU iris database has entries from 1 to 100, inclusive
CPU iris database and graph database is empty
CPU modifications and persisted_state tables are empty
GPU modifications and persisted_state tables are empty
There are zero deletions on S3

Test:
index genesis up to 100

Postconditions:
GPU iris database has entries from 1 to 100, inclusive
CPU iris database has entries from 1 to 100, inclusive
CPU modifications table is empty
CPU persisted_state table shows the max indexed modification is 0 and max indexed iris is 100
CPU graph database matches the output of plaintext genesis
There are zero deletions on S3

## 101

Preconditions:
same as 100

Test:
index genesis up to 50
index genesis up to 100

Postconditions:
same as 100

## 102

Preconditions:
GPU iris database has entries from 1 to 100, inclusive
CPU iris database and graph database is empty
CPU modifications and persisted_state tables are empty
GPU persisted_state table is empty
S3 deletions: [1, 10, 20, 50, 100]

Test:
index genesis up to 100

Postconditions:
GPU iris database has entries from 1 to 100, inclusive
CPU iris database has entries from 1 to 100, inclusive
CPU modifications table is empty
CPU persisted_state table shows the max indexed modification is 0 and max indexed iris is 100
CPU graph database matches the output of plaintext genesis
CPU graph database at layer zero has 95 links (100 irises - 5 deletions)

## 103

Preconditions:
GPU iris database has entries from 1 to 100, inclusive
CPU iris database and graph database is empty
CPU modifications and persisted_state tables are empty
GPU persisted_state table is empty
GPU modifications table has entries for irises which have not been indexed yet.
CPU graph database at layer zero has 100 links
There are zero deletions on S3

Test:
index genesis up to 100

Postconditions:
GPU iris database has entries from 1 to 100, inclusive
CPU iris database has entries from 1 to 100, inclusive
CPU modifications table is empty
CPU persisted_state table shows the max indexed modification is 3 and max indexed iris is 100
CPU graph database matches the output of plaintext genesis
There are zero deletions on S3

## 104

Preconditions:
GPU iris database has entries from 1 to 100, inclusive
CPU iris database and graph database is empty
CPU modifications and persisted_state tables are empty
GPU persisted_state table is empty
GPU modifications table is empty
CPU graph database at layer zero has 100 links
There are zero deletions on S3

Test:
index genesis up to 50
increment the version for iris id 3 in the GPU database and add a reset_update modification for it
increment the version for iris id 5 in the GPU database and add a reauth modification for it
index genesis up to 100

Postconditions:
GPU iris database has 100 entries
CPU iris database has 100 entries
CPU iris database reflects irises updated by new modifications after the first run
CPU persisted_state table shows the max indexed modification is 2 and max indexed iris is 100
CPU graph database matches the output of plaintext genesis
CPU graph database at layer zero has 102 links

## 105

Pre-conditions:
- GPU iris database has entries from 1 to 100, inclusive
- CPU iris database and graph database is empty
- GPU modifications and persisted_state tables are empty
- CPU modifications and persisted_state tables are empty
- CPU graph database at layer zero has 100 links
- S3 deletions: `[]`

Test:
- Update the GPU database with simulated modifications:
    - Id 1: `ResetUpdate` modification for serial id 5, persisted
    - Id 2: `Uniqueness` modification for serial id 15, persisted
    - Id 3: `Reauth` modification for serial id 25, not persisted
    - Id 4: `Uniqueness` modification for serial id 55, not persisted
- Increment versions in GPU database of irises affected by persisted `ResetUpdate` and `Reauth` modifications
- Run genesis up to serial id 50
- Update the GPU database with simulated modifications:
    - Persist modifications with ids 3 and 4
    - Id 5: `ResetUpdate` modification for serial id 60, persisted
    - Id 6: `Reauth` modification for serial id 70, persisted
    - Id 7: `ResetUpdat` modification for serial id 10, persisted
    - Id 8: `Reauth` modification for serial id 20, persisted
    - Id 9: `ResetUpdate` modification for serial id 30, not persisted
- Increment versions in GPU database of irises affected by newly persisted `ResetUpdate` and `Reauth` modifications
- Run genesis up to serial id 100

Post-conditions
- GPU iris database has 100 entries
- CPU iris database has 100 entries
- CPU iris database reflects iris versions updated by modifications
- CPU persisted_state table shows the max indexed modification is 8 and max indexed iris is 100
- CPU graph database matches the output of plaintext genesis
- CPU graph database at layer zero has 103 links


## 106

Pre-conditions:
- GPU iris database has entries from 1 to 100, inclusive
- CPU iris database and graph database is empty
- GPU modifications and persisted_state tables are empty
- CPU modifications and persisted_state tables are empty
- CPU graph database at layer zero has 100 links
- S3 deletions: `[7, 12, 39, 77, 100]`

Test:
- Run genesis process twice, using the same procedure and modifications as in test 105

Post-conditions
- GPU iris database has 100 entries
- CPU iris database has 100 entries
- CPU iris database reflects iris versions updated by modifications
- CPU persisted_state table shows the max indexed modification is 8 and max indexed iris is 100
- CPU graph database matches the output of plaintext genesis
- CPU graph database at layer zero has 98 links
