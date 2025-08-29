# Test cases

In all genesis e2e tests, databases are initialized as follows:
- GPU iris database has entries from 1 to 100, inclusive
- GPU graph database is empty
- CPU iris database and graph database is empty
- GPU modifications and persisted_state tables are empty
- CPU modifications and persisted_state tables are empty

## 100

Preconditions:
- Databases are initialized
- S3 deletions: `[]`

Test:
- Index genesis up to 100

Postconditions:
- GPU iris database has 100 entries
- CPU iris database has 100 entries
- CPU modifications table is empty
- CPU persisted_state table shows the max indexed modification is 0 and max indexed iris is 100
- CPU graph database matches the output of plaintext genesis

## 101

Preconditions:
- Same as in test 100

Test:
- Index genesis up to 50
- Index genesis up to 100

Postconditions:
- Same as in test 100

## 102

Preconditions:
- Databases are initialized
- S3 deletions: `[1, 10, 20, 50, 100]`

Test:
- Index genesis up to 100

Postconditions:
- GPU iris database has 100 entries
- CPU iris database has 100 entries
- CPU modifications table is empty
- CPU `persisted_state` table shows the max indexed modification is 0 and max indexed iris is 100
- CPU graph database matches the output of plaintext genesis
- CPU graph database at layer zero has 95 nodes (100 irises - 5 deletions)

## 103

Preconditions:
- Databases are initialized
- S3 deletions: `[]`

Test:
- Update the GPU database with simulated modifications:
    - Id 1: `ResetUpdate` modification for serial id 5, persisted
    - Id 2: `Reauth` modification for serial id 15, persisted
    - Id 3: `Uniqueness` modification for serial id 25, persisted
- Index genesis up to 100

Postconditions:
- GPU iris database has 100 entries
- CPU iris database has 100 entries
- CPU modifications table is empty
- CPU `persisted_state` table shows the max indexed modification is 2 and max indexed iris is 100
- CPU graph database matches the output of plaintext genesis
- CPU graph database at layer zero has 100 nodes

## 104

Preconditions:
- Databases are initialized
- S3 deletions: `[]`

Test:
- Index genesis up to 50
- Update the GPU database with simulated modifications:
    - Id 1: `ResetUpdate` modification for serial id 5, persisted
    - Id 2: `Reauth` modification for serial id 15, persisted
- Increment versions in GPU database of irises affected by newly persisted `ResetUpdate` and `Reauth` modifications
- Index genesis up to 100

Postconditions:
- GPU iris database has 100 entries
- CPU iris database has 100 entries
- CPU iris database contains vector ids matching those in the plaintext genesis CPU database
- CPU `persisted_state` table shows the max indexed modification is 2 and max indexed iris is 100
- CPU graph database matches the output of plaintext genesis
- CPU graph database at layer zero has 102 nodes

## 105

Pre-conditions:
- Databases are initialized
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
- CPU iris database contains vector ids matching those in the plaintext genesis CPU database
- CPU `persisted_state` table shows the max indexed modification is 8 and max indexed iris is 100
- CPU graph database matches the output of plaintext genesis
- CPU graph database at layer zero has 103 nodes


## 106

Pre-conditions:
- Databases are initialized
- S3 deletions: `[7, 12, 39, 77, 100]`

Test:
- Run genesis process twice, using the same procedure and modifications as in test 105

Post-conditions
- GPU iris database has 100 entries
- CPU iris database has 100 entries
- CPU iris database contains vector ids matching those in the plaintext genesis CPU database
- CPU `persisted_state` table shows the max indexed modification is 8 and max indexed iris is 100
- CPU graph database matches the output of plaintext genesis
- CPU graph database at layer zero has 98 nodes
