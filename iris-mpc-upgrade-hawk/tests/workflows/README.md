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

## 107

Pre-conditions:
- Databases are initialized
- S3 deletions: `[]`

Test:
- Run genesis up to serial id 50
- Insert 10 extra irises directly into the CPU database (simulating a dirty CPU state where the CPU database has 60 irises but the S3 checkpoint only covers 50)
- Run genesis up to serial id 100
  - Rollback detects the mismatch between the CPU database and the S3 checkpoint and removes the extra irises before continuing

Post-conditions:
- GPU iris database has 100 entries
- CPU iris database has 100 entries
- CPU iris database contains vector ids matching those in the plaintext genesis CPU database
- CPU modifications table is empty
- CPU `persisted_state` table shows the max indexed modification is 0 and max indexed iris is 100
- CPU graph database matches the output of plaintext genesis

## 108

Pre-conditions:
- Databases are initialized
- S3 deletions: `[]`

Test:
- Run genesis up to serial id 25
- Run genesis up to serial id 50
- Delete the genesis graph checkpoint for party 0 from the CPU store (parties 1 and 2 retain their checkpoint entries pointing to S3)
- Run genesis up to serial id 75
  - Party 0 recovers its graph from the S3 checkpoint held by parties 1 and 2

Post-conditions:
- GPU iris database has 100 entries
- CPU iris database has 75 entries
- CPU iris database contains vector ids matching those in the plaintext genesis CPU database
- CPU modifications table is empty
- CPU `persisted_state` table shows the max indexed modification is 0 and max indexed iris is 75
- CPU graph database matches the output of plaintext genesis

## 109

Pre-conditions:
- Databases are initialized
- S3 deletions: `[]`

Test:
- Run 1: Index genesis up to serial id 25 with checkpoint frequency 10 and pruning mode `OlderNonArchival`
- Run 2: Index genesis up to serial id 50 with checkpoint frequency 50 and pruning mode `None`
- Run 3: Index genesis up to serial id 75 with checkpoint frequency 50 and pruning mode `OlderNonArchival`
- Run 4: Index genesis up to serial id 100 with checkpoint frequency 50 and pruning mode `AllOlder`

Post-conditions:
- GPU iris database has 100 entries
- CPU iris database has 100 entries
- CPU modifications table is empty
- CPU `persisted_state` table shows the max indexed modification is 0 and max indexed iris is 100
- After Run 1: 3 genesis graph checkpoints exist (created at irises 10, 20, 25)
- After Run 2: 4 genesis graph checkpoints exist (no pruning with `None` mode keeps previous checkpoints)
- After Run 3: 3 genesis graph checkpoints exist (pruning mode `OlderNonArchival` removes older non-archival checkpoints)
- After Run 4: 2 genesis graph checkpoint exists (pruning mode `AllOlder` removes all but the latest prior checkpoint)
