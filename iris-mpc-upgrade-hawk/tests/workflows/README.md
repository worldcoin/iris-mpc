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
There are zero deletions on S3

## 105
Preconditions:
GPU iris database has entries from 1 to 100, inclusive
CPU iris database and graph database is empty
CPU modifications and persisted_state tables are empty
GPU persisted_state table is empty
GPU modifications table is empty
CPU graph database at layer zero has 100 links
There are zero deletions on S3

Test:
update the GPU database with simulated modifications:
- add reset_update modification id 1, for serial id 5, persisted, and increment the associated iris version
- add uniquness modification id 2, for serial id 15, persisted, and increment the associated iris version
- reauth modification id 3, for serial id 25, not persisted
- uniqueness modification id 4, for serial id 55, not persisted
index genesis up to 50
update the GPU database with simulated modifications:
- mark modification with id 3 as persisted, and increment the associated iris version
- mark modification with id 4 as persisted
- add reset_update modification id 5, for serial id 60, persisted, and increment the associated iris version
- add reauth modification id 6, for serial id 70, persisted, and increment the associated iris version
- add reset_update modification id 7, for serial id 10, persisted, and increment the associated iris version
- add reauth modification id 8, for serial id 20, persisted and increment the associated iris version
- add reset_update modification id 9, for serial id 30, not persisted
index genesis up to 100

Postconditions:
GPU iris database has 100 entries
CPU iris database has 100 entries
CPU iris database reflects irises updated by new and persisted modifications after the first run
CPU persisted_state table shows the max indexed modification is 8 and max indexed iris is 100
CPU graph database matches the output of plaintext genesis
CPU graph database at layer zero has 103 links
There are zero deletions on S3
