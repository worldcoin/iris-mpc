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
S3 deletions: [1, 2, 3, 4, 5]

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
overwrite the iris code for iris id 3 in the GPU database and add a reset_update modification for it
overwrite the version for iris id 5 in the GPU database and add a reauth modification for it
index genesis up to 100

Postconditions:
GPU iris database has 100 entries
CPU iris database has 100 entries
CPU persisted_state table shows the max indexed modification is 5 and max indexed iris is 100
CPU graph database matches the output of plaintext genesis
CPU graph database at layer zero has 100 links
There are zero deletions on S3