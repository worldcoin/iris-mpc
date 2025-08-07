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
GPU modifications table has 3 uniqueness modifications, for ids 1-3, which genesis ignores
There are zero deletions on S3

Test:
index genesis up to 100

Postconditions:
GPU iris database has entries from 1 to 100, inclusive
CPU iris database has entries from 1 to 100, inclusive
CPU modifications table is empty
CPU persisted_state table shows the max indexed modification is 0 and max indexed iris is 100
CPU graph database matches the output of plaintext genesis
CPU graph database at layer zero has 100 links
There are zero deletions on S3


## 104
Preconditions:
index genesis up to 50
GPU iris database has entries from 1 to 100, inclusive
CPU iris database and graph database are indexed up to 50
CPU modifications table is empty
CPU persisted state table shows max indexed iris as 50
GPU modifications table has the following modifications:
    ids 51-75 are reset modifications for irises 1-25
    ids 76-100 are reauth modifications for irises 26-50
There are zero deletions on S3

Test:
index genesis up to 100

Postconditions:
GPU iris database has 100 entries
CPU iris database has 100 entries
CPU modifications table 50 entries
CPU persisted_state table shows the max indexed modification is 50 and max indexed iris is 100
CPU graph database matches the output of plaintext genesis
CPU graph database at layer zero has 50 links
There are zero deletions on S3