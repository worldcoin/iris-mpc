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
GPU modifications and persisted_state tables are empty
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
GPU modifications and persisted_state tables are empty
S3 deletions: [101, 102, 103, 104, 105]

Test:
index genesis up to 100

Postconditions:
GPU iris database has entries from 1 to 100, inclusive
CPU iris database has entries from 1 to 100, inclusive
CPU modifications table is empty
CPU persisted_state table shows the max indexed modification is 0 and max indexed iris is 100
CPU graph database matches the output of plaintext genesis
CPU graph database at layer zero has 100 links because none of the deletions apply

