# Testing in staging update 2025-05-02
1. Seeded db using db-init jobs supplying 2M irises(1M identities) from a file with synthetic irises
2. Image used was build from a branch `build-init-db-container` in iris-mpc repo https://github.com/worldcoin/iris-mpc/tree/build-initdb-container
    - Seeding 1M identities takes 30h+
    - Job can fail silently (on the last run 5k of irises was missing in the middle of the collection)
    - Latest commit build an image to just write iris shares to patch a failed previous run
3. Code needed to be adopted to run the same job in each node's vpc and write out only shares for the specific party
   - Luckily the random is not time-based but creates a deterministic sequence from a seed so this was viable
4. Reused role and permissions of the hnsw nodes and reused sns-requests s3 bucket for supplying the data file
5. Finally achieved 1M identities in db and nodes booting and staying stable
6. Snapshoted dbs at 100k,250k,500k,1M identities, snapshots prefixed with `hsnw-`
7. Pushed code used locally to `testing-in-staging` branch in `iris-mpc`

Future work
- Decide if we should invest and create a CICD pipeline to deploy this job, that will require modifications of the binary to be configured with env vars
- For further testing we need to instrument the code correctly, metrics we want:
  - Batch duration
  - Round count
  - Bandwith used
  - DB queries
- Once we have the metrics, we still have 1M identities in the synthetic file, lets create a job that will produce traffic on staging using orb-tools based of that file and observe the metrics on a longer time period
- `orb-tools` repo has a branch `temp/load-plaintext-codes-from-file` which allows to send a request based on data file. 