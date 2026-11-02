# Stage ingestion reset

Run only during the coordinated messaging cutover, after manually confirming that
old traffic is drained and the worker, all three iris matching deployments and
iris statistics publishers have stopped. Do not clear DI tables.

The helper reads the existing modifications-retention CronJob and prints a one-off
Job. It preserves the database secret, schema, pod labels, service account, image
and networking configuration. It does not submit a Job or change any CronJob.

For each party, use its explicit context (never the current-context default):

```sh
PARTY=0
CONTEXT=arn:aws:eks:eu-central-1:024848486749:cluster/ampc-hnsw-0-stage
python3 scripts/stage-messaging-cutover/ingestion-reset.py --party "$PARTY" --mode dry-run > /tmp/iris-ingestion-dry-run.json
kubectl --context "$CONTEXT" create -f /tmp/iris-ingestion-dry-run.json
JOB=$(python3 -c 'import json; print(json.load(open("/tmp/iris-ingestion-dry-run.json"))["metadata"]["name"])')
kubectl --context "$CONTEXT" -n ampc-hnsw wait --for=condition=complete "job/$JOB" --timeout=30m
kubectl --context "$CONTEXT" -n ampc-hnsw logs "job/$JOB"
```

Repeat for parties 1 and 2, using accounts `024848486818` and `024848486770`
and clusters `ampc-hnsw-1-stage` and `ampc-hnsw-2-stage` respectively.

After reviewing all three dry-run counts, repeat the commands with `--mode delete`
and a separate output filename. The rule targets only `ingested_requests`, using
`received_at` with zero retention. Logs report deleted rows. Independently verify
`COUNT(*) = 0` in each matching schema before restarting; future-dated rows are not
eligible under this timestamp predicate. Drain and database checks are manual.

Completed Jobs expire after 24 hours. Before restart, confirm every submitted
cleanup Job has finished and remove the Jobs to prevent accidental reruns:

```sh
kubectl --context "$CONTEXT" -n ampc-hnsw delete jobs -l pop.worldcoin.org/purpose=iris-ingestion-reset
```

No scheduled zero-retention rule is introduced. Existing retention rules and old
messaging infrastructure remain unchanged.
