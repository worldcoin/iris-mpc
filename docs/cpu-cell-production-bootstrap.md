# First production CPU cell

Each party has one rolling Deployment, `ampc-hnsw-b`. Its name stays fixed while
its pod revisions alternate between the logical DNS slots `pod-a` and `pod-b`.
It starts at zero replicas. The dormant template pins the stage-tested binary
and omits the deployment and handoff IDs. Only a running pod's init container
writes its slot record.

1. Before cutover, verify the production release digest, the DNS waiter
   permissions, connectivity to ports 3000 and 3090, and capacity for two
   workers per party. Verify that the release supports the production S3
   importer in candidate mode before the first warm rollout.
2. In the agreed cold-start window, stop the legacy `ampc-hnsw` and active
   Iris anon-stats pods on all three parties. Suspend the Iris exporter and
   scheduled retention CronJobs in the AMPC and SMPCv2 clusters. Confirm that
   the pods have exited and any existing database Jobs have finished, including
   manually started Jobs. Suspending a CronJob only prevents new scheduled runs.
   Keep the legacy release installed at zero replicas because it owns the
   service account used by the new pods.
   Requests can queue while the first managed cohort loads.
3. In one activation PR, pin every production Iris application and Job template
   to its corresponding digest from the same release. Restore the previous
   anon-stats replica counts and CronJob schedules, leaving the graph and
   ingested-request reset jobs suspended. For the rolling Deployment, set one
   replica and `SMPC__STARTUP_MODE=active` on logical A, enable migrations and
   set one fresh `SMPC__CPU_CELL__DEPLOYMENT_ID` across all three parties.
   Use the existing production active probes on port 3000. Leave the handoff
   fields unset. Verify Argo synced the release images on every party, anon-stats
   is ready, and a fresh signup completes across all three parties. After the
   migrations run, use a migration-compatible release for recovery.
4. For the next release, update the same rolling Deployment to candidate mode
   on logical B. Use the port 3090 probes in the dormant template, A's source
   UUID and port 3000 status URLs, and one fresh destination UUID. Follow the
   [warm release steps](cpu-cell-stage-slots.md) using the production values.
