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
2. In the agreed cold-start window, stop all three legacy `ampc-hnsw` consumers
   and confirm their pods have exited. Keep the legacy release installed at
   zero replicas because it owns the service account used by the new pods.
   Requests can queue while the first managed cohort loads.
3. In a separate rollout change, pin the tagged release, set one replica and
   `SMPC__STARTUP_MODE=active` on logical A, enable database migrations and
   set one fresh `SMPC__CPU_CELL__DEPLOYMENT_ID` across all three parties.
   Use the existing production active probes on port 3000. Leave the handoff
   fields unset. Verify all three parties serve and process a fresh signup.
4. For the next release, update the same rolling Deployment to candidate mode
   on logical B. Use the port 3090 probes in the dormant template, A's source
   UUID and port 3000 status URLs, and one fresh destination UUID. Follow the
   [warm release steps](cpu-cell-stage-slots.md) using the production values.
