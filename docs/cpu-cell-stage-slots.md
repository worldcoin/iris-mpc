# Passive CPU cells in stage

The existing `ampc-hnsw` deployment keeps serving. `ampc-hnsw-a` and
`ampc-hnsw-b` are separate, initially disabled candidate deployments in the same
namespace. They share the party's existing database, queues, `application`
secret, pod security group and `ampc-hnsw` service account. Neither slot serves
requests until the later drain/reconciliation/activation work is implemented.
Do not start another active process alongside the legacy serving deployment.

## Render a slot

Use common-app **2.7.4 or later**, which supports reusing a named service account.
The cluster-apps bootstrap defines the same ordered values list:

```sh
party=0
slot=b
helm template "ampc-hnsw-$slot" "$COMMON_APP_CHART" --namespace ampc-hnsw \
  -f deploy/stage/common-values-ampc-hnsw.yaml \
  -f "deploy/stage/ampc-hnsw-$party-stage/values-ampc-hnsw.yaml" \
  -f deploy/stage/common-values-cpu-cell.yaml \
  -f "deploy/stage/ampc-hnsw-$party-stage/values-ampc-hnsw-$slot.yaml"
```

Slots default to zero replicas. Enable only the candidate being measured, in
the configuration Argo reconciles, after confirming spare capacity. Each
candidate inherits the serving workload's CPU/memory requests and placement;
the stage pool must accommodate both. The pinned candidate image includes
passive startup; the older serving release does not.

Each slot has distinct selectors and `pod-a.{party}.stage.hnsw.worldcoin.dev`
or `pod-b.{party}.stage.hnsw.worldcoin.dev` peer names. The existing party
certificates must cover those names. The DNS init container waits for Route53
INSYNC before preload and uses CREATE rather than UPSERT to reject collisions.

## Observe preload

Candidate status binds only to loopback. Exec probes use the image's curl:
`/health` is liveness, `/startup` confirms preload, and `/ready` always fails.
No Service is created. A candidate therefore remains Kubernetes NotReady even
after successful preload; do not use Deployment Available or Argo Healthy as
the acceptance signal. Inspect the exact candidate pod/container:

```sh
kubectl --context "$CONTEXT" -n ampc-hnsw exec "$CANDIDATE_POD" \
  -c "ampc-hnsw-$slot" -- curl --fail --silent --max-time 5 \
  http://127.0.0.1:3090/candidate-status
```

Require `state=loaded`, `serving=false`, the expected party and nonzero loaded
rows for all three parties. Record each process UUID, image, row count and
preload duration. Check that none of the legacy serving pods restarted.

Measure the same offered signup-service request rate before and during B
preload. Record terminal outcomes and latency, serving-cell latency/error
metrics, Postgres CPU/connections/read I/O, and B preload duration. Keep baseline
and preload windows separate; a successful preload alone is not evidence of
acceptable serving or database impact. No fixture reset or queue replacement
is needed.

The passive process does not bind peer sockets. Validate the certificate SANs
and mutual TLS on the slot DNS/peer ports with an isolated test listener before
claiming the networking is ready for activation.

## Retire or reuse a slot

1. Commit zero replicas to the source Argo reconciles and wait for that exact
   revision to sync. Do not rely on `kubectl scale`: self-healing can recreate it.
2. Confirm the Deployment has observed zero replicas and no slot Pods remain,
   including terminating Pods and DNS init containers. Unreachable nodes or
   uncertain termination block DNS removal/reuse.
3. Read the exact slot A record and delete that record with its observed IP/TTL.
   Never delete another slot's name or the serving `node.{party}` records.
4. Before enabling the slot again, confirm its name is absent and keep the old
   revision disabled. An old init container cannot UPSERT over a new owner.

A pod/init restart can leave a record behind. CREATE deliberately fails closed
until the same retirement checks and exact-record deletion are complete.
Rollback for this passive stage test is disabling the candidate and retiring
its records; the existing serving deployment remains in place. These files do
not implement promotion or production rollout.
