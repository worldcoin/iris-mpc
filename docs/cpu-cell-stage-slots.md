# Warm CPU cell releases in stage

Each party uses its existing `ampc-hnsw-b` Kubernetes Deployment for warm
releases. Its name and selector stay fixed. A and B identify **logical slots**:
successive pod revisions alternate between the two fixed peer DNS names. The
legacy `ampc-hnsw` and separate `ampc-hnsw-a` Deployments stay at zero.

## Prepare one release PR

1. Confirm the previous rollout finished on all three parties and its old pods
   have terminated. Check the current logical slot and deployment UUID in the
   live pod template. The next slot must be free.
2. Update the three stage `values-ampc-hnsw-b.yaml` files together. Pin the
   image, flip `PEERS__SLOT` and the DNS init container's `CELL_SLOT`, and use
   that slot for coordination hostnames and metrics names. Keep the fixed A/B
   peer lists. Set the source status URLs (port 3000) and source deployment UUID
   to the current serving release. Generate one fresh destination UUID for the
   new rollout and use it in all three parties, even when reusing an image.
   Omit `SMPC__CPU_CELL__HANDOFF__ID` for candidates supporting the default.
   For an older warm-handoff image, explicitly set the handoff ID to that same
   fresh UUID. Keep the original IDs for any retry of an existing release.
3. Review the rendered pod templates, including inherited values, as a single
   rollout. Any warm pod-template change needs the opposite slot: a same-slot
   candidate would point the serving DNS name at its own pod during startup. Serialize releases through old-pod termination.

To render party 0 with the same values order as Argo:

```sh
party=0
base=deploy/stage
p="$base/ampc-hnsw-$party-stage"
helm template ampc-hnsw-b "$COMMON_APP_CHART" --namespace ampc-hnsw \
  -f "$base/common-values-ampc-hnsw.yaml" -f "$p/values-ampc-hnsw.yaml" \
  -f "$base/common-values-cpu-cell.yaml" -f "$p/values-ampc-hnsw-b.yaml"
```

## What happens after merge

Argo updates one Deployment per party. With one desired replica, one surge and
zero unavailable, Kubernetes creates the next pod alongside its predecessor.
The DNS init container upserts the candidate's own slot record. Capacity must
support both pods during preload. While any candidate waits for capacity or
preload, the source cohort can keep serving.

The existing three-party protocol drains the source, reconciles the candidates
and activates them. The candidate's `/startup` can succeed after preload.
Its `/ready` on port 3090 succeeds only after serving agreement. Each party's
Deployment controller then scales down its old ReplicaSet. The old DNS record
stays in place until that slot's next init container replaces it.

For stage verification, observe both ReplicaSets by pod-template hash. Check
the destination slot and `serving=true` on all three new pods, the disappearance
of old pods, peer DNS resolution and fresh and duplicate signup outcomes.
A pause after sealing requires forward recovery. Do not undo the Deployment
to the previous pod template. If termination is uncertain, fence the old
process before reusing its slot. See [CPU cell ownership and recovery](cpu-cell-ownership.md).

Common-app 2.7.4 sets the Deployment progress deadline to ten minutes. A slow
preload can therefore show `ProgressDeadlineExceeded` before the startup probe
budget expires. The controller continues the rollout when the pod becomes
available. Inspect scheduling, DNS init and candidate status to locate a stall.
