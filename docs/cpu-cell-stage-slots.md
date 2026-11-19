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
certificates must cover those names. The DNS init container completes Route53
registration before preload. Initially it creates the address only if the name
is free. Prepared releases also register the deployment UUID as described below.
Resolver propagation is verified separately before activation;
the passive loader does not use the peer network.

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

Once both slots have used candidate manifests and the previous slot has been
retired as described below, prepare the next stage release from the repository
root with Git, Bash, jq, yq v4 and uuidgen:

```sh
scripts/prepare-cpu-cell-stage-release.sh "$PINNED_STAGE_IMAGE"
git diff -- deploy/stage
```

Pass the published `ghcr.io/worldcoin/iris-mpc-linear-scan-amd64:<commit>@sha256:<digest>`
reference. The script reads the six slot manifests, uses the last handoff's
source to identify the slot to reuse, and writes one shared deployment ID and
handoff ID into its three manifests. Repeating the command for an already
prepared rollout preserves both IDs. Review the diff in the deployment PR.
The script reads the values in the Git working tree. Before enabling a slot
whose desired replica count is zero, verify that it has actually retired.

1. Commit zero replicas to the source Argo reconciles and wait for that exact
   revision to sync. Do not rely on `kubectl scale`: self-healing can recreate it.
2. Confirm the Deployment has observed zero replicas and no slot Pods remain,
   including terminating Pods and DNS init containers. Unreachable nodes or
   uncertain termination block DNS removal/reuse.
3. For a slot last deployed without `DNS_DEPLOYMENT_ID`, remove its legacy
   A record using the observed IP and TTL, and confirm the name is free. The
   first prepared release creates the A record and a TXT record containing its
   deployment UUID together. An existing unmarked address blocks registration.
4. For subsequent releases, leave both records in place. The init container
   replaces the address and owner in one Route53 transaction that requires the
   previous deployment's exact TXT record. A retry succeeds if both the address
   and the new owner already match. A missing or unexpected owner, or a restart
   of the same deployment at a different IP, requires investigation.

## Coordinated handoff prerequisites

Before enabling handoff, verify healthy CPU capacity for both cells, peer
DNS/TLS/network access, database headroom, and `route53:GetChange` on each
stage workload role. Route53 INSYNC does not prove peer DNS resolution or TLS.

Establish protocol 2 with a coordinated cold upgrade and no handoff in flight.
Candidate startup migrates before compatibility checks; `MIGRATE=false` does
not protect an incompatible serving release. Stop/fence legacy writers first.

During that baseline, configure Active A with one shared deployment UUID, fixed
`CPU_CELL__PEERS__A` and `CPU_CELL__PEERS__B` candidate-status URL lists, and
`CPU_CELL__PEERS__SLOT=a` (all prefixed with `SMPC__`). Restore A's serving HTTP
probes on port 3000 instead of the passive exec probes on 3090. The later warm
release changes B only: pin its
image digest, use one fresh B deployment UUID and one fresh handoff UUID shared
across all three parties, and supply A's status URLs
(`http://<host>:<healthcheck-port>/handoff-status`). Keep the same A/B URL lists
and set `CPU_CELL__PEERS__SLOT=b`.
Keep B's candidate probes through promotion.

The fixed peer lists also route B→A without restarting serving B. Retire old
processes and follow the DNS ownership procedure above before slot reuse. Validate
both directions through signup-service and exercise cold recovery under
POP-4378/4379/4380. After sealing, failures must not automatically resume A.
Keep journal cleanup and production rollout separate.
