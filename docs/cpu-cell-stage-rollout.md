# Stage A/B rollout preparation

This is the rollout sequence for POP-4377. The checked-in slots remain disabled
and passive. It is not an activation manifest. Use the existing
`ampc-hnsw-a` and `ampc-hnsw-b` Argo applications on all three stage parties;
do not create another application or intent-writing Job.

## Prepare while activation is under review

1. Validate all six slots against the actual `worldcoin-foundation/common-app`
   chart used by Argo. Version 2.7.4 is the reference for these slots:

   ```sh
   COMMON_APP_CHART=/path/to/common-app bun scripts/tests/test-cpu-cell-stage.ts
   ```

   This checks inert defaults, selectors, inherited database/queue/TLS/security
   group configuration, probes and DNS failure handling. Docker cases use the
   actual DNS init image with networking disabled and AWS stubbed.
2. Finish serving activation review and the next-destination routing fix below.
   Select an approved image by source commit and immutable registry digest.
   The linear-scan image workflow builds main/release events, not arbitrary PR
   pushes. Confirm the intended build completed; a green Argo sync alone does
   not establish which binary is running. Production release tags are separate.
3. Snapshot each party's parent/child Argo revisions and sync policies. Establish
   which GitOps revision will own the test. Parent reconciliation or a revision
   reset can otherwise replace a manually pinned child. Restore only the
   policies/revisions recorded for that party.
4. Clear the infrastructure gates: healthy non-terminating CPU nodes and
   provisioning controller, room for one extra CPU pod/node per party, account
   quota/subnet capacity, and measured database headroom. Keep A serving if B
   cannot schedule. Do not reserve hardware or change instance types implicitly.
5. Verify certificate SANs for both slot names, actual peer TLS, security-group
   and routing reachability for serving/MPC ports and candidate status port
   3090. Handoff HTTP status is protected by the existing peer network boundary;
   UUID equality is not authentication. The reused Pod Identity role also needs
   `route53:GetChange` for the DNS waiter. A local stub does not verify IAM.

Route53 INSYNC confirms the authoritative CREATE change completed. Check actual
resolution to each intended Pod IP separately, including from peer pods.
Retirement and exact-record removal follow [the slot procedure](cpu-cell-stage-slots.md#retire-or-reuse-a-slot).
Never replace CREATE with UPSERT to bypass a stale owner.

## Establish the first compatible source

Protocol 2 drops the original candidate-process column. Candidate startup runs
migrations before checking source compatibility; `DATABASE__MIGRATE=false` in
legacy values does not prevent that startup path. Do not launch a new candidate
beside an incompatible source.

Use one coordinated cold baseline with no handoff in flight:

1. Stop and fence the legacy `ampc-hnsw` writers across all parties through their
   owning GitOps configuration. Check observed zero replicas and absence of old
   pods/init containers; uncertain node/process termination blocks progress.
2. Retire stale A-slot DNS records using the existing exact-record procedure.
3. Configure slot A as Active with the compatible image and one fresh deployment
   UUID shared across the three parties. Set A's `CANDIDATE_STATUS_URLS` to the
   future B operator endpoints now, before A starts serving. Keep legacy
   `ampc-hnsw` and slot B at
   zero replicas. Use `Recreate` with one process per slot.
4. Use the normal startup/migration path. Verify local migration history,
   durable C, startup agreement, readiness and a signup-service baseline before
   enabling any handoff. Do not assume a deleting node will remain available
   when its last serving pod stops.

An Active A slot must override the passive exec probes: remove their `exec`
handlers and restore the common serving HTTP probes on health port 3000.
The passive 3090 listener is not started by ordinary Active startup. Keep the
slot-specific DNS initializer, named service account, selectors and TLS mounts.

## Deliver the handoff through GitOps

Use release-specific final values in the existing applications' ordered Helm
values lists. Review the complete three-party diff before syncing. Release
inputs are the immutable image, source/destination deployment UUIDs and a fresh
handoff UUID. Deployment UUIDs identify releases, not reusable `a`/`b` labels.
The source settings below must already be installed in the baseline. Starting
the warm handoff updates destination B only; adding settings to serving A at
that point would restart it. Include the future next-destination configuration
from POP-4560 in B's original release as well.

| Setting | Serving source A | New destination B |
| --- | --- | --- |
| `replicaCount` | 1 | 1 |
| `SMPC__STARTUP_MODE` | `active` | `candidate` |
| `SMPC__CPU_CELL__DEPLOYMENT_ID` | source release UUID | destination release UUID |
| `SMPC__CPU_CELL__CANDIDATE_STATUS_URLS` | B operator URLs | B operator URLs |
| `SMPC__CPU_CELL__HANDOFF__ID` | absent | fresh handoff UUID |
| `SMPC__CPU_CELL__HANDOFF__SOURCE_DEPLOYMENT_ID` | absent | source release UUID |
| `SMPC__CPU_CELL__HANDOFF__SOURCE_STATUS_URLS` | absent | A serving URLs |

For party indices 0, 1, 2, serving URLs are
`http://pod-a.{party}.stage.hnsw.worldcoin.dev:3000/handoff-status` and B
operator URLs are
`http://pod-b.{party}.stage.hnsw.worldcoin.dev:3090/handoff-status`.
Use the slot-specific serving hostnames on B for its own future serving cohort.
The local bind hostname may be `0.0.0.0`; peer URLs must use resolvable names.

Keep B's candidate probes throughout normal promotion: its status listener
remains alive and readiness follows serving activation. The configured intent
makes the runtime record the request, wait for all candidates, drain A,
reconcile B and promote automatically. There is no second promotion merge,
operator SQL insertion or separate vote. Passive-only preload measurements are
a separate preparatory test, not an extra step in that normal release flow.

## Prove both directions and recovery

The current promotion code retains B operator URLs for activation and later
source polling. Consequently B cannot yet discover the next A candidate.
Before claiming repeatability, add explicit next-destination endpoints for
subsequent source coordination while preserving B's current activation peers.
Verify two consecutive handoffs in the existing three-party fixture. DNS
aliases or replacing the current activation peer list do not solve this safely.

After that fix, repeat the release with roles swapped and fresh release/handoff
UUIDs. Retire the old A pods/init containers and exact DNS records before reusing
the A slot. Do not restart the serving B merely to update future candidate URLs.

The stage acceptance run must record:

- Same offered signup-service load before/during preload, outcomes, latency,
  database CPU/I/O/connections and queue age. An unschedulable B leaves A serving.
- All-party completion C/publication agreement, reconciled registry metadata,
  exact current serving cohort and first B result through signup-service.
- Request/result accounting across A→B and B→A, including pending work. Different
  retained completed-journal histories are permitted; keep journal cleanup off.
- Missing parties, failed preload/sync, interrupted drain/activation and process
  replacement. Failure after sealing never automatically resumes A.
- The coordinated cold-recovery procedure, recovery duration, and slot reuse.

POP-4378/4380 cover executable recovery and failure cases; POP-4379 covers full
request/result correctness. Component tests do not replace this stage evidence.
Keep production deferred to the approved tagged release after the A/B work.
