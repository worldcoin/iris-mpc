# Startup Peer-Visibility Deadlock: Frozen verified_peers Bug

## Summary

When a single party (e.g., party 2) crashes and restarts while the other two parties have already passed the startup peer-visibility gates, the restarting party generates a new UUID that the other two parties will never acknowledge. This is because `verified_peers` — the state that records which peer UUIDs each party has observed — is **populated once during startup and then frozen forever**, exposed immutably via the `/health` endpoint. The restarting party times out after 900s, crashes, restarts with yet another UUID, and repeats forever.

**2026-07-22 incident: Party 2 crashed → all three parties stuck for 145+ minutes.**

---

## Root Cause: verified_peers is Write-Once, Expose-Forever

### The Design
- Each process maintains an `Arc<Mutex<HashSet<String>>> verified_peers` — a set of peer UUIDs it has observed
- The `/health` HTTP endpoint exposes this set to other peers: `{ uuid: "my-uuid", verified_peers: {peer1, peer2, peer3} }`
- Both `wait_for_others_unready()` and `wait_until_startup_visibility_is_complete()` query `/health` and update `verified_peers`

### The Flaw
Once a party exits both startup-sync functions and enters the iris/graph load phase, it **never calls those functions again**. Its `verified_peers` HashSet is frozen and exposed to the world.

If another party restarts with a new UUID, the frozen parties will never know about it — there's no mechanism to re-update `verified_peers` after the startup gates pass.

---

## Detailed Sequence

### Phase 1: Healthy Startup (2026-07-21 22:33)

All three parties start fresh and execute startup synchronization:

**Party 0 (UUID: `1baece17`):**
1. `wait_for_others_unready()`: queries `/health` → inserts `{1baece17, ecce5a72, 724aca6f}` into `verified_peers`
2. `wait_until_startup_visibility_is_complete()`: queries `/health` again → validates all peers report all UUIDs → passes
3. Enters iris/graph load phase
4. **`verified_peers` is now frozen at `{1baece17, ecce5a72, 724aca6f}`**

**Party 1 & 2:** Similar execution, same final `verified_peers` set.

**What other parties observe via `/health` endpoint:**
- Party 0's response: `{ uuid: "1baece17", verified_peers: {1baece17, ecce5a72, 724aca6f} }`
- Party 1's response: `{ uuid: "ecce5a72", verified_peers: {ecce5a72, 1baece17, 724aca6f} }`
- Party 2's response: `{ uuid: "724aca6f", verified_peers: {724aca6f, 1baece17, ecce5a72} }`

### Phase 2: Party 2 Crashes (~22:35 – 00:50 UTC)

Parties 0 and 1 are in the iris/graph load phase (never exit). Party 2 dies.

### Phase 3: Party 2 Restarts (00:35 UTC, attempt 1 of 9)

**Party 2 restarts with new UUID: `a7c85a4e`**

1. `wait_for_others_unready()`: queries `/health`, inserts `{a7c85a4e, 1baece17, ecce5a72}` into its own `verified_peers`
2. `wait_until_startup_visibility_is_complete()` enters its retry loop:
   
   **Iteration N:**
   - Queries Party 0's `/health`:
     ```
     Response: { uuid: "1baece17", verified_peers: {1baece17, ecce5a72, 724aca6f} }
     ```
   - Queries Party 1's `/health`:
     ```
     Response: { uuid: "ecce5a72", verified_peers: {ecce5a72, 1baece17, 724aca6f} }
     ```
   - Builds expected set: `{a7c85a4e, 1baece17, ecce5a72}` (my UUID + peers' UUIDs)
   - Checks Party 0: missing from its verified_peers: `{a7c85a4e}` ❌
   - Checks Party 1: missing from its verified_peers: `{a7c85a4e}` ❌
   - **Both peers are missing my new UUID**
   - Sleep 1s, retry

3. **Loop continues for 900s:**
   - Parties 0 and 1 are stuck in iris/graph load, never re-enter the sync functions
   - `/health` responses remain frozen: `verified_peers: {..., 724aca6f}` (old UUID)
   - Party 2 never sees its own UUID in either peer's verified set
   - After 900s timeout → log error, exit(1)

4. **Restart cycle repeats** (9 times total during incident)
   - Each restart generates a fresh UUID (e.g., `f1a2b3c4`, `d5e6f7a8`, etc.)
   - Each times out at exactly 900s
   - Meanwhile, parties 0 and 1 stay `READY=0`, waiting for party 2 downstream

### Phase 4: Manual Intervention — Simultaneous Restart

**User runs:** 
```bash
kubectl rollout restart deploy/ampc-hnsw on all three clusters
```

All three processes die and restart together:

1. Fresh processes, all generate new UUIDs (unrelated to previous ones)
2. All three simultaneously enter `wait_for_others_unready()`
3. All three simultaneously populate `verified_peers` with the new UUIDs
4. All three enter `wait_until_startup_visibility_is_complete()`
5. All three see each other's new UUIDs in `/health` responses → all pass within 1s
6. All three log: `"All nodes have full peer visibility during startup"` ✓
7. All three proceed to iris/graph load and eventually become `READY=1/1`

---

## Why This Design Fails on Staggered Restarts

### Assumption
The design assumes **"all peers restart together or none restart during startup."**
- If all restart together → all enter visibility loop → all see each other's fresh UUIDs → all pass
- If none restart → visibility is already established → no new UUIDs to observe

### Reality
**A single unexpected crash breaks the assumption:**
- One peer crashes (crash reason unknown; could be OOMKill, node eviction, transient error)
- Other peers are already past the visibility gates (their `verified_peers` is frozen)
- Crashed peer restarts with new UUID
- No mechanism exists to notify already-past peers or to re-update their frozen sets
- Cascade: all three parties become unavailable

---

## Implementation Locations

- **Peer-visibility sync:** `ampc-server-utils/src/server_coordination.rs:645–720`
  - `wait_until_startup_visibility_is_complete()`
  - `fn current_startup_uuid_set()`
  - `fn missing_startup_visibility()`

- **Server startup entry:** `iris-mpc/src/server/mod.rs:~125–130`
  - Calls `wait_for_others_unready()` then `wait_until_startup_visibility_is_complete()`

- **Health endpoint:** `ampc-server-utils/src/server_coordination.rs:57–275`
  - `/health` and `/ready` handlers that expose `verified_peers`

---

## Original Implementation

- **Author:** Wojciech Sromek
- **Commit:** `1b5800147` (2026-05-26)
- **PR:** #2152 (POP-3883: Prevent node 0 locking after becoming unready)
- **Original code:** iris-mpc/src/server/mod.rs

- **Migration to ampc-common:** Stuart Woodbury + Bryan Gillespie
- **Commit:** `bf96c3e8` (2026-06-17)
- **PR:** #106 in ampc-common

---

## Incident Timeline

| Time | Event |
|------|-------|
| 22:33 UTC (2026-07-21) | All three parties start, pass visibility gates, enter iris/graph load |
| ~22:35–00:50 UTC | Party 2 crashes (root cause unknown) |
| 00:35–00:50 UTC | Party 2 crash-loops 9 times, timing out at 900s each; parties 0 & 1 stuck at `READY=0` |
| 01:09 UTC | Manual `rollout restart` on all three parties simultaneously |
| 01:09+ UTC | All three restart, pass visibility with fresh UUIDs, load iris/graph, become `READY=1/1` |
| **Total outage:** 145+ minutes |
| **Data loss:** None (graph checkpoint consensus prevents divergence) |
| **Service impact:** Complete cluster unavailability |

---

## Possible Fixes

### Short-term (Mitigate)
1. **Reduce timeout:** Lower `startup_sync_timeout_secs` from 900s to something like 120s so deadlock resolves faster and alerts trigger sooner
2. **Better observability:** Add alerting for "any party in startup loop for >600s" → auto-restart all three

### Medium-term (Recover from Staggered Restarts)
1. **Re-enterable visibility loop:** Allow parties that are past the gates to re-enter if a peer restarts
2. **Liveness-based reset:** If a peer's UUID disappears from `/health` responses, clear it from `verified_peers` and allow re-checking
3. **Deterministic UUIDs:** Generate startup UUIDs deterministically (e.g., hash of `party_id + current_epoch`) so restarting produces the same UUID, avoiding the freshness problem

### Long-term (Architectural)
1. **Decouple liveness from visibility:** The startup sync mixes "I'm alive and reachable" with "I've observed all peers." Separate these concerns.
2. **Ephemeral visibility:** Make visibility state per-process-lifetime and don't expose it via HTTP; instead, have peers actively keep each other informed when they observe new UUIDs
3. **Three-phase commit:** Require an explicit "all peers agree on the startup set" consensus before any peer can proceed (similar to what the graph checkpoint protocol does with barriers)


---

## Architectural Flaw: Static Initialization vs. Dynamic State Machine

### Current Design: "Run Once, Freeze Forever"

The startup-visibility protocol is implemented as a **one-time sequential ritual**:

```
Startup → wait_for_others_unready() → wait_until_startup_visibility_is_complete() → iris/graph load → READY
                                    ↓
                            verified_peers is frozen
                            `/health` reports stale UUIDs forever
```

Once both functions complete, the peer is committed to a **static snapshot** of the startup set. The state machine has no way to recover if that snapshot becomes invalid (a peer dies and restarts with a new UUID).

### The Problem: Three Asynchronous Peers, One Frozen State Machine

- **Party 0** completes gates at T=10s, freezes at `{1baece17, ecce5a72, 724aca6f}`
- **Party 1** completes gates at T=12s, freezes at `{ecce5a72, 1baece17, 724aca6f}`
- **Party 2** crashes at T=150s, restarts at T=200s with new UUID `a7c85a4e`
- **Party 2** tries to validate against parties 0 & 1's frozen state → deadlock

All three are running the "same" state machine, but they're executing asynchronously. Once any peer advances past the gates, it's no longer participating in the protocol, so it can't respond to new peers joining late.

### Proposed: Continuous State Machine

Each peer should run a **state machine that never exits**:

```
loop:
  Poll /health of all peers
  Collect UUIDs and verified_peers sets
  Validate: "do all peers report all UUIDs?"
    ✓ Yes  → state = READY (but keep polling)
    ✗ No   → state = NOT_READY (reset, re-validate)
  If any peer's UUID changes → reset state machine
  If any peer disappears     → reset state machine
  Sleep, retry
```

Benefits:
1. **Resilient to staggered restarts:** A late-joining peer with a new UUID is detected, state resets, and consensus is re-established
2. **Self-healing:** If a peer briefly disappears, the state machine automatically tries to recover rather than deadlocking
3. **No frozen snapshots:** `verified_peers` is always current, never exposing stale data via `/health`
4. **Decoupled from iris/graph load:** Startup visibility doesn't need to complete before heavy loading can begin; it's actively maintained in the background

### Why the Graph Checkpoint Protocol Avoids This

The graph checkpoint protocol (in `iris-mpc-cpu/src/checkpoint_protocol/`) uses:
- **Retries:** transient failures trigger re-attempts, not fatal exits
- **Barriers:** explicit synchronization points where all peers must arrive before anyone proceeds
- **Active consensus:** parties continue communicating until agreement is reached

The peer-visibility protocol could learn from this: make it a continuous invariant, not a one-time gate.

