# Possible Fix: Restart-Tolerant Startup Sync (pre-ready only)

**Goal:** allow a party to crash and re-join the startup/sync sequence *before* the fleet
crosses the ready barrier, while keeping today's "a peer restart forces a full-fleet
restart" policy *after* it. The re-join must not reload iris codes or the graph — only the
tracking of peer UUIDs (`/health` `verified_peers`) and the ready check are redone.

**Scope of analysis:** `iris-mpc/src/server/mod.rs::server_main` — the `server_main` that
`iris-mpc-hawk` reaches via `iris-mpc-bins/bin/iris-mpc/server/iris_mpc_hawk.rs`. The
coordination primitives it calls live in the pinned `ampc-server-utils`
(`ampc-common` rev `9a3d6a4`, `src/server_coordination.rs`).

Companion document: `startup_sync_bug.md` (incident narrative, 2026-07-22).

---

## 1. Current startup sequence and its monitoring regimes

`server_main` has three phases, and they differ in how peers are watched:

| Phase | Code | Peer monitoring |
|---|---|---|
| A — gates | `mod.rs:139-146`: `start_coordination_server_with_extra_routes`, `wait_for_others_unready`, `wait_until_startup_visibility_is_complete` | live `/health` polling; `verified_peers` written |
| **B — load** | `mod.rs:148-284`: `get_sync_result`, `sync_modifications`, `sync_graph_mutations`, SQS/claim recovery, `init_hawk_actor`, `start_results_thread` | **none.** `verified_peers` is frozen and still served from `/health` |
| C — barrier | `mod.rs:286-295`: `init_heartbeat_task`, `set_node_ready`, `wait_for_others_ready` | heartbeat live; peer UUID mismatch ⇒ graceful shutdown |

Phase B is the long one (iris + graph load — minutes) and is the only phase with no peer
monitoring at all.

### The failure

Nothing writes `verified_peers` after `mod.rs:146`. A peer that dies during phase B reboots
with a fresh UUID that the two survivors can never acknowledge, because they are not
running any code that would insert it. Consequences:

- the restarted peer loops in `wait_until_startup_visibility_is_complete`
  (`server_coordination.rs:727`) until `startup_sync_timeout_secs` (900s), exits, reboots
  with yet another UUID, and repeats;
- the survivors reach `wait_for_others_ready` (`server_coordination.rs:584`), whose outer
  loop has **no deadline**, and park there forever. This is the "hangs forever". It is
  doing its job — never enter MPC without all three parties — it simply can never be
  satisfied.

Both documented self-healing paths misfire specifically in phase B. The doc comment on
`wait_for_others_unready` (`server_coordination.rs:283-302`) relies on either
(a) a peer still in its own startup recording our UUID on its next poll, or (b) a
fully-ready peer self-healing via the heartbeat's UUID-mismatch check. In phase B we are
neither still-polling nor ready, and we have no heartbeat, so neither path fires.

---

## 2. Why the pre-ready / post-ready split is sound

The invariant is already documented in this file, at `mod.rs:194-199`:

> SAFETY DEPENDENCY: peers serve their SyncState from a snapshot taken at THEIR boot, so an
> exchanged frontier can under-report persists made after that peer booted. This is sound
> today only because the startup unready-gate + heartbeat teardown force full-fleet
> restarts (every frontier is rebuilt after its party's last commit). If coordination is
> ever relaxed to allow solo restarts, this exchange must move to a live DB read in the
> sync endpoint.

Read carefully, this justifies exactly the split we want:

- **Pre-ready:** no party has ever run a batch, so there are no persists after any peer's
  boot snapshot. An exchanged frontier cannot be stale, and a solo restart invalidates
  nothing another party is holding. A re-joining party re-derives its own state from its
  own boot anyway. Relaxing coordination *here* does not trip the warning above.
- **Post-ready:** batches can commit, so a peer's snapshot can under-report and a solo
  re-join is unsafe. Full-fleet restart must remain the policy.

The ready barrier is therefore the correct place to draw the line, and the pre-ready
re-join genuinely needs to redo only the UUID tracking and the ready check — not the
sync-state exchange, not the loads. (The claim-release / skip-ahead work in phase B is
idempotent and is re-run by the restarting party itself.)

---

## 3. Proposed fix

### 3.1 A startup visibility maintainer that owns `verified_peers` until the fleet is ready

Spawn one background task immediately after `start_coordination_server_with_extra_routes`
(`mod.rs:139`), holding the returned `Arc<Mutex<HashSet<String>>>` and the
`ShutdownHandler`. It polls each peer's `/health` on a short fixed interval and:

1. **inserts** each peer's current UUID into `verified_peers`. `verified_peers` therefore
   never freezes during phase B, so a peer restarting mid-load is verified within one poll
   interval and its `wait_until_startup_visibility_is_complete` passes on its next
   iteration. *This single change removes the crash loop.*
2. **prunes** UUIDs that no peer advertises any more, so we stop publishing dead
   incarnations. Without pruning, `missing_startup_visibility`
   (`server_coordination.rs:821`) can be satisfied by ghost UUIDs.
3. records the last-seen UUID per party and bumps a `peer_epoch: AtomicU64` on any change,
   logging the transition.
4. after the fleet is ready, **seals**: it stops mutating `verified_peers`, and a UUID
   change now calls `shutdown_handler.trigger_manual_shutdown()` — restoring today's
   "restart the fleet" policy.

Implementation notes:

- Poll per host, built from `get_check_addresses(&config.node_hostnames,
  &config.healthcheck_ports, "health")`, rather than calling
  `try_get_endpoint_other_nodes`. Two reasons: that helper discards the party index (it
  sorts by index then maps it away, `server_coordination.rs:617`), and its per-node retry
  loop is unbounded inside a `startup_sync_timeout_secs` budget — wrong shape for a
  steady-state poller. The maintainer wants "best effort snapshot each tick, never fail".
- An unreachable peer is *not* an error pre-ready: record it as absent, prune its UUID,
  keep polling. Tolerating the gap is the point.
- Suggested interval: a local `const STARTUP_VISIBILITY_POLL: Duration =
  Duration::from_secs(2);`. No config change needed.

Suggested surface:

```rust
pub struct StartupVisibility { /* verified_peers, peer_uuids, epoch, sealed */ }

impl StartupVisibility {
    /// Spawns the poller on `task_monitor`; takes ownership of `verified_peers`.
    pub fn spawn(
        config: &ServerCoordinationConfig,
        task_monitor: &mut TaskMonitor,
        shutdown_handler: &Arc<ShutdownHandler>,
        verified_peers: Arc<Mutex<HashSet<String>>>,
    ) -> Self;

    /// Party id -> currently observed UUID. Absent parties are omitted.
    pub fn peer_uuids(&self) -> BTreeMap<usize, String>;

    pub fn epoch(&self) -> u64;

    /// Resolves the first time `epoch() != from`.
    pub async fn epoch_changed_from(&self, from: u64);

    /// Stop maintaining `verified_peers`; from now on a peer UUID change
    /// triggers graceful shutdown.
    pub fn seal(&self);
}
```

### 3.2 Additional pre-ready check: peer UUIDs must not have changed

Before we announce readiness, explicitly verify that the peer UUIDs we are about to become
ready *against* are the same ones we proved full visibility for. A difference means a peer
restarted during phase B, and becoming ready against a dead incarnation is what strands the
newcomer (we would freeze/seal holding a UUID nobody has).

This is a distinct check from the epoch watch in 3.3: this one runs *before*
`set_node_ready` and gates the announcement itself; the epoch watch runs *after* and
handles a restart that races the barrier.

```rust
// Snapshot taken at the phase-A gate (mod.rs:145), i.e. the UUID set we proved
// full mutual visibility for.
let mut proven_uuids = visibility.peer_uuids();
let mut churn = 0u32;

// --- additional pre-ready check ---------------------------------------------
// Peer UUIDs must be identical to the set the visibility gate was proven with.
loop {
    let current_uuids = visibility.peer_uuids();
    if current_uuids == proven_uuids {
        break;
    }
    churn += 1;
    tracing::warn!(
        "Peer UUIDs changed during startup load (proven={:?}, current={:?}, churn={}); \
         a peer restarted before the fleet was ready — re-proving startup visibility \
         without reloading irises or graph",
        proven_uuids, current_uuids, churn,
    );
    metrics::counter!("startup_peer_restart_before_ready").increment(1);

    // Cheap, HTTP-only: re-prove that every live party sees every other live party.
    wait_until_startup_visibility_is_complete(
        &server_coord_config, &verified_peers, &my_uuid,
    ).await?;

    // Re-anchor on what we just proved, so a peer that flaps cannot livelock us
    // against a stale snapshot.
    proven_uuids = visibility.peer_uuids();
}
```

The loop terminates when the observed peer set is stable across a proven gate. `churn` is
kept for the log and the metric — a peer flapping repeatedly is visible rather than silent.
Deliberately no exit path here: exiting mints a new UUID and kills our health listener,
which is the documented cause of wedges (`server_coordination.rs:288-296`).

### 3.3 Make the ready barrier re-enterable

`mod.rs:294-295` becomes:

```rust
set_node_ready(Arc::clone(&is_ready_flag));

loop {
    let epoch = visibility.epoch();

    // HTTP-only; no iris/graph reload.
    wait_until_startup_visibility_is_complete(
        &server_coord_config, &verified_peers, &my_uuid,
    ).await?;

    tokio::select! {
        r = wait_for_others_ready(&server_coord_config) => r?,
        _ = visibility.epoch_changed_from(epoch) => {
            tracing::warn!("peer restarted while waiting at the ready barrier; re-syncing");
            continue;
        }
    }

    if visibility.epoch() == epoch {
        break; // all three ready, no restart raced us
    }
    tracing::warn!("peer UUID changed as the barrier completed; re-proving visibility");
}

visibility.seal(); // pre-ready tolerance ends here
```

Nothing in this loop touches the iris store or the graph.

### 3.4 Move `init_heartbeat_task` after the barrier

`init_heartbeat_task` (`mod.rs:286`) currently starts *before* `set_node_ready`, where it
actively fights the tolerance we want: its UUID-mismatch branch triggers graceful shutdown
on a pre-ready peer restart (`server_coordination.rs:~505`), and its init path `panic!`s if
a peer is briefly unreachable. Its initial reachability gate is subsumed by the ready
barrier, so move the call to after the barrier loop.

Preferably drop it from this binary entirely and let the sealed maintainer be the
post-ready watcher: the maintainer already holds the correct post-barrier UUIDs, whereas
`init_heartbeat_task` learns UUIDs at *its own* init and would silently adopt a peer that
restarted in the `seal()` → heartbeat-init gap. If the heartbeat is kept for its other
duties (image-name check, `shutting_down` propagation, consecutive-failure teardown), start
it after `seal()` and accept the small overlap.

### 3.5 Leave `wait_for_others_ready` unbounded

Do not add a process exit to the barrier. With 3.1–3.3 in place the barrier is satisfiable
again, and exiting would re-introduce the new-UUID churn the crate's own doc comment warns
about. Instead add a periodic `warn!` plus a gauge/counter so "at the barrier for > N
minutes" is alertable.

---

## 4. Resulting state machine

```
coordination server up
  └─ maintainer spawned ──────────────────────────────────────────┐ (writes verified_peers,
                                                                  │  tracks peer UUIDs,
  phase A gates (unready + visibility) ──> proven_uuids           │  bumps epoch)
                                                                  │
  phase B load (irises, graph, sync)  <-- peer may restart here   │
    peer restart => maintainer inserts new UUID within one tick   │
                    (newcomer's own gates now pass)               │
                                                                  │
  pre-ready UUID check (3.2): current_uuids == proven_uuids?      │
    no  -> re-prove visibility, re-anchor, loop  (no reload)      │
    yes -> set_node_ready                                         │
                                                                  │
  barrier (3.3): wait_for_others_ready, epoch-interruptible       │
    epoch change -> re-prove visibility, retry  (no reload)       │
                                                                  │
  seal() ─────────────────────────────────────────────────────────┘ (stops writing;
                                                                     UUID change =>
  main server loop                                                   graceful shutdown)
```

**Invariant:** `verified_peers` is live for exactly the window in which a peer restart is
safe, and frozen for exactly the window in which it is not.

---

## 5. Residual races

1. **Restart immediately after `seal()`.** We seal at T, a peer restarts at T+ε. It can no
   longer be verified, so it retries to its deadline while our watcher observes the UUID
   change and triggers graceful shutdown → full-fleet restart. Slower than the pre-ready
   path, but it terminates instead of wedging, and it is the intended post-ready policy.
2. **Staggered seals.** Party A seals while B is still pre-ready and C restarts: A holds
   C's dead UUID. Mitigated by sealing only *after* `wait_for_others_ready` succeeds with a
   stable epoch — all three ready means nobody is pre-ready. The remaining window is
   case 1.
3. **Flapping peer.** Handled by re-anchoring `proven_uuids` on each re-proof (3.2) so we
   never compare against a stale snapshot; visible via `churn` and the counter.

---

## 6. Placement and blast radius

Implementable entirely in-repo — no `ampc-common` bump. Everything needed is public:
`ReadyProbeResponse`, `get_check_addresses`, `try_get_endpoint_other_nodes`,
`wait_until_startup_visibility_is_complete`, `set_node_ready`, `wait_for_others_ready`, and
the `Arc<Mutex<HashSet<String>>>` handed back at `mod.rs:139`.

- new `iris-mpc/src/server/startup_visibility.rs` (~150 lines)
- edits confined to `iris-mpc/src/server/mod.rs:139` and `mod.rs:286-295`

Same phase-B blind spot exists, unfixed, at three other call sites — worth upstreaming the
maintainer into `ampc-server-utils` afterwards rather than copying it:

- `iris-mpc-bins/bin/iris-mpc-anon-stats-server/main.rs:995`
- `iris-mpc-bins/bin/iris-mpc/server.rs:1082`
- `iris-mpc-upgrade-hawk/src/genesis/setup.rs:228` (longest load of the three)

---

## 7. Testable assertions

1. Kill party 2 during phase B; parties 0 and 1 insert its new UUID within one poll
   interval and all three reach `READY=1/1` without manual intervention.
2. Kill party 2 during phase B *after* parties 0 and 1 have passed their phase-A gates; the
   pre-ready check (3.2) fires on 0 and 1, logs `startup_peer_restart_before_ready`, and
   they become ready only against party 2's *new* UUID.
3. Kill party 2 after all three are ready; parties 0 and 1 trigger graceful shutdown
   (unchanged behaviour).
4. `verified_peers` never contains a UUID no peer advertises, at any point in phases A–C.
5. No iris/graph reload occurs on any pre-ready re-sync — assert `init_hawk_actor` is
   entered exactly once per process lifetime.
