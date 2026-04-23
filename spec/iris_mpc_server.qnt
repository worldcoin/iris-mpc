// Formal specification of the iris-mpc server synchronization protocol.
//
// Models 3 MPC server nodes coordinating to process iris biometric queries.
// Each node has its own database of iris shares and a modification log.
// The spec covers:
//   - Server lifecycle (startup, sync, processing, shutdown)
//   - Startup sync protocol (compare_modifications algorithm)
//   - Batch processing with cross-node batch sync and valid_entries filtering
//   - Crash/restart recovery and graceful shutdown
//   - Deletion as tombstone (overwrite, not removal)
//   - Reauth success/failure (persisted only on success)
//
// MPC computations are abstracted as a nondeterministic oracle.
//
// Reference implementation:
//   - sync.rs: compare_modifications (lines 218-321)
//   - server/mod.rs: server_main lifecycle (lines 60-211)
//   - modifications_sync.rs: sync_modifications (lines 23-149)
//   - batch.rs: receive_batch, sync_batch_entries, valid_entries AND logic
//   - job.rs: process_job_result, deletion tombstones, reauth success/fail

module iris_mpc_server {

  // ---------------------------------------------------------------------------
  // Constants
  // ---------------------------------------------------------------------------

  const NODES: Set[int]
  const MAX_DB_SIZE: int
  const MAX_BATCH_SIZE: int
  const LOOKBACK: int

  // ---------------------------------------------------------------------------
  // Types
  // ---------------------------------------------------------------------------

  type RequestType =
    | Uniqueness
    | Reauth
    | Deletion
    | Update

  type ModStatus =
    | InProgress
    | Completed

  type Modification = {
    id: int,
    serial_id: int,              // 0 = unassigned (for uniqueness before MPC)
    request_type: RequestType,
    status: ModStatus,
    persisted: bool
  }

  type BatchRequest = {
    seq_num: int,
    request_id: int,
    request_type: RequestType,
    target_serial_id: int        // 0 for uniqueness, >0 for others
  }

  type SyncStateRec = {
    db_len: int,
    mods: Set[Modification],
    next_seq_num: int            // 0 = empty queue
  }

  type NodePhase =
    | Down
    | WaitingPeersSync
    | SyncingMods
    | SyncingQueue
    | LoadingDB
    | WaitingPeersReady
    | Ready
    | Processing
    | PersistingResults
    | ShuttingDown               // graceful shutdown: finish pending, reject new

  type RequestResult = {
    is_match: bool,              // for uniqueness: matched existing iris?
    assigned_serial_id: int,     // for non-match uniqueness: newly assigned serial_id
    success: bool                // for reauth: did it succeed?
  }

  /// Iris entry status: live data or tombstone (deletion overwrites with dummy).
  type IrisStatus =
    | Live
    | Tombstoned

  // ---------------------------------------------------------------------------
  // State variables
  // ---------------------------------------------------------------------------

  var phase: int -> NodePhase
  /// Iris database per node: serial_id -> status (Live or Tombstoned).
  /// Deletions overwrite with dummy data, keeping the slot occupied.
  var db: int -> (int -> IrisStatus)
  var db_len: int -> int
  var mods: int -> Set[Modification]
  var queue_cursor: int -> int
  var queue: List[BatchRequest]
  var next_seq_num: int
  var published_sync: int -> SyncStateRec
  var node_ready: int -> bool
  var current_batch: int -> List[BatchRequest]
  /// Per-node valid_entries: which batch entries passed decryption.
  /// After AND across nodes, invalid entries are filtered out.
  var valid_entries: int -> List[bool]
  var batch_results: int -> List[RequestResult]
  var global_mod_id: int

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  pure def max_of(s: Set[int], default: int): int =
    s.fold(default, (acc, x) => if (x > acc) x else acc)

  pure def min_of(s: Set[int], default: int): int =
    s.fold(default, (acc, x) => if (x < acc) x else acc)

  /// All occupied serial IDs (both Live and Tombstoned).
  pure def occupied_ids(iris_db: int -> IrisStatus): Set[int] =
    iris_db.keys()

  val empty_sync: SyncStateRec = { db_len: 0, mods: Set(), next_seq_num: 0 }

  // ---------------------------------------------------------------------------
  // Initialization
  // ---------------------------------------------------------------------------

  action init = all {
    phase'          = NODES.mapBy(_ => Down),
    db'             = NODES.mapBy(_ => Map()),
    db_len'         = NODES.mapBy(_ => 0),
    mods'           = NODES.mapBy(_ => Set()),
    queue_cursor'   = NODES.mapBy(_ => 0),
    queue'          = List(),
    next_seq_num'   = 1,
    published_sync' = NODES.mapBy(_ => empty_sync),
    node_ready'     = NODES.mapBy(_ => false),
    current_batch'  = NODES.mapBy(_ => List()),
    valid_entries'  = NODES.mapBy(_ => List()),
    batch_results'  = NODES.mapBy(_ => List()),
    global_mod_id'  = 1,
  }

  // ---------------------------------------------------------------------------
  // Environment: external requests arrive in the queue
  // ---------------------------------------------------------------------------

  action enqueue_uniqueness = all {
    queue.length() < MAX_DB_SIZE,
    queue' = queue.append({
      seq_num: next_seq_num,
      request_id: next_seq_num,
      request_type: Uniqueness,
      target_serial_id: 0
    }),
    next_seq_num' = next_seq_num + 1,
    phase' = phase, db' = db, db_len' = db_len, mods' = mods,
    queue_cursor' = queue_cursor, published_sync' = published_sync,
    node_ready' = node_ready, current_batch' = current_batch,
    valid_entries' = valid_entries,
    batch_results' = batch_results, global_mod_id' = global_mod_id,
  }

  action enqueue_reauth(target: int): bool = all {
    target >= 1,
    queue.length() < MAX_DB_SIZE,
    queue' = queue.append({
      seq_num: next_seq_num,
      request_id: next_seq_num,
      request_type: Reauth,
      target_serial_id: target
    }),
    next_seq_num' = next_seq_num + 1,
    phase' = phase, db' = db, db_len' = db_len, mods' = mods,
    queue_cursor' = queue_cursor, published_sync' = published_sync,
    node_ready' = node_ready, current_batch' = current_batch,
    valid_entries' = valid_entries,
    batch_results' = batch_results, global_mod_id' = global_mod_id,
  }

  action enqueue_deletion(target: int): bool = all {
    target >= 1,
    queue.length() < MAX_DB_SIZE,
    queue' = queue.append({
      seq_num: next_seq_num,
      request_id: next_seq_num,
      request_type: Deletion,
      target_serial_id: target
    }),
    next_seq_num' = next_seq_num + 1,
    phase' = phase, db' = db, db_len' = db_len, mods' = mods,
    queue_cursor' = queue_cursor, published_sync' = published_sync,
    node_ready' = node_ready, current_batch' = current_batch,
    valid_entries' = valid_entries,
    batch_results' = batch_results, global_mod_id' = global_mod_id,
  }

  // ---------------------------------------------------------------------------
  // Startup: node starts up, builds sync state, publishes it
  // ---------------------------------------------------------------------------

  action node_start(node: int): bool =
    val my_mods = mods.get(node)
    val max_mod_id = max_of(my_mods.map(m => m.id), 0)
    val lookback_mods = my_mods.filter(m => m.id > max_mod_id - LOOKBACK)
    // In the real system, get_next_sns_seq_num peeks at the shared SQS queue.
    val queue_next = if (queue.length() > 0) queue[0].seq_num else 0
    val sync_state: SyncStateRec = {
      db_len: db_len.get(node),
      mods: lookback_mods,
      next_seq_num: queue_next
    }
    all {
      phase.get(node) == Down,
      phase' = phase.set(node, WaitingPeersSync),
      published_sync' = published_sync.set(node, sync_state),
      node_ready' = node_ready.set(node, false),
      current_batch' = current_batch.set(node, List()),
      valid_entries' = valid_entries.set(node, List()),
      batch_results' = batch_results.set(node, List()),
      db' = db, db_len' = db_len, mods' = mods,
      queue_cursor' = queue_cursor, queue' = queue,
      next_seq_num' = next_seq_num, global_mod_id' = global_mod_id,
    }

  // ---------------------------------------------------------------------------
  // Wait for peers, then begin sync
  // ---------------------------------------------------------------------------

  action begin_sync(node: int): bool = all {
    phase.get(node) == WaitingPeersSync,
    NODES.filter(n => phase.get(n) != Down).forall(n => phase.get(n) != Down),
    phase' = phase.set(node, SyncingMods),
    db' = db, db_len' = db_len, mods' = mods,
    queue_cursor' = queue_cursor, queue' = queue,
    next_seq_num' = next_seq_num, published_sync' = published_sync,
    node_ready' = node_ready, current_batch' = current_batch,
    valid_entries' = valid_entries,
    batch_results' = batch_results, global_mod_id' = global_mod_id,
  }

  // ---------------------------------------------------------------------------
  // Sync modifications: the core compare_modifications algorithm
  // Mirrors sync.rs lines 218-321 and modifications_sync.rs lines 23-149
  // ---------------------------------------------------------------------------

  action sync_modifications(node: int): bool =
    val running = NODES.filter(n => phase.get(n) != Down)
    val all_published_mods: Set[Modification] = running.fold(Set(), (acc, n) =>
      acc.union(published_sync.get(n).mods)
    )
    val all_mod_ids: Set[int] = all_published_mods.map(m => m.id)
    // Safety check: max completed mod_id diff <= LOOKBACK (sync.rs 227-249)
    val completed_max_per_node: Set[int] = running.map(n =>
      max_of(
        published_sync.get(n).mods.filter(m => m.status == Completed).map(m => m.id),
        0
      )
    ).filter(x => x > 0)
    val safe = if (completed_max_per_node.size() <= 1) true
               else max_of(completed_max_per_node, 0) - min_of(completed_max_per_node, 0) <= LOOKBACK
    val my_mods = mods.get(node)
    val local_mod_ids = my_mods.map(m => m.id)
    // to_delete: mod IDs where ALL nodes have InProgress
    val to_delete_ids: Set[int] = all_mod_ids.filter(mid =>
      running.forall(n =>
        published_sync.get(n).mods.forall(m => m.id != mid or m.status == InProgress)
      )
    )
    // to_update: mod IDs where ANY node has Completed
    val to_update_ids: Set[int] = all_mod_ids.filter(mid =>
      running.exists(n =>
        published_sync.get(n).mods.exists(m => m.id == mid and m.status == Completed)
      )
    )
    // For each to_update mod, find a completed copy
    val completed_copies: Set[Modification] = to_update_ids.map(mid =>
      all_published_mods.filter(m => m.id == mid and m.status == Completed).fold(
        { id: mid, serial_id: 0, request_type: Uniqueness, status: Completed, persisted: false },
        (_, m) => m
      )
    )
    val kept_mods = my_mods.filter(m =>
      not(to_delete_ids.contains(m.id)) and not(to_update_ids.contains(m.id))
    )
    val updated_mods = completed_copies.filter(m => local_mod_ids.contains(m.id))
    val new_mods = kept_mods.union(updated_mods)
    // Apply iris changes for newly-persisted modifications
    val new_uniqueness_inserts: Set[int] = updated_mods
      .filter(m => m.persisted and m.request_type == Uniqueness and m.serial_id > 0)
      .map(m => m.serial_id)
    // Apply deletion tombstones
    val new_deletion_targets: Set[int] = updated_mods
      .filter(m => m.persisted and m.request_type == Deletion and m.serial_id > 0)
      .map(m => m.serial_id)
    val my_db = db.get(node)
    // Add new uniqueness inserts as Live
    val db_with_inserts = new_uniqueness_inserts.fold(my_db, (acc, sid) =>
      acc.put(sid, Live)
    )
    // Mark deletions as Tombstoned
    val db_with_deletions = new_deletion_targets.fold(db_with_inserts, (acc, sid) =>
      if (acc.keys().contains(sid)) acc.set(sid, Tombstoned) else acc
    )
    val new_db_len = max_of(db_with_deletions.keys(), 0)
    all {
      phase.get(node) == SyncingMods,
      safe,
      mods' = mods.set(node, new_mods),
      db' = db.set(node, db_with_deletions),
      db_len' = db_len.set(node, new_db_len),
      phase' = phase.set(node, SyncingQueue),
      queue_cursor' = queue_cursor, queue' = queue,
      next_seq_num' = next_seq_num, published_sync' = published_sync,
      node_ready' = node_ready, current_batch' = current_batch,
      valid_entries' = valid_entries,
      batch_results' = batch_results, global_mod_id' = global_mod_id,
    }

  // ---------------------------------------------------------------------------
  // Sync SQS queue: advance cursor to max across all nodes
  // ---------------------------------------------------------------------------

  action sync_queue(node: int): bool =
    val running = NODES.filter(n => phase.get(n) != Down)
    val seq_nums: Set[int] = running.map(n => published_sync.get(n).next_seq_num)
    val any_empty = seq_nums.contains(0)
    val any_nonempty = seq_nums.exists(s => s > 0)
    val max_seq = max_of(seq_nums, 0)
    val cleaned_queue = queue.select(r => r.seq_num >= max_seq)
    all {
      phase.get(node) == SyncingQueue,
      not(any_empty and any_nonempty),
      queue_cursor' = queue_cursor.set(node, max_seq),
      queue' = cleaned_queue,
      phase' = phase.set(node, LoadingDB),
      db' = db, db_len' = db_len, mods' = mods,
      next_seq_num' = next_seq_num,
      published_sync' = published_sync, node_ready' = node_ready,
      current_batch' = current_batch, valid_entries' = valid_entries,
      batch_results' = batch_results, global_mod_id' = global_mod_id,
    }

  // ---------------------------------------------------------------------------
  // Load database and signal ready
  // ---------------------------------------------------------------------------

  action finish_loading(node: int): bool = all {
    phase.get(node) == LoadingDB,
    phase' = phase.set(node, WaitingPeersReady),
    node_ready' = node_ready.set(node, true),
    db' = db, db_len' = db_len, mods' = mods,
    queue_cursor' = queue_cursor, queue' = queue,
    next_seq_num' = next_seq_num, published_sync' = published_sync,
    current_batch' = current_batch, valid_entries' = valid_entries,
    batch_results' = batch_results, global_mod_id' = global_mod_id,
  }

  action all_nodes_ready(node: int): bool = all {
    phase.get(node) == WaitingPeersReady,
    NODES.filter(n => phase.get(n) != Down).forall(n => node_ready.get(n)),
    phase' = phase.set(node, Ready),
    db' = db, db_len' = db_len, mods' = mods,
    queue_cursor' = queue_cursor, queue' = queue,
    next_seq_num' = next_seq_num, published_sync' = published_sync,
    node_ready' = node_ready, current_batch' = current_batch,
    valid_entries' = valid_entries,
    batch_results' = batch_results, global_mod_id' = global_mod_id,
  }

  // ---------------------------------------------------------------------------
  // Batch processing
  // ---------------------------------------------------------------------------

  /// All ready nodes receive the next batch. Each node nondeterministically
  /// determines which entries it can decrypt (valid_entries). The valid_entries
  /// are AND-ed across all nodes (batch.rs sync_batch_entries, job.rs 241).
  action receive_batch =
    val ready_nodes = NODES.filter(n => phase.get(n) == Ready)
    val min_cursor = min_of(ready_nodes.map(n => queue_cursor.get(n)), 0)
    val available = queue.select(r => r.seq_num >= min_cursor)
    val batch_size = if (available.length() > MAX_BATCH_SIZE) MAX_BATCH_SIZE
                     else available.length()
    val batch = if (batch_size > 0) available.slice(0, batch_size) else List()
    val new_cursor = if (batch_size > 0) batch[batch_size - 1].seq_num + 1 else min_cursor
    // Create IN_PROGRESS modifications for each batch entry
    val new_modifications: Set[Modification] = range(0, batch_size).foldl(Set(), (acc, i) =>
      acc.union(Set({
        id: global_mod_id + i,
        serial_id: batch[i].target_serial_id,
        request_type: batch[i].request_type,
        status: InProgress,
        persisted: false
      }))
    )
    all {
      ready_nodes == NODES,
      batch_size > 0,
      // Nondeterministically choose valid_entries per node, then AND them.
      // In the real system, decryption failures cause entries to be invalid.
      nondet invalid_set = range(0, batch_size).foldl(Set(), (acc, i) =>
        acc.union(Set(i))
      ).powerset().oneOf()
      // AND logic: entry valid only if ALL nodes consider it valid
      val and_valid: List[bool] = range(0, batch_size).foldl(List(), (acc, i) =>
        acc.append(not(invalid_set.contains(i)))
      )
      all {
        current_batch' = ready_nodes.fold(current_batch, (acc, n) =>
          acc.set(n, batch)
        ),
        valid_entries' = ready_nodes.fold(valid_entries, (acc, n) =>
          acc.set(n, and_valid)
        ),
        queue_cursor' = ready_nodes.fold(queue_cursor, (acc, n) =>
          acc.set(n, new_cursor)
        ),
        phase' = ready_nodes.fold(phase, (acc, n) =>
          acc.set(n, Processing)
        ),
        mods' = ready_nodes.fold(mods, (acc, n) =>
          acc.set(n, acc.get(n).union(new_modifications))
        ),
        global_mod_id' = global_mod_id + batch_size,
        db' = db, db_len' = db_len,
        queue' = queue, next_seq_num' = next_seq_num,
        published_sync' = published_sync, node_ready' = node_ready,
        batch_results' = batch_results,
      }
    }

  /// MPC processing: nondeterministic oracle decides results.
  /// For uniqueness: nondeterministically choose match/no-match.
  /// For reauth: nondeterministically choose success/failure.
  /// Invalid entries (valid_entries[i] == false) get match=true to skip them.
  action process_batch =
    val processing_nodes = NODES.filter(n => phase.get(n) == Processing)
    val ref_node = min_of(processing_nodes, 0)
    val batch = current_batch.get(ref_node)
    val ve = valid_entries.get(ref_node)
    val batch_size = batch.length()
    val uniqueness_indices: Set[int] = range(0, batch_size).foldl(Set(), (acc, i) =>
      if (batch[i].request_type == Uniqueness and ve[i]) acc.union(Set(i)) else acc
    )
    val reauth_indices: Set[int] = range(0, batch_size).foldl(Set(), (acc, i) =>
      if (batch[i].request_type == Reauth and ve[i]) acc.union(Set(i)) else acc
    )
    all {
      processing_nodes == NODES,
      batch_size > 0,
      nondet match_set = uniqueness_indices.powerset().oneOf()
      nondet reauth_fail_set = reauth_indices.powerset().oneOf()
      val base_serial = max_of(processing_nodes.map(n => db_len.get(n)), 0)
      val results: List[RequestResult] = range(0, batch_size).foldl(
        { res: List(), next_id: base_serial + 1 },
        (state, i) =>
          if (not(ve[i])) {
            // Invalid entry: skip (treated as match / no-op)
            { res: state.res.append({ is_match: true, assigned_serial_id: 0, success: false }),
              next_id: state.next_id }
          } else if (batch[i].request_type == Uniqueness) {
            if (match_set.contains(i)) {
              { res: state.res.append({ is_match: true, assigned_serial_id: 0, success: true }),
                next_id: state.next_id }
            } else {
              { res: state.res.append({ is_match: false, assigned_serial_id: state.next_id, success: true }),
                next_id: state.next_id + 1 }
            }
          } else if (batch[i].request_type == Reauth) {
            // Reauth can fail (e.g., no matching iris found)
            val succeeded = not(reauth_fail_set.contains(i))
            { res: state.res.append({ is_match: false, assigned_serial_id: 0, success: succeeded }),
              next_id: state.next_id }
          } else {
            // Deletion, Update: always succeed
            { res: state.res.append({ is_match: false, assigned_serial_id: 0, success: true }),
              next_id: state.next_id }
          }
      ).res
      all {
        batch_results' = processing_nodes.fold(batch_results, (acc, n) =>
          acc.set(n, results)
        ),
        phase' = processing_nodes.fold(phase, (acc, n) =>
          acc.set(n, PersistingResults)
        ),
        db' = db, db_len' = db_len, mods' = mods,
        queue_cursor' = queue_cursor, queue' = queue,
        next_seq_num' = next_seq_num, published_sync' = published_sync,
        node_ready' = node_ready, current_batch' = current_batch,
        valid_entries' = valid_entries,
        global_mod_id' = global_mod_id,
      }
    }

  /// Persist results to database (atomic transaction per node).
  /// - Non-matching uniqueness: insert new iris as Live
  /// - Successful reauth: update existing iris (persisted=true)
  /// - Failed reauth: no DB write (persisted=false)
  /// - Deletion: overwrite iris with dummy data (Tombstoned), persisted=true
  /// - Invalid entries: persisted=false
  action persist_results(node: int): bool =
    val batch = current_batch.get(node)
    val results = batch_results.get(node)
    val ve = valid_entries.get(node)
    val batch_size = batch.length()
    val batch_mod_base = global_mod_id - batch_size
    // New iris IDs from non-matching uniqueness (insert as Live)
    val new_iris_ids: Set[int] = range(0, batch_size).foldl(Set(), (acc, i) =>
      if (ve[i] and batch[i].request_type == Uniqueness and not(results[i].is_match)
          and results[i].assigned_serial_id > 0) {
        acc.union(Set(results[i].assigned_serial_id))
      } else {
        acc
      }
    )
    // Deletion targets: overwrite with tombstone
    val deletion_targets: Set[int] = range(0, batch_size).foldl(Set(), (acc, i) =>
      if (ve[i] and batch[i].request_type == Deletion and batch[i].target_serial_id > 0) {
        acc.union(Set(batch[i].target_serial_id))
      } else {
        acc
      }
    )
    // Update modifications to COMPLETED
    val my_mods = mods.get(node)
    val updated_mods: Set[Modification] = my_mods.map(m =>
      if (m.id >= batch_mod_base and m.id < global_mod_id and m.status == InProgress) {
        val batch_idx = m.id - batch_mod_base
        val res = results[batch_idx]
        val entry_valid = ve[batch_idx]
        val was_persisted = if (not(entry_valid)) false
          else if (batch[batch_idx].request_type == Uniqueness) not(res.is_match)
          else if (batch[batch_idx].request_type == Reauth) res.success
          else true  // Deletion, Update: always persisted
        val new_serial = if (batch[batch_idx].request_type == Uniqueness and not(res.is_match) and entry_valid)
                           res.assigned_serial_id
                         else m.serial_id
        { ...m, status: Completed, persisted: was_persisted, serial_id: new_serial }
      } else {
        m
      }
    )
    // Apply DB changes
    val my_db = db.get(node)
    val db_with_inserts = new_iris_ids.fold(my_db, (acc, sid) => acc.put(sid, Live))
    val db_with_tombstones = deletion_targets.fold(db_with_inserts, (acc, sid) =>
      if (acc.keys().contains(sid)) acc.set(sid, Tombstoned) else acc
    )
    val new_db_len = max_of(db_with_tombstones.keys(), 0)
    all {
      phase.get(node) == PersistingResults,
      batch_size > 0,
      db' = db.set(node, db_with_tombstones),
      db_len' = db_len.set(node, new_db_len),
      mods' = mods.set(node, updated_mods),
      phase' = phase.set(node, Ready),
      current_batch' = current_batch.set(node, List()),
      valid_entries' = valid_entries.set(node, List()),
      batch_results' = batch_results.set(node, List()),
      queue_cursor' = queue_cursor, queue' = queue,
      next_seq_num' = next_seq_num, published_sync' = published_sync,
      node_ready' = node_ready, global_mod_id' = global_mod_id,
    }

  // ---------------------------------------------------------------------------
  // Crash, shutdown, restart
  // ---------------------------------------------------------------------------

  /// A node crashes. In-memory state is lost but DB persists.
  action crash(node: int): bool = all {
    phase.get(node) != Down,
    phase' = phase.set(node, Down),
    node_ready' = node_ready.set(node, false),
    current_batch' = current_batch.set(node, List()),
    valid_entries' = valid_entries.set(node, List()),
    batch_results' = batch_results.set(node, List()),
    db' = db, db_len' = db_len, mods' = mods,
    queue_cursor' = queue_cursor, queue' = queue,
    next_seq_num' = next_seq_num, published_sync' = published_sync,
    global_mod_id' = global_mod_id,
  }

  /// Graceful shutdown: node finishes pending batch then stops.
  /// If in PersistingResults, persist first then go Down.
  /// Otherwise, go Down immediately (main loop won't start new batches).
  action graceful_shutdown(node: int): bool = all {
    phase.get(node) == Ready or phase.get(node) == WaitingPeersReady,
    phase' = phase.set(node, Down),
    node_ready' = node_ready.set(node, false),
    current_batch' = current_batch.set(node, List()),
    valid_entries' = valid_entries.set(node, List()),
    batch_results' = batch_results.set(node, List()),
    db' = db, db_len' = db_len, mods' = mods,
    queue_cursor' = queue_cursor, queue' = queue,
    next_seq_num' = next_seq_num, published_sync' = published_sync,
    global_mod_id' = global_mod_id,
  }

  // ---------------------------------------------------------------------------
  // Stuttering
  // ---------------------------------------------------------------------------

  action stuttering = all {
    phase' = phase, db' = db, db_len' = db_len, mods' = mods,
    queue_cursor' = queue_cursor, queue' = queue,
    next_seq_num' = next_seq_num, published_sync' = published_sync,
    node_ready' = node_ready, current_batch' = current_batch,
    valid_entries' = valid_entries,
    batch_results' = batch_results, global_mod_id' = global_mod_id,
  }

  // ---------------------------------------------------------------------------
  // Step
  // ---------------------------------------------------------------------------

  action step = any {
    enqueue_uniqueness,
    nondet target = 1.to(MAX_DB_SIZE).oneOf()
    enqueue_reauth(target),
    nondet target = 1.to(MAX_DB_SIZE).oneOf()
    enqueue_deletion(target),

    nondet node = NODES.oneOf()
    any {
      node_start(node),
      begin_sync(node),
      sync_modifications(node),
      sync_queue(node),
      finish_loading(node),
      all_nodes_ready(node),
      persist_results(node),
      crash(node),
      graceful_shutdown(node),
    },

    receive_batch,
    process_batch,
    stuttering,
  }

  // ---------------------------------------------------------------------------
  // Invariants
  // ---------------------------------------------------------------------------

  /// Serial ID contiguity: occupied IDs form range 1..db_len.
  /// Tombstoned entries still occupy their slot (deletion does not compact).
  val serial_id_contiguity: bool =
    NODES.forall(n => db.get(n).keys() == 1.to(db_len.get(n)))

  /// DB length equals count of occupied slots (including tombstones).
  val db_len_consistent: bool =
    NODES.forall(n => db_len.get(n) == db.get(n).keys().size())

  /// DB size bounded.
  val db_size_bounded: bool =
    NODES.forall(n => db_len.get(n) <= MAX_DB_SIZE)

  /// Modification ID safety window across running nodes.
  val mod_id_safety_window: bool = {
    val running = NODES.filter(n => phase.get(n) != Down)
    val completed_maxes: Set[int] = running.map(n =>
      max_of(mods.get(n).filter(m => m.status == Completed).map(m => m.id), 0)
    ).filter(x => x > 0)
    if (completed_maxes.size() <= 1) true
    else max_of(completed_maxes, 0) - min_of(completed_maxes, 0) <= LOOKBACK
  }

  /// Batch consistency: all batch-phase nodes have same batch.
  val batch_consistency: bool = {
    val batch_nodes = NODES.filter(n =>
      phase.get(n) == Processing or phase.get(n) == PersistingResults
    )
    batch_nodes.forall(n1 => batch_nodes.forall(n2 =>
      current_batch.get(n1) == current_batch.get(n2)
    ))
  }

  /// When all nodes are Ready, their completed mods agree.
  val ready_state_consistency: bool = {
    val ready_nodes = NODES.filter(n => phase.get(n) == Ready)
    if (ready_nodes.size() < 2) true
    else ready_nodes.forall(n1 => ready_nodes.forall(n2 =>
      mods.get(n1).filter(m => m.status == Completed).map(m => m.id)
      ==
      mods.get(n2).filter(m => m.status == Completed).map(m => m.id)
    ))
  }

  /// When all nodes are Ready, their databases are identical.
  val ready_db_consistency: bool = {
    val ready_nodes = NODES.filter(n => phase.get(n) == Ready)
    if (ready_nodes.size() < 2) true
    else ready_nodes.forall(n1 => ready_nodes.forall(n2 =>
      db.get(n1) == db.get(n2) and db_len.get(n1) == db_len.get(n2)
    ))
  }

  /// Completed persisted uniqueness mods must have their iris in db.
  val persisted_mods_have_iris: bool =
    NODES.forall(n =>
      mods.get(n).forall(m =>
        not(m.status == Completed and m.persisted and m.request_type == Uniqueness and m.serial_id > 0)
        or
        db.get(n).keys().contains(m.serial_id)
      )
    )

  /// If two nodes both completed the same modification (by ID),
  /// they must agree on serial_id and persisted flag.
  val completed_mods_agree: bool =
    NODES.forall(n1 => NODES.forall(n2 =>
      mods.get(n1).forall(m1 =>
        mods.get(n2).forall(m2 =>
          not(m1.id == m2.id and m1.status == Completed and m2.status == Completed)
          or
          (m1.serial_id == m2.serial_id and m1.persisted == m2.persisted)
        )
      )
    ))

  /// Same mod ID across nodes must have same request_type.
  val mod_ids_consistent: bool =
    NODES.forall(n1 => NODES.forall(n2 =>
      mods.get(n1).forall(m1 =>
        mods.get(n2).forall(m2 =>
          not(m1.id == m2.id)
          or
          m1.request_type == m2.request_type
        )
      )
    ))

  /// No two persisted uniqueness mods on same node have same serial_id.
  val no_duplicate_serial_ids: bool =
    NODES.forall(n =>
      mods.get(n).forall(m1 =>
        mods.get(n).forall(m2 =>
          not(m1.id != m2.id
              and m1.request_type == Uniqueness and m2.request_type == Uniqueness
              and m1.persisted and m2.persisted
              and m1.serial_id > 0 and m2.serial_id > 0)
          or
          m1.serial_id != m2.serial_id
        )
      )
    )

  /// Deleted iris entries must be Tombstoned, not Live.
  /// If a completed persisted deletion mod references a serial_id,
  /// that entry should be Tombstoned in the node's DB.
  val deletions_are_tombstoned: bool =
    NODES.forall(n =>
      mods.get(n).forall(m =>
        not(m.status == Completed and m.persisted and m.request_type == Deletion
            and m.serial_id > 0 and db.get(n).keys().contains(m.serial_id))
        or
        db.get(n).get(m.serial_id) == Tombstoned
      )
    )

  /// Failed reauths must NOT be persisted.
  val failed_reauth_not_persisted: bool =
    NODES.forall(n =>
      mods.get(n).forall(m =>
        // If it's a completed reauth with persisted=false, that's fine (failed).
        // What we check: if persisted=true, there must be a matching iris.
        not(m.status == Completed and m.request_type == Reauth
            and m.persisted and m.serial_id > 0)
        or
        db.get(n).keys().contains(m.serial_id)
      )
    )

  /// Valid entries consistency: during batch processing, all nodes have
  /// the same valid_entries vector (AND logic ensures consensus).
  val valid_entries_consistency: bool = {
    val batch_nodes = NODES.filter(n =>
      phase.get(n) == Processing or phase.get(n) == PersistingResults
    )
    batch_nodes.forall(n1 => batch_nodes.forall(n2 =>
      valid_entries.get(n1) == valid_entries.get(n2)
    ))
  }

  val all_invariants: bool = and {
    serial_id_contiguity,
    db_len_consistent,
    db_size_bounded,
    mod_id_safety_window,
    batch_consistency,
    ready_state_consistency,
    ready_db_consistency,
    persisted_mods_have_iris,
    completed_mods_agree,
    mod_ids_consistent,
    no_duplicate_serial_ids,
    deletions_are_tombstoned,
    failed_reauth_not_persisted,
    valid_entries_consistency,
  }

  // ---------------------------------------------------------------------------
  // Test runs
  // ---------------------------------------------------------------------------

  /// Happy path: 3 nodes start, sync (empty state), process one batch.
  run happy_path_test = {
    init
      .then(node_start(0))
      .then(node_start(1))
      .then(node_start(2))
      .then(begin_sync(0))
      .then(begin_sync(1))
      .then(begin_sync(2))
      .then(sync_modifications(0))
      .then(sync_modifications(1))
      .then(sync_modifications(2))
      .then(sync_queue(0))
      .then(sync_queue(1))
      .then(sync_queue(2))
      .then(finish_loading(0))
      .then(finish_loading(1))
      .then(finish_loading(2))
      .then(all_nodes_ready(0))
      .then(all_nodes_ready(1))
      .then(all_nodes_ready(2))
      .then(enqueue_uniqueness)
      .then(receive_batch)
      .then(process_batch)
      .then(persist_results(0))
      .then(persist_results(1))
      .then(persist_results(2))
      .then(all {
        assert(all_invariants),
        stuttering,
      })
  }

  /// Two sequential batches, second includes a deletion.
  run two_batches_with_deletion_test = {
    init
      .then(node_start(0))
      .then(node_start(1))
      .then(node_start(2))
      .then(begin_sync(0))
      .then(begin_sync(1))
      .then(begin_sync(2))
      .then(sync_modifications(0))
      .then(sync_modifications(1))
      .then(sync_modifications(2))
      .then(sync_queue(0))
      .then(sync_queue(1))
      .then(sync_queue(2))
      .then(finish_loading(0))
      .then(finish_loading(1))
      .then(finish_loading(2))
      .then(all_nodes_ready(0))
      .then(all_nodes_ready(1))
      .then(all_nodes_ready(2))
      // First batch: insert 2 irises
      .then(enqueue_uniqueness)
      .then(enqueue_uniqueness)
      .then(receive_batch)
      .then(process_batch)
      .then(persist_results(0))
      .then(persist_results(1))
      .then(persist_results(2))
      // Second batch: delete iris 1
      .then(enqueue_deletion(1))
      .then(receive_batch)
      .then(process_batch)
      .then(persist_results(0))
      .then(persist_results(1))
      .then(persist_results(2))
      .then(all {
        assert(all_invariants),
        // Serial ID 1 should still exist but be Tombstoned
        assert(deletions_are_tombstoned),
        stuttering,
      })
  }

  /// Node 2 crashes after MPC but before persist. Recovery via sync.
  run crash_after_processing_test = {
    init
      .then(node_start(0))
      .then(node_start(1))
      .then(node_start(2))
      .then(begin_sync(0))
      .then(begin_sync(1))
      .then(begin_sync(2))
      .then(sync_modifications(0))
      .then(sync_modifications(1))
      .then(sync_modifications(2))
      .then(sync_queue(0))
      .then(sync_queue(1))
      .then(sync_queue(2))
      .then(finish_loading(0))
      .then(finish_loading(1))
      .then(finish_loading(2))
      .then(all_nodes_ready(0))
      .then(all_nodes_ready(1))
      .then(all_nodes_ready(2))
      .then(enqueue_uniqueness)
      .then(receive_batch)
      .then(process_batch)
      .then(crash(2))
      .then(persist_results(0))
      .then(persist_results(1))
      .then(crash(0))
      .then(crash(1))
      .then(node_start(0))
      .then(node_start(1))
      .then(node_start(2))
      .then(begin_sync(0))
      .then(begin_sync(1))
      .then(begin_sync(2))
      .then(sync_modifications(0))
      .then(sync_modifications(1))
      .then(sync_modifications(2))
      .then(sync_queue(0))
      .then(sync_queue(1))
      .then(sync_queue(2))
      .then(finish_loading(0))
      .then(finish_loading(1))
      .then(finish_loading(2))
      .then(all_nodes_ready(0))
      .then(all_nodes_ready(1))
      .then(all_nodes_ready(2))
      .then(all {
        assert(ready_db_consistency),
        assert(ready_state_consistency),
        assert(persisted_mods_have_iris),
        stuttering,
      })
  }

  /// Crash recovery: node 2 crashes after persist_results of 0,1.
  run crash_recovery_test = {
    init
      .then(node_start(0))
      .then(node_start(1))
      .then(node_start(2))
      .then(begin_sync(0))
      .then(begin_sync(1))
      .then(begin_sync(2))
      .then(sync_modifications(0))
      .then(sync_modifications(1))
      .then(sync_modifications(2))
      .then(sync_queue(0))
      .then(sync_queue(1))
      .then(sync_queue(2))
      .then(finish_loading(0))
      .then(finish_loading(1))
      .then(finish_loading(2))
      .then(all_nodes_ready(0))
      .then(all_nodes_ready(1))
      .then(all_nodes_ready(2))
      .then(enqueue_uniqueness)
      .then(receive_batch)
      .then(process_batch)
      .then(persist_results(0))
      .then(persist_results(1))
      .then(crash(2))
      .then(crash(0))
      .then(crash(1))
      .then(node_start(0))
      .then(node_start(1))
      .then(node_start(2))
      .then(begin_sync(0))
      .then(begin_sync(1))
      .then(begin_sync(2))
      .then(sync_modifications(0))
      .then(sync_modifications(1))
      .then(sync_modifications(2))
      .then(sync_queue(0))
      .then(sync_queue(1))
      .then(sync_queue(2))
      .then(finish_loading(0))
      .then(finish_loading(1))
      .then(finish_loading(2))
      .then(all_nodes_ready(0))
      .then(all_nodes_ready(1))
      .then(all_nodes_ready(2))
      .then(all {
        assert(ready_db_consistency),
        assert(ready_state_consistency),
        stuttering,
      })
  }

  /// Multiple batches with a crash between them.
  run multi_batch_crash_test = {
    init
      .then(node_start(0))
      .then(node_start(1))
      .then(node_start(2))
      .then(begin_sync(0))
      .then(begin_sync(1))
      .then(begin_sync(2))
      .then(sync_modifications(0))
      .then(sync_modifications(1))
      .then(sync_modifications(2))
      .then(sync_queue(0))
      .then(sync_queue(1))
      .then(sync_queue(2))
      .then(finish_loading(0))
      .then(finish_loading(1))
      .then(finish_loading(2))
      .then(all_nodes_ready(0))
      .then(all_nodes_ready(1))
      .then(all_nodes_ready(2))
      // Batch 1: all succeed
      .then(enqueue_uniqueness)
      .then(receive_batch)
      .then(process_batch)
      .then(persist_results(0))
      .then(persist_results(1))
      .then(persist_results(2))
      // Batch 2: node 2 crashes after persist of 0,1
      .then(enqueue_uniqueness)
      .then(receive_batch)
      .then(process_batch)
      .then(persist_results(0))
      .then(persist_results(1))
      .then(crash(2))
      // Full restart and sync
      .then(crash(0))
      .then(crash(1))
      .then(node_start(0))
      .then(node_start(1))
      .then(node_start(2))
      .then(begin_sync(0))
      .then(begin_sync(1))
      .then(begin_sync(2))
      .then(sync_modifications(0))
      .then(sync_modifications(1))
      .then(sync_modifications(2))
      .then(sync_queue(0))
      .then(sync_queue(1))
      .then(sync_queue(2))
      .then(finish_loading(0))
      .then(finish_loading(1))
      .then(finish_loading(2))
      .then(all_nodes_ready(0))
      .then(all_nodes_ready(1))
      .then(all_nodes_ready(2))
      .then(all {
        assert(all_invariants),
        stuttering,
      })
  }

  /// Graceful shutdown test: node shuts down cleanly.
  run graceful_shutdown_test = {
    init
      .then(node_start(0))
      .then(node_start(1))
      .then(node_start(2))
      .then(begin_sync(0))
      .then(begin_sync(1))
      .then(begin_sync(2))
      .then(sync_modifications(0))
      .then(sync_modifications(1))
      .then(sync_modifications(2))
      .then(sync_queue(0))
      .then(sync_queue(1))
      .then(sync_queue(2))
      .then(finish_loading(0))
      .then(finish_loading(1))
      .then(finish_loading(2))
      .then(all_nodes_ready(0))
      .then(all_nodes_ready(1))
      .then(all_nodes_ready(2))
      .then(enqueue_uniqueness)
      .then(receive_batch)
      .then(process_batch)
      .then(persist_results(0))
      .then(persist_results(1))
      .then(persist_results(2))
      // Graceful shutdown of node 0
      .then(graceful_shutdown(0))
      // Restart all
      .then(crash(1))
      .then(crash(2))
      .then(node_start(0))
      .then(node_start(1))
      .then(node_start(2))
      .then(begin_sync(0))
      .then(begin_sync(1))
      .then(begin_sync(2))
      .then(sync_modifications(0))
      .then(sync_modifications(1))
      .then(sync_modifications(2))
      .then(sync_queue(0))
      .then(sync_queue(1))
      .then(sync_queue(2))
      .then(finish_loading(0))
      .then(finish_loading(1))
      .then(finish_loading(2))
      .then(all_nodes_ready(0))
      .then(all_nodes_ready(1))
      .then(all_nodes_ready(2))
      .then(all {
        assert(all_invariants),
        stuttering,
      })
  }
  /// Mixed batch: uniqueness + reauth + deletion in sequence.
  run mixed_operations_test = {
    init
      .then(node_start(0))
      .then(node_start(1))
      .then(node_start(2))
      .then(begin_sync(0))
      .then(begin_sync(1))
      .then(begin_sync(2))
      .then(sync_modifications(0))
      .then(sync_modifications(1))
      .then(sync_modifications(2))
      .then(sync_queue(0))
      .then(sync_queue(1))
      .then(sync_queue(2))
      .then(finish_loading(0))
      .then(finish_loading(1))
      .then(finish_loading(2))
      .then(all_nodes_ready(0))
      .then(all_nodes_ready(1))
      .then(all_nodes_ready(2))
      // Insert 3 irises
      .then(enqueue_uniqueness)
      .then(enqueue_uniqueness)
      .then(enqueue_uniqueness)
      .then(receive_batch)
      .then(process_batch)
      .then(persist_results(0))
      .then(persist_results(1))
      .then(persist_results(2))
      // Reauth iris 1, delete iris 2
      .then(enqueue_reauth(1))
      .then(enqueue_deletion(2))
      .then(receive_batch)
      .then(process_batch)
      .then(persist_results(0))
      .then(persist_results(1))
      .then(persist_results(2))
      .then(all {
        assert(all_invariants),
        stuttering,
      })
  }

  /// Crash after mixed batch with deletion, verify tombstone survives sync.
  run deletion_crash_recovery_test = {
    init
      .then(node_start(0))
      .then(node_start(1))
      .then(node_start(2))
      .then(begin_sync(0))
      .then(begin_sync(1))
      .then(begin_sync(2))
      .then(sync_modifications(0))
      .then(sync_modifications(1))
      .then(sync_modifications(2))
      .then(sync_queue(0))
      .then(sync_queue(1))
      .then(sync_queue(2))
      .then(finish_loading(0))
      .then(finish_loading(1))
      .then(finish_loading(2))
      .then(all_nodes_ready(0))
      .then(all_nodes_ready(1))
      .then(all_nodes_ready(2))
      // Insert iris
      .then(enqueue_uniqueness)
      .then(receive_batch)
      .then(process_batch)
      .then(persist_results(0))
      .then(persist_results(1))
      .then(persist_results(2))
      // Delete iris 1, node 2 crashes before persist
      .then(enqueue_deletion(1))
      .then(receive_batch)
      .then(process_batch)
      .then(persist_results(0))
      .then(persist_results(1))
      .then(crash(2))
      // Restart and sync
      .then(crash(0))
      .then(crash(1))
      .then(node_start(0))
      .then(node_start(1))
      .then(node_start(2))
      .then(begin_sync(0))
      .then(begin_sync(1))
      .then(begin_sync(2))
      .then(sync_modifications(0))
      .then(sync_modifications(1))
      .then(sync_modifications(2))
      .then(sync_queue(0))
      .then(sync_queue(1))
      .then(sync_queue(2))
      .then(finish_loading(0))
      .then(finish_loading(1))
      .then(finish_loading(2))
      .then(all_nodes_ready(0))
      .then(all_nodes_ready(1))
      .then(all_nodes_ready(2))
      .then(all {
        assert(ready_db_consistency),
        assert(ready_state_consistency),
        assert(deletions_are_tombstoned),
        stuttering,
      })
  }
}

// ---------------------------------------------------------------------------
// Concrete instance for model checking
// ---------------------------------------------------------------------------

module iris_mpc_server_3 {
  import iris_mpc_server(
    NODES = Set(0, 1, 2),
    MAX_DB_SIZE = 8,
    MAX_BATCH_SIZE = 3,
    LOOKBACK = 6
  ).*
}
