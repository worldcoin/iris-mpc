//! Pure set computation for the genesis version-join delta.
//!
//! Graph work is a uniform repair — remove every key for the serial, then
//! search + reinsert at the current source version — applied to any serial
//! that disagrees with the source on any axis: graph version behind or ahead,
//! several keys (ghosts), absent from the graph, or an HNSW iris row that is
//! missing or at the wrong version (a row-level disagreement means the graph
//! entry may have been built against wrong content, so graph-key equality
//! alone is not trustworthy).
//!
//! Plans are computed per eye and unioned by the driver: source and HNSW row
//! versions are per-serial (eye-invariant), so only per-eye graph state can
//! differ, and asymmetric per-eye damage is repairable by a uniform repair
//! (reinsertion is a harmless refresh for a clean eye).
//!
//! Deletion is invisible to version comparison (it bumps the source version
//! and writes dummy shares); [`partition_repair`] separates tombstones (remove
//! only) from live serials (remove + reinsert).

use iris_mpc_common::{SerialId, VersionId};
use std::collections::{HashMap, HashSet};

/// Per-class counts over one eye's repair set; logging/metrics only. The
/// graph classes are mutually exclusive per serial; the store classes are
/// independent of them.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RepairReasons {
    /// `max graph version < source version`.
    pub version_behind: usize,
    /// `max graph version > source version`.
    pub version_ahead: usize,
    /// Several graph keys for the serial, max version agreeing with source.
    pub multi_version: usize,
    /// Present in source but absent from the graph.
    pub graph_missing: usize,
    /// HNSW iris row absent.
    pub row_missing: usize,
    /// HNSW iris row version differs from source.
    pub row_mismatch: usize,
}

/// The version-comparison output of [`compute_version_join`] for one eye. All
/// serial vectors are sorted ascending, so processing order is identical
/// across parties.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VersionJoinPlan {
    /// Serials needing a graph repair: remove every existing key, then reinsert
    /// at the current source version iff live (see [`partition_repair`]).
    pub graph_repair: Vec<SerialId>,
    /// Per-class breakdown of `graph_repair`, for logging only.
    pub repair_reasons: RepairReasons,
    /// Serials with no HNSW row at all; INSERTed at the row-write flush.
    /// Subset of `graph_repair`. Eye-invariant.
    pub missing_hnsw_rows: Vec<SerialId>,
    /// `serial ≤ max_serial` absent from the source pool. Logged, not acted
    /// on. Eye-invariant.
    pub source_missing: Vec<SerialId>,
}

/// Final action sets after the deletion split. All row writes are deferred to
/// a single flush once the repaired graph is durable (see [`Self::row_writes`]):
/// a row that turns source-consistent earlier would erase its own repair
/// trigger while the distrusted graph could still be lost.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RepairPlan {
    /// Live serials: remove all existing keys, then MPC search + insert at the
    /// current source version; the HNSW row is rewritten at flush.
    pub graph_replay: Vec<SerialId>,
    /// Deletion tombstones with graph presence: remove all keys, no reinsert.
    pub graph_remove: Vec<SerialId>,
    /// Serials without an HNSW row: INSERTed (source content) at flush.
    pub insert_missing_rows: Vec<SerialId>,
    /// Tombstones with a stale HNSW row: overwritten from source content at
    /// flush (no replay covers their row).
    pub stale_tombstone_rows: Vec<SerialId>,
}

impl RepairPlan {
    /// Row writes for the flush: `(inserts, updates)`, both sorted. Inserts
    /// are the absent rows; updates are every replayed serial plus the
    /// tombstone overwrites, minus the freshly inserted rows.
    pub fn row_writes(&self) -> (Vec<SerialId>, Vec<SerialId>) {
        let inserts = self.insert_missing_rows.clone();
        let insert_set: HashSet<SerialId> = inserts.iter().copied().collect();
        let mut updates: Vec<SerialId> = self
            .graph_replay
            .iter()
            .chain(self.stale_tombstone_rows.iter())
            .filter(|s| !insert_set.contains(s))
            .copied()
            .collect();
        updates.sort_unstable();
        updates.dedup();
        (inserts, updates)
    }
}

/// Fold `(serial, version)` pairs into all versions per serial. Per-serial
/// order follows the input iterator — sort before deriving ordered operations.
pub fn versions_per_serial(
    pairs: impl IntoIterator<Item = (SerialId, VersionId)>,
) -> HashMap<SerialId, Vec<VersionId>> {
    let mut out: HashMap<SerialId, Vec<VersionId>> = HashMap::new();
    for (serial, version) in pairs {
        out.entry(serial).or_default().push(version);
    }
    out
}

/// Compute one eye's version-join plan over `1..=max_serial`. Pure: no I/O.
///
/// `graph_versions` must carry every graph key per serial (see
/// [`versions_per_serial`]). Deletions are not distinguished here;
/// [`partition_repair`] resolves the tombstones.
pub fn compute_version_join(
    graph_versions: &HashMap<SerialId, Vec<VersionId>>,
    source_versions: &HashMap<SerialId, VersionId>,
    hnsw_versions: &HashMap<SerialId, VersionId>,
    max_serial: SerialId,
) -> VersionJoinPlan {
    let mut plan = VersionJoinPlan::default();

    for serial in 1..=max_serial {
        let Some(&v_src) = source_versions.get(&serial) else {
            plan.source_missing.push(serial);
            continue;
        };

        // Graph axis.
        let mut repair = match graph_versions.get(&serial) {
            None => {
                plan.repair_reasons.graph_missing += 1;
                true
            }
            Some(versions) => {
                let v_max = *versions
                    .iter()
                    .max()
                    .expect("versions_per_serial: non-empty");
                if v_max < v_src {
                    plan.repair_reasons.version_behind += 1;
                    true
                } else if v_max > v_src {
                    plan.repair_reasons.version_ahead += 1;
                    true
                } else if versions.len() > 1 {
                    plan.repair_reasons.multi_version += 1;
                    true
                } else {
                    false
                }
            }
        };

        // Store axis: any row disagreement is a repair too. A missing row is
        // additionally INSERTed at flush; a stale live row is rewritten by
        // the replay's flush entry, and a stale tombstone row by the
        // overwrite `partition_repair` routes it to.
        match hnsw_versions.get(&serial) {
            None => {
                plan.repair_reasons.row_missing += 1;
                plan.missing_hnsw_rows.push(serial);
                repair = true;
            }
            Some(&v_hnsw) if v_hnsw != v_src => {
                plan.repair_reasons.row_mismatch += 1;
                repair = true;
            }
            Some(_) => {}
        }

        if repair {
            plan.graph_repair.push(serial);
        }
    }

    plan
}

/// Split the (cross-eye union) repair set into per-action lists.
///
/// `deleted` holds the tombstones (serials whose source content is the
/// deletion dummy); `serials_in_graph` holds serials with at least one graph
/// key in either eye. Tombstones without graph presence get no graph job;
/// tombstones with a stale HNSW row go to `stale_tombstone_rows`, those
/// without a row are already covered by `insert_missing_rows` (inserted with
/// source content, i.e. the tombstone).
pub fn make_repair_plan(
    repair: &[SerialId],
    missing_hnsw_rows: &[SerialId],
    deleted: &HashSet<SerialId>,
    serials_in_graph: &HashSet<SerialId>,
    source_versions: &HashMap<SerialId, VersionId>,
    hnsw_versions: &HashMap<SerialId, VersionId>,
) -> RepairPlan {
    let mut out = RepairPlan {
        insert_missing_rows: missing_hnsw_rows.to_vec(),
        ..Default::default()
    };

    for &serial in repair {
        if !deleted.contains(&serial) {
            out.graph_replay.push(serial);
            continue;
        }
        if serials_in_graph.contains(&serial) {
            out.graph_remove.push(serial);
        }
        let stale_row = match (hnsw_versions.get(&serial), source_versions.get(&serial)) {
            (Some(v_hnsw), Some(v_src)) => v_hnsw != v_src,
            _ => false,
        };
        if stale_row {
            out.stale_tombstone_rows.push(serial);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(entries: &[(SerialId, &[VersionId])]) -> HashMap<SerialId, Vec<VersionId>> {
        entries.iter().map(|(s, vs)| (*s, vs.to_vec())).collect()
    }

    fn map(pairs: &[(SerialId, VersionId)]) -> HashMap<SerialId, VersionId> {
        pairs.iter().copied().collect()
    }

    fn set(serials: &[SerialId]) -> HashSet<SerialId> {
        serials.iter().copied().collect()
    }

    fn split(
        plan: &VersionJoinPlan,
        deleted: &HashSet<SerialId>,
        graph_versions: &HashMap<SerialId, Vec<VersionId>>,
        source_versions: &HashMap<SerialId, VersionId>,
        hnsw_versions: &HashMap<SerialId, VersionId>,
    ) -> RepairPlan {
        let serials_in_graph: HashSet<SerialId> = graph_versions.keys().copied().collect();
        make_repair_plan(
            &plan.graph_repair,
            &plan.missing_hnsw_rows,
            deleted,
            &serials_in_graph,
            source_versions,
            hnsw_versions,
        )
    }

    #[test]
    fn versions_per_serial_collects_all_keys() {
        let m = versions_per_serial([(1u32, 3i16), (1, 1), (2, 5)]);
        assert_eq!(m.get(&1), Some(&vec![3, 1]));
        assert_eq!(m.get(&2), Some(&vec![5]));
    }

    #[test]
    fn version_behind_is_repair() {
        let g = graph(&[(1, &[3])]);
        let plan = compute_version_join(&g, &map(&[(1, 4)]), &map(&[(1, 3)]), 1);
        assert_eq!(plan.graph_repair, vec![1]);
        assert_eq!(plan.repair_reasons.version_behind, 1);
        // Stale row counted, but the replay rewrites it (no direct action).
        assert_eq!(plan.repair_reasons.row_mismatch, 1);
        assert!(plan.missing_hnsw_rows.is_empty());
    }

    #[test]
    fn version_ahead_and_multi_version_are_repair() {
        // 1: graph ahead of source; 2: ghost + live agreeing on max.
        let g = graph(&[(1, &[5]), (2, &[2, 4])]);
        let source = map(&[(1, 4), (2, 4)]);
        let hnsw = map(&[(1, 4), (2, 4)]);
        let plan = compute_version_join(&g, &source, &hnsw, 2);
        assert_eq!(plan.graph_repair, vec![1, 2]);
        assert_eq!(plan.repair_reasons.version_ahead, 1);
        assert_eq!(plan.repair_reasons.multi_version, 1);
        assert!(plan.missing_hnsw_rows.is_empty());
    }

    #[test]
    fn graph_missing_is_repair() {
        let plan = compute_version_join(&graph(&[]), &map(&[(1, 1)]), &map(&[(1, 1)]), 1);
        assert_eq!(plan.graph_repair, vec![1]);
        assert_eq!(plan.repair_reasons.graph_missing, 1);
    }

    #[test]
    fn hnsw_row_mismatch_with_clean_graph_is_repair() {
        // Clean graph; HNSW row behind (1) and ahead (2) of source. A
        // row-level disagreement disqualifies the graph entry too.
        let g = graph(&[(1, &[5]), (2, &[5])]);
        let source = map(&[(1, 5), (2, 5)]);
        let hnsw = map(&[(1, 4), (2, 6)]);
        let plan = compute_version_join(&g, &source, &hnsw, 2);
        assert_eq!(plan.graph_repair, vec![1, 2]);
        assert_eq!(plan.repair_reasons.row_mismatch, 2);
        assert!(plan.missing_hnsw_rows.is_empty());

        // Live rows: no direct row action — the replay rewrites them.
        let repair = split(&plan, &set(&[]), &g, &source, &hnsw);
        assert_eq!(repair.graph_replay, vec![1, 2]);
        assert!(repair.stale_tombstone_rows.is_empty());
        assert!(repair.insert_missing_rows.is_empty());
    }

    #[test]
    fn missing_hnsw_row_is_repair_and_inserted() {
        // 1: graph behind + missing row; 2: clean graph + missing row.
        let g = graph(&[(1, &[1]), (2, &[2])]);
        let source = map(&[(1, 2), (2, 2)]);
        let plan = compute_version_join(&g, &source, &map(&[]), 2);
        assert_eq!(plan.graph_repair, vec![1, 2]);
        assert_eq!(plan.missing_hnsw_rows, vec![1, 2]);
        assert_eq!(plan.repair_reasons.row_missing, 2);
    }

    #[test]
    fn source_missing_is_logged_not_acted_on() {
        let g = graph(&[(2, &[1])]);
        let plan = compute_version_join(&g, &map(&[(2, 1)]), &map(&[(2, 1)]), 2);
        assert_eq!(plan.source_missing, vec![1]);
        assert!(plan.graph_repair.is_empty());
    }

    #[test]
    fn empty_diff_yields_empty_plan() {
        let g = graph(&[(1, &[1]), (2, &[1]), (3, &[1])]);
        let versions = map(&[(1, 1), (2, 1), (3, 1)]);
        let plan = compute_version_join(&g, &versions, &versions, 3);
        assert_eq!(plan, VersionJoinPlan::default());
    }

    #[test]
    fn partition_routes_live_and_tombstone_repair() {
        // 1: live, behind → replay. 2: tombstone with ghosts → remove-only,
        // stale row → overwrite. 3: tombstone never indexed → no graph job,
        // missing row inserted (with tombstone content).
        let g = graph(&[(1, &[1]), (2, &[1, 2])]);
        let source = map(&[(1, 2), (2, 3), (3, 2)]);
        let hnsw = map(&[(1, 1), (2, 1)]);
        let plan = compute_version_join(&g, &source, &hnsw, 3);
        assert_eq!(plan.graph_repair, vec![1, 2, 3]);
        assert_eq!(plan.missing_hnsw_rows, vec![3]);

        let repair = split(&plan, &set(&[2, 3]), &g, &source, &hnsw);
        assert_eq!(repair.graph_replay, vec![1]);
        assert_eq!(repair.graph_remove, vec![2]);
        assert_eq!(repair.stale_tombstone_rows, vec![2]);
        assert_eq!(repair.insert_missing_rows, vec![3]);
    }

    #[test]
    fn row_writes_updates_exclude_fresh_inserts() {
        // 1: replayed, row present → update. 2: replayed, row absent →
        // insert only. 3: tombstone overwrite → update. 4: tombstone,
        // row absent → insert only.
        let plan = RepairPlan {
            graph_replay: vec![1, 2],
            graph_remove: vec![3, 4],
            insert_missing_rows: vec![2, 4],
            stale_tombstone_rows: vec![3],
        };
        let (inserts, updates) = plan.row_writes();
        assert_eq!(inserts, vec![2, 4]);
        assert_eq!(updates, vec![1, 3]);
    }

    #[test]
    fn split_without_tombstones_is_all_replay() {
        let g = graph(&[(1, &[1])]);
        let source = map(&[(1, 2)]);
        let hnsw = map(&[(1, 2)]);
        let plan = compute_version_join(&g, &source, &hnsw, 1);
        let repair = split(&plan, &set(&[]), &g, &source, &hnsw);
        assert_eq!(repair.graph_replay, vec![1]);
        assert!(repair.graph_remove.is_empty());
        assert!(repair.stale_tombstone_rows.is_empty());
    }

    #[test]
    fn per_eye_plans_differ_only_on_graph_axis() {
        // Ghost in one eye only: that eye classifies multi_version, the other
        // is clean. The driver unions the sets; both eyes get a uniform repair.
        let g_left = graph(&[(1, &[1, 2])]);
        let g_right = graph(&[(1, &[2])]);
        let source = map(&[(1, 2)]);
        let hnsw = map(&[(1, 2)]);
        let left = compute_version_join(&g_left, &source, &hnsw, 1);
        let right = compute_version_join(&g_right, &source, &hnsw, 1);
        assert_eq!(left.graph_repair, vec![1]);
        assert!(right.graph_repair.is_empty());
        assert_eq!(left.missing_hnsw_rows, right.missing_hnsw_rows);
        assert_eq!(left.source_missing, right.source_missing);
    }
}
