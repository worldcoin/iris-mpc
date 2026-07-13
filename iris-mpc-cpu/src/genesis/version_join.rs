//! Pure set computation for the version-join delta mode.
//!
//! Given per-serial version state for the materialized graph, the source iris
//! store, and the HNSW iris store, derive the action sets that drive the
//! delta. Graph work is uniform surgery — remove every existing key for the
//! serial, then search + reinsert at the current source version — applied to
//! any serial whose graph lineage is suspect: version behind or ahead of the
//! source, several keys for one serial (ghosts), or absent from the graph.
//!
//! Deletion is the one case version comparison cannot see: a deletion bumps
//! the source version and replaces the content with the party's deterministic
//! dummy shares. [`VersionJoinPlan::split`] therefore separates surgery
//! serials whose source content is the deletion tombstone (remove only, no
//! reinsert) from live ones (remove + reinsert).

use iris_mpc_common::{SerialId, VersionId};
use std::collections::{HashMap, HashSet};

/// Why serials entered the surgery set; one exclusive class per serial.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SurgeryReasons {
    /// `max graph version < source version`.
    pub version_behind: usize,
    /// `max graph version > source version`.
    pub version_ahead: usize,
    /// Several graph keys for the serial, max version agreeing with source.
    pub multi_version: usize,
    /// Present in source but absent from the graph.
    pub graph_missing: usize,
}

impl SurgeryReasons {
    pub fn total(&self) -> usize {
        self.version_behind + self.version_ahead + self.multi_version + self.graph_missing
    }
}

/// The version-comparison output of [`compute_version_join`]. All serial
/// vectors are sorted ascending (the function iterates serials in order), so
/// processing order is identical across parties.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VersionJoinPlan {
    /// Serials needing graph surgery: remove every existing key, then search +
    /// reinsert at the current source version iff the source content is live
    /// (see [`Self::split`]).
    pub graph_surgery: Vec<SerialId>,
    /// Per-class breakdown of `graph_surgery`.
    pub surgery_reasons: SurgeryReasons,
    /// Non-surgery serials whose HNSW iris row disagrees with the source.
    /// Overwrite the row from the pool.
    pub store_repair: Vec<SerialId>,
    /// Serials (surgery or not) whose HNSW row is missing entirely; INSERTed
    /// during the store-repair phase so a later replay UPDATE finds a row.
    /// Always a subset of `store_repair`.
    pub missing_hnsw_rows: Vec<SerialId>,
    /// `serial ≤ max_serial` absent from the source pool. Logged, not acted on.
    pub source_missing: Vec<SerialId>,
}

/// Final action sets after the deletion split.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SurgeryPlan {
    /// Live serials: remove all existing keys, then MPC search + insert at the
    /// current source version, and persist the HNSW row.
    pub graph_replay: Vec<SerialId>,
    /// Deletion tombstones with graph presence: remove all keys, no reinsert.
    pub graph_remove: Vec<SerialId>,
    /// Serials whose HNSW iris row is overwritten from the pool.
    pub store_repair: Vec<SerialId>,
    /// Subset of `store_repair` to INSERT instead of UPDATE.
    pub missing_hnsw_rows: Vec<SerialId>,
}

/// Fold `(serial, version)` pairs into all versions per serial. The graph can
/// hold several keys for one serial (ghosts left by prior deltas); the max is
/// the live candidate and every key is a removal target. Per-serial order
/// follows the input iterator — sort before deriving ordered operations.
pub fn versions_per_serial(
    pairs: impl IntoIterator<Item = (SerialId, VersionId)>,
) -> HashMap<SerialId, Vec<VersionId>> {
    let mut out: HashMap<SerialId, Vec<VersionId>> = HashMap::new();
    for (serial, version) in pairs {
        out.entry(serial).or_default().push(version);
    }
    out
}

/// Compute the version-join plan. Pure: no I/O.
///
/// `graph_versions` must carry every graph key per serial (see
/// [`versions_per_serial`]). Serials are considered over `1..=max_serial`.
/// Deletions are not distinguished here — a deleted serial classifies by its
/// (bumped) source version like any other; [`VersionJoinPlan::split`] resolves
/// the tombstones.
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

        // Graph axis: any suspect lineage goes to surgery.
        let surgery = match graph_versions.get(&serial) {
            None => {
                plan.surgery_reasons.graph_missing += 1;
                true
            }
            Some(versions) => {
                let v_max = *versions
                    .iter()
                    .max()
                    .expect("versions_per_serial: non-empty");
                if v_max < v_src {
                    plan.surgery_reasons.version_behind += 1;
                    true
                } else if v_max > v_src {
                    plan.surgery_reasons.version_ahead += 1;
                    true
                } else if versions.len() > 1 {
                    plan.surgery_reasons.multi_version += 1;
                    true
                } else {
                    false
                }
            }
        };
        if surgery {
            plan.graph_surgery.push(serial);
        }

        // Store axis: reconcile the HNSW iris row. A missing row is inserted
        // even for surgery serials (the replay persistence path is an UPDATE);
        // a present-but-stale row under surgery is left to that path, except
        // for tombstones, which `split` routes back to store repair.
        match hnsw_versions.get(&serial) {
            None => {
                plan.store_repair.push(serial);
                plan.missing_hnsw_rows.push(serial);
            }
            Some(&v_hnsw) if v_hnsw != v_src && !surgery => {
                plan.store_repair.push(serial);
            }
            Some(_) => {}
        }
    }

    plan
}

impl VersionJoinPlan {
    /// Split surgery into replay vs remove-only using `deleted` (serials whose
    /// source content equals the deletion tombstone). Tombstones without graph
    /// presence are dropped (nothing to remove, nothing to insert); tombstones
    /// with a stale HNSW row are added to `store_repair` (no replay will
    /// rewrite their row).
    pub fn split(
        &self,
        deleted: &HashSet<SerialId>,
        graph_versions: &HashMap<SerialId, Vec<VersionId>>,
        source_versions: &HashMap<SerialId, VersionId>,
        hnsw_versions: &HashMap<SerialId, VersionId>,
    ) -> SurgeryPlan {
        let mut out = SurgeryPlan {
            store_repair: self.store_repair.clone(),
            missing_hnsw_rows: self.missing_hnsw_rows.clone(),
            ..Default::default()
        };

        for &serial in &self.graph_surgery {
            if !deleted.contains(&serial) {
                out.graph_replay.push(serial);
                continue;
            }
            if graph_versions.contains_key(&serial) {
                out.graph_remove.push(serial);
            }
            let stale_row = match (hnsw_versions.get(&serial), source_versions.get(&serial)) {
                (Some(v_hnsw), Some(v_src)) => v_hnsw != v_src,
                _ => false,
            };
            if stale_row {
                out.store_repair.push(serial);
            }
        }
        out.store_repair.sort_unstable();

        out
    }
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

    #[test]
    fn versions_per_serial_collects_all_keys() {
        let m = versions_per_serial([(1u32, 3i16), (1, 1), (2, 5)]);
        assert_eq!(m.get(&1), Some(&vec![3, 1]));
        assert_eq!(m.get(&2), Some(&vec![5]));
    }

    #[test]
    fn version_behind_is_surgery_without_store_repair() {
        // Graph behind source; stale HNSW row is left to the replay path.
        let g = graph(&[(1, &[3])]);
        let plan = compute_version_join(&g, &map(&[(1, 4)]), &map(&[(1, 3)]), 1);
        assert_eq!(plan.graph_surgery, vec![1]);
        assert_eq!(plan.surgery_reasons.version_behind, 1);
        assert!(plan.store_repair.is_empty());
    }

    #[test]
    fn version_ahead_and_multi_version_are_surgery() {
        // 1: graph ahead of source; 2: ghost + live agreeing on max.
        let g = graph(&[(1, &[5]), (2, &[2, 4])]);
        let source = map(&[(1, 4), (2, 4)]);
        let hnsw = map(&[(1, 4), (2, 4)]);
        let plan = compute_version_join(&g, &source, &hnsw, 2);
        assert_eq!(plan.graph_surgery, vec![1, 2]);
        assert_eq!(plan.surgery_reasons.version_ahead, 1);
        assert_eq!(plan.surgery_reasons.multi_version, 1);
        assert!(plan.store_repair.is_empty());
    }

    #[test]
    fn graph_missing_is_surgery() {
        let plan = compute_version_join(&graph(&[]), &map(&[(1, 1)]), &map(&[(1, 1)]), 1);
        assert_eq!(plan.graph_surgery, vec![1]);
        assert_eq!(plan.surgery_reasons.graph_missing, 1);
    }

    #[test]
    fn hnsw_mismatch_both_directions_is_store_repair() {
        // Clean graph; HNSW row behind (1) and ahead (2) of source.
        let g = graph(&[(1, &[5]), (2, &[5])]);
        let source = map(&[(1, 5), (2, 5)]);
        let hnsw = map(&[(1, 4), (2, 6)]);
        let plan = compute_version_join(&g, &source, &hnsw, 2);
        assert!(plan.graph_surgery.is_empty());
        assert_eq!(plan.store_repair, vec![1, 2]);
        assert!(plan.missing_hnsw_rows.is_empty());
    }

    #[test]
    fn missing_hnsw_row_is_inserted_even_under_surgery() {
        // 1: surgery + missing row; 2: clean graph + missing row.
        let g = graph(&[(1, &[1]), (2, &[2])]);
        let source = map(&[(1, 2), (2, 2)]);
        let plan = compute_version_join(&g, &source, &map(&[]), 2);
        assert_eq!(plan.graph_surgery, vec![1]);
        assert_eq!(plan.store_repair, vec![1, 2]);
        assert_eq!(plan.missing_hnsw_rows, vec![1, 2]);
    }

    #[test]
    fn source_missing_is_logged_not_acted_on() {
        let g = graph(&[(2, &[1])]);
        let plan = compute_version_join(&g, &map(&[(2, 1)]), &map(&[(2, 1)]), 2);
        assert_eq!(plan.source_missing, vec![1]);
        assert!(plan.graph_surgery.is_empty());
        assert!(plan.store_repair.is_empty());
    }

    #[test]
    fn empty_diff_yields_empty_plan() {
        let g = graph(&[(1, &[1]), (2, &[1]), (3, &[1])]);
        let versions = map(&[(1, 1), (2, 1), (3, 1)]);
        let plan = compute_version_join(&g, &versions, &versions, 3);
        assert_eq!(plan, VersionJoinPlan::default());
    }

    #[test]
    fn split_routes_live_and_tombstone_surgery() {
        // 1: live, behind → replay. 2: tombstone with ghosts → remove-only,
        // stale row → store repair. 3: tombstone never indexed → dropped.
        let g = graph(&[(1, &[1]), (2, &[1, 2])]);
        let source = map(&[(1, 2), (2, 3), (3, 2)]);
        let hnsw = map(&[(1, 1), (2, 1)]);
        let plan = compute_version_join(&g, &source, &hnsw, 3);
        assert_eq!(plan.graph_surgery, vec![1, 2, 3]);
        // 3 has no HNSW row → inserted (with tombstone content) via store repair.
        assert_eq!(plan.missing_hnsw_rows, vec![3]);

        let surgery = plan.split(&set(&[2, 3]), &g, &source, &hnsw);
        assert_eq!(surgery.graph_replay, vec![1]);
        assert_eq!(surgery.graph_remove, vec![2]);
        assert_eq!(surgery.store_repair, vec![2, 3]);
        assert_eq!(surgery.missing_hnsw_rows, vec![3]);
    }

    #[test]
    fn split_without_tombstones_is_all_replay() {
        let g = graph(&[(1, &[1])]);
        let source = map(&[(1, 2)]);
        let hnsw = map(&[(1, 2)]);
        let plan = compute_version_join(&g, &source, &hnsw, 1);
        let surgery = plan.split(&set(&[]), &g, &source, &hnsw);
        assert_eq!(surgery.graph_replay, vec![1]);
        assert!(surgery.graph_remove.is_empty());
        assert!(surgery.store_repair.is_empty());
    }
}
