//! Pure set computation for the version-join delta mode.
//!
//! Given `(serial → version)` maps for the materialized graph, the source iris
//! store, and the HNSW iris store, derive the two action sets that drive the
//! delta: `graph_replay` (nodes whose content is stale in the graph and must be
//! re-searched + re-inserted) and `store_repair` (HNSW iris rows that disagree
//! with the source but need no graph work). Anomalies that the version
//! comparison cannot safely act on are collected separately.

use iris_mpc_common::{SerialId, VersionId};
use std::collections::{HashMap, HashSet};

/// Anomaly classes surfaced by [`compute_version_join`]. Each is logged and
/// counted; only `missing_hnsw_row` is acted upon (via INSERT — see
/// [`VersionJoinPlan::missing_hnsw_rows`]).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VersionJoinAnomalies {
    /// `source_version < graph_version` — graph ahead of source (impossible in a
    /// source-built lineage).
    pub graph_ahead: Vec<SerialId>,
    /// Present in source but absent from the graph — never indexed.
    pub graph_missing: Vec<SerialId>,
    /// Present in source but absent from the HNSW iris store — repaired via
    /// INSERT rather than UPDATE.
    pub missing_hnsw_row: Vec<SerialId>,
    /// `serial ≤ max_serial` absent from the source pool.
    pub source_missing: Vec<SerialId>,
}

impl VersionJoinAnomalies {
    /// Total number of anomalous serials across all classes.
    pub fn total(&self) -> usize {
        self.graph_ahead.len()
            + self.graph_missing.len()
            + self.missing_hnsw_row.len()
            + self.source_missing.len()
    }
}

/// The plan produced by [`compute_version_join`]. All serial vectors are sorted
/// ascending (the function iterates serials in order), so replay order is
/// identical across parties.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VersionJoinPlan {
    /// Serials whose graph node is stale: MPC search + insert at the current
    /// source version, then persist the HNSW row.
    pub graph_replay: Vec<SerialId>,
    /// Serials whose HNSW iris row disagrees with the source but need no graph
    /// work. Overwrite the row from the pool.
    pub store_repair: Vec<SerialId>,
    /// Subset of `store_repair` whose HNSW row is missing entirely; INSERT
    /// instead of UPDATE.
    pub missing_hnsw_rows: Vec<SerialId>,
    /// Anomalies to log; not acted upon (except `missing_hnsw_row`).
    pub anomalies: VersionJoinAnomalies,
}

/// Fold `(serial, version)` pairs into a map keeping the **maximum** version per
/// serial. The base graph can hold several versions of one serial (ghosts left
/// by prior delta runs), so the live version is the maximum.
pub fn max_version_per_serial(
    pairs: impl IntoIterator<Item = (SerialId, VersionId)>,
) -> HashMap<SerialId, VersionId> {
    let mut out: HashMap<SerialId, VersionId> = HashMap::new();
    for (serial, version) in pairs {
        out.entry(serial)
            .and_modify(|v| {
                if version > *v {
                    *v = version;
                }
            })
            .or_insert(version);
    }
    out
}

/// Compute the version-join plan. Pure: no I/O.
///
/// `graph_versions` must already carry the max version per serial (see
/// [`max_version_per_serial`]). `excluded` serials are dropped from every set
/// (deletion is the one case version comparison cannot see). Serials are
/// considered over `1..=max_serial`.
pub fn compute_version_join(
    graph_versions: &HashMap<SerialId, VersionId>,
    source_versions: &HashMap<SerialId, VersionId>,
    hnsw_versions: &HashMap<SerialId, VersionId>,
    excluded: &HashSet<SerialId>,
    max_serial: SerialId,
) -> VersionJoinPlan {
    let mut plan = VersionJoinPlan::default();

    for serial in 1..=max_serial {
        if excluded.contains(&serial) {
            continue;
        }

        let Some(&v_src) = source_versions.get(&serial) else {
            plan.anomalies.source_missing.push(serial);
            continue;
        };

        // Graph axis: decide whether the node's content is stale.
        let in_replay = match graph_versions.get(&serial) {
            None => {
                plan.anomalies.graph_missing.push(serial);
                false
            }
            Some(&v_graph) => {
                if v_src > v_graph {
                    plan.graph_replay.push(serial);
                    true
                } else {
                    if v_src < v_graph {
                        plan.anomalies.graph_ahead.push(serial);
                    }
                    false
                }
            }
        };

        // Store axis: reconcile the HNSW iris row. Replay serials get their row
        // rewritten by the replay-result persistence path, so skip them here.
        if !in_replay {
            match hnsw_versions.get(&serial) {
                None => {
                    plan.store_repair.push(serial);
                    plan.missing_hnsw_rows.push(serial);
                    plan.anomalies.missing_hnsw_row.push(serial);
                }
                Some(&v_hnsw) if v_hnsw != v_src => {
                    plan.store_repair.push(serial);
                }
                Some(_) => {}
            }
        }
    }

    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(SerialId, VersionId)]) -> HashMap<SerialId, VersionId> {
        pairs.iter().copied().collect()
    }

    fn excl(serials: &[SerialId]) -> HashSet<SerialId> {
        serials.iter().copied().collect()
    }

    #[test]
    fn source_bump_triggers_graph_replay_not_double_counted_store_repair() {
        // serial 1: source ahead of graph AND hnsw stale → replay only.
        let graph = map(&[(1, 3)]);
        let source = map(&[(1, 4)]);
        let hnsw = map(&[(1, 3)]);
        let plan = compute_version_join(&graph, &source, &hnsw, &excl(&[]), 1);
        assert_eq!(plan.graph_replay, vec![1]);
        assert!(
            plan.store_repair.is_empty(),
            "replay serial must not also be store_repair"
        );
        assert_eq!(plan.anomalies.total(), 0);
    }

    #[test]
    fn hnsw_mismatch_both_directions_is_store_repair() {
        // serial 1: hnsw behind source; serial 2: hnsw ahead of source.
        // Graph matches source for both → no replay.
        let graph = map(&[(1, 5), (2, 5)]);
        let source = map(&[(1, 5), (2, 5)]);
        let hnsw = map(&[(1, 4), (2, 6)]);
        let plan = compute_version_join(&graph, &source, &hnsw, &excl(&[]), 2);
        assert!(plan.graph_replay.is_empty());
        assert_eq!(plan.store_repair, vec![1, 2]);
        assert!(plan.missing_hnsw_rows.is_empty());
        assert_eq!(plan.anomalies.total(), 0);
    }

    #[test]
    fn max_version_semantics_ghost_and_live() {
        // Graph holds ghost at v=1 and live at v=2 for serial 1; collapsed to max=2.
        let graph = max_version_per_serial([(1u32, 1i16), (1, 2)]);
        assert_eq!(graph.get(&1), Some(&2));

        // source at v+1 (==2) → no replay.
        let source_current = map(&[(1, 2)]);
        let hnsw = map(&[(1, 2)]);
        let plan = compute_version_join(&graph, &source_current, &hnsw, &excl(&[]), 1);
        assert!(plan.graph_replay.is_empty());
        assert_eq!(plan.anomalies.total(), 0);

        // source at v+2 (==3) → replay.
        let source_ahead = map(&[(1, 3)]);
        let plan = compute_version_join(&graph, &source_ahead, &hnsw, &excl(&[]), 1);
        assert_eq!(plan.graph_replay, vec![1]);
    }

    #[test]
    fn excluded_serials_dropped_from_both_sets() {
        // serial 1 would replay, serial 2 would store_repair — both excluded.
        let graph = map(&[(1, 1), (2, 2)]);
        let source = map(&[(1, 2), (2, 2)]);
        let hnsw = map(&[(1, 1), (2, 1)]);
        let plan = compute_version_join(&graph, &source, &hnsw, &excl(&[1, 2]), 2);
        assert!(plan.graph_replay.is_empty());
        assert!(plan.store_repair.is_empty());
        assert_eq!(plan.anomalies.total(), 0);
    }

    #[test]
    fn each_anomaly_class_lands_in_anomalies_not_action_sets() {
        // 1: graph ahead of source.
        // 2: in source, absent from graph.
        // 3: in source, absent from hnsw (acted on via insert + counted).
        // 4: absent from source pool.
        let graph = map(&[(1, 5), (4, 1)]);
        let source = map(&[(1, 4), (2, 2), (3, 2)]);
        let hnsw = map(&[(1, 4), (2, 2), (4, 1)]);
        let plan = compute_version_join(&graph, &source, &hnsw, &excl(&[]), 4);

        assert_eq!(plan.anomalies.graph_ahead, vec![1]);
        assert_eq!(plan.anomalies.graph_missing, vec![2, 3]);
        assert_eq!(plan.anomalies.missing_hnsw_row, vec![3]);
        assert_eq!(plan.anomalies.source_missing, vec![4]);

        // graph_ahead does not replay; store axis matched (1) so no repair for 1.
        assert!(plan.graph_replay.is_empty());
        // serial 3 missing from hnsw → store_repair + missing_hnsw_rows.
        assert_eq!(plan.store_repair, vec![3]);
        assert_eq!(plan.missing_hnsw_rows, vec![3]);
    }

    #[test]
    fn empty_diff_yields_empty_plan() {
        let graph = map(&[(1, 1), (2, 1), (3, 1)]);
        let source = map(&[(1, 1), (2, 1), (3, 1)]);
        let hnsw = map(&[(1, 1), (2, 1), (3, 1)]);
        let plan = compute_version_join(&graph, &source, &hnsw, &excl(&[]), 3);
        assert_eq!(plan, VersionJoinPlan::default());
    }
}
