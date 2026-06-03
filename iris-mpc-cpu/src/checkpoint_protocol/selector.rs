//! Snapshot-consensus policy. Given each party's recent-checkpoints list,
//! pick the row everyone agrees on (or report no agreement).
//!
//! Two impls:
//!
//! - [`StrictLatest`] requires every party's newest checkpoint to be
//!   byte-identical. Matches the original Phase 1 behaviour: any divergence
//!   on the freshest row aborts the cycle.
//!
//! - [`MostRecentCommon`] walks the local recent list (newest first) and
//!   returns the first entry present in every peer's list — the genesis
//!   `find_common_checkpoint` policy. Tolerates a single party missing the
//!   latest row (e.g. crash between hash-consensus and DB insert) by falling
//!   back to the previous common ancestor.

use crate::checkpoint_protocol::CheckpointMeta;

/// Policy for selecting the agreed base from each party's recent list.
///
/// All input lists are assumed to be **newest first** (matches
/// `MutationStore::recent_checkpoints` and the genesis
/// `get_genesis_graph_checkpoints` ordering).
pub trait BaseSelector: Send + Sync {
    fn pick(
        &self,
        my_recent: &[CheckpointMeta],
        peer_lists: &[Vec<CheckpointMeta>],
    ) -> Option<CheckpointMeta>;
}

/// "All parties' newest checkpoint must match exactly, or fail."
#[derive(Default, Clone, Copy, Debug)]
pub struct StrictLatest;

impl BaseSelector for StrictLatest {
    fn pick(
        &self,
        my_recent: &[CheckpointMeta],
        peer_lists: &[Vec<CheckpointMeta>],
    ) -> Option<CheckpointMeta> {
        let mine = my_recent.first()?;
        for peer in peer_lists {
            let theirs = peer.first()?;
            if !theirs.same_checkpoint(mine) {
                return None;
            }
        }
        Some(mine.clone())
    }
}

/// "Walk my recent list newest-first; return the first entry present in
/// every peer's list."
///
/// Mirrors genesis's `find_common_checkpoint`. Tolerates per-party history
/// divergence within the recent-checkpoints window — pick the most recent
/// ancestor everyone has.
#[derive(Default, Clone, Copy, Debug)]
pub struct MostRecentCommon;

impl BaseSelector for MostRecentCommon {
    fn pick(
        &self,
        my_recent: &[CheckpointMeta],
        peer_lists: &[Vec<CheckpointMeta>],
    ) -> Option<CheckpointMeta> {
        my_recent
            .iter()
            .find(|cp| {
                peer_lists
                    .iter()
                    .all(|peer| peer.iter().any(|p| p.same_checkpoint(cp)))
            })
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cp(id: i64) -> CheckpointMeta {
        CheckpointMeta {
            checkpoint_id: id,
            s3_key: format!("cp/{id}"),
            last_indexed_iris_id: 0,
            last_indexed_modification_id: 0,
            graph_mutation_id: Some(id * 100),
            blake3_hash: format!("hash_{id}"),
            graph_version: 1,
        }
    }

    // ── StrictLatest ─────────────────────────────────────────────────────

    #[test]
    fn strict_all_newest_agree_picks_it() {
        let mine = vec![cp(5), cp(4)];
        let peers = vec![vec![cp(5), cp(4)], vec![cp(5), cp(3)]];
        assert_eq!(StrictLatest.pick(&mine, &peers), Some(cp(5)));
    }

    #[test]
    fn strict_one_peer_diverges_on_newest_returns_none() {
        let mine = vec![cp(5), cp(4)];
        let peers = vec![vec![cp(5), cp(4)], vec![cp(4), cp(3)]];
        assert_eq!(StrictLatest.pick(&mine, &peers), None);
    }

    #[test]
    fn strict_empty_my_list_returns_none() {
        let peers = vec![vec![cp(5)]];
        assert_eq!(StrictLatest.pick(&[], &peers), None);
    }

    #[test]
    fn strict_empty_peer_list_returns_none() {
        let mine = vec![cp(5)];
        let peers = vec![vec![]];
        assert_eq!(StrictLatest.pick(&mine, &peers), None);
    }

    // ── MostRecentCommon ─────────────────────────────────────────────────

    #[test]
    fn common_all_three_share_newest_picks_it() {
        let mine = vec![cp(5), cp(4)];
        let peers = vec![vec![cp(5), cp(4)], vec![cp(5)]];
        assert_eq!(MostRecentCommon.pick(&mine, &peers), Some(cp(5)));
    }

    /// One peer is one row behind on the newest checkpoint. The protocol
    /// must fall back to the previous common ancestor.
    #[test]
    fn common_one_peer_behind_falls_back_to_previous() {
        let mine = vec![cp(5), cp(4), cp(3)];
        let peers = vec![vec![cp(5), cp(4), cp(3)], vec![cp(4), cp(3)]];
        assert_eq!(MostRecentCommon.pick(&mine, &peers), Some(cp(4)));
    }

    #[test]
    fn common_no_overlap_returns_none() {
        let mine = vec![cp(5), cp(4)];
        let peers = vec![vec![cp(10), cp(9)], vec![cp(8), cp(7)]];
        assert_eq!(MostRecentCommon.pick(&mine, &peers), None);
    }

    #[test]
    fn common_only_two_of_three_agree_returns_none() {
        // I + peer 0 share cp(5); peer 1 does not.
        let mine = vec![cp(5), cp(4)];
        let peers = vec![vec![cp(5), cp(4)], vec![cp(6)]];
        // cp(5) fails (peer 1 lacks it); cp(4) fails (peer 1 lacks it) → None.
        assert_eq!(MostRecentCommon.pick(&mine, &peers), None);
    }

    /// Newest-first ordering: when multiple checkpoints are shared, the
    /// caller's order (newest first) decides which one we pick.
    #[test]
    fn common_picks_newest_of_multiple_shared() {
        let mine = vec![cp(5), cp(4), cp(3)];
        let peers = vec![vec![cp(5), cp(4), cp(3)], vec![cp(5), cp(4), cp(3)]];
        assert_eq!(MostRecentCommon.pick(&mine, &peers), Some(cp(5)));
    }

    #[test]
    fn common_empty_my_list_returns_none() {
        let peers = vec![vec![cp(5)]];
        assert_eq!(MostRecentCommon.pick(&[], &peers), None);
    }

    /// Two peers (the realistic 3-party case): both must contain the
    /// candidate. A row only one peer has is not a common ancestor.
    #[test]
    fn common_three_party_needs_both_peers() {
        let mine = vec![cp(5), cp(4), cp(3)];
        let peers = vec![vec![cp(5)], vec![cp(4)]];
        // cp(5): peer 1 lacks → skip. cp(4): peer 0 lacks → skip. None.
        assert_eq!(MostRecentCommon.pick(&mine, &peers), None);
    }

    /// Cross-party checkpoint_id divergence: each party's DB assigns a
    /// different auto-increment id for the same logical checkpoint.
    /// The selector must still find agreement via blake3_hash.
    #[test]
    fn common_cross_party_checkpoint_id_differs() {
        // All three parties have the same checkpoint but with different DB ids.
        let mine = vec![CheckpointMeta {
            checkpoint_id: 1,
            s3_key: "cp/5".into(),
            last_indexed_iris_id: 0,
            last_indexed_modification_id: 50,
            graph_mutation_id: Some(500),
            blake3_hash: "hash_5".into(),
            graph_version: 1,
        }];
        let peer0 = vec![CheckpointMeta {
            checkpoint_id: 3, // different DB id
            s3_key: "cp/5".into(),
            last_indexed_iris_id: 0,
            last_indexed_modification_id: 50,
            graph_mutation_id: Some(500),
            blake3_hash: "hash_5".into(),
            graph_version: 1,
        }];
        let peer1 = vec![CheckpointMeta {
            checkpoint_id: 7, // different DB id
            s3_key: "cp/5".into(),
            last_indexed_iris_id: 0,
            last_indexed_modification_id: 50,
            graph_mutation_id: Some(500),
            blake3_hash: "hash_5".into(),
            graph_version: 1,
        }];
        let result = MostRecentCommon.pick(&mine, &vec![peer0, peer1]);
        assert!(
            result.is_some(),
            "should find common checkpoint despite differing checkpoint_ids"
        );
        assert_eq!(result.unwrap().blake3_hash, "hash_5");
    }

    /// Same cross-party divergence test for StrictLatest.
    #[test]
    fn strict_cross_party_checkpoint_id_differs() {
        let mine = vec![CheckpointMeta {
            checkpoint_id: 1,
            s3_key: "cp/5".into(),
            last_indexed_iris_id: 0,
            last_indexed_modification_id: 50,
            graph_mutation_id: Some(500),
            blake3_hash: "hash_5".into(),
            graph_version: 1,
        }];
        let peer0 = vec![CheckpointMeta {
            checkpoint_id: 3,
            s3_key: "cp/5".into(),
            last_indexed_iris_id: 0,
            last_indexed_modification_id: 50,
            graph_mutation_id: Some(500),
            blake3_hash: "hash_5".into(),
            graph_version: 1,
        }];
        let result = StrictLatest.pick(&mine, &vec![peer0]);
        assert!(
            result.is_some(),
            "should agree despite differing checkpoint_ids"
        );
    }
}
