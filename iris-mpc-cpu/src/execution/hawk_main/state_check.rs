use ampc_secret_sharing::RingElement;
use eyre::{bail, eyre, Result};
use futures::join;
use iris_mpc_common::vector_id::VectorId;
use serde::{Deserialize, Serialize};
use siphasher::sip::SipHasher13;
use std::{
    hash::{Hash, Hasher},
    sync::{
        atomic::{AtomicU8, Ordering},
        Arc,
    },
    time::Duration,
};

use crate::{
    execution::hawk_main::{BothOrient, LEFT, RIGHT},
    genesis::{SYNC_DONE, SYNC_ERROR},
    network::mpc::{NetworkValue, StateChecksum},
};

use super::{BothEyes, HawkSession};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SetHash {
    accumulator: u64,
}

impl SetHash {
    pub fn add_unordered(&mut self, value: impl Hash) {
        self.accumulator = self.accumulator.wrapping_add(Self::hash(value));
    }

    pub fn remove(&mut self, value: impl Hash) {
        self.accumulator = self.accumulator.wrapping_sub(Self::hash(value));
    }

    pub fn checksum(&self) -> u64 {
        self.accumulator
    }

    pub(crate) fn hash(value: impl Hash) -> u64 {
        let mut hasher = SipHasher13::default();
        value.hash(&mut hasher);
        hasher.finish()
    }
}

// ── Binary-search state machine for iris diff debugging ──────────────────

pub(crate) struct OpenRange {
    pub(crate) lo: u32,
    pub(crate) hi: u32,
    /// Hashes for [self, prev, next] over this range.
    hashes: [u64; 3],
}

/// Networking-free binary search over the serial-ID space.
///
/// Keeps a worklist of ranges where at least two of the three parties
/// disagree.  Each round splits every range at the midpoint, and the
/// caller is responsible for exchanging left-half hashes with the two
/// neighbours (via the network in production, or directly in tests).
pub(crate) struct IrisDiffSearch {
    worklist: Vec<OpenRange>,
    found: Vec<u32>,
    max_samples: usize,
    max_rounds: usize,
    round: usize,
}

impl IrisDiffSearch {
    pub fn new(global_hi: u32, hashes: [u64; 3], max_samples: usize) -> Self {
        Self {
            worklist: vec![OpenRange {
                lo: 0,
                hi: global_hi,
                hashes,
            }],
            found: Vec::new(),
            max_samples,
            max_rounds: 40,
            round: 0,
        }
    }

    /// Extract leaf-level findings from the worklist, then compute this
    /// party's left-half hashes for the remaining ranges.
    ///
    /// Returns `None` when the search is complete (worklist empty, sample
    /// cap reached, or round limit hit).  On `Some`, returns the ranges
    /// to split and this party's left-half hashes — pass both to
    /// [`complete_round`] after exchanging hashes with neighbours.
    pub fn prepare_round(
        &mut self,
        range_hash: impl Fn(u32, u32) -> u64,
    ) -> Option<(Vec<OpenRange>, Vec<u64>)> {
        if self.worklist.is_empty()
            || self.found.len() >= self.max_samples
            || self.round >= self.max_rounds
        {
            return None;
        }
        self.round += 1;

        let current = std::mem::take(&mut self.worklist);
        let mut to_split = Vec::new();
        for r in current {
            if r.hi - r.lo <= 1 {
                if self.found.len() < self.max_samples {
                    self.found.push(r.lo);
                }
            } else {
                to_split.push(r);
            }
        }

        if to_split.is_empty() {
            return None;
        }

        let my_lefts: Vec<u64> = to_split
            .iter()
            .map(|r| {
                let mid = r.lo + (r.hi - r.lo) / 2;
                range_hash(r.lo, mid)
            })
            .collect();

        Some((to_split, my_lefts))
    }

    /// Feed the exchanged left-half hashes back and advance the search.
    pub fn complete_round(
        &mut self,
        to_split: Vec<OpenRange>,
        my_lefts: &[u64],
        prev_lefts: &[u64],
        next_lefts: &[u64],
    ) {
        for (i, r) in to_split.into_iter().enumerate() {
            // Cap the worklist to prevent blow-up when many serial IDs
            // differ — each range can spawn 2 children per round.
            if self.found.len() + self.worklist.len() >= self.max_samples * 2 {
                break;
            }

            let mid = r.lo + (r.hi - r.lo) / 2;

            let ml = my_lefts[i];
            let pl = prev_lefts[i];
            let nl = next_lefts[i];

            let mr = r.hashes[0].wrapping_sub(ml);
            let pr = r.hashes[1].wrapping_sub(pl);
            let nr = r.hashes[2].wrapping_sub(nl);

            if ml != pl || ml != nl {
                self.worklist.push(OpenRange {
                    lo: r.lo,
                    hi: mid,
                    hashes: [ml, pl, nl],
                });
            }
            if mr != pr || mr != nr {
                self.worklist.push(OpenRange {
                    lo: mid,
                    hi: r.hi,
                    hashes: [mr, pr, nr],
                });
            }
        }
    }

    /// Drain any remaining leaf-level items and return the sorted results.
    pub fn into_found(mut self) -> Vec<u32> {
        for r in &self.worklist {
            if r.hi - r.lo <= 1 && self.found.len() < self.max_samples {
                self.found.push(r.lo);
            }
        }
        self.found.sort_unstable();
        self.found
    }
}

// ── HawkSession methods ──────────────────────────────────────────────────

impl HawkSession {
    /// Returns true if there is a mismatch in shutdown states between nodes.
    pub async fn sync_peers(
        shutdown_flag: bool,
        sync_status: Arc<AtomicU8>,
        sessions: &BothEyes<Vec<HawkSession>>,
    ) -> Result<bool> {
        let session = &sessions[0][0];

        let decode_u16 = |msg| match msg {
            Ok(NetworkValue::RingElement16(elem)) => Ok(elem.0),
            Ok(other) => {
                tracing::error!("Unexpected message variant in sync: {:?}", other);
                Err(eyre!("Could not deserialize sync result"))
            }
            Err(e) => {
                tracing::error!("Network receive error in sync: {e}");
                Err(e)
            }
        };

        // Consensus loop — exchange "ready to exit" flags until all
        // parties are ready. A party is ready when persistence completes (DONE)
        // or the results thread crashes (ERROR).
        // this step is combined with an exchange of the shutdown flag
        //
        // Receives are retried independently of sends to handle inter-party
        // timing drift — e.g. when one party finishes an S3 checkpoint upload
        // well before the others and enters this loop first.  Retrying only
        // the receive avoids filling the peer's mpsc channel with duplicate
        // sends.
        // TODO: upstream `ampc-common` should expose typed network errors so
        // we can distinguish timeouts from permanent failures here.
        let shutdown: u16 = if shutdown_flag { 1 } else { 0 } << 8;
        let mut prev;
        let mut next;
        loop {
            let my_status = sync_status.load(Ordering::Relaxed);
            let msg = NetworkValue::RingElement16(RingElement(my_status as u16 | shutdown));

            {
                let mut store = session.aby3_store.write().await;
                let net = &mut store.session.network_session;
                net.send_prev(msg.clone()).await?;
                net.send_next(msg).await?;
            }

            prev = loop {
                let mut store = session.aby3_store.write().await;
                let net = &mut store.session.network_session;
                match decode_u16(net.receive_prev().await) {
                    Ok(val) => break val,
                    Err(e) => {
                        tracing::warn!("Retrying sync receive_prev after error: {e}");
                        drop(store);
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
            };

            next = loop {
                let mut store = session.aby3_store.write().await;
                let net = &mut store.session.network_session;
                match decode_u16(net.receive_next().await) {
                    Ok(val) => break val,
                    Err(e) => {
                        tracing::warn!("Retrying sync receive_next after error: {e}");
                        drop(store);
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
            };

            let ready = my_status >= SYNC_DONE;
            if ready && (prev & 0xFF) as u8 >= SYNC_DONE && (next & 0xFF) as u8 >= SYNC_DONE {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Error exchange — if any party's results thread died, all
        // parties bail with an error.
        let my_error = sync_status.load(Ordering::Relaxed) == SYNC_ERROR;
        let prev_error = (prev & 0xFF) as u8 == SYNC_ERROR;
        let next_error = (next & 0xFF) as u8 == SYNC_ERROR;
        if my_error || prev_error || next_error {
            bail!("Results thread terminated before persistence completed");
        }

        // Compare shutdown flags
        let prev_shutdown = (prev >> 8) as u8 == 1;
        let next_shutdown = (next >> 8) as u8 == 1;
        Ok(prev_shutdown != shutdown_flag || next_shutdown != shutdown_flag)
    }

    pub async fn prf_check(sessions: &BothOrient<BothEyes<Vec<HawkSession>>>) -> Result<()> {
        // make a function because the borrow checker can't track the lifetimes properly if this was a closure
        async fn squeeze_rng(session: &HawkSession) -> Result<()> {
            let mut store = session.aby3_store.write().await;
            let prf = &mut store.session.prf;

            let my_share = prf.gen_zero_share::<u128>();
            let my_msg = || NetworkValue::PrfCheck(my_share);

            let net = &mut store.session.network_session;
            net.send_prev(my_msg()).await?;
            net.send_next(my_msg()).await?;

            let decode = |msg| match msg {
                Ok(NetworkValue::PrfCheck(c)) => Ok(c),
                Ok(other) => {
                    tracing::error!("Unexpected message variant in PRF check: {:?}", other);
                    Err(eyre!("Could not deserialize PrfCheck"))
                }
                Err(e) => {
                    tracing::error!("Network receive error in PRF check: {e}");
                    Err(e)
                }
            };
            let prev_share = decode(net.receive_prev().await)?;
            let next_share = decode(net.receive_next().await)?;

            if (prev_share + my_share + next_share).convert() != 0_u128 {
                bail!("PRFs are out of sync");
            }
            Ok(())
        }

        let _ = futures::future::try_join_all(
            sessions
                .iter()
                .flat_map(|orient| orient.iter().flat_map(|eyes| eyes.iter()))
                .map(squeeze_rng),
        )
        .await?;
        Ok(())
    }

    pub async fn state_check(sessions: BothEyes<&HawkSession>) -> Result<()> {
        let (left_state, right_state) = join!(
            HawkSession::state_check_side(sessions[LEFT]),
            HawkSession::state_check_side(sessions[RIGHT]),
        );

        let left_state = left_state?;
        let right_state = right_state?;
        left_state.check_left_vs_right(&right_state)?;
        Ok(())
    }

    async fn state_check_side(session: &HawkSession) -> Result<StateChecksum> {
        let my_state = session.checksum().await;

        let (prev_state, next_state) = {
            let mut store = session.aby3_store.write().await;
            let net = &mut store.session.network_session;

            let my_msg = || NetworkValue::StateChecksum(my_state.clone());
            net.send_prev(my_msg()).await?;
            net.send_next(my_msg()).await?;

            let decode = |msg| match msg {
                Ok(NetworkValue::StateChecksum(c)) => Ok(c),
                Ok(other) => {
                    tracing::error!("Unexpected message variant in state check: {:?}", other);
                    Err(eyre!("Could not deserialize state checksum"))
                }
                Err(e) => {
                    tracing::error!("Network receive error in state check: {e}");
                    Err(e)
                }
            };
            let prev = decode(net.receive_prev().await)?;
            let next = decode(net.receive_next().await)?;
            (prev, next)
        };

        if prev_state != my_state || next_state != my_state {
            if prev_state.irises != my_state.irises || next_state.irises != my_state.irises {
                if let Err(e) = Self::debug_iris_diff(
                    session,
                    my_state.irises,
                    prev_state.irises,
                    next_state.irises,
                )
                .await
                {
                    tracing::warn!("Iris diff debug protocol failed: {e}");
                }
            }
            bail!(
                "Party states have diverged: my_state={my_state:?} prev_state={prev_state:?} next_state={next_state:?}"
            );
        }
        Ok(my_state)
    }

    /// When iris checksums diverge, binary-search the serial-ID space using
    /// prefix sums to find up to K specific serial IDs that differ between
    /// parties.
    ///
    /// Uses [`IrisDiffSearch`] for the core algorithm; this method only adds
    /// the networking (prefix-sum construction, global-range agreement, and
    /// per-round hash exchange).
    async fn debug_iris_diff(
        session: &HawkSession,
        my_iris_hash: u64,
        prev_iris_hash: u64,
        next_iris_hash: u64,
    ) -> Result<()> {
        const MAX_DIFF_SAMPLES: usize = 64;

        tracing::warn!("Iris checksums differ — starting binary-search debug protocol");

        // ── build prefix sums locally ────────────────────────────────────
        let storage = session.aby3_store.read().await.storage.clone();
        let prefix = storage.read().await.prefix_sums();
        let my_len = (prefix.len() - 1) as u32;

        // ── agree on the global serial-ID range ──────────────────────────
        let global_hi = {
            let msg = NetworkValue::RingElement64(RingElement(my_len as u64));
            let mut store = session.aby3_store.write().await;
            let net = &mut store.session.network_session;
            net.send_prev(msg.clone()).await?;
            net.send_next(msg).await?;
            let decode = |msg| match msg {
                Ok(NetworkValue::RingElement64(r)) => Ok(r.0 as u32),
                other => Err(eyre!("Expected RingElement64 for len: {other:?}")),
            };
            let prev_len = decode(net.receive_prev().await)?;
            let next_len = decode(net.receive_next().await)?;
            my_len.max(prev_len).max(next_len)
        };

        if global_hi == 0 {
            tracing::warn!("All iris stores are empty, nothing to diff");
            return Ok(());
        }

        let range_hash = |lo: u32, hi: u32| -> u64 {
            let lo_c = lo.min(my_len) as usize;
            let hi_c = hi.min(my_len) as usize;
            prefix[hi_c].wrapping_sub(prefix[lo_c])
        };

        // ── drive the binary search ──────────────────────────────────────
        let mut search = IrisDiffSearch::new(
            global_hi,
            [my_iris_hash, prev_iris_hash, next_iris_hash],
            MAX_DIFF_SAMPLES,
        );

        while let Some((to_split, my_lefts)) = search.prepare_round(range_hash) {
            let (prev_lefts, next_lefts) = {
                let packed: Vec<RingElement<u64>> =
                    my_lefts.iter().map(|&h| RingElement(h)).collect();
                let msg = NetworkValue::VecRing64(packed);
                let mut store = session.aby3_store.write().await;
                let net = &mut store.session.network_session;
                net.send_prev(msg.clone()).await?;
                net.send_next(msg).await?;
                let decode = |msg| match msg {
                    Ok(NetworkValue::VecRing64(v)) => {
                        Ok(v.into_iter().map(|r| r.0).collect::<Vec<u64>>())
                    }
                    other => Err(eyre!("Expected VecRing64: {other:?}")),
                };
                (
                    decode(net.receive_prev().await)?,
                    decode(net.receive_next().await)?,
                )
            };

            if prev_lefts.len() != my_lefts.len() || next_lefts.len() != my_lefts.len() {
                bail!(
                    "Debug protocol desync: expected {} hashes, got prev={}, next={}",
                    my_lefts.len(),
                    prev_lefts.len(),
                    next_lefts.len()
                );
            }

            search.complete_round(to_split, &my_lefts, &prev_lefts, &next_lefts);
        }

        // ── log findings ─────────────────────────────────────────────────
        let found = search.into_found();
        let irises = storage.read().await;
        tracing::warn!(
            "Iris diff found {} differing serial IDs (cap {MAX_DIFF_SAMPLES}):",
            found.len(),
        );
        for &sid in &found {
            match irises.get_current_version(sid) {
                Some(version) => {
                    let vid = VectorId::new(sid, version);
                    tracing::warn!("  serial_id={sid}: my entry = {vid}");
                }
                None => {
                    tracing::warn!("  serial_id={sid}: absent from my store");
                }
            }
        }
        if found.len() >= MAX_DIFF_SAMPLES {
            tracing::warn!("  … capped at {MAX_DIFF_SAMPLES} samples, more may exist");
        }

        Ok(())
    }

    async fn checksum(&self) -> StateChecksum {
        StateChecksum {
            irises: self.aby3_store.read().await.checksum().await,
            graph: self.graph_store.read().await.checksum(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::hawkers::shared_irises::SharedIrises;
    use itertools::Itertools;

    #[test]
    fn test_set_hash() {
        let mut digests = vec![];

        let a = 12_u64;
        let b = VectorId::from_serial_id(34);
        let c = (a, &vec![b; 10]);

        let mut set_hash = SetHash::default();
        digests.push(set_hash.checksum());

        set_hash.add_unordered(a);
        digests.push(set_hash.checksum());

        set_hash.add_unordered(b);
        digests.push(set_hash.checksum());

        set_hash.add_unordered(c);
        digests.push(set_hash.checksum());

        assert!(digests.iter().all_unique());

        let different_order = {
            let mut set_hash = SetHash::default();
            set_hash.add_unordered(c);
            set_hash.add_unordered(a);
            set_hash.remove(c);
            set_hash.add_unordered(c);
            set_hash.add_unordered(b);
            set_hash.checksum()
        };
        assert_eq!(digests.pop().unwrap(), different_order);

        set_hash.remove(c);
        assert_eq!(digests.pop().unwrap(), set_hash.checksum());

        set_hash.remove(b);
        assert_eq!(digests.pop().unwrap(), set_hash.checksum());

        set_hash.remove(a);
        assert_eq!(digests.pop().unwrap(), set_hash.checksum());
        assert_eq!(SetHash::default().checksum(), set_hash.checksum());
    }

    #[test]
    fn test_prefix_sums_consistent_with_set_hash() {
        let mut store: SharedIrises<u8> = SharedIrises::default();

        // Insert some entries with gaps and different versions.
        store.insert(VectorId::from_serial_id(1), 0);
        store.insert(VectorId::from_serial_id(3), 0);
        store.insert(VectorId::new(5, 2), 0);
        store.insert(VectorId::from_serial_id(10), 0);

        let prefix = store.prefix_sums();

        // Total prefix sum should equal the store's set_hash checksum.
        let total = *prefix.last().unwrap();
        assert_eq!(total, store.set_hash.checksum());

        // Range [0, len) should also equal the total.
        let len = store.get_points().len() as u32;
        let full_range = prefix[len as usize].wrapping_sub(prefix[0]);
        assert_eq!(full_range, total);

        // A sub-range covering only serial_id=3 should match just that
        // entry's hash.
        let range_3 = prefix[4].wrapping_sub(prefix[3]);
        assert_eq!(range_3, SetHash::hash(VectorId::from_serial_id(3)));

        // An empty sub-range (no entries) should be 0.
        let range_empty = prefix[3].wrapping_sub(prefix[2]); // serial_id=2: no entry
        assert_eq!(range_empty, 0);
    }

    // ── helpers for IrisDiffSearch tests ──────────────────────────────

    /// Build a clamped range-hash closure for a store.
    fn range_hash_fn(store: &SharedIrises<u8>) -> impl Fn(u32, u32) -> u64 {
        let prefix = store.prefix_sums();
        let len = (prefix.len() - 1) as u32;
        move |lo: u32, hi: u32| -> u64 {
            let lo_c = lo.min(len) as usize;
            let hi_c = hi.min(len) as usize;
            prefix[hi_c].wrapping_sub(prefix[lo_c])
        }
    }

    /// Run the full binary search for 3 parties and return the sorted
    /// diff serial-IDs that party 0 would report.
    fn run_diff(stores: &[SharedIrises<u8>; 3], max_samples: usize) -> Vec<u32> {
        let rh0 = range_hash_fn(&stores[0]);
        let rh1 = range_hash_fn(&stores[1]);
        let rh2 = range_hash_fn(&stores[2]);

        let lens: Vec<u32> = stores.iter().map(|s| s.get_points().len() as u32).collect();
        let global_hi = *lens.iter().max().unwrap();

        let hashes: [u64; 3] = [
            stores[0].set_hash.checksum(),
            stores[1].set_hash.checksum(),
            stores[2].set_hash.checksum(),
        ];

        let mut search = IrisDiffSearch::new(global_hi, hashes, max_samples);

        // hashes = [party0, party1, party2] — prev=party1, next=party2.
        while let Some((to_split, my_lefts)) = search.prepare_round(&rh0) {
            let prev_lefts: Vec<u64> = to_split
                .iter()
                .map(|r| {
                    let mid = r.lo + (r.hi - r.lo) / 2;
                    rh1(r.lo, mid)
                })
                .collect();
            let next_lefts: Vec<u64> = to_split
                .iter()
                .map(|r| {
                    let mid = r.lo + (r.hi - r.lo) / 2;
                    rh2(r.lo, mid)
                })
                .collect();
            search.complete_round(to_split, &my_lefts, &prev_lefts, &next_lefts);
        }

        search.into_found()
    }

    // ── actual tests ─────────────────────────────────────────────────

    #[test]
    fn test_diff_identical_stores() {
        let mut stores: [SharedIrises<u8>; 3] = Default::default();
        for sid in 1..=100 {
            for s in &mut stores {
                s.insert(VectorId::from_serial_id(sid), 0);
            }
        }
        // All identical → nothing found.
        let found = run_diff(&stores, 64);
        assert!(found.is_empty(), "expected no diffs, got {found:?}");
    }

    #[test]
    fn test_diff_single_version_mismatch() {
        let mut stores: [SharedIrises<u8>; 3] = Default::default();
        for sid in 1..=100 {
            for s in &mut stores {
                s.insert(VectorId::from_serial_id(sid), 0);
            }
        }
        // Party 0 updates serial_id=42 to version 1.
        stores[0].insert(VectorId::new(42, 1), 0);

        let found = run_diff(&stores, 64);
        assert_eq!(found, vec![42], "expected diff at serial_id=42");
    }

    #[test]
    fn test_diff_extra_entry_in_one_party() {
        let mut stores: [SharedIrises<u8>; 3] = Default::default();
        for sid in 1..=50 {
            for s in &mut stores {
                s.insert(VectorId::from_serial_id(sid), 0);
            }
        }
        // Party 2 has an extra entry.
        stores[2].insert(VectorId::from_serial_id(51), 0);

        let found = run_diff(&stores, 64);
        assert_eq!(found, vec![51]);
    }

    #[test]
    fn test_diff_multiple_scattered() {
        let mut stores: [SharedIrises<u8>; 3] = Default::default();
        for sid in 1..=1000 {
            for s in &mut stores {
                s.insert(VectorId::from_serial_id(sid), 0);
            }
        }

        // Three differences spread across the range.
        stores[0].insert(VectorId::new(7, 1), 0); // version mismatch
        stores[1].insert(VectorId::from_serial_id(1001), 0); // extra in party 1
        stores[2].insert(VectorId::new(500, 3), 0); // version mismatch

        let found = run_diff(&stores, 64);
        assert_eq!(found.len(), 3);
        assert!(found.contains(&7));
        assert!(found.contains(&500));
        assert!(found.contains(&1001));
    }

    #[test]
    fn test_diff_different_store_lengths() {
        let mut stores: [SharedIrises<u8>; 3] = Default::default();
        // Party 0: serial_ids 1..=10
        for sid in 1..=10 {
            stores[0].insert(VectorId::from_serial_id(sid), 0);
        }
        // Party 1: serial_ids 1..=10
        for sid in 1..=10 {
            stores[1].insert(VectorId::from_serial_id(sid), 0);
        }
        // Party 2: serial_ids 1..=15 (5 extra)
        for sid in 1..=15 {
            stores[2].insert(VectorId::from_serial_id(sid), 0);
        }

        let found = run_diff(&stores, 64);
        assert_eq!(found, vec![11, 12, 13, 14, 15]);
    }

    #[test]
    fn test_diff_respects_sample_cap() {
        let mut stores: [SharedIrises<u8>; 3] = Default::default();
        // Party 0 has 200 entries, parties 1 and 2 have none.
        for sid in 1..=200 {
            stores[0].insert(VectorId::from_serial_id(sid), 0);
        }

        let cap = 10;
        let found = run_diff(&stores, cap);
        assert_eq!(found.len(), cap, "should cap at {cap}, got {}", found.len());
        // All returned IDs should be valid serial_ids in [1, 200].
        for &sid in &found {
            assert!((1..=200).contains(&sid));
        }
    }

    #[test]
    fn test_diff_adjacent_entries() {
        let mut stores: [SharedIrises<u8>; 3] = Default::default();
        for sid in 1..=100 {
            for s in &mut stores {
                s.insert(VectorId::from_serial_id(sid), 0);
            }
        }
        // Two adjacent differences.
        stores[0].insert(VectorId::new(50, 1), 0);
        stores[0].insert(VectorId::new(51, 1), 0);

        let found = run_diff(&stores, 64);
        assert_eq!(found, vec![50, 51]);
    }

    #[test]
    fn test_diff_all_three_parties_differ() {
        let mut stores: [SharedIrises<u8>; 3] = Default::default();
        for sid in 1..=100 {
            for s in &mut stores {
                s.insert(VectorId::from_serial_id(sid), 0);
            }
        }
        // All three parties have different versions for serial_id=10.
        stores[0].insert(VectorId::new(10, 1), 0);
        stores[1].insert(VectorId::new(10, 2), 0);
        // stores[2] keeps version 0

        let found = run_diff(&stores, 64);
        assert_eq!(found, vec![10]);
    }
}
