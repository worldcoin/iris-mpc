use crate::shares::share::DistanceShare;
use std::hash::{Hash, Hasher};

pub type DistanceBundle1D = Vec<DistanceShare<u16>>;
pub type LiftedDistanceBundle1D = Vec<DistanceShare<u32>>;
pub type MinLiftedDistance1D = DistanceShare<u32>;

pub struct AnonStats1DMapping {
    stats: Vec<(i64, DistanceBundle1D)>,
}

impl AnonStats1DMapping {
    pub fn new(mut stats: Vec<(i64, DistanceBundle1D)>) -> Self {
        // ensure it is sorted by id
        stats.sort_by_key(|(id, _)| *id);
        AnonStats1DMapping { stats }
    }

    pub fn len(&self) -> usize {
        self.stats.len()
    }
    pub fn is_empty(&self) -> bool {
        self.stats.is_empty()
    }
    pub fn get_ranges(&self) -> (i64, i64) {
        if self.stats.is_empty() {
            (0, 0)
        } else {
            (self.stats.first().unwrap().0, self.stats.last().unwrap().0)
        }
    }
    pub fn get_id_hash(&self) -> u64 {
        let mut hasher = siphasher::sip::SipHasher13::new();
        for (id, _) in &self.stats {
            id.hash(&mut hasher);
        }
        hasher.finish()
    }

    pub fn into_bundles(self) -> Vec<DistanceBundle1D> {
        self.stats.into_iter().map(|(_, bundle)| bundle).collect()
    }
}
