use std::hash::{Hash, Hasher};

use iris_mpc_common::job::Eye;
use iris_mpc_cpu::{execution::hawk_main::Orientation, shares::share::DistanceShare};

pub type DistanceBundle1D = Vec<DistanceShare<u16>>;
pub type LiftedDistanceBundle1D = Vec<DistanceShare<u32>>;
pub type MinLiftedDistance1D = DistanceShare<u32>;

pub type DistanceBundle2D = (DistanceBundle1D, DistanceBundle1D);
pub type LiftedDistanceBundle2D = (LiftedDistanceBundle1D, LiftedDistanceBundle1D);
pub type MinLiftedDistance2D = (DistanceShare<u32>, DistanceShare<u32>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnonStatsOrigin {
    pub side: Option<Eye>,
    pub orientation: Orientation,
    pub context: AnonStatsContext,
}

impl From<AnonStatsOrigin> for i16 {
    fn from(origin: AnonStatsOrigin) -> Self {
        let side_val: i16 = match origin.side {
            Some(Eye::Left) => 0,
            Some(Eye::Right) => 1,
            None => 2,
        };
        let orientation_val: i16 = match origin.orientation {
            Orientation::Normal => 0,
            Orientation::Mirror => 1,
        };
        // 2 bits side + 1 bit orientation + 8 bits context
        (side_val << 9) | (orientation_val << 8) | (origin.context as i16)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AnonStatsContext {
    GPU = 0,
    HNSW = 1,
}

pub struct AnonStatsMapping<T> {
    stats: Vec<(i64, T)>,
}

impl<T> AnonStatsMapping<T> {
    pub fn new(mut stats: Vec<(i64, T)>) -> Self {
        // ensure it is sorted by id
        stats.sort_by_key(|(id, _)| *id);
        AnonStatsMapping { stats }
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

    pub fn into_bundles(self) -> Vec<T> {
        self.stats.into_iter().map(|(_, bundle)| bundle).collect()
    }

    pub fn truncate(&mut self, new_size: usize) {
        self.stats.truncate(new_size);
    }
}
