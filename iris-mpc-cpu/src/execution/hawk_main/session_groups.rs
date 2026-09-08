use crate::execution::hawk_main::HAWK_BASE_ROTATIONS_MASK;

use super::{BothEyes, BothOrient, HawkSession, Orientation, LEFT, RIGHT};

pub struct SessionGroups {
    pub for_search: BothOrient<BothEyes<Vec<HawkSession>>>,
    pub for_intra_batch: BothOrient<BothEyes<Vec<HawkSession>>>,
}

impl SessionGroups {
    // For each request, we may use parallel sessions for:
    // both orientations, both eyes, search+intra_batch, rotations.
    pub const N_SESSIONS_PER_REQUEST: usize =
        2 * 2 * 2 * (HAWK_BASE_ROTATIONS_MASK.count_ones() as usize);

    /// Number of sessions available to one search orientation and eye.
    ///
    /// The network allocates `N_SESSIONS_PER_REQUEST` sessions for every unit
    /// of request parallelism. Splitting by eye and then into the two search
    /// and two intra-batch groups leaves one session per configured base
    /// rotation. This is also the operational full-scan concurrency knob.
    pub const fn search_sessions_per_group(request_parallelism: usize) -> usize {
        request_parallelism * HAWK_BASE_ROTATIONS_MASK.count_ones() as usize
    }

    // Group the sessions per orientation, eye, and search+intra_batch.
    pub fn new(sessions: BothEyes<Vec<HawkSession>>) -> Self {
        let [left, right] = sessions;
        let [l0, l1, l2, l3] = split_in_four(left);
        let [r0, r1, r2, r3] = split_in_four(right);
        Self {
            for_search: [[l0, r0], [l1, r1]],
            for_intra_batch: [[l2, r2], [l3, r3]],
        }
    }

    // This takes &mut to enforce that it is not used in parallel with other methods.
    pub fn for_state_check(&mut self) -> BothEyes<&HawkSession> {
        [&self.for_search[0][LEFT][0], &self.for_search[0][RIGHT][0]]
    }

    pub fn for_mutations(&mut self, orient: Orientation) -> &BothEyes<Vec<HawkSession>> {
        &self.for_search[orient as usize]
    }

    pub fn for_search(&self, orient: Orientation) -> &BothEyes<Vec<HawkSession>> {
        &self.for_search[orient as usize]
    }

    pub fn for_intra_batch(&self, orient: Orientation) -> BothEyes<Vec<HawkSession>> {
        self.for_intra_batch[orient as usize].clone()
    }
}

fn split_in_four<T>(mut a: Vec<T>) -> [Vec<T>; 4] {
    let n = a.len();
    assert!(
        n.is_multiple_of(4),
        "Expected length to be divisible by 4, got {n}"
    );
    let quarter = n / 4;

    let d = a.split_off(quarter * 3);
    let c = a.split_off(quarter * 2);
    let b = a.split_off(quarter);
    [a, b, c, d]
}

#[cfg(test)]
mod tests {
    use super::SessionGroups;

    #[test]
    fn request_parallelism_maps_to_search_session_count() {
        const REQUEST_PARALLELISM: usize = 11;

        // The network creates 24 sessions per request group. Alternating
        // left/right assignment leaves 12 per eye, and quartering by
        // orientation and search/intra-batch leaves three search sessions.
        assert_eq!(SessionGroups::N_SESSIONS_PER_REQUEST, 24);
        let sessions_per_eye = REQUEST_PARALLELISM * SessionGroups::N_SESSIONS_PER_REQUEST / 2;
        let groups = super::split_in_four(vec![(); sessions_per_eye]);
        assert!(groups
            .iter()
            .all(|group| group.len() == 3 * REQUEST_PARALLELISM));
        assert_eq!(
            SessionGroups::search_sessions_per_group(REQUEST_PARALLELISM),
            33
        );
    }
}
