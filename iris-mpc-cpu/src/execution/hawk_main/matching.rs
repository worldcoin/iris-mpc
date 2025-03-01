use super::{BothEyes, InsertPlan, VectorId};
use itertools::{izip, Itertools};
use std::{collections::HashMap, iter::repeat};

const LEFT: usize = 0;
const RIGHT: usize = 1;

pub struct BatchStep1(Vec<Step1>);

impl BatchStep1 {
    pub fn new(plans: &BothEyes<Vec<InsertPlan>>) -> Self {
        // Join the results of both eyes into results per eye pair.
        Self(
            izip!(&plans[LEFT], &plans[RIGHT])
                .map(|(left, right)| Step1::new([left, right]))
                .collect_vec(),
        )
    }

    pub fn is_matches(&self) -> Vec<bool> {
        self.0
            .iter()
            .map(|step1| step1.step2().is_match())
            .collect_vec()
    }
}

pub struct Step1 {
    neighbors:    BothEyes<Vec<VectorId>>,
    match_counts: BothEyes<usize>,
}

impl Step1 {
    pub fn new(plans: BothEyes<&InsertPlan>) -> Self {
        Self {
            neighbors:    [
                plans[LEFT].nearest_neighbors(),
                plans[RIGHT].nearest_neighbors(),
            ],
            match_counts: [plans[LEFT].match_count(), plans[RIGHT].match_count()],
        }
    }

    pub fn step2(&self) -> Step2 {
        let mut full_join: HashMap<VectorId, BothEyes<Option<bool>>> = HashMap::new();

        for side in [LEFT, RIGHT] {
            let is_match = repeat(true)
                .take(self.match_counts[side])
                .chain(repeat(false));

            for (vector_id, is_match) in izip!(&self.neighbors[side], is_match) {
                full_join.entry(*vector_id).or_default()[side] = Some(is_match);
            }
        }

        let mut step2 = Step2::default();

        for (vector_id, is_match_lr) in full_join {
            match is_match_lr {
                [Some(l), Some(r)] => step2.inner_join.push((vector_id, [l, r])),
                [Some(l), None] => step2.anti_join[LEFT].push((vector_id, l)),
                [None, Some(r)] => step2.anti_join[RIGHT].push((vector_id, r)),
                _ => unreachable!(),
            }
        }

        step2
    }
}

#[derive(Default)]
pub struct Step2 {
    inner_join: Vec<(VectorId, BothEyes<bool>)>,
    anti_join:  BothEyes<Vec<(VectorId, bool)>>,
}

impl Step2 {
    /// *AND* policy: only match, if both eyes match (like `mergeDbResults`).
    /// TODO: Account for rotated and mirrored versions.
    pub fn is_match(&self) -> bool {
        self.inner_join.iter().any(|(_, [l, r])| *l && *r)
    }
}
