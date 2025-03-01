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

    pub fn step2(&self) -> BatchStep2 {
        BatchStep2(self.0.iter().map(Step1::step2).collect_vec())
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

pub struct BatchStep2(Vec<Step2>);

impl BatchStep2 {
    pub fn missing_vector_ids(&self) -> Vec<BothEyes<Vec<VectorId>>> {
        self.0
            .iter()
            .map(|step2| [LEFT, RIGHT].map(|side| step2.missing_vector_ids(side).collect_vec()))
            .collect_vec()
    }

    pub fn step3(&self, other_is_match: &[BothEyes<HashMap<VectorId, bool>>]) -> BatchStep3 {
        assert_eq!(self.0.len(), other_is_match.len());
        BatchStep3(
            izip!(&self.0, other_is_match)
                .map(|(step2, other_is_match)| step2.step3(other_is_match))
                .collect_vec(),
        )
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

    fn missing_vector_ids(&self, side: usize) -> impl Iterator<Item = VectorId> + '_ {
        let other_side = 1 - side;
        self.anti_join[other_side]
            .iter()
            .filter(|(_, is_match)| *is_match)
            .map(|(id, _)| *id)
    }

    fn step3(&self, other_is_match: &BothEyes<HashMap<VectorId, bool>>) -> Step3 {
        let mut step3 = Step3 {
            inner_join: self.inner_join.clone(), // TODO: without clone.
        };

        for (id, left) in &self.anti_join[LEFT] {
            if let Some(right) = other_is_match[RIGHT].get(id) {
                step3.inner_join.push((*id, [*left, *right]));
            }
        }

        for (id, right) in &self.anti_join[RIGHT] {
            if let Some(left) = other_is_match[LEFT].get(id) {
                step3.inner_join.push((*id, [*left, *right]));
            }
        }

        step3
    }
}

pub struct BatchStep3(Vec<Step3>);

impl BatchStep3 {
    pub fn is_matches(&self) -> Vec<bool> {
        self.0.iter().map(Step3::is_match).collect_vec()
    }
}

struct Step3 {
    inner_join: Vec<(VectorId, BothEyes<bool>)>,
}

impl Step3 {
    /// *AND* policy: only match, if both eyes match (like `mergeDbResults`).
    /// TODO: Account for rotated and mirrored versions.
    fn is_match(&self) -> bool {
        self.inner_join.iter().any(|(_, [l, r])| *l && *r)
    }
}
