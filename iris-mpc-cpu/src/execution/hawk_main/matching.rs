use super::{
    rot::VecRots, BothEyes, InsertPlan, MapEdges, VecEdges, VecRequests, VectorId, LEFT, RIGHT,
};
use itertools::{izip, Itertools};
use std::collections::HashMap;

/// The matching algorithm follows these steps:
///
/// 1. Organize the results of the nearest neighbor search with
///    `BatchStep1::new`. Then:
///
///    1.a. Get the vectors found on only one side with `missing_vector_ids()`.
///    1.b. Fetch the other side and calculate their `is_match` with MPC.
///    1.c. Give this back to `step2(missing_is_match)`.
///
/// 2. `BatchStep2::is_matches`: Combine it all into the final match decisions.
pub struct BatchStep1(VecRequests<Step1>);

impl BatchStep1 {
    pub fn new(plans: &BothEyes<VecRequests<VecRots<InsertPlan>>>) -> Self {
        // Join the results of both eyes into results per eye pair.
        Self(
            izip!(&plans[LEFT], &plans[RIGHT])
                .map(|(left, right)| Step1::new([left, right]))
                .collect_vec(),
        )
    }

    pub fn missing_vector_ids(&self) -> BothEyes<VecRequests<VecEdges<VectorId>>> {
        [LEFT, RIGHT].map(|side| {
            self.0
                .iter()
                .map(|step| step.missing_vector_ids(side))
                .collect_vec()
        })
    }

    pub fn step2(self, missing_is_match: &BothEyes<VecRequests<MapEdges<bool>>>) -> BatchStep2 {
        assert_eq!(self.0.len(), missing_is_match[LEFT].len());
        assert_eq!(self.0.len(), missing_is_match[RIGHT].len());
        BatchStep2(
            izip!(self.0, &missing_is_match[LEFT], &missing_is_match[RIGHT])
                .map(|(step, missing_left, missing_right)| {
                    step.step2([missing_left, missing_right])
                })
                .collect_vec(),
        )
    }
}

struct Step1 {
    inner_join: VecEdges<(VectorId, BothEyes<bool>)>,
    anti_join: BothEyes<VecEdges<VectorId>>,
}

impl Step1 {
    fn new(results: BothEyes<&VecRots<InsertPlan>>) -> Step1 {
        let mut full_join: MapEdges<BothEyes<bool>> = HashMap::new();

        for (side, rotations) in izip!([LEFT, RIGHT], results) {
            // Merge matches from all rotations.
            for rotation in rotations.iter() {
                for vector_id in rotation.match_ids() {
                    full_join.entry(vector_id).or_default()[side] = true;
                }
            }
        }

        let mut step1 = Step1::with_capacity(full_join.len());

        for (vector_id, is_match_lr) in full_join {
            match is_match_lr {
                [true, true] => step1.inner_join.push((vector_id, [true, true])),
                [true, false] => step1.anti_join[LEFT].push(vector_id),
                [false, true] => step1.anti_join[RIGHT].push(vector_id),
                [false, false] => {}
            }
        }

        step1
    }

    fn with_capacity(capacity: usize) -> Self {
        Step1 {
            inner_join: Vec::with_capacity(capacity),
            anti_join: [
                Vec::with_capacity(capacity / 2),
                Vec::with_capacity(capacity / 2),
            ],
        }
    }

    fn missing_vector_ids(&self, side: usize) -> VecEdges<VectorId> {
        let other_side = 1 - side;
        self.anti_join[other_side].clone()
    }

    fn step2(self, missing_is_match: BothEyes<&MapEdges<bool>>) -> Step2 {
        let mut step2 = Step2 {
            full_join: self.inner_join,
        };

        for id in &self.anti_join[LEFT] {
            if let Some(right) = missing_is_match[RIGHT].get(id) {
                step2.full_join.push((*id, [true, *right]));
            }
        }

        for id in &self.anti_join[RIGHT] {
            if let Some(left) = missing_is_match[LEFT].get(id) {
                step2.full_join.push((*id, [*left, true]));
            }
        }

        step2
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchStep2(VecRequests<Step2>);

impl BatchStep2 {
    pub fn is_matches(&self) -> VecRequests<bool> {
        self.0.iter().map(Step2::is_match).collect_vec()
    }

    pub fn filter_map<F, OUT>(&self, f: F) -> VecRequests<VecEdges<OUT>>
    where
        F: Fn(&(VectorId, BothEyes<bool>)) -> Option<OUT>,
    {
        self.0.iter().map(|step| step.filter_map(&f)).collect_vec()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Step2 {
    full_join: VecEdges<(VectorId, BothEyes<bool>)>,
}

impl Step2 {
    /// *AND* policy: only match, if both eyes match (like `mergeDbResults`).
    /// TODO: Account for rotated and mirrored versions.
    fn is_match(&self) -> bool {
        self.full_join.iter().any(|(_, [l, r])| *l && *r)
    }

    fn filter_map<F, OUT>(&self, f: F) -> VecEdges<OUT>
    where
        F: Fn(&(VectorId, BothEyes<bool>)) -> Option<OUT>,
    {
        self.full_join.iter().filter_map(f).collect_vec()
    }
}
