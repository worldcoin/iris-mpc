use super::{BothEyes, InsertPlan, MapEdges, VecEdges, VecRequests, VectorId, LEFT, RIGHT};
use itertools::{izip, Itertools};
use std::{collections::HashMap, iter::repeat};

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
    pub fn new(plans: &BothEyes<VecRequests<InsertPlan>>) -> Self {
        // Join the results of both eyes into results per eye pair.
        Self(
            izip!(&plans[LEFT], &plans[RIGHT])
                .map(|(left, right)| Step1::new([left, right]))
                .collect_vec(),
        )
    }

    pub fn missing_vector_ids(&self) -> VecRequests<BothEyes<VecEdges<VectorId>>> {
        self.0
            .iter()
            .map(|step| [LEFT, RIGHT].map(|side| step.missing_vector_ids(side)))
            .collect_vec()
    }

    pub fn step2(self, missing_is_match: &VecRequests<BothEyes<MapEdges<bool>>>) -> BatchStep2 {
        assert_eq!(self.0.len(), missing_is_match.len());
        BatchStep2(
            izip!(self.0, missing_is_match)
                .map(|(step, missing_is_match)| step.step2(missing_is_match))
                .collect_vec(),
        )
    }
}

#[derive(Default)]
struct Step1 {
    inner_join: VecEdges<(VectorId, BothEyes<bool>)>,
    anti_join:  BothEyes<VecEdges<(VectorId, bool)>>,
}

impl Step1 {
    fn new(plans: BothEyes<&InsertPlan>) -> Step1 {
        let mut full_join: MapEdges<BothEyes<Option<bool>>> = HashMap::new();

        for (side, plan) in izip!([LEFT, RIGHT], plans) {
            let is_match = repeat(true).take(plan.match_count()).chain(repeat(false));

            for (vector_id, is_match) in izip!(plan.nearest_neighbors(), is_match) {
                full_join.entry(vector_id).or_default()[side] = Some(is_match);
            }
        }

        let mut step1 = Step1::default();
        step1.inner_join.reserve(full_join.len());

        for (vector_id, is_match_lr) in full_join {
            match is_match_lr {
                [Some(l), Some(r)] if l || r => step1.inner_join.push((vector_id, [l, r])),
                [Some(l), None] if l => step1.anti_join[LEFT].push((vector_id, l)),
                [None, Some(r)] if r => step1.anti_join[RIGHT].push((vector_id, r)),
                _ => {}
            }
        }

        step1
    }

    fn missing_vector_ids(&self, side: usize) -> VecEdges<VectorId> {
        let other_side = 1 - side;
        self.anti_join[other_side]
            .iter()
            .map(|(id, _)| *id)
            .collect_vec()
    }

    fn step2(self, missing_is_match: &BothEyes<MapEdges<bool>>) -> Step2 {
        let mut step2 = Step2 {
            full_join: self.inner_join,
        };

        for (id, left) in &self.anti_join[LEFT] {
            if let Some(right) = missing_is_match[RIGHT].get(id) {
                step2.full_join.push((*id, [*left, *right]));
            }
        }

        for (id, right) in &self.anti_join[RIGHT] {
            if let Some(left) = missing_is_match[LEFT].get(id) {
                step2.full_join.push((*id, [*left, *right]));
            }
        }

        step2
    }
}

pub struct BatchStep2(VecRequests<Step2>);

impl BatchStep2 {
    pub fn is_matches(&self) -> VecRequests<bool> {
        self.0.iter().map(Step2::is_match).collect_vec()
    }
}

struct Step2 {
    full_join: VecEdges<(VectorId, BothEyes<bool>)>,
}

impl Step2 {
    /// *AND* policy: only match, if both eyes match (like `mergeDbResults`).
    /// TODO: Account for rotated and mirrored versions.
    fn is_match(&self) -> bool {
        self.full_join.iter().any(|(_, [l, r])| *l && *r)
    }
}
