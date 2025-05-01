use super::{
    rot::VecRots, BothEyes, InsertPlan, MapEdges, VecEdges, VecRequests, VectorId, LEFT, RIGHT,
};
use itertools::{chain, izip, Itertools};
use std::collections::HashMap;

/// Since the two separate HSNW for left and right return separate vectors of matching ids, we
/// cannot do the trivial AND/OR matching procedure from v2, since the other side might not have
/// considered that id at all. This however does not mean it would not match, so for all ids that
/// are given back for one side we do a manual comparison in the other side to get a full
/// left-right match pair. Only then do we continue to the final matching logic.
///
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
    pub fn new(
        plans: &BothEyes<VecRequests<VecRots<InsertPlan>>>,
        luc_ids: VecRequests<Vec<VectorId>>,
    ) -> Self {
        // Join the results of both eyes into results per eye pair.
        Self(
            izip!(&plans[LEFT], &plans[RIGHT], luc_ids)
                .map(|(left, right, luc)| Step1::new([left, right], luc))
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
    luc_ids: Vec<VectorId>,
}

impl Step1 {
    fn new(search_results: BothEyes<&VecRots<InsertPlan>>, luc_ids: Vec<VectorId>) -> Step1 {
        let mut full_join: MapEdges<BothEyes<bool>> = HashMap::new();

        for (side, rotations) in izip!([LEFT, RIGHT], search_results) {
            // Merge matches from all rotations.
            for rotation in rotations.iter() {
                for vector_id in rotation.match_ids() {
                    full_join.entry(vector_id).or_default()[side] = true;
                }
            }
        }

        let mut step1 = Step1::with_capacity(full_join.len());
        step1.luc_ids = luc_ids;

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
            luc_ids: Vec::new(),
        }
    }

    fn missing_vector_ids(&self, side: usize) -> VecEdges<VectorId> {
        let other_side = 1 - side;
        let anti_join = &self.anti_join[other_side];

        chain!(anti_join, &self.luc_ids).cloned().collect_vec()
    }

    fn step2(self, missing_is_match: BothEyes<&MapEdges<bool>>) -> Step2 {
        let luc_results = self
            .luc_ids
            .iter()
            .map(|id| {
                let is_match =
                    [LEFT, RIGHT].map(|side| *missing_is_match[side].get(id).unwrap_or(&false));
                (*id, is_match)
            })
            .collect_vec();

        let mut step2 = Step2 {
            full_join: self.inner_join,
            luc_results,
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
    luc_results: VecEdges<(VectorId, BothEyes<bool>)>,
}

impl Step2 {
    /// Search *AND* policy: only match if both eyes match (like `mergeDbResults`).
    ///
    /// LUC *OR* policy: "Local" irises match if either side matches.
    fn is_match(&self) -> bool {
        let search = self.full_join.iter().any(|(_, [l, r])| *l && *r);

        let luc = self.luc_results.iter().any(|(_, [l, r])| *l || *r);

        search || luc
    }

    fn filter_map<F, OUT>(&self, f: F) -> VecEdges<OUT>
    where
        F: Fn(&(VectorId, BothEyes<bool>)) -> Option<OUT>,
    {
        self.full_join.iter().filter_map(f).collect_vec()
    }
}
