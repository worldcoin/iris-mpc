use super::{
    rot::{Rotations, VecRotationSupport},
    BothEyes, VecRequests,
};
use eyre::{eyre, Result};
use futures::future::JoinAll;
use itertools::Itertools;
use std::{collections::HashMap, future::Future};
use tokio::{sync::mpsc::UnboundedReceiver, task::JoinError};

const N_EYES: usize = 2;

/// A schedule is a collections of batches to process in parallel.
pub struct Schedule {
    n_sessions: usize,
    n_requests: usize,
    n_rotations: usize,
}

/// A batch is a list of tasks to do within the same unit of parallelism.
/// Our unit of parallelism is one session of one eye side.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Batch {
    pub i_session: usize,
    pub i_eye: usize,
    pub tasks: Vec<Task>,
}

/// A task within a batch is something to do with one rotation of one request.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Task {
    pub i_eye: usize,
    pub i_request: usize,
    pub i_rotation: usize,
    pub is_central: bool,
}

impl Task {
    pub fn id(&self) -> TaskId {
        (self.i_eye, self.i_request, self.i_rotation)
    }
}

pub type TaskId = (usize, usize, usize);

impl Schedule {
    pub fn new(n_sessions: usize, n_requests: usize, n_rotations: usize) -> Self {
        Self {
            n_sessions,
            n_requests,
            n_rotations,
        }
    }

    /// Enumerate all combinations of eye sides, requests, and rotations.
    /// Distribute the tasks over a number of sessions.
    /// Note: Should be used exclusively for intra_match_is_batch
    /// as it is optimized for its logic
    pub fn intra_match_batches(&self) -> Vec<Batch> {
        let n_tasks = self.n_requests * self.n_rotations;
        let batch_size = n_tasks / self.n_sessions;
        let rest_size = n_tasks % self.n_sessions;

        (0..N_EYES)
            .flat_map(|i_eye| {
                let mut task_iter = (0..self.n_rotations).flat_map(move |i_rotation| {
                    range_forward_backward(self.n_requests).map(move |i_request| Task {
                        i_eye,
                        i_request,
                        i_rotation,
                        is_central: (i_rotation == self.n_rotations / 2),
                    })
                });

                (0..self.n_sessions).map(move |i_session| {
                    // Some sessions get one more task if n_sessions does not divide n_tasks.
                    let one_more = (i_session < rest_size) as usize;

                    let tasks = task_iter.by_ref().take(batch_size + one_more).collect_vec();

                    Batch {
                        i_eye,
                        i_session,
                        tasks,
                    }
                })
            })
            .collect_vec()
    }

    /// Enumerate all combinations of eye sides, requests, and rotations.
    /// Distribute the tasks over a number of sessions.
    /// This method is search-aware and weighs central rotations as higher workloads than non-central ones
    pub fn search_batches(&self) -> Vec<Batch> {
        let n_tasks = self.n_requests * self.n_rotations;
        let batch_size = n_tasks / self.n_sessions;
        let rest_size = n_tasks % self.n_sessions;

        (0..N_EYES)
            .flat_map(|i_eye| {
                // Iterate requests first and rotations second (contrast with intra_match_batches)
                // This ensures that heavier center tasks are grouped with their (lighter) rotations
                // This order also opens up the possibility of better memory access patterns, assuming rotations
                // of a fixed iris behave similarly
                let mut task_iter = (0..self.n_requests).flat_map(move |i_request| {
                    (0..self.n_rotations).map(move |i_rotation| Task {
                        i_eye,
                        i_request,
                        i_rotation,
                        is_central: (i_rotation == self.n_rotations / 2),
                    })
                });

                (0..self.n_sessions).map(move |i_session| {
                    // Some sessions get one more task if n_sessions does not divide n_tasks.
                    let one_more = (i_session < rest_size) as usize;

                    let tasks = task_iter.by_ref().take(batch_size + one_more).collect_vec();

                    Batch {
                        i_eye,
                        i_session,
                        tasks,
                    }
                })
            })
            .collect_vec()
    }

    pub fn organize_results<T, ROT: Rotations>(
        &self,
        mut results: HashMap<TaskId, T>,
    ) -> Result<BothEyes<VecRequests<VecRotationSupport<T, ROT>>>> {
        let [l, r] = [0, 1].map(|i_eye| {
            (0..self.n_requests)
                .map(|i_request| {
                    (0..self.n_rotations)
                        .map(|i_rotation| {
                            let task = Task {
                                i_eye,
                                i_request,
                                i_rotation,
                                is_central: (i_rotation == self.n_rotations / 2),
                            };
                            results
                                .remove(&task.id())
                                .ok_or_else(|| eyre!("missing result {:?}", task))
                        })
                        .collect::<Result<Vec<_>>>()
                        .map(VecRotationSupport::from)
                })
                .collect::<Result<Vec<_>>>()
        });
        Ok([l?, r?])
    }
}

pub async fn parallelize<F, T>(tasks: impl Iterator<Item = F>) -> Result<Vec<T>>
where
    F: Future<Output = Result<T>> + Send + 'static,
    F::Output: Send + 'static,
{
    tasks
        .map(tokio::spawn)
        .collect::<JoinAll<_>>()
        .await
        .into_iter()
        .collect::<Result<Result<Vec<T>>, JoinError>>()?
}

pub async fn collect_results<T>(
    mut rx: UnboundedReceiver<(TaskId, T)>,
) -> Result<HashMap<TaskId, T>> {
    rx.close();

    let mut results = HashMap::new();
    while let Some((task_id, result)) = rx.recv().await {
        results.insert(task_id, result);
    }
    Ok(results)
}

/// Like (0..n) but alternating between forward and backward iteration.
/// The work of a task can depend on the index in `intra_batch_is_match`.
/// This helps to distribute the indexes fairly among the sessions.
fn range_forward_backward(n: usize) -> impl Iterator<Item = usize> {
    let forward = 0..n / 2;
    let backward = (n / 2..n).rev();
    forward.interleave(backward)
}

#[cfg(test)]
mod test {
    use super::*;
    use iris_mpc_common::ROTATIONS;

    #[test]
    fn test_intra_match_schedule() {
        for n_rotations in [1, ROTATIONS] {
            test_intra_match_schedule_impl(1, 0, n_rotations);
            test_intra_match_schedule_impl(1, 1, n_rotations);
            test_intra_match_schedule_impl(1, 2, n_rotations);
            test_intra_match_schedule_impl(10, 1, n_rotations);
            test_intra_match_schedule_impl(1, 10, n_rotations);
            test_intra_match_schedule_impl(7, 10, n_rotations);
            test_intra_match_schedule_impl(10, 30, n_rotations);
            test_intra_match_schedule_impl(10, 97, n_rotations);
        }
    }

    fn test_intra_match_schedule_impl(n_sessions: usize, n_requests: usize, n_rotations: usize) {
        let n_eyes = N_EYES;
        let n_batches = n_eyes * n_sessions;
        let n_tasks = n_eyes * n_requests * n_rotations;

        let batches = Schedule::new(n_sessions, n_requests, n_rotations).intra_match_batches();
        assert_eq!(batches.len(), n_batches);

        let count_tasks: usize = batches.iter().map(|b| b.tasks.len()).sum();
        assert_eq!(count_tasks, n_tasks);

        let unique_sessions = batches
            .iter()
            .map(|b| (b.i_eye, b.i_session))
            .unique()
            .count();
        assert_eq!(unique_sessions, n_batches);

        let unique_tasks = batches
            .iter()
            .flat_map(|b| {
                assert!(b.i_eye < n_eyes);
                assert!(b.i_session < n_sessions);

                b.tasks.iter().map(|t| {
                    assert!(t.i_request < n_requests);
                    assert!(t.i_rotation < n_rotations);

                    (b.i_eye, t.i_request, t.i_rotation)
                })
            })
            .unique()
            .count();
        assert_eq!(unique_tasks, n_tasks);
    }

    #[test]
    fn test_search_schedule() {
        for n_rotations in [1, ROTATIONS] {
            test_search_schedule_impl(1, 0, n_rotations);
            test_search_schedule_impl(1, 1, n_rotations);
            test_search_schedule_impl(1, 2, n_rotations);
            test_search_schedule_impl(10, 1, n_rotations);
            test_search_schedule_impl(1, 10, n_rotations);
            test_search_schedule_impl(7, 10, n_rotations);
            test_search_schedule_impl(10, 30, n_rotations);
            test_search_schedule_impl(10, 97, n_rotations);
        }
    }

    fn test_search_schedule_impl(n_sessions: usize, n_requests: usize, n_rotations: usize) {
        let n_eyes = N_EYES;
        let n_batches = n_eyes * n_sessions;
        let n_tasks = n_eyes * n_requests * n_rotations;

        let batches = Schedule::new(n_sessions, n_requests, n_rotations).search_batches();
        assert_eq!(batches.len(), n_batches);

        let count_tasks: usize = batches.iter().map(|b| b.tasks.len()).sum();
        assert_eq!(count_tasks, n_tasks);

        let unique_sessions = batches
            .iter()
            .map(|b| (b.i_eye, b.i_session))
            .unique()
            .count();
        assert_eq!(unique_sessions, n_batches);

        let unique_tasks = batches
            .iter()
            .flat_map(|b| {
                assert!(b.i_eye < n_eyes);
                assert!(b.i_session < n_sessions);

                b.tasks.iter().map(|t| {
                    assert!(t.i_request < n_requests);
                    assert!(t.i_rotation < n_rotations);

                    (b.i_eye, t.i_request, t.i_rotation)
                })
            })
            .unique()
            .count();
        assert_eq!(unique_tasks, n_tasks);

        // Check central-rotation load balance
        let minmax = batches
            .iter()
            .map(|batch| {
                batch
                    .tasks
                    .iter()
                    .map(|task| task.is_central as usize)
                    .sum::<usize>()
            })
            .minmax();
        let dif = match minmax {
            itertools::MinMaxResult::NoElements => 0,
            itertools::MinMaxResult::OneElement(_) => 0,
            itertools::MinMaxResult::MinMax(min, max) => max - min,
        };
        // The difference might be 2 in contrived cases, but it's
        // not worth addressing
        assert!(dif <= 1);
    }

    #[test]
    fn test_range_forward_backward() {
        assert!(range_forward_backward(0).collect_vec().is_empty());
        assert_eq!(range_forward_backward(1).collect_vec(), vec![0]);
        assert_eq!(
            range_forward_backward(7).collect_vec(),
            vec![0, 6, 1, 5, 2, 4, 3]
        );
        assert_eq!(
            range_forward_backward(8).collect_vec(),
            vec![0, 7, 1, 6, 2, 5, 3, 4]
        );
    }
}
