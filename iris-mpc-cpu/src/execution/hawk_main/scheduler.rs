use itertools::Itertools;

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
    pub i_request: usize,
    pub i_rotation: usize,
}

/// Enumerate all combinations of eye sides, requests, and rotations.
/// Distribute the tasks over a number of sessions.
pub fn schedule(
    n_sessions: usize,
    n_eyes: usize,
    n_requests: usize,
    n_rotations: usize,
) -> Vec<Batch> {
    let n_tasks = n_requests * n_rotations;
    let batch_size = n_tasks / n_sessions;
    let rest_size = n_tasks % n_sessions;

    (0..n_eyes)
        .flat_map(|i_eye| {
            let mut task_iter = (0..n_rotations).flat_map(|i_rotation| {
                range_forward_backward(n_requests).map(move |i_request| Task {
                    i_request,
                    i_rotation,
                })
            });

            (0..n_sessions).map(move |i_session| {
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
    fn test_schedule() {
        test_schedule_impl(1, 0);
        test_schedule_impl(1, 1);
        test_schedule_impl(1, 2);
        test_schedule_impl(10, 1);
        test_schedule_impl(1, 10);
        test_schedule_impl(7, 10);
        test_schedule_impl(10, 30);
        test_schedule_impl(10, 97);
    }

    fn test_schedule_impl(n_sessions: usize, n_requests: usize) {
        let n_eyes = 2;
        let n_batches = n_eyes * n_sessions;
        let n_rotations = ROTATIONS;
        let n_tasks = n_eyes * n_requests * n_rotations;

        let batches = schedule(n_sessions, n_eyes, n_requests, n_rotations);
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
