use eyre::Result;

use crate::hnsw::VectorStore;
/// An implementation of the Quickselect algorithm that works on indices and uses
/// an external oracle for comparisons.
pub struct Quickselect {
    left: usize,
    right: usize,
    kth_index: usize, // 0-based index of the element we're looking for
    perm: Vec<usize>,
    result: Option<usize>,
}

impl Quickselect {
    /// Creates a new Quickselect instance to find the k-th smallest element.
    /// `n`: The total number of elements (from 0 to n-1).
    /// `k`: The 1-based rank of the element to find (e.g., k=1 for smallest, k=n for largest).
    pub fn new(n: usize, k: usize) -> Self {
        assert!(k > 0 && k <= n, "k must be between 1 and n");
        Quickselect {
            left: 0,
            right: n,
            kth_index: k - 1, // Convert 1-based k to 0-based index
            perm: (0..n).collect::<Vec<_>>(),
            result: None,
        }
    }

    /// Returns true if the algorithm has found the k-th element.
    pub fn is_done(&self) -> bool {
        self.result.is_some()
    }

    /// Returns the k-th element's index if the algorithm is done.
    pub fn get_result(&self) -> Option<usize> {
        self.result
    }

    /// Generates the next set of comparisons needed.
    /// Compares elements in the current search space against a pivot.
    pub fn get_next_cmps(&self) -> Vec<(usize, usize)> {
        if self.is_done() {
            return Vec::new();
        }
        let pivot_idx = self.get_pivot_index();
        let pivot_val = self.perm[pivot_idx];

        (self.left..self.right)
            .filter(|&i| i != pivot_idx)
            .map(|i| (self.perm[i], pivot_val))
            .collect()
    }

    /// Processes the results of the comparisons and narrows the search space.
    pub fn step(&mut self, results: Vec<bool>) {
        if self.is_done() {
            return;
        }

        let expected_len = (self.right - self.left).saturating_sub(1);
        assert_eq!(
            results.len(),
            expected_len,
            "Mismatch between number of elements and results"
        );

        let pivot_idx = self.get_pivot_index();
        let pivot_val = self.perm[pivot_idx];

        let elements_with_results: Vec<(usize, bool)> = (self.left..self.right)
            .filter(|&i| i != pivot_idx)
            .map(|i| self.perm[i])
            .zip(results)
            .collect();

        let mut current_idx = self.left;

        // Place all elements that are less than the pivot.
        for &(val, is_less) in &elements_with_results {
            if is_less {
                self.perm[current_idx] = val;
                current_idx += 1;
            }
        }

        // The new index for the pivot is right after the "lesser" elements.
        let pivot_new_idx = current_idx;
        self.perm[pivot_new_idx] = pivot_val;
        current_idx += 1;

        // Place all elements that are greater than or equal to the pivot.
        for &(val, is_less) in &elements_with_results {
            if !is_less {
                self.perm[current_idx] = val;
                current_idx += 1;
            }
        }

        // Update state for next iteration
        match self.kth_index.cmp(&pivot_new_idx) {
            std::cmp::Ordering::Equal => {
                self.result = Some(self.perm[pivot_new_idx]);
            }
            std::cmp::Ordering::Less => {
                self.right = pivot_new_idx;
            }
            std::cmp::Ordering::Greater => {
                self.left = pivot_new_idx + 1;
            }
        }
    }

    /// Calculates the pivot index as the middle of the current interval
    fn get_pivot_index(&self) -> usize {
        self.left + (self.right - self.left) / 2
    }
}

/// Runs quickselect for `data` using `oracle` to evaluate batches of comparisons.
/// Returns a permutation `P` which satisfies `(i < k - 1) -> data[P[i]] <= data[P[k - 1]]`
/// and `i >= k - 1 -> data[P[i] >= data[P[k - 1]]`.
///
/// `oracle` should return true for pairs where `lhs < rhs` and false for others.
pub fn run_quickselect_test<T: Clone + Ord>(data: &[T], k: usize) -> Vec<usize>
where
{
    let n = data.len();
    let mut qs = Quickselect::new(n, k);

    // Loop until the algorithm finds the result
    while !qs.is_done() {
        let comparisons = qs
            .get_next_cmps()
            .iter()
            .map(|&(i, j)| (data[i].clone(), data[j].clone()))
            .collect::<Vec<_>>();

        let results = comparisons
            .iter()
            .map(|(lhs, rhs)| lhs < rhs)
            .collect::<Vec<bool>>();
        qs.step(results);
    }
    qs.perm
}

/// Runs quickselect for `data` using `oracle` to evaluate batches of comparisons.
/// Returns a permutation `P` which satisfies `(i < k - 1) -> data[P[i]] <= data[P[k - 1]]`
/// and `i >= k - 1 -> data[P[i] >= data[P[k - 1]]`.
///
/// `oracle` should return true for pairs where `lhs < rhs` and false for others.
pub async fn run_quickselect_with_store<V: VectorStore>(
    store: &mut V,
    data: &[V::DistanceRef],
    k: usize,
) -> Result<Vec<usize>>
where
{
    let n = data.len();
    let mut qs = Quickselect::new(n, k);

    // Loop until the algorithm finds the result
    while !qs.is_done() {
        let comparisons = qs
            .get_next_cmps()
            .iter()
            .map(|&(i, j)| (data[i].clone(), data[j].clone()))
            .collect::<Vec<_>>();

        let results = store.less_than_batch(&comparisons).await?;
        qs.step(results);
    }
    Ok(qs.perm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};

    fn get_expected<T: Ord + Clone>(data: &[T], k: usize) -> T {
        let mut sorted_data = data.to_vec();
        sorted_data.sort();
        sorted_data[k - 1].clone()
    }

    /// A helper to run the tests and perform all checks.
    fn run_and_verify<T: Ord + Clone + std::fmt::Debug>(data: &[T], k: usize) {
        let expected = get_expected(data, k);
        let final_perm = run_quickselect_test(data, k);

        let km1th = data[final_perm[k - 1]].clone();
        // 1. Check if the k-th value is correct.
        assert_eq!(km1th, expected, "The k-th value is incorrect");

        // All elements before the result must be less than or equal to it.
        for i in 0..(k - 1) {
            let current_val = &data[final_perm[i]];
            assert!(
                current_val <= &km1th,
                "Partition Fail: Element {:?} at perm index {} should be <= {:?} (k-th element)",
                current_val,
                i,
                km1th
            );
        }

        // All elements after the result must be greater than or equal to it.
        for i in k..final_perm.len() {
            let current_val = &data[final_perm[i]];
            assert!(
                current_val >= &km1th,
                "Partition Fail: Element {:?} at perm index {} should be >= {:?} (k-th element)",
                current_val,
                i,
                km1th
            );
        }
    }

    #[test]
    fn test_edge_case_find_minimum() {
        let data = vec![8, 1, 5, 9, 2, 0, 4];
        run_and_verify(&data, 1);
    }

    #[test]
    fn test_edge_case_find_maximum() {
        let data = vec![8, 1, 5, 9, 2, 0, 4];
        run_and_verify(&data, data.len());
    }

    #[test]
    fn test_edge_case_with_duplicates() {
        let data = vec![7, 2, 7, 5, 2, 9, 5, 9, 1];
        run_and_verify(&data, 4);
    }

    #[test]
    fn test_structured_input_reverse_sorted() {
        let data = vec![70, 60, 50, 40, 30, 20, 10];
        run_and_verify(&data, 6);
    }

    #[test]
    fn test_randomized_configurations() {
        let mut rng = thread_rng();
        for _ in 0..100 {
            let n = rng.gen_range(1..=500);
            let k = rng.gen_range(1..=n);

            let data: Vec<i32> = (0..n).map(|_| rng.gen_range(-1000..=1000)).collect();
            run_and_verify(&data, k);
        }
    }
}
