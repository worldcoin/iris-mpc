use std::collections::HashMap;

use crate::hnsw::VectorStore;
use eyre::Result;

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
            .filter(|&i| i != pivot_idx) // Don't compare the pivot with itself
            .map(|i| (self.perm[i], pivot_val))
            .collect()
    }

    /// Processes the results of the comparisons and narrows the search space.
    pub fn accept_results(&mut self, results: Vec<bool>) {
        if self.is_done() {
            return;
        }

        let pivot_idx = self.get_pivot_index();
        let pivot_val = self.perm[pivot_idx];

        // --- In-Place Partitioning ---
        let values_to_compare: Vec<_> = (self.left..self.right)
            .filter(|&i| i != pivot_idx)
            .map(|i| self.perm[i])
            .collect();
        assert_eq!(
            values_to_compare.len(),
            results.len(),
            "Mismatch between comparisons and results"
        );
        let results_map: HashMap<usize, bool> =
            values_to_compare.into_iter().zip(results).collect();

        self.perm.swap(pivot_idx, self.right - 1);

        let mut store_idx = self.left;
        for i in self.left..(self.right - 1) {
            if *results_map.get(&self.perm[i]).unwrap_or(&false) {
                self.perm.swap(i, store_idx);
                store_idx += 1;
            }
        }

        self.perm.swap(store_idx, self.right - 1);
        let pivot_new_idx = store_idx;

        // --- Update State for Next Iteration ---
        if self.kth_index == pivot_new_idx {
            self.result = Some(pivot_val);
        } else if self.kth_index < pivot_new_idx {
            self.right = pivot_new_idx;
        } else {
            self.left = pivot_new_idx + 1;
        }

        if !self.is_done() && self.right > self.left && self.right - self.left <= 1 {
            self.result = Some(self.perm[self.left]);
        }
    }

    /// Calculates the pivot index without risking integer overflow.
    fn get_pivot_index(&self) -> usize {
        self.left + (self.right - self.left) / 2
    }
}

/// Test harness that simulates an external comparison oracle.
/// It runs the Quickselect algorithm on a concrete slice of data.
///
/// Returns a tuple containing:
/// 1. A reference to the found k-th element.
/// 2. The final state of the permutation vector.
/// 3. The final index of the k-th element within the permutation.
pub fn run_quickselect_on_data<T: Ord>(data: &[T], k: usize) -> (&T, Vec<usize>, usize) {
    let n = data.len();
    let mut qs = Quickselect::new(n, k);

    // Loop until the algorithm finds the result
    while !qs.is_done() {
        let comparisons = qs.get_next_cmps();

        // If there are no more comparisons to make, the algorithm should be done.
        // This can happen if the search space is narrowed to a single element.
        if comparisons.is_empty() {
            // Force quit and let the result be determined by the final state
            break;
        }

        // Simulate the oracle: perform the comparisons on the concrete data
        let results: Vec<bool> = comparisons
            .iter()
            .map(|&(idx1, idx2)| data[idx1] < data[idx2])
            .collect();

        // Feed the results back to the algorithm
        qs.accept_results(results);
    }

    // Get the final index and return the corresponding value and state
    let result_index = qs
        .get_result()
        .expect("Algorithm finished but no result was found");

    let final_k_idx = qs
        .perm
        .iter()
        .position(|&p| p == result_index)
        .expect("Result index not found in final permutation");

    (&data[result_index], qs.perm, final_k_idx)
}

pub async fn run_quickselect_on_vectors<S: VectorStore>(
    store: &mut S,
    data: &[(S::VectorRef, S::DistanceRef)],
    k: usize,
) -> Result<Vec<(S::VectorRef, S::DistanceRef)>> {
    let n = data.len();
    let mut qs = Quickselect::new(n, k);

    // Loop until the algorithm finds the result
    while !qs.is_done() {
        let comparisons = qs.get_next_cmps();

        // If there are no more comparisons to make, the algorithm should be done.
        // This can happen if the search space is narrowed to a single element.
        if comparisons.is_empty() {
            // Force quit and let the result be determined by the final state
            break;
        }

        // Simulate the oracle: perform the comparisons on the concrete data
        let distances = comparisons
            .iter()
            .map(|&(idx1, idx2)| (data[idx1].1.clone(), data[idx2].1.clone()))
            .collect::<Vec<_>>();
        let results = store.less_than_batch(&distances).await?;

        // Feed the results back to the algorithm
        qs.accept_results(results);
    }

    let mut new_neighborhood = qs.perm.iter().map(|i| data[*i].clone()).collect::<Vec<_>>();
    new_neighborhood.truncate(k);

    Ok(new_neighborhood)
}

//------------------------------------------------------------------//
//                          TESTS                                   //
//------------------------------------------------------------------//

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to get the expected answer by sorting.
    fn get_expected<T: Ord + Clone>(data: &[T], k: usize) -> T {
        let mut sorted_data = data.to_vec();
        sorted_data.sort();
        sorted_data[k - 1].clone()
    }

    /// A helper to run the tests and perform all checks.
    fn run_and_verify<T: Ord + Clone + std::fmt::Debug>(data: &[T], k: usize) {
        let expected = get_expected(data, k);
        let (result_val, final_perm, final_k_idx) = run_quickselect_on_data(data, k);

        // 1. Check if the k-th value is correct.
        assert_eq!(*result_val, expected);

        // 2. Check the partitioning property:
        // All elements before the k-th element in the final permutation must be less than or equal to it.
        for i in 0..final_k_idx {
            let current_val = &data[final_perm[i]];
            assert!(
                current_val <= result_val,
                "Partition Fail: Element {:?} at perm index {} should be <= {:?} (k-th element)",
                current_val,
                i,
                result_val
            );
        }

        // All elements after the k-th element must be greater than or equal to it.
        for i in (final_k_idx + 1)..final_perm.len() {
            let current_val = &data[final_perm[i]];
            assert!(
                current_val >= result_val,
                "Partition Fail: Element {:?} at perm index {} should be >= {:?} (k-th element)",
                current_val,
                i,
                result_val
            );
        }
    }

    #[test]
    fn test_find_minimum() {
        let data = vec![8, 1, 5, 9, 2, 0, 4];
        run_and_verify(&data, 1);
    }

    #[test]
    fn test_find_maximum() {
        let data = vec![8, 1, 5, 9, 2, 0, 4];
        run_and_verify(&data, data.len());
    }

    #[test]
    fn test_find_median() {
        let data = vec![3, 7, 8, 5, 2, 1, 9, 6, 4]; // 9 elements
        run_and_verify(&data, 5);
    }

    #[test]
    fn test_with_duplicates() {
        let data = vec![7, 2, 7, 5, 2, 9, 5, 9, 1];
        run_and_verify(&data, 4);
    }

    #[test]
    fn test_with_sorted_array() {
        let data = vec![10, 20, 30, 40, 50, 60, 70];
        run_and_verify(&data, 3);
    }

    #[test]
    fn test_with_reverse_sorted_array() {
        let data = vec![70, 60, 50, 40, 30, 20, 10];
        run_and_verify(&data, 6);
    }

    #[test]
    fn test_large_array() {
        let data: Vec<i32> = (0..1000).rev().collect(); // 999, 998, ..., 0
        run_and_verify(&data, 251);
    }

    #[test]
    fn test_two_elements() {
        let data = vec![100, 2];
        run_and_verify(&data, 1);
    }
}
