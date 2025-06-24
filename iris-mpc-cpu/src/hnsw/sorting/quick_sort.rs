use eyre::{bail, Result};
use itertools::izip;

use crate::hnsw::VectorStore;

/// Represents a subcall of an ongoing execution of a quick-sort algorithm
/// over a partially sorted list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartialQuickSort {
    /// Index of left endpoint of sort interval, inclusive
    pub left: usize,

    /// Length of sort interval prefix which is assumed to be pre-sorted
    pub sorted_len: usize,

    /// Length of sort interval suffix which is a priori unsorted
    pub unsorted_len: usize,
}

/// Internally represents the sorting state of a `PartialQuickSort` struct.
enum SortState {
    /// Sort interval is fully sorted
    Sorted,

    /// Sort interval is partially sorted, with specified number of sorted entries
    PartiallySorted,

    /// Sort interval is fully unsorted
    Unsorted,
}

impl PartialQuickSort {
    /// Return the pairs of indices which should be compared with a "less than"
    /// operator to continue to the next layer of recursion for the sort.
    ///
    /// The results of these comparisons are used in the `step` function to
    /// update a list during sorting and continue to the next level of recursion.
    pub fn next_cmps(&self) -> Result<Vec<(usize, usize)>> {
        match self.state() {
            SortState::Sorted => {
                bail!("PartialQuickSort instance is already sorted");
            }
            SortState::Unsorted => Ok((1..self.right()).map(|x| (x, 0)).collect()),
            SortState::PartiallySorted => {
                let pivot_idx = self.pivot_idx();
                let left_u = self.left + self.sorted_len;

                Ok((left_u..self.right()).map(|x| (x, pivot_idx)).collect())
            }
        }
    }

    /// Update `lst` and step to the next layer of recursion for the sort, using
    /// the comparison results for index pairs given by a call to `next_cmps`.
    ///
    /// This function copies elements of `lst` from the represented index
    /// interval into the corresponding interval in `target` after partitioning
    /// around the pivot element, as follows:
    ///
    /// - Moves the pivot element to the correct sorted index `pivot_idx`
    ///
    /// - Moves all elements less than the pivot element to the left of
    ///   `pivot_idx` and all elements greater than the pivot element to the
    ///   right of `pivot_idx`
    ///
    /// - Maintains the relative ordering of elements up to these shifts, in
    ///   particular positioning sorted sublists in the left part of the
    ///   resulting subintervals
    ///
    /// - In particular, unsorted items "equal" to the pivot element are sorted
    ///   to after the pivot element, and the overall sort is stable
    ///
    /// After the elements of `lst` are copied to `target` as described, the
    /// current `PartialQuickSort` struct is mutated to represent the left
    /// recursive subinterval, and a new struct is returned representing the
    /// right recursive subinterval.
    pub fn step<Item: Clone>(
        &mut self,
        cmp_results: &[bool],
        lst: &[Item],
        target: &mut [Item],
    ) -> Result<Self> {
        // Index of pivot element, and start index of the unsorted part of the
        // sort interval, inclusive.  If fully unsorted, then the first element
        // by itself is sorted.
        let (pivot_idx, unsorted_start) = match self.state() {
            SortState::Sorted => {
                bail!("PartialQuickSort instance is already sorted");
            }
            SortState::Unsorted => (0, 1),
            SortState::PartiallySorted => (self.pivot_idx(), self.left + self.sorted_len),
        };

        // Count elements going to left and right subintervals
        let sorted_len_left = pivot_idx - self.left;
        let sorted_len_right = unsorted_start - pivot_idx - 1;
        let unsorted_len_left = cmp_results.iter().filter(|b| **b).count();
        let unsorted_len_right = cmp_results.len() - unsorted_len_left;

        // Move pivot element to sorted index in `target`
        let pivot_target = self.left + sorted_len_left + unsorted_len_left;
        target[pivot_target] = lst[pivot_idx].clone();

        // Copy sorted subintervals into `target`
        target[(self.left)..pivot_idx].clone_from_slice(&lst[self.left..pivot_idx]);
        target[(pivot_target + 1)..(pivot_target + 1 + sorted_len_right)]
            .clone_from_slice(&lst[(pivot_idx + 1)..unsorted_start]);

        // Step through unsorted interval and copy elements into left and right sides
        // of `target` depending on the outcomes in `cmp_results`
        let mut l_idx = pivot_idx;
        let mut r_idx = pivot_target + 1 + sorted_len_right;
        for (val, cmp) in izip!(lst[unsorted_start..self.right()].iter(), cmp_results) {
            if *cmp {
                target[l_idx] = val.clone();
                l_idx += 1;
            } else {
                target[r_idx] = val.clone();
                r_idx += 1;
            }
        }

        // Update `self` to left recursive sort
        self.unsorted_len = unsorted_len_left;
        self.sorted_len = sorted_len_left;

        // Return right recursive sort
        let right_part = Self {
            left: pivot_target + 1,
            sorted_len: sorted_len_right,
            unsorted_len: unsorted_len_right,
        };
        Ok(right_part)
    }

    /// Return whether the quicksort block represented by this struct is finished.
    ///
    /// This is the case either if the full block has length 0 or 1, or if the
    /// full sort interval is covered by the known-sorted prefix.
    pub fn is_finished(&self) -> bool {
        matches!(self.state(), SortState::Sorted)
    }

    /// Return the index of the right endpoint of the sort interval, exclusive.
    pub fn right(&self) -> usize {
        self.left + self.sorted_len + self.unsorted_len
    }

    /// Return the `SortState` representing the current status of this quicksort
    /// struct.  An interval of length 0 or 1 is sorted, and an interval of length
    /// 2 or greater is sorted, unsorted, or partially sorted depending on the
    /// lengths of the sorted and unsorted intervals.
    fn state(&self) -> SortState {
        if self.sorted_len + self.unsorted_len <= 1 {
            // Length 0 or 1 list is sorted
            SortState::Sorted
        } else {
            // Interval is length at least 2
            if self.sorted_len == 0 {
                SortState::Unsorted
            } else if self.unsorted_len == 0 {
                SortState::Sorted
            } else {
                SortState::PartiallySorted
            }
        }
    }

    /// Return the pivot index for this sort step, set to the midpoint of the
    /// sorted subinterval (rounded down) when the subinterval is nonempty, or
    /// 0 if the sorted subinterval is empty.
    ///
    /// This selection works well in the context we are using this sort, where
    /// the sorted subinterval is generally large relative to the unsorted
    /// part, and the unsorted part consists of items with relatively uniform
    /// ordering.
    pub fn pivot_idx(&self) -> usize {
        self.left + self.sorted_len / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eyre::Result;

    #[test]
    fn test_qs_step() -> Result<()> {
        // sort by tuple first element, second element to check stability
        let lst = vec![
            (0u32, 7u32),
            (1, 3),
            (2, 0),
            (3, 5),
            (4, 2),
            (3, 4),
            (2, 2),
            (3, 3),
        ];

        let mut qs_step = PartialQuickSort {
            left: 2,
            sorted_len: 3,
            unsorted_len: 2,
        };

        let mut lst2 = vec![(0u32, 0u32); lst.len()];
        let lst2_after = vec![
            (0u32, 0u32),
            (0, 0),
            (2, 0),
            (2, 2),
            (3, 5),
            (4, 2),
            (3, 4),
            (0, 0),
        ];

        let cmp_pairs = qs_step.next_cmps()?;
        let cmp_results: Vec<_> = cmp_pairs
            .iter()
            .map(|(idx1, idx2)| lst[*idx1].0 < lst[*idx2].0)
            .collect();

        let next_qs_step = qs_step.step(&cmp_results, &lst, &mut lst2)?;

        // [ (2, 0) | (2, 2) ]
        assert_eq!(
            qs_step,
            PartialQuickSort {
                left: 2,
                sorted_len: 1,
                unsorted_len: 1,
            }
        );

        // [ (4, 2) | (3, 4) ]
        assert_eq!(
            next_qs_step,
            PartialQuickSort {
                left: 5,
                sorted_len: 1,
                unsorted_len: 1,
            }
        );

        assert_eq!(lst2, lst2_after);

        Ok(())
    }

    #[test]
    fn test_qs_step_unsorted() -> Result<()> {
        let lst = vec![4u32, 1, 3, 7, 5, 6, 2];

        let mut qs_step = PartialQuickSort {
            left: 0,
            sorted_len: 0,
            unsorted_len: 7,
        };

        let mut lst2 = vec![0u32; lst.len()];
        let lst2_after = vec![1, 3, 2, 4u32, 7, 5, 6];

        let cmp_pairs = qs_step.next_cmps()?;
        let cmp_results: Vec<_> = cmp_pairs
            .iter()
            .map(|(idx1, idx2)| lst[*idx1] < lst[*idx2])
            .collect();

        let next_qs_step = qs_step.step(&cmp_results, &lst, &mut lst2)?;

        // [ 1 3 2 ]
        assert_eq!(
            qs_step,
            PartialQuickSort {
                left: 0,
                sorted_len: 0,
                unsorted_len: 3,
            }
        );

        // [ 7 5 6 ]
        assert_eq!(
            next_qs_step,
            PartialQuickSort {
                left: 4,
                sorted_len: 0,
                unsorted_len: 3,
            }
        );

        assert_eq!(lst2, lst2_after);

        Ok(())
    }
}
