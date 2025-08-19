use eyre::{bail, OptionExt, Result};
use itertools::izip;

use crate::hnsw::VectorStore;

/// Internally represents the sorting state of a `PartialQuickSort` struct.
enum SortState {
    /// Sort interval is fully sorted
    Sorted,

    /// Sort interval is partially sorted, with specified number of sorted entries
    PartiallySorted,

    /// Sort interval is fully unsorted
    Unsorted,
}

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
            SortState::Unsorted => Ok(((self.left + 1)..self.right())
                .map(|x| (x, self.left))
                .collect()),
            SortState::PartiallySorted => {
                let pivot_idx = self.pivot_idx();
                let left_u = self.left + self.sorted_len;

                Ok((left_u..self.right()).map(|x| (x, pivot_idx)).collect())
            }
        }
    }

    /// Update `src` list and step to the next layer of recursion using the
    /// comparison results for index pairs given by a call to `next_cmps`.
    ///
    /// This function copies elements of `src` from the represented index
    /// interval into the corresponding interval in `dst` after partitioning
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
    /// - As a special case, unsorted items "equal" to the pivot element are
    ///   sorted to after the pivot element, and the overall sort is stable
    ///
    /// After the elements of `src` are copied to `dst` as described, the
    /// current `PartialQuickSort` struct is mutated to represent the left
    /// recursive subinterval, and a new struct is returned representing the
    /// right recursive subinterval.
    pub fn step<Item: Clone>(
        &mut self,
        cmp_results: &[bool],
        src: &[Item],
        dst: &mut [Item],
    ) -> Result<Self> {
        // Index of pivot element, and start index of the unsorted part of the
        // sort interval, inclusive.  If fully unsorted, then the first element
        // by itself is sorted.
        let (pivot_idx, unsorted_start) = match self.state() {
            SortState::Sorted => {
                bail!("PartialQuickSort instance is already sorted");
            }
            SortState::Unsorted => (self.left, self.left + 1),
            SortState::PartiallySorted => (self.pivot_idx(), self.left + self.sorted_len),
        };

        // Count elements going to left and right subintervals
        let sorted_len_left = pivot_idx - self.left;
        let sorted_len_right = unsorted_start - pivot_idx - 1;
        let unsorted_len_left = cmp_results.iter().filter(|b| **b).count();
        let unsorted_len_right = cmp_results.len() - unsorted_len_left;

        // Move pivot element to sorted index in `dst`
        let pivot_target = self.left + sorted_len_left + unsorted_len_left;
        dst[pivot_target] = src[pivot_idx].clone();

        // Copy sorted subintervals into `dst`
        dst[(self.left)..pivot_idx].clone_from_slice(&src[self.left..pivot_idx]);
        dst[(pivot_target + 1)..(pivot_target + 1 + sorted_len_right)]
            .clone_from_slice(&src[(pivot_idx + 1)..unsorted_start]);

        // Step through unsorted interval and copy elements into left and right
        // sides of `dst` depending on the outcomes in `cmp_results`
        let mut l_idx = pivot_idx;
        let mut r_idx = pivot_target + 1 + sorted_len_right;
        for (val, cmp) in izip!(src[unsorted_start..self.right()].iter(), cmp_results) {
            if *cmp {
                dst[l_idx] = val.clone();
                l_idx += 1;
            } else {
                dst[r_idx] = val.clone();
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
        self.left + (self.sorted_len - 1) / 2
    }
}

/// Apply the parallel quicksort algorithm to the given list using `store` as
/// the `VectorStore` for doing comparisons.  `buffer` is a mutable slice used
/// for processing which must be at least as large as `list`, and `sorted_len`
/// gives the number of elements at the start of `list` which are known to be
/// in sorted order already.
pub async fn apply_quicksort<V: VectorStore>(
    store: &mut V,
    list: &mut [(V::VectorRef, V::DistanceRef)],
    buffer: &mut [(V::VectorRef, V::DistanceRef)],
    sorted_len: usize,
) -> Result<()> {
    let len = list.len();
    if len == 0 {
        return Ok(());
    }

    if buffer.len() < len {
        bail!("Buffer is too small for the list being sorted")
    }
    let buffer = &mut buffer[0..len];

    #[derive(Clone)]
    struct LocalSort {
        // Base partial quicksort
        pub sort: PartialQuickSort,

        // Index range for pending comparison results
        pub cmp_results_range: Option<(usize, usize)>,
    }

    let top_level_sort = PartialQuickSort {
        left: 0,
        sorted_len,
        unsorted_len: len - sorted_len,
    };
    if top_level_sort.is_finished() {
        return Ok(());
    }

    let mut sorts: Vec<LocalSort> = Vec::with_capacity(len);
    sorts.push(LocalSort {
        sort: top_level_sort,
        cmp_results_range: None,
    });

    // Main processing loop
    let mut cmps: Vec<(usize, usize)> = Vec::with_capacity(len);
    while !sorts.is_empty() {
        // Collect comparisons
        for LocalSort {
            sort,
            cmp_results_range,
        } in sorts.iter_mut()
        {
            let local_cmps = sort.next_cmps()?;
            *cmp_results_range = Some((cmps.len(), cmps.len() + local_cmps.len()));
            cmps.extend(local_cmps);
        }

        // Collect distances and evaluate comparisons
        let distances: Vec<_> = cmps
            .drain(..)
            .filter_map(|(idx1, idx2)| match (list.get(idx1), list.get(idx2)) {
                (Some((_, d1)), Some((_, d2))) => Some((d1.clone(), d2.clone())),
                _ => None,
            })
            .collect();
        let cmp_results = store.less_than_batch(&distances).await?;

        // Execute quicksort steps for each active recursive sort
        let mut new_sorts = Vec::with_capacity(sorts.len());
        for LocalSort {
            sort,
            cmp_results_range,
        } in sorts.iter_mut()
        {
            let (local_cmps_left, local_cmps_right) = cmp_results_range
                .take()
                .ok_or_eyre("Unable to find expected local comparisons range")?;
            let new_sort = sort.step(
                &cmp_results[local_cmps_left..local_cmps_right],
                list,
                buffer,
            )?;
            new_sorts.push(new_sort);
        }
        list.clone_from_slice(buffer);

        // Update with new recursive sorts
        sorts.extend(new_sorts.into_iter().map(|sort| LocalSort {
            sort,
            cmp_results_range: None,
        }));

        // Remove all finished parts
        sorts.retain(|s| !s.sort.is_finished());
    }

    Ok(())
}

/// Apply partial quicksort to a list of `PartialOrd` elements which has
/// sorted prefix of length `sorted_len`, by recursive application.
///
/// Function is meant primarily for testing use.
pub fn apply_quicksort_recursive<T: PartialOrd + Clone>(
    list: &mut [T],
    sorted_len: usize,
) -> Result<()> {
    let mut buffer: Vec<_> = list.into();
    let initial_sort = PartialQuickSort {
        left: 0,
        sorted_len,
        unsorted_len: list.len() - sorted_len,
    };

    fn apply_quicksort_recursive_interior<T: PartialOrd + Clone>(
        list: &mut [T],
        buffer: &mut [T],
        mut sort: PartialQuickSort,
    ) -> Result<()> {
        if !sort.is_finished() {
            let (left, right) = (sort.left, sort.right());

            let cmps = sort.next_cmps()?;

            let cmp_results: Vec<_> = cmps
                .into_iter()
                .map(|(idx1, idx2)| list[idx1] < list[idx2])
                .collect();

            let next_sort = sort.step(&cmp_results, list, buffer)?;
            list[left..right].clone_from_slice(&buffer[left..right]);

            apply_quicksort_recursive_interior(list, buffer, sort)?;
            apply_quicksort_recursive_interior(list, buffer, next_sort)?;
        }

        Ok(())
    }

    apply_quicksort_recursive_interior(list, &mut buffer, initial_sort)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eyre::Result;
    use rand::Rng;

    #[test]
    fn test_qs_step() -> Result<()> {
        // sort by tuple first element; second element used to check stability
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

    #[test]
    fn test_qs_random() -> Result<()> {
        let mut rng = rand::thread_rng();

        for _ in 0..10 {
            let sorted_length = rng.gen_range(128..512);
            let unsorted_length = rng.gen_range(128..512);
            let length = sorted_length + unsorted_length;

            for _ in 0..10 {
                let mut vals1: Vec<u64> = (0..length).map(|_| rng.gen_range(0..100)).collect();
                let mut vals2 = vals1.clone();

                vals1.get_mut(0..sorted_length).unwrap().sort();
                apply_quicksort_recursive(&mut vals1, sorted_length)?;

                vals2.sort();

                assert_eq!(vals1, vals2);
            }
        }

        Ok(())
    }
}
