/// Represents an ongoing binary search to identify the insertion index of a
/// query within a sorted array.  The next index for comparison at each stage
/// is generated by the `next` function, and once the results of that
/// comparison are available, the struct is updated using the `update`
/// function.  After the search is completed, the resulting insertion index is
/// retrieved with a call to `result`.
///
/// Comparison results are expected to be the outcome of a "strict less than"
/// operation by default, where the sorted array is in increasing order, and
/// under this interpretation, the resulting insertion index places the query
/// after all elements "less than or equal to" the value, and before all
/// elements "greater than" the value.
pub struct BinarySearch {
    /// Index of left endpoint of search interval, inclusive
    pub left: usize,

    /// Index of right endpoint of search interval, exclusive
    pub right: usize,
}

impl BinarySearch {
    /// Update the given binary search with comparison result `cmp_result`
    /// describing the outcome of `query < arr[middle]`.
    pub fn update(&mut self, cmp_result: bool) {
        if !self.is_finished() {
            let middle = midpoint(self.left, self.right);
            if cmp_result {
                self.right = middle;
            } else {
                self.left = middle + 1;
            }
        }
    }

    /// Returns whether the binary search has converged to an insertion index.
    pub fn is_finished(&self) -> bool {
        self.left >= self.right
    }

    /// If additional comparisons are needed, returns the array index `middle`
    /// giving the element which should be compared against the query next in
    /// the binary search.
    ///
    /// A `Some` value indicates that additional comparisons are needed, and a
    /// `None` value indicates that search has already converged to a result.
    pub fn next(&self) -> Option<usize> {
        if self.is_finished() {
            None
        } else {
            Some(midpoint(self.left, self.right))
        }
    }

    /// Returns `Some(insertion_index)` if the search has converged to a result.
    /// Otherwise, returns `None`.
    pub fn result(&self) -> Option<usize> {
        if self.is_finished() {
            Some(self.left)
        } else {
            None
        }
    }
}

#[inline(always)]
fn midpoint(left: usize, right: usize) -> usize {
    (left + right) / 2
}
