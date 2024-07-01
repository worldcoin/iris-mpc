use crate::N_QUERIES;

/// Determines, on which devices the non-match results will be inserted.
pub fn distribute_insertions(results: &[usize], db_sizes: &[usize]) -> Vec<Vec<usize>> {
    let mut ret = vec![vec![]; db_sizes.len()];
    let start = db_sizes
        .iter()
        .position(|&x| x == *db_sizes.iter().min().unwrap())
        .unwrap();

    let mut c = start;
    for r in results {
        ret[c].push(*r);
        c = (c + 1) % db_sizes.len();
    }
    ret
}

/// Calculates the index, at which the inserted entry will be found later in the
/// db.
pub fn calculate_insertion_indices(
    merged_results: &mut [u32],
    insertion_list: &[Vec<usize>],
    db_sizes: &[usize],
) -> Vec<bool> {
    let mut matches = vec![true; N_QUERIES];
    let mut last_index = db_sizes.iter().sum::<usize>() as u32;
    let mut c: usize = 0;
    let mut min_index = 0;
    let mut min_index_val = usize::MAX;
    for i in 0..insertion_list.len() {
        if insertion_list[i].len() > 0 && insertion_list[i][0] < min_index_val {
            min_index_val = insertion_list[i][0];
            min_index = i;
        }
    }
    loop {
        for i in 0..insertion_list.len() {
            let ii = (i + min_index) % insertion_list.len();
            if c >= insertion_list[ii].len() {
                return matches;
            }
            merged_results[insertion_list[ii][c]] = last_index;
            matches[insertion_list[ii][c]] = false;
            last_index += 1;
        }
        c += 1;
    }
}

/// Merges the results across all devices. Returns the smallest matching index.
pub fn get_merged_results(host_results: &[Vec<u32>], n_devices: usize) -> Vec<u32> {
    let mut results = vec![];
    for j in 0..host_results[0].len() {
        let mut match_entry = u32::MAX;
        for i in 0..host_results.len() {
            let match_idx = host_results[i][j] * n_devices as u32 + i as u32;
            if host_results[i][j] != u32::MAX && match_idx < match_entry {
                match_entry = match_idx;
            }
        }

        results.push(match_entry);

        // DEBUG
        println!(
            "Query {}: match={} [index: {}]",
            j,
            match_entry != u32::MAX,
            match_entry
        );
    }
    results
}
