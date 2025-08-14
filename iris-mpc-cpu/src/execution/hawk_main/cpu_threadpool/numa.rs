use std::cmp;

use core_affinity::CoreId;

const SHARD_COUNT: usize = 2;

pub fn select_core_ids(shard_index: usize) -> Vec<CoreId> {
    let shard_index = shard_index % SHARD_COUNT;

    let mut core_ids = core_affinity::get_core_ids().unwrap();
    core_ids.reverse();
    assert!(!core_ids.is_empty());

    let shard_size = core_ids.len() / SHARD_COUNT;
    let start = shard_index * shard_size;
    let end = cmp::min(start + shard_size, core_ids.len());

    core_ids[start..end].to_vec()
}
