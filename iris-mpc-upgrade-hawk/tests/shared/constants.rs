const DB_SIZE: usize = 1000;
const DB_RNG_SEED: u64 = 0xdeadbeef;
const INTERNAL_RNG_SEED: u64 = 0xdeadbeef;
const NUM_BATCHES: usize = 5;

const HNSW_PARALLELISM_REQUEST: usize = 1;
const HNSW_PARALLELISM_STREAM: usize = 1;
const HNSW_PARALLELISM_CONNECTION: usize = 1;

const MAX_BATCH_SIZE: usize = 5;
const MAX_DELETIONS_PER_BATCH: usize = 0;
const MAX_RESET_UPDATES_PER_BATCH: usize = 0;

const HNSW_EF_CONSTR: usize = 320;
const HNSW_M: usize = 256;
const HNSW_EF_SEARCH: usize = 256;
