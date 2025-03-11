use kameo::Actor;

// ------------------------------------------------------------------------
// Declaration + state.
// ------------------------------------------------------------------------

// Actor: Issues query/insert operations over in-memory HNSW graph.
#[derive(Actor, Default)]
pub struct GraphIndexer {}
