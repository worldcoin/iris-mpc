use kameo::Actor;

// ------------------------------------------------------------------------
// Declaration + state.
// ------------------------------------------------------------------------

// Actor: Writes HNSW graph data to store.
#[derive(Actor, Default)]
pub struct GraphDataWriter {}
