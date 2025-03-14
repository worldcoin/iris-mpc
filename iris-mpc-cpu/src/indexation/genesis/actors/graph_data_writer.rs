use kameo::Actor;

// ------------------------------------------------------------------------
// Declaration + state.
// ------------------------------------------------------------------------

// Name for logging purposes.
const _: &str = "GraphDataWriter";

// Actor: Writes HNSW graph data to store.
#[derive(Actor, Default)]
pub struct GraphDataWriter {}
