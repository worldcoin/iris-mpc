//! This submodule is based off of a local port of functionality in the
//! hawk-pack crate, which implements a generic functionality for HNSW graph
//! search:
//!
//! (<https://github.com/Inversed-Tech/hawk-pack/>)
//!
//! This local copy simplifies some of the generic functionality present in the
//! source crate, and modifies the implementation to better support the design
//! and performance constraints of the iris uniqueness application.

pub mod graph;
pub mod metrics;
pub mod searcher;
pub mod vector_store;

pub use graph::{layered_graph::GraphMem, neighborhood::SortedNeighborhood};
pub use searcher::{HnswParams, HnswSearcher};
pub use vector_store::VectorStore;
