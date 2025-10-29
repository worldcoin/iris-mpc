//! This module contains the implementation of the different types of stores that can be used in the HNSW protocol.
//! The stores are used to store the vectors or their references that are indexed by the HNSW protocol.
//! The stores also contain the logic to perform the necessary low-level operations on the stored vectors such as
//! - distance computation,
//! - matching,
//! - distance comparison,
//! - insertion into the store,
//! - preprocessing of queries (vectors) to the HNSW protocol.
//!
//! Each store implements the `Store` trait which defines the common interface for all the stores.
//! The `Store` trait is defined in `hnsw::vector_store.rs`.

/// Store with vectors in secret shared form.
/// The underlying operations are secure multi-party computation (MPC) operations.
pub mod aby3;

/// Store with vectors in plaintext form.
pub mod plaintext_store;

/// Data structure for shared in-memory irises
pub mod shared_irises;

pub mod build_plaintext;

pub mod naive_knn_plaintext;

const WITH_MIN_ROTA: bool = true;
