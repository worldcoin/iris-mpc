#![allow(clippy::needless_range_loop)]
pub mod config;
pub mod error;
pub mod galois;
pub mod galois_engine;
pub mod helpers;
pub mod id;
pub mod iris_db;
pub mod shamir;

pub const IRIS_CODE_LENGTH: usize = 12800;
