//! This module implements stable types to represent serialization formats which
//! will can be stably referenced while the codebase is changing.  In general
//! this module aims to have one submodule for each "file format" which is
//! produced or consumed by the codebase at some stage of development, with
//! possible exceptions for lower-level data types related to serialization
//! formats of external libraries.

pub mod graph_v0;
pub mod graph_v1;
pub mod graph_v2;
pub mod graph_v2_pair;
pub mod graph_v3;
pub mod iris_base64;
