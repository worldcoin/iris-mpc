use std::{
    fs::File,
    io::{BufReader, BufWriter},
};

use clap::ValueEnum;
use eyre::Result;
use iris_mpc_common::IrisVectorId;
use serde::{Deserialize, Serialize};

use crate::{
    hnsw::graph::{
        layered_graph::{self, GraphMem},
        neighborhood,
    },
    utils::serialization::types::{
        graph_v0::{read_graph_v0, GraphV0},
        graph_v1::{self, read_graph_v1, GraphV1},
    },
};

#[derive(Clone, Debug, ValueEnum, Copy, Serialize, Deserialize, PartialEq)]
pub enum GraphFormat {
    /// Binary format without cached distances
    V1,

    /// Binary format with cached `(u16, u16)` distances
    V0,
}

// fn write_bin<T, P>(data: T, path: P) -> Result<()>
// where
//     T: Serialize,
//     P: AsRef<std::path::Path>,
// {
//     let file = File::create(path)?;
//     let writer = BufWriter::new(file);
//     bincode::serialize_into(writer, &data)?;
//     Ok(())
// }

// fn read_bin<T, P>(path: P) -> Result<T>
// where
//     T: DeserializeOwned,
//     P: AsRef<std::path::Path>,
// {
//     let file = File::open(path)?;
//     let reader = BufReader::new(file);
//     let data: T = bincode::deserialize_from(reader)?;
//     Ok(data)
// }

pub fn read_graph_from_file<P: AsRef<std::path::Path>>(
    path: P,
    format: GraphFormat,
) -> Result<GraphMem<IrisVectorId>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    match format {
        GraphFormat::V1 => {
            let graph = read_graph_v1(&mut reader)?;
            Ok(graph.into())
        }
        GraphFormat::V0 => {
            let graph = read_graph_v0(&mut reader)?;
            Ok(graph.into())
        }
    }
}

pub fn write_graph_to_file<P: AsRef<std::path::Path>>(
    path: P,
    _data: &GraphMem<IrisVectorId>,
) -> Result<()> {
    let file = File::create(path)?;
    let _writer = BufWriter::new(file);

    todo!()
}

/* --------------- Conversion GraphMem -> GraphV1 ------------------- */

// TODO -- finish conversion boilerplate

impl From<IrisVectorId> for graph_v1::VectorId {
    fn from(value: IrisVectorId) -> Self {
        graph_v1::VectorId {
            id: value.serial_id(),
            version: value.version_id(),
        }
    }
}

impl From<layered_graph::EntryPoint<IrisVectorId>> for graph_v1::EntryPoint {
    fn from(value: layered_graph::EntryPoint<IrisVectorId>) -> Self {
        graph_v1::EntryPoint {
            point: value.point.into(),
            layer: value.layer,
        }
    }
}

impl From<neighborhood::SortedEdgeIds<IrisVectorId>> for graph_v1::EdgeIds {
    fn from(value: neighborhood::SortedEdgeIds<IrisVectorId>) -> Self {
        graph_v1::EdgeIds(value.0.into_iter().map(Into::into).collect())
    }
}

// ...

impl From<GraphMem<IrisVectorId>> for GraphV1 {
    fn from(_value: GraphMem<IrisVectorId>) -> Self {
        todo!()
    }
}

/* --------------- Conversion GraphV1 -> GraphMem ------------------- */

impl From<graph_v1::VectorId> for IrisVectorId {
    fn from(value: graph_v1::VectorId) -> Self {
        IrisVectorId::new(value.id, value.version)
    }
}

impl From<graph_v1::EntryPoint> for layered_graph::EntryPoint<IrisVectorId> {
    fn from(value: graph_v1::EntryPoint) -> Self {
        layered_graph::EntryPoint {
            point: value.point.into(),
            layer: value.layer,
        }
    }
}

impl From<graph_v1::EdgeIds> for neighborhood::SortedEdgeIds<IrisVectorId> {
    fn from(value: graph_v1::EdgeIds) -> Self {
        neighborhood::SortedEdgeIds(value.0.into_iter().map(Into::into).collect())
    }
}

// ...

impl From<GraphV1> for GraphMem<IrisVectorId> {
    fn from(_value: GraphV1) -> Self {
        todo!()
    }
}

/* --------------- Conversion GraphV0 -> GraphMem ------------------- */

//...

impl From<GraphV0> for GraphMem<IrisVectorId> {
    fn from(_value: GraphV0) -> Self {
        todo!()
    }
}

// TODO: utility function to read and write "pair of graph", serialized as type `GraphV1Pair`
