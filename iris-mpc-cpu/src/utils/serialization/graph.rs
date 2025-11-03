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
        layered_graph::{self, GraphMem, Layer},
        neighborhood,
    },
    utils::serialization::types::{
        graph_v0::{self, read_graph_v0, GraphV0},
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

impl From<Layer<IrisVectorId>> for graph_v1::Layer {
    fn from(value: Layer<IrisVectorId>) -> Self {
        graph_v1::Layer {
            links: value
                .links
                .into_iter()
                .map(|(v, nb)| (v.into(), nb.into()))
                .collect(),
        }
    }
}

impl From<GraphMem<IrisVectorId>> for GraphV1 {
    fn from(value: GraphMem<IrisVectorId>) -> Self {
        graph_v1::GraphV1 {
            entry_point: value.entry_point.map(|ep| ep.into()),
            layers: value
                .layers
                .into_iter()
                .map(|layer| layer.into())
                .collect::<Vec<_>>(),
        }
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

impl From<graph_v1::Layer> for Layer<IrisVectorId> {
    fn from(value: graph_v1::Layer) -> Self {
        let mut layer = Layer::new();
        for (v, nb) in value.links.into_iter() {
            layer.set_links(v.into(), nb.into());
        }
        layer
    }
}

impl From<GraphV1> for GraphMem<IrisVectorId> {
    fn from(value: GraphV1) -> Self {
        GraphMem {
            entry_point: value.entry_point.map(|ep| ep.into()),
            layers: value.layers.into_iter().map(|layer| layer.into()).collect(),
        }
    }
}

/* --------------- Conversion GraphV0 -> GraphMem ------------------- */

impl From<graph_v0::PointId> for IrisVectorId {
    fn from(value: graph_v0::PointId) -> Self {
        // The V0 format only stores a u32 ID. We assume version 0.
        IrisVectorId::new(value.0, 0)
    }
}

impl From<graph_v0::EntryPoint> for layered_graph::EntryPoint<IrisVectorId> {
    fn from(value: graph_v0::EntryPoint) -> Self {
        layered_graph::EntryPoint {
            point: value.point.into(),
            layer: value.layer,
        }
    }
}

impl From<graph_v0::Edges> for neighborhood::SortedEdgeIds<IrisVectorId> {
    fn from(value: graph_v0::Edges) -> Self {
        // V0 stores distances (PointId, (u16, u16)), we drop the distances.
        neighborhood::SortedEdgeIds(
            value
                .0
                .into_iter()
                .map(|(point_id, _distances)| point_id.into())
                .collect(),
        )
    }
}

impl From<graph_v0::Layer> for Layer<IrisVectorId> {
    fn from(value: graph_v0::Layer) -> Self {
        let mut layer = Layer::new();
        for (point_id, edges) in value.links.into_iter() {
            layer.set_links(point_id.into(), edges.into());
        }
        layer
    }
}

impl From<GraphV0> for GraphMem<IrisVectorId> {
    fn from(value: GraphV0) -> Self {
        GraphMem {
            entry_point: value.entry_point.map(|ep| ep.into()),
            layers: value.layers.into_iter().map(|layer| layer.into()).collect(),
        }
    }
}
