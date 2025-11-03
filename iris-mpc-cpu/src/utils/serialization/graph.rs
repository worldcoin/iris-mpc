use std::{fmt::Display, fs::File, io::BufReader, str::FromStr};

use clap::ValueEnum;
use eyre::Result;
use iris_mpc_common::IrisVectorId;
use serde::{Deserialize, Serialize};

use crate::{
    hnsw::{
        graph::{
            layered_graph::{self, GraphMem, Layer},
            neighborhood,
        },
        vector_store::Ref,
    },
    utils::serialization::types::{
        graph_v0::{self, read_graph_v0, GraphV0},
        graph_v1::{self, read_graph_v1, GraphV1},
        graph_v2,
    },
};

#[derive(Clone, Debug, ValueEnum, Copy, Serialize, Deserialize, PartialEq)]
pub enum GraphFormat {
    /// Binary format without cached distances
    V1,

    /// Binary format with cached `(u16, u16)` distances
    V0,

    /// Current format (unstable)
    GraphMem,
}

pub fn read_graph_current<V: Ref + Display + FromStr, R: std::io::Read>(
    reader: &mut R,
) -> eyre::Result<GraphMem<V>> {
    let data = bincode::deserialize_from(reader)?;
    Ok(data)
}

pub fn write_graph_v1<V: Ref + Display + FromStr, W: std::io::Write>(
    writer: &mut W,
    data: &GraphMem<V>,
) -> eyre::Result<()> {
    bincode::serialize_into(writer, data)?;
    Ok(())
}

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
        GraphFormat::GraphMem => Ok(read_graph_current(&mut reader)?),
    }
}

/* --------------- Conversion GraphMem -> GraphV1 ------------------- */

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
            entry_point: value.entry_point.first().cloned().map(|ep| ep.into()),
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
            entry_point: value
                .entry_point
                .map(|e| e.into())
                .into_iter()
                .collect::<Vec<_>>(),
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
            entry_point: value
                .entry_point
                .map(|ep| ep.into())
                .into_iter()
                .collect::<Vec<_>>(),
            layers: value.layers.into_iter().map(|layer| layer.into()).collect(),
        }
    }
}

/* --------------- Conversion GraphMem -> GraphV2 ------------------- */

impl From<IrisVectorId> for graph_v2::VectorId {
    fn from(value: IrisVectorId) -> Self {
        graph_v2::VectorId {
            id: value.serial_id(),
            version: value.version_id(),
        }
    }
}

impl From<layered_graph::EntryPoint<IrisVectorId>> for graph_v2::EntryPoint {
    fn from(value: layered_graph::EntryPoint<IrisVectorId>) -> Self {
        graph_v2::EntryPoint {
            point: value.point.into(),
            layer: value.layer,
        }
    }
}

impl From<neighborhood::SortedEdgeIds<IrisVectorId>> for graph_v2::EdgeIds {
    fn from(value: neighborhood::SortedEdgeIds<IrisVectorId>) -> Self {
        graph_v2::EdgeIds(value.0.into_iter().map(Into::into).collect())
    }
}

impl From<Layer<IrisVectorId>> for graph_v2::Layer {
    fn from(value: Layer<IrisVectorId>) -> Self {
        graph_v2::Layer {
            links: value
                .links
                .into_iter()
                .map(|(v, nb)| (v.into(), nb.into()))
                .collect(),
        }
    }
}

/* --------------- Conversion GraphV2 -> GraphMem ------------------- */

impl From<graph_v2::VectorId> for IrisVectorId {
    fn from(value: graph_v2::VectorId) -> Self {
        IrisVectorId::new(value.id, value.version)
    }
}

impl From<graph_v2::EntryPoint> for layered_graph::EntryPoint<IrisVectorId> {
    fn from(value: graph_v2::EntryPoint) -> Self {
        layered_graph::EntryPoint {
            point: value.point.into(),
            layer: value.layer,
        }
    }
}

impl From<graph_v2::EdgeIds> for neighborhood::SortedEdgeIds<IrisVectorId> {
    fn from(value: graph_v2::EdgeIds) -> Self {
        neighborhood::SortedEdgeIds(value.0.into_iter().map(Into::into).collect())
    }
}

impl From<graph_v2::Layer> for Layer<IrisVectorId> {
    fn from(value: graph_v2::Layer) -> Self {
        let mut layer = Layer::new();
        for (v, nb) in value.links.into_iter() {
            layer.set_links(v.into(), nb.into());
        }
        layer
    }
}

impl From<graph_v2::GraphV2> for GraphMem<IrisVectorId> {
    fn from(value: graph_v2::GraphV2) -> Self {
        GraphMem {
            // V2 uses a Vec<EntryPoint>, which matches GraphMem
            entry_point: value.entry_point.into_iter().map(|e| e.into()).collect(),
            layers: value.layers.into_iter().map(|layer| layer.into()).collect(),
        }
    }
}
