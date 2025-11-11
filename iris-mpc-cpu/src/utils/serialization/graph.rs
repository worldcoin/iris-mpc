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
        graph_v2::{self, read_graph_v2},
        graph_v3::{self, read_graph_v3},
    },
};

#[derive(Clone, Debug, ValueEnum, Copy, Serialize, Deserialize, PartialEq)]
pub enum GraphFormat {
    /// Current format (unstable)
    GraphMem,

    /// - Binary format
    /// - Multiple entry-points <-- DIFF with V2
    /// - VectorId = (SerialId, VersionId)
    /// - Contains layer checksums
    /// - Edges store VectorIds only
    V3,

    /// - Binary format
    /// - Single entry-point
    /// - VectorId = (SerialId, VersionId) <-- DIFF with V1
    /// - Contains layer checksums <-- DIFF with V1
    /// - Edges store VectorIds only
    V2,

    /// - Binary format
    /// - Single entry-point
    /// - VectorId = SerialId only
    /// - No layer checksums
    /// - Edges store VectorIds only <-- DIFF with V0
    V1,

    /// - Binary format
    /// - Single entry-point
    /// - VectorId = SerialId only
    /// - No layer checksums
    /// - Edges store VectorIds and cached `(u16, u16)` distances
    V0,
}

// Designated method for writing a GraphMem
// Currently goes through V3
pub fn write_graph_current<W: std::io::Write>(
    writer: &mut W,
    data: GraphMem<IrisVectorId>,
) -> eyre::Result<()> {
    bincode::serialize_into::<_, graph_v3::GraphV3>(writer, &(data.into()))?;
    Ok(())
}

// Specifically write to file
pub fn write_graph_to_file<P: AsRef<std::path::Path>>(
    path: P,
    data: GraphMem<IrisVectorId>,
) -> eyre::Result<()> {
    let file = File::open(path)?;
    let mut writer = BufWriter::new(file);
    write_graph_current(&mut writer, data)
}

// Designated method for reading a GraphMem
// Currently goes through V3
pub fn read_graph_current<R: std::io::Read>(
    reader: &mut R,
) -> eyre::Result<GraphMem<IrisVectorId>> {
    let data = bincode::deserialize_from::<_, graph_v3::GraphV3>(reader)?.into();
    Ok(data)
}

pub fn read_graph_from_file<P: AsRef<std::path::Path>>(
    path: P,
    format: GraphFormat,
) -> Result<GraphMem<IrisVectorId>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    match format {
        GraphFormat::V3 => {
            let graph = read_graph_v3(&mut reader)?;
            Ok(graph.into())
        }
        GraphFormat::V2 => {
            let graph = read_graph_v2(&mut reader)?;
            Ok(graph.into())
        }
        GraphFormat::V1 => {
            let graph = read_graph_v1(&mut reader)?;
            Ok(graph.into())
        }
        GraphFormat::V0 => {
            let graph = read_graph_v0(&mut reader)?;
            Ok(graph.into())
        }
        GraphFormat::GraphMem => read_graph_current(&mut reader),
    }
}

pub fn read_graph_pair<R: std::io::Read, G: for<'a> Deserialize<'a>>(
    reader: &mut R,
) -> eyre::Result<[G; 2]> {
    let data = bincode::deserialize_from(reader)?;
    Ok(data)
}

pub fn write_graph_pair_current<W: std::io::Write>(
    writer: &mut W,
    data: [GraphMem<IrisVectorId>; 2],
) -> eyre::Result<()> {
    let data = data.map(|graph| graph.into());
    bincode::serialize_into::<_, [graph_v3::GraphV3; 2]>(writer, &data)?;
    Ok(())
}

pub fn read_graph_pair_current<R: std::io::Read>(
    reader: &mut R,
) -> eyre::Result<[GraphMem<IrisVectorId>; 2]> {
    let data = read_graph_pair::<_, graph_v3::GraphV3>(reader)?.map(|graph| graph.into());
    Ok(data)
}
pub fn read_graph_pair_from_file<P: AsRef<std::path::Path>>(
    path: P,
    format: GraphFormat,
) -> Result<[GraphMem<IrisVectorId>; 2]> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    match format {
        GraphFormat::V3 => {
            let graphs = read_graph_pair::<_, graph_v3::GraphV3>(&mut reader)?;
            Ok(graphs.map(|graph| graph.into()))
        }
        GraphFormat::V2 => {
            let graphs = read_graph_pair::<_, graph_v2::GraphV2>(&mut reader)?;
            Ok(graphs.map(|graph| graph.into()))
        }
        GraphFormat::V1 => {
            let graphs = read_graph_pair::<_, graph_v1::GraphV1>(&mut reader)?;
            Ok(graphs.map(|graph| graph.into()))
        }
        GraphFormat::V0 => {
            let graphs = read_graph_pair::<_, graph_v0::GraphV0>(&mut reader)?;
            Ok(graphs.map(|graph| graph.into()))
        }
        GraphFormat::GraphMem => read_graph_pair_current(&mut reader),
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

/* --------------- Conversion GraphV1 -> GraphMem ------------------- */

impl From<graph_v1::VectorId> for IrisVectorId {
    fn from(value: graph_v1::VectorId) -> Self {
        IrisVectorId::new(value.0, 0)
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

        // value.set_hash is ignored;
        // instead the set_hash is recomputed implicitly in the set_links calls
        for (v, nb) in value.links.into_iter() {
            layer.set_links(v.into(), nb.into());
        }
        layer
    }
}

impl From<graph_v2::GraphV2> for GraphMem<IrisVectorId> {
    fn from(value: graph_v2::GraphV2) -> Self {
        GraphMem {
            // GraphMem uses a Vec<EntryPoint>, V2 uses Option<EntryPoint>.
            entry_point: value
                .entry_point
                .map(|e| e.into())
                .into_iter()
                .collect::<Vec<_>>(),
            layers: value.layers.into_iter().map(|layer| layer.into()).collect(),
        }
    }
}

/* --------------- Conversion GraphV3 -> GraphMem ------------------- */

impl From<graph_v3::VectorId> for IrisVectorId {
    fn from(value: graph_v3::VectorId) -> Self {
        IrisVectorId::new(value.id, value.version)
    }
}

impl From<graph_v3::EntryPoint> for layered_graph::EntryPoint<IrisVectorId> {
    fn from(value: graph_v3::EntryPoint) -> Self {
        layered_graph::EntryPoint {
            point: value.point.into(),
            layer: value.layer,
        }
    }
}

impl From<graph_v3::EdgeIds> for neighborhood::SortedEdgeIds<IrisVectorId> {
    fn from(value: graph_v3::EdgeIds) -> Self {
        neighborhood::SortedEdgeIds(value.0.into_iter().map(Into::into).collect())
    }
}

impl From<graph_v3::Layer> for Layer<IrisVectorId> {
    fn from(value: graph_v3::Layer) -> Self {
        let mut layer = Layer::new();
        for (v, nb) in value.links.into_iter() {
            layer.set_links(v.into(), nb.into());
        }
        layer
    }
}

impl From<graph_v3::GraphV3> for GraphMem<IrisVectorId> {
    fn from(value: graph_v3::GraphV3) -> Self {
        GraphMem {
            // V3 uses a Vec<EntryPoint>, which matches GraphMem
            entry_point: value.entry_point.into_iter().map(|e| e.into()).collect(),
            layers: value.layers.into_iter().map(|layer| layer.into()).collect(),
        }
    }
}

/* --------------- Conversion GraphMem -> GraphV3 ------------------- */

impl From<IrisVectorId> for graph_v3::VectorId {
    fn from(value: IrisVectorId) -> Self {
        graph_v3::VectorId {
            id: value.serial_id(),
            version: value.version_id(),
        }
    }
}

impl From<layered_graph::EntryPoint<IrisVectorId>> for graph_v3::EntryPoint {
    fn from(value: layered_graph::EntryPoint<IrisVectorId>) -> Self {
        graph_v3::EntryPoint {
            point: value.point.into(),
            layer: value.layer,
        }
    }
}

impl From<neighborhood::SortedEdgeIds<IrisVectorId>> for graph_v3::EdgeIds {
    fn from(value: neighborhood::SortedEdgeIds<IrisVectorId>) -> Self {
        graph_v3::EdgeIds(value.0.into_iter().map(Into::into).collect())
    }
}

impl From<Layer<IrisVectorId>> for graph_v3::Layer {
    fn from(value: Layer<IrisVectorId>) -> Self {
        graph_v3::Layer {
            links: value
                .links
                .into_iter()
                .map(|(v, nb)| (v.into(), nb.into()))
                .collect(),
            // This is ignored when deserializing
            // But it needs to exist
            set_hash: 0,
        }
    }
}

impl From<GraphMem<IrisVectorId>> for graph_v3::GraphV3 {
    fn from(value: GraphMem<IrisVectorId>) -> Self {
        graph_v3::GraphV3 {
            entry_point: value.entry_point.into_iter().map(|ep| ep.into()).collect(),
            layers: value.layers.into_iter().map(|layer| layer.into()).collect(),
        }
    }
}
