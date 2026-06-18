use std::{
    fmt::Display,
    fs::File,
    io::{BufReader, BufWriter, Cursor},
    path::Path,
};

use clap::ValueEnum;
use eyre::{bail, Result};
use iris_mpc_common::VectorId;
use serde::{Deserialize, Serialize};

use crate::{
    hnsw::graph::layered_graph::{self, GraphMem, Layer},
    utils::serialization::types::{
        graph_v0::{self, read_graph_v0, GraphV0},
        graph_v1::{self, read_graph_v1, GraphV1},
        graph_v2::{self, read_graph_v2, GraphV2},
        graph_v3::{self, read_graph_v3, GraphV3},
        graph_v4::{self, read_graph_v4, GraphV4},
    },
};

/* --------------------- Graph Serialization ------------------------ */

#[derive(Clone, Debug, ValueEnum, Copy, Serialize, Deserialize, PartialEq)]
pub enum GraphFormat {
    /// Designated current stable format for `GraphMem` serialization.
    Current,

    /// Stable graph serialization format Version 4.
    ///
    /// - Binary format
    /// - Multiple entry-points
    /// - VectorId = (SerialId, VersionId)
    /// - Contains layer checksums
    /// - Edges store VectorIds only
    /// - Sequence number for timestamping graph mutations <-- DIFF with V3
    V4,

    /// Stable graph serialization format Version 3.
    ///
    /// - Binary format
    /// - Multiple entry-points <-- DIFF with V2
    /// - VectorId = (SerialId, VersionId)
    /// - Contains layer checksums
    /// - Edges store VectorIds only
    V3,

    /// Stable graph serialization format Version 2.
    ///
    /// - Binary format
    /// - Single entry-point
    /// - VectorId = (SerialId, VersionId) <-- DIFF with V1
    /// - Contains layer checksums <-- DIFF with V1
    /// - Edges store VectorIds only
    V2,

    /// Stable graph serialization format Version 1.
    ///
    /// - Binary format
    /// - Single entry-point
    /// - VectorId = SerialId only
    /// - No layer checksums
    /// - Edges store VectorIds only <-- DIFF with V0
    V1,

    /// Stable graph serialization format Version 0.
    ///
    /// - Binary format
    /// - Single entry-point
    /// - VectorId = SerialId only
    /// - No layer checksums
    /// - Edges store VectorIds and cached `(u16, u16)` distances
    V0,

    /// Direct serialization from `GraphMem` derived format.
    ///
    /// This format type is provided for compatibility only -- please prefer use
    /// of stable serialization formats.
    Raw,
}

impl GraphFormat {
    /// Convert GraphFormat to its corresponding i32 value for storage.
    pub fn version(&self) -> i32 {
        match self {
            GraphFormat::Current | GraphFormat::V4 => 4,
            GraphFormat::V3 => 3,
            GraphFormat::V2 => 2,
            GraphFormat::V1 => 1,
            GraphFormat::V0 => 0,
            GraphFormat::Raw => {
                // Raw format should not be used for storage, but we assign a sentinel value
                -1
            }
        }
    }
}

/// Array of all concrete graph formats
pub const ALL_CONCRETE_GRAPH_FORMATS: [GraphFormat; 6] = [
    GraphFormat::V4,
    GraphFormat::V3,
    GraphFormat::V2,
    GraphFormat::V1,
    GraphFormat::V0,
    GraphFormat::Raw,
];

impl Display for GraphFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            GraphFormat::Current => "Current",
            GraphFormat::V4 => "V4",
            GraphFormat::V3 => "V3",
            GraphFormat::V2 => "V2",
            GraphFormat::V1 => "V1",
            GraphFormat::V0 => "V0",
            GraphFormat::Raw => "Raw",
        };
        write!(f, "{}", s)
    }
}

impl TryFrom<i32> for GraphFormat {
    type Error = eyre::Error;

    fn try_from(value: i32) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(GraphFormat::V0),
            1 => Ok(GraphFormat::V1),
            2 => Ok(GraphFormat::V2),
            3 => Ok(GraphFormat::V3),
            4 => Ok(GraphFormat::V4),
            -1 => Ok(GraphFormat::Raw),
            _ => Err(eyre::eyre!("unsupported graph format version: {}", value)),
        }
    }
}

/// Designated method for reading a `GraphMem`. Currently goes through the
/// `GraphV4` serialization type.
pub fn read_graph_current<R: std::io::Read>(reader: &mut R) -> eyre::Result<GraphMem> {
    let data = bincode::deserialize_from::<_, GraphV4>(reader)?.into();
    Ok(data)
}

/// Designated method for writing a `GraphMem`. Currently goes through the
/// `GraphV4` serialization type.
pub fn write_graph_current<W: std::io::Write>(writer: &mut W, data: GraphMem) -> Result<()> {
    bincode::serialize_into::<_, GraphV4>(writer, &(data.into()))?;
    Ok(())
}

/// Read a serialized `GraphMem` using the derived `Deserialize` trait
/// implementation.
///
/// _Provided for compatibility only_ -- please prefer use of stable serialization
/// formats.
pub fn read_graph_raw<R: std::io::Read>(reader: &mut R) -> Result<GraphMem> {
    let data = bincode::deserialize_from::<_, GraphMem>(reader)?;
    Ok(data)
}

/// Write a serialized `GraphMem` using the derived `Serialize` trait
/// implementation.
///
/// _Provided for compatibility only_ -- please prefer use of stable serialization
/// formats.
pub fn write_graph_raw<W: std::io::Write>(writer: &mut W, data: GraphMem) -> Result<()> {
    bincode::serialize_into::<_, GraphMem>(writer, &data)?;
    Ok(())
}

/// Read a `GraphMem` with a specified serialization format.
pub fn read_graph<R: std::io::Read>(reader: &mut R, format: GraphFormat) -> Result<GraphMem> {
    match format {
        GraphFormat::Current => read_graph_current(reader),
        GraphFormat::V4 => {
            let graph = read_graph_v4(reader)?;
            Ok(graph.into())
        }
        GraphFormat::V3 => {
            let graph = read_graph_v3(reader)?;
            Ok(graph.into())
        }
        GraphFormat::V2 => {
            let graph = read_graph_v2(reader)?;
            Ok(graph.into())
        }
        GraphFormat::V1 => {
            let graph = read_graph_v1(reader)?;
            Ok(graph.into())
        }
        GraphFormat::V0 => {
            let graph = read_graph_v0(reader)?;
            Ok(graph.into())
        }
        GraphFormat::Raw => read_graph_raw(reader),
    }
}

/// Read a `GraphMem` from file with a specified serialization format.
pub fn read_graph_from_file<P: AsRef<Path>>(path: P, format: GraphFormat) -> Result<GraphMem> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    read_graph(&mut reader, format)
}

/// Read a graph from file with unknown serialization format.  Returns deserialization
/// results using the most recent graph format for which deserialization is successful,
/// or an `Err` result if no graph format is valid.
pub fn try_read_graph_from_file<P: AsRef<Path>>(path: P) -> Result<GraphMem> {
    let data = std::fs::read(&path)?;

    for format in ALL_CONCRETE_GRAPH_FORMATS {
        let mut cursor = Cursor::new(&data);
        let graph_res = read_graph(&mut cursor, format);
        if graph_res.is_ok() {
            return graph_res;
        }
    }

    bail!("Unable to deserialize graph from file");
}

/// Attempt to deserialize the provided `data` slice into graphs using all
/// available graph formats, returning a list of formats for which
/// deserialization was successful.
pub fn check_valid_graph_formats(data: &[u8]) -> Vec<GraphFormat> {
    let mut valid_formats = Vec::new();

    for format in ALL_CONCRETE_GRAPH_FORMATS {
        let mut cursor = Cursor::new(data);
        let graph_res = read_graph(&mut cursor, format);
        if graph_res.is_ok() {
            valid_formats.push(format);
        }
    }

    valid_formats
}

/// Write a `GraphMem` to file using the `GraphFormat::Current` serialization
/// format, currently `GraphV4`.
pub fn write_graph_to_file<P: AsRef<Path>>(path: P, data: GraphMem) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    write_graph_current(&mut writer, data)
}

/* ------------------ Graph Pair Serialization ---------------------- */

/// Method to read a pair of serialized structs of the same type.
fn read_pair<R: std::io::Read + ?Sized, G: for<'a> Deserialize<'a>>(
    reader: &mut R,
) -> Result<[G; 2]> {
    let data = bincode::deserialize_from(reader)?;
    Ok(data)
}

/// Designated method for reading a pair of `GraphMem` structs. Currently goes
/// through the `GraphV4` serialization type.
pub fn read_graph_pair_current<R: std::io::Read + ?Sized>(reader: &mut R) -> Result<[GraphMem; 2]> {
    let data = read_pair::<_, GraphV4>(reader)?.map(|graph| graph.into());
    Ok(data)
}

/// Designated method for writing a pair of `GraphMem` structs. Currently goes
/// through the `GraphV4` serialization type.
pub fn write_graph_pair_current<W: std::io::Write>(
    writer: &mut W,
    data: [GraphMem; 2],
) -> Result<()> {
    let data = data.map(|graph| graph.into());
    bincode::serialize_into::<_, [GraphV4; 2]>(writer, &data)?;
    Ok(())
}

/// Read a serialized pair of serialized `GraphMem` structs using the derived
/// `Deserialize` trait implementation.
///
/// _Provided for compatibility only_ -- please prefer use of stable serialization
/// formats.
pub fn read_graph_pair_raw<R: std::io::Read + ?Sized>(reader: &mut R) -> Result<[GraphMem; 2]> {
    let data = read_pair::<_, GraphMem>(reader)?;
    Ok(data)
}

/// Write a pair of serialized `GraphMem` structs using the derived
/// `Deserialize` trait implementation.
///
/// _Provided for compatibility only_ -- please prefer use of stable serialization
/// formats.
pub fn write_graph_pair_raw<W: std::io::Write>(writer: &mut W, data: [GraphMem; 2]) -> Result<()> {
    bincode::serialize_into::<_, [GraphMem; 2]>(writer, &data)?;
    Ok(())
}

/// Read a pair of `GraphMem` structs with a specified serialization format.
pub fn read_graph_pair<R: std::io::Read + ?Sized>(
    reader: &mut R,
    format: GraphFormat,
) -> Result<[GraphMem; 2]> {
    match format {
        GraphFormat::Current => read_graph_pair_current(reader),
        GraphFormat::V4 => {
            let graphs = read_pair::<_, GraphV4>(reader)?;
            Ok(graphs.map(|graph| graph.into()))
        }
        GraphFormat::V3 => {
            let graphs = read_pair::<_, GraphV3>(reader)?;
            Ok(graphs.map(|graph| graph.into()))
        }
        GraphFormat::V2 => {
            let graphs = read_pair::<_, GraphV2>(reader)?;
            Ok(graphs.map(|graph| graph.into()))
        }
        GraphFormat::V1 => {
            let graphs = read_pair::<_, GraphV1>(reader)?;
            Ok(graphs.map(|graph| graph.into()))
        }
        GraphFormat::V0 => {
            let graphs = read_pair::<_, GraphV0>(reader)?;
            Ok(graphs.map(|graph| graph.into()))
        }
        GraphFormat::Raw => read_graph_pair_raw(reader),
    }
}

/// Read a pair of `GraphMem` structs from file with a specified graph
/// serialization format.
pub fn read_graph_pair_from_file<P: AsRef<Path>>(
    path: P,
    format: GraphFormat,
) -> Result<[GraphMem; 2]> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    read_graph_pair(&mut reader, format)
}

/// Read a graph pair from file with unknown serialization format.  Returns
/// deserialization results using the most recent graph format for which
/// deserialization is successful, or an `Err` result if no graph format is
/// valid.
pub fn try_read_graph_pair_from_file<P: AsRef<Path>>(path: P) -> Result<[GraphMem; 2]> {
    let data = std::fs::read(&path)?;

    for format in ALL_CONCRETE_GRAPH_FORMATS {
        let mut cursor = Cursor::new(&data);
        let graph_pair_res = read_graph_pair(&mut cursor, format);
        if graph_pair_res.is_ok() {
            return graph_pair_res;
        }
    }

    bail!("Unable to deserialize graph pair from file");
}

/// Attempt to deserialize the provided `data` slice into graph pairs using all
/// available graph formats, returning a list of formats for which
/// deserialization was successful.
pub fn check_valid_graph_pair_formats(data: &[u8]) -> Vec<GraphFormat> {
    let mut valid_formats = Vec::new();

    for format in ALL_CONCRETE_GRAPH_FORMATS {
        let mut cursor = Cursor::new(data);
        let graph_res = read_graph_pair(&mut cursor, format);
        if graph_res.is_ok() {
            valid_formats.push(format);
        }
    }

    valid_formats
}

/// Write a pair of `GraphMem` structs to file using the `GraphFormat::Current`
/// graph serialization format, currently `GraphV4`.
pub fn write_graph_pair_to_file<P: AsRef<Path>>(path: P, data: [GraphMem; 2]) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    write_graph_pair_current(&mut writer, data)
}

/* --------------- Conversion GraphV0 -> GraphMem ------------------- */

impl From<graph_v0::PointId> for VectorId {
    fn from(value: graph_v0::PointId) -> Self {
        // The V0 format only stores a u32 ID. We assume version 0.
        VectorId::new(value.0, 0)
    }
}

impl From<graph_v0::EntryPoint> for layered_graph::EntryPoint {
    fn from(value: graph_v0::EntryPoint) -> Self {
        layered_graph::EntryPoint {
            point: value.point.into(),
            layer: value.layer,
        }
    }
}

impl From<graph_v0::Layer> for Layer {
    fn from(value: graph_v0::Layer) -> Self {
        let mut layer = Layer::new();
        for (point_id, edges) in value.links.into_iter() {
            layer.set_links(
                point_id.into(),
                edges.0.into_iter().map(|x| x.0.into()).collect(),
            );
        }
        layer
    }
}

impl From<graph_v0::GraphV0> for GraphMem {
    fn from(value: GraphV0) -> Self {
        GraphMem {
            entry_points: value
                .entry_point
                .map(|ep| ep.into())
                .into_iter()
                .collect::<Vec<_>>(),
            layers: value.layers.into_iter().map(|layer| layer.into()).collect(),
            last_update_seq_no: 0,
        }
    }
}

/* --------------- Conversion GraphV1 -> GraphMem ------------------- */

impl From<graph_v1::VectorId> for VectorId {
    fn from(value: graph_v1::VectorId) -> Self {
        VectorId::new(value.0, 0)
    }
}

impl From<graph_v1::EntryPoint> for layered_graph::EntryPoint {
    fn from(value: graph_v1::EntryPoint) -> Self {
        layered_graph::EntryPoint {
            point: value.point.into(),
            layer: value.layer,
        }
    }
}

impl From<graph_v1::Layer> for Layer {
    fn from(value: graph_v1::Layer) -> Self {
        let mut layer = Layer::new();
        for (v, nb) in value.links.into_iter() {
            layer.set_links(
                VectorId::new(v.0, 1),
                nb.0.into_iter().map(|x| VectorId::new(x.0, 1)).collect(),
            );
        }
        layer
    }
}

impl From<graph_v1::GraphV1> for GraphMem {
    fn from(value: GraphV1) -> Self {
        GraphMem {
            entry_points: value
                .entry_point
                .map(|e| e.into())
                .into_iter()
                .collect::<Vec<_>>(),
            layers: value.layers.into_iter().map(|layer| layer.into()).collect(),
            last_update_seq_no: 0,
        }
    }
}

/* --------------- Conversion GraphV2 -> GraphMem ------------------- */

impl From<graph_v2::VectorId> for VectorId {
    fn from(value: graph_v2::VectorId) -> Self {
        VectorId::new(value.id, value.version)
    }
}

impl From<graph_v2::EntryPoint> for layered_graph::EntryPoint {
    fn from(value: graph_v2::EntryPoint) -> Self {
        layered_graph::EntryPoint {
            point: value.point.into(),
            layer: value.layer,
        }
    }
}

impl From<graph_v2::Layer> for Layer {
    fn from(value: graph_v2::Layer) -> Self {
        let mut layer = Layer::new();

        // value.set_hash is ignored;
        // instead the set_hash is recomputed implicitly in the set_links calls
        for (v, nb) in value.links.into_iter() {
            layer.set_links(v.into(), nb.0.into_iter().map(|x| x.into()).collect());
        }
        layer
    }
}

impl From<graph_v2::GraphV2> for GraphMem {
    fn from(value: graph_v2::GraphV2) -> Self {
        GraphMem {
            // GraphMem uses a Vec<EntryPoint>, V2 uses Option<EntryPoint>.
            entry_points: value
                .entry_point
                .map(|e| e.into())
                .into_iter()
                .collect::<Vec<_>>(),
            layers: value.layers.into_iter().map(|layer| layer.into()).collect(),
            last_update_seq_no: 0,
        }
    }
}

/* --------------- Conversion GraphV3 -> GraphMem ------------------- */

impl From<graph_v3::VectorId> for VectorId {
    fn from(value: graph_v3::VectorId) -> Self {
        VectorId::new(value.id, value.version)
    }
}

impl From<graph_v3::EntryPoint> for layered_graph::EntryPoint {
    fn from(value: graph_v3::EntryPoint) -> Self {
        layered_graph::EntryPoint {
            point: value.point.into(),
            layer: value.layer,
        }
    }
}

impl From<graph_v3::Layer> for Layer {
    fn from(value: graph_v3::Layer) -> Self {
        let mut layer = Layer::new();
        for (v, nb) in value.links.into_iter() {
            layer.set_links(v.into(), nb.0.into_iter().map(|x| x.into()).collect());
        }
        layer
    }
}

impl From<graph_v3::GraphV3> for GraphMem {
    fn from(value: graph_v3::GraphV3) -> Self {
        GraphMem {
            // V3 uses a Vec<EntryPoint>, which matches GraphMem
            entry_points: value.entry_point.into_iter().map(|e| e.into()).collect(),
            layers: value.layers.into_iter().map(|layer| layer.into()).collect(),
            last_update_seq_no: 0,
        }
    }
}

/* --------------- Conversion GraphV4 -> GraphMem ------------------- */

impl From<graph_v4::VectorId> for VectorId {
    fn from(value: graph_v4::VectorId) -> Self {
        VectorId::new(value.id, value.version)
    }
}

impl From<graph_v4::EntryPoint> for layered_graph::EntryPoint {
    fn from(value: graph_v4::EntryPoint) -> Self {
        layered_graph::EntryPoint {
            point: value.point.into(),
            layer: value.layer,
        }
    }
}

impl From<graph_v4::Layer> for Layer {
    fn from(value: graph_v4::Layer) -> Self {
        let mut layer = Layer::new();
        for (v, nb) in value.links.into_iter() {
            layer.set_links(v.into(), nb.0.into_iter().map(|x| x.into()).collect());
        }
        layer
    }
}

impl From<graph_v4::GraphV4> for GraphMem {
    fn from(value: graph_v4::GraphV4) -> Self {
        GraphMem {
            entry_points: value.entry_points.into_iter().map(|e| e.into()).collect(),
            layers: value.layers.into_iter().map(|layer| layer.into()).collect(),
            last_update_seq_no: value.last_update_seq_no,
        }
    }
}

/* --------------- Conversion GraphMem -> GraphV4 ------------------- */

impl From<VectorId> for graph_v4::VectorId {
    fn from(value: VectorId) -> Self {
        graph_v4::VectorId {
            id: value.serial_id(),
            version: value.version_id(),
        }
    }
}

impl From<layered_graph::EntryPoint> for graph_v4::EntryPoint {
    fn from(value: layered_graph::EntryPoint) -> Self {
        graph_v4::EntryPoint {
            point: value.point.into(),
            layer: value.layer,
        }
    }
}

impl From<Layer> for graph_v4::Layer {
    fn from(value: Layer) -> Self {
        let set_hash = value.checksum();
        graph_v4::Layer {
            links: value
                .links
                .into_iter()
                .map(|(v, nb)| {
                    (
                        v.into(),
                        graph_v4::EdgeIds(nb.into_iter().map(|x| x.into()).collect()),
                    )
                })
                .collect(),
            set_hash,
        }
    }
}

impl From<GraphMem> for graph_v4::GraphV4 {
    fn from(value: GraphMem) -> Self {
        graph_v4::GraphV4 {
            entry_points: value.entry_points.into_iter().map(|ep| ep.into()).collect(),
            layers: value.layers.into_iter().map(|layer| layer.into()).collect(),
            last_update_seq_no: value.last_update_seq_no,
        }
    }
}

/* ----------- Streaming Deserialization (single-copy path) ---------- */

/// Read a pair of [`GraphMem`] structs layer-by-layer into [`GraphMem`]
/// directly, without ever materialising the full intermediate `GraphVN`
/// struct alongside the destination graph.
///
/// For V4/Current each [`graph_v4::Layer`] (one `HashMap`) is read
/// entry-by-entry and fed into [`Layer::set_links`] immediately, so peak
/// transient memory is roughly one layer's worth of link data rather than
/// one entire graph.  V3 uses the same path — its layer binary layout is
/// identical to V4 (same field types, no `last_update_seq_no`).
///
/// V0/V1/V2/Raw fall back to [`read_graph_pair`]; those are rare migration
/// paths and the graphs tend to be smaller.
pub fn read_graph_pair_streaming<R: std::io::Read + ?Sized>(
    reader: &mut R,
    format: GraphFormat,
) -> Result<[GraphMem; 2]> {
    match format {
        GraphFormat::Current | GraphFormat::V4 => Ok([
            read_graph_v4_streaming(reader)?,
            read_graph_v4_streaming(reader)?,
        ]),
        GraphFormat::V3 => Ok([
            read_graph_v3_streaming(reader)?,
            read_graph_v3_streaming(reader)?,
        ]),
        _ => read_graph_pair(reader, format),
    }
}

/// Streaming reader for a single `GraphV4`.
///
/// Reads `entry_points`, then deserialises each `Layer` individually via
/// [`read_hashed_layer_streaming`], then reads `last_update_seq_no`.  Each
/// layer is converted to [`Layer`] and appended before the
/// next layer is read, so no two full-graph copies coexist.
fn read_graph_v4_streaming<R: std::io::Read + ?Sized>(reader: &mut R) -> Result<GraphMem> {
    let entry_points: Vec<graph_v4::EntryPoint> = bincode::deserialize_from(&mut *reader)
        .map_err(|e| eyre::eyre!("v4 streaming entry_points: {e}"))?;

    let layer_count_u64: u64 = bincode::deserialize_from(&mut *reader)
        .map_err(|e| eyre::eyre!("v4 streaming layer_count: {e}"))?;
    let layer_count = usize::try_from(layer_count_u64)
        .map_err(|_| eyre::eyre!("v4 streaming layer_count overflows usize: {layer_count_u64}"))?;
    let mut layers = Vec::with_capacity(layer_count);
    for i in 0..layer_count {
        layers.push(
            read_hashed_layer_streaming(reader)
                .map_err(|e| eyre::eyre!("v4 streaming layer {i}: {e}"))?,
        );
    }

    let last_update_seq_no: u64 = bincode::deserialize_from(&mut *reader)
        .map_err(|e| eyre::eyre!("v4 streaming last_update_seq_no: {e}"))?;

    Ok(GraphMem {
        entry_points: entry_points.into_iter().map(|e| e.into()).collect(),
        layers,
        last_update_seq_no,
    })
}

/// Streaming reader for a single `GraphV3`.
///
/// V3 differs from V4 only in having `entry_point` (same binary layout) and
/// no `last_update_seq_no` field.  Layers share the same on-wire layout as
/// V4 (see [`read_hashed_layer_streaming`]).
fn read_graph_v3_streaming<R: std::io::Read + ?Sized>(reader: &mut R) -> Result<GraphMem> {
    // Field name is `entry_point` in GraphV3 vs `entry_points` in GraphV4,
    // but bincode uses positional encoding so the bytes are identical.
    let entry_points: Vec<graph_v3::EntryPoint> = bincode::deserialize_from(&mut *reader)
        .map_err(|e| eyre::eyre!("v3 streaming entry_points: {e}"))?;

    let layer_count_u64: u64 = bincode::deserialize_from(&mut *reader)
        .map_err(|e| eyre::eyre!("v3 streaming layer_count: {e}"))?;
    let layer_count = usize::try_from(layer_count_u64)
        .map_err(|_| eyre::eyre!("v3 streaming layer_count overflows usize: {layer_count_u64}"))?;
    let mut layers = Vec::with_capacity(layer_count);
    for i in 0..layer_count {
        layers.push(
            read_hashed_layer_streaming(reader)
                .map_err(|e| eyre::eyre!("v3 streaming layer {i}: {e}"))?,
        );
    }

    // V3 has no `last_update_seq_no`; default to 0.
    Ok(GraphMem {
        entry_points: entry_points.into_iter().map(|e| e.into()).collect(),
        layers,
        last_update_seq_no: 0,
    })
}

/// Deserialise one layer entry-by-entry into [`Layer`].
///
/// Works for both V3 and V4 layers because their on-wire layout is
/// identical: both store `HashMap<{u32, i16}, Vec<{u32, i16}>>`
/// (bincode: `u64 count` + N × `(VectorId, EdgeIds)`) followed by a
/// `u64 set_hash`.  We decode using `graph_v4` types — safe because
/// `graph_v3::VectorId` and `graph_v4::VectorId` are the same struct.
///
/// The serialised `set_hash` is read and discarded; [`Layer::set_links`]
/// recomputes it incrementally as each edge list is inserted.
fn read_hashed_layer_streaming<R: std::io::Read + ?Sized>(reader: &mut R) -> Result<Layer> {
    // HashMap bincode layout: u64 count + count × (key, value)
    let link_count: u64 =
        bincode::deserialize_from(&mut *reader).map_err(|e| eyre::eyre!("link_count: {e}"))?;

    let mut layer = Layer::new();
    for _ in 0..link_count {
        let v: graph_v4::VectorId =
            bincode::deserialize_from(&mut *reader).map_err(|e| eyre::eyre!("VectorId: {e}"))?;
        let edges: graph_v4::EdgeIds =
            bincode::deserialize_from(&mut *reader).map_err(|e| eyre::eyre!("EdgeIds: {e}"))?;
        layer.set_links(
            v.into(),
            edges
                .0
                .into_iter()
                .map(|x: graph_v4::VectorId| x.into())
                .collect(),
        );
    }

    // Discard the stored set_hash; Layer::set_links already recomputed it.
    let _: u64 =
        bincode::deserialize_from(&mut *reader).map_err(|e| eyre::eyre!("set_hash: {e}"))?;

    Ok(layer)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Multi-node layer inserted out of key order, so `HashMap` iteration order
    /// is overwhelmingly unlikely to coincide with sorted order — i.e. the test
    /// actually exercises the link sorting.
    fn sample_graph() -> GraphMem {
        let v = VectorId::from_serial_id;
        let mut layer = Layer::new();
        for &n in &[7u32, 3, 9, 1, 5, 8, 2, 6, 4] {
            layer.set_links(v(n), vec![v(n + 1), v(n + 2)]);
        }
        let mut g = GraphMem::new();
        g.entry_points = vec![layered_graph::EntryPoint {
            point: v(1),
            layer: 0,
        }];
        g.layers = vec![layer];
        g
    }

    /// The `Current` (GraphV4) file format must serialize byte-identically to the
    /// raw `GraphMem` wire format used by `upload_graph_checkpoint` and the
    /// materializer. If they drift, checkpoints minted via `graph-utils` won't
    /// match those minted by the sidecar and cross-party BLAKE3 consensus breaks.
    #[test]
    fn current_serialization_matches_raw_graphmem() {
        let g = sample_graph();
        let mut cur = Vec::new();
        write_graph_current(&mut cur, g.clone()).unwrap();
        let mut raw = Vec::new();
        write_graph_raw(&mut raw, g).unwrap();
        assert_eq!(
            cur, raw,
            "GraphV4/Current serialization diverged from raw GraphMem bytes"
        );
    }
}
