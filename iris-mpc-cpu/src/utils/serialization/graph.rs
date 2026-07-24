use std::{
    collections::{HashMap, HashSet},
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
    hnsw::graph::{
        encoded_neighborhood::EncodedNeighborhood,
        layered_graph::{self, GraphMem, Layer, NodeInit},
    },
    utils::serialization::types::{
        graph_v0::{self, read_graph_v0, GraphV0},
        graph_v1::{self, read_graph_v1, GraphV1},
        graph_v2::{self, read_graph_v2, GraphV2},
        graph_v3::{self, read_graph_v3, GraphV3},
        graph_v4::{self, read_graph_v4, GraphV4},
        graph_v5::{self, read_graph_v5, GraphV5},
    },
};

/* --------------------- Graph Serialization ------------------------ */

#[derive(Clone, Debug, ValueEnum, Copy, Serialize, Deserialize, PartialEq)]
pub enum GraphFormat {
    /// Designated current stable format for `GraphMem` serialization.
    Current,

    /// Stable graph serialization format Version 5.
    ///
    /// - Binary format
    /// - Multiple entry-points
    /// - VectorId = SerialId
    /// - Contains layer checksums
    /// - Edges store VectorIds only
    /// - Sequence number for timestamping graph mutations
    /// - Neighborhoods contain the last updated sequence number <-- DIFF with V4
    /// - Contains a HashMap of (VectorId, sequence number) to invalidate old edges <-- DIFF with V4
    /// - Neighbor lists are Rice-coded blobs (`EncodedNeighborhood`) <-- DIFF with V4
    V5,

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
            GraphFormat::Current | GraphFormat::V5 => 5,
            GraphFormat::V4 => 4,
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
pub const ALL_CONCRETE_GRAPH_FORMATS: [GraphFormat; 7] = [
    GraphFormat::V5,
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
            GraphFormat::V5 => "V5",
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
            5 => Ok(GraphFormat::V5),
            -1 => Ok(GraphFormat::Raw),
            _ => Err(eyre::eyre!("unsupported graph format version: {}", value)),
        }
    }
}

/// Designated method for reading a `GraphMem`. Currently goes through the
/// `GraphV5` serialization type.
pub fn read_graph_current<R: std::io::Read>(reader: &mut R) -> eyre::Result<GraphMem> {
    let data = bincode::deserialize_from::<_, GraphV5>(reader)?.into();
    Ok(data)
}

/// Designated method for writing a `GraphMem`. Currently goes through the
/// `GraphV5` serialization type.
pub fn write_graph_current<W: std::io::Write>(writer: &mut W, data: GraphMem) -> Result<()> {
    bincode::serialize_into::<_, GraphV5>(writer, &(data.into()))?;
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
        GraphFormat::V5 => {
            let graph = read_graph_v5(reader)?;
            Ok(graph.into())
        }
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
/// format, currently `GraphV5`.
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

/// Convert a decoded wire-format pair into `GraphMem`s.
///
/// Deliberately serial: each `into()` drains its source entry-by-entry, so
/// peak transient memory stays at ~1×E (E = edge payload). Converting the
/// eyes in parallel would free each source on a different allocator arena
/// than it was decoded on, stranding it under glibc's per-thread arenas and
/// pushing peak RSS toward ~2×E.
fn convert_pair<G>(pair: [G; 2]) -> [GraphMem; 2]
where
    G: Into<GraphMem>,
{
    let [left, right] = pair;
    [left.into(), right.into()]
}

/// Designated method for reading a pair of `GraphMem` structs. Currently goes
/// through the `GraphV5` serialization type.
pub fn read_graph_pair_current<R: std::io::Read + ?Sized>(reader: &mut R) -> Result<[GraphMem; 2]> {
    Ok(convert_pair(read_pair::<_, GraphV5>(reader)?))
}

/// Designated method for writing a pair of `GraphMem` structs. Currently goes
/// through the `GraphV5` serialization type.
pub fn write_graph_pair_current<W: std::io::Write>(
    writer: &mut W,
    data: [GraphMem; 2],
) -> Result<()> {
    let data = data.map(|graph| graph.into());
    bincode::serialize_into::<_, [GraphV5; 2]>(writer, &data)?;
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
        GraphFormat::V5 => Ok(convert_pair(read_pair::<_, GraphV5>(reader)?)),
        GraphFormat::V4 => Ok(convert_pair(read_pair::<_, GraphV4>(reader)?)),
        GraphFormat::V3 => Ok(convert_pair(read_pair::<_, GraphV3>(reader)?)),
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

/// Inputs for read-time pruning of a legacy (V3/V4) base checkpoint.
pub struct LegacyPruneContext {
    /// Serial → current-version table from the iris store (`stream_iris_ids`).
    /// An edge to Z survives only if its stored target version equals Z's
    /// version here.
    pub version_map: HashMap<u32, i16>,
    /// Deleted serials (S3 deletion list). Dropped regardless of version: a
    /// deletion keeps the iris row, and we don't assume its version bump reaches
    /// `version_map`, so version-match alone may keep edges to a deleted serial.
    pub deleted: HashSet<u32>,
}

/// Read a `GraphMem` pair, physically dropping edges the runtime would
/// version-skip: an edge to Z survives iff Z is not deleted and the edge's
/// stored version equals Z's current version (`version_map`). Surviving nodes
/// seed the staleness clock to 0. Genesis uses this to materialize a legacy
/// V3/V4 base; V5/Current edges are version-free (skipped lazily by
/// `get_active_links`) and fall through to the plain reader.
///
/// INVARIANT: `version_map` must be identical across parties or the resulting
/// checksum diverges — relies on `version_id` being public and replicated.
pub fn read_graph_pair_pruned<R: std::io::Read + ?Sized>(
    reader: &mut R,
    format: GraphFormat,
    prune: &LegacyPruneContext,
) -> Result<[GraphMem; 2]> {
    match format {
        GraphFormat::V4 => Ok(read_pair::<_, GraphV4>(reader)?.map(|g| prune_graph_v4(g, prune))),
        GraphFormat::V3 => Ok(read_pair::<_, GraphV3>(reader)?.map(|g| prune_graph_v3(g, prune))),
        _ => read_graph_pair(reader, format),
    }
}

/// Build pruned `Layer`s and a `node_init` clock (seq 0, version from
/// `version_map`) from a legacy graph. `$layers`/`$entry_points` are moved out
/// (distinct fields → partial move). At most one version per serial matches
/// `version_map`, so multi-version stragglers collapse deterministically onto
/// the live entry. `set_links_trusted` recomputes each layer's `set_hash`.
macro_rules! legacy_prune_to_mem {
    ($layers:expr, $entry_points:expr, $last_update_seq_no:expr, $prune:expr) => {{
        let src_layers = $layers;
        let prune = $prune;
        let live_at = |id: u32, version: i16| {
            !prune.deleted.contains(&id) && prune.version_map.get(&id) == Some(&version)
        };
        let mut layers = Vec::with_capacity(src_layers.len());
        for layer in src_layers {
            let mut out = Layer::with_capacity(layer.links.len());
            for (key, edges) in layer.links {
                if !live_at(key.id, key.version) {
                    continue;
                }
                let kept: Vec<u32> = edges
                    .0
                    .into_iter()
                    .filter(|e| live_at(e.id, e.version))
                    .map(|e| e.id)
                    .collect();
                out.set_links_trusted(key.id, kept, 0);
            }
            layers.push(out);
        }
        // A kept node's key version equals its registry-current version (that's
        // what `live_at` checked), so it seeds the graph's version truth.
        let node_init = layers
            .iter()
            .flat_map(|l| l.links.keys())
            .map(|&v| {
                let version = *prune
                    .version_map
                    .get(&v)
                    .expect("pruned node must be in version_map");
                (v, NodeInit { seq_no: 0, version })
            })
            .collect();
        GraphMem::from_parts(
            $entry_points.into_iter().map(|e| e.into()).collect(),
            layers,
            $last_update_seq_no,
            node_init,
        )
    }};
}

fn prune_graph_v4(value: GraphV4, prune: &LegacyPruneContext) -> GraphMem {
    legacy_prune_to_mem!(
        value.layers,
        value.entry_points,
        value.last_update_seq_no,
        prune
    )
}

fn prune_graph_v3(value: GraphV3, prune: &LegacyPruneContext) -> GraphMem {
    // V3 has no `last_update_seq_no` and names its entry points `entry_point`.
    legacy_prune_to_mem!(value.layers, value.entry_point, 0, prune)
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
/// graph serialization format, currently `GraphV5`.
pub fn write_graph_pair_to_file<P: AsRef<Path>>(path: P, data: [GraphMem; 2]) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    write_graph_pair_current(&mut writer, data)
}

/* --------------- Conversion GraphV0 -> GraphMem ------------------- */

impl From<graph_v0::PointId> for VectorId {
    fn from(value: graph_v0::PointId) -> Self {
        VectorId::from_serial_id(value.0)
    }
}

impl From<graph_v0::EntryPoint> for layered_graph::EntryPoint {
    fn from(value: graph_v0::EntryPoint) -> Self {
        layered_graph::EntryPoint {
            point: value.point.0,
            layer: value.layer,
        }
    }
}

impl From<graph_v0::Layer> for Layer {
    fn from(value: graph_v0::Layer) -> Self {
        let mut layer = Layer::new();
        for (point_id, edges) in value.links.into_iter() {
            layer.set_links_trusted(point_id.0, edges.0.into_iter().map(|x| x.0 .0).collect(), 0);
        }
        layer
    }
}

impl From<graph_v0::GraphV0> for GraphMem {
    fn from(value: GraphV0) -> Self {
        let layers: Vec<Layer> = value.layers.into_iter().map(|layer| layer.into()).collect();
        let node_init = layers
            .iter()
            .flat_map(|l| l.links.keys())
            .map(|&v| (v, NodeInit::default()))
            .collect();
        GraphMem::from_parts(
            value
                .entry_point
                .map(|ep| ep.into())
                .into_iter()
                .collect::<Vec<_>>(),
            layers,
            0,
            node_init,
        )
    }
}

/* --------------- Conversion GraphV1 -> GraphMem ------------------- */

impl From<graph_v1::VectorId> for VectorId {
    fn from(value: graph_v1::VectorId) -> Self {
        VectorId::from_serial_id(value.0)
    }
}

impl From<graph_v1::EntryPoint> for layered_graph::EntryPoint {
    fn from(value: graph_v1::EntryPoint) -> Self {
        layered_graph::EntryPoint {
            point: value.point.0,
            layer: value.layer,
        }
    }
}

impl From<graph_v1::Layer> for Layer {
    fn from(value: graph_v1::Layer) -> Self {
        let mut layer = Layer::new();
        for (v, nb) in value.links.into_iter() {
            layer.set_links_trusted(v.0, nb.0.into_iter().map(|x| x.0).collect(), 0);
        }
        layer
    }
}

impl From<graph_v1::GraphV1> for GraphMem {
    fn from(value: GraphV1) -> Self {
        let layers: Vec<Layer> = value.layers.into_iter().map(|layer| layer.into()).collect();
        let node_init = layers
            .iter()
            .flat_map(|l| l.links.keys())
            .map(|&v| (v, NodeInit::default()))
            .collect();
        GraphMem::from_parts(
            value
                .entry_point
                .map(|e| e.into())
                .into_iter()
                .collect::<Vec<_>>(),
            layers,
            0,
            node_init,
        )
    }
}

/* --------------- Conversion GraphV2 -> GraphMem ------------------- */

impl From<graph_v2::VectorId> for VectorId {
    fn from(value: graph_v2::VectorId) -> Self {
        VectorId::from_serial_id(value.id)
    }
}

impl From<graph_v2::EntryPoint> for layered_graph::EntryPoint {
    fn from(value: graph_v2::EntryPoint) -> Self {
        layered_graph::EntryPoint {
            point: value.point.id,
            layer: value.layer,
        }
    }
}

impl From<graph_v2::Layer> for Layer {
    fn from(value: graph_v2::Layer) -> Self {
        let mut layer = Layer::new();

        // value.set_hash is ignored;
        // instead the set_hash is recomputed implicitly in the set_links_trusted calls
        for (v, nb) in value.links.into_iter() {
            layer.set_links_trusted(v.id, nb.0.into_iter().map(|x| x.id).collect(), 0);
        }
        layer
    }
}

impl From<graph_v2::GraphV2> for GraphMem {
    fn from(value: graph_v2::GraphV2) -> Self {
        let layers: Vec<Layer> = value.layers.into_iter().map(|layer| layer.into()).collect();
        let node_init = layers
            .iter()
            .flat_map(|l| l.links.keys())
            .map(|&v| (v, NodeInit::default()))
            .collect();
        // GraphMem uses a Vec<EntryPoint>, V2 uses Option<EntryPoint>.
        GraphMem::from_parts(
            value
                .entry_point
                .map(|e| e.into())
                .into_iter()
                .collect::<Vec<_>>(),
            layers,
            0,
            node_init,
        )
    }
}

/* --------------- Conversion GraphV3 -> GraphMem ------------------- */

impl From<graph_v3::VectorId> for VectorId {
    fn from(value: graph_v3::VectorId) -> Self {
        VectorId::from_serial_id(value.id)
    }
}

impl From<graph_v3::EntryPoint> for layered_graph::EntryPoint {
    fn from(value: graph_v3::EntryPoint) -> Self {
        layered_graph::EntryPoint {
            point: value.point.id,
            layer: value.layer,
        }
    }
}

impl From<graph_v3::Layer> for Layer {
    fn from(value: graph_v3::Layer) -> Self {
        // Recompute set_hash via set_links_trusted rather than trusting the stored
        // value: older checkpoints carry a set_hash from a prior fold algorithm.
        // Pre-size the map so the bulk insert doesn't rehash.
        let mut layer = Layer::with_capacity(value.links.len());
        for (v, nb) in value.links.into_iter() {
            layer.set_links_trusted(v.id, nb.0.into_iter().map(|x| x.id).collect(), 0);
        }
        layer
    }
}

impl From<graph_v3::GraphV3> for GraphMem {
    fn from(value: graph_v3::GraphV3) -> Self {
        let layers: Vec<Layer> = value.layers.into_iter().map(|layer| layer.into()).collect();
        let node_init = layers
            .iter()
            .flat_map(|l| l.links.keys())
            .map(|&v| (v, NodeInit::default()))
            .collect();
        // V3 uses a Vec<EntryPoint>, which matches GraphMem
        GraphMem::from_parts(
            value.entry_point.into_iter().map(|e| e.into()).collect(),
            layers,
            0,
            node_init,
        )
    }
}

/* --------------- Conversion GraphV4 -> GraphMem ------------------- */

impl From<graph_v4::VectorId> for VectorId {
    fn from(value: graph_v4::VectorId) -> Self {
        // V4 stored (id, version); drop the version field.
        VectorId::from_serial_id(value.id)
    }
}

impl From<graph_v4::EntryPoint> for layered_graph::EntryPoint {
    fn from(value: graph_v4::EntryPoint) -> Self {
        layered_graph::EntryPoint {
            point: value.point.id,
            layer: value.layer,
        }
    }
}

impl From<graph_v4::Layer> for Layer {
    fn from(value: graph_v4::Layer) -> Self {
        // See `From<graph_v3::Layer>`: recompute set_hash, pre-sized map.
        let mut layer = Layer::with_capacity(value.links.len());
        for (v, nb) in value.links.into_iter() {
            layer.set_links_trusted(v.id, nb.0.into_iter().map(|x| x.id).collect(), 0);
        }
        layer
    }
}

impl From<graph_v4::GraphV4> for GraphMem {
    fn from(value: graph_v4::GraphV4) -> Self {
        let layers: Vec<Layer> = value.layers.into_iter().map(|layer| layer.into()).collect();
        let node_init = layers
            .iter()
            .flat_map(|l| l.links.keys())
            .map(|&v| (v, NodeInit::default()))
            .collect();
        GraphMem::from_parts(
            value.entry_points.into_iter().map(|e| e.into()).collect(),
            layers,
            value.last_update_seq_no,
            node_init,
        )
    }
}

/* --------------- Conversion GraphMem -> GraphV4 ------------------- */

impl From<VectorId> for graph_v4::VectorId {
    fn from(value: VectorId) -> Self {
        // V4 required a version field; default to 0 when writing back.
        graph_v4::VectorId {
            id: value.serial_id(),
            version: 0,
        }
    }
}

impl From<layered_graph::EntryPoint> for graph_v4::EntryPoint {
    fn from(value: layered_graph::EntryPoint) -> Self {
        graph_v4::EntryPoint {
            point: graph_v4::VectorId {
                id: value.point,
                version: 0,
            },
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
                        graph_v4::VectorId { id: v, version: 0 },
                        graph_v4::EdgeIds(
                            nb.neighbors()
                                .iter()
                                .map(|&x| graph_v4::VectorId { id: x, version: 0 })
                                .collect(),
                        ),
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

/* --------------- Conversion GraphV5 -> GraphMem ------------------- */

impl From<graph_v5::EntryPoint> for layered_graph::EntryPoint {
    fn from(value: graph_v5::EntryPoint) -> Self {
        layered_graph::EntryPoint {
            point: value.point,
            layer: value.layer,
        }
    }
}

impl From<graph_v5::Layer> for Layer {
    fn from(value: graph_v5::Layer) -> Self {
        let mut layer = Layer::with_capacity(value.links.len());
        for (v, nb) in value.links.into_iter() {
            layer.set_links_encoded_trusted(
                v,
                EncodedNeighborhood::from_bytes(nb.neighbors.0.into_boxed_slice()),
                nb.updated_seq_no,
            );
        }
        layer
    }
}

impl From<graph_v5::NodeInit> for NodeInit {
    fn from(value: graph_v5::NodeInit) -> Self {
        NodeInit {
            seq_no: value.seq_no,
            version: value.version,
        }
    }
}

impl From<graph_v5::GraphV5> for GraphMem {
    fn from(value: GraphV5) -> Self {
        GraphMem::from_parts(
            value.entry_points.into_iter().map(|e| e.into()).collect(),
            value.layers.into_iter().map(|layer| layer.into()).collect(),
            value.last_update_seq_no,
            value
                .node_init
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect(),
        )
    }
}

/* --------------- Conversion GraphMem -> GraphV5 ------------------- */

impl From<layered_graph::EntryPoint> for graph_v5::EntryPoint {
    fn from(value: layered_graph::EntryPoint) -> Self {
        graph_v5::EntryPoint {
            point: value.point,
            layer: value.layer,
        }
    }
}

impl From<Layer> for graph_v5::Layer {
    fn from(value: Layer) -> Self {
        let set_hash = value.checksum();
        graph_v5::Layer {
            links: value
                .links
                .into_iter()
                .map(|(v, nb)| {
                    let updated_seq_no = nb.seq_no();
                    (
                        v,
                        graph_v5::Neighborhood {
                            neighbors: graph_v5::EncodedNeighborhoodBytes(
                                nb.into_encoded().into_bytes().into_vec(),
                            ),
                            updated_seq_no,
                        },
                    )
                })
                .collect(),
            set_hash,
        }
    }
}

impl From<NodeInit> for graph_v5::NodeInit {
    fn from(value: NodeInit) -> Self {
        graph_v5::NodeInit {
            seq_no: value.seq_no,
            version: value.version,
        }
    }
}

impl From<GraphMem> for graph_v5::GraphV5 {
    fn from(value: GraphMem) -> Self {
        graph_v5::GraphV5 {
            entry_points: value.entry_points.into_iter().map(|ep| ep.into()).collect(),
            layers: value.layers.into_iter().map(|layer| layer.into()).collect(),
            node_init: value
                .node_init
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect(),
            last_update_seq_no: value.last_update_seq_no,
        }
    }
}

/* ----------- Streaming Deserialization (single-copy path) ---------- */

/// Read a pair of [`GraphMem`] structs from a byte stream.
///
/// `bincode::deserialize_from` pulls bytes incrementally from `reader`, and the
/// `From<GraphVN>` conversions move neighbor `Vec`s into the destination graph,
/// so peak transient memory is roughly the wire payload — no second full-graph
/// copy. The streaming win comes from byte-streaming the input `reader`, not
/// from the decode.
pub fn read_graph_pair_streaming<R: std::io::Read + ?Sized>(
    reader: &mut R,
    format: GraphFormat,
) -> Result<[GraphMem; 2]> {
    read_graph_pair(reader, format)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Multi-node layer inserted out of key order, so `HashMap` iteration order
    /// is overwhelmingly unlikely to coincide with sorted order — i.e. the test
    /// actually exercises the link sorting.
    fn sample_graph() -> GraphMem {
        let mut layer = Layer::new();
        for &n in &[7u32, 3, 9, 1, 5, 8, 2, 6, 4] {
            layer.set_links_trusted(n, vec![n + 1, n + 2], 0);
        }
        // Seed the content clock at 0 for every node, matching the From<GraphVN>
        // converters; the checksum folds node_init.
        let node_init = layer
            .get_links_map()
            .keys()
            .map(|&k| (k, NodeInit::default()))
            .collect();
        let entry_points = vec![layered_graph::EntryPoint { point: 1, layer: 0 }];
        GraphMem::from_parts(entry_points, vec![layer], 0, node_init)
    }

    /// Round-trip a `GraphMem` through the current (V5) serialization format and
    /// verify the deserialized graph compares equal to the original.
    #[test]
    fn current_v5_round_trip() {
        let g = sample_graph();
        let mut buf = Vec::new();
        write_graph_current(&mut buf, g.clone()).unwrap();
        let g2 = read_graph_current(&mut buf.as_slice()).unwrap();
        assert_eq!(
            g, g2,
            "GraphV5/Current round-trip produced a different graph"
        );
    }

    /// Checkpoints are written by serializing `[GraphMem; 2]` directly but
    /// read back as `GraphV5`; the two wire layouts must be byte-identical.
    /// The fixture's clocks are non-zero and distinct over out-of-order keys
    /// so field-order drift and BTreeMap-sort vs HashMap-iteration ordering
    /// are both observable (uniform-zero clocks would alias the orderings and
    /// pass regardless).
    #[test]
    fn graphmem_direct_write_matches_current_and_reads_back() {
        let mut layer = Layer::new();
        for (i, &n) in [7u32, 3, 9, 1, 5, 8, 2, 6, 4].iter().enumerate() {
            layer.set_links_trusted(n, vec![n + 1, n + 2], i as u64 + 1);
        }
        let node_init = layer
            .get_links_map()
            .keys()
            .map(|&k| {
                (
                    k,
                    NodeInit {
                        seq_no: k as u64 * 10 + 1,
                        version: k as i16 + 1,
                    },
                )
            })
            .collect();
        let entry_points = vec![layered_graph::EntryPoint { point: 1, layer: 0 }];
        let g = GraphMem::from_parts(entry_points, vec![layer], 42, node_init);

        let direct = bincode::serialize(&[g.clone(), g.clone()]).unwrap();
        let mut via_current = Vec::new();
        write_graph_pair_current(&mut via_current, [g.clone(), g.clone()]).unwrap();
        assert_eq!(
            direct, via_current,
            "GraphMem-direct bytes differ from write_graph_pair_current (GraphV5) bytes"
        );

        let pair = read_graph_pair(&mut Cursor::new(&direct), GraphFormat::Current).unwrap();
        for restored in pair {
            assert_eq!(
                restored, g,
                "genesis-written bytes did not read back via V5 reader"
            );
        }
    }

    /// Per-layer checksum and links survive a write→read round trip: the
    /// recomputed `set_hash` matches the source graph's.
    #[test]
    fn read_graph_pair_preserves_set_hash() {
        let g = sample_graph();
        let mut buf = Vec::new();
        write_graph_pair_current(&mut buf, [g.clone(), g.clone()]).unwrap();

        for fmt in [GraphFormat::Current, GraphFormat::V5] {
            let pair = read_graph_pair(&mut Cursor::new(&buf), fmt).unwrap();
            for restored in pair {
                assert_eq!(restored.layers.len(), g.layers.len());
                for (orig, got) in g.layers.iter().zip(restored.layers.iter()) {
                    assert_eq!(
                        orig.checksum(),
                        got.checksum(),
                        "set_hash drifted on read ({fmt:?})"
                    );
                    assert_eq!(
                        orig.get_links_map(),
                        got.get_links_map(),
                        "links drifted on read ({fmt:?})"
                    );
                }
            }
        }
    }

    /// Serialize a graph pair in the on-wire layout for `fmt`.
    fn write_pair_in_format(g: &GraphMem, fmt: GraphFormat) -> Vec<u8> {
        use crate::utils::serialization::types::{graph_v3, graph_v4};
        let mut buf = Vec::new();
        match fmt {
            GraphFormat::V4 => {
                let pair: [graph_v4::GraphV4; 2] = [g.clone().into(), g.clone().into()];
                bincode::serialize_into(&mut buf, &pair).unwrap();
            }
            GraphFormat::V3 => {
                let to_v3 = |g: &GraphMem| graph_v3::GraphV3 {
                    entry_point: g
                        .entry_points
                        .iter()
                        .map(|ep| graph_v3::EntryPoint {
                            point: graph_v3::VectorId {
                                id: ep.point,
                                version: 0,
                            },
                            layer: ep.layer,
                        })
                        .collect(),
                    layers: g
                        .layers
                        .iter()
                        .map(|layer| graph_v3::Layer {
                            links: layer
                                .get_links_map()
                                .iter()
                                .map(|(v, nb)| {
                                    (
                                        graph_v3::VectorId { id: *v, version: 0 },
                                        graph_v3::EdgeIds(
                                            nb.neighbors()
                                                .iter()
                                                .map(|x| graph_v3::VectorId { id: *x, version: 0 })
                                                .collect(),
                                        ),
                                    )
                                })
                                .collect(),
                            set_hash: layer.checksum(),
                        })
                        .collect(),
                };
                let pair = [to_v3(g), to_v3(g)];
                bincode::serialize_into(&mut buf, &pair).unwrap();
            }
            other => panic!("unsupported test format {other:?}"),
        }
        buf
    }

    /// `read_graph_pair_streaming` yields a graph pair equal to
    /// `read_graph_pair` and to the original, including per-layer `checksum()`,
    /// for every stable layer-hashed format.
    #[test]
    fn streaming_matches_derived_and_original() {
        let g = sample_graph();

        for fmt in [GraphFormat::V3, GraphFormat::V4, GraphFormat::V5] {
            let buf = if fmt == GraphFormat::V5 {
                let mut b = Vec::new();
                write_graph_pair_current(&mut b, [g.clone(), g.clone()]).unwrap();
                b
            } else {
                write_pair_in_format(&g, fmt)
            };

            let derived = read_graph_pair(&mut Cursor::new(&buf), fmt).unwrap();
            let streamed = read_graph_pair_streaming(&mut Cursor::new(&buf), fmt).unwrap();

            for (s, d) in streamed.iter().zip(derived.iter()) {
                assert_eq!(s.checksum(), d.checksum(), "streamed != derived ({fmt:?})");
                assert_eq!(s.checksum(), g.checksum(), "streamed != original ({fmt:?})");
                assert_eq!(s.layers.len(), g.layers.len());
                for (orig, got) in g.layers.iter().zip(s.layers.iter()) {
                    assert_eq!(orig.get_links_map(), got.get_links_map());
                }
            }
        }
    }

    /// Prune-at-read drops edges whose stored target version differs from the
    /// target's registry-current version, plus edges to absent targets. Every
    /// surviving node's clock seeds at seq 0 with its registry-current version.
    #[test]
    fn prune_drops_version_drifted_and_dangling_edges() {
        use crate::utils::serialization::types::graph_v4;
        let vid = |id: u32, version: i16| graph_v4::VectorId { id, version };
        // Registry says node 3 ("Z") current version is 2. Node 1 ("A") links to
        // the stale Z@1 and to an absent target 9@0 (not in the registry); node 2
        // ("B") links to the fresh Z@2.
        let mut links = HashMap::new();
        links.insert(vid(1, 0), graph_v4::EdgeIds(vec![vid(3, 1), vid(9, 0)]));
        links.insert(vid(2, 0), graph_v4::EdgeIds(vec![vid(3, 2)]));
        links.insert(vid(3, 2), graph_v4::EdgeIds(vec![]));
        let layer = graph_v4::Layer { links, set_hash: 0 };
        let g = graph_v4::GraphV4 {
            entry_points: vec![],
            layers: vec![layer],
            last_update_seq_no: 7,
        };

        let prune = LegacyPruneContext {
            version_map: HashMap::from([(1u32, 0i16), (2, 0), (3, 2)]),
            deleted: HashSet::new(),
        };
        let mem = prune_graph_v4(g, &prune);
        let m = mem.layers[0].get_links_map();
        assert!(m[&1u32].neighbors().is_empty(), "stale + dangling dropped");
        assert_eq!(m[&2u32].neighbors(), [3u32], "fresh edge kept");
        assert!(m[&3u32].neighbors().is_empty());
        assert_eq!(mem.last_update_seq_no, 7);
        for (k, version) in [(1u32, 0i16), (2, 0), (3, 2)] {
            assert_eq!(
                mem.node_init.get(&k),
                Some(&NodeInit { seq_no: 0, version }),
                "clock seeds at seq 0 with the registry-current version"
            );
        }
    }

    /// A legacy graph carrying two version entries for one serial (a `RemoveNode`
    /// straggler) collapses deterministically onto the registry-current entry:
    /// the stale entry's out-edges are dropped, inbound edges to the old version
    /// are dropped, and the serial appears once with the live entry's edges.
    /// Only one version can match the registry value, so there is no
    /// HashMap-iteration-order dependence.
    #[test]
    fn prune_collapses_multi_version_serial_to_live_entry() {
        use crate::utils::serialization::types::graph_v4;
        let vid = |id: u32, version: i16| graph_v4::VectorId { id, version };
        // Serial 3 has both a stale entry (@1, out-edge to 2) and the live entry
        // (@2, out-edge to 1); registry-current is 2. Serial 1 links to the stale
        // 3@1; serial 2 to 3@2.
        let mut links = HashMap::new();
        links.insert(vid(1, 0), graph_v4::EdgeIds(vec![vid(3, 1)]));
        links.insert(vid(2, 0), graph_v4::EdgeIds(vec![vid(3, 2)]));
        links.insert(vid(3, 1), graph_v4::EdgeIds(vec![vid(2, 0)]));
        links.insert(vid(3, 2), graph_v4::EdgeIds(vec![vid(1, 0)]));
        let g = graph_v4::GraphV4 {
            entry_points: vec![],
            layers: vec![graph_v4::Layer { links, set_hash: 0 }],
            last_update_seq_no: 0,
        };

        let prune = LegacyPruneContext {
            version_map: HashMap::from([(1u32, 0i16), (2, 0), (3, 2)]),
            deleted: HashSet::new(),
        };
        let mem = prune_graph_v4(g, &prune);
        let m = mem.layers[0].get_links_map();
        // Serial 3 present once, with the live (@2) entry's out-edges (-> 1).
        assert_eq!(m[&3u32].neighbors(), [1u32], "live entry's edges kept");
        assert!(
            m[&1u32].neighbors().is_empty(),
            "inbound edge to 3@1 dropped"
        );
        assert_eq!(m[&2u32].neighbors(), [3u32], "inbound edge to 3@2 kept");
        assert_eq!(
            mem.node_init.get(&3u32),
            Some(&NodeInit {
                seq_no: 0,
                version: 2
            })
        );
    }

    /// A deleted serial is dropped outright — node and inbound edges — even
    /// when the registry version still matches: the deletion list is the
    /// authority, not a version bump.
    #[test]
    fn prune_drops_deleted_serial_and_its_inbound_edges() {
        use crate::utils::serialization::types::graph_v4;
        let vid = |id: u32, version: i16| graph_v4::VectorId { id, version };
        // Serial 3 is deleted but present as key @5, and the registry still
        // reports 3@5 as current, so serial 1's edge 3@5 *matches* by version —
        // only the deletion list distinguishes it. Serial 2 is live.
        let mut links = HashMap::new();
        links.insert(vid(1, 0), graph_v4::EdgeIds(vec![vid(3, 5), vid(2, 0)]));
        links.insert(vid(2, 0), graph_v4::EdgeIds(vec![]));
        links.insert(vid(3, 5), graph_v4::EdgeIds(vec![vid(2, 0)]));
        let g = graph_v4::GraphV4 {
            entry_points: vec![],
            layers: vec![graph_v4::Layer { links, set_hash: 0 }],
            last_update_seq_no: 0,
        };

        let prune = LegacyPruneContext {
            version_map: HashMap::from([(1u32, 0i16), (2, 0), (3, 5)]),
            deleted: HashSet::from([3u32]),
        };
        let mem = prune_graph_v4(g, &prune);
        let m = mem.layers[0].get_links_map();
        assert!(!m.contains_key(&3u32), "deleted serial absent as a node");
        assert_eq!(m[&1u32].neighbors(), [2u32], "edge to deleted 3 dropped");
        assert!(m[&2u32].neighbors().is_empty());
        assert!(
            !mem.node_init.contains_key(&3u32),
            "deleted serial absent from clock"
        );
    }

    /// `From<graph_v3::Layer>` recomputes `set_hash` from links and ignores
    /// the stored value (older checkpoints carry an older-algorithm hash).
    #[test]
    fn from_graph_v3_layer_recomputes_ignoring_stored_set_hash() {
        use crate::utils::serialization::types::graph_v3;
        let wire = |id: u32| graph_v3::VectorId { id, version: 0 };
        // Ordered slice (not HashMap iteration) per the iter_over_hash_type lint.
        let entries: [(u32, Vec<u32>); 3] = [(7, vec![8, 9]), (3, vec![4]), (5, vec![])];
        let mut links: std::collections::HashMap<graph_v3::VectorId, graph_v3::EdgeIds> =
            std::collections::HashMap::new();
        let mut reference = Layer::new();
        for (n, nbs) in &entries {
            let wnbs: Vec<graph_v3::VectorId> = nbs.iter().map(|&x| wire(x)).collect();
            reference.set_links_trusted(wire(*n).id, wnbs.iter().map(|x| x.id).collect(), 0);
            links.insert(wire(*n), graph_v3::EdgeIds(wnbs));
        }
        let recomputed = reference.checksum();

        let got: Layer = graph_v3::Layer {
            links,
            // Deliberately wrong stored value; the conversion must ignore it.
            set_hash: recomputed.wrapping_add(0xDEAD_BEEF),
        }
        .into();

        assert_eq!(
            got.checksum(),
            recomputed,
            "V3 conversion must recompute set_hash, not trust the stored value"
        );
        assert_eq!(
            got.get_links_map(),
            reference.get_links_map(),
            "V3 conversion must preserve links"
        );
    }
}
