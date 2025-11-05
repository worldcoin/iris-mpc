//! Implements a data serialization format representing an ordered pair of
//! `GraphMem` structs, with left and right sides.

/// Data type for the long-term serialization file format encoding a left/right
/// pair of `GraphV1` HNSW graphs.
pub type GraphV2Pair = [super::graph_v2::GraphV2; 2];

/* ------------------------------- I/O ------------------------------ */

pub fn read_graph_v1_pair<R: std::io::Read>(reader: &mut R) -> eyre::Result<GraphV2Pair> {
    let data = bincode::deserialize_from(reader)?;
    Ok(data)
}

pub fn write_graph_v1_pair<W: std::io::Write>(
    writer: &mut W,
    data: &GraphV2Pair,
) -> eyre::Result<()> {
    bincode::serialize_into(writer, data)?;
    Ok(())
}
