use crate::serialization::{ReadPacked, WritePacked};
use eyre::Result;
use serde::{Deserialize, Serialize};
use std::{
    io::{Read, Write},
    mem::size_of,
    ops::{Deref, DerefMut},
};

/// A sorted list of edge IDs (without distances).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct SortedEdgeIds<V>(pub Vec<V>);

impl<V: ReadPacked> ReadPacked for SortedEdgeIds<V> {
    fn read_packed<R: Read>(reader: &mut R) -> Result<Self> {
        let mut len_bytes = [0u8; size_of::<usize>()];
        reader.read_exact(&mut len_bytes)?;
        let len = usize::from_le_bytes(len_bytes);
        let mut edges = Vec::with_capacity(len);
        for _ in 0..len {
            edges.push(V::read_packed(reader)?);
        }
        Ok(SortedEdgeIds(edges))
    }
}

impl<V: WritePacked> WritePacked for SortedEdgeIds<V> {
    fn write_packed<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&(self.0.len()).to_le_bytes())?;
        for edge in &self.0 {
            edge.write_packed(writer)?;
        }
        Ok(())
    }
}

impl<V: WritePacked> SortedEdgeIds<V> {
    pub fn to_packed(self) -> Result<Vec<u8>> {
        let mut ret = vec![];
        self.write_packed(&mut ret)?;
        Ok(ret)
    }
}

impl<V> SortedEdgeIds<V> {
    pub fn from_ascending_vec(edges: Vec<V>) -> Self {
        SortedEdgeIds(edges)
    }

    pub fn trim_to_k_nearest(&mut self, k: usize) {
        self.0.truncate(k);
    }
}

impl<V> Default for SortedEdgeIds<V> {
    fn default() -> Self {
        SortedEdgeIds(vec![])
    }
}

impl<V> Deref for SortedEdgeIds<V> {
    type Target = Vec<V>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<V> DerefMut for SortedEdgeIds<V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
