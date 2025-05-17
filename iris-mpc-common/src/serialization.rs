//! These traits are used for custom binary serialization. They were initially
//! developed to reduce the size of the HNSW graph database, to reduce load times.

use eyre::Result;
use std::io::{Read, Write};

pub trait WritePacked {
    fn write_packed<W: Write>(&self, writer: &mut W) -> Result<()>;
}

pub trait ReadPacked: Sized {
    fn read_packed<R: Read>(reader: &mut R) -> Result<Self>;
}
