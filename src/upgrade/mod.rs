use crate::IRIS_CODE_LENGTH;
use eyre::{bail, Result};
use std::{
    fs::File,
    io::{BufWriter, Write},
};

pub mod packets;
pub mod upgrade;

pub trait OldIrisShareSource {
    /// loads an 1-of-2 additive share of the iris code with id `share_id`
    fn load_code_share(&self, share_id: u64) -> std::io::Result<[u16; IRIS_CODE_LENGTH]>;
    /// loads the maks of the iris code with id `share_id`
    fn load_mask(&self, share_id: u64) -> std::io::Result<[bool; IRIS_CODE_LENGTH]>;
}

pub trait NewIrisShareSink {
    fn store_code_share(
        &self,
        share_id: u64,
        share: &[u16; IRIS_CODE_LENGTH],
    ) -> std::io::Result<()>;
    fn store_mask_share(
        &self,
        share_id: u64,
        share: &[u16; IRIS_CODE_LENGTH],
    ) -> std::io::Result<()>;
}

#[derive(Debug, Clone)]
pub struct IrisShareTestFileSink {
    path: std::path::PathBuf,
}

impl IrisShareTestFileSink {
    pub fn new(folder: std::path::PathBuf) -> Result<Self> {
        if !folder.is_dir() {
            bail!("{} is not a directory", folder.display());
        }
        Ok(Self { path: folder })
    }
}

impl NewIrisShareSink for IrisShareTestFileSink {
    fn store_code_share(
        &self,
        share_id: u64,
        share: &[u16; IRIS_CODE_LENGTH],
    ) -> std::io::Result<()> {
        let mut file = BufWriter::new(File::create(
            self.path.join(format!("code_share_{}", share_id)),
        )?);
        for s in share {
            write!(file, "{}\n", s)?;
        }
        file.flush()
    }

    fn store_mask_share(
        &self,
        share_id: u64,
        share: &[u16; IRIS_CODE_LENGTH],
    ) -> std::io::Result<()> {
        let mut file = BufWriter::new(File::create(
            self.path.join(format!("mask_share_{}", share_id)),
        )?);
        for s in share {
            write!(file, "{}\n", s)?;
        }
        file.flush()
    }
}
