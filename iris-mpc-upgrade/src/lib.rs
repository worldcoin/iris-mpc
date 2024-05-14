use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Mutex;
use std::{convert::Infallible, str::FromStr, sync::atomic::AtomicU64};

use color_eyre::eyre::{bail, Result};
use sha2::digest::Digest;

pub mod packets;
pub mod prf;
pub mod shamir;
pub mod share;
pub mod upgrade;

pub const IRIS_CODE_LEN: usize = 12800;

#[derive(Debug, Copy, Clone)]
pub struct Seed([u8; 16]);

impl FromStr for Seed {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Seed(
            sha2::Sha256::digest(s.as_bytes())[0..16]
                .try_into()
                .unwrap(),
        ))
    }
}

impl From<[u8; 16]> for Seed {
    fn from(seed: [u8; 16]) -> Self {
        Seed(seed)
    }
}

#[derive(Debug)]
pub struct Counters {
    pub submitted_counter: AtomicU64,
    pub active_counter: AtomicU64,
    pub finished_counter: AtomicU64,
    pub finished_ids: Mutex<Vec<u64>>,
}

impl Counters {
    pub fn new() -> Self {
        Self {
            finished_counter: AtomicU64::new(0),
            submitted_counter: AtomicU64::new(0),
            active_counter: AtomicU64::new(0),
            finished_ids: Mutex::new(Vec::new()),
        }
    }
}

pub trait OldIrisShareSource {
    /// loads an 1-of-2 additive share of the iris code with id `share_id`
    fn load_code_share(&self, share_id: u64) -> std::io::Result<Vec<u16>>;
    /// loads the maks of the iris code with id `share_id`
    fn load_mask(&self, share_id: u64) -> std::io::Result<Vec<bool>>;
}

pub trait NewIrisShareSink {
    fn store_code_share(&self, share_id: u64, share: Vec<u16>) -> std::io::Result<()>;
    fn store_mask_share(&self, share_id: u64, share: Vec<u16>) -> std::io::Result<()>;
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
    fn store_code_share(&self, share_id: u64, share: Vec<u16>) -> std::io::Result<()> {
        let mut file = BufWriter::new(File::create(
            self.path.join(format!("code_share_{}", share_id)),
        )?);
        for s in share {
            write!(file, "{}\n", s)?;
        }
        file.flush()
    }

    fn store_mask_share(&self, share_id: u64, share: Vec<u16>) -> std::io::Result<()> {
        let mut file = BufWriter::new(File::create(
            self.path.join(format!("mask_share_{}", share_id)),
        )?);
        for s in share {
            write!(file, "{}\n", s)?;
        }
        file.flush()
    }
}

/// An enum representing the party ID
#[derive(std::cmp::Eq, std::cmp::PartialEq, Clone, Copy, Debug)]
#[repr(u8)]
pub enum PartyID {
    /// Party 0
    ID0 = 0,
    /// Party 1
    ID1 = 1,
    /// Party 2
    ID2 = 2,
}

impl PartyID {
    /// get next ID
    pub fn next_id(&self) -> Self {
        match *self {
            PartyID::ID0 => PartyID::ID1,
            PartyID::ID1 => PartyID::ID2,
            PartyID::ID2 => PartyID::ID0,
        }
    }

    /// get previous ID
    pub fn prev_id(&self) -> Self {
        match *self {
            PartyID::ID0 => PartyID::ID2,
            PartyID::ID1 => PartyID::ID0,
            PartyID::ID2 => PartyID::ID1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PartyIDError(String);

impl std::error::Error for PartyIDError {}

impl std::fmt::Display for PartyIDError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid party ID: {}", &self.0)
    }
}

impl TryFrom<usize> for PartyID {
    type Error = PartyIDError;

    fn try_from(other: usize) -> Result<Self, Self::Error> {
        match other {
            0 => Ok(PartyID::ID0),
            1 => Ok(PartyID::ID1),
            2 => Ok(PartyID::ID2),
            i => Err(PartyIDError(format!("Invalid party ID: {}", i))),
        }
    }
}

impl TryFrom<u8> for PartyID {
    type Error = PartyIDError;

    #[inline(always)]
    fn try_from(other: u8) -> Result<Self, Self::Error> {
        (other as usize).try_into()
    }
}

impl FromStr for PartyID {
    type Err = PartyIDError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<usize>()
            .map_err(|e| PartyIDError(e.to_string()))?
            .try_into()
    }
}

impl From<PartyID> for u8 {
    #[inline(always)]
    fn from(other: PartyID) -> Self {
        other as u8
    }
}

impl From<PartyID> for usize {
    #[inline(always)]
    fn from(other: PartyID) -> Self {
        other as usize
    }
}

impl std::fmt::Display for PartyID {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", *self as usize)
    }
}
