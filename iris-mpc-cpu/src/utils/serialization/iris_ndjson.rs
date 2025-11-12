use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::Path,
    sync::Arc,
};

use clap::ValueEnum;
use eyre::Result;
use iris_mpc_common::{
    iris_db::iris::{IrisCode, IrisCodeArray},
    IrisVectorId,
};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::{
    hawkers::{aby3::aby3_store::DistanceFn, plaintext_store::PlaintextStore},
    utils::serialization::types::iris_base64::{
        read_from_iris_ndjson, write_to_iris_ndjson, Base64IrisCode,
    },
};

// Note: all utilities relating to conversion between serialization data formats
// and application data formats go here rather than in type definitions.

impl From<&IrisCode> for Base64IrisCode {
    fn from(value: &IrisCode) -> Self {
        Self {
            iris_codes: value.code.to_base64().unwrap(),
            mask_codes: value.mask.to_base64().unwrap(),
        }
    }
}

impl From<&Base64IrisCode> for IrisCode {
    fn from(value: &Base64IrisCode) -> Self {
        Self {
            code: IrisCodeArray::from_base64(&value.iris_codes).unwrap(),
            mask: IrisCodeArray::from_base64(&value.mask_codes).unwrap(),
        }
    }
}

#[derive(Clone, Debug, ValueEnum, Copy, Serialize, Deserialize, PartialEq)]
pub enum IrisSelection {
    All,
    Even,
    Odd,
}

pub fn irises_from_ndjson_iter(
    path: &Path,
    limit: Option<usize>,
    selection: IrisSelection,
) -> Result<impl Iterator<Item = IrisCode>> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let stream = read_from_iris_ndjson(reader);

    let (skip, step) = match selection {
        IrisSelection::All => (0, 1),
        IrisSelection::Even => (0, 2),
        IrisSelection::Odd => (1, 2),
    };

    Ok(stream
        .skip(skip)
        .step_by(step)
        .map(|json_pt| (&json_pt.unwrap()).into())
        .take(limit.unwrap_or(usize::MAX)))
}

/// Read `limit` iris codes from the specified file into an in-memory `Vec`, or
/// all iris codes of `limit` is `None`.  Read all, even, or odd iris codes from the
/// file depending on the value of `selection`.
pub fn irises_from_ndjson(
    path: &Path,
    limit: Option<usize>,
    selection: IrisSelection,
) -> Result<Vec<IrisCode>> {
    let stream_iterator = irises_from_ndjson_iter(path, limit, selection)?;
    let codes = stream_iterator.collect_vec();

    // Check that enough codes were present in file
    if let Some(num) = limit {
        if codes.len() != num {
            let path_str = path.as_os_str().to_str().unwrap_or("[invalid UTF-8 path]");
            eyre::bail!(format!(
                "File {} contains too few entries; number read: {}",
                path_str,
                codes.len()
            ));
        }
    }

    Ok(codes)
}

impl PlaintextStore {
    pub fn to_ndjson_file(&self, path: &Path) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let serial_ids = self.storage.get_sorted_serial_ids();
        let irises = serial_ids.into_iter().map(|serial_id| {
            let iris_code_arc = self
                .storage
                .get_vector_by_serial_id(serial_id)
                .expect("key not found in store");
            (&**iris_code_arc).into()
        });

        write_to_iris_ndjson(&mut writer, irises)?;
        writer.flush()?;
        Ok(())
    }
}

// TODO: refactor into function which write from `Vec<IrisCode>`, plus wrapper which takes a PlaintextStore
