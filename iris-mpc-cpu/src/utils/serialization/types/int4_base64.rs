//! Base64-NDJSON serialization for `Int4Vector` used by deep-ID binaries.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use eyre::{eyre, Result};
use serde::{Deserialize, Serialize};

use crate::hawkers::plaintext_deep_id_store::{Int4Vector, INT4_PACKED_BYTES};

/// On-disk representation: the 256 packed bytes, base64-encoded.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Base64Int4Vector {
    pub packed_b64: String,
}

impl From<&Int4Vector> for Base64Int4Vector {
    fn from(v: &Int4Vector) -> Self {
        Self {
            packed_b64: STANDARD.encode(v.packed),
        }
    }
}

impl TryFrom<&Base64Int4Vector> for Int4Vector {
    type Error = eyre::Report;

    fn try_from(v: &Base64Int4Vector) -> Result<Self> {
        let bytes = STANDARD.decode(&v.packed_b64)?;
        if bytes.len() != INT4_PACKED_BYTES {
            return Err(eyre!(
                "expected {} packed bytes, got {}",
                INT4_PACKED_BYTES,
                bytes.len()
            ));
        }
        let mut packed = [0u8; INT4_PACKED_BYTES];
        packed.copy_from_slice(&bytes);
        Ok(Int4Vector { packed })
    }
}

pub fn read_from_int4_ndjson<R: std::io::Read>(
    reader: R,
) -> impl Iterator<Item = Result<Base64Int4Vector>> {
    serde_json::Deserializer::from_reader(reader)
        .into_iter()
        .map(|res| res.map_err(Into::into))
}

pub fn write_to_int4_ndjson<W, I>(writer: &mut W, data: I) -> Result<()>
where
    W: std::io::Write,
    I: IntoIterator<Item = Base64Int4Vector>,
{
    for json_pt in data {
        serde_json::to_writer(&mut *writer, &json_pt)?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aes_prng::AesRng;
    use rand::SeedableRng;

    #[test]
    fn roundtrip_through_ndjson() {
        let mut rng = AesRng::seed_from_u64(0xDEADBEEF);
        let originals: Vec<Int4Vector> = (0..16).map(|_| Int4Vector::random(&mut rng)).collect();

        let mut buf: Vec<u8> = Vec::new();
        let encoded = originals.iter().map(Base64Int4Vector::from);
        write_to_int4_ndjson(&mut buf, encoded).unwrap();

        let decoded: Vec<Int4Vector> = read_from_int4_ndjson(buf.as_slice())
            .map(|r| Int4Vector::try_from(&r.unwrap()).unwrap())
            .collect();

        assert_eq!(decoded.len(), originals.len());
        for (orig, got) in originals.iter().zip(decoded.iter()) {
            assert_eq!(orig.packed, got.packed);
        }
    }
}
