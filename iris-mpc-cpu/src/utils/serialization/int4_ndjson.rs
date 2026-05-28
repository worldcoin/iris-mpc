use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::Path,
};

use eyre::{bail, Result};

use crate::{
    hawkers::plaintext_deep_id_store::Int4Vector,
    utils::serialization::types::int4_base64::{
        read_from_int4_ndjson, write_to_int4_ndjson, Base64Int4Vector,
    },
};

/// Stream `Int4Vector`s out of an NDJSON file. `limit` truncates the stream.
///
/// Each yielded item is a `Result` because per-line failures (malformed JSON,
/// invalid base64, wrong packed length, out-of-domain nibbles) are recoverable
/// — callers should surface them rather than panic mid-stream.
pub fn int4_vectors_from_ndjson_iter(
    path: &Path,
    limit: Option<usize>,
) -> Result<impl Iterator<Item = Result<Int4Vector>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let stream = read_from_int4_ndjson(reader);

    Ok(stream
        .map(|json_pt| Int4Vector::try_from(&json_pt?))
        .take(limit.unwrap_or(usize::MAX)))
}

/// Read `limit` Int4Vectors from the file into a `Vec`, or all if `limit` is None.
pub fn int4_vectors_from_ndjson(path: &Path, limit: Option<usize>) -> Result<Vec<Int4Vector>> {
    let vectors = int4_vectors_from_ndjson_iter(path, limit)?.collect::<Result<Vec<_>>>()?;

    if let Some(num) = limit {
        if vectors.len() != num {
            let path_str = path.as_os_str().to_str().unwrap_or("[invalid UTF-8 path]");
            bail!(
                "File {} contains too few entries; number read: {}",
                path_str,
                vectors.len()
            );
        }
    }

    Ok(vectors)
}

/// Write a slice of `Int4Vector`s to an NDJSON file.
pub fn write_int4_vectors_ndjson(path: &Path, vectors: &[Int4Vector]) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let encoded = vectors.iter().map(Base64Int4Vector::from);
    write_to_int4_ndjson(&mut writer, encoded)?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aes_prng::AesRng;
    use rand::SeedableRng;
    use tempfile::tempdir;

    #[test]
    fn file_roundtrip() {
        let mut rng = AesRng::seed_from_u64(0xBEEF);
        let originals: Vec<Int4Vector> = (0..8).map(|_| Int4Vector::random(&mut rng)).collect();

        let dir = tempdir().unwrap();
        let path = dir.path().join("v.ndjson");

        write_int4_vectors_ndjson(&path, &originals).unwrap();
        let read_back = int4_vectors_from_ndjson(&path, Some(8)).unwrap();

        assert_eq!(read_back.len(), originals.len());
        for (a, b) in originals.iter().zip(read_back.iter()) {
            assert_eq!(a.packed, b.packed);
        }
    }

    #[test]
    fn limit_truncates() {
        let mut rng = AesRng::seed_from_u64(0xBEEF);
        let originals: Vec<Int4Vector> = (0..8).map(|_| Int4Vector::random(&mut rng)).collect();

        let dir = tempdir().unwrap();
        let path = dir.path().join("v.ndjson");
        write_int4_vectors_ndjson(&path, &originals).unwrap();

        let read_back = int4_vectors_from_ndjson(&path, Some(3)).unwrap();
        assert_eq!(read_back.len(), 3);
        for (a, b) in originals.iter().take(3).zip(read_back.iter()) {
            assert_eq!(a.packed, b.packed);
        }
    }
}
