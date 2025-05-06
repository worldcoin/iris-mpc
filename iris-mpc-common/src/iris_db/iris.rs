use crate::galois_engine::degree4::GaloisRingIrisCodeShare;
use base64::{prelude::BASE64_STANDARD, Engine};
use eyre::bail;
use rand::{
    distributions::{Bernoulli, Distribution},
    Rng,
};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

pub const MATCH_THRESHOLD_RATIO: f64 = 0.375;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct IrisCodeArray(#[serde(with = "BigArray")] pub [u64; Self::IRIS_CODE_SIZE_U64]);
impl Default for IrisCodeArray {
    fn default() -> Self {
        Self::ZERO
    }
}

impl IrisCodeArray {
    pub const IRIS_CODE_SIZE: usize = 12800;
    pub const IRIS_CODE_SIZE_BYTES: usize = (Self::IRIS_CODE_SIZE + 7) / 8;
    pub const IRIS_CODE_SIZE_U64: usize = (Self::IRIS_CODE_SIZE + 63) / 64;
    pub const ZERO: Self = IrisCodeArray([0; Self::IRIS_CODE_SIZE_U64]);
    pub const ONES: Self = IrisCodeArray([u64::MAX; Self::IRIS_CODE_SIZE_U64]);
    #[inline]
    pub fn set_bit(&mut self, i: usize, val: bool) {
        let word = i / 64;
        let bit = i % 64;
        if val {
            self.0[word] |= 1u64 << bit;
        } else {
            self.0[word] &= !(1u64 << bit);
        }
    }
    pub fn bits(&self) -> Bits<'_> {
        Bits {
            code: self,
            current: 0,
            index: 0,
        }
    }
    #[inline]
    pub fn get_bit(&self, i: usize) -> bool {
        let word = i / 64;
        let bit = i % 64;
        (self.0[word] >> bit) & 1 == 1
    }
    #[inline]
    pub fn flip_bit(&mut self, i: usize) {
        let word = i / 64;
        let bit = i % 64;
        self.0[word] ^= 1u64 << bit;
    }

    #[inline]
    pub fn random_rng<R: Rng>(rng: &mut R) -> Self {
        let mut code = IrisCodeArray::ZERO;
        rng.fill(code.as_raw_mut_slice());
        code
    }

    pub fn count_ones(&self) -> usize {
        self.0.iter().map(|c| c.count_ones() as usize).sum()
    }

    pub fn as_raw_slice(&self) -> &[u8] {
        bytemuck::cast_slice(&self.0)
    }
    pub fn as_raw_mut_slice(&mut self) -> &mut [u8] {
        bytemuck::cast_slice_mut(&mut self.0)
    }

    /// Decode from base64 string compatible with Open IRIS
    pub fn from_base64(s: &str) -> eyre::Result<Self> {
        let decoded_bytes = BASE64_STANDARD.decode(s)?;
        if decoded_bytes.len() % 8 != 0 {
            bail!("Invalid length for u64 array");
        }

        Ok(Self(
            decoded_bytes
                .chunks_exact(8)
                .map(|chunk| {
                    let mut arr = [0u8; 8];
                    arr.copy_from_slice(chunk);
                    u64::from_be_bytes(arr).reverse_bits()
                })
                .collect::<Vec<u64>>()
                .try_into()
                .map_err(|_| eyre::eyre!("Expected exactly 200 elements"))?,
        ))
    }

    /// Encode to base64 string compatible with Open IRIS
    pub fn to_base64(&self) -> eyre::Result<String> {
        Ok(BASE64_STANDARD.encode(
            self.0
                .iter()
                .flat_map(|&x| x.reverse_bits().to_be_bytes())
                .collect::<Vec<_>>(),
        ))
    }
}

impl std::ops::BitAndAssign for IrisCodeArray {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        for i in 0..Self::IRIS_CODE_SIZE_U64 {
            self.0[i] &= rhs.0[i];
        }
    }
}
impl std::ops::BitAnd for IrisCodeArray {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        let mut res = IrisCodeArray::ZERO;
        for i in 0..Self::IRIS_CODE_SIZE_U64 {
            res.0[i] = self.0[i] & rhs.0[i];
        }
        res
    }
}
impl std::ops::BitXorAssign for IrisCodeArray {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        for i in 0..Self::IRIS_CODE_SIZE_U64 {
            self.0[i] ^= rhs.0[i];
        }
    }
}
impl std::ops::BitXor for IrisCodeArray {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        let mut res = IrisCodeArray::ZERO;
        for i in 0..Self::IRIS_CODE_SIZE_U64 {
            res.0[i] = self.0[i] ^ rhs.0[i];
        }
        res
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct IrisCode {
    pub code: IrisCodeArray,
    pub mask: IrisCodeArray,
}
impl Default for IrisCode {
    fn default() -> Self {
        Self {
            code: IrisCodeArray::ZERO,
            mask: IrisCodeArray::ONES,
        }
    }
}

impl IrisCode {
    pub const IRIS_CODE_SIZE: usize = IrisCodeArray::IRIS_CODE_SIZE;

    pub fn random_rng<R: Rng>(rng: &mut R) -> Self {
        let mut code = IrisCode {
            code: IrisCodeArray::random_rng(rng),
            mask: IrisCodeArray::ONES,
        };

        // remove about 10% of the mask bits
        // masks are duplicated in the last dimension, so we always need to set the bits
        // pairwise <https://github.com/worldcoin/iris/blob/e43e32748fd6800aa1ee11b0e79261d5ed62d776/src/iris/nodes/encoder/iris_encoder.py#L46>
        for _ in 0..Self::IRIS_CODE_SIZE / 10 / 2 {
            let i = rng.gen_range(0..Self::IRIS_CODE_SIZE / 2);
            code.mask.set_bit(2 * i, false);
            code.mask.set_bit(2 * i + 1, false);
        }

        code
    }

    /// Return the fractional Hamming distance between two iris codes, represented
    /// as a single floating point value.
    pub fn get_distance(&self, other: &Self) -> f64 {
        let (code_distance, combined_mask_len) = self.get_distance_fraction(other);
        code_distance as f64 / combined_mask_len as f64
    }

    /// Return the fractional Hamming distance between two iris codes, represented
    /// as `u16` numerator and denominator.
    pub fn get_distance_fraction(&self, other: &Self) -> (u16, u16) {
        let combined_mask = self.mask & other.mask;
        let combined_mask_len = combined_mask.count_ones();

        let combined_code = (self.code ^ other.code) & combined_mask;
        let code_distance = combined_code.count_ones();

        (code_distance as u16, combined_mask_len as u16)
    }

    /// Return the fractional Hamming distance between two iris codes, represented
    /// as the `i16` dot product of associated masked-bit vectors and the `u16` size
    /// of the common unmasked region.
    pub fn get_dot_distance_fraction(&self, other: &Self) -> (i16, u16) {
        let (code_distance, combined_mask_len) = self.get_distance_fraction(other);

        // `code_distance` gives the number of common unmasked bits which are
        // different between two iris codes, and `combined_mask_len` gives the
        // total number of common unmasked bits. The dot product of masked-bit
        // vectors adds 1 for each unmasked bit which is equal, and subtracts 1
        // for each unmasked bit which is unequal; so this can be computed by
        // starting with 1 for every unmasked bit, and subtracting 2 for every
        // unequal unmasked bit, as follows.
        let dot_product = combined_mask_len.wrapping_sub(2 * code_distance) as i16;

        (dot_product, combined_mask_len)
    }

    pub fn is_close(&self, other: &Self) -> bool {
        self.get_distance(other) < MATCH_THRESHOLD_RATIO
    }

    pub fn get_similar_iris<R: Rng>(&self, rng: &mut R) -> IrisCode {
        let mut res = self.clone();
        // flip a few bits in mask and code (like 5%)
        let dist = Bernoulli::new(0.05).unwrap();
        for i in 0..IrisCode::IRIS_CODE_SIZE {
            if dist.sample(rng) {
                res.code.flip_bit(i);
            }
            if dist.sample(rng) {
                res.mask.flip_bit(i);
            }
        }

        res
    }

    pub fn mirrored(&self) -> IrisCode {
        let mut mirrored = IrisCode::default();
        for i in 0..IrisCode::IRIS_CODE_SIZE {
            let new_i = GaloisRingIrisCodeShare::remap_old_to_new_index(i);
            let mirrored_new_i = GaloisRingIrisCodeShare::remap_new_to_mirrored_index(new_i);
            let mirrored_i = GaloisRingIrisCodeShare::remap_new_to_old_index(mirrored_new_i);
            mirrored.mask.set_bit(mirrored_i, self.mask.get_bit(i));
            let b = i % 2;
            let code_bit = self.code.get_bit(i);
            mirrored
                .code
                .set_bit(mirrored_i, if b == 0 { code_bit } else { !code_bit });
        }
        mirrored
    }
}

pub struct Bits<'a> {
    code: &'a IrisCodeArray,
    current: u64,
    index: usize,
}

impl Iterator for Bits<'_> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= IrisCodeArray::IRIS_CODE_SIZE {
            None
        } else {
            if self.index % 64 == 0 {
                self.current = self.code.0[self.index / 64];
            }
            let res = self.current & 1 == 1;
            self.current >>= 1;
            self.index += 1;
            Some(res)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            IrisCodeArray::IRIS_CODE_SIZE - self.index,
            Some(IrisCodeArray::IRIS_CODE_SIZE - self.index),
        )
    }
}

impl ExactSizeIterator for Bits<'_> {}

#[cfg(test)]
mod tests {
    use super::{IrisCode, IrisCodeArray};
    use eyre::{Context, ContextCompat};
    use float_eq::assert_float_eq;
    use std::collections::HashMap;

    #[test]
    fn bit_iter_eq_get_bit() {
        let mut rng = rand::thread_rng();
        let iris = super::IrisCode::random_rng(&mut rng);
        for (i, bit) in iris.code.bits().enumerate() {
            assert_eq!(iris.code.get_bit(i), bit);
        }
    }

    #[test]
    fn decode_from_string() {
        let (code_str, rotations) =
            parse_test_data(include_str!("../example-data/all_rotations.txt")).unwrap();
        let code = IrisCodeArray::from_base64(code_str).unwrap();
        let mut decoded_str = String::new();
        for i in 0..IrisCodeArray::IRIS_CODE_SIZE {
            decoded_str += &format!("{}", code.get_bit(i) as u8);
        }
        assert_eq!(
            decoded_str,
            *rotations.get(&0).unwrap(),
            "Decoded bit string does not match expected"
        );
        assert_eq!(code_str, code.to_base64().unwrap());
    }
    #[test]
    fn test_mirrored_iris_code() {
        // Use the same test data as in match_mirrored_codes
        let lines = include_str!("../example-data/flipped_codes.txt")
            .lines()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();

        // Parse the original and flipped iris codes
        let code = IrisCodeArray::from_base64(lines[0]).unwrap();
        let mask = IrisCodeArray::from_base64(lines[1]).unwrap();

        let flipped_code = IrisCodeArray::from_base64(lines[2]).unwrap();
        let flipped_mask = IrisCodeArray::from_base64(lines[3]).unwrap();

        // Create IrisCode objects
        let original_iris = IrisCode { code, mask };
        let flipped_iris = IrisCode {
            code: flipped_code,
            mask: flipped_mask,
        };

        // Check that the mirrored flipped iris matches the original
        let mirrored_iris = flipped_iris.mirrored();
        let distance = original_iris.get_distance(&mirrored_iris);
        assert_float_eq!(distance, 0.0, abs <= 1e-6);
    }
    pub fn parse_test_data(s: &str) -> eyre::Result<(&str, HashMap<i32, String>)> {
        let lines = s.lines();
        let mut lines = lines.map(|s| s.trim()).filter(|s| !s.is_empty());
        let code: &str = lines.next().context("Missing code")?;
        let mut rotations = HashMap::new();

        for line in lines {
            let mut parts = line.splitn(2, ':');
            let rotation = parts
                .next()
                .context("Missing rotation number")?
                .trim()
                .parse()
                .context("Invalid rotation")?;
            let bit_str = parts.next().context("Missing bit string")?;
            rotations.insert(rotation, bit_str.trim().replace(' ', "").to_string());
        }

        Ok((code, rotations))
    }
}
