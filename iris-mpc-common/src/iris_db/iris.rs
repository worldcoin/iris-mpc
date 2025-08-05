use crate::galois_engine::degree4::GaloisRingIrisCodeShare;
use base64::{prelude::BASE64_STANDARD, Engine};
use eyre::bail;
use eyre::Result;
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
    pub const CODE_COLS: usize = 200;
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

    pub fn from_bits(bits: &[bool]) -> Self {
        let mut code = IrisCodeArray::ZERO;
        for (i, bit) in bits.iter().enumerate() {
            code.set_bit(i, *bit);
        }
        code
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
    pub fn from_base64(s: &str) -> Result<Self> {
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
    pub fn to_base64(&self) -> Result<String> {
        Ok(BASE64_STANDARD.encode(
            self.0
                .iter()
                .flat_map(|&x| x.reverse_bits().to_be_bytes())
                .collect::<Vec<_>>(),
        ))
    }

    pub fn rotate_left(&mut self, by: usize) {
        let mut bits = self.bits().collect::<Vec<_>>();
        bits.chunks_exact_mut(Self::CODE_COLS * 4)
            .for_each(|chunk| chunk.rotate_left(by * 4));
        *self = IrisCodeArray::from_bits(&bits);
    }

    pub fn rotate_right(&mut self, by: usize) {
        let mut bits = self.bits().collect::<Vec<_>>();
        bits.chunks_exact_mut(Self::CODE_COLS * 4)
            .for_each(|chunk| chunk.rotate_right(by * 4));
        *self = IrisCodeArray::from_bits(&bits);
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
    pub const CODE_COLS: usize = 200;
    pub const ROTATIONS_PER_DIRECTION: usize = 15;

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

    /// Generate a random iris code but with pattern_size number of equal bits in each column.
    /// The purpose of this is to have iris codes that keep similarity across rotations.
    /// The distance between one of those iris codes and its nth rotation (in each direction)
    /// is (min(n, pattern_size) * 1/(2 * pattern_size)).
    pub fn random_rng_with_pattern<R: Rng>(rng: &mut R, pattern_size: usize) -> Self {
        assert!(pattern_size <= IrisCode::CODE_COLS);
        assert!(IrisCode::CODE_COLS % pattern_size == 0);
        const LEN: usize = IrisCode::CODE_COLS * 4;
        let mut code = IrisCode::random_rng(rng);
        for i in (0..Self::IRIS_CODE_SIZE).step_by(LEN) {
            for j in (0..LEN).step_by(pattern_size * 4) {
                for k in 0..pattern_size {
                    for l in 0..4 {
                        code.code
                            .set_bit(i + j + k * 4 + l, code.code.get_bit(i + j + l));
                    }
                }
            }
        }
        code
    }

    /// Return the fractional Hamming distance between two iris codes, represented
    /// as a single floating point value.
    pub fn get_distance(&self, other: &Self) -> f64 {
        let (code_distance, combined_mask_len) = self.get_distance_fraction(other);
        code_distance as f64 / combined_mask_len as f64
    }

    /// Return the minimum distance of an iris code against all rotations of another iris code.
    pub fn get_min_distance(&self, other: &Self) -> f64 {
        let mut min_distance = f64::INFINITY;
        for rotation in other.all_rotations() {
            let distance = rotation.get_distance(self);
            if distance < min_distance {
                min_distance = distance;
            }
        }
        min_distance
    }

    /// Return the minimum distance of an iris code against all rotations of another iris code.
    pub fn get_min_distance_against_many(&self, others: &[Self]) -> f64 {
        let mut min_distance = f64::INFINITY;
        for rotation in others {
            let distance = rotation.get_distance(self);
            if distance < min_distance {
                min_distance = distance;
            }
        }
        min_distance
    }

    /// Return the minimum distance of an iris code against all rotations of another iris code.
    pub fn get_distances_against_many(&self, others: &[Self]) -> Vec<f64> {
        others
            .iter()
            .map(|rotation| rotation.get_distance(self))
            .collect()
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

    pub fn is_close_with_rotations(&self, other: &Self) -> bool {
        self.get_min_distance(other) < MATCH_THRESHOLD_RATIO
    }

    pub fn get_similar_iris<R: Rng>(&self, rng: &mut R, approx_diff_factor: f64) -> IrisCode {
        let mut res = self.clone();
        let dist = Bernoulli::new(approx_diff_factor).unwrap();
        for i in 0..IrisCode::IRIS_CODE_SIZE {
            if dist.sample(rng) {
                res.code.flip_bit(i);
            }
            if dist.sample(rng) {
                // flip both the imaginary and real bits
                res.mask.flip_bit(i - (i % 2));
                res.mask.flip_bit(i - (i % 2) + 1);
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

    fn rotate_right(&mut self, by: usize) {
        self.code.rotate_right(by);
        self.mask.rotate_right(by);
    }

    fn rotate_left(&mut self, by: usize) {
        self.code.rotate_left(by);
        self.mask.rotate_left(by);
    }

    pub fn all_rotations(&self) -> Vec<IrisCode> {
        let mut code = self.clone();
        code.rotate_left(Self::ROTATIONS_PER_DIRECTION + 1);
        let mut rotations = Vec::new();
        for _ in 0..Self::ROTATIONS_PER_DIRECTION * 2 + 1 {
            code.rotate_right(1);
            rotations.push(code.clone());
        }
        rotations
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
    use eyre::Result;
    use eyre::{Context, ContextCompat};
    use float_eq::assert_float_eq;
    use std::cmp::min;
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
    pub fn parse_test_data(s: &str) -> Result<(&str, HashMap<i32, String>)> {
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

    #[test]
    fn test_rotations() {
        let mut rng = rand::thread_rng();
        let iris = IrisCode::random_rng(&mut rng);
        assert_eq!(
            iris,
            iris.all_rotations()[IrisCode::ROTATIONS_PER_DIRECTION]
        );
    }

    #[test]
    fn test_random_rng_with_pattern() {
        let mut rng = rand::thread_rng();
        let pattern_size = 4;
        let iris0 = IrisCode::random_rng_with_pattern(&mut rng, pattern_size);
        let rotations0 = iris0.all_rotations();

        for (i, rotation) in rotations0.iter().enumerate() {
            assert_float_eq!(
                rotation.get_distance(&iris0),
                min(
                    (IrisCode::ROTATIONS_PER_DIRECTION as f64 - i as f64).abs() as usize,
                    pattern_size
                ) as f64
                    / (2.0 * pattern_size as f64),
                abs <= 0.05
            );
        }
    }

    #[test]
    fn test_min_distance_random() {
        let mut rng = rand::thread_rng();
        let iris0 = IrisCode::random_rng(&mut rng);
        let mut iris1 = iris0.get_similar_iris(&mut rng, 0.025);
        iris1.rotate_left(1);
        assert_float_eq!(iris0.get_distance(&iris1), 0.5, abs <= 0.05);
        assert_float_eq!(iris0.get_min_distance(&iris1), 0.0, abs <= 0.05);
    }

    #[test]
    fn test_min_distance_example_data() {
        let lines = include_str!("../example-data/random_codes.txt")
            .lines()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();

        let t1_code = IrisCodeArray::from_base64(lines[0]).unwrap();
        let t1_mask = IrisCodeArray::from_base64(lines[1]).unwrap();
        let t2_code = IrisCodeArray::from_base64(lines[2]).unwrap();
        let t2_mask = IrisCodeArray::from_base64(lines[3]).unwrap();

        let dist_0 = lines[4].parse::<f64>().unwrap();
        let dist_15 = lines[5].parse::<f64>().unwrap();

        let t1_iris = IrisCode {
            code: t1_code,
            mask: t1_mask,
        };
        let t2_iris = IrisCode {
            code: t2_code,
            mask: t2_mask,
        };

        assert_float_eq!(t1_iris.get_distance(&t2_iris), dist_0, abs <= 1e-6);
        assert_float_eq!(t1_iris.get_min_distance(&t2_iris), dist_15, abs <= 1e-6);
    }
}
