use crate::galois_engine::degree4::{GaloisRingIrisCodeShare, IrisRotation};
use crate::IRIS_CODE_LENGTH;
use base64::{prelude::BASE64_STANDARD, Engine};
use bytemuck::cast_slice;
use eyre::bail;
use eyre::Result;
use itertools::izip;
use rand::seq::SliceRandom;
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

/// Iris code representation using base64 encoding compatible with Open IRIS
#[derive(Serialize, Deserialize)]
pub struct IrisCodeBase64 {
    pub iris_codes: String,
    pub mask_codes: String,
}

/// Convertor: IrisCode -> IrisCodeBase64.
impl From<&IrisCode> for IrisCodeBase64 {
    fn from(value: &IrisCode) -> Self {
        Self {
            iris_codes: value.code.to_base64().unwrap(),
            mask_codes: value.mask.to_base64().unwrap(),
        }
    }
}

/// Convertor: IrisCodeBase64 -> IrisCode.
impl From<&IrisCodeBase64> for IrisCode {
    fn from(value: &IrisCodeBase64) -> Self {
        Self {
            code: IrisCodeArray::from_base64(&value.iris_codes).unwrap(),
            mask: IrisCodeArray::from_base64(&value.mask_codes).unwrap(),
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

    /// Return the rotated fractional Hamming distance between two iris codes
    /// represented with bit encodings for which `u64` limbs restrict to
    /// geometric columns, as produced by [IrisCode::transform_iris_code_array].
    ///
    /// The returned distance is represented by a `u16` tuple of the fraction
    /// numerator and denominator.
    ///
    /// Iterates over left array as though it has been rotated according to the
    /// iris rotation. This involves splitting the array at the index for which
    /// cyclic rotation wraps, and doing hamming distance computations over two
    /// separate slices.
    fn get_distance_fraction_with_rotation(
        left_code: &[u64; Self::CODE_COLS],
        left_mask: &[u64; Self::CODE_COLS],
        right_code: &[u64; Self::CODE_COLS],
        right_mask: &[u64; Self::CODE_COLS],
        rotation: &IrisRotation,
    ) -> (u16, u16) {
        // what this code mimics:
        // let combined_mask = self.mask & other.mask;
        // let combined_mask_len = combined_mask.count_ones();
        //
        //  let combined_code = (self.code ^ other.code) & combined_mask;
        // let code_distance = combined_code.count_ones();

        let chunk_size = Self::CODE_COLS;
        let skip = match rotation {
            IrisRotation::Center => 0,
            IrisRotation::Left(rot) => *rot,
            IrisRotation::Right(rot) => chunk_size - *rot,
        };

        let mut code_distance = 0u16;
        let mut combined_mask_len = 0u16;

        // Split the rotation into two contiguous loops,
        // allowing the compiler to vectorize
        let (left_code1, left_code2) = left_code.split_at(skip);
        let (left_mask1, left_mask2) = left_mask.split_at(skip);
        let (right_code1, right_code2) = right_code.split_at(chunk_size - skip);
        let (right_mask1, right_mask2) = right_mask.split_at(chunk_size - skip);

        for (lc, lm, rc, rm) in izip!(left_code1, left_mask1, right_code2, right_mask2) {
            let combined_mask = lm & rm;
            combined_mask_len += combined_mask.count_ones() as u16;
            let combined_code = (lc ^ rc) & combined_mask;
            code_distance += combined_code.count_ones() as u16;
        }

        for (lc, lm, rc, rm) in izip!(left_code2, left_mask2, right_code1, right_mask1) {
            let combined_mask = lm & rm;
            combined_mask_len += combined_mask.count_ones() as u16;
            let combined_code = (lc ^ rc) & combined_mask;
            code_distance += combined_code.count_ones() as u16;
        }

        (code_distance, combined_mask_len)
    }

    /// Transform an iris code array to a bit embedding for which iris rotations
    /// respect `u64` limbs, i.e each limb represents the bits in one geometric
    /// column of the iris code.
    ///
    /// The current "standard" embedding encodes bits in a way that matches the
    /// representation as MPC shares, which is optimized for transformations
    /// needed over the Galois ring secret sharing scheme.  Bits in this scheme
    /// appear in blocks of `800 = 4 * IrisCode::CODE_COLS`, which represent
    /// blocks of 4 spaced out geometric rows for each geometric column:
    ///
    /// ```text
    /// (r, c) = (0, 0), (4, 0), (8, 0), (12, 0), (0, 1), (4, 1), (8, 1), (12, 1), ...
    ///       (0, 199), (4, 199), (8, 199), (12, 199), (1, 0), (5, 0), (9, 0), (13, 0), ...
    /// ```
    ///
    /// See [GaloisRingIrisCodeShare] for the explicit mapping of coordinates.
    ///
    /// To convert this to a representation where each `u64` limb holds only
    /// bits in a single geometric column which can then be rotated directly to
    /// represent iris code rotations, the 4-bit blocks of the first 800 bits of
    /// the array are mapped into the first 4 bits each of 200 new `u64` limbs,
    /// the 4-bit blocks of the next 800 bits of the array are mapped each to
    /// the second 4 bits of the new limbs, and so on through the full array.
    ///
    /// The resulting embedding into `u64` limbs encodes all bits from a single
    /// geometric column for both wavelengths and both the real and imaginary
    /// parts in the following geometric row order:
    ///
    /// ```text
    /// r = 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15
    /// ```
    ///
    /// This ordering is repeated for the real bits for each of the two wavelet
    /// frequencies in succession, and then the same again for the imaginary
    /// bits.
    fn transform_iris_code_array(input: &IrisCodeArray) -> [u64; Self::CODE_COLS] {
        // r is zeroed
        let mut r = [0; Self::CODE_COLS];

        // first 200 4 bit blocks -> 800 bits -> 25 u32
        let input_u32 = cast_slice::<u64, u32>(&input.0);
        for (chunk_idx, chunk) in input_u32.chunks_exact(25).enumerate() {
            // do the first 4 bits first, next the second 4 bits, etc.
            let rshift = chunk_idx << 2; // multiply chunk_idx by 4
            let mut output_idx = 0;
            for &half_word in chunk.iter() {
                let half_word = half_word as u64;
                r[output_idx] |= (half_word & 0xF) << rshift;
                r[output_idx + 1] |= ((half_word >> 4) & 0xF) << rshift;
                r[output_idx + 2] |= ((half_word >> 8) & 0xF) << rshift;
                r[output_idx + 3] |= ((half_word >> 12) & 0xF) << rshift;
                r[output_idx + 4] |= ((half_word >> 16) & 0xF) << rshift;
                r[output_idx + 5] |= ((half_word >> 20) & 0xF) << rshift;
                r[output_idx + 6] |= ((half_word >> 24) & 0xF) << rshift;
                r[output_idx + 7] |= ((half_word >> 28) & 0xF) << rshift;

                output_idx += 8;
            }
        }

        r
    }

    /// Return the minimum distance of an iris code against all rotations of another iris code
    /// using the IrisRotation enum. This avoids generating all rotation copies.
    /// snote that the rotations are applied to Other.
    pub fn get_min_distance_fraction_rotation_aware(&self, other: &Self) -> (u16, u16) {
        let mut min_distance = (u16::MAX, u16::MAX);

        let self_code = Self::transform_iris_code_array(&self.code);
        let self_mask = Self::transform_iris_code_array(&self.mask);
        let other_code = Self::transform_iris_code_array(&other.code);
        let other_mask = Self::transform_iris_code_array(&other.mask);

        // go through all rotations of other
        for rotation in IrisRotation::all() {
            let distance = Self::get_distance_fraction_with_rotation(
                &other_code,
                &other_mask,
                &self_code,
                &self_mask,
                &rotation,
            );
            if distance.0 as u32 * (min_distance.1 as u32)
                < distance.1 as u32 * min_distance.0 as u32
            {
                min_distance = distance;
            }
        }

        min_distance
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

    /// Return the minimum distance of an iris code against all rotations of another iris code.
    pub fn get_min_distance_fraction(&self, other: &Self) -> (u16, u16) {
        let mut min_distance = (u16::MAX, u16::MAX);
        for rotation in other.all_rotations() {
            let distance = rotation.get_distance_fraction(self);
            if distance.0 as u32 * (min_distance.1 as u32)
                < distance.1 as u32 * min_distance.0 as u32
            {
                min_distance = distance;
            }
        }
        min_distance
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

    pub fn get_graded_similar_iris<R: Rng>(
        &self,
        rng: &mut R,
        target_distance: (u16, u16),
    ) -> IrisCode {
        let mut visible_bits = (0..IRIS_CODE_LENGTH)
            .filter_map(|i| self.mask.get_bit(i).then_some(i))
            .collect::<Vec<_>>();

        visible_bits.shuffle(rng);

        // Compute the ideal number of differing bits in the result
        let (num, denom) = target_distance;
        let neq_cnt =
            ((num as usize) * visible_bits.len() + (denom / 2) as usize) / (denom as usize);

        let mut result = self.clone();
        for i in visible_bits.iter().take(neq_cnt) {
            result.code.flip_bit(*i);
        }

        result
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
    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
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

    #[test]
    fn test_get_distance_fraction_with_rotation() {
        use crate::galois_engine::degree4::IrisRotation;
        let mut rng = rand::thread_rng();

        // Test that get_distance_fraction_with_rotation
        // produces the same result as get_distance_fraction
        for _ in 0..10 {
            let iris1 = IrisCode::random_rng(&mut rng);
            let iris2 = IrisCode::random_rng(&mut rng);

            let iris1_code = IrisCode::transform_iris_code_array(&iris1.code);
            let iris1_mask = IrisCode::transform_iris_code_array(&iris1.mask);
            let iris2_code = IrisCode::transform_iris_code_array(&iris2.code);
            let iris2_mask = IrisCode::transform_iris_code_array(&iris2.mask);

            let standard_result = iris1.get_distance_fraction(&iris2);
            let rotation_result = IrisCode::get_distance_fraction_with_rotation(
                &iris2_code,
                &iris2_mask,
                &iris1_code,
                &iris1_mask,
                &IrisRotation::Center,
            );
            assert_eq!(standard_result, rotation_result);

            for left_rot in 1..=15 {
                let rotation_result = IrisCode::get_distance_fraction_with_rotation(
                    &iris2_code,
                    &iris2_mask,
                    &iris1_code,
                    &iris1_mask,
                    &IrisRotation::Left(left_rot),
                );
                let mut rotated_iris2 = iris2.clone();
                rotated_iris2.rotate_left(left_rot);
                let standard_result = iris1.get_distance_fraction(&rotated_iris2);
                assert_eq!(standard_result, rotation_result);
            }

            for right_rot in 1..=15 {
                let rotation_result = IrisCode::get_distance_fraction_with_rotation(
                    &iris2_code,
                    &iris2_mask,
                    &iris1_code,
                    &iris1_mask,
                    &IrisRotation::Right(right_rot),
                );
                let mut rotated_iris2 = iris2.clone();
                rotated_iris2.rotate_right(right_rot);
                let standard_result = iris1.get_distance_fraction(&rotated_iris2);
                assert_eq!(standard_result, rotation_result);
            }
        }
    }

    #[test]
    fn test_get_graded_similar_iris() {
        let mut base_rng = SmallRng::seed_from_u64(42);
        let mut iris_a = IrisCode::random_rng(&mut base_rng);

        // Start with full mask and flip 1200 bits
        iris_a.mask = IrisCodeArray::ONES;
        let mut indices = (0..(IrisCode::IRIS_CODE_SIZE / 2))
            .take(600)
            .collect::<Vec<_>>();
        indices.shuffle(&mut base_rng);
        for &i in indices.iter() {
            iris_a.mask.flip_bit(2 * i);
            iris_a.mask.flip_bit(2 * i + 1);
        }

        let dist_b_target = (1, 8);
        let dist_c_target = (1, 4);

        let mut rng_for_b = base_rng.clone();
        let mut rng_for_c = base_rng.clone();

        let iris_b = iris_a.get_graded_similar_iris(&mut rng_for_b, dist_b_target);
        let iris_c = iris_a.get_graded_similar_iris(&mut rng_for_c, dist_c_target);

        let dist_a_b = iris_a.get_distance(&iris_b);
        let expected_dist_b = dist_b_target.0 as f64 / dist_b_target.1 as f64;

        assert_float_eq!(dist_a_b, expected_dist_b, abs <= 1e-5);

        let dist_a_c = iris_a.get_distance(&iris_c);
        let expected_dist_c = dist_c_target.0 as f64 / dist_c_target.1 as f64;
        assert_float_eq!(dist_a_c, expected_dist_c, abs <= 1e-5);

        let dist_b_c = iris_b.get_distance(&iris_c);
        let expected_dist_b_c = expected_dist_c - expected_dist_b;
        assert_float_eq!(dist_b_c, expected_dist_b_c, abs <= 1e-5);

        assert_eq!(iris_a.mask, iris_b.mask);
        assert_eq!(iris_a.mask, iris_c.mask);
    }
}
