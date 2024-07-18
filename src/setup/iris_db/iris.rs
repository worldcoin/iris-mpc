use base64::{prelude::BASE64_STANDARD, Engine};
use eyre::bail;
use rand::{
    distributions::{Bernoulli, Distribution},
    Rng,
};

pub const MATCH_THRESHOLD_RATIO: f64 = 0.375;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IrisCodeArray(pub [u64; Self::IRIS_CODE_SIZE_U64]);
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
            code:    self,
            current: 0,
            index:   0,
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

#[derive(Clone, Debug)]
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
        for _ in 0..Self::IRIS_CODE_SIZE / 10 {
            let i = rng.gen_range(0..Self::IRIS_CODE_SIZE);
            code.mask.set_bit(i, false);
        }

        code
    }

    pub fn get_distance(&self, other: &Self) -> f64 {
        let combined_mask = self.mask & other.mask;
        let combined_mask_len = combined_mask.count_ones();

        let combined_code = (self.code ^ other.code) & combined_mask;
        let code_distance = combined_code.count_ones();
        code_distance as f64 / combined_mask_len as f64
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
}

pub struct Bits<'a> {
    code:    &'a IrisCodeArray,
    current: u64,
    index:   usize,
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
    use super::IrisCodeArray;
    use eyre::{Context, ContextCompat};
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
