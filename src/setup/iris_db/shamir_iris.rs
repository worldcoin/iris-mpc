use super::iris::{IrisCode, IrisCodeArray};
use crate::setup::shamir::{Shamir, P32};
use rand::Rng;

#[derive(Debug)]
pub struct ShamirIris {
    pub code: [u16; IrisCodeArray::IRIS_CODE_SIZE],
    pub mask: [u16; IrisCodeArray::IRIS_CODE_SIZE],
}

impl Clone for ShamirIris {
    fn clone(&self) -> Self {
        Self {
            code: self.code,
            mask: self.mask,
        }
    }
}

impl Default for ShamirIris {
    fn default() -> Self {
        Self {
            code: [0; IrisCodeArray::IRIS_CODE_SIZE],
            mask: [0; IrisCodeArray::IRIS_CODE_SIZE],
        }
    }
}

impl ShamirIris {
    const COLS: usize = 200;

    fn share_bit<R: Rng>(code: bool, mask: bool, rng: &mut R) -> ([u16; 3], [u16; 3]) {
        // code needs to be encoded before sharing
        let val = (code & mask) as u32;
        let to_share = ((mask as u32 + P32 + P32 - val - val) % P32) as u16;
        let code_shares = Shamir::share_d1(to_share, rng);

        // mask is directly shared
        let mask_shares = Shamir::share_d1(mask as u16, rng);
        (code_shares, mask_shares)
    }

    pub fn share_iris<R: Rng>(iris: &IrisCode, rng: &mut R) -> [ShamirIris; 3] {
        let mut result = [
            ShamirIris::default(),
            ShamirIris::default(),
            ShamirIris::default(),
        ];
        for (bitindex, (c_bit, m_bit)) in iris.code.bits().zip(iris.mask.bits()).enumerate() {
            let (code_shares, mask_shares) = Self::share_bit(c_bit, m_bit, rng);
            for (res, (code, mask)) in result
                .iter_mut()
                .zip(code_shares.into_iter().zip(mask_shares.into_iter()))
            {
                res.code[bitindex] = code;
                res.mask[bitindex] = mask;
            }
        }

        result
    }

    pub fn all_rotations(&self) -> Vec<ShamirIris> {
        let mut reference = self.clone();
        let mut result = vec![];
        reference.rotate_left(16 * 4);
        for _ in 0..31 {
            reference.rotate_right(4);
            result.push(reference.clone());
        }
        result
    }

    pub fn rotate_right(&mut self, by: usize) {
        self.code
            .chunks_exact_mut(Self::COLS * 4)
            .for_each(|chunk| chunk.rotate_right(by));

        self.mask
            .chunks_exact_mut(Self::COLS * 4)
            .for_each(|chunk| chunk.rotate_right(by));
    }

    pub fn rotate_left(&mut self, by: usize) {
        self.code
            .chunks_exact_mut(Self::COLS * 4)
            .for_each(|chunk| chunk.rotate_left(by));

        self.mask
            .chunks_exact_mut(Self::COLS * 4)
            .for_each(|chunk| chunk.rotate_left(by));
    }
}
