pub type CompactGaloisRingShares = Vec<Vec<u8>>;

pub mod degree4 {
    use crate::{
        galois::degree4::{basis, GaloisRingElement, ShamirGaloisRingShare},
        iris_db::iris::{IrisCode, IrisCodeArray},
        IRIS_CODE_LENGTH, MASK_CODE_LENGTH, PRE_PROC_IRIS_CODE_LENGTH, PRE_PROC_ROW_PADDING,
    };
    use base64::{prelude::BASE64_STANDARD, Engine};
    use eyre::Result;
    use rand::{rngs::StdRng, CryptoRng, Rng, SeedableRng};
    use serde::{Deserialize, Serialize};
    use serde_big_array::BigArray;

    const CODE_COLS: usize = 200;

    /// A representation of a 100% relative distance as a fraction.
    /// The numerator is chosen as -1 meaning 1 difference.
    /// The denominator is chosen as 1 meaning 1 unmasked bits.
    /// When treated as additive shares, it becomes -3 / 3, also representing 100% distance.
    pub const SHARE_OF_MAX_DISTANCE: (u16, u16) = (u16::MAX, 1);

    fn preprocess_coefs(id: usize, coefs: &mut [u16]) {
        let lagrange_coeffs = ShamirGaloisRingShare::deg_2_lagrange_polys_at_zero();
        for i in (0..coefs.len()).step_by(4) {
            let element = GaloisRingElement::<basis::Monomial>::from_coefs([
                coefs[i],
                coefs[i + 1],
                coefs[i + 2],
                coefs[i + 3],
            ]);
            // include lagrange coeffs
            let element: GaloisRingElement<basis::Monomial> = element * lagrange_coeffs[id - 1];
            let element = element.to_basis_B();
            coefs[i] = element.coefs[0];
            coefs[i + 1] = element.coefs[1];
            coefs[i + 2] = element.coefs[2];
            coefs[i + 3] = element.coefs[3];
        }
    }

    fn rotate_coefs_right(coefs: &mut [u16], by: usize) {
        coefs
            .chunks_exact_mut(CODE_COLS * 4)
            .for_each(|chunk| chunk.rotate_right(by * 4));
    }

    fn rotate_coefs_left(coefs: &mut [u16], by: usize) {
        coefs
            .chunks_exact_mut(CODE_COLS * 4)
            .for_each(|chunk| chunk.rotate_left(by * 4));
    }

    fn trick_dot<const D: usize>(left: &[u16; D], right: &[u16; D]) -> u16 {
        let mut sum = 0u16;
        for i in 0..D {
            sum = sum.wrapping_add(left[i].wrapping_mul(right[i]));
        }
        sum
    }

    // iterate over left.coefs as though it has been rotated according to the iris rotation.
    // this involves chunking by CODE_COLS * 4 and then rotating the chunk
    // the rotation is accomplished by operating over two slices
    fn rotation_aware_trick_dot<const D: usize>(
        left: &[u16; D],
        right: &[u16; D],
        rotation: &IrisRotation,
    ) -> u16 {
        let skip = match rotation {
            IrisRotation::Center => 0,
            IrisRotation::Left(rot) => rot * 4,
            IrisRotation::Right(rot) => (CODE_COLS * 4) - (rot * 4),
        };

        let mut sum = 0u16;
        let chunk_size = CODE_COLS * 4;

        for (left_slice, right_slice) in left
            .chunks_exact(chunk_size)
            .zip(right.chunks_exact(chunk_size))
        {
            // Split the rotation into two contiguous loops,
            // allowing the compiler to vectorize
            let (left1, left2) = left_slice.split_at(skip);
            let (right1, right2) = right_slice.split_at(chunk_size - skip);

            for (l, r) in left1.iter().zip(right2.iter()) {
                sum = sum.wrapping_add(l.wrapping_mul(*r));
            }

            for (l, r) in left2.iter().zip(right1.iter()) {
                sum = sum.wrapping_add(l.wrapping_mul(*r));
            }
        }
        sum
    }

    pub fn rotation_aware_trick_dot_padded(
        left: &[u16; PRE_PROC_IRIS_CODE_LENGTH],
        right: &[u16; IRIS_CODE_LENGTH],
        rotation: &IrisRotation,
    ) -> u16 {
        let skip = match rotation {
            IrisRotation::Center => 60, // no padding (60 added on each side)
            IrisRotation::Left(rot) => 60 + (rot * 4),
            IrisRotation::Right(rot) => 60 - (rot * 4),
        };

        let mut sum = 0u16;
        const UNPADDED_ROW_LEN: usize = CODE_COLS * 4; // 800 elements per row
        const PADDED_CHUNK_SIZE: usize = UNPADDED_ROW_LEN + PRE_PROC_ROW_PADDING; // 920 elements per padded row

        // Process each row
        for (row_idx, chunk) in left.chunks_exact(PADDED_CHUNK_SIZE).enumerate() {
            // Calculate the starting index in the padded chunk
            // Each row used to be elements 0..=799 but now has:
            // - elements 740..=799 prepended (60 elements)
            // - elements 0..=59 appended (60 elements)
            // So we need to start at index `skip` to get 800 consecutive elements
            let start_idx = skip;
            let end_idx = start_idx + UNPADDED_ROW_LEN;

            // Extract the slice we need for this row
            let left_slice = &chunk[start_idx..end_idx];

            // Get corresponding slice from self
            let right_start = row_idx * UNPADDED_ROW_LEN;
            let right_end = right_start + UNPADDED_ROW_LEN;
            let right_slice = &right[right_start..right_end];

            // Compute dot product for this row
            // use explicit indices for the loop to try to help the compiler optimize with SIMD
            for i in 0..UNPADDED_ROW_LEN {
                sum = sum.wrapping_add(left_slice[i].wrapping_mul(right_slice[i]));
            }
        }

        sum
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct GaloisRingTrimmedMaskCodeShare {
        /// The 1-based ID of the Lagrange evaluation point. This id = party_id + 1.
        /// This field appears in serializations.
        pub id: usize,
        #[serde(with = "BigArray")]
        pub coefs: [u16; MASK_CODE_LENGTH],
    }

    impl From<GaloisRingIrisCodeShare> for GaloisRingTrimmedMaskCodeShare {
        fn from(iris_share: GaloisRingIrisCodeShare) -> Self {
            let mut coefs = [0; MASK_CODE_LENGTH];
            coefs.copy_from_slice(&iris_share.coefs[..MASK_CODE_LENGTH]);

            GaloisRingTrimmedMaskCodeShare {
                id: iris_share.id,
                coefs,
            }
        }
    }

    impl From<&GaloisRingIrisCodeShare> for GaloisRingTrimmedMaskCodeShare {
        fn from(iris_share: &GaloisRingIrisCodeShare) -> Self {
            let mut coefs = [0; MASK_CODE_LENGTH];
            coefs.copy_from_slice(&iris_share.coefs[..MASK_CODE_LENGTH]);

            GaloisRingTrimmedMaskCodeShare {
                id: iris_share.id,
                coefs,
            }
        }
    }

    impl GaloisRingTrimmedMaskCodeShare {
        /// Wrap a mask share. party_id is 0-based.
        #[inline(always)]
        pub fn new(coefs: [u16; MASK_CODE_LENGTH], party_id: usize) -> Self {
            Self {
                id: party_id + 1,
                coefs,
            }
        }

        /// Empty mask share. party_id is 0-based.
        pub fn default_for_party(party_id: usize) -> Self {
            GaloisRingTrimmedMaskCodeShare {
                id: party_id + 1,
                coefs: [0u16; MASK_CODE_LENGTH],
            }
        }

        pub fn preprocess_mask_code_query_share(&mut self) {
            preprocess_coefs(self.id, &mut self.coefs);
        }

        pub fn all_rotations(&self) -> Vec<GaloisRingTrimmedMaskCodeShare> {
            let mut reference = self.clone();
            let mut result = vec![];
            rotate_coefs_left(&mut reference.coefs, 16);
            for _ in 0..31 {
                rotate_coefs_right(&mut reference.coefs, 1);
                result.push(reference.clone());
            }
            result
        }

        pub fn trick_dot(&self, other: &GaloisRingTrimmedMaskCodeShare) -> u16 {
            trick_dot(&self.coefs, &other.coefs)
        }

        pub fn rotation_aware_trick_dot(
            &self,
            other: &GaloisRingTrimmedMaskCodeShare,
            rotation: &IrisRotation,
        ) -> u16 {
            rotation_aware_trick_dot(&self.coefs, &other.coefs, rotation)
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct GaloisRingIrisCodeShare {
        /// The 1-based ID of the Lagrange evaluation point. This id = party_id + 1.
        /// This field appears in serializations.
        pub id: usize,
        #[serde(with = "BigArray")]
        pub coefs: [u16; IRIS_CODE_LENGTH],
    }

    impl GaloisRingIrisCodeShare {
        // Maps from an index in a flattened array of the new shape to the
        // index in a flattened array of the original shape.
        //
        //          New shape         --> Original shape
        // (b, w, r % 4,   c, r // 4) --> ( r,   c, w, b)
        // (2, 2,     4, 200,      4) --> (16, 200, 2, 2)
        pub const fn remap_new_to_old_index(i: usize) -> usize {
            let b = i / 6400;
            let w = i % 6400 / 3200;
            let r1 = i % 3200 / 800;
            let c = i % 800 / 4;
            let r2 = i % 4;
            let r = r2 * 4 + r1;
            800 * r + c * 4 + w * 2 + b
        }

        pub const fn remap_old_to_new_index(i: usize) -> usize {
            let r = i / 800;
            let c = i % 800 / 4;
            let w = i % 4 / 2;
            let b = i % 2;
            let r1 = r % 4;
            let r2 = r / 4;
            b * 6400 + w * 3200 + r1 * 800 + c * 4 + r2
        }

        pub const fn remap_new_to_mirrored_index(i: usize) -> usize {
            let b = i / 6400;
            let w = i % 6400 / 3200;
            let r1 = i % 3200 / 800;
            let c = i % 800 / 4;
            let r2 = i % 4;
            let half_width = 100;
            let flipped_c = if c < 100 {
                half_width - 1 - c
            } else {
                199 - (c - half_width)
            };
            b * 6400 + w * 3200 + r1 * 800 + flipped_c * 4 + r2
        }

        fn flipped_imaginary_coeffs(&self) -> Self {
            let mut res = self.clone();
            let element = GaloisRingElement::ZERO;
            let mut rng = StdRng::seed_from_u64(0);
            let share = ShamirGaloisRingShare::encode_3_mat(&element.coefs, &mut rng);
            for i in IRIS_CODE_LENGTH / 2..IRIS_CODE_LENGTH {
                res.coefs[i] = share[self.id - 1].y.coefs[i % 4].wrapping_sub(res.coefs[i]);
            }
            res
        }

        /// Wrap a code share. party_id is 0-based.
        #[inline(always)]
        pub fn new(coefs: [u16; IRIS_CODE_LENGTH], party_id: usize) -> Self {
            Self {
                id: party_id + 1,
                coefs,
            }
        }

        /// Empty code share. party_id is 0-based.
        pub fn default_for_party(party_id: usize) -> Self {
            GaloisRingIrisCodeShare {
                id: party_id + 1,
                coefs: [0u16; IRIS_CODE_LENGTH],
            }
        }

        pub fn encode_iris_code<R: CryptoRng + Rng>(
            iris_code: &IrisCodeArray,
            mask_code: &IrisCodeArray,
            rng: &mut R,
        ) -> [GaloisRingIrisCodeShare; 3] {
            let mut shares = [
                GaloisRingIrisCodeShare {
                    id: 1,
                    coefs: [0; IRIS_CODE_LENGTH],
                },
                GaloisRingIrisCodeShare {
                    id: 2,
                    coefs: [0; IRIS_CODE_LENGTH],
                },
                GaloisRingIrisCodeShare {
                    id: 3,
                    coefs: [0; IRIS_CODE_LENGTH],
                },
            ];
            let encode_mask_code = |i| {
                let mask = mask_code.get_bit(i) as u16;
                let code = iris_code.get_bit(i) as u16;
                mask.wrapping_sub(2 * (code & mask))
            };
            for i in (0..IRIS_CODE_LENGTH).step_by(4) {
                let element = GaloisRingElement::<basis::A>::from_coefs([
                    encode_mask_code(Self::remap_new_to_old_index(i)),
                    encode_mask_code(Self::remap_new_to_old_index(i + 1)),
                    encode_mask_code(Self::remap_new_to_old_index(i + 2)),
                    encode_mask_code(Self::remap_new_to_old_index(i + 3)),
                ]);
                let element = element.to_monomial();
                let share = ShamirGaloisRingShare::encode_3_mat(&element.coefs, rng);
                for j in 0..3 {
                    shares[j].coefs[i] = share[j].y.coefs[0];
                    shares[j].coefs[i + 1] = share[j].y.coefs[1];
                    shares[j].coefs[i + 2] = share[j].y.coefs[2];
                    shares[j].coefs[i + 3] = share[j].y.coefs[3];
                }
            }
            shares
        }

        pub fn encode_mask_code<R: CryptoRng + Rng>(
            mask_code: &IrisCodeArray,
            rng: &mut R,
        ) -> [GaloisRingIrisCodeShare; 3] {
            let mut shares = [
                GaloisRingIrisCodeShare {
                    id: 1,
                    coefs: [0; IRIS_CODE_LENGTH],
                },
                GaloisRingIrisCodeShare {
                    id: 2,
                    coefs: [0; IRIS_CODE_LENGTH],
                },
                GaloisRingIrisCodeShare {
                    id: 3,
                    coefs: [0; IRIS_CODE_LENGTH],
                },
            ];
            for i in (0..IRIS_CODE_LENGTH).step_by(4) {
                let element = GaloisRingElement::<basis::A>::from_coefs([
                    mask_code.get_bit(Self::remap_new_to_old_index(i)) as u16,
                    mask_code.get_bit(Self::remap_new_to_old_index(i + 1)) as u16,
                    mask_code.get_bit(Self::remap_new_to_old_index(i + 2)) as u16,
                    mask_code.get_bit(Self::remap_new_to_old_index(i + 3)) as u16,
                ]);
                let element = element.to_monomial();
                let share = ShamirGaloisRingShare::encode_3_mat(&element.coefs, rng);
                for j in 0..3 {
                    shares[j].coefs[i] = share[j].y.coefs[0];
                    shares[j].coefs[i + 1] = share[j].y.coefs[1];
                    shares[j].coefs[i + 2] = share[j].y.coefs[2];
                    shares[j].coefs[i + 3] = share[j].y.coefs[3];
                }
            }
            shares
        }

        #[allow(clippy::assertions_on_constants)]
        pub fn reencode_extended_iris_code<R: CryptoRng + Rng>(
            iris_code: &[u16; IRIS_CODE_LENGTH],
            rng: &mut R,
        ) -> [GaloisRingIrisCodeShare; 3] {
            assert!(IRIS_CODE_LENGTH % 4 == 0);
            let mut shares = [
                GaloisRingIrisCodeShare {
                    id: 1,
                    coefs: [0; IRIS_CODE_LENGTH],
                },
                GaloisRingIrisCodeShare {
                    id: 2,
                    coefs: [0; IRIS_CODE_LENGTH],
                },
                GaloisRingIrisCodeShare {
                    id: 3,
                    coefs: [0; IRIS_CODE_LENGTH],
                },
            ];
            for i in (0..IRIS_CODE_LENGTH).step_by(4) {
                let element = GaloisRingElement::<basis::A>::from_coefs([
                    iris_code[Self::remap_new_to_old_index(i)],
                    iris_code[Self::remap_new_to_old_index(i + 1)],
                    iris_code[Self::remap_new_to_old_index(i + 2)],
                    iris_code[Self::remap_new_to_old_index(i + 3)],
                ]);
                let element = element.to_monomial();
                let share = ShamirGaloisRingShare::encode_3_mat(&element.coefs, rng);
                for j in 0..3 {
                    shares[j].coefs[i] = share[j].y.coefs[0];
                    shares[j].coefs[i + 1] = share[j].y.coefs[1];
                    shares[j].coefs[i + 2] = share[j].y.coefs[2];
                    shares[j].coefs[i + 3] = share[j].y.coefs[3];
                }
            }
            shares
        }

        pub fn preprocess_iris_code_query_share(&mut self) {
            preprocess_coefs(self.id, &mut self.coefs);
        }

        pub fn full_dot(&self, other: &GaloisRingIrisCodeShare) -> u16 {
            let mut sum = 0u16;
            let lagrange_coeffs = ShamirGaloisRingShare::deg_2_lagrange_polys_at_zero();
            for i in (0..IRIS_CODE_LENGTH).step_by(4) {
                let x = GaloisRingElement::from_coefs([
                    self.coefs[i],
                    self.coefs[i + 1],
                    self.coefs[i + 2],
                    self.coefs[i + 3],
                ]);
                let y = GaloisRingElement::from_coefs([
                    other.coefs[i],
                    other.coefs[i + 1],
                    other.coefs[i + 2],
                    other.coefs[i + 3],
                ]);
                let z = x * y;
                let z = z * lagrange_coeffs[self.id - 1];
                let z = z.to_basis_B();
                sum = sum.wrapping_add(z.coefs[0]);
            }
            sum
        }

        pub fn trick_dot(&self, other: &GaloisRingIrisCodeShare) -> u16 {
            trick_dot(&self.coefs, &other.coefs)
        }

        pub fn rotation_aware_trick_dot(
            &self,
            other: &GaloisRingIrisCodeShare,
            // rotation applied to self
            rotation: &IrisRotation,
        ) -> u16 {
            rotation_aware_trick_dot(&self.coefs, &other.coefs, rotation)
        }

        pub fn all_rotations(&self) -> Vec<GaloisRingIrisCodeShare> {
            let mut reference = self.clone();
            let mut result = vec![];
            rotate_coefs_left(&mut reference.coefs, 16);
            for _ in 0..31 {
                rotate_coefs_right(&mut reference.coefs, 1);
                result.push(reference.clone());
            }
            result
        }

        pub fn to_base64(&self) -> String {
            let as_vec_u8 = bincode::serialize(&self).expect("to serialize");
            BASE64_STANDARD.encode::<Vec<u8>>(as_vec_u8)
        }

        pub fn from_base64(s: &str) -> Result<Self> {
            let decoded_bytes = BASE64_STANDARD.decode(s)?;
            Ok(bincode::deserialize(&decoded_bytes)?)
        }

        pub fn mirrored_code(&self) -> Self {
            self.mirrored(true)
        }

        pub fn mirrored_mask(&self) -> Self {
            self.mirrored(false)
        }

        fn mirrored(&self, flip_imaginary: bool) -> Self {
            // Flip the coefficients corresponding to the imaginary bits
            let share = if flip_imaginary {
                self.flipped_imaginary_coeffs()
            } else {
                self.clone()
            };

            // Mirror the coefficients
            let mut res = share.clone();
            for i in 0..IRIS_CODE_LENGTH {
                res.coefs[Self::remap_new_to_mirrored_index(i)] = share.coefs[i];
            }

            res
        }
    }

    #[derive(Clone)]
    pub struct GaloisShares {
        /// Iris code from a request.
        pub code: GaloisRingIrisCodeShare,
        /// Mask from the request.
        pub mask: GaloisRingTrimmedMaskCodeShare,
        /// Iris rotations (centered iris in the middle).
        pub code_rotated: Vec<GaloisRingIrisCodeShare>,
        /// Mask rotations (centered mask in the middle).
        pub mask_rotated: Vec<GaloisRingTrimmedMaskCodeShare>,
        /// Iris rotations with Lagrange interpolations.
        pub code_interpolated: Vec<GaloisRingIrisCodeShare>,
        /// Mask rotations with Lagrange interpolations.
        pub mask_interpolated: Vec<GaloisRingTrimmedMaskCodeShare>,
        /// Iris mirrored with Lagrange interpolations.
        pub code_mirrored: Vec<GaloisRingIrisCodeShare>,
        /// Mask mirrored with Lagrange interpolations.
        pub mask_mirrored: Vec<GaloisRingTrimmedMaskCodeShare>,
    }

    pub fn preprocess_iris_message_shares(
        code_share: GaloisRingIrisCodeShare,
        mask_share: GaloisRingTrimmedMaskCodeShare,
        code_share_mirrored: GaloisRingIrisCodeShare,
        mask_share_mirrored: GaloisRingTrimmedMaskCodeShare,
    ) -> Result<GaloisShares> {
        let mut code_share = code_share;
        let mut mask_share = mask_share;

        // Original for storage.
        let store_iris_shares = code_share.clone();
        let store_mask_shares = mask_share.clone();

        // With rotations for in-memory database.
        let db_iris_shares = code_share.all_rotations();
        let db_mask_shares = mask_share.all_rotations();

        // With Lagrange interpolation.
        GaloisRingIrisCodeShare::preprocess_iris_code_query_share(&mut code_share);
        GaloisRingTrimmedMaskCodeShare::preprocess_mask_code_query_share(&mut mask_share);

        // Mirrored share and mask.
        // Only interested in the Lagrange interpolated share and mask for the mirrored case.
        let mut code_share_mirrored = code_share_mirrored;
        let mut mask_share_mirrored = mask_share_mirrored;

        // With Lagrange interpolation.
        GaloisRingIrisCodeShare::preprocess_iris_code_query_share(&mut code_share_mirrored);
        GaloisRingTrimmedMaskCodeShare::preprocess_mask_code_query_share(&mut mask_share_mirrored);

        Ok(GaloisShares {
            code: store_iris_shares,
            mask: store_mask_shares,
            code_rotated: db_iris_shares,
            mask_rotated: db_mask_shares,
            code_interpolated: code_share.all_rotations(),
            mask_interpolated: mask_share.all_rotations(),
            code_mirrored: code_share_mirrored.all_rotations(),
            mask_mirrored: mask_share_mirrored.all_rotations(),
        })
    }

    pub struct FullGaloisRingIrisCodeShare {
        pub code: GaloisRingIrisCodeShare,
        pub mask: GaloisRingTrimmedMaskCodeShare,
    }

    impl FullGaloisRingIrisCodeShare {
        pub fn encode_iris_code(
            iris: &IrisCode,
            rng: &mut (impl Rng + CryptoRng),
        ) -> [FullGaloisRingIrisCodeShare; 3] {
            let [code0, code1, code2] =
                GaloisRingIrisCodeShare::encode_iris_code(&iris.code, &iris.mask, rng);
            let [mask0, mask1, mask2] = GaloisRingIrisCodeShare::encode_mask_code(&iris.mask, rng);
            [
                FullGaloisRingIrisCodeShare {
                    code: code0,
                    mask: mask0.into(),
                },
                FullGaloisRingIrisCodeShare {
                    code: code1,
                    mask: mask1.into(),
                },
                FullGaloisRingIrisCodeShare {
                    code: code2,
                    mask: mask2.into(),
                },
            ]
        }
    }

    pub enum IrisRotation {
        Center,
        Left(usize),
        Right(usize),
    }

    impl IrisRotation {
        pub fn all() -> IrisRotationIter {
            IrisRotationIter::new()
        }
    }

    pub struct IrisRotationIter {
        idx: isize,
    }

    impl Default for IrisRotationIter {
        fn default() -> Self {
            Self::new()
        }
    }

    impl IrisRotationIter {
        pub fn new() -> Self {
            Self {
                idx: -1 * (IrisCode::ROTATIONS_PER_DIRECTION as isize),
            }
        }
    }

    impl Iterator for IrisRotationIter {
        type Item = IrisRotation;

        fn next(&mut self) -> Option<Self::Item> {
            const LEFT: isize = 0 - IrisCode::ROTATIONS_PER_DIRECTION as isize;
            const RIGHT: isize = IrisCode::ROTATIONS_PER_DIRECTION as isize;
            match self.idx {
                LEFT..=-1 => {
                    let rot = IrisRotation::Left(-self.idx as usize);
                    self.idx += 1;
                    Some(rot)
                }
                0 => {
                    self.idx += 1;
                    Some(IrisRotation::Center)
                }
                1..=RIGHT => {
                    let rot = IrisRotation::Right((self.idx) as usize);
                    self.idx += 1;
                    Some(rot)
                }
                _ => None,
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::{
            galois_engine::degree4::{
                rotate_coefs_left, rotate_coefs_right, GaloisRingIrisCodeShare,
                GaloisRingTrimmedMaskCodeShare, IrisRotation,
            },
            iris_db::iris::IrisCodeArray,
            MASK_CODE_LENGTH,
        };
        use float_eq::assert_float_eq;
        use rand::{thread_rng, Rng};

        #[test]
        fn galois_dot_trick() {
            let rng = &mut thread_rng();
            for _ in 0..10 {
                let iris_db = IrisCodeArray::random_rng(rng);
                let iris_query = IrisCodeArray::random_rng(rng);
                let shares = GaloisRingIrisCodeShare::encode_mask_code(&iris_db, rng);
                let mut query_shares = GaloisRingIrisCodeShare::encode_mask_code(&iris_query, rng);
                query_shares
                    .iter_mut()
                    .for_each(|share| share.preprocess_iris_code_query_share());
                let mut dot = [0; 3];
                for i in 0..3 {
                    dot[i] = shares[i].trick_dot(&query_shares[i]);
                }
                let dot = dot.iter().fold(0u16, |acc, x| acc.wrapping_add(*x));
                let expected = (iris_db & iris_query).count_ones();
                assert_eq!(dot, expected as u16);
            }
        }
        #[test]
        fn rotation_aware_dot_trick() {
            let rng = &mut thread_rng();
            for _ in 0..10 {
                let iris_db = IrisCodeArray::random_rng(rng);
                let iris_query = IrisCodeArray::random_rng(rng);
                let mut shares = GaloisRingIrisCodeShare::encode_mask_code(&iris_db, rng);
                let mut query_shares = GaloisRingIrisCodeShare::encode_mask_code(&iris_query, rng);
                query_shares
                    .iter_mut()
                    .for_each(|share| share.preprocess_iris_code_query_share());

                // use these for the  test
                let left = &shares[0];
                let right = &query_shares[0];

                // do rotation aware trick dot first
                let mut dots = vec![];
                for rotation in IrisRotation::all() {
                    dots.push(left.rotation_aware_trick_dot(right, &rotation));
                }

                let left = &mut shares[0];
                let right = &query_shares[0];
                let mut dots2 = vec![];
                rotate_coefs_left(&mut left.coefs, 16);
                for _ in 0..31 {
                    rotate_coefs_right(&mut left.coefs, 1);
                    dots2.push(left.trick_dot(right));
                }
                assert_eq!(dots, dots2);
            }
        }
        #[test]
        fn galois_dot_full() {
            let rng = &mut thread_rng();
            for _ in 0..10 {
                let iris_db = IrisCodeArray::random_rng(rng);
                let iris_query = IrisCodeArray::random_rng(rng);
                let shares = GaloisRingIrisCodeShare::encode_mask_code(&iris_db, rng);
                let query_shares = GaloisRingIrisCodeShare::encode_mask_code(&iris_query, rng);
                let mut dot = [0; 3];
                for i in 0..3 {
                    dot[i] = shares[i].full_dot(&query_shares[i]);
                }
                let dot = dot.iter().fold(0u16, |acc, x| acc.wrapping_add(*x));
                let expected = (iris_db & iris_query).count_ones();
                assert_eq!(dot, expected as u16);
            }
        }

        #[test]
        fn hamming_distance_galois() {
            let rng = &mut thread_rng();
            let lines = include_str!("example-data/random_codes.txt")
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

            let mask = t1_mask & t2_mask;
            let plain_distance =
                ((t1_code ^ t2_code) & mask).count_ones() as f64 / mask.count_ones() as f64;

            let t1_code_shares = GaloisRingIrisCodeShare::encode_iris_code(&t1_code, &t1_mask, rng);
            let t1_mask_shares =
                GaloisRingIrisCodeShare::encode_mask_code(&t1_mask, rng).map(|s| s.into());

            let mut t2_code_shares =
                GaloisRingIrisCodeShare::encode_iris_code(&t2_code, &t2_mask, rng);
            let mut t2_mask_shares = GaloisRingIrisCodeShare::encode_mask_code(&t2_mask, rng);

            let t2_code_shares_rotated = t2_code_shares
                .iter_mut()
                .map(|share| share.all_rotations())
                .collect::<Vec<_>>();

            let t2_mask_shares_rotated = t2_mask_shares
                .iter_mut()
                .map(|share| {
                    let trimmed: GaloisRingTrimmedMaskCodeShare = share.clone().into();
                    trimmed.all_rotations()
                })
                .collect::<Vec<_>>();

            let mut min_dist = f64::MAX;
            for rot_idx in 0..31 {
                let query_code_shares = t2_code_shares_rotated
                    .iter()
                    .map(|shares| shares[rot_idx].clone())
                    .collect::<Vec<_>>();
                let query_mask_shares = t2_mask_shares_rotated
                    .iter()
                    .map(|shares| shares[rot_idx].clone())
                    .collect::<Vec<_>>();

                let res = calculate_distance(
                    &t1_code_shares,
                    &t1_mask_shares,
                    &query_code_shares,
                    &query_mask_shares,
                );

                // Without rotations
                if rot_idx == 15 {
                    assert_float_eq!(
                        dist_0,
                        res,
                        abs <= 1e-6,
                        "galois impl distance doesn't match expected"
                    );
                    assert_float_eq!(
                        plain_distance,
                        res,
                        abs <= 1e-6,
                        "galois impl distance doesn't match reference impl"
                    );
                }

                if res < min_dist {
                    min_dist = res;
                }
            }

            // Minimum distance
            assert_float_eq!(dist_15, min_dist, abs <= 1e-6);
        }

        fn calculate_distance(
            code_shares: &[GaloisRingIrisCodeShare],
            mask_shares: &[GaloisRingTrimmedMaskCodeShare],
            query_code_shares: &[GaloisRingIrisCodeShare],
            query_mask_shares: &[GaloisRingTrimmedMaskCodeShare],
        ) -> f64 {
            let mut query_code_shares_preprocessed = vec![];
            let mut query_mask_shares_preprocessed = vec![];

            for i in 0..3 {
                let mut code = query_code_shares[i].clone();
                let mut mask = query_mask_shares[i].clone();
                code.preprocess_iris_code_query_share();
                mask.preprocess_mask_code_query_share();
                query_code_shares_preprocessed.push(code);
                query_mask_shares_preprocessed.push(mask);
            }

            let mut dot_codes = [0; 3];
            for i in 0..3 {
                dot_codes[i] = code_shares[i].trick_dot(&query_code_shares_preprocessed[i]);
            }
            let dot_codes = dot_codes.iter().fold(0u16, |acc, x| acc.wrapping_add(*x));

            // dot product for masks
            let mut dot_masks = [0; 3];
            for i in 0..3 {
                // trick dot for mask
                dot_masks[i] = 0u16;
                for j in 0..MASK_CODE_LENGTH {
                    dot_masks[i] = dot_masks[i].wrapping_add(
                        mask_shares[i].coefs[j]
                            .wrapping_mul(query_mask_shares_preprocessed[i].coefs[j]),
                    );
                }
                dot_masks[i] = dot_masks[i].wrapping_mul(2);
            }
            let dot_masks = dot_masks.iter().fold(0u16, |acc, x| acc.wrapping_add(*x));

            0.5f64 - (dot_codes as i16) as f64 / (2f64 * dot_masks as f64)
        }

        #[test]
        fn base64_shares() {
            let mut rng = thread_rng();
            let code = IrisCodeArray::random_rng(&mut rng);
            let shares = GaloisRingIrisCodeShare::encode_mask_code(&code, &mut rng);
            for i in 0..3 {
                let s = shares[i].to_base64();
                let decoded = GaloisRingIrisCodeShare::from_base64(&s).unwrap();
                assert_eq!(shares[i].coefs, decoded.coefs);
            }
        }

        #[test]
        fn match_mirrored_codes() {
            let lines = include_str!("example-data/flipped_codes.txt")
                .lines()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>();

            let code = IrisCodeArray::from_base64(lines[0]).unwrap();
            let mask = IrisCodeArray::from_base64(lines[1]).unwrap();

            let flipped_code = IrisCodeArray::from_base64(lines[2]).unwrap();
            let flipped_mask = IrisCodeArray::from_base64(lines[3]).unwrap();

            let mask = mask & flipped_mask;
            let plain_distance =
                ((code ^ flipped_code) & mask).count_ones() as f64 / mask.count_ones() as f64;

            let mut rng = thread_rng();
            let iris_code_shares =
                GaloisRingIrisCodeShare::encode_iris_code(&code, &mask, &mut rng);
            let mask_shares = GaloisRingIrisCodeShare::encode_mask_code(&mask, &mut rng);

            let iris_code_shares_incoming =
                GaloisRingIrisCodeShare::encode_iris_code(&flipped_code, &flipped_mask, &mut rng);
            let mask_shares_incoming =
                GaloisRingIrisCodeShare::encode_mask_code(&flipped_mask, &mut rng);

            let iris_code_shares_flipped = iris_code_shares_incoming
                .iter()
                .map(|share| share.mirrored_code())
                .collect::<Vec<_>>();
            let mask_shares_flipped = mask_shares_incoming
                .iter()
                .map(|share| share.mirrored_mask())
                .collect::<Vec<_>>();

            let distance = calculate_distance(
                &iris_code_shares,
                &mask_shares.clone().map(|s| s.into()),
                &iris_code_shares_incoming,
                &mask_shares_incoming.map(|s| s.into()),
            );

            assert_float_eq!(distance, plain_distance, abs <= 1e-6);
            assert_float_eq!(distance, 0.5, abs <= 0.1);

            let distance = calculate_distance(
                &iris_code_shares,
                &mask_shares.clone().map(|s| s.into()),
                &iris_code_shares_flipped,
                &mask_shares_flipped
                    .iter()
                    .map(|s| s.into())
                    .collect::<Vec<_>>(),
            );

            assert_float_eq!(distance, 0.0, abs <= 1e-6);
        }

        #[test]
        fn check_remap() {
            let mut rng = thread_rng();
            let index: usize = rng.gen_range(0..12800);
            assert_eq!(
                GaloisRingIrisCodeShare::remap_old_to_new_index(
                    GaloisRingIrisCodeShare::remap_new_to_old_index(index)
                ),
                index
            );
            assert_eq!(
                GaloisRingIrisCodeShare::remap_new_to_old_index(
                    GaloisRingIrisCodeShare::remap_old_to_new_index(index)
                ),
                index
            );
        }
    }
}
