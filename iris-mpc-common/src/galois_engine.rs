pub type CompactGaloisRingShares = Vec<Vec<u8>>;

pub mod degree4 {
    use crate::{
        galois::degree4::{basis, GaloisRingElement, ShamirGaloisRingShare},
        iris_db::iris::{IrisCode, IrisCodeArray},
        IRIS_CODE_LENGTH, MASK_CODE_LENGTH,
    };
    use base64::{prelude::BASE64_STANDARD, Engine};
    use rand::{CryptoRng, Rng};
    use serde::{Deserialize, Serialize};
    use serde_big_array::BigArray;

    const CODE_COLS: usize = 200;

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

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct GaloisRingTrimmedMaskCodeShare {
        pub id:    usize,
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
        pub fn default_for_party(party_id: usize) -> Self {
            GaloisRingTrimmedMaskCodeShare {
                id:    party_id,
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
            let mut sum = 0u16;
            for i in 0..MASK_CODE_LENGTH {
                sum = sum.wrapping_add(self.coefs[i].wrapping_mul(other.coefs[i]));
            }
            sum
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct GaloisRingIrisCodeShare {
        pub id:    usize,
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
        pub const fn remap_index(i: usize) -> usize {
            let b = i / 6400;
            let w = i % 6400 / 3200;
            let r1 = i % 3200 / 800;
            let c = i % 800 / 4;
            let r2 = i % 4;
            let r = r2 * 4 + r1;
            800 * r + c * 4 + w * 2 + b
        }

        pub fn new(id: usize, coefs: [u16; IRIS_CODE_LENGTH]) -> Self {
            Self { id, coefs }
        }

        pub fn default_for_party(party_id: usize) -> Self {
            GaloisRingIrisCodeShare {
                id:    party_id,
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
                    id:    1,
                    coefs: [0; IRIS_CODE_LENGTH],
                },
                GaloisRingIrisCodeShare {
                    id:    2,
                    coefs: [0; IRIS_CODE_LENGTH],
                },
                GaloisRingIrisCodeShare {
                    id:    3,
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
                    encode_mask_code(Self::remap_index(i)),
                    encode_mask_code(Self::remap_index(i + 1)),
                    encode_mask_code(Self::remap_index(i + 2)),
                    encode_mask_code(Self::remap_index(i + 3)),
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
                    id:    1,
                    coefs: [0; IRIS_CODE_LENGTH],
                },
                GaloisRingIrisCodeShare {
                    id:    2,
                    coefs: [0; IRIS_CODE_LENGTH],
                },
                GaloisRingIrisCodeShare {
                    id:    3,
                    coefs: [0; IRIS_CODE_LENGTH],
                },
            ];
            for i in (0..IRIS_CODE_LENGTH).step_by(4) {
                let element = GaloisRingElement::<basis::A>::from_coefs([
                    mask_code.get_bit(Self::remap_index(i)) as u16,
                    mask_code.get_bit(Self::remap_index(i + 1)) as u16,
                    mask_code.get_bit(Self::remap_index(i + 2)) as u16,
                    mask_code.get_bit(Self::remap_index(i + 3)) as u16,
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
                    id:    1,
                    coefs: [0; IRIS_CODE_LENGTH],
                },
                GaloisRingIrisCodeShare {
                    id:    2,
                    coefs: [0; IRIS_CODE_LENGTH],
                },
                GaloisRingIrisCodeShare {
                    id:    3,
                    coefs: [0; IRIS_CODE_LENGTH],
                },
            ];
            for i in (0..IRIS_CODE_LENGTH).step_by(4) {
                let element = GaloisRingElement::<basis::A>::from_coefs([
                    iris_code[Self::remap_index(i)],
                    iris_code[Self::remap_index(i + 1)],
                    iris_code[Self::remap_index(i + 2)],
                    iris_code[Self::remap_index(i + 3)],
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
            let mut sum = 0u16;
            for i in 0..IRIS_CODE_LENGTH {
                sum = sum.wrapping_add(self.coefs[i].wrapping_mul(other.coefs[i]));
            }
            sum
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

        pub fn from_base64(s: &str) -> eyre::Result<Self> {
            let decoded_bytes = BASE64_STANDARD.decode(s)?;
            Ok(bincode::deserialize(&decoded_bytes)?)
        }
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

    #[cfg(test)]
    mod tests {
        use crate::{
            galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
            iris_db::iris::IrisCodeArray,
            MASK_CODE_LENGTH,
        };
        use float_eq::assert_float_eq;
        use rand::thread_rng;

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
            let t1_mask_shares = GaloisRingIrisCodeShare::encode_mask_code(&t1_mask, rng);

            let t2_code_shares = GaloisRingIrisCodeShare::encode_iris_code(&t2_code, &t2_mask, rng);
            let t2_mask_shares = GaloisRingIrisCodeShare::encode_mask_code(&t2_mask, rng);

            let mut t2_code_shares_rotated = t2_code_shares
                .iter()
                .map(|share| share.all_rotations())
                .collect::<Vec<_>>();

            let mut t2_mask_shares_rotated = t2_mask_shares
                .iter()
                .map(|share| {
                    let trimmed: GaloisRingTrimmedMaskCodeShare = share.clone().into();
                    trimmed.all_rotations()
                })
                .collect::<Vec<_>>();

            let mut min_dist = f64::MAX;
            for rot_idx in 0..31 {
                t2_code_shares_rotated
                    .iter_mut()
                    .for_each(|share| share[rot_idx].preprocess_iris_code_query_share());

                t2_mask_shares_rotated
                    .iter_mut()
                    .for_each(|share| share[rot_idx].preprocess_mask_code_query_share());

                // dot product for codes
                let mut dot_codes = [0; 3];
                for i in 0..3 {
                    dot_codes[i] = t1_code_shares[i].trick_dot(&t2_code_shares_rotated[i][rot_idx]);
                }
                let dot_codes = dot_codes.iter().fold(0u16, |acc, x| acc.wrapping_add(*x));

                // dot product for masks
                let mut dot_masks = [0; 3];
                for i in 0..3 {
                    // trick dot for mask
                    dot_masks[i] = 0u16;
                    for j in 0..MASK_CODE_LENGTH {
                        dot_masks[i] = dot_masks[i].wrapping_add(
                            t1_mask_shares[i].coefs[j]
                                .wrapping_mul(t2_mask_shares_rotated[i][rot_idx].coefs[j]),
                        );
                    }
                    dot_masks[i] = dot_masks[i].wrapping_mul(2);
                }
                let dot_masks = dot_masks.iter().fold(0u16, |acc, x| acc.wrapping_add(*x));

                let res = 0.5f64 - (dot_codes as i16) as f64 / (2f64 * dot_masks as f64);

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
    }
}
