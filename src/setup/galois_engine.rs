pub mod degree2 {
    use crate::setup::{
        galois::degree2::{GaloisRingElement, ShamirGaloisRingShare},
        iris_db::iris::IrisCodeArray,
    };
    use rand::{CryptoRng, Rng};

    #[derive(Debug, Clone)]
    pub struct GaloisRingIrisCodeShare {
        pub id:    usize,
        pub coefs: [u16; 12800],
    }

    impl GaloisRingIrisCodeShare {
        const COLS: usize = 200;

        pub fn new(id: usize, coefs: [u16; 12800]) -> Self {
            Self { id, coefs }
        }

        pub fn encode_iris_code<R: CryptoRng + Rng>(
            iris_code: &IrisCodeArray,
            mask_code: &IrisCodeArray,
            rng: &mut R,
        ) -> [GaloisRingIrisCodeShare; 3] {
            let mut shares = [
                GaloisRingIrisCodeShare {
                    id:    1,
                    coefs: [0; 12800],
                },
                GaloisRingIrisCodeShare {
                    id:    2,
                    coefs: [0; 12800],
                },
                GaloisRingIrisCodeShare {
                    id:    3,
                    coefs: [0; 12800],
                },
            ];
            let encode_mask_code = |i| {
                let m = mask_code.get_bit(i) as u16;
                let c = iris_code.get_bit(i) as u16;
                m.wrapping_sub(2 * (c & m))
            };
            for i in (0..12800).step_by(2) {
                let element = GaloisRingElement {
                    coefs: [encode_mask_code(i), encode_mask_code(i + 1)],
                };
                let share = ShamirGaloisRingShare::encode_3_mat(&element.coefs, rng);
                for j in 0..3 {
                    shares[j].coefs[i] = share[j].y.coefs[0];
                    shares[j].coefs[i + 1] = share[j].y.coefs[1];
                }
            }
            shares
        }

        pub fn encode_mask_code<R: CryptoRng + Rng>(
            iris_code: &IrisCodeArray,
            rng: &mut R,
        ) -> [GaloisRingIrisCodeShare; 3] {
            let mut shares = [
                GaloisRingIrisCodeShare {
                    id:    1,
                    coefs: [0; 12800],
                },
                GaloisRingIrisCodeShare {
                    id:    2,
                    coefs: [0; 12800],
                },
                GaloisRingIrisCodeShare {
                    id:    3,
                    coefs: [0; 12800],
                },
            ];
            for i in (0..12800).step_by(2) {
                let element = GaloisRingElement {
                    coefs: [iris_code.get_bit(i) as u16, iris_code.get_bit(i + 1) as u16],
                };
                let share = ShamirGaloisRingShare::encode_3_mat(&element.coefs, rng);
                for j in 0..3 {
                    shares[j].coefs[i] = share[j].y.coefs[0];
                    shares[j].coefs[i + 1] = share[j].y.coefs[1];
                }
            }
            shares
        }

        pub fn preprocess_iris_code_query_share(&mut self) {
            let lagrange_coeffs = ShamirGaloisRingShare::deg_3_lagrange_polys_at_zero();
            for i in (0..12800).step_by(2) {
                let new_share = GaloisRingElement {
                    coefs: [self.coefs[i], self.coefs[i + 1]],
                };
                let adjusted_self = new_share * lagrange_coeffs[self.id - 1];
                // we write the bits back into the flat array in the "wrong" order, such that we
                // can do simple dot product later
                self.coefs[i] = adjusted_self.coefs[0];
                self.coefs[i + 1] = adjusted_self.coefs[1]; // Note the order
                                                            // of bits
            }
        }

        pub fn preprocess_iris_code_query_shares(
            mut shares: [GaloisRingIrisCodeShare; 3],
        ) -> [GaloisRingIrisCodeShare; 3] {
            for i in 0..3 {
                shares[i].preprocess_iris_code_query_share();
            }
            shares
        }

        pub fn full_dot(&self, other: &GaloisRingIrisCodeShare) -> u16 {
            let mut sum = 0u16;
            for i in (0..12800).step_by(2) {
                let x = GaloisRingElement {
                    coefs: [self.coefs[i], self.coefs[i + 1]],
                };
                let y = GaloisRingElement {
                    coefs: [other.coefs[i], other.coefs[i + 1]],
                };
                let z = x * y;
                sum = sum.wrapping_add(z.coefs[0]);
            }
            sum
        }
        pub fn trick_dot(&self, other: &GaloisRingIrisCodeShare) -> u16 {
            let mut sum = 0u16;
            for i in 0..12800 {
                sum = sum.wrapping_add(self.coefs[i].wrapping_mul(other.coefs[i]));
            }
            sum
        }

        pub fn all_rotations(&self) -> Vec<GaloisRingIrisCodeShare> {
            let mut reference = self.clone();
            let mut result = vec![];
            reference.rotate_left(16);
            for _ in 0..31 {
                reference.rotate_right(1);
                result.push(reference.clone());
            }
            result
        }

        pub fn rotate_right(&mut self, by: usize) {
            self.coefs
                .chunks_exact_mut(Self::COLS * 4)
                .for_each(|chunk| chunk.rotate_right(by * 4));
        }

        pub fn rotate_left(&mut self, by: usize) {
            self.coefs
                .chunks_exact_mut(Self::COLS * 4)
                .for_each(|chunk| chunk.rotate_left(by * 4));
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::setup::{
            galois_engine::degree2::GaloisRingIrisCodeShare, iris_db::iris::IrisCodeArray,
        };
        use rand::thread_rng;

        #[test]
        fn galois_dot_trick() {
            for _ in 0..10 {
                let iris_db = IrisCodeArray::random_rng(&mut thread_rng());
                let iris_query = IrisCodeArray::random_rng(&mut thread_rng());
                let shares = GaloisRingIrisCodeShare::encode_mask_code(&iris_db, &mut thread_rng());
                let query_shares =
                    GaloisRingIrisCodeShare::encode_mask_code(&iris_query, &mut thread_rng());
                let query_shares =
                    GaloisRingIrisCodeShare::preprocess_iris_code_query_shares(query_shares);
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
            for _ in 0..10 {
                let iris_db = IrisCodeArray::random_rng(&mut thread_rng());
                let iris_query = IrisCodeArray::random_rng(&mut thread_rng());
                let shares = GaloisRingIrisCodeShare::encode_mask_code(&iris_db, &mut thread_rng());
                let query_shares =
                    GaloisRingIrisCodeShare::encode_mask_code(&iris_query, &mut thread_rng());
                let query_shares =
                    GaloisRingIrisCodeShare::preprocess_iris_code_query_shares(query_shares);
                let mut dot = [0; 3];
                for i in 0..3 {
                    dot[i] = shares[i].full_dot(&query_shares[i]);
                }
                let dot = dot.iter().fold(0u16, |acc, x| acc.wrapping_add(*x));
                let expected = (iris_db & iris_query).count_ones();
                assert_eq!(dot, expected as u16);
            }
        }
    }
}

pub mod degree4 {
    use crate::setup::{
        galois::degree4::{
            basis::{self},
            GaloisRingElement, ShamirGaloisRingShare,
        },
        iris_db::iris::IrisCodeArray,
    };
    use rand::{CryptoRng, Rng};

    #[derive(Debug, Clone)]
    pub struct GaloisRingIrisCodeShare {
        pub id:    usize,
        pub coefs: [u16; 12800],
    }

    impl GaloisRingIrisCodeShare {
        const COLS: usize = 200;

        pub fn new(id: usize, coefs: [u16; 12800]) -> Self {
            Self { id, coefs }
        }

        pub fn encode_iris_code<R: CryptoRng + Rng>(
            iris_code: &IrisCodeArray,
            mask_code: &IrisCodeArray,
            rng: &mut R,
        ) -> [GaloisRingIrisCodeShare; 3] {
            let mut shares = [
                GaloisRingIrisCodeShare {
                    id:    1,
                    coefs: [0; 12800],
                },
                GaloisRingIrisCodeShare {
                    id:    2,
                    coefs: [0; 12800],
                },
                GaloisRingIrisCodeShare {
                    id:    3,
                    coefs: [0; 12800],
                },
            ];
            let encode_mask_code = |i| {
                let m = mask_code.get_bit(i) as u16;
                let c = iris_code.get_bit(i) as u16;
                m.wrapping_sub(2 * (c & m))
            };
            for i in (0..12800).step_by(4) {
                let element = GaloisRingElement::<basis::A>::from_coefs([
                    encode_mask_code(i),
                    encode_mask_code(i + 1),
                    encode_mask_code(i + 2),
                    encode_mask_code(i + 3),
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
            iris_code: &IrisCodeArray,
            rng: &mut R,
        ) -> [GaloisRingIrisCodeShare; 3] {
            let mut shares = [
                GaloisRingIrisCodeShare {
                    id:    1,
                    coefs: [0; 12800],
                },
                GaloisRingIrisCodeShare {
                    id:    2,
                    coefs: [0; 12800],
                },
                GaloisRingIrisCodeShare {
                    id:    3,
                    coefs: [0; 12800],
                },
            ];
            for i in (0..12800).step_by(4) {
                let element = GaloisRingElement::<basis::A>::from_coefs([
                    iris_code.get_bit(i) as u16,
                    iris_code.get_bit(i + 1) as u16,
                    iris_code.get_bit(i + 2) as u16,
                    iris_code.get_bit(i + 3) as u16,
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
            let lagrange_coeffs = ShamirGaloisRingShare::deg_3_lagrange_polys_at_zero();
            for i in (0..12800).step_by(4) {
                let element = GaloisRingElement::<basis::Monomial>::from_coefs([
                    self.coefs[i],
                    self.coefs[i + 1],
                    self.coefs[i + 2],
                    self.coefs[i + 3],
                ]);
                // include lagrange coeffs
                let element = element * lagrange_coeffs[self.id - 1];
                let element = element.to_basis_B();
                self.coefs[i] = element.coefs[0];
                self.coefs[i + 1] = element.coefs[1];
                self.coefs[i + 2] = element.coefs[2];
                self.coefs[i + 3] = element.coefs[3];
            }
        }

        pub fn full_dot(&self, other: &GaloisRingIrisCodeShare) -> u16 {
            let mut sum = 0u16;
            let lagrange_coeffs = ShamirGaloisRingShare::deg_3_lagrange_polys_at_zero();
            for i in (0..12800).step_by(4) {
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
            for i in 0..12800 {
                sum = sum.wrapping_add(self.coefs[i].wrapping_mul(other.coefs[i]));
            }
            sum
        }
        pub fn all_rotations(&self) -> Vec<GaloisRingIrisCodeShare> {
            let mut reference = self.clone();
            let mut result = vec![];
            reference.rotate_left(16);
            for _ in 0..31 {
                reference.rotate_right(1);
                result.push(reference.clone());
            }
            result
        }
        pub fn rotate_right(&mut self, by: usize) {
            self.coefs
                .chunks_exact_mut(Self::COLS * 4)
                .for_each(|chunk| chunk.rotate_right(by * 4));
        }

        pub fn rotate_left(&mut self, by: usize) {
            self.coefs
                .chunks_exact_mut(Self::COLS * 4)
                .for_each(|chunk| chunk.rotate_left(by * 4));
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::setup::{
            galois_engine::degree4::GaloisRingIrisCodeShare, iris_db::iris::IrisCodeArray,
        };
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
    }
}
