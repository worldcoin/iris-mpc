pub mod degree2 {
    use crate::setup::{
        galois::degree2::{GaloisRingElement, ShamirGaloisRingShare},
        iris_db::iris::IrisCodeArray,
    };

    #[derive(Debug, Clone)]
    pub struct GaloisRingIrisCodeShare {
        pub id:    usize,
        pub coefs: [u16; 12800],
    }

    impl GaloisRingIrisCodeShare {
        pub fn encode_iris_code(iris_code: &IrisCodeArray) -> [GaloisRingIrisCodeShare; 3] {
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
                let share = ShamirGaloisRingShare::encode_3_mat(&element.coefs);
                for j in 0..3 {
                    shares[j].coefs[i] = share[j].y.coefs[0];
                    shares[j].coefs[i + 1] = share[j].y.coefs[1];
                }
            }
            shares
        }
        pub fn preprocess_iris_code_query_shares(
            mut shares: [GaloisRingIrisCodeShare; 3],
        ) -> [GaloisRingIrisCodeShare; 3] {
            let lagrange_coeffs = ShamirGaloisRingShare::deg_3_lagrange_polys_at_zero();
            for i in (0..12800).step_by(2) {
                for j in 0..3 {
                    let share = GaloisRingElement {
                        coefs: [shares[j].coefs[i], shares[j].coefs[i + 1]],
                    };
                    let adjusted_share = share * lagrange_coeffs[j];
                    // we write the bits back into the flat array in the "wrong" order, such that we
                    // can do simple dot product later
                    shares[j].coefs[i] = adjusted_share.coefs[0];
                    shares[j].coefs[i + 1] = adjusted_share.coefs[1]; // Note the order of bits
                }
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
                let shares = GaloisRingIrisCodeShare::encode_iris_code(&iris_db);
                let query_shares = GaloisRingIrisCodeShare::encode_iris_code(&iris_query);
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
                let shares = GaloisRingIrisCodeShare::encode_iris_code(&iris_db);
                let query_shares = GaloisRingIrisCodeShare::encode_iris_code(&iris_query);
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
        galois::degree4::{GaloisRingElement, ShamirGaloisRingShare},
        iris_db::iris::IrisCodeArray,
    };

    #[derive(Debug, Clone)]
    pub struct GaloisRingIrisCodeShare {
        pub id:    usize,
        pub coefs: [u16; 12800],
    }

    impl GaloisRingIrisCodeShare {
        pub fn encode_iris_code(iris_code: &IrisCodeArray) -> [GaloisRingIrisCodeShare; 3] {
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
                let element = GaloisRingElement {
                    coefs: [
                        iris_code.get_bit(i) as u16,
                        iris_code.get_bit(i + 1) as u16,
                        iris_code.get_bit(i + 2) as u16,
                        iris_code.get_bit(i + 3) as u16,
                    ],
                };
                let share = ShamirGaloisRingShare::encode_3_mat(&element.coefs);
                for j in 0..3 {
                    shares[j].coefs[i] = share[j].y.coefs[0];
                    shares[j].coefs[i + 1] = share[j].y.coefs[1];
                    shares[j].coefs[i + 2] = share[j].y.coefs[2];
                    shares[j].coefs[i + 3] = share[j].y.coefs[3];
                }
            }
            shares
        }
        pub fn encode_iris_code_query(iris_code: &IrisCodeArray) -> [GaloisRingIrisCodeShare; 3] {
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
                let element = GaloisRingElement {
                    coefs: [
                        iris_code.get_bit(i) as u16,
                        iris_code.get_bit(i + 3) as u16, // Note the order of bits
                        iris_code.get_bit(i + 2) as u16,
                        iris_code.get_bit(i + 1) as u16, // Note the order of bits
                    ],
                };
                let share = ShamirGaloisRingShare::encode_3_mat(&element.coefs);
                let lagrange_coeffs = ShamirGaloisRingShare::deg_3_lagrange_polys_at_zero();
                for j in 0..3 {
                    let adjusted_share = share[j].y * lagrange_coeffs[j];
                    // we write the bits back into the flat array in the "wrong" order, such that we
                    // can do simple dot product later
                    shares[j].coefs[i] = adjusted_share.coefs[0];
                    shares[j].coefs[i + 3] = adjusted_share.coefs[1]; // Note the order of bits
                    shares[j].coefs[i + 2] = adjusted_share.coefs[2];
                    shares[j].coefs[i + 1] = adjusted_share.coefs[3]; // Note the order of bits
                }
            }
            shares
        }

        pub fn full_dot(&self, other: &GaloisRingIrisCodeShare) -> u16 {
            let mut sum = 0u16;
            for i in (0..12800).step_by(4) {
                let x = GaloisRingElement {
                    coefs: [
                        self.coefs[i],
                        self.coefs[i + 1],
                        self.coefs[i + 2],
                        self.coefs[i + 3],
                    ],
                };
                let y = GaloisRingElement {
                    coefs: [
                        other.coefs[i],
                        other.coefs[i + 3],
                        other.coefs[i + 2],
                        other.coefs[i + 1],
                    ],
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
    }

    #[cfg(test)]
    mod tests {
        use crate::setup::iris_db::iris::IrisCodeArray;
        use rand::thread_rng;

        #[test]
        fn galois_dot_trick() {
            for _ in 0..10 {
                let iris_db = IrisCodeArray::random_rng(&mut thread_rng());
                let iris_query = IrisCodeArray::random_rng(&mut thread_rng());
                let shares = super::GaloisRingIrisCodeShare::encode_iris_code(&iris_db);
                let query_shares =
                    super::GaloisRingIrisCodeShare::encode_iris_code_query(&iris_query);
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
                let shares = super::GaloisRingIrisCodeShare::encode_iris_code(&iris_db);
                let query_shares =
                    super::GaloisRingIrisCodeShare::encode_iris_code_query(&iris_query);
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
