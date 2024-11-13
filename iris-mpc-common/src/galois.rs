pub mod degree4 {
    use crate::id::PartyID;
    use basis::{Basis, Monomial};
    use rand::{CryptoRng, Rng};
    use std::marker::PhantomData;

    pub mod basis {
        pub trait Basis {}
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct A;
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct B;
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct Monomial;

        impl Basis for A {}
        impl Basis for B {}
        impl Basis for Monomial {}
    }

    /// An element of the Galois ring `$\mathbb{Z}_{2^{16}}[x]/(x^4 - x - 1)$`.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct GaloisRingElement<B: Basis> {
        pub coefs: [u16; 4],
        basis:     PhantomData<B>,
    }

    impl GaloisRingElement<Monomial> {
        pub const ZERO: GaloisRingElement<Monomial> = GaloisRingElement {
            coefs: [0, 0, 0, 0],
            basis: PhantomData,
        };
        pub const ONE: GaloisRingElement<Monomial> = GaloisRingElement {
            coefs: [1, 0, 0, 0],
            basis: PhantomData,
        };
        pub const EXCEPTIONAL_SEQUENCE: [GaloisRingElement<Monomial>; 16] = [
            GaloisRingElement::from_coefs([0, 0, 0, 0]),
            GaloisRingElement::from_coefs([1, 0, 0, 0]),
            GaloisRingElement::from_coefs([0, 1, 0, 0]),
            GaloisRingElement::from_coefs([1, 1, 0, 0]),
            GaloisRingElement::from_coefs([0, 0, 1, 0]),
            GaloisRingElement::from_coefs([1, 0, 1, 0]),
            GaloisRingElement::from_coefs([0, 1, 1, 0]),
            GaloisRingElement::from_coefs([1, 1, 1, 0]),
            GaloisRingElement::from_coefs([0, 0, 0, 1]),
            GaloisRingElement::from_coefs([1, 0, 0, 1]),
            GaloisRingElement::from_coefs([0, 1, 0, 1]),
            GaloisRingElement::from_coefs([1, 1, 0, 1]),
            GaloisRingElement::from_coefs([0, 0, 1, 1]),
            GaloisRingElement::from_coefs([1, 0, 1, 1]),
            GaloisRingElement::from_coefs([0, 1, 1, 1]),
            GaloisRingElement::from_coefs([1, 1, 1, 1]),
        ];
        pub fn encode1(x: &[u16]) -> Option<Vec<Self>> {
            if x.len() % 4 != 0 {
                return None;
            }
            Some(
                x.chunks_exact(4)
                    .map(|c| GaloisRingElement {
                        coefs: [c[0], c[3], c[2], c[1]],
                        basis: PhantomData,
                    })
                    .collect(),
            )
        }
        pub fn encode2(x: &[u16]) -> Option<Vec<Self>> {
            if x.len() % 4 != 0 {
                return None;
            }
            Some(
                x.chunks_exact(4)
                    .map(|c| GaloisRingElement {
                        coefs: [c[0], c[1], c[2], c[3]],
                        basis: PhantomData,
                    })
                    .collect(),
            )
        }

        /// Inverse of the element, if it exists
        ///
        /// # Panics
        ///
        /// This function panics if the element has no inverse
        pub fn inverse(&self) -> Self {
            // hard-coded inverses for some elements we need
            // too lazy to implement the general case in rust
            // and we do not need the general case, since this is only used for the lagrange
            // polys, which can be pre-computed anyway

            if self.coefs.iter().all(|x| x % 2 == 0) {
                panic!("Element has no inverse");
            }

            // inversion by exponentition by (p^r -1) * p^(m-1) - 1, with p = 2, r = 4, m =
            // 16
            const P: u32 = 2;
            const R: u32 = 4;
            const M: u32 = 16;
            const EXP: u32 = (P.pow(R) - 1) * P.pow(M - 1) - 1;

            self.pow(EXP)
        }

        /// Basic exponentiation by squaring, not constant time
        pub fn pow(&self, mut exp: u32) -> Self {
            if exp == 0 {
                return Self::ONE;
            }
            let mut x = self.clone();
            let mut y = Self::ONE;
            while exp > 1 {
                if exp % 2 == 1 {
                    y = x * y;
                    exp = exp - 1;
                }
                x = x * x;
                exp = exp / 2;
            }
            x * y
        }

        #[allow(non_snake_case)]
        pub fn to_basis_A(&self) -> GaloisRingElement<basis::A> {
            // Multiplication with matrix (S)^-1
            // [    1     0     0     0]
            // [ 7454     1     0     0]
            // [35057 40342     1     0]
            // [37176 61738  8525     1]
            GaloisRingElement {
                coefs: [
                    self.coefs[0],
                    self.coefs[1].wrapping_add(self.coefs[0].wrapping_mul(7454)),
                    self.coefs[2]
                        .wrapping_add(self.coefs[0].wrapping_mul(35057))
                        .wrapping_add(self.coefs[1].wrapping_mul(40342)),
                    self.coefs[3]
                        .wrapping_add(self.coefs[0].wrapping_mul(37176))
                        .wrapping_add(self.coefs[1].wrapping_mul(61738))
                        .wrapping_add(self.coefs[2].wrapping_mul(8525)),
                ],
                basis: PhantomData,
            }
        }

        #[allow(non_snake_case)]
        pub fn to_basis_B(&self) -> GaloisRingElement<basis::B> {
            // Multiplication with matrix (S*S)^-1
            // [16038 45700 28361 37176]
            // [45700 28361 37176 61738]
            // [28361 37176 61738  8525]
            // [37176 61738  8525     1]
            GaloisRingElement {
                coefs: [
                    (self.coefs[0].wrapping_mul(16038))
                        .wrapping_add(self.coefs[1].wrapping_mul(45700))
                        .wrapping_add(self.coefs[2].wrapping_mul(28361))
                        .wrapping_add(self.coefs[3].wrapping_mul(37176)),
                    (self.coefs[0].wrapping_mul(45700))
                        .wrapping_add(self.coefs[1].wrapping_mul(28361))
                        .wrapping_add(self.coefs[2].wrapping_mul(37176))
                        .wrapping_add(self.coefs[3].wrapping_mul(61738)),
                    (self.coefs[0].wrapping_mul(28361))
                        .wrapping_add(self.coefs[1].wrapping_mul(37176))
                        .wrapping_add(self.coefs[2].wrapping_mul(61738))
                        .wrapping_add(self.coefs[3].wrapping_mul(8525)),
                    (self.coefs[0].wrapping_mul(37176))
                        .wrapping_add(self.coefs[1].wrapping_mul(61738))
                        .wrapping_add(self.coefs[2].wrapping_mul(8525))
                        .wrapping_add(self.coefs[3].wrapping_mul(1)),
                ],
                basis: PhantomData,
            }
        }
    }

    impl<B: Basis> std::ops::Add for GaloisRingElement<B> {
        type Output = Self;
        fn add(self, rhs: Self) -> Self::Output {
            self.add(&rhs)
        }
    }
    impl<B: Basis> std::ops::Add<&GaloisRingElement<B>> for GaloisRingElement<B> {
        type Output = Self;
        fn add(mut self, rhs: &Self) -> Self::Output {
            for i in 0..4 {
                self.coefs[i] = self.coefs[i].wrapping_add(rhs.coefs[i]);
            }
            self
        }
    }

    impl<B: Basis> std::ops::Sub for GaloisRingElement<B> {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self::Output {
            self.sub(&rhs)
        }
    }
    impl<B: Basis> std::ops::Sub<&GaloisRingElement<B>> for GaloisRingElement<B> {
        type Output = Self;
        fn sub(mut self, rhs: &Self) -> Self::Output {
            for i in 0..4 {
                self.coefs[i] = self.coefs[i].wrapping_sub(rhs.coefs[i]);
            }
            self
        }
    }

    impl<B: Basis> std::ops::Neg for GaloisRingElement<B> {
        type Output = Self;

        fn neg(self) -> Self::Output {
            GaloisRingElement {
                coefs: [
                    self.coefs[0].wrapping_neg(),
                    self.coefs[1].wrapping_neg(),
                    self.coefs[2].wrapping_neg(),
                    self.coefs[3].wrapping_neg(),
                ],
                basis: PhantomData,
            }
        }
    }

    impl std::ops::Mul for GaloisRingElement<Monomial> {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self::Output {
            self.mul(&rhs)
        }
    }
    impl std::ops::Mul<&GaloisRingElement<Monomial>> for GaloisRingElement<Monomial> {
        type Output = Self;
        fn mul(self, rhs: &Self) -> Self::Output {
            GaloisRingElement {
                coefs: [
                    (self.coefs[0].wrapping_mul(rhs.coefs[0]))
                        .wrapping_add(self.coefs[1].wrapping_mul(rhs.coefs[3]))
                        .wrapping_add(self.coefs[2].wrapping_mul(rhs.coefs[2]))
                        .wrapping_add(self.coefs[3].wrapping_mul(rhs.coefs[1])),
                    (self.coefs[0].wrapping_mul(rhs.coefs[1]))
                        .wrapping_add(self.coefs[1].wrapping_mul(rhs.coefs[0]))
                        .wrapping_add(self.coefs[1].wrapping_mul(rhs.coefs[3]))
                        .wrapping_add(self.coefs[2].wrapping_mul(rhs.coefs[2]))
                        .wrapping_add(self.coefs[3].wrapping_mul(rhs.coefs[1]))
                        .wrapping_add(self.coefs[2].wrapping_mul(rhs.coefs[3]))
                        .wrapping_add(self.coefs[3].wrapping_mul(rhs.coefs[2])),
                    (self.coefs[0].wrapping_mul(rhs.coefs[2]))
                        .wrapping_add(self.coefs[1].wrapping_mul(rhs.coefs[1]))
                        .wrapping_add(self.coefs[2].wrapping_mul(rhs.coefs[0]))
                        .wrapping_add(self.coefs[2].wrapping_mul(rhs.coefs[3]))
                        .wrapping_add(self.coefs[3].wrapping_mul(rhs.coefs[2]))
                        .wrapping_add(self.coefs[3].wrapping_mul(rhs.coefs[3])),
                    (self.coefs[0].wrapping_mul(rhs.coefs[3]))
                        .wrapping_add(self.coefs[1].wrapping_mul(rhs.coefs[2]))
                        .wrapping_add(self.coefs[2].wrapping_mul(rhs.coefs[1]))
                        .wrapping_add(self.coefs[3].wrapping_mul(rhs.coefs[0]))
                        .wrapping_add(self.coefs[3].wrapping_mul(rhs.coefs[3])),
                ],
                basis: PhantomData,
            }
        }
    }

    impl GaloisRingElement<basis::A> {
        pub fn to_monomial(&self) -> GaloisRingElement<Monomial> {
            // Multiplication with the matrix S
            // S = [
            //     [    1     0     0     0]
            //     [58082     1     0     0]
            //     [60579 25194     1     0]
            //     [17325 51956 57011     1]
            // ]
            GaloisRingElement {
                coefs: [
                    self.coefs[0],
                    self.coefs[1].wrapping_add(self.coefs[0].wrapping_mul(58082)),
                    self.coefs[2]
                        .wrapping_add(self.coefs[0].wrapping_mul(60579))
                        .wrapping_add(self.coefs[1].wrapping_mul(25194)),
                    self.coefs[3]
                        .wrapping_add(self.coefs[0].wrapping_mul(17325))
                        .wrapping_add(self.coefs[1].wrapping_mul(51956))
                        .wrapping_add(self.coefs[2].wrapping_mul(57011)),
                ],
                basis: PhantomData,
            }
        }
    }

    impl<B: Basis> GaloisRingElement<B> {
        pub const fn from_coefs(coefs: [u16; 4]) -> Self {
            GaloisRingElement {
                coefs,
                basis: PhantomData,
            }
        }
        pub fn random(rng: &mut (impl Rng + CryptoRng)) -> Self {
            GaloisRingElement {
                coefs: rng.gen(),
                basis: PhantomData,
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ShamirGaloisRingShare {
        pub id: usize,
        pub y:  GaloisRingElement<Monomial>,
    }
    impl std::ops::Add for ShamirGaloisRingShare {
        type Output = Self;
        fn add(self, rhs: Self) -> Self::Output {
            assert_eq!(self.id, rhs.id, "ids must be euqal");
            ShamirGaloisRingShare {
                id: self.id,
                y:  self.y + rhs.y,
            }
        }
    }
    impl std::ops::Mul for ShamirGaloisRingShare {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self::Output {
            assert_eq!(self.id, rhs.id, "ids must be euqal");
            ShamirGaloisRingShare {
                id: self.id,
                y:  self.y * rhs.y,
            }
        }
    }
    impl std::ops::Sub for ShamirGaloisRingShare {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self::Output {
            assert_eq!(self.id, rhs.id, "ids must be euqal");
            ShamirGaloisRingShare {
                id: self.id,
                y:  self.y - rhs.y,
            }
        }
    }

    impl ShamirGaloisRingShare {
        pub fn encode_3<R: CryptoRng + Rng>(
            input: &GaloisRingElement<Monomial>,
            rng: &mut R,
        ) -> [ShamirGaloisRingShare; 3] {
            let coefs = [*input, GaloisRingElement::random(rng)];
            (1..=3)
                .map(|i| {
                    let element = GaloisRingElement::EXCEPTIONAL_SEQUENCE[i];
                    let share = coefs[0] + coefs[1] * element;
                    ShamirGaloisRingShare { id: i, y: share }
                })
                .collect::<Vec<_>>()
                .as_slice()
                .try_into()
                .unwrap()
        }

        pub fn encode_3_mat<R: CryptoRng + Rng>(
            input: &[u16; 4],
            rng: &mut R,
        ) -> [ShamirGaloisRingShare; 3] {
            let invec = [
                input[0],
                input[1],
                input[2],
                input[3],
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
            ];
            let share1 = ShamirGaloisRingShare {
                id: 1,
                y:  GaloisRingElement::from_coefs([
                    invec[0].wrapping_add(invec[4]),
                    invec[1].wrapping_add(invec[5]),
                    invec[2].wrapping_add(invec[6]),
                    invec[3].wrapping_add(invec[7]),
                ]),
            };
            let share2 = ShamirGaloisRingShare {
                id: 2,
                y:  GaloisRingElement::from_coefs([
                    invec[0].wrapping_add(invec[7]),
                    invec[1].wrapping_add(invec[4]).wrapping_add(invec[7]),
                    invec[2].wrapping_add(invec[5]),
                    invec[3].wrapping_add(invec[6]),
                ]),
            };
            let share3 = ShamirGaloisRingShare {
                id: 3,
                y:  GaloisRingElement::from_coefs([
                    share2.y.coefs[0].wrapping_add(invec[4]),
                    share2.y.coefs[1].wrapping_add(invec[5]),
                    share2.y.coefs[2].wrapping_add(invec[6]),
                    share2.y.coefs[3].wrapping_add(invec[7]),
                ]),
            };
            [share1, share2, share3]
        }

        pub fn deg_1_lagrange_polys_at_zero(
            my_id: PartyID,
            other_id: PartyID,
        ) -> GaloisRingElement<Monomial> {
            let mut res = GaloisRingElement::ONE;
            let i = usize::from(my_id) + 1;
            let j = usize::from(other_id) + 1;
            res = res * (-GaloisRingElement::EXCEPTIONAL_SEQUENCE[j]);
            res = res
                * (GaloisRingElement::EXCEPTIONAL_SEQUENCE[i]
                    - GaloisRingElement::EXCEPTIONAL_SEQUENCE[j])
                    .inverse();
            res
        }

        pub fn deg_2_lagrange_polys_at_zero() -> [GaloisRingElement<Monomial>; 3] {
            let mut res = [GaloisRingElement::ONE; 3];
            for i in 1..=3 {
                for j in 1..=3 {
                    if j != i {
                        res[i - 1] = res[i - 1] * (-GaloisRingElement::EXCEPTIONAL_SEQUENCE[j]);
                        res[i - 1] = res[i - 1]
                            * (GaloisRingElement::EXCEPTIONAL_SEQUENCE[i]
                                - GaloisRingElement::EXCEPTIONAL_SEQUENCE[j])
                                .inverse();
                    }
                }
            }
            res
        }

        pub fn reconstruct_deg_2_shares(
            shares: &[ShamirGaloisRingShare; 3],
        ) -> GaloisRingElement<Monomial> {
            let lagrange_polys_at_zero = Self::deg_2_lagrange_polys_at_zero();
            shares
                .iter()
                .map(|s| s.y * lagrange_polys_at_zero[s.id - 1])
                .reduce(|a, b| a + b)
                .unwrap()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{GaloisRingElement, ShamirGaloisRingShare};
        use crate::galois::degree4::basis;

        #[test]
        fn exceptional_sequence_is_pairwise_diff_invertible() {
            for i in 0..GaloisRingElement::EXCEPTIONAL_SEQUENCE.len() {
                for j in 0..GaloisRingElement::EXCEPTIONAL_SEQUENCE.len() {
                    if i != j {
                        let diff = GaloisRingElement::EXCEPTIONAL_SEQUENCE[i]
                            - GaloisRingElement::EXCEPTIONAL_SEQUENCE[j];
                        assert_eq!(diff.inverse() * diff, GaloisRingElement::ONE);
                    }
                }
            }
        }

        #[test]
        fn random_inverses() {
            for _ in 0..100 {
                let mut g_e = GaloisRingElement::random(&mut rand::thread_rng());
                // make it have an inverse
                g_e.coefs.iter_mut().for_each(|x| *x |= 1);

                assert_eq!(g_e.inverse() * g_e, GaloisRingElement::ONE);
            }
        }
        #[test]
        fn sharing() {
            let input1 = GaloisRingElement::random(&mut rand::thread_rng());
            let input2 = GaloisRingElement::random(&mut rand::thread_rng());

            let shares1 = ShamirGaloisRingShare::encode_3(&input1, &mut rand::thread_rng());
            let shares2 = ShamirGaloisRingShare::encode_3(&input2, &mut rand::thread_rng());
            let shares_mul = [
                shares1[0] * shares2[0],
                shares1[1] * shares2[1],
                shares1[2] * shares2[2],
            ];

            let reconstructed = ShamirGaloisRingShare::reconstruct_deg_2_shares(&shares_mul);
            let expected = input1 * input2;

            assert_eq!(reconstructed, expected);
        }
        #[test]
        fn sharing_mat() {
            let input1 = GaloisRingElement::random(&mut rand::thread_rng());
            let input2 = GaloisRingElement::random(&mut rand::thread_rng());

            let shares1 =
                ShamirGaloisRingShare::encode_3_mat(&input1.coefs, &mut rand::thread_rng());
            let shares2 =
                ShamirGaloisRingShare::encode_3_mat(&input2.coefs, &mut rand::thread_rng());
            let shares_mul = [
                shares1[0] * shares2[0],
                shares1[1] * shares2[1],
                shares1[2] * shares2[2],
            ];

            let reconstructed = ShamirGaloisRingShare::reconstruct_deg_2_shares(&shares_mul);
            let expected = input1 * input2;

            assert_eq!(reconstructed, expected);
        }

        fn dot(a: &[u16; 4], b: &[u16; 4]) -> u16 {
            a.iter()
                .zip(b.iter())
                .fold(0u16, |x, (a, b)| x.wrapping_add(a.wrapping_mul(*b)))
        }

        #[test]
        fn basis_conversions() {
            // essentially re-implementing the checks from
            // gr4_encoding_example.sage
            let input1 = GaloisRingElement::<basis::A>::random(&mut rand::thread_rng());
            let input2 = GaloisRingElement::<basis::A>::random(&mut rand::thread_rng());
            let result = dot(&input1.coefs, &input2.coefs);
            let monomial1 = input1.to_monomial();
            assert!(monomial1.to_basis_A() == input1);
            let monomial2 = input2.to_monomial();
            let res2 = monomial1 * monomial2;
            // TODO this vector needs to be adapted
            // let test_lin_comb = [1, 50642, 57413, 17471];
            // assert_eq!(result, dot(&res2.coefs, &test_lin_comb));

            let res3 = res2.to_basis_B();
            assert_eq!(res3.coefs[0], result);

            let basis_b1 = monomial1.to_basis_B();

            let result4 = dot(&basis_b1.coefs, &monomial2.coefs);
            assert_eq!(result, result4);
        }
    }
}
