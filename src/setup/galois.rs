use rand::{CryptoRng, Rng};

/// An element of the Galois ring $\mathbb{Z}_{2^{16}}[x]/(x^4 - x - 1)$.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GaloisRingElement {
    pub coefs: [u16; 4],
}

impl GaloisRingElement {
    pub const ZERO: GaloisRingElement = GaloisRingElement {
        coefs: [0, 0, 0, 0],
    };
    pub const ONE: GaloisRingElement = GaloisRingElement {
        coefs: [1, 0, 0, 0],
    };
    pub const EXCEPTIONAL_SEQUENCE: [GaloisRingElement; 4] = [
        GaloisRingElement::ZERO,
        GaloisRingElement::ONE,
        GaloisRingElement {
            coefs: [0, 1, 0, 0],
        },
        GaloisRingElement {
            coefs: [1, 1, 0, 0],
        },
    ];
    pub fn encode1(x: &[u16]) -> Option<Vec<Self>> {
        if x.len() % 4 != 0 {
            return None;
        }
        Some(
            x.chunks_exact(4)
                .map(|c| GaloisRingElement {
                    coefs: [c[0], c[3], c[2], c[1]],
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
                })
                .collect(),
        )
    }

    pub fn random(rng: &mut (impl Rng + CryptoRng)) -> Self {
        GaloisRingElement { coefs: rng.gen() }
    }

    pub fn inverse(&self) -> Self {
        // hard-coded inverses for some elements we need
        // too lazy to implement the general case in rust
        // and we do not need the general case, since this is only used for the lagrange polys, which can be pre-computed anyway

        if *self == GaloisRingElement::ZERO {
            panic!("Division by zero");
        }

        if *self == GaloisRingElement::ONE {
            return GaloisRingElement::ONE;
        }

        if *self == -GaloisRingElement::ONE {
            return -GaloisRingElement::ONE;
        }
        if *self
            == (GaloisRingElement {
                coefs: [0, 1, 0, 0],
            })
        {
            return GaloisRingElement {
                coefs: [65535, 0, 0, 1],
            };
        }
        if *self
            == (GaloisRingElement {
                coefs: [0, 65535, 0, 0],
            })
        {
            return GaloisRingElement {
                coefs: [1, 0, 0, 65535],
            };
        }
        if *self
            == (GaloisRingElement {
                coefs: [1, 1, 0, 0],
            })
        {
            return GaloisRingElement {
                coefs: [2, 65535, 1, 65535],
            };
        }
        if *self
            == (GaloisRingElement {
                coefs: [1, 65535, 0, 0],
            })
        {
            return GaloisRingElement {
                coefs: [0, 65535, 65535, 65535],
            };
        }
        if *self
            == (GaloisRingElement {
                coefs: [65535, 1, 0, 0],
            })
        {
            return GaloisRingElement {
                coefs: [0, 1, 1, 1],
            };
        }

        panic!("No inverse for {:?} in LUT", self);
    }
}

impl std::ops::Add for GaloisRingElement {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}
impl std::ops::Add<&GaloisRingElement> for GaloisRingElement {
    type Output = Self;
    fn add(mut self, rhs: &Self) -> Self::Output {
        for i in 0..4 {
            self.coefs[i] = self.coefs[i].wrapping_add(rhs.coefs[i]);
        }
        self
    }
}

impl std::ops::Sub for GaloisRingElement {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self - &rhs
    }
}
impl std::ops::Sub<&GaloisRingElement> for GaloisRingElement {
    type Output = Self;
    fn sub(mut self, rhs: &Self) -> Self::Output {
        for i in 0..4 {
            self.coefs[i] = self.coefs[i].wrapping_sub(rhs.coefs[i]);
        }
        self
    }
}

impl std::ops::Neg for GaloisRingElement {
    type Output = Self;

    fn neg(self) -> Self::Output {
        GaloisRingElement {
            coefs: [
                self.coefs[0].wrapping_neg(),
                self.coefs[1].wrapping_neg(),
                self.coefs[2].wrapping_neg(),
                self.coefs[3].wrapping_neg(),
            ],
        }
    }
}

impl std::ops::Mul for GaloisRingElement {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}
impl std::ops::Mul<&GaloisRingElement> for GaloisRingElement {
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
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShamirGaloisRingShare {
    pub id: usize,
    pub y: GaloisRingElement,
}
impl std::ops::Add for ShamirGaloisRingShare {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.id, rhs.id, "ids must be euqal");
        ShamirGaloisRingShare {
            id: self.id,
            y: self.y + &rhs.y,
        }
    }
}
impl std::ops::Mul for ShamirGaloisRingShare {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.id, rhs.id, "ids must be euqal");
        ShamirGaloisRingShare {
            id: self.id,
            y: self.y * &rhs.y,
        }
    }
}
impl std::ops::Sub for ShamirGaloisRingShare {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.id, rhs.id, "ids must be euqal");
        ShamirGaloisRingShare {
            id: self.id,
            y: self.y - &rhs.y,
        }
    }
}

impl ShamirGaloisRingShare {
    pub fn encode_3(input: &GaloisRingElement) -> [ShamirGaloisRingShare; 3] {
        let mut rng = rand::thread_rng();
        let coefs = [*input, GaloisRingElement::random(&mut rng)];
        (1..=3)
            .map(|i| {
                let element = GaloisRingElement::EXCEPTIONAL_SEQUENCE[i];
                let share = coefs[0] + coefs[1] * &element;
                ShamirGaloisRingShare { id: i, y: share }
            })
            .collect::<Vec<_>>()
            .as_slice()
            .try_into()
            .unwrap()
    }

    pub fn encode_3_mat(input: &[u16; 4]) -> [ShamirGaloisRingShare; 3] {
        let mut rng = rand::thread_rng();
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
            y: GaloisRingElement {
                coefs: [
                    invec[0].wrapping_add(invec[4]),
                    invec[1].wrapping_add(invec[5]),
                    invec[2].wrapping_add(invec[6]),
                    invec[3].wrapping_add(invec[7]),
                ],
            },
        };
        let share2 = ShamirGaloisRingShare {
            id: 2,
            y: GaloisRingElement {
                coefs: [
                    invec[0].wrapping_add(invec[7]),
                    invec[1].wrapping_add(invec[4]).wrapping_add(invec[7]),
                    invec[2].wrapping_add(invec[5]),
                    invec[3].wrapping_add(invec[6]),
                ],
            },
        };
        let share3 = ShamirGaloisRingShare {
            id: 3,
            y: GaloisRingElement {
                coefs: [
                    share2.y.coefs[0].wrapping_add(invec[4]),
                    share2.y.coefs[1].wrapping_add(invec[5]),
                    share2.y.coefs[2].wrapping_add(invec[6]),
                    share2.y.coefs[3].wrapping_add(invec[7]),
                ],
            },
        };
        [share1, share2, share3]
    }

    pub fn deg_3_lagrange_polys_at_zero() -> [GaloisRingElement; 3] {
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

    pub fn reconstruct_deg_2_shares(shares: &[ShamirGaloisRingShare; 3]) -> GaloisRingElement {
        let lagrange_polys_at_zero = Self::deg_3_lagrange_polys_at_zero();
        shares
            .iter()
            .map(|s| s.y * &lagrange_polys_at_zero[s.id - 1])
            .reduce(|a, b| a + b)
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::{GaloisRingElement, ShamirGaloisRingShare};

    #[test]
    fn inverses() {
        for g_e in [
            GaloisRingElement::ONE,
            -GaloisRingElement::ONE,
            GaloisRingElement::EXCEPTIONAL_SEQUENCE[2],
            GaloisRingElement::EXCEPTIONAL_SEQUENCE[3],
            GaloisRingElement::EXCEPTIONAL_SEQUENCE[1] - GaloisRingElement::EXCEPTIONAL_SEQUENCE[2],
            GaloisRingElement::EXCEPTIONAL_SEQUENCE[1] - GaloisRingElement::EXCEPTIONAL_SEQUENCE[3],
            GaloisRingElement::EXCEPTIONAL_SEQUENCE[2] - GaloisRingElement::EXCEPTIONAL_SEQUENCE[1],
            GaloisRingElement::EXCEPTIONAL_SEQUENCE[2] - GaloisRingElement::EXCEPTIONAL_SEQUENCE[3],
            GaloisRingElement::EXCEPTIONAL_SEQUENCE[3] - GaloisRingElement::EXCEPTIONAL_SEQUENCE[1],
            GaloisRingElement::EXCEPTIONAL_SEQUENCE[3] - GaloisRingElement::EXCEPTIONAL_SEQUENCE[2],
        ] {
            assert_eq!(g_e.inverse() * g_e, GaloisRingElement::ONE);
        }
    }
    #[test]
    fn sharing() {
        let input1 = GaloisRingElement::random(&mut rand::thread_rng());
        let input2 = GaloisRingElement::random(&mut rand::thread_rng());

        let shares1 = ShamirGaloisRingShare::encode_3(&input1);
        let shares2 = ShamirGaloisRingShare::encode_3(&input2);
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

        let shares1 = ShamirGaloisRingShare::encode_3_mat(&input1.coefs);
        let shares2 = ShamirGaloisRingShare::encode_3_mat(&input2.coefs);
        let shares_mul = [
            shares1[0] * shares2[0],
            shares1[1] * shares2[1],
            shares1[2] * shares2[2],
        ];

        let reconstructed = ShamirGaloisRingShare::reconstruct_deg_2_shares(&shares_mul);
        let expected = input1 * input2;

        assert_eq!(reconstructed, expected);
    }
}
