use super::{reduce, MODULUS};
use crate::protocol::shared_iris::GaloisRingSharedIris;
use ampc_actor_utils::{
    execution::session::{Session, SessionHandles},
    network::mpc::NetworkValue,
    protocol::{binary::extract_msb_batch, ops::galois_ring_to_rep3},
};
use ampc_secret_sharing::shares::{bit::Bit, ring_impl::RingElement, share::Share};
use eyre::{ensure, Result};
use iris_mpc_common::{
    galois::degree4::{basis, GaloisRingElement, ShamirGaloisRingShare},
    IRIS_CODE_LENGTH, MASK_CODE_LENGTH,
};
use rand::Rng;

/// Degree-one Shamir evaluations at party points 1, 2, 3, in the original
/// `(b, w, r % 4, column, r / 4)` coefficient order. All values are canonical.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldIris {
    pub code: Vec<u16>,
    pub mask: Vec<u16>,
}

impl FieldIris {
    /// Mirror in decoded coordinate space: reverse each angular half and flip
    /// the imaginary code sign. The transformation is linear on field shares.
    pub fn mirrored(&self) -> Self {
        use iris_mpc_common::galois_engine::degree4::GaloisRingIrisCodeShare;
        let mut output = self.clone();
        for i in 0..IRIS_CODE_LENGTH {
            let j = GaloisRingIrisCodeShare::remap_new_to_mirrored_index(i);
            output.code[j] = if i >= MASK_CODE_LENGTH {
                reduce(-i64::from(self.code[i]))
            } else {
                self.code[i]
            };
            if i < MASK_CODE_LENGTH {
                output.mask[j] = self.mask[i];
            }
        }
        output
    }
}

/// Replicated *field* shares. These deliberately do not implement the power-of-two
/// ring interfaces: reducing an existing ring share modulo p is not a conversion.
#[derive(Clone, Copy, Default)]
struct ReplicatedField {
    a: u16,
    b: u16,
}

impl ReplicatedField {
    fn xor(self, rhs: Self, product: Self) -> Self {
        Self {
            a: reduce(i64::from(self.a) + i64::from(rhs.a) - 2 * i64::from(product.a)),
            b: reduce(i64::from(self.b) + i64::from(rhs.b) - 2 * i64::from(product.b)),
        }
    }

    /// Component x_i is known to parties i and i+1. Assign it the linear
    /// polynomial L_i(z)=(z_missing-z)/z_missing, where missing=(i+2)%3+1.
    /// Then L_i(0)=1 and the party missing x_i needs only its zero contribution.
    /// Summing these polynomials gives a degree-one sharing of sum_i x_i.
    fn shamir(self, party: usize) -> u16 {
        const INVERSES: [i64; 3] = [1, 26_101, 34_801];
        let weight = |component: usize| {
            let missing = (component + 2) % 3 + 1;
            (missing as i64 - (party + 1) as i64) * INVERSES[missing - 1]
        };
        reduce(i64::from(self.a) * weight(party) + i64::from(self.b) * weight((party + 2) % 3))
    }
}

/// One ABY3 multiplication round in F_p. Uniform pairwise PRF masks randomize
/// every transmitted additive component; neither inputs nor products are opened.
async fn multiply(
    session: &mut Session,
    lhs: &[ReplicatedField],
    rhs: &[ReplicatedField],
) -> Result<Vec<ReplicatedField>> {
    ensure!(
        lhs.len() == rhs.len(),
        "field multiplication length mismatch"
    );
    let own: Vec<_> = lhs
        .iter()
        .zip(rhs)
        .map(|(x, y)| {
            // Rejection sampling is intentional: reducing random u16s modulo p
            // would bias the zero shares. Each adjacent PRF consumes identically.
            let my = session.prf.my_prf.gen_range(0..MODULUS);
            let prev = session.prf.prev_prf.gen_range(0..MODULUS);
            RingElement(reduce(
                i64::from(x.a) * i64::from(y.a)
                    + i64::from(x.a) * i64::from(y.b)
                    + i64::from(x.b) * i64::from(y.a)
                    + i64::from(my)
                    - i64::from(prev),
            ))
        })
        .collect();
    session
        .network_session
        .send_next(NetworkValue::VecRing16(own.clone()))
        .await?;
    let NetworkValue::VecRing16(prev) = session.network_session.receive_prev().await? else {
        eyre::bail!("expected field multiplication vector");
    };
    ensure!(
        prev.len() == own.len(),
        "field multiplication response length mismatch"
    );
    ensure!(
        prev.iter().all(|x| x.0 < MODULUS),
        "noncanonical field share"
    );
    Ok(own
        .into_iter()
        .zip(prev)
        .map(|(a, b)| ReplicatedField { a: a.0, b: b.0 })
        .collect())
}

/// Inject XOR-shared bits into F_p using b0 XOR b1 XOR b2 =
/// (b0+b1-2*b0*b1) XOR b2. Both multiplications are privately reshared.
async fn inject_bits(session: &mut Session, bits: &[Share<Bit>]) -> Result<Vec<ReplicatedField>> {
    let party = session.own_role().index();
    let component = |index: usize| {
        bits.iter()
            .map(|bit| ReplicatedField {
                a: u16::from(party == index && bit.a.0.convert()),
                b: u16::from((party + 2) % 3 == index && bit.b.0.convert()),
            })
            .collect::<Vec<_>>()
    };
    let b0 = component(0);
    let b1 = component(1);
    let product = multiply(session, &b0, &b1).await?;
    let first: Vec<_> = b0
        .into_iter()
        .zip(b1)
        .zip(product)
        .map(|((a, b), p)| a.xor(b, p))
        .collect();
    let b2 = component(2);
    let product = multiply(session, &first, &b2).await?;
    Ok(first
        .into_iter()
        .zip(b2)
        .zip(product)
        .map(|((a, b), p)| a.xor(b, p))
        .collect())
}

/// Convert original degree-one Galois-ring iris shares into degree-one F_p
/// shares, without reconstructing an iris or sending unmasked contributions.
///
/// First apply the Galois-ring reconstruction weights and undo basis A locally;
/// masked resharing then gives replicated Z_65536 coordinates. A well-formed
/// code coordinate is v in {-1,0,1}, so v=LSB(v)-2*MSB(v); masks are bits.
/// LSB is local in XOR sharing, whereas MSB and field bit injection are MPC.
/// Finally the replicated field components define degree-one Shamir shares.
///
/// This shares the existing protocol's semi-honest security model and assumes
/// the same valid ternary-code/binary-mask input encoding. It is not a proof of
/// client input validity. All parties must supply the same ordered record batch.
pub async fn convert_irises(
    session: &mut Session,
    irises: &[&GaloisRingSharedIris],
) -> Result<Vec<FieldIris>> {
    let party = session.own_role().index();
    ensure!(
        irises
            .iter()
            .all(|x| x.code.id == party + 1 && x.mask.id == party + 1),
        "iris share belongs to another party"
    );
    let lagrange = ShamirGaloisRingShare::deg_2_lagrange_polys_at_zero()[party];
    let mut contributions =
        Vec::with_capacity(irises.len() * (IRIS_CODE_LENGTH + MASK_CODE_LENGTH));
    // Codes first, then masks, so all signs form one contiguous MPC batch.
    for coefs in irises
        .iter()
        .map(|iris| iris.code.coefs.as_slice())
        .chain(irises.iter().map(|iris| iris.mask.coefs.as_slice()))
    {
        for block in coefs.chunks_exact(4) {
            let element = GaloisRingElement::<basis::Monomial>::from_coefs(block.try_into()?);
            contributions.extend((element * lagrange).to_basis_A().coefs.map(RingElement));
        }
    }
    let ring = galois_ring_to_rep3(session, contributions).await?;
    let code_count = irises.len() * IRIS_CODE_LENGTH;
    let mut bits: Vec<_> = ring
        .iter()
        .map(|x| {
            Share::new(
                RingElement(Bit::new(x.a.0 & 1 != 0)),
                RingElement(Bit::new(x.b.0 & 1 != 0)),
            )
        })
        .collect();
    bits.extend(extract_msb_batch(session, &ring[..code_count]).await?);
    let field = inject_bits(session, &bits).await?;
    let (lsb, sign) = field.split_at(ring.len());
    let mut output = Vec::with_capacity(irises.len());
    for i in 0..irises.len() {
        let code = (i * IRIS_CODE_LENGTH..(i + 1) * IRIS_CODE_LENGTH)
            .map(|j| reduce(i64::from(lsb[j].shamir(party)) - 2 * i64::from(sign[j].shamir(party))))
            .collect();
        let mask = lsb[code_count + i * MASK_CODE_LENGTH..code_count + (i + 1) * MASK_CODE_LENGTH]
            .iter()
            .map(|x| x.shamir(party))
            .collect();
        output.push(FieldIris { code, mask });
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ampc_actor_utils::execution::local::LocalRuntime;
    use futures::future::try_join_all;
    use iris_mpc_common::{
        galois_engine::degree4::GaloisRingIrisCodeShare, iris_db::iris::IrisCode,
    };
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[tokio::test]
    async fn convert_real_galois_shares_without_opening() -> Result<()> {
        let mut rng = ChaCha20Rng::seed_from_u64(2348);
        let plaintext = [
            IrisCode::random_rng(&mut rng),
            IrisCode::random_rng(&mut rng),
        ];
        let shared: Vec<_> = plaintext
            .iter()
            .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.clone()))
            .collect();
        let runtime = LocalRuntime::mock_setup_with_channel().await?;
        let outputs = try_join_all(runtime.sessions.into_iter().enumerate().map(
            |(party, mut session)| {
                let inputs = shared.iter().map(|iris| &iris[party]).collect::<Vec<_>>();
                async move { convert_irises(&mut session, &inputs).await }
            },
        ))
        .await?;
        for (record, iris) in plaintext.iter().enumerate() {
            for index in 0..IRIS_CODE_LENGTH {
                let old = GaloisRingIrisCodeShare::remap_new_to_old_index(index);
                let expected = reduce(
                    i64::from(iris.mask.get_bit(old)) * (1 - 2 * i64::from(iris.code.get_bit(old))),
                );
                let x: [i64; 3] =
                    std::array::from_fn(|party| i64::from(outputs[party][record].code[index]));
                // All pairs reconstruct the same constant of a degree-one polynomial.
                assert_eq!(
                    reduce(2 * x[0] - x[1]),
                    expected,
                    "pair 0/1, coordinate {index}"
                );
                assert_eq!(reduce((3 * x[0] - x[2]) * 26_101), expected);
                assert_eq!(reduce(3 * x[1] - 2 * x[2]), expected);
                if index < MASK_CODE_LENGTH {
                    let m: [i64; 3] =
                        std::array::from_fn(|party| i64::from(outputs[party][record].mask[index]));
                    assert_eq!(reduce(2 * m[0] - m[1]), u16::from(iris.mask.get_bit(old)));
                    assert_eq!(
                        reduce(3 * m[1] - 2 * m[2]),
                        u16::from(iris.mask.get_bit(old))
                    );
                }
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn full_protocol_matches_original_all_rotations_and_mirror() -> Result<()> {
        use crate::protocol::ntt::{score_chunk, SpectralIris, SpectralQuery};
        let mut rng = ChaCha20Rng::seed_from_u64(2386);
        let mut irises: Vec<_> = (0..9).map(|_| IrisCode::random_rng(&mut rng)).collect();
        // Self-matches, unrelated targets, and an empty mask exercise the entire
        // score range; nine records exercise a full SIMD tile plus a tail.
        irises[8].mask = iris_mpc_common::iris_db::iris::IrisCodeArray::ZERO;
        let shared: Vec<_> = irises
            .into_iter()
            .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris))
            .collect();
        let mut expected = vec![0u16; shared.len() * 2 * 31 * 2];
        for party in 0..3 {
            for (orientation, mut query) in [shared[0][party].clone(), shared[0][party].mirrored()]
                .into_iter()
                .enumerate()
            {
                query.code.preprocess_iris_code_query_share();
                query.mask.preprocess_mask_code_query_share();
                let codes = query.code.all_rotations();
                let masks = query.mask.all_rotations();
                for (target, raw) in shared.iter().enumerate() {
                    for r in 0..31 {
                        let i = (target * 2 + orientation) * 62 + 2 * r;
                        expected[i] =
                            expected[i].wrapping_add(codes[r].trick_dot(&raw[party].code));
                        expected[i + 1] = expected[i + 1]
                            .wrapping_add(2u16.wrapping_mul(masks[r].trick_dot(&raw[party].mask)));
                    }
                }
            }
        }
        let runtime = LocalRuntime::mock_setup_with_channel().await?;
        let results = try_join_all(runtime.sessions.into_iter().enumerate().map(
            |(party, mut session)| {
                let inputs: Vec<_> = shared.iter().map(|iris| &iris[party]).collect();
                async move {
                    let field = convert_irises(&mut session, &inputs).await?;
                    let db: Vec<_> = field.iter().map(SpectralIris::prepare).collect();
                    let mirror = field[0].mirrored();
                    let query = SpectralQuery::prepare(&[&field[0], &mirror], party);
                    let local = score_chunk(&query, &db.iter().collect::<Vec<_>>());
                    Ok::<_, eyre::Report>(local)
                }
            },
        ))
        .await?;
        for (i, pair) in expected.chunks_exact(2).enumerate() {
            let want = reduce(2 * i64::from(pair[0] as i16) - i64::from(pair[1]) / 2);
            let actual = reduce(results.iter().map(|scores| i64::from(scores[i])).sum());
            assert_eq!(actual, want, "target/orientation/rotation slot {i}");
        }
        Ok(())
    }
}
