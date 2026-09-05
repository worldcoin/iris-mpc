//! Exact anonymous-statistics threshold directly from local F_52201 scores.
//! Only the final predicate is opened by the caller. No edaBits are consumed.
use super::{reduce, MODULUS};
use ampc_actor_utils::{
    execution::session::{Session, SessionHandles},
    network::mpc::NetworkValue,
};
use ampc_secret_sharing::shares::{
    bit::Bit, ring_impl::RingElement, share::Share, vecshare_bittranspose::Transpose64, VecShare,
};
use eyre::{ensure, Result};
use num_traits::Zero;
use rand::Rng;

/// Refresh degree-two local contributions, then binary-share u=(a+b) mod p
/// and v=c. This is ABY3's two-way split with a field reduction before masking.
/// Pairwise PRFs mask every message; unrefreshed dot contributions never leave
/// their owner. All parties send their independent split chunk before receiving.
async fn split(session: &mut Session, scores: &[u16]) -> Result<(VecShare<u16>, VecShare<u16>)> {
    ensure!(
        scores.iter().all(|&x| x < MODULUS),
        "noncanonical field score"
    );
    let role = session.own_role().index();
    // Batch the PRG work. Rejection above an exact multiple of p avoids any
    // modulo bias; using u32 candidates makes rejection very rare.
    let (mine, previous_masks) = session.prf.gen_rands_batch::<u32>(scores.len());
    const LIMIT: u64 = (1u64 << 32) / MODULUS as u64 * MODULUS as u64;
    let mut uniform = |mut value: u32, mine: bool| {
        while u64::from(value) >= LIMIT {
            value = if mine {
                session.prf.my_prf.gen()
            } else {
                session.prf.prev_prf.gen()
            };
        }
        (value % u32::from(MODULUS)) as u16
    };
    let masks: Vec<_> = mine
        .into_iter()
        .zip(previous_masks)
        .map(|(a, b)| (uniform(a.0, true), uniform(b.0, false)))
        .collect();
    let own: Vec<_> = scores
        .iter()
        .zip(masks)
        .map(|(&x, (mine, prev))| {
            RingElement(reduce(
                i64::from(x) + i64::from(mine) - i64::from(prev)
                    + if role == 0 { 32000 } else { 0 },
            ))
        })
        .collect();
    session
        .network_session
        .send_next(NetworkValue::VecRing16(own.clone()))
        .await?;
    let NetworkValue::VecRing16(previous) = session.network_session.receive_prev().await? else {
        eyre::bail!("expected refreshed field components");
    };
    ensure!(
        previous.len() == scores.len() && previous.iter().all(|x| x.0 < MODULUS),
        "invalid refreshed field components"
    );
    let n = scores.len();
    let lengths = [n / 3, (n - n / 3) / 2, n - n / 3 - (n - n / 3) / 2];
    let starts = [0, lengths[0], lengths[0] + lengths[1]];
    let mut u = vec![Share::<u16>::zero(); n];
    let mut v = vec![Share::<u16>::zero(); n];
    let mut sent = Vec::with_capacity(lengths[role]);
    for owner in 0..3 {
        for i in starts[owner]..starts[owner] + lengths[owner] {
            if owner == role {
                let mask: u16 = session.prf.prev_prf.gen();
                let value = reduce(i64::from(own[i].0) + i64::from(previous[i].0)) ^ mask;
                sent.push(RingElement(value));
                u[i] = Share::new(RingElement(value), RingElement(mask));
            } else if (owner + 1) % 3 == role {
                v[i].a = own[i];
            } else {
                u[i].a = RingElement(session.prf.my_prf.gen());
                v[i].b = previous[i];
            }
        }
    }
    session
        .network_session
        .send_next(NetworkValue::VecRing16(sent))
        .await?;
    let NetworkValue::VecRing16(received) = session.network_session.receive_prev().await? else {
        eyre::bail!("expected masked binary split");
    };
    let owner = (role + 2) % 3;
    ensure!(
        received.len() == lengths[owner],
        "invalid binary split length"
    );
    for (i, value) in (starts[owner]..starts[owner] + lengths[owner]).zip(received) {
        u[i].b = value;
    }
    Ok((VecShare::new_vec(u), VecShare::new_vec(v)))
}

/// Batch independent packed AND gates into one communication layer.
async fn and_layer(
    session: &mut Session,
    inputs: &[(Share<u64>, Share<u64>)],
) -> Result<Vec<Share<u64>>> {
    let (mine, previous) = session.prf.gen_rands_batch::<u64>(inputs.len());
    let local: Vec<_> = inputs
        .iter()
        .zip(mine)
        .zip(previous)
        .map(|(((a, b), r), s)| (a & b) ^ r ^ s)
        .collect();
    session
        .network_session
        .send_next(NetworkValue::VecRing64(local.clone()))
        .await?;
    let NetworkValue::VecRing64(remote) = session.network_session.receive_prev().await? else {
        eyre::bail!("expected packed field-threshold ANDs");
    };
    ensure!(
        remote.len() == local.len(),
        "invalid field-threshold AND length"
    );
    Ok(local
        .into_iter()
        .zip(remote)
        .map(|(a, b)| Share::new(a, b))
        .collect())
}

/// Return the same NOT-accepted bit as the existing anonymous threshold.
/// Inputs are local contributions to g=2*C-m, where M=2m. For valid irises,
/// Y=g+32000 is in [0,51200]. After splitting, W=u+v is at most 104400:
/// [Y>=32000] = [W>=32000] XOR [W>=52201] XOR [W>=84201].
/// The addition and three public comparisons stream low bits to high bits
/// together: 46 packed ANDs in 17 layers, without arithmetic score conversion.
pub async fn anon_stats_greater_than(
    session: &mut Session,
    scores: &[u16],
) -> Result<Vec<Share<Bit>>> {
    if scores.is_empty() {
        return Ok(Vec::new());
    }
    let (u, v) = split(session, scores).await?;
    let u = u.transpose_pack_u64();
    let v = v.transpose_pack_u64();
    let words = scores.len().div_ceil(64);
    let zero = Share::<u64>::zero();
    let one = Share::from_const(u64::MAX, session.own_role());
    let mut carry = vec![zero; words];
    let mut ge = [vec![zero; words], vec![zero; words], vec![zero; words]];
    let thresholds = [32000, 52201, 84201];
    // The comparisons against p and p+32000 share their low eight bits.
    // The low eight bits of 32000 are zero. Share the prefix and simplify
    // constant gates; at bit 8 the two different updates reuse one product.
    for bit in 0..16 {
        let sum: Vec<_> = (0..words)
            .map(|j| u[bit].get_at(j) ^ v[bit].get_at(j) ^ carry[j])
            .collect();
        let mut inputs = Vec::with_capacity(4 * words);
        inputs
            .extend((0..words).map(|j| (u[bit].get_at(j) ^ carry[j], v[bit].get_at(j) ^ carry[j])));
        match bit {
            0 => {}
            1..=8 => inputs.extend((0..words).map(|j| (sum[j], ge[1][j]))),
            _ => {
                for g in &ge {
                    inputs.extend((0..words).map(|j| (sum[j], g[j])));
                }
            }
        }
        let products = and_layer(session, &inputs).await?;
        for j in 0..words {
            carry[j] ^= products[j];
        }
        if bit == 0 {
            ge[0].fill(one);
            ge[1].clone_from(&sum);
            ge[2].clone_from(&sum);
        } else if bit <= 8 {
            for j in 0..words {
                let product = products[words + j];
                if bit == 8 {
                    ge[0][j] = sum[j];
                    ge[2][j] = sum[j] ^ ge[1][j] ^ product;
                    ge[1][j] = product;
                } else {
                    ge[1][j] = if thresholds[1] >> bit & 1 != 0 {
                        product
                    } else {
                        sum[j] ^ ge[1][j] ^ product
                    };
                    ge[2][j] = ge[1][j];
                }
            }
        } else {
            for (index, (g, threshold)) in ge.iter_mut().zip(thresholds).enumerate() {
                for j in 0..words {
                    let product = products[(index + 1) * words + j];
                    g[j] = if threshold >> bit & 1 != 0 {
                        product
                    } else {
                        sum[j] ^ g[j] ^ product
                    };
                }
            }
        }
    }
    // If W's top bit is set, the two lower thresholds are both true and
    // cancel. Otherwise the upper threshold is false. One mux replaces all
    // three final comparison gates. Count: 16 + 8 + 7*3 + 1 = 46 ANDs.
    let inputs: Vec<_> = (0..words)
        .map(|j| (carry[j], ge[0][j] ^ ge[1][j] ^ ge[2][j]))
        .collect();
    let mux = and_layer(session, &inputs).await?;
    let mut out = VecShare::new_vec(
        (0..words)
            .map(|j| ge[0][j] ^ ge[1][j] ^ mux[j] ^ one)
            .collect(),
    )
    .convert_to_bits();
    out.truncate(scores.len());
    Ok(out.inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ampc_actor_utils::{execution::local::LocalRuntime, protocol::binary::open_bin};
    use futures::future::try_join_all;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    #[tokio::test]
    async fn all_field_values_and_share_wraps() -> Result<()> {
        let mut rng = ChaCha20Rng::seed_from_u64(52201);
        let inputs: Vec<[u16; 3]> = (0..MODULUS)
            .map(|y| {
                let a = rng.gen_range(0..MODULUS);
                let b = rng.gen_range(0..MODULUS);
                [
                    a,
                    b,
                    reduce(i64::from(y) - 32000 - i64::from(a) - i64::from(b)),
                ]
            })
            .collect();
        let runtime = LocalRuntime::mock_setup_with_channel().await?;
        let results = try_join_all(runtime.sessions.into_iter().enumerate().map(
            |(party, mut session)| {
                let input: Vec<_> = inputs.iter().map(|x| x[party]).collect();
                async move {
                    let mut result = Vec::new();
                    // Nonmultiples of 3 and 64 exercise split and packed tails.
                    for batch in input.chunks(4093) {
                        let shares = anon_stats_greater_than(&mut session, batch).await?;
                        result.extend(
                            open_bin(&mut session, &shares)
                                .await?
                                .into_iter()
                                .map(bool::from),
                        );
                    }
                    Ok::<_, eyre::Report>(result)
                }
            },
        ))
        .await?;
        for result in results {
            for (y, rejected) in result.into_iter().enumerate() {
                assert_eq!(rejected, y < 32000, "field representative {y}");
            }
        }
        Ok(())
    }
}
