//! Exact anonymous-statistics threshold directly from local F_52201 scores.
//! Only the final predicate is opened by the caller. No edaBits are consumed.
use super::{reduce, MODULUS};
use ampc_actor_utils::{
    execution::session::{Session, SessionHandles},
    network::mpc::NetworkValue,
};
use ampc_secret_sharing::shares::{
    bit::Bit,
    ring_impl::{RingElement, VecRingElement},
    share::Share,
    vecshare_bittranspose::Transpose64,
    VecShare,
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
    let own: VecRingElement<u16> = scores
        .iter()
        .zip(masks)
        .map(|(&x, (mine, prev))| {
            RingElement(reduce(
                i64::from(x) + i64::from(mine) - i64::from(prev)
                    + if role == 0 { 32768 } else { 0 },
            ))
        })
        .collect::<Vec<_>>()
        .into();
    session.network_session.send_ring_vec_next(&own).await?;
    let previous = session
        .network_session
        .receive_ring_vec_prev::<u16>()
        .await?;
    ensure!(
        previous.len() == scores.len() && previous.0.iter().all(|x| x.0 < MODULUS),
        "invalid refreshed field components"
    );
    let n = scores.len();
    let lengths = [n / 3, (n - n / 3) / 2, n - n / 3 - (n - n / 3) / 2];
    let starts = [0, lengths[0], lengths[0] + lengths[1]];
    let mut u = vec![Share::<u16>::zero(); n];
    let mut v = vec![Share::<u16>::zero(); n];
    let mut sent = Vec::with_capacity(lengths[role]);
    // Draw the XOR pads in bulk, using both bytes of each u16. The adjacent
    // party draws the same length from the same pairwise stream, including
    // uneven and empty split chunks.
    let mine = session.prf.gen_rands_mine::<u16>(lengths[(role + 1) % 3]);
    let previous_masks = session.prf.gen_rands_prev::<u16>(lengths[role]);
    for owner in 0..3 {
        for i in starts[owner]..starts[owner] + lengths[owner] {
            if owner == role {
                let mask = previous_masks.0[i - starts[owner]].0;
                let value = reduce(i64::from(own.0[i].0) + i64::from(previous.0[i].0)) ^ mask;
                sent.push(RingElement(value));
                u[i] = Share::new(RingElement(value), RingElement(mask));
            } else if (owner + 1) % 3 == role {
                v[i].a = own.0[i];
            } else {
                u[i].a = mine.0[i - starts[owner]];
                v[i].b = previous.0[i];
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
    let local: VecRingElement<u64> = inputs
        .iter()
        .zip(mine)
        .zip(previous)
        .map(|(((a, b), r), s)| (a & b) ^ r ^ s)
        .collect::<Vec<_>>()
        .into();
    session.network_session.send_ring_vec_next(&local).await?;
    let remote = session
        .network_session
        .receive_ring_vec_prev::<u64>()
        .await?;
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
/// g is in [-32000,19200], so Y=g+32768 is in [768,51968] and fits F_52201.
/// Accept exactly when bit 15 of the canonical Y is set. After splitting,
/// W=u+v is at most 104400. Let B=bit15(W), C=bit16(W), and
/// L=[low15(W)>=19433], where 19433=52201-32768. Then
/// [Y>=32768] = B XOR (L AND (B XOR C)). The two possible wrap comparisons
/// share all 15 low bits. This uses 16 adder ANDs, 14 comparison ANDs, and
/// one final mux: 31 packed ANDs in 17 layers, without score conversion.
async fn anon_stats_greater_than_packed(
    session: &mut Session,
    scores: &[u16],
) -> Result<Vec<Share<u64>>> {
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
    let mut ge = vec![zero; words];
    let mut high = vec![zero; words];
    const LOW_THRESHOLD: u16 = MODULUS - 32768;
    for bit in 0..16 {
        let sum: Vec<_> = (0..words)
            .map(|j| u[bit].get_at(j) ^ v[bit].get_at(j) ^ carry[j])
            .collect();
        let mut inputs = Vec::with_capacity(2 * words);
        inputs
            .extend((0..words).map(|j| (u[bit].get_at(j) ^ carry[j], v[bit].get_at(j) ^ carry[j])));
        if (1..15).contains(&bit) {
            inputs.extend((0..words).map(|j| (sum[j], ge[j])));
        }
        let products = and_layer(session, &inputs).await?;
        for j in 0..words {
            carry[j] ^= products[j];
        }
        match bit {
            // The low bit of 19433 is one, so its initial comparison is free.
            0 => ge = sum,
            15 => high = sum,
            _ => {
                for j in 0..words {
                    let product = products[words + j];
                    ge[j] = if LOW_THRESHOLD >> bit & 1 != 0 {
                        product
                    } else {
                        sum[j] ^ ge[j] ^ product
                    };
                }
            }
        }
    }
    let inputs: Vec<_> = (0..words).map(|j| (ge[j], high[j] ^ carry[j])).collect();
    let mux = and_layer(session, &inputs).await?;
    Ok((0..words).map(|j| high[j] ^ mux[j] ^ one).collect())
}

/// Keep a secret-shared result for callers that need further binary MPC.
pub async fn anon_stats_greater_than(
    session: &mut Session,
    scores: &[u16],
) -> Result<Vec<Share<Bit>>> {
    let packed = anon_stats_greater_than_packed(session, scores).await?;
    let mut bits = VecShare::new_vec(packed).convert_to_bits();
    bits.truncate(scores.len());
    Ok(bits.inner())
}

/// Open only the existing anonymous predicate, directly from its packed shares.
/// Avoid expanding shares to individual bits only to repack them for the wire.
/// Padding bits are zeroed and excluded from the public output.
pub async fn open_anon_stats_matches(session: &mut Session, scores: &[u16]) -> Result<Vec<bool>> {
    if scores.is_empty() {
        return Ok(Vec::new());
    }
    let shares = anon_stats_greater_than_packed(session, scores).await?;
    let byte_count = scores.len().div_ceil(8);
    let mut packed = Vec::with_capacity(shares.len() * 8);
    for share in &shares {
        packed.extend_from_slice(&share.b.0.to_le_bytes());
    }
    packed.truncate(byte_count);
    if !scores.len().is_multiple_of(8) {
        *packed.last_mut().unwrap() &= (1 << (scores.len() % 8)) - 1;
    }
    session
        .network_session
        .send_next(NetworkValue::VecRingBit(packed, scores.len()))
        .await?;
    let NetworkValue::VecRingBit(remote, bit_count) =
        session.network_session.receive_prev().await?
    else {
        eyre::bail!("expected packed anonymous predicate bits");
    };
    ensure!(
        bit_count == scores.len() && remote.len() == byte_count,
        "anonymous predicate bit count mismatch"
    );
    let mut result = Vec::with_capacity(bit_count);
    for (share, bytes) in shares.into_iter().zip(remote.chunks(8)) {
        let local = (share.a ^ share.b).0.to_le_bytes();
        for (a, b) in local.into_iter().zip(bytes) {
            let accepted = !(a ^ b);
            for bit in 0..(bit_count - result.len()).min(8) {
                result.push(accepted >> bit & 1 != 0);
            }
        }
    }
    Ok(result)
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
                    reduce(i64::from(y) - 32768 - i64::from(a) - i64::from(b)),
                ]
            })
            .collect();
        let runtime = LocalRuntime::mock_setup_with_channel().await?;
        let results = try_join_all(runtime.sessions.into_iter().enumerate().map(
            |(party, mut session)| {
                let input: Vec<_> = inputs.iter().map(|x| x[party]).collect();
                async move {
                    let mut result = Vec::new();
                    // Empty and tiny split chunks, then nonmultiples of 3 and
                    // 64, exercise pairwise PRF alignment and packed tails.
                    let ranges = [0..0, 0..1, 1..3, 3..6].into_iter().chain(
                        (6..input.len())
                            .step_by(4093)
                            .map(|start| start..(start + 4093).min(input.len())),
                    );
                    for range in ranges {
                        let shares =
                            anon_stats_greater_than(&mut session, &input[range.clone()]).await?;
                        let opened: Vec<_> = open_bin(&mut session, &shares)
                            .await?
                            .into_iter()
                            .map(bool::from)
                            .collect();
                        let direct =
                            open_anon_stats_matches(&mut session, &input[range.clone()]).await?;
                        assert!(opened
                            .iter()
                            .zip(&direct)
                            .all(|(&rejected, &accepted)| rejected != accepted));
                        assert_eq!(opened.len(), direct.len());
                        result.extend(opened);
                    }
                    Ok::<_, eyre::Report>(result)
                }
            },
        ))
        .await?;
        for result in results {
            for (y, rejected) in result.into_iter().enumerate() {
                assert_eq!(rejected, y < 32768, "field representative {y}");
            }
        }
        Ok(())
    }
}
