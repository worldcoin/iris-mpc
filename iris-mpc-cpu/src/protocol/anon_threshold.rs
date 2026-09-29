//! Anonymous-statistics threshold of the exact linear scan, evaluated on the
//! local additive dot-product contributions without a Rep3 refresh.
//!
//! With `c` the code dot product and `m = 2b` the full mask dot product (`b`
//! is the dot product of the trimmed masks), `FHD > 3/8` is equivalent to
//! `4c - m < 0`, which is `2c - b < 0`. Iris codes bound `|c| <= m <= 12800`,
//! so `2c - b` lies in `[-32000, 19200]` and fits a signed 16-bit value: the
//! comparison is the MSB of `2c - b mod 2^16`. Every party computes its term
//! `2c_i - b_i` locally from its Galois-ring contributions, so the three
//! terms are a 3-out-of-3 additive sharing of that value.
//!
//! Instead of refreshing the value into Rep3 and reducing three components
//! with a full-adder layer, the parties convert it to a two-term binary
//! sharing `x = y + z` in one round. Let `h` be the z-holder, `n = h + 1` and
//! `p = h + 2`, and let `r_hn` and `r_hp` come from the PRFs that `h` shares
//! with `n` and with `p`:
//!
//! - `n` sends `x_n + r_hn` to `p` and `p` sends `x_p + r_hp` to `n`. Both then
//!   know `y = x_n + x_p + r_hn + r_hp`. A value two parties know is a single
//!   replicated component, so `y` enters Rep3 for free.
//! - `h` knows `z = x_h - r_hn - r_hp`. It bit-slices `z`, masks it with a pad
//!   `rho` from the PRF shared with `n`, and sends `z ^ rho` to `p`. The
//!   shares of `z` are then `(rho, z ^ rho, 0)`.
//!
//! Each message is masked by randomness its receiver does not know, and every
//! party sends 16 bits per value. The MSB of `y + z` then takes a ripple carry
//! of 15 AND gates: `c_1 = y_0 & z_0`, `c_{k+1} = ((y_k ^ c_k) & (z_k ^ c_k)) ^
//! c_k`, and `msb = y_15 ^ z_15 ^ c_15`. Opening costs one more bit, so the
//! dense comparison sends 32 bits per party and value, against 99 bits for a
//! Rep3 refresh followed by an 18-bit comparison circuit.
//!
//! All values are processed bit-sliced: 64 comparisons per `u64` word, with
//! the 16 bit planes of one value vector stored plane-major in flat buffers.

use crate::execution::session::{Session, SessionHandles};
use ampc_secret_sharing::shares::{
    ring_impl::{RingElement, VecRingElement},
    share::Share,
    vecshare::VecShare,
};
use eyre::{bail, ensure, Result};
use rand::RngCore;
use tracing::instrument;

const BITS: usize = 16;

/// Rep3 binary shares of packed bits: lane `j` of word `w` is value `64 w + j`.
/// Party `i` holds `a = a_i` and `b = a_{i-1}`.
struct PackedBits {
    a: Vec<u64>,
    b: Vec<u64>,
}

fn fill_u16(rng: &mut impl RngCore, out: &mut [u16]) {
    rng.fill_bytes(bytemuck::cast_slice_mut(out));
}

fn fill_u64(rng: &mut impl RngCore, out: &mut [u64]) {
    rng.fill_bytes(bytemuck::cast_slice_mut(out));
}

fn as_words(values: &[RingElement<u64>]) -> &[u64] {
    RingElement::convert_slice(values)
}

fn words_to_ring(words: Vec<u64>) -> VecRingElement<u64> {
    VecRingElement(words.into_iter().map(RingElement).collect())
}

/// Transpose 64 `u16` values into 16 words: bit `k` of `input[j]` becomes bit
/// `j` of `output[k]`.
#[inline]
fn transpose_u16_block(input: &[u16; 64]) -> [u64; BITS] {
    let mut result = [0_u64; BITS];
    for (bit, output) in result.iter_mut().enumerate() {
        *output = u64::from(input[bit])
            | (u64::from(input[16 + bit]) << 16)
            | (u64::from(input[32 + bit]) << 32)
            | (u64::from(input[48 + bit]) << 48);
    }
    let mut mask = 0x00ff_00ff_00ff_00ff_u64;
    let mut shift = 8_u32;
    while shift != 0 {
        let mut index = 0;
        while index < BITS {
            let swap = ((result[index] >> shift) ^ result[index + shift as usize]) & mask;
            result[index + shift as usize] ^= swap;
            result[index] ^= swap << shift;
            index = (index + shift as usize + 1) & !(shift as usize);
        }
        shift >>= 1;
        mask ^= mask << shift;
    }
    result
}

/// [`transpose_u16_block`] with AVX-512BW: the sign bits of 32 words are one
/// mask instruction, and doubling the words moves the next bit into the sign.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
fn transpose_u16_block_avx512(input: &[u16; 64]) -> [u64; BITS] {
    use std::arch::x86_64::{_mm512_add_epi16, _mm512_loadu_si512, _mm512_movepi16_mask};
    // SAFETY: both loads read 32 u16 values inside `input`.
    let (mut low, mut high) = unsafe {
        (
            _mm512_loadu_si512(input.as_ptr().cast()),
            _mm512_loadu_si512(input.as_ptr().add(32).cast()),
        )
    };
    let mut result = [0_u64; BITS];
    for bit in (0..BITS).rev() {
        result[bit] =
            u64::from(_mm512_movepi16_mask(low)) | (u64::from(_mm512_movepi16_mask(high)) << 32);
        low = _mm512_add_epi16(low, low);
        high = _mm512_add_epi16(high, high);
    }
    result
}

/// Bit-slice `values` into `planes`, plane-major: bit `k` of value `64 w + j`
/// becomes bit `j` of `planes[k * words + w]`. The tail block is zero-padded.
fn bit_slice(values: &[u16], planes: &mut [u64], words: usize) {
    debug_assert_eq!(planes.len(), BITS * words);
    debug_assert_eq!(values.len().div_ceil(64), words);
    #[cfg(target_arch = "x86_64")]
    let transpose: fn(&[u16; 64]) -> [u64; BITS] =
        if std::arch::is_x86_feature_detected!("avx512bw") {
            // SAFETY: the CPU supports AVX-512BW.
            |block: &[u16; 64]| unsafe { transpose_u16_block_avx512(block) }
        } else {
            transpose_u16_block
        };
    #[cfg(not(target_arch = "x86_64"))]
    let transpose: fn(&[u16; 64]) -> [u64; BITS] = transpose_u16_block;
    for (word, chunk) in values.chunks(64).enumerate() {
        let block = match <&[u16; 64]>::try_from(chunk) {
            Ok(block) => transpose(block),
            Err(_) => {
                let mut padded = [0_u16; 64];
                padded[..chunk.len()].copy_from_slice(chunk);
                transpose(&padded)
            }
        };
        for (bit, plane_word) in block.into_iter().enumerate() {
            planes[bit * words + word] = plane_word;
        }
    }
}

/// One layer of Rep3 AND gates on packed words: returns this party's share
/// `(local, received)` of `x & w` for `x = (xa, xb)` and `w = (wa, wb)`.
async fn and_packed(
    session: &mut Session,
    xa: impl Fn(usize) -> u64,
    xb: impl Fn(usize) -> u64,
    wa: impl Fn(usize) -> u64,
    wb: impl Fn(usize) -> u64,
    words: usize,
    scratch: &mut [u64],
) -> Result<(Vec<u64>, Vec<u64>)> {
    // Zero sharing of the AND output: mine ^ prev, drawn from the pairwise PRFs.
    fill_u64(session.prf.get_my_prf(), &mut scratch[..words]);
    let mut local = vec![0_u64; words];
    fill_u64(session.prf.get_prev_prf(), &mut local);
    for (index, out) in local.iter_mut().enumerate() {
        let (xa, xb, wa, wb) = (xa(index), xb(index), wa(index), wb(index));
        *out ^= (xa & wa) ^ (xa & wb) ^ (xb & wa) ^ scratch[index];
    }
    let network = &mut session.network_session;
    let local = words_to_ring(local);
    network.send_ring_vec_next(&local).await?;
    let received = network.receive_ring_vec_prev::<u64>().await?;
    if received.len() != words {
        bail!(
            "anonymous-threshold AND received {} words, expected {words}",
            received.len()
        );
    }
    Ok((
        RingElement::convert_vec(local.0),
        RingElement::convert_vec(received.0),
    ))
}

/// Rep3 shares of `MSB(sum of the parties' values mod 2^16)`, where this
/// party holds the additive term `value(i)` of value `i < len`. See the module
/// documentation.
async fn msb_from_additive(
    session: &mut Session,
    len: usize,
    value: impl Fn(usize) -> u16,
) -> Result<PackedBits> {
    // Rotating the z-holder with the session spreads its slightly different
    // load over the parties.
    let z_holder = session.session_id().0 as usize % 3;
    msb_from_additive_with_z_holder(session, len, value, z_holder).await
}

/// Word `index` of bit plane `bit` of a share component; `None` is an all-zero
/// component.
#[inline(always)]
fn plane_word(planes: Option<&[u64]>, words: usize, bit: usize, index: usize) -> u64 {
    planes.map_or(0, |planes| planes[bit * words + index])
}

async fn msb_from_additive_with_z_holder(
    session: &mut Session,
    len: usize,
    value: impl Fn(usize) -> u16,
    z_holder: usize,
) -> Result<PackedBits> {
    let words = len.div_ceil(64);
    if len == 0 {
        return Ok(PackedBits {
            a: Vec::new(),
            b: Vec::new(),
        });
    }
    let role = session.own_role().index();
    ensure!(
        role < 3 && z_holder < 3,
        "the anonymous threshold requires three parties"
    );

    // Plane-major shares of y and z. Each role holds exactly two nonzero
    // components (the others are `None`): h holds z = (rho, z ^ rho); n holds
    // y = (y, 0) and z = (0, rho); p holds y = (0, y) and z = (z ^ rho, 0).
    let plane_words = BITS * words;
    type Components = (
        Option<Vec<u64>>,
        Option<Vec<u64>>,
        Option<Vec<u64>>,
        Option<Vec<u64>>,
    );
    let (ya, yb, za, zb): Components = match (role + 3 - z_holder) % 3 {
        0 => {
            // h: `my` PRF is shared with n, `prev` with p.
            let mut z = vec![0_u16; len];
            fill_u16(session.prf.get_my_prf(), &mut z);
            let mut r_hp = vec![0_u16; len];
            fill_u16(session.prf.get_prev_prf(), &mut r_hp);
            let mut rho = vec![0_u64; plane_words];
            fill_u64(session.prf.get_my_prf(), &mut rho);
            for (index, (z, &r_p)) in z.iter_mut().zip(&r_hp).enumerate() {
                *z = value(index).wrapping_sub(*z).wrapping_sub(r_p);
            }
            drop(r_hp);
            let mut masked = vec![0_u64; plane_words];
            bit_slice(&z, &mut masked, words);
            drop(z);
            for (masked, rho) in masked.iter_mut().zip(&rho) {
                *masked ^= rho;
            }
            let masked = words_to_ring(masked);
            session.network_session.send_ring_vec_prev(&masked).await?;
            (
                None,
                None,
                Some(rho),
                Some(RingElement::convert_vec(masked.0)),
            )
        }
        1 => {
            // n: `prev` is h, `next` is p.
            let mut own = vec![0_u16; len];
            fill_u16(session.prf.get_prev_prf(), &mut own);
            let mut rho = vec![0_u64; plane_words];
            fill_u64(session.prf.get_prev_prf(), &mut rho);
            let own = VecRingElement(
                own.into_iter()
                    .enumerate()
                    .map(|(index, r)| RingElement(value(index).wrapping_add(r)))
                    .collect(),
            );
            session.network_session.send_ring_vec_next(&own).await?;
            let mut other = session
                .network_session
                .receive_ring_vec_next::<u16>()
                .await?;
            if other.len() != len {
                bail!(
                    "anonymous-threshold conversion received {} values, expected {len}",
                    other.len()
                );
            }
            for (other, own) in other.0.iter_mut().zip(&own.0) {
                other.0 = other.0.wrapping_add(own.0);
            }
            let mut planes = vec![0_u64; plane_words];
            bit_slice(RingElement::convert_slice(&other.0), &mut planes, words);
            (Some(planes), None, None, Some(rho))
        }
        _ => {
            // p: `prev` is n, `next` is h.
            let mut own = vec![0_u16; len];
            fill_u16(session.prf.get_my_prf(), &mut own);
            let own = VecRingElement(
                own.into_iter()
                    .enumerate()
                    .map(|(index, r)| RingElement(value(index).wrapping_add(r)))
                    .collect(),
            );
            session.network_session.send_ring_vec_prev(&own).await?;
            let mut other = session
                .network_session
                .receive_ring_vec_prev::<u16>()
                .await?;
            let masked = session
                .network_session
                .receive_ring_vec_next::<u64>()
                .await?;
            if other.len() != len || masked.len() != plane_words {
                bail!("anonymous-threshold conversion received a message of unexpected length");
            }
            for (other, own) in other.0.iter_mut().zip(&own.0) {
                other.0 = other.0.wrapping_add(own.0);
            }
            let mut planes = vec![0_u64; plane_words];
            bit_slice(RingElement::convert_slice(&other.0), &mut planes, words);
            (
                None,
                Some(planes),
                Some(RingElement::convert_vec(masked.0)),
                None,
            )
        }
    };
    let (ya, yb, za, zb) = (ya.as_deref(), yb.as_deref(), za.as_deref(), zb.as_deref());
    let mut scratch = vec![0_u64; words];

    // Bit 0: the carry out is y_0 & z_0.
    let (mut ca, mut cb) = and_packed(
        session,
        |i| plane_word(ya, words, 0, i),
        |i| plane_word(yb, words, 0, i),
        |i| plane_word(za, words, 0, i),
        |i| plane_word(zb, words, 0, i),
        words,
        &mut scratch,
    )
    .await?;
    // Bits 1..15: c_{k+1} = ((y_k ^ c_k) & (z_k ^ c_k)) ^ c_k.
    for bit in 1..BITS - 1 {
        let (ga, gb) = {
            let (ca, cb) = (&ca, &cb);
            and_packed(
                session,
                |i| plane_word(ya, words, bit, i) ^ ca[i],
                |i| plane_word(yb, words, bit, i) ^ cb[i],
                |i| plane_word(za, words, bit, i) ^ ca[i],
                |i| plane_word(zb, words, bit, i) ^ cb[i],
                words,
                &mut scratch,
            )
            .await?
        };
        for (c, g) in ca.iter_mut().zip(&ga) {
            *c ^= g;
        }
        for (c, g) in cb.iter_mut().zip(&gb) {
            *c ^= g;
        }
    }
    let top = BITS - 1;
    for (index, (ca, cb)) in ca.iter_mut().zip(cb.iter_mut()).enumerate() {
        *ca ^= plane_word(ya, words, top, index) ^ plane_word(za, words, top, index);
        *cb ^= plane_word(yb, words, top, index) ^ plane_word(zb, words, top, index);
    }
    Ok(PackedBits { a: ca, b: cb })
}

/// Rep3 shares of the anonymous-statistics comparison `FHD > 3/8` for every
/// `(code, trimmed mask)` pair in `interleaved_dots`, which holds this
/// party's additive Galois-ring contributions. The trimmed mask dot is the
/// value *before* the usual doubling to the full mask scale.
///
/// Returns packed shares: a `1` bit means the distance is above the
/// threshold (not an anonymous-statistics match). Lane `j` of word `w` is pair
/// `64 w + j`; lanes past the input length are padding.
#[instrument(level = "trace", target = "searcher::network", skip_all)]
pub async fn fhd_greater_than_anon_stats_from_trimmed_additive(
    session: &mut Session,
    interleaved_dots: &[RingElement<u16>],
) -> Result<VecShare<u64>> {
    let bits = anon_stats_threshold_bits(session, interleaved_dots).await?;
    Ok(VecShare::new_vec(
        bits.a
            .into_iter()
            .zip(bits.b)
            .map(|(a, b)| Share::new(RingElement(a), RingElement(b)))
            .collect(),
    ))
}

async fn anon_stats_threshold_bits(
    session: &mut Session,
    interleaved_dots: &[RingElement<u16>],
) -> Result<PackedBits> {
    ensure!(
        interleaved_dots.len().is_multiple_of(2),
        "anonymous-threshold input must contain interleaved code/mask pairs"
    );
    // `2c - b` of pair `i`, computed as it is consumed.
    let dots = RingElement::convert_slice(interleaved_dots);
    msb_from_additive(session, dots.len() / 2, |index| {
        dots[2 * index]
            .wrapping_mul(2)
            .wrapping_sub(dots[2 * index + 1])
    })
    .await
}

/// Evaluate the anonymous-statistics comparison on local additive
/// contributions and open it; see
/// [`fhd_greater_than_anon_stats_from_trimmed_additive`]. Returns the opened
/// words, where a `0` bit marks an anonymous-statistics match.
#[instrument(level = "trace", target = "searcher::network", skip_all)]
pub async fn open_anon_stats_threshold_from_trimmed_additive(
    session: &mut Session,
    interleaved_dots: &[RingElement<u16>],
) -> Result<Vec<u64>> {
    let bits = anon_stats_threshold_bits(session, interleaved_dots).await?;
    open_packed_bits(session, bits).await
}

async fn open_packed_bits(session: &mut Session, bits: PackedBits) -> Result<Vec<u64>> {
    let PackedBits { a, b } = bits;
    if a.is_empty() {
        return Ok(Vec::new());
    }
    let words = a.len();
    let network = &mut session.network_session;
    let b = words_to_ring(b);
    network.send_ring_vec_next(&b).await?;
    let previous = network.receive_ring_vec_prev::<u64>().await?;
    if previous.len() != words {
        bail!(
            "packed opening received {} words, expected {words}",
            previous.len()
        );
    }
    Ok(a.into_iter()
        .zip(as_words(&b.0))
        .zip(as_words(&previous.0))
        .map(|((a, b), previous)| a ^ b ^ previous)
        .collect())
}

/// Open packed binary Rep3 shares, one round and one bit per lane. Returns the
/// opened words, with lane `j` of word `w` holding bit `64 w + j`.
#[instrument(level = "trace", target = "searcher::network", skip_all)]
pub async fn open_bin_packed(session: &mut Session, shares: &VecShare<u64>) -> Result<Vec<u64>> {
    let (a, b): (Vec<u64>, Vec<u64>) = shares.iter().map(|share| (share.a.0, share.b.0)).unzip();
    open_packed_bits(session, PackedBits { a, b }).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::local::LocalRuntime;
    use aes_prng::AesRng;
    use ampc_actor_utils::protocol::test_utils::create_array_sharing;
    use rand::{Rng, SeedableRng};
    use tokio::task::JoinSet;

    #[test]
    fn bit_slice_matches_naive_transpose() {
        let mut rng = AesRng::seed_from_u64(0x5eed);
        for len in [1_usize, 63, 64, 65, 200] {
            let values = (0..len).map(|_| rng.gen::<u16>()).collect::<Vec<_>>();
            let words = len.div_ceil(64);
            let mut planes = vec![0_u64; BITS * words];
            bit_slice(&values, &mut planes, words);
            for (index, value) in values.iter().enumerate() {
                for bit in 0..BITS {
                    let got = (planes[bit * words + index / 64] >> (index % 64)) & 1;
                    assert_eq!(got, u64::from((value >> bit) & 1));
                }
            }
            // Padding lanes are zero.
            for bit in 0..BITS {
                for lane in len..words * 64 {
                    assert_eq!((planes[bit * words + lane / 64] >> (lane % 64)) & 1, 0);
                }
            }
        }
    }

    /// Plaintext `(code, trimmed mask)` pairs inside the iris bounds, with the
    /// boundary `b = 2c` and the extreme ranges.
    fn sample_pairs(rng: &mut AesRng, count: usize) -> Vec<(i64, i64)> {
        let mut pairs: Vec<(i64, i64)> = vec![
            (0, 0),
            (100, 200),
            (100, 201),
            (100, 199),
            (-12800, 6400),
            (12800, 6400),
            (-1, 1),
            (2, 3),
            (-6, 3),
        ];
        while pairs.len() < count {
            let b = rng.gen_range(0..=6400_i64);
            let c = rng.gen_range(-2 * b..=2 * b);
            pairs.push((c, b));
        }
        pairs
    }

    async fn open_threshold(interleaved: Vec<RingElement<u16>>, sessions_seed: usize) -> Vec<u64> {
        let sessions = LocalRuntime::mock_sessions_with_channel().await.unwrap();
        let shares = {
            let mut rng = AesRng::seed_from_u64(sessions_seed as u64);
            let plain = RingElement::convert_vec(interleaved);
            create_array_sharing(&mut rng, &plain)
        };
        let mut jobs = JoinSet::new();
        for (party, session) in sessions.into_iter().enumerate() {
            let local = shares
                .of_party(party)
                .iter()
                .map(|share| share.a)
                .collect::<Vec<_>>();
            jobs.spawn(async move {
                let mut session = session.lock().await;
                let packed =
                    fhd_greater_than_anon_stats_from_trimmed_additive(&mut session, &local)
                        .await
                        .unwrap();
                let opened = open_bin_packed(&mut session, &packed).await.unwrap();
                let fused = open_anon_stats_threshold_from_trimmed_additive(&mut session, &local)
                    .await
                    .unwrap();
                assert_eq!(opened, fused);
                opened
            });
        }
        let results = jobs.join_all().await;
        assert_eq!(results[0], results[1]);
        assert_eq!(results[1], results[2]);
        results.into_iter().next().unwrap()
    }

    #[tokio::test]
    async fn anon_threshold_matches_plaintext() {
        let mut rng = AesRng::seed_from_u64(0x2c_b375);
        for count in [9, 64, 3_000] {
            let pairs = sample_pairs(&mut rng, count);
            let interleaved = pairs
                .iter()
                .flat_map(|&(c, b)| [RingElement(c as u16), RingElement(b as u16)])
                .collect::<Vec<_>>();
            let words = open_threshold(interleaved, count).await;
            assert_eq!(words.len(), count.div_ceil(64));
            for (index, &(c, b)) in pairs.iter().enumerate() {
                let greater = (words[index / 64] >> (index % 64)) & 1 == 1;
                assert_eq!(greater, b > 2 * c, "pair {:?}", (c, b));
            }
        }
    }

    /// The z-holder rotates with the session id; every role assignment must
    /// agree with the plaintext.
    #[tokio::test]
    async fn anon_threshold_is_correct_for_every_z_holder() {
        let mut rng = AesRng::seed_from_u64(7);
        let pairs = sample_pairs(&mut rng, 130);
        let interleaved = pairs
            .iter()
            .flat_map(|&(c, b)| [RingElement(c as u16), RingElement(b as u16)])
            .collect::<Vec<_>>();
        let sessions = LocalRuntime::mock_sessions_with_channel().await.unwrap();
        let shares = create_array_sharing(&mut rng, &RingElement::convert_vec(interleaved));
        let mut jobs = JoinSet::new();
        for (party, session) in sessions.into_iter().enumerate() {
            let local = shares
                .of_party(party)
                .iter()
                .map(|share| share.a)
                .collect::<Vec<_>>();
            jobs.spawn(async move {
                let mut session = session.lock().await;
                let mut opened = Vec::new();
                for z_holder in 0..3 {
                    let expression = RingElement::convert_slice(&local)
                        .chunks_exact(2)
                        .map(|pair| pair[0].wrapping_mul(2).wrapping_sub(pair[1]))
                        .collect::<Vec<_>>();
                    let bits = msb_from_additive_with_z_holder(
                        &mut session,
                        expression.len(),
                        |index| expression[index],
                        z_holder,
                    )
                    .await
                    .unwrap();
                    opened.push(open_packed_bits(&mut session, bits).await.unwrap());
                }
                opened
            });
        }
        let results = jobs.join_all().await;
        assert_eq!(results[0], results[1]);
        assert_eq!(results[1], results[2]);
        for words in &results[0] {
            for (index, &(c, b)) in pairs.iter().enumerate() {
                let greater = (words[index / 64] >> (index % 64)) & 1 == 1;
                assert_eq!(greater, b > 2 * c);
            }
        }
    }

    #[tokio::test]
    async fn anon_threshold_accepts_empty_input() {
        assert!(open_threshold(Vec::new(), 1).await.is_empty());
    }
}
