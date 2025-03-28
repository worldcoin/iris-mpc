use crate::{
    execution::session::{Session, SessionHandles},
    network::value::NetworkValue,
    shares::{
        bit::Bit,
        int_ring::IntRing2k,
        ring_impl::RingElement,
        share::Share,
        vecshare::{SliceShare, VecShare},
    },
};
use eyre::{eyre, Error};
use itertools::{izip, Itertools};
use num_traits::{One, Zero};
use rand::{distributions::Standard, prelude::Distribution, Rng};
use std::ops::SubAssign;
use tracing::{instrument, trace, trace_span, Instrument};

pub(crate) fn transposed_padded_len(len: usize) -> usize {
    let padded_len = (len + 63) / 64;
    padded_len * 64
}

pub(crate) fn a2b_pre<T: IntRing2k>(
    session: &Session,
    x: Share<T>,
) -> eyre::Result<(Share<T>, Share<T>, Share<T>)> {
    let (a, b) = x.get_ab();

    let mut x1 = Share::zero();
    let mut x2 = Share::zero();
    let mut x3 = Share::zero();

    match session.own_role()?.index() {
        0 => {
            x1.a = a;
            x3.b = b;
        }
        1 => {
            x2.a = a;
            x1.b = b;
        }
        2 => {
            x3.a = a;
            x2.b = b;
        }
        _ => {
            return Err(eyre!(
                "Cannot deal with roles that have index outside of the set [0, 1, 2]"
            ))
        }
    }
    Ok((x1, x2, x3))
}

pub(crate) fn transposed_pack_xor_assign<T: IntRing2k>(x1: &mut [VecShare<T>], x2: &[VecShare<T>]) {
    let len = x1.len();
    debug_assert_eq!(len, x2.len());

    for (x1, x2) in x1.iter_mut().zip(x2.iter()) {
        *x1 ^= x2.as_slice();
    }
}

pub(crate) fn transposed_pack_xor<T: IntRing2k>(
    x1: &[VecShare<T>],
    x2: &[VecShare<T>],
) -> Vec<VecShare<T>> {
    let len = x1.len();
    debug_assert_eq!(len, x2.len());

    let mut res = Vec::with_capacity(len);
    for (x1, x2) in x1.iter().zip(x2.iter()) {
        res.push(x1.as_slice() ^ x2.as_slice());
    }
    res
}

pub(crate) async fn and_many_send(
    session: &mut Session,
    a: SliceShare<'_, u64>,
    b: SliceShare<'_, u64>,
) -> Result<Vec<RingElement<u64>>, Error>
where
    Standard: Distribution<u64>,
{
    if a.len() != b.len() {
        return Err(eyre!("InvalidSize in and_many_send"));
    }
    let mut shares_a = Vec::with_capacity(a.len());
    for (a_, b_) in a.iter().zip(b.iter()) {
        let rand = session.prf.gen_binary_zero_share::<u64>();
        let mut c = a_ & b_;
        c ^= rand;
        shares_a.push(c);
    }

    let network = &mut session.network_session;
    let message = shares_a.clone();
    let message = if message.len() == 1 {
        NetworkValue::RingElement64(message[0])
    } else {
        NetworkValue::VecRing64(message)
    };
    network.send_next(message.to_network()).await?;
    Ok(shares_a)
}

pub(crate) async fn and_many_receive(
    session: &mut Session,
) -> Result<Vec<RingElement<u64>>, Error> {
    let shares_b = {
        let serialized_other_share = session.network_session.receive_prev().await;
        match NetworkValue::from_network(serialized_other_share) {
            Ok(NetworkValue::RingElement64(message)) => Ok(vec![message]),
            Ok(NetworkValue::VecRing64(message)) => Ok(message),
            _ => Err(eyre!("Error in receiving in and_many operation")),
        }
    }?;
    Ok(shares_b)
}

pub(crate) async fn and_many(
    session: &mut Session,
    a: SliceShare<'_, u64>,
    b: SliceShare<'_, u64>,
) -> Result<VecShare<u64>, Error>
where
    Standard: Distribution<u64>,
{
    let shares_a = and_many_send(session, a, b).await?;
    let shares_b = and_many_receive(session).await?;
    let complete_shares = VecShare::from_ab(shares_a, shares_b);
    Ok(complete_shares)
}

#[instrument(level = "trace", target = "searcher::network", skip(session, x1, x2))]
pub(crate) async fn transposed_pack_and(
    session: &mut Session,
    x1: Vec<VecShare<u64>>,
    x2: Vec<VecShare<u64>>,
) -> Result<Vec<VecShare<u64>>, Error> {
    if x1.len() != x2.len() {
        return Err(eyre!(
            "Inputs have different length {} {}",
            x1.len(),
            x2.len()
        ));
    }
    let chunk_sizes = x1.iter().map(VecShare::len).collect::<Vec<_>>();
    let chunk_sizes2 = x2.iter().map(VecShare::len).collect::<Vec<_>>();
    if chunk_sizes != chunk_sizes2 {
        return Err(eyre!("VecShare lengths are not equal"));
    }

    let x1 = VecShare::flatten(x1);
    let x2 = VecShare::flatten(x2);
    let mut shares_a = and_many_send(session, x1.as_slice(), x2.as_slice()).await?;
    let mut shares_b = and_many_receive(session).await?;

    let mut res = Vec::with_capacity(chunk_sizes.len());
    for l in chunk_sizes {
        let a = shares_a.drain(..l).collect();
        let b = shares_b.drain(..l).collect();
        res.push(VecShare::from_ab(a, b));
    }
    Ok(res)
}

#[instrument(level = "trace", target = "searcher::network", skip_all)]
async fn binary_add_3_get_two_carries(
    session: &mut Session,
    x1: Vec<VecShare<u64>>,
    x2: Vec<VecShare<u64>>,
    x3: Vec<VecShare<u64>>,
    truncate_len: usize,
) -> Result<(VecShare<Bit>, VecShare<Bit>), Error>
where
    Standard: Distribution<u64>,
{
    let len = x1.len();
    debug_assert!(len == x2.len() && len == x3.len());

    // Full adder to get 2 * c and s
    let mut x2x3 = x2;
    transposed_pack_xor_assign(&mut x2x3, &x3);
    let s = transposed_pack_xor(&x1, &x2x3);
    let mut x1x3 = x1;
    transposed_pack_xor_assign(&mut x1x3, &x3);
    let mut c = transposed_pack_and(session, x1x3, x2x3).await?;
    transposed_pack_xor_assign(&mut c, &x3);

    // Add 2c + s via a ripple carry adder
    // LSB of c is 0
    // First round: half adder can be skipped due to LSB of c being 0
    let mut a = s;
    let mut b = c;

    let and_many_span = trace_span!(target: "searcher::network", "and_many_calls", n = b.len());

    // First full adder (carry is 0)
    let mut c = and_many(session, a[1].as_slice(), b[0].as_slice())
        .instrument(and_many_span.clone())
        .await?;

    // For last round
    let mut b_msb = b.pop().expect("Enough elements present");

    // 2 -> k
    for (a_, b_) in a.iter_mut().skip(2).zip(b.iter_mut().skip(1)) {
        *a_ ^= c.as_slice();
        *b_ ^= c.as_slice();
        let tmp_c = and_many(session, a_.as_slice(), b_.as_slice())
            .instrument(and_many_span.clone())
            .await?;
        c ^= tmp_c;
    }

    // Finally, last bit of a is 0
    let res2 = and_many(session, b_msb.as_slice(), c.as_slice())
        .instrument(and_many_span)
        .await?;
    b_msb ^= c;

    // Extract bits for outputs
    let mut res1 = b_msb.convert_to_bits();
    res1.truncate(truncate_len);
    let mut res2 = res2.convert_to_bits();
    res2.truncate(truncate_len);

    Ok((res1, res2))
}

#[instrument(level = "trace", target = "searcher::network", skip_all)]
async fn bit_inject_ot_2round_helper(
    session: &mut Session,
    input: VecShare<Bit>,
) -> Result<VecShare<u16>, Error>
where
    Standard: Distribution<u16>,
{
    let len = input.len();
    let mut wc = Vec::with_capacity(len);
    let mut shares = VecShare::with_capacity(len);
    let prf = &mut session.prf;

    for inp in input.into_iter() {
        // new share
        let c3 = prf.get_prev_prf().gen::<RingElement<u16>>();
        shares.push(Share::new(RingElement::zero(), c3));

        // mask of the ot
        let w0 = prf.get_prev_prf().gen::<RingElement<u16>>();
        let w1 = prf.get_prev_prf().gen::<RingElement<u16>>();

        let choice = inp.get_a().convert().convert();
        if choice {
            wc.push(w1);
        } else {
            wc.push(w0);
        }
    }
    let next_id = session.next_identity()?;
    let network = &mut session.network_session;
    trace!(target: "searcher::network", action = "send", party = ?next_id, bytes = 0, rounds = 1);
    network
        .send_next(NetworkValue::VecRing16(wc).to_network())
        .await?;

    let c1 = {
        let reply = network.receive_next().await;
        match NetworkValue::from_network(reply) {
            Ok(NetworkValue::VecRing16(val)) => Ok(val),
            _ => Err(eyre!("Could not deserialize properly in bit inject")),
        }
    }?;

    // Receive Reshare
    for (s, c1) in shares.iter_mut().zip(c1) {
        s.a = c1;
    }
    Ok(shares)
}

#[instrument(level = "trace", target = "searcher::network", skip_all)]
async fn bit_inject_ot_2round_receiver(
    session: &mut Session,
    input: VecShare<Bit>,
) -> Result<VecShare<u16>, Error> {
    let prev_id = session.prev_identity()?;
    let network = &mut session.network_session;

    let (m0, m1, wc) = {
        let reply_m0_and_m1 = network.receive_next().await;
        let m0_and_m1 = NetworkValue::vec_from_network(reply_m0_and_m1).unwrap();
        assert!(
            m0_and_m1.len() == 2,
            "Deserialized vec in bit inject is wrong length"
        );
        let (m0, m1) = m0_and_m1.into_iter().collect_tuple().unwrap();

        let m0 = match m0 {
            NetworkValue::VecRing16(val) => Ok(val),
            _ => Err(eyre!("Could not deserialize properly in bit inject")),
        };

        let m1 = match m1 {
            NetworkValue::VecRing16(val) => Ok(val),
            _ => Err(eyre!("Could not deserialize properly in bit inject")),
        };

        let reply_wc = network.receive_prev().await;
        let wc = match NetworkValue::from_network(reply_wc) {
            Ok(NetworkValue::VecRing16(val)) => Ok(val),
            _ => Err(eyre!("Could not deserialize properly in bit inject")),
        };
        (m0, m1, wc)
    };

    let (m0, m1, wc) = (m0?, m1?, wc?);

    let len = input.len();
    let mut shares = VecShare::with_capacity(len);
    let mut send = Vec::with_capacity(len);

    for ((inp, wc), (m0, m1)) in input
        .into_iter()
        .zip(wc.into_iter())
        .zip(m0.into_iter().zip(m1.into_iter()))
    {
        // new share
        let c2 = session.prf.get_my_prf().gen::<RingElement<u16>>();

        let choice = inp.get_b().convert().convert();
        let xor = if choice { wc ^ m1 } else { wc ^ m0 };

        send.push(xor);
        shares.push(Share::new(c2, xor));
    }

    // Reshare to Helper
    trace!(target: "searcher::network", action = "send", party = ?prev_id, bytes = 0, rounds = 1);
    network
        .send_prev(NetworkValue::VecRing16(send).to_network())
        .await?;

    Ok(shares)
}

#[instrument(level = "trace", target = "searcher::network", skip_all)]
async fn bit_inject_ot_2round_sender(
    session: &mut Session,
    input: VecShare<Bit>,
) -> Result<VecShare<u16>, Error> {
    let len = input.len();
    let mut m0 = Vec::with_capacity(len);
    let mut m1 = Vec::with_capacity(len);
    let mut shares = VecShare::with_capacity(len);
    let prf = &mut session.prf;

    for inp in input.into_iter() {
        let (a, b) = inp.get_ab();
        // new shares
        let (c3, c2) = prf.gen_rands::<RingElement<u16>>();
        // mask of the ot
        let w0 = prf.get_my_prf().gen::<RingElement<u16>>();
        let w1 = prf.get_my_prf().gen::<RingElement<u16>>();

        shares.push(Share::new(c3, c2));
        let c = c3 + c2;
        let xor = RingElement(u16::from((a ^ b).convert().convert()));
        let m0_ = xor - c;
        let m1_ = (xor ^ RingElement::one()) - c;
        m0.push(m0_ ^ w0);
        m1.push(m1_ ^ w1);
    }

    let prev_id = session.prev_identity()?;
    let m0_and_m1: Vec<NetworkValue> = [m0, m1]
        .into_iter()
        .map(NetworkValue::VecRing16)
        .collect::<Vec<_>>();
    trace!(target: "searcher::network", action = "send", party = ?prev_id, bytes = 0, rounds = 1);
    // Reshare to Helper
    session
        .network_session
        .send_prev(NetworkValue::vec_to_network(&m0_and_m1))
        .await?;
    Ok(shares)
}

// TODO this is unbalanced, so a real implementation should actually rotate
// parties around
pub(crate) async fn bit_inject_ot_2round(
    session: &mut Session,
    input: VecShare<Bit>,
) -> Result<VecShare<u16>, Error> {
    let res = match session.own_role()?.index() {
        0 => {
            // OT Helper
            bit_inject_ot_2round_helper(session, input).await?
        }
        1 => {
            // OT Receiver
            bit_inject_ot_2round_receiver(session, input).await?
        }
        2 => {
            // OT Sender
            bit_inject_ot_2round_sender(session, input).await?
        }
        _ => {
            return Err(eyre!(
                "Cannot deal with roles outside of the set [0, 1, 2] in bit_inject_ot"
            ))
        }
    };
    Ok(res)
}

pub(crate) fn mul_lift_2k<const K: u64>(val: &Share<u16>) -> Share<u32>
where
    u32: From<u16>,
{
    let a = (u32::from(val.a.0)) << K;
    let b = (u32::from(val.b.0)) << K;
    Share::new(RingElement(a), RingElement(b))
}

pub(crate) fn mul_lift_2k_many<const K: u64>(vals: SliceShare<u16>) -> VecShare<u32> {
    VecShare::new_vec(vals.iter().map(mul_lift_2k::<K>).collect())
}

pub(crate) async fn lift<const K: usize>(
    session: &mut Session,
    shares: VecShare<u16>,
) -> eyre::Result<VecShare<u32>> {
    let len = shares.len();
    let padded_len = transposed_padded_len(len);

    let mut x_a = VecShare::with_capacity(padded_len);
    for share in shares.iter() {
        x_a.push(Share::new(
            RingElement(share.a.0 as u32),
            RingElement(share.b.0 as u32),
        ));
    }

    let x = shares.transpose_pack_u64();

    let len_ = x.len();
    let mut x1 = Vec::with_capacity(len_);
    let mut x2 = Vec::with_capacity(len_);
    let mut x3 = Vec::with_capacity(len_);

    for x_ in x.into_iter() {
        let len__ = x_.len();
        let mut x1_ = VecShare::with_capacity(len__);
        let mut x2_ = VecShare::with_capacity(len__);
        let mut x3_ = VecShare::with_capacity(len__);
        for x__ in x_.into_iter() {
            let (x1__, x2__, x3__) = a2b_pre(session, x__)?;
            x1_.push(x1__);
            x2_.push(x2__);
            x3_.push(x3__);
        }
        x1.push(x1_);
        x2.push(x2_);
        x3.push(x3_);
    }

    let (mut b1, b2) = binary_add_3_get_two_carries(session, x1, x2, x3, len).await?;
    b1.extend(b2);

    // We need bit1 * 2^{ShareRing::K+1} mod 2^{ShareRing::K+K}  and bit2 *
    // 2^{ShareRing::K+1} mod 2^{ShareRing::K+K} So we inject bit to mod
    // 2^{K-1} and mod 2^{K-2} and use the mul_lift_2k function TODO: This
    // one is not optimized: We send too much, since we need less than K
    // bits
    debug_assert!(K <= 16); // otherwise u16 does not work
    let mut b = bit_inject_ot_2round(session, b1).await?;
    let (b1, b2) = b.split_at_mut(len);

    // Make the result mod 2^{K-1} and mod 2^{K-2} (Not required since we bitextract
    // the correct one later) Self::share_bit_mod(&mut b1, K as u32);
    // Self::share_bit_mod(&mut b2, K as u32 - 1);

    let b1 = mul_lift_2k_many::<{ u16::K as u64 }>(b1.to_slice());
    let b2 = mul_lift_2k_many::<{ u16::K as u64 + 1 }>(b2.to_slice());

    // Finally, compute the result
    x_a.sub_assign(b1);
    x_a.sub_assign(b2);
    Ok(x_a)
}

// MSB related code
#[allow(dead_code)]
pub(crate) async fn binary_add_3_get_msb(
    session: &mut Session,
    x1: Vec<VecShare<u64>>,
    x2: Vec<VecShare<u64>>,
    mut x3: Vec<VecShare<u64>>,
) -> Result<VecShare<u64>, Error> {
    let len = x1.len();
    debug_assert!(len == x2.len() && len == x3.len());

    // Full adder to get 2 * c and s
    let mut x2x3 = x2;
    transposed_pack_xor_assign(&mut x2x3, &x3);
    let s = transposed_pack_xor(&x1, &x2x3);
    let mut x1x3 = x1;
    transposed_pack_xor_assign(&mut x1x3, &x3);
    // 2 * c
    x1x3.pop().expect("Enough elements present");
    x2x3.pop().expect("Enough elements present");
    x3.pop().expect("Enough elements present");
    let mut c = transposed_pack_and(session, x1x3, x2x3).await?;
    transposed_pack_xor_assign(&mut c, &x3);

    // Add 2c + s via a ripple carry adder
    // LSB of c is 0
    // First round: half adder can be skipped due to LSB of c being 0
    let mut a = s;
    let mut b = c;

    let and_many_span = trace_span!(target: "searcher::network", "and_many_calls", n = b.len() - 1);

    // First full adder (carry is 0)
    let mut c = and_many(session, a[1].as_slice(), b[0].as_slice())
        .instrument(and_many_span.clone())
        .await?;

    // For last round
    let mut a_msb = a.pop().expect("Enough elements present");
    let b_msb = b.pop().expect("Enough elements present");

    // 2 -> k-1
    for (a_, b_) in a.iter_mut().skip(2).zip(b.iter_mut().skip(1)) {
        *a_ ^= c.as_slice();
        *b_ ^= c.as_slice();
        let tmp_c = and_many(session, a_.as_slice(), b_.as_slice())
            .instrument(and_many_span.clone())
            .await?;
        c ^= tmp_c;
    }

    a_msb ^= b_msb;
    a_msb ^= c;

    // Extract bits for outputs
    let res = a_msb;

    Ok(res)
}

/// Returns the MSB of the sum of three 32-bit integers using the binary parallel prefix adder tree.
/// Input integers are given in binary form.
pub(crate) async fn binary_add_3_get_msb_prefix(
    session: &mut Session,
    x1: Vec<VecShare<u64>>,
    x2: Vec<VecShare<u64>>,
    mut x3: Vec<VecShare<u64>>,
) -> Result<VecShare<u64>, Error> {
    let len = x1.len();
    debug_assert!(len == x2.len() && len == x3.len());
    debug_assert!(len == 32);

    // Let x1, x2, x3 are integers modulo 2^k.
    //
    // Full adder where x3 plays the role of an input carry, i.e.,
    // c = (x1 AND x2) XOR (x3 AND (x1 XOR x2)) and
    // s = x1 XOR x2 XOR x3
    // Note that x1 + x2 + x3 = 2 * c + s mod 2^k
    let mut x2x3 = x2;
    transposed_pack_xor_assign(&mut x2x3, &x3);
    // x1 XOR x2 XOR x3
    let mut s = transposed_pack_xor(&x1, &x2x3);

    // Compute 2 * c mod 2^k
    // x1 XOR x3
    let mut x1x3 = x1;
    transposed_pack_xor_assign(&mut x1x3, &x3);
    // Chop off the MSBs of these values as they are anyway removed by 2 * c later on.
    x1x3.pop().expect("No elements here");
    x2x3.pop().expect("No elements here");
    x3.pop().expect("No elements here");
    // (x1 XOR x3) AND (x2 XOR x3) = (x1 AND x2) XOR (x3 AND (x1 XOR x2)) XOR x3
    let mut c = transposed_pack_and(session, x1x3, x2x3).await?;
    // (x1 AND x2) XOR (x3 AND (x1 XOR x2))
    transposed_pack_xor_assign(&mut c, &x3);

    // Find the MSB of 2 * c + s using the parallel prefix adder
    // The LSB of 2 * c is zero, so we can ignore the LSB of s
    let mut a = s.drain(1..).collect::<Vec<_>>();
    let mut b = c;

    // Compute carry propagates p = a XOR b and carry generates g = a AND b
    let mut p = transposed_pack_xor(&a, &b);
    // The MSB of g is used to compute the carry of the whole sum; we don't need it as there is reduction modulo 2^k
    a.pop();
    b.pop();
    let g = transposed_pack_and(session, a, b).await?;
    // The MSB of p is needed to compute the MSB of the sum, but it doesn't needed for the carry computation
    let msb_p = p.pop().expect("No elements here");

    // Compute the carry for the MSB of the sum
    //
    // Reduce the above vectors according to the following rule:
    // p = (p0, p1, p2, p3,...) -> (p0 AND p1, p2 AND p3,...)
    // g = (g0, g1, g2, g3,...) -> (g1 XOR g0 AND p1, g3 XOR g2 AND p3,...)
    // Note that p0 is not needed to compute g, thus we can omit it as follows
    // p = (p1, p2, p3,...) -> (p2 AND p3, p4 AND p5...)
    let mut temp_p = p.drain(1..).collect::<Vec<_>>();
    let mut temp_g = g;

    while temp_g.len() != 1 {
        let (maybe_extra_p, maybe_extra_g) = if temp_g.len() % 2 == 1 {
            (temp_p.pop(), temp_g.pop())
        } else {
            (None, None)
        };

        // Split the vectors into even and odd indexed elements
        // Note that the starting index of temp_p is 1 due to removal of p0 above
        let (even_p, odd_p): (Vec<_>, Vec<_>) = temp_p
            .clone()
            .into_iter()
            .enumerate()
            .partition(|(i, _)| i % 2 == 1);
        let even_p: Vec<_> = even_p.into_iter().map(|(_, x)| x).collect();
        let odd_p: Vec<_> = odd_p.into_iter().map(|(_, x)| x).collect();
        let (even_g, odd_g): (Vec<_>, Vec<_>) = temp_g
            .clone()
            .into_iter()
            .enumerate()
            .partition(|(i, _)| i % 2 == 0);
        let even_g: Vec<_> = even_g.into_iter().map(|(_, x)| x).collect();
        let odd_g: Vec<_> = odd_g.into_iter().map(|(_, x)| x).collect();

        // Merge even_p and even_g to multiply them by odd_p at once
        // This corresponds to computing
        //            (p2 AND p3, p4 AND p5,...) and
        // (g0 AND p1, g2 AND p3, g4 AND p5,...) as above
        let mut even_p_with_even_g = even_p;
        let new_p_len = even_p_with_even_g.len();
        even_p_with_even_g.extend(even_g);
        // Remove p1 to multiply even_p with odd_p
        let mut odd_p_doubled = odd_p[1..].to_vec();
        odd_p_doubled.extend(odd_p);

        let mut tmp = transposed_pack_and(session, even_p_with_even_g, odd_p_doubled).await?;

        // Update p
        temp_p = tmp.drain(..new_p_len).collect();
        if let Some(extra_p) = maybe_extra_p {
            temp_p.push(extra_p);
        }

        // Finish computing (g1 XOR g0 AND p1, g3 XOR g2 AND p3,...) and update g
        temp_g = transposed_pack_xor(&tmp, &odd_g);
        if let Some(extra_g) = maybe_extra_g {
            temp_g.push(extra_g);
        }
    }
    // a_msb XOR b_msb XOR top carry
    let msb = msb_p ^ temp_g.pop().unwrap();
    Ok(msb)
}

// Extracts bit at position K
async fn extract_msb<const K: usize>(
    session: &mut Session,
    x: Vec<VecShare<u64>>,
) -> Result<VecShare<u64>, Error> {
    let len = x.len();

    let mut x1 = Vec::with_capacity(len);
    let mut x2 = Vec::with_capacity(len);
    let mut x3 = Vec::with_capacity(len);

    for x_ in x.into_iter() {
        let len_ = x_.len();
        let mut x1_ = VecShare::with_capacity(len_);
        let mut x2_ = VecShare::with_capacity(len_);
        let mut x3_ = VecShare::with_capacity(len_);
        for x__ in x_.into_iter() {
            let (x1__, x2__, x3__) = a2b_pre(session, x__)?;
            x1_.push(x1__);
            x2_.push(x2__);
            x3_.push(x3__);
        }
        x1.push(x1_);
        x2.push(x2_);
        x3.push(x3_);
    }

    binary_add_3_get_msb_prefix(session, x1, x2, x3).await
}

pub async fn extract_msb_u32<const K: usize>(
    session: &mut Session,
    x_: VecShare<u32>,
) -> Result<VecShare<u64>, Error> {
    let x = x_.transpose_pack_u64_with_len::<K>();
    extract_msb::<K>(session, x).await
}

// TODO a dedicated bitextraction for just one element would be more
// efficient
#[instrument(level = "trace", target = "searcher::network", skip_all)]
pub async fn single_extract_msb_u32<const K: usize>(
    session: &mut Session,
    x: Share<u32>,
) -> Result<Share<Bit>, Error> {
    let (a, b) = extract_msb_u32::<{ u32::BITS as usize }>(session, VecShare::new_vec(vec![x]))
        .await?
        .get_at(0)
        .get_ab();

    Ok(Share::new(a.get_bit_as_bit(0), b.get_bit_as_bit(0)))
}

#[instrument(level = "trace", target = "searcher::network", skip_all)]
pub async fn extract_msb_u32_batch(
    session: &mut Session,
    x: &[Share<u32>],
) -> eyre::Result<Vec<Share<Bit>>> {
    let res_len = x.len();
    let mut res = Vec::with_capacity(res_len);

    let packed_bits =
        extract_msb_u32::<{ u32::BITS as usize }>(session, VecShare::new_vec(x.to_vec())).await?;
    let mut packed_bits_iter = packed_bits.into_iter();

    while res.len() < res_len {
        let (a, b) = packed_bits_iter.next().unwrap().get_ab_ref();
        for i in 0..64 {
            if res.len() == res_len {
                break;
            }
            res.push(Share::new(a.get_bit_as_bit(i), b.get_bit_as_bit(i)));
        }
    }

    Ok(res)
}

#[instrument(level = "trace", target = "searcher::network", skip_all)]
pub async fn open_bin(session: &mut Session, shares: &[Share<Bit>]) -> eyre::Result<Vec<Bit>> {
    let network = &mut session.network_session;
    let message = if shares.len() == 1 {
        NetworkValue::RingElementBit(shares[0].b).to_network()
    } else {
        // TODO: could be optimized by packing bits
        let bits = shares
            .iter()
            .map(|x| NetworkValue::RingElementBit(x.b))
            .collect::<Vec<_>>();
        NetworkValue::vec_to_network(&bits)
    };

    network.send_next(message).await?;

    // receiving from previous party
    let c = {
        let serialized_other_shares = network.receive_prev().await;
        if shares.len() == 1 {
            match NetworkValue::from_network(serialized_other_shares) {
                Ok(NetworkValue::RingElementBit(message)) => Ok(vec![message]),
                Err(e) => Err(eyre!("Error in receiving in open_bin operation: {}", e)),
                _ => Err(eyre!("Wrong value type is received in open_bin operation")),
            }
        } else {
            match NetworkValue::vec_from_network(serialized_other_shares) {
                Ok(v) => {
                    if matches!(v[0], NetworkValue::RingElementBit(_)) {
                        Ok(v.into_iter()
                            .map(|x| match x {
                                NetworkValue::RingElementBit(message) => message,
                                _ => unreachable!(),
                            })
                            .collect())
                    } else {
                        Err(eyre!("Wrong value type is received in open_bin operation"))
                    }
                }
                Err(e) => Err(eyre!("Error in receiving in open_bin operation: {}", e)),
            }
        }
    }?;

    // XOR shares with the received shares
    izip!(shares.iter(), c.iter())
        .map(|(s, c)| Ok((s.a ^ s.b ^ c).convert()))
        .collect::<eyre::Result<Vec<_>>>()
}
