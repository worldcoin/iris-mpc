use crate::{
    execution::session::{Session, SessionHandles},
    network::value::{NetworkInt, NetworkValue},
    shares::{
        bit::Bit,
        int_ring::IntRing2k,
        ring_impl::{RingElement, VecRingElement},
        share::Share,
        vecshare::{SliceShare, VecShare},
    },
};
use eyre::{bail, eyre, Error, Result};
use iris_mpc_common::fast_metrics::FastHistogram;
use itertools::{izip, Itertools};
use num_traits::{One, Zero};
use rand::{distributions::Standard, prelude::Distribution, Rng};
use std::{cell::RefCell, ops::SubAssign};
use tracing::{instrument, trace_span, Instrument};

thread_local! {
    static ROUNDS_METRICS: RefCell<FastHistogram> = RefCell::new(
        FastHistogram::new("smpc.rounds")
    );
}

/// Splits the components of the given arithmetic share into 3 secret shares as described in Section 5.3 of the ABY3 paper.
///
/// The parties own the following arithmetic shares of x = x1 + x2 + x3:
///
/// |share component|Party 0 | Party 1 | Party 2 |
/// |---------------|--------|---------|---------|
/// |a              |x1      |x2       |x3       |
/// |b              |x3      |x1       |x2       |
///
/// The function returns the shares in the following order:
///
/// shares of x1
/// |share component|Party 0 | Party 1 | Party 2 |
/// |---------------|--------|---------|---------|
/// |a              |x1      |0        |0        |
/// |b              |0       |x1       |0        |
///
/// shares of x2
/// |share component|Party 0 | Party 1 | Party 2 |
/// |---------------|--------|---------|---------|
/// |a              |0       |x2       |0        |
/// |b              |0       |0        |x2       |
///
/// shares of x3
/// |share component|Party 0 | Party 1 | Party 2 |
/// |---------------|--------|---------|---------|
/// |a              |0       |0        |x3       |
/// |b              |x3      |0        |0        |
fn a2b_pre<T: IntRing2k>(session: &Session, x: Share<T>) -> Result<(Share<T>, Share<T>, Share<T>)> {
    let (a, b) = x.get_ab();

    let mut x1 = Share::zero();
    let mut x2 = Share::zero();
    let mut x3 = Share::zero();

    match session.own_role().index() {
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
            bail!("Cannot deal with roles that have index outside of the set [0, 1, 2]")
        }
    }
    Ok((x1, x2, x3))
}

/// Computes in place binary XOR of two vectors of bit-sliced shares.
fn transposed_pack_xor_assign<T: IntRing2k>(x1: &mut [VecShare<T>], x2: &[VecShare<T>]) {
    let len = x1.len();
    debug_assert_eq!(len, x2.len());

    for (x1, x2) in x1.iter_mut().zip(x2.iter()) {
        *x1 ^= x2.as_slice();
    }
}

/// Computes binary XOR of two vectors of bit-sliced shares.
fn transposed_pack_xor<T: IntRing2k>(x1: &[VecShare<T>], x2: &[VecShare<T>]) -> Vec<VecShare<T>> {
    let len = x1.len();
    debug_assert_eq!(len, x2.len());

    let mut res = Vec::with_capacity(len);
    for (x1, x2) in x1.iter().zip(x2.iter()) {
        res.push(x1.as_slice() ^ x2.as_slice());
    }
    res
}

/// Computes and sends a local share of the AND of two vectors of bit-sliced shares.
async fn and_many_iter_send<T: IntRing2k + NetworkInt>(
    session: &mut Session,
    a: impl Iterator<Item = Share<T>>,
    b: impl Iterator<Item = Share<T>>,
    size_hint: usize,
) -> Result<Vec<RingElement<T>>, Error>
where
    Standard: Distribution<T>,
{
    // Caller should ensure that size_hint == a.len() == b.len()
    let mut shares_a = VecRingElement::with_capacity(size_hint);
    for (a_, b_) in a.zip(b) {
        let rand = session.prf.gen_binary_zero_share::<T>();
        let mut c = &a_ & &b_;
        c ^= rand;
        shares_a.push(c);
    }

    let network = &mut session.network_session;
    network.send_ring_vec_next(&shares_a).await?;
    Ok(shares_a.0)
}

async fn and_many_send<T: IntRing2k + NetworkInt>(
    session: &mut Session,
    a: SliceShare<'_, T>,
    b: SliceShare<'_, T>,
) -> Result<Vec<RingElement<T>>, Error>
where
    Standard: Distribution<T>,
{
    if a.len() != b.len() {
        bail!("InvalidSize in and_many_send");
    }
    let mut shares_a = VecRingElement::with_capacity(a.len());
    for (a_, b_) in a.iter().zip(b.iter()) {
        let rand = session.prf.gen_binary_zero_share::<T>();
        let mut c = a_ & b_;
        c ^= rand;
        shares_a.push(c);
    }

    let network = &mut session.network_session;
    network.send_ring_vec_next(&shares_a).await?;
    Ok(shares_a.0)
}

/// Receives a share of the AND of two vectors of bit-sliced shares.
async fn and_many_receive<T: IntRing2k + NetworkInt>(
    session: &mut Session,
) -> Result<Vec<RingElement<T>>, Error> {
    let shares_b = {
        let other_share = session.network_session.receive_prev().await;

        match other_share {
            Ok(v) => T::into_vec(v),
            Err(e) => Err(eyre!("Error in and_many_receive: {e}")),
        }
    }?;
    Ok(shares_b)
}

/// Low-level SMPC protocol to compute the AND of two vectors of bit-sliced shares.
pub(crate) async fn and_many<T: IntRing2k + NetworkInt>(
    session: &mut Session,
    a: SliceShare<'_, T>,
    b: SliceShare<'_, T>,
) -> Result<VecShare<T>, Error>
where
    Standard: Distribution<T>,
{
    let shares_a = and_many_send(session, a, b).await?;
    let shares_b = and_many_receive(session).await?;
    let complete_shares = VecShare::from_ab(shares_a, shares_b);
    Ok(complete_shares)
}

/// Reduce the given vector of bit-vector shares by computing their element-wise AND.
///
/// Each vector in `v` is expected to have `len` bits.
pub(crate) async fn and_product(
    session: &mut Session,
    v: Vec<VecShare<Bit>>,
    len: usize,
) -> Result<VecShare<Bit>, Error> {
    if v.is_empty() {
        bail!("Input vector is empty");
    }
    for vec_share in &v {
        if vec_share.len() != len {
            bail!("Input vector shares have different lengths");
        }
    }

    let mut res = v;
    while res.len() > 1 {
        // if the length is odd, we save the last column to add it back later
        let maybe_last_column = if res.len() % 2 == 1 { res.pop() } else { None };
        let half_len = res.len() / 2;
        let left_bits: VecShare<u64> =
            VecShare::new_vec(res.drain(..half_len).flatten().collect_vec()).pack();
        let right_bits: VecShare<u64> =
            VecShare::new_vec(res.drain(..).flatten().collect_vec()).pack();
        let and_bits = and_many(session, left_bits.as_slice(), right_bits.as_slice()).await?;
        let mut and_bits = and_bits.convert_to_bits();
        let num_and_bits = half_len * len;
        and_bits.truncate(num_and_bits);
        res = and_bits
            .inner()
            .chunks(len)
            .map(|chunk| VecShare::new_vec(chunk.to_vec()))
            .collect_vec();
        res.extend(maybe_last_column);
    }
    res.pop().ok_or(eyre!("Not enough elements"))
}

/// Computes binary AND of two vectors of bit-sliced shares.
#[instrument(level = "trace", target = "searcher::network", skip(session, x1, x2))]
async fn transposed_pack_and<T: IntRing2k + NetworkInt>(
    session: &mut Session,
    x1: Vec<VecShare<T>>,
    x2: Vec<VecShare<T>>,
) -> Result<Vec<VecShare<T>>, Error>
where
    Standard: Distribution<T>,
{
    if x1.len() != x2.len() {
        bail!("Inputs have different length {} {}", x1.len(), x2.len());
    }
    let x1_length = x1.len();

    let chunk_sizes = x1.iter().map(VecShare::len).collect::<Vec<_>>();
    for (chunk_size1, chunk_size2) in izip!(chunk_sizes.iter(), x2.iter().map(VecShare::len)) {
        if *chunk_size1 != chunk_size2 {
            bail!("VecShare lengths are not equal");
        }
    }

    let x1 = VecShare::flatten(x1);
    let x2 = VecShare::flatten(x2);
    let mut shares_a = and_many_iter_send(session, x1, x2, x1_length).await?;
    let mut shares_b = and_many_receive(session).await?;

    // Unflatten the shares vectors
    let mut res = Vec::with_capacity(chunk_sizes.len());
    for l in chunk_sizes {
        let a = shares_a.drain(..l);
        let b = shares_b.drain(..l);
        res.push(VecShare::from_iter_ab(a, b));
    }
    Ok(res)
}

/// Computes the sum of three integers using the binary ripple-carry adder and return two resulting overflow carries.
/// Input integers are given in binary form.
#[instrument(level = "trace", target = "searcher::network", skip_all)]
async fn binary_add_3_get_two_carries<T: IntRing2k + NetworkInt>(
    session: &mut Session,
    x1: Vec<VecShare<T>>,
    x2: Vec<VecShare<T>>,
    x3: Vec<VecShare<T>>,
    truncate_len: usize,
) -> Result<(VecShare<Bit>, VecShare<Bit>), Error>
where
    Standard: Distribution<T>,
{
    let len = x1.len();
    if len != x2.len() || len != x3.len() {
        bail!(
            "Inputs have different length {} {} {}",
            len,
            x2.len(),
            x3.len()
        );
    };

    if len < 16 {
        bail!("Input length should be at least 16: {len}");
    }

    // Let x1, x2, x3 are integers modulo 2^k.
    //
    // Full adder where x3 plays the role of an input carry yields
    // c = (x1 AND x2) XOR (x3 AND (x1 XOR x2)) and
    // s = x1 XOR x2 XOR x3
    // Note that x1 + x2 + x3 = 2 * c + s mod 2^k

    let mut x2x3 = x2;
    transposed_pack_xor_assign(&mut x2x3, &x3);
    // x1 XOR x2 XOR x3
    let mut s = transposed_pack_xor(&x1, &x2x3);
    let mut x1x3 = x1;
    transposed_pack_xor_assign(&mut x1x3, &x3);
    // (x1 XOR x3) AND (x2 XOR x3) = (x1 AND x2) XOR (x3 AND (x1 XOR x2)) XOR x3
    let mut c = transposed_pack_and(session, x1x3, x2x3).await?;
    // (x1 AND x2) XOR (x3 AND (x1 XOR x2))
    transposed_pack_xor_assign(&mut c, &x3);

    // Find the MSB of 2 * c + s using the parallel prefix adder
    let and_many_span = trace_span!(target: "searcher::network", "and_many_calls", n = c.len());

    // First full adder (carry is 0)
    // The LSB of 2 * c is zero, so we can ignore the LSB of s
    let mut carry = and_many(session, s[1].as_slice(), c[0].as_slice())
        .instrument(and_many_span.clone())
        .await?;

    // Keep the MSB of c to compute the carries
    let mut c_msb = c.pop().ok_or(eyre!("Not enough elements"))?;

    // Compute carry of the sum of 2*c without MSB and s
    for (s_, c_) in s.iter_mut().skip(2).zip(c.iter_mut().skip(1)) {
        *s_ ^= carry.as_slice();
        *c_ ^= carry.as_slice();
        let tmp_c = and_many(session, s_.as_slice(), c_.as_slice())
            .instrument(and_many_span.clone())
            .await?;
        carry ^= tmp_c;
    }

    // Top carry
    let res2 = and_many(session, c_msb.as_slice(), carry.as_slice())
        .instrument(and_many_span)
        .await?;
    // Carry next to top
    c_msb ^= carry;

    // Extract bits for outputs
    let mut res1 = c_msb.convert_to_bits();
    res1.truncate(truncate_len);
    let mut res2 = res2.convert_to_bits();
    res2.truncate(truncate_len);

    Ok((res1, res2))
}

#[instrument(level = "trace", target = "searcher::network", skip_all)]
async fn bit_inject_ot_2round_helper<T: IntRing2k + NetworkInt>(
    session: &mut Session,
    input: VecShare<Bit>,
) -> Result<VecShare<T>, Error>
where
    Standard: Distribution<T>,
{
    let len = input.len();
    let mut wc = Vec::with_capacity(len);
    let mut shares = VecShare::with_capacity(len);
    let prf = &mut session.prf;

    for inp in input.into_iter() {
        // New random share (also generated by Sender)
        let c3 = prf.get_prev_prf().gen::<RingElement<T>>();
        shares.push(Share::new(RingElement::zero(), c3));

        // Random masks of the OT (also generated by Sender)
        let w0 = prf.get_prev_prf().gen::<RingElement<T>>();
        let w1 = prf.get_prev_prf().gen::<RingElement<T>>();

        // Choose the bit share of the current party unknown to Sender to pick w0 or w1
        let choice = inp.get_a().convert().convert();
        if choice {
            wc.push(w1);
        } else {
            wc.push(w0);
        }
    }

    let network = &mut session.network_session;

    // Send masks to Receiver
    network.send_next(T::new_network_vec(wc)).await?;

    ROUNDS_METRICS.with_borrow_mut(|rounds_metrics| {
        rounds_metrics.record(1.0);
    });

    // Receive m0 or m1 from Receiver
    let m0_or_m1 = match network.receive_next().await {
        Ok(nv) => T::into_vec(nv)?,
        Err(e) => return Err(eyre!("Could not deserialize properly in bit inject: {e}")),
    };

    // Set the first share to the value of m0 or m1
    for (s, mb) in shares.iter_mut().zip(m0_or_m1) {
        s.a = mb;
    }
    Ok(shares)
}

#[instrument(level = "trace", target = "searcher::network", skip_all)]
async fn bit_inject_ot_2round_receiver<T: IntRing2k + NetworkInt>(
    session: &mut Session,
    input: VecShare<Bit>,
) -> Result<VecShare<T>>
where
    Standard: Distribution<T>,
{
    let network = &mut session.network_session;

    let (m0, m1, wc) = {
        let m0_and_m1 = match network.receive_next().await {
            Ok(v) => NetworkValue::vec_from_network(v)?,
            _ => bail!("Cannot deserialize m0 and m1 into vec"),
        };
        if m0_and_m1.len() != 2 {
            bail!(
                "Deserialized vec in bit inject is wrong length: {}",
                m0_and_m1.len()
            );
        }
        let (m0, m1) = m0_and_m1
            .into_iter()
            .map(T::into_vec)
            .collect_tuple()
            .ok_or(eyre!("Cannot deserialize m0 and m1 into tuple"))?;

        let wc = match network.receive_prev().await {
            Ok(v) => T::into_vec(v),
            Err(e) => Err(eyre!("Could not deserialize properly in bit inject: {e}")),
        };
        (m0, m1, wc)
    };

    let (m0, m1, wc) = (m0?, m1?, wc?);

    let len = input.len();
    let mut shares = VecShare::with_capacity(len);
    let mut unmasked_m = Vec::with_capacity(len);

    for ((inp, wc), (m0, m1)) in input
        .into_iter()
        .zip(wc.into_iter())
        .zip(m0.into_iter().zip(m1.into_iter()))
    {
        // New random share (also generated by Sender)
        let c2 = session.prf.get_my_prf().gen::<RingElement<T>>();

        // Unmask m0 or m1 depending on the bit share owned by the current party.
        let choice = inp.get_b().convert().convert();
        let mb = if choice { wc ^ m1 } else { wc ^ m0 };

        unmasked_m.push(mb);
        shares.push(Share::new(c2, mb));
    }

    // Send unmasked m to Helper
    network.send_prev(T::new_network_vec(unmasked_m)).await?;

    ROUNDS_METRICS.with_borrow_mut(|rounds_metrics| {
        rounds_metrics.record(1.0);
    });

    Ok(shares)
}

#[instrument(level = "trace", target = "searcher::network", skip_all)]
async fn bit_inject_ot_2round_sender<T: IntRing2k + NetworkInt>(
    session: &mut Session,
    input: VecShare<Bit>,
) -> Result<VecShare<T>, Error>
where
    Standard: Distribution<T>,
{
    let len = input.len();
    let mut m0 = Vec::with_capacity(len);
    let mut m1 = Vec::with_capacity(len);
    let mut shares = VecShare::with_capacity(len);
    let prf = &mut session.prf;

    for inp in input.into_iter() {
        let (a, b) = inp.get_ab();
        // New random shares
        let (c3, c2) = prf.gen_rands::<RingElement<T>>();
        // Random masks of the OT
        let w0 = prf.get_my_prf().gen::<RingElement<T>>();
        let w1 = prf.get_my_prf().gen::<RingElement<T>>();

        shares.push(Share::new(c3, c2));
        let c = c3 + c2;
        let xor = RingElement(T::from((a ^ b).convert().convert()));
        let m0_ = xor - c;
        let m1_ = (xor ^ RingElement::one()) - c;
        m0.push(m0_ ^ w0);
        m1.push(m1_ ^ w1);
    }

    let m0_and_m1: Vec<NetworkValue> = [m0, m1]
        .into_iter()
        .map(T::new_network_vec)
        .collect::<Vec<_>>();

    // Send m0 and m1 to Receiver
    session
        .network_session
        .send_prev(NetworkValue::vec_to_network(m0_and_m1))
        .await?;

    ROUNDS_METRICS.with_borrow_mut(|rounds_metrics| {
        rounds_metrics.record(1.0);
    });

    Ok(shares)
}

/// Conducts a 3 party OT protocol to inject bits into shares of type T.
/// The specifics of the protocol can be found in the ABY3 paper (Section 5.4.1).
///
/// Party 2 sends twice more than other parties.
/// So a real implementation should actually rotate parties around.
///
/// The protocol itself is unbalanced, but we balance the assignment of roles
/// in a round-robin over sessions.
pub(crate) async fn bit_inject_ot_2round<T: IntRing2k + NetworkInt>(
    session: &mut Session,
    input: VecShare<Bit>,
) -> Result<VecShare<T>, Error>
where
    Standard: Distribution<T>,
{
    let role_index = (session.own_role().index() + session.session_id().0 as usize) % 3;
    let res = match role_index {
        0 => {
            // OT Helper
            bit_inject_ot_2round_helper::<T>(session, input).await?
        }
        1 => {
            // OT Receiver
            bit_inject_ot_2round_receiver::<T>(session, input).await?
        }
        2 => {
            // OT Sender
            bit_inject_ot_2round_sender::<T>(session, input).await?
        }
        _ => {
            bail!("Cannot deal with roles outside of the set [0, 1, 2] in bit_inject_ot")
        }
    };
    Ok(res)
}

/// Lifts the given shares of u16 to shares of u32 by multiplying them by 2^k.
///
/// This works since for any k-bit value b = x + y + z mod 2^16 with k < 16, it holds
/// (x >> l) + (y >> l) + (z >> l) = (b >> l) mod 2^32 for any l <= 32-k.
pub(crate) fn mul_lift_2k<const K: u64>(val: &Share<u16>) -> Share<u32> {
    let a = (u32::from(val.a.0)) << K;
    let b = (u32::from(val.b.0)) << K;
    Share::new(RingElement(a), RingElement(b))
}

/// Lifts the given shares of u16 to shares of u32 by multiplying them by 2^k.
fn mul_lift_2k_many<const K: u64>(vals: SliceShare<u16>) -> VecShare<u32> {
    VecShare::new_vec(vals.iter().map(mul_lift_2k::<K>).collect())
}

/// Lifts the given shares of u16 to shares of u32.
pub(crate) async fn lift(session: &mut Session, shares: VecShare<u16>) -> Result<VecShare<u32>> {
    let len = shares.len();
    let mut padded_len = (len + 63) / 64;
    padded_len *= 64;

    // Interpret the shares as u32
    let mut x_a = VecShare::with_capacity(padded_len);
    for share in shares.iter() {
        x_a.push(Share::new(
            RingElement(share.a.0 as u32),
            RingElement(share.b.0 as u32),
        ));
    }

    // Bit-slice the shares into 64-bit shares
    let x = shares.transpose_pack_u64();

    // Prepare the local input shares to be summed by the binary adder
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

    // Sum the binary shares using the binary parallel prefix adder.
    // Since the input shares are u16 and we sum over u32, the two carries arise, i.e.,
    // x1 + x2 + x3 = x + b1 * 2^16 + b2 * 2^17 mod 2^32
    let (mut b1, b2) = binary_add_3_get_two_carries(session, x1, x2, x3, len).await?;

    // Lift b1 and b2 into u16 via bit injection
    // This slightly deviates from Algorithm 10 from ePrint/2024/705 as bit injection to integers modulo 2^15 doesn't give any advantage.
    b1.extend(b2);
    let mut b = bit_inject_ot_2round(session, b1).await?;
    let (b1, b2) = b.split_at_mut(len);

    // Lift b1 and b2 into u32 and multiply them by 2^16 and 2^17, respectively.
    // This can be done by computing b1 as u32 << 16 and b2 as u32 << 17.
    let b1 = mul_lift_2k_many::<16>(b1.to_slice());
    let b2 = mul_lift_2k_many::<17>(b2.to_slice());

    // Compute x1 + x2 + x3 - b1 * 2^16 - b2 * 2^17 = x mod 2^32
    x_a.sub_assign(b1);
    x_a.sub_assign(b2);
    Ok(x_a)
}

/// Returns the MSB of the sum of three 32-bit integers using the binary ripple-carry adder.
/// Input integers are given in binary form.
///
/// NOTE: This adder has a linear multiplicative depth, which is way worse than the logarithmic depth of the parallel prefix adder below.
/// However, its throughput is almost two times better.
#[allow(dead_code)]
async fn binary_add_3_get_msb<T: IntRing2k + NetworkInt>(
    session: &mut Session,
    x1: Vec<VecShare<T>>,
    x2: Vec<VecShare<T>>,
    mut x3: Vec<VecShare<T>>,
) -> Result<VecShare<T>, Error>
where
    Standard: Distribution<T>,
{
    let len = x1.len();
    if len != x2.len() || len != x3.len() {
        bail!(
            "Inputs have different length {} {} {}",
            len,
            x2.len(),
            x3.len()
        );
    };

    if len < 32 {
        bail!("Input length should be at least 32: {len}");
    }

    // Let x1, x2, x3 are integers modulo 2^k.
    //
    // Full adder where x3 plays the role of an input carry yields
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
    x1x3.pop().ok_or(eyre!("Not enough elements"))?;
    x2x3.pop().ok_or(eyre!("Not enough elements"))?;
    x3.pop().ok_or(eyre!("Not enough elements"))?;
    // (x1 XOR x3) AND (x2 XOR x3) = (x1 AND x2) XOR (x3 AND (x1 XOR x2)) XOR x3
    let mut c = transposed_pack_and(session, x1x3, x2x3).await?;
    // (x1 AND x2) XOR (x3 AND (x1 XOR x2))
    transposed_pack_xor_assign(&mut c, &x3);

    // Find the MSB of 2 * c + s using the parallel prefix adder
    let and_many_span = trace_span!(target: "searcher::network", "and_many_calls", n = c.len() - 1);

    // First full adder (carry is 0)
    // The LSB of 2 * c is zero, so we can ignore the LSB of s
    let mut carry = and_many(session, s[1].as_slice(), c[0].as_slice())
        .instrument(and_many_span.clone())
        .await?;

    // To compute the MSB of the sum we have to add the MSB of s and c later
    let s_msb = s.pop().ok_or(eyre!("Not enough elements"))?;
    let c_msb = c.pop().ok_or(eyre!("Not enough elements"))?;

    // Compute carry for the MSB of the sum
    for (s_, c_) in s.iter_mut().skip(2).zip(c.iter_mut().skip(1)) {
        // carry = s_ AND c_ XOR carry AND (s_ XOR c_) = (s_ XOR carry) AND (c_ XOR carry) XOR carry
        *s_ ^= carry.as_slice();
        *c_ ^= carry.as_slice();
        let tmp_c = and_many(session, s_.as_slice(), c_.as_slice())
            .instrument(and_many_span.clone())
            .await?;
        carry ^= tmp_c;
    }

    // Return the MSB of the sum
    Ok(s_msb ^ c_msb ^ carry)
}

/// Returns the MSB of the sum of three integers of type T using the binary parallel prefix adder tree.
/// Input integers are given in binary form.
async fn binary_add_3_get_msb_prefix<T: IntRing2k + NetworkInt>(
    session: &mut Session,
    x1: Vec<VecShare<T>>,
    x2: Vec<VecShare<T>>,
    mut x3: Vec<VecShare<T>>,
) -> Result<VecShare<T>, Error>
where
    Standard: Distribution<T>,
{
    let len = x1.len();
    if len != x2.len() || len != x3.len() {
        bail!(
            "Inputs have different length {} {} {}",
            len,
            x2.len(),
            x3.len()
        );
    };

    // Let x1, x2, x3 are integers modulo 2^k.
    //
    // Full adder where x3 plays the role of an input carry yields
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
    x1x3.pop().ok_or(eyre!("Not enough elements"))?;
    x2x3.pop().ok_or(eyre!("Not enough elements"))?;
    x3.pop().ok_or(eyre!("Not enough elements"))?;
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
    let msb_p = p.pop().ok_or(eyre!("Not enough elements"))?;

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
        // We anticipate concatenations to allocate vecs with the correct capacity
        // and minimize cloning/collecting

        // To assess correctness of these sizes, note that the encompassing while loop
        // maintains the invariant `temp_g.len() - 1 = temp_p.len()`

        let mut even_p_with_even_g = Vec::with_capacity(temp_g.len() - 1);
        let mut odd_p_temp = Vec::with_capacity(temp_p.len() / 2 + 1);
        let mut odd_g = Vec::with_capacity(temp_g.len() / 2);

        for (i, p) in temp_p.into_iter().enumerate() {
            if i % 2 == 1 {
                even_p_with_even_g.push(p);
            } else {
                odd_p_temp.push(p);
            }
        }
        let new_p_len = even_p_with_even_g.len();

        for (i, g) in temp_g.into_iter().enumerate() {
            if i % 2 == 0 {
                even_p_with_even_g.push(g);
            } else {
                odd_g.push(g);
            }
        }

        // Now `even_p_with_even_g` contains merged even_p and even_g to multiply
        // them by odd_p at once
        // This corresponds to computing
        //            (p2 AND p3, p4 AND p5,...) and
        // (g0 AND p1, g2 AND p3, g4 AND p5,...) as above

        let mut odd_p_doubled = Vec::with_capacity(odd_p_temp.len() * 2 - 1);
        // Remove p1 to multiply even_p with odd_p
        odd_p_doubled.extend(odd_p_temp.iter().skip(1).cloned());
        odd_p_doubled.extend(odd_p_temp);

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
    let msb = msb_p
        ^ temp_g
            .pop()
            .ok_or(eyre!("Should contain exactly 1 carry"))?;
    Ok(msb)
}

/// Extracts the MSBs of given bit-sliced arithmetic shares.
/// The input is supposed to be given in a transposed form such that the i-th `VecShare<T>` contains the i-th bits of the given arithmetic shares.
/// This function follow the arithmetic-to-binary (A2B) conversion protocol from the ABY3 framework (see Section 5.3, Bit Decomposition).
/// The only difference is that the binary circuit returns only the MSB of the sum.
///
/// The generic T type is only used to batch bits and has no relation to the underlying type of the input arithmetic shares.
async fn extract_msb<T: IntRing2k + NetworkInt>(
    session: &mut Session,
    x: Vec<VecShare<T>>,
) -> Result<VecShare<T>, Error>
where
    Standard: Distribution<T>,
{
    let len = x.len();

    // Prepare the local input shares to be summed by the binary adder
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

    // Sum the binary shares using the binary parallel prefix adder and return the MSB
    binary_add_3_get_msb_prefix(session, x1, x2, x3).await
}

/// Extracts the MSBs of the secret shared input values in a bit-sliced form as u64 shares, i.e., the i-th bit of the j-th u64 secret share is the MSB of the (j * 64 + i)-th input value.
async fn extract_msb_u32(session: &mut Session, x_: VecShare<u32>) -> Result<VecShare<u64>, Error> {
    let x = x_.transpose_pack_u64();
    extract_msb::<u64>(session, x).await
}

/// Extracts the MSB of the secret shared input value.
pub(crate) async fn single_extract_msb_u32(
    session: &mut Session,
    x: Share<u32>,
) -> Result<Share<Bit>, Error> {
    let (a, b) = extract_msb_u32(session, VecShare::new_vec(vec![x]))
        .await?
        .get_at(0)
        .get_ab();

    Ok(Share::new(a.get_bit_as_bit(0), b.get_bit_as_bit(0)))
}

/// Extracts the secret shared MSBs of the secret shared input values.
#[instrument(level = "trace", target = "searcher::network", skip_all)]
pub(crate) async fn extract_msb_u32_batch(
    session: &mut Session,
    x: &[Share<u32>],
) -> Result<Vec<Share<Bit>>> {
    let res_len = x.len();
    let mut res = Vec::with_capacity(res_len);

    let packed_bits = extract_msb_u32(session, VecShare::new_vec(x.to_vec())).await?;

    'outer: for bit_batch in packed_bits.into_iter() {
        let (a, b) = bit_batch.get_ab();
        for i in 0..64 {
            res.push(Share::new(a.get_bit_as_bit(i), b.get_bit_as_bit(i)));
            if res.len() == res_len {
                break 'outer;
            }
        }
    }

    Ok(res)
}

/// Opens a vector of binary additive replicated secret shares as described in the ABY3 framework.
///
/// In particular, each party holds a share of the form `(a, b)` where `a` and `b` are already known to the next and previous parties, respectively.
/// Thus, the current party should send its `b` share to the next party and receive the `b` share from the previous party.
/// `a XOR b XOR previous b` yields the opened bit.
#[instrument(level = "trace", target = "searcher::network", skip_all)]
pub(crate) async fn open_bin(session: &mut Session, shares: &[Share<Bit>]) -> Result<Vec<Bit>> {
    let network = &mut session.network_session;
    let message = if shares.len() == 1 {
        NetworkValue::RingElementBit(shares[0].b)
    } else {
        // TODO: could be optimized by packing bits
        let bits = shares
            .iter()
            .map(|x| NetworkValue::RingElementBit(x.b))
            .collect::<Vec<_>>();
        NetworkValue::vec_to_network(bits)
    };

    network.send_next(message).await?;

    // Receiving `b` from previous party
    let b_from_previous = {
        let other_shares = network
            .receive_prev()
            .await
            .map_err(|e| eyre!("Error in receiving in open_bin operation: {}", e))?;
        if shares.len() == 1 {
            match other_shares {
                NetworkValue::RingElementBit(message) => Ok(vec![message]),
                _ => Err(eyre!("Wrong value type is received in open_bin operation")),
            }
        } else {
            match NetworkValue::vec_from_network(other_shares) {
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
    izip!(shares.iter(), b_from_previous.iter())
        .map(|(s, prev_b)| Ok((s.a ^ s.b ^ prev_b).convert()))
        .collect::<Result<Vec<_>>>()
}

/// Extracts the MSBs of the secret shared input values in a bit-sliced form as u64 shares, i.e., the i-th bit of the j-th u64 secret share is the MSB of the (j * 64 + i)-th input value.
async fn extract_msb_u16(session: &mut Session, x_: VecShare<u16>) -> Result<VecShare<u64>, Error> {
    let x = x_.transpose_pack_u64();
    extract_msb::<u64>(session, x).await
}

/// Extracts the MSB of the secret shared input value.
pub(crate) async fn extract_msb_u16_batch(
    session: &mut Session,
    x: &[Share<u16>],
) -> Result<Vec<Share<Bit>>> {
    let res_len = x.len();
    let mut res = Vec::with_capacity(res_len);

    let packed_bits = extract_msb_u16(session, VecShare::new_vec(x.to_vec())).await?;

    'outer: for bit_batch in packed_bits.into_iter() {
        let (a, b) = bit_batch.get_ab();
        for i in 0..64 {
            res.push(Share::new(a.get_bit_as_bit(i), b.get_bit_as_bit(i)));
            if res.len() == res_len {
                break 'outer;
            }
        }
    }

    Ok(res)
}
