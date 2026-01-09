use crate::{
    execution::session::{Session, SessionHandles},
    network::value::{NetworkInt, NetworkValue},
    shares::{
        bit::Bit,
        int_ring::IntRing2k,
        ring_impl::RingElement,
        share::Share,
        vecshare::{SliceShare, VecShare},
    },
};
use std::io::Write;

use aes_prng::AesRng;
use ark_std::{end_timer, start_timer};
use eyre::{bail, eyre, Error, Result};
use iris_mpc_common::fast_metrics::FastHistogram;
use itertools::{izip, Itertools};
use num_traits::{One, Zero};
use rand::prelude::*;
use rand::{distributions::Standard, prelude::Distribution, Rng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{
    cell::RefCell,
    ops::SubAssign,
    sync::atomic::{AtomicU64, AtomicU8, Ordering},
};
use tracing::{instrument, trace_span, Instrument};

// Global counters for FSS timing (using atomic types for thread-safe accumulation)
static FSS_PRG_KEYGEN_COUNT: AtomicU64 = AtomicU64::new(0);
static FSS_PRG_KEYGEN_SUM_MICROS: AtomicU64 = AtomicU64::new(0);
static FSS_PRG_KEYGEN_MIN_MICROS: AtomicU64 = AtomicU64::new(u64::MAX);
static FSS_PRG_KEYGEN_MAX_MICROS: AtomicU64 = AtomicU64::new(0);

static FSS_ICF_EVAL_COUNT: AtomicU64 = AtomicU64::new(0);
static FSS_ICF_EVAL_SUM_MICROS: AtomicU64 = AtomicU64::new(0);
static FSS_ICF_EVAL_MIN_MICROS: AtomicU64 = AtomicU64::new(u64::MAX);
static FSS_ICF_EVAL_MAX_MICROS: AtomicU64 = AtomicU64::new(0);

static FSS_ICF_KEYGEN_COUNT: AtomicU64 = AtomicU64::new(0);
static FSS_ICF_KEYGEN_SUM_MICROS: AtomicU64 = AtomicU64::new(0);
static FSS_ICF_KEYGEN_MIN_MICROS: AtomicU64 = AtomicU64::new(u64::MAX);
static FSS_ICF_KEYGEN_MAX_MICROS: AtomicU64 = AtomicU64::new(0);

static FSS_NETWORK_COUNT: AtomicU64 = AtomicU64::new(0);
static FSS_NETWORK_SUM_MICROS: AtomicU64 = AtomicU64::new(0);
static FSS_NETWORK_MIN_MICROS: AtomicU64 = AtomicU64::new(u64::MAX);
static FSS_NETWORK_MAX_MICROS: AtomicU64 = AtomicU64::new(0);
static FSS_NETWORK_BYTES_SENT: AtomicU64 = AtomicU64::new(0);

/// Record a PRG keygen timing measurement in microseconds.
#[inline]
fn record_prg_keygen_micros(micros: u64) {
    FSS_PRG_KEYGEN_COUNT.fetch_add(1, Ordering::Relaxed);
    FSS_PRG_KEYGEN_SUM_MICROS.fetch_add(micros, Ordering::Relaxed);

    // Update min
    let mut current_min = FSS_PRG_KEYGEN_MIN_MICROS.load(Ordering::Relaxed);
    while micros < current_min {
        match FSS_PRG_KEYGEN_MIN_MICROS.compare_exchange(
            current_min,
            micros,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => current_min = actual,
        }
    }

    // Update max
    let mut current_max = FSS_PRG_KEYGEN_MAX_MICROS.load(Ordering::Relaxed);
    while micros > current_max {
        match FSS_PRG_KEYGEN_MAX_MICROS.compare_exchange(
            current_max,
            micros,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => current_max = actual,
        }
    }
}

/// Record an ICF eval timing measurement in microseconds.
#[inline]
fn record_icf_eval_micros(micros: u64) {
    FSS_ICF_EVAL_COUNT.fetch_add(1, Ordering::Relaxed);
    FSS_ICF_EVAL_SUM_MICROS.fetch_add(micros, Ordering::Relaxed);

    // Update min
    let mut current_min = FSS_ICF_EVAL_MIN_MICROS.load(Ordering::Relaxed);
    while micros < current_min {
        match FSS_ICF_EVAL_MIN_MICROS.compare_exchange(
            current_min,
            micros,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => current_min = actual,
        }
    }

    // Update max
    let mut current_max = FSS_ICF_EVAL_MAX_MICROS.load(Ordering::Relaxed);
    while micros > current_max {
        match FSS_ICF_EVAL_MAX_MICROS.compare_exchange(
            current_max,
            micros,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => current_max = actual,
        }
    }
}

/// Record an ICF keygen timing measurement in microseconds.
#[inline]
fn record_icf_keygen_micros(micros: u64) {
    FSS_ICF_KEYGEN_COUNT.fetch_add(1, Ordering::Relaxed);
    FSS_ICF_KEYGEN_SUM_MICROS.fetch_add(micros, Ordering::Relaxed);

    // Update min
    let mut current_min = FSS_ICF_KEYGEN_MIN_MICROS.load(Ordering::Relaxed);
    while micros < current_min {
        match FSS_ICF_KEYGEN_MIN_MICROS.compare_exchange(
            current_min,
            micros,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => current_min = actual,
        }
    }

    // Update max
    let mut current_max = FSS_ICF_KEYGEN_MAX_MICROS.load(Ordering::Relaxed);
    while micros > current_max {
        match FSS_ICF_KEYGEN_MAX_MICROS.compare_exchange(
            current_max,
            micros,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => current_max = actual,
        }
    }
}

/// Record a network operation timing measurement in microseconds.
#[inline]
fn record_network_micros(micros: u64) {
    FSS_NETWORK_COUNT.fetch_add(1, Ordering::Relaxed);
    FSS_NETWORK_SUM_MICROS.fetch_add(micros, Ordering::Relaxed);

    // Update min
    let mut current_min = FSS_NETWORK_MIN_MICROS.load(Ordering::Relaxed);
    while micros < current_min {
        match FSS_NETWORK_MIN_MICROS.compare_exchange(
            current_min,
            micros,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => current_min = actual,
        }
    }

    // Update max
    let mut current_max = FSS_NETWORK_MAX_MICROS.load(Ordering::Relaxed);
    while micros > current_max {
        match FSS_NETWORK_MAX_MICROS.compare_exchange(
            current_max,
            micros,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => current_max = actual,
        }
    }
}

/// Record bytes sent over the network.
#[inline]
fn record_network_bytes_sent(bytes: u64) {
    FSS_NETWORK_BYTES_SENT.fetch_add(bytes, Ordering::Relaxed);
}

thread_local! {
    static ROUNDS_METRICS: RefCell<FastHistogram> = RefCell::new(
        FastHistogram::new("smpc.rounds")
    );
}

use fss_rs::icf::{IcShare, Icf, InG, IntvFn, OutG};
use fss_rs::prg::Aes128MatyasMeyerOseasPrg;

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
    let mut shares_a = Vec::with_capacity(size_hint);
    for (a_, b_) in a.zip(b) {
        let rand = session.prf.gen_binary_zero_share::<T>();
        let mut c = &a_ & &b_;
        c ^= rand;
        shares_a.push(c);
    }

    let network = &mut session.network_session;
    let messages = shares_a.clone();
    let message = if messages.len() == 1 {
        T::new_network_element(messages[0])
    } else {
        T::new_network_vec(messages)
    };
    network.send_next(message).await?;
    Ok(shares_a)
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
    let mut shares_a = Vec::with_capacity(a.len());
    for (a_, b_) in a.iter().zip(b.iter()) {
        let rand = session.prf.gen_binary_zero_share::<T>();
        let mut c = a_ & b_;
        c ^= rand;
        shares_a.push(c);
    }

    let network = &mut session.network_session;
    let messages = shares_a.clone();
    let message = if messages.len() == 1 {
        T::new_network_element(messages[0])
    } else {
        T::new_network_vec(messages)
    };
    network.send_next(message).await?;
    Ok(shares_a)
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
async fn and_many<T: IntRing2k + NetworkInt>(
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

/// Computes binary AND of two vectors of bit-sliced shares.
/// Input is 2 Vec<VecShare<T>> i.e.< VecShare of bit 0, VecShare of bit 1 ...>
/// First checks that the shares are same length (they have to have both the same number of bits)
/// Flattens the shares into the format:
///     x1: VecShare< Share of bit 0, Share of bit 1 ... > same for x2
/// and_many_send: each party i calculates (a_i*b_i)^key and sends to next party
/// and_many_receive: each party receives the other party's share (calculated by send)
/// "Unflatten" the shares again, to their original form
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
    transposed_pack_xor_assign(&mut x1x3, &x3); // this is x1x3 = x1 XOR x3
                                                // (x1 XOR x3) AND (x2 XOR x3) = (x1 AND x2) XOR (x3 AND (x1 XOR x2)) XOR x3
    let mut c = transposed_pack_and(session, x1x3, x2x3).await?; // c = (x1 XOR x3) AND (x2 XOR x3)
    transposed_pack_xor_assign(&mut c, &x3); // c = (x1 AND x2) XOR (x3 AND (x1 XOR x2))
                                             // do the above to remove the last ... XOR x3 (because x3 XOR x3 = 0 and x XOR 0 = x)

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

/// our implementation of binary_add_3_get_msb_prefix
///  t |        Party 0                            |        Party 1                           |                Party 2 (dealer)
///  --+-------------------------------------------+------------------------------------------+----------------------------------------
///    | has shares (d0, d2), and keys k0, k2      | has shares (d1, d0), and keys k1, k0     | has shares (d2, d1), and keys k2, k1
///  1 | Use k2 to generate a prf key r2 (also r') | Use k1 to generate a prf key r1 (also r')| Use k2,k1 to generate prf keys r2,r1
///  2 | send_next(d2+r2)   to Party 1             | send_prev(d1+r1)   to Party 0            | (kFss0, kFss1) = gen_IC(r_in = r1+r2)
///  3 | d1+r1 = receive_next() from Party 1       | d2+r2 = receive_prev() from Party 0      | send_next(kFss0) To Party 0
///  4 | kFss0 = receive_prev() from Party 2       |                                          | send_prev(kFss1) To Party 1
///  5 |                                           | [A] kFss1 = receive_next() from Party 2  |
///  6 | Can reconstruct: d+r = d1+r1+d2+r2+d0     | [B] Can reconstruct: d+r = d1+r1+d2+r2+d0|<-- potentially swap A and B (check this)
///  7 | f_x_0 = eval_IC(kFss0, d+r)               | f_x_1 = eval_IC(kFss1, d+r)              |
///  8 | use k0 to generate r' (done above)        | use k0 to generate r'  (done above)      | gamma = gen_zero_shares(k2) [use this for RSS below]
///  -------------------------------------------------Doing 2 out of 3 RSS in this part ---------------------------------------
///  - | My share is t0 = f_x_0+r'                 | My share is t1=f_x_1-r'                  | we want to split t0 and t1 to 2 out of 3 RSS
///  9 | alpha = gen_zero_shares(k0)               | beta = gen_zero_shares(k1)               | send_next(gamma)
///  10| send_next(t0+alpha)                       | send_next(t1+beta)                       | t1+beta = receive_prev()
///  11| gamma = receive_prev()                    | t0+alpha = receive_prev()                |
///  - | My shares are: (t0+alpha, gamma)          | My shares are: (t1+beta, t0+alpha)       | My shares are: (t1+beta, gamma)
///  --------------------------------------Previous version if we don't want 2-out-of-3 RSS ------------------------------------
///   | send_prev(r'+f_x_0) to Party 2 <send LSB> | send_next(f_x_1-r') to Party 2 <send LSB>|
///   | send_next(f_x_0 + r') to Party 1   <..>   | send_prev(f_x_1-r') to Party 0    <..>   | f_x_1-r' = receive_prev() from Party 1
///   | f_x_1 - r'= receive_next() from Party 1   | f_x_0 + r' = receive_prev() from Party 0 | f_x_0+r' = receive_next() from Party 0
///   ---- everyone has the same 2 shares, open_bin will just add the two shares, no communication
///  these shares will be BITS not u128
async fn add_3_get_msb_fss(session: &mut Session, x: &Share<u32>) -> Result<Share<Bit>, Error>
where
    Standard: Distribution<u32>,
{
    todo!(); //do this so compiler doesn't complain
}
//     // Input is Share {a,b}, in the notation below we have:
//     // Party0: a=d0, b=d2
//     // Party1: a=d1, b=d0
//     // Party2: a=d2, b=d1

//     // Get party number
//     let role = session.own_role().index();
//     // Depending on the role, do different stuff
//     match role {
//         0 => {
//             //Generate r2 prf key, keep the r0 key for later
//             let (r_prime, _) = session.prf.gen_rands::<RingElement<u32>>().clone(); //_ is r2
//             let r2 = RingElement(0);

//             // Send d2+r2 to party 1
//             let d2r2 = x.b + r2;
//             session
//                 .network_session
//                 .send_next(u32::new_network_element(d2r2))
//                 .await?;

//             // Receive d1+r1 from party 1
//             let d1r1 = match session.network_session.receive_next().await {
//                 Ok(v) => u32::into_vec(v),
//                 Err(e) => Err(eyre!("FSS: Party 0 cannot receive d1+r1 from party 1: {e}")),
//             }?;

//             // Receive my fss_key from dealer
//             let k_fss_0_vec = match session.network_session.receive_prev().await {
//                 Ok(v) => u32::into_vec(v),
//                 Err(e) => Err(eyre!("Party 1 cannot receive my fss key from dealer {e}")),
//             }?;

//             // Extract the Vec<RingElement<u32>> into Vec<u32>
//             let k_fss_0_single: Vec<u32> = RingElement::<u32>::convert_vec(k_fss_0_vec);

//             // Deserialize to find original IcShare
//             let k_fss_0_icshare: IcShare = IcShare::deserialize(&k_fss_0_single)?;

//             // we need this to handle signed numbers, if input is unsigned no need to add N/2
//             let n_falf_u32 = 1u32 << 31;

//             // reconstruct d+r [recall x.a=d0], d1r1 is a vector with 1 element because of the network message
//             let d_plus_r: RingElement<u32> = d1r1[0] + d2r2 + x.a + RingElement(n_falf_u32); // this should be wrapping addition, implemented by RingElement

//             let n_half = InG::from(n_falf_u32);

//             // make the interval so that we return 1 when MSB == 1
//             // this is (our number + n/2 ) % n, modulo is handled by U32Group
//             let p = InG::from(1u32 << 31) + n_half;
//             let q = InG::from(u32::MAX) + n_half; // modulo is handled by U32Group

//             let keys: Vec<[u8; 16]> = vec![[0u8; 16]; 4];
//             let prg =
//                 Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&std::array::from_fn(|i| &keys[i]));
//             let icf = Icf::new(p, q, prg); //

//             let timer_eval = start_timer!(|| "Eval timer");
//             let f_x_0 = icf.eval(
//                 false,
//                 &k_fss_0_icshare,
//                 fss_rs::group::int::U32Group(d_plus_r.0),
//             ); // evaluating f(x) for party 0// this should return 128 bits
//             end_timer!(timer_eval);

//             //Send f_x_0+r' to party 2 -- but we need to do some conversions first:
//             // Convert f_x_0 from ByteGroup<16> to RingElement<u128>
//             let f_x_0_ring_128: RingElement<u128> =
//                 RingElement::<u128>(u128::from_le_bytes(f_x_0.0));

//             // Add it to r_prime
//             let f_x_0_plus_r_prime = RingElement::<u128>(u128::from(r_prime.0)) ^ f_x_0_ring_128;

//             // We only need to send the LSb of f(x)_0+r' to the dealer and party 1
//             let bit_of_share_0 = RingElement(Bit::new((f_x_0_plus_r_prime.0 & 1) != 0));

//             session // send to party 1
//                 .network_session
//                 .send_next(NetworkValue::RingElementBit(bit_of_share_0))
//                 .await?;

//             session // send to the dealer (party 2)
//                 .network_session
//                 .send_prev(NetworkValue::RingElementBit(bit_of_share_0))
//                 .await?;

//             // Receive Bit of share of party 1
//             let bit_of_share_1: RingElement<Bit> =
//                 match session.network_session.receive_next().await {
//                     Ok(v) => match v {
//                         NetworkValue::RingElementBit(b) => Ok(b),
//                         other => Err(eyre!("expected RingElementBit, got {other:?}")),
//                     },
//                     Err(e) => Err(eyre!(
//                         "Party 1 cannot receive bit from the other party: {e}"
//                     )),
//                 }?;

//             Ok(Share::new(bit_of_share_0, bit_of_share_1))
//         }
//         1 => {
//             // Generate (my) r1 prf key, keep the r0 key for later
//             let (_, r_prime) = session.prf.gen_rands::<RingElement<u32>>().clone(); //_ is r1
//             let r1 = RingElement(0);

//             // Send d1+r1 to party 0
//             let d1r1 = x.a + r1;
//             session
//                 .network_session
//                 .send_prev(u32::new_network_element(d1r1))
//                 .await?;

//             // Receive d2+r2 from party 0
//             let d2r2 = match session.network_session.receive_prev().await {
//                 Ok(v) => u32::into_vec(v),
//                 Err(e) => Err(eyre!("FSS: Party 1 cannot receive d2+r2 from party 0: {e}")),
//             }?;

//             //Receive the fss key from the dealer (party 2)
//             let k_fss_1_vec = match session.network_session.receive_next().await {
//                 Ok(v) => u32::into_vec(v),
//                 Err(e) => Err(eyre!("Party 1 cannot receive my fss key from dealer {e}")),
//             }?;

//             // Extract the Vec<RingElement<u32>> into Vec<u32>
//             let k_fss_1_single: Vec<u32> = RingElement::<u32>::convert_vec(k_fss_1_vec);

//             // Deserialize to find original IcShare
//             let k_fss_1_icshare: IcShare = IcShare::deserialize(&k_fss_1_single)?;

//             // we need this to handle signed numbers, if input is unsigned no need to add N/2
//             let n_falf_u32 = 1u32 << 31;

//             // reconstruct d+r, recall d0=x.b
//             let d_plus_r = d1r1 + d2r2[0] + x.b + RingElement(n_falf_u32);

//             let n_half = InG::from(n_falf_u32);
//             // make the interval so that we return 1 when MSB == 1
//             // this is (our number + n/2 ) % n, modulo is handled by U32Group
//             let p = InG::from(1u32 << 31) + n_half;
//             let q = InG::from(u32::MAX) + n_half; // modulo is handled by U32Group

//             let keys: Vec<[u8; 16]> = vec![[0u8; 16]; 4];
//             let prg =
//                 Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&std::array::from_fn(|i| &keys[i]));
//             let icf = Icf::new(p, q, prg);
//             let f_x_1 = icf.eval(
//                 true,
//                 &k_fss_1_icshare,
//                 fss_rs::group::int::U32Group(d_plus_r.0),
//             ); // evaluating f(x) for party 1 this should return 128 bits

//             //Send f_x_1-r' to party 2, we need to do some conversions first
//             // Convert f_x_1 from ByteGroup<16> to RingElement<u128>
//             let f_x_1_ring_128: RingElement<u128> =
//                 RingElement::<u128>(u128::from_le_bytes(f_x_1.0));
//             // println!("Party 1, evaluated f(x)_1 u128 is {f_x_1_ring_128:?}\n    \n  {f_x_1:?}");

//             // Do t1 = f_x_1 - r_prime
//             let f_x_1_minus_r_prime = f_x_1_ring_128 ^ RingElement::<u128>(u128::from(r_prime.0));

//             // We only need to send the LSb of f(x)_0+r' to the dealer and party 1
//             let bit_of_share_1 = RingElement(Bit::new((f_x_1_minus_r_prime.0 & 1) != 0));

//             session // send to party 2 (dealer)
//                 .network_session
//                 .send_next(NetworkValue::RingElementBit(bit_of_share_1))
//                 .await?;

//             session // send to party 0
//                 .network_session
//                 .send_prev(NetworkValue::RingElementBit(bit_of_share_1))
//                 .await?;

//             // Receive bit of share from party 0
//             let bit_of_share_0: RingElement<Bit> =
//                 match session.network_session.receive_prev().await {
//                     Ok(v) => match v {
//                         NetworkValue::RingElementBit(b) => Ok(b),
//                         other => Err(eyre!("expected RingElementBit, got {other:?}")),
//                     },
//                     Err(e) => Err(eyre!(
//                         "Party 1 cannot receive bit from the other party: {e}"
//                     )),
//                 }?;

//             Ok(Share::new(bit_of_share_0, bit_of_share_1))
//         }
//         2 => {
//             let (_r2, _r1) = session.prf.gen_rands::<RingElement<u32>>().clone();
//             let r2 = RingElement(0);
//             let r1 = RingElement(0);

//             // Setting up the Interval Containment function
//             let keys: Vec<[u8; 16]> = vec![[0u8; 16]; 4];
//             let prg =
//                 Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&std::array::from_fn(|i| &keys[i]));

//             let r1_plus_r2_u32: u32 = (r1 + r2).convert();

//             // we need this to handle signed numbers, if input is unsigned no need to add N/2
//             let n_half = InG::from(1u32 << 31);

//             // make the interval so that we return 1 when MSB == 1
//             // this is (our number + n/2 ) % n, modulo is handled by U32Group
//             let p = InG::from(1u32 << 31) + n_half;
//             let q = InG::from(u32::MAX) + n_half; // modulo is handled by U32Group
//                                                   // println!("Interval is p={p:?}, q={q:?}");
//             let f = IntvFn {
//                 r_in: InG::from(r1_plus_r2_u32), //rin = r1+r2
//                 r_out: OutG::from(0u128),        // rout=0
//             };

//             // now we can call gen to generate the FSS keys for each party
//             let icf = Icf::new(p, q, prg);
//             let timer2 = start_timer!(|| "Gen timer");
//             let (k_fss_0_pre_ser, k_fss_1_pre_ser): (IcShare, IcShare) = {
//                 let mut rng = rand::thread_rng();
//                 icf.gen(f, &mut rng)
//             };
//             end_timer!(timer2); //     k_fss_0_pre_ser, k_fss_1_pre_ser
//             let timer = start_timer!(|| "serialize timer"); // );
//             let (k_fss_0, k_fss_1): (Vec<u32>, Vec<u32>) =
//                 (k_fss_0_pre_ser.serialize()?, k_fss_1_pre_ser.serialize()?);
//             end_timer!(timer);

//             // Convert them into Ve<RingElement<u32>> to send them through the network
//             let key0_vec = RingElement::<u32>::convert_vec_rev(k_fss_0);
//             let key1_vec = RingElement::<u32>::convert_vec_rev(k_fss_1);

//             // Send the FSS keys to parties 0 and 1, so they can do Eval
//             session
//                 .network_session
//                 .send_next(NetworkInt::new_network_vec(key0_vec))
//                 .await?; //next is party 0
//             session
//                 .network_session
//                 .send_prev(NetworkInt::new_network_vec(key1_vec))
//                 .await?; //previous is party 1

//             // Receive bit of share from party 0
//             let bit_of_share_0: RingElement<Bit> =
//                 match session.network_session.receive_next().await {
//                     Ok(v) => match v {
//                         NetworkValue::RingElementBit(b) => Ok(b),
//                         other => Err(eyre!("expected RingElementBit, got {other:?}")),
//                     },
//                     Err(e) => Err(eyre!(
//                         "Party 1 cannot receive bit from the other party: {e}"
//                     )),
//                 }?;

//             // Receive bit of share from party 1
//             let bit_of_share_1: RingElement<Bit> =
//                 match session.network_session.receive_prev().await {
//                     Ok(v) => match v {
//                         NetworkValue::RingElementBit(b) => Ok(b),
//                         other => Err(eyre!("expected RingElementBit, got {other:?}")),
//                     },
//                     Err(e) => Err(eyre!(
//                         "Party 1 cannot receive bit from the other party: {e}"
//                     )),
//                 }?;
//             Ok(Share::new(bit_of_share_0, bit_of_share_1))
//         }
//         _ => {
//             // this is not a valid party number
//             Err(eyre!("Party no is invalid for FSS: "))
//         }
//     }
// }

/// Batched version of the function above
/// Instead of handling one request at a time, get a batch of size ???
/// Main differences: each party
async fn add_3_get_msb_fss_batch(
    session: &mut Session,
    x: &[Share<u32>],
) -> Result<Vec<Share<Bit>>, Error>
where
    Standard: Distribution<u32>,
{
    todo!("Implement the batch version of add_3_get_msb_fss_batch"); // do this for now so we don't have an issue with InG being u16 instead of u32
}
//     // Input is Share {a,b}, in the notation below we have:
//     // Party0: a=d0, b=d2
//     // Party1: a=d1, b=d0
//     // Party2: a=d2, b=d1

//     // Get party number
//     let role = session.own_role().index();
//     // Depending on the role, do different stuff
//     match role {
//         0 => {
//             //Generate all r2 prf key, keep the r0 keys for later
//             let batch_size = x.len();
//             // println!("party 0: Batch size is {}", batch_size);
//             let mut r_prime_keys = Vec::with_capacity(batch_size);
//             let mut d2r2_vec = Vec::with_capacity(batch_size);
//             for i in 0..batch_size {
//                 let (r_prime_temp, _) = session.prf.gen_rands::<RingElement<u32>>().clone(); //_ is r2
//                 r_prime_keys.push(RingElement::<u128>(u128::from(r_prime_temp.0))); //convet to this for later
//                 d2r2_vec.push(x[i].b + RingElement(0)); // change this to take the second thing gen_rands returns
//             }

//             // Send the vector of d2+r2 to party 1
//             let clone_d2r2_vec = d2r2_vec.clone();
//             session
//                 .network_session
//                 .send_next(u32::new_network_vec(clone_d2r2_vec))
//                 .await?;

//             // Receive d1+r1 from party 1
//             let d1r1 = match session.network_session.receive_next().await {
//                 Ok(v) => u32::into_vec(v),
//                 Err(e) => Err(eyre!("FSS: Party 0 cannot receive d1+r1 from party 1: {e}")),
//             }?;

//             // Receive batch_size number of fss keys from dealer
//             let k_fss_0_vec = match session.network_session.receive_prev().await {
//                 Ok(v) => u32::into_vec(v),
//                 Err(e) => Err(eyre!("Party 0 cannot receive my fss key from dealer {e}")),
//             }?;

//             // Set up the function for FSS
//             // we need this below to handle signed numbers, if input is unsigned no need to add N/2
//             let n_falf_u32 = 1u32 << 31;
//             let n_half = InG::from(n_falf_u32);
//             // make the interval so that we return 1 when MSB == 1
//             // this is (our number + n/2 ) % n, modulo is handled by U32Group
//             let p = InG::from(1u32 << 31) + n_half;
//             let q = InG::from(u32::MAX) + n_half; // modulo is handled by U32Group
//             let keys: Vec<[u8; 16]> = vec![[0u8; 16]; 4];
//             let mut f_x_0_bits = Vec::with_capacity(batch_size); // store all the eval results

//             // Deserialize each to find original IcShare and call eval
//             let key_words_fss_0: Vec<u32> = RingElement::<u32>::convert_vec(k_fss_0_vec); //need to un-flatten key vector
//             let mut offset: usize = 0;
//             for i in 0..batch_size {
//                 // // Need to "unflatten" to get batch_size number of fss keys
//                 let curr_key_byte_len = 1 + (key_words_fss_0[offset] as usize + 3) / 4; // offset index has the byte length, then find total u32s for this key

//                 // Get current key
//                 let k_fss_0_icshare: IcShare =
//                     IcShare::deserialize(&key_words_fss_0[offset..offset + curr_key_byte_len])?;
//                 offset += curr_key_byte_len; //update offset to point to next cell that contains size of next key

//                 // reconstruct the input d+r [recall x.a=d0] for each x[i]
//                 let d_plus_r: RingElement<u32> =
//                     d1r1[i] + d2r2_vec[i] + x[i].a + RingElement(n_falf_u32);
//                 // this should be wrapping addition, implemented by RingElement

//                 // Now we're ready to call eval
//                 let prg =
//                     Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&std::array::from_fn(|i| &keys[i]));
//                 let icf = Icf::new(p, q, prg);
//                 //Call eval & convert from from ByteGroup<16> to RingElement<u128>
//                 let temp_eval = RingElement::<u128>(u128::from_le_bytes(
//                     icf.eval(
//                         false,
//                         &k_fss_0_icshare,
//                         fss_rs::group::int::U32Group(d_plus_r.0),
//                     )
//                     .0,
//                 ));

//                 // Add the respective r_prime and add to the vector of results and take only the LSB
//                 // Make them RingElements so it's easy to send to network
//                 f_x_0_bits.push(RingElement(Bit::new(
//                     ((temp_eval ^ r_prime_keys[i]).0 & 1) != 0,
//                 )));
//             }

//             // Prepare them in a vector to send to dealer and next party
//             let f_0_res_network: Vec<NetworkValue> = f_x_0_bits
//                 .iter()
//                 .copied()
//                 .map(NetworkValue::RingElementBit)
//                 .collect();

//             let cloned_f_0_res_network = f_0_res_network.clone();
//             session // send to party 1
//                 .network_session
//                 .send_next(NetworkValue::vec_to_network(cloned_f_0_res_network))
//                 .await?;

//             session // send to the dealer (party 2)
//                 .network_session
//                 .send_prev(NetworkValue::vec_to_network(f_0_res_network))
//                 .await?;

//             // Receive Bits of share of party 1 --> this is a vec of network values
//             let f_x_1_bits_net = match session.network_session.receive_next().await {
//                 Ok(v) => NetworkValue::vec_from_network(v),
//                 Err(e) => return Err(eyre!("Party 0 cannot receive bit shares from party 1: {e}")),
//             }?;

//             // Convert Vec<NetworkValue> to Vec<RingElement<Bit>>
//             let f_x_1_bits: Vec<RingElement<Bit>> = f_x_1_bits_net
//                 .into_iter()
//                 .map(|nv| match nv {
//                     NetworkValue::RingElementBit(b) => Ok(b),
//                     other => Err(eyre!("expected RingElementBit, got {:?}", other)),
//                 })
//                 .collect::<Result<_, _>>()?;

//             // Return a vector of Share<Bit> where the a is from f_x_0_bits
//             // and the b is from f_x_1_bits
//             let shares: Vec<Share<Bit>> = f_x_0_bits
//                 .into_iter()
//                 .zip(f_x_1_bits)
//                 .map(|(a, b)| Share { a, b })
//                 .collect();
//             Ok(shares)
//         }
//         1 => {
//             let batch_size = x.len();
//             // eprintln!("party 1: Batch size is {}", batch_size);
//             std::io::stderr().flush().ok();
//             let mut r_prime_keys = Vec::with_capacity(batch_size);
//             let mut d1r1_vec = Vec::with_capacity(batch_size);
//             for i in 0..batch_size {
//                 let (_, r_prime_temp) = session.prf.gen_rands::<RingElement<u32>>().clone(); //_ is r1
//                 r_prime_keys.push(RingElement::<u128>(u128::from(r_prime_temp.0))); //convet to this for later
//                 d1r1_vec.push(x[i].a + RingElement(0)); // change this to take the first thing gen_rands returns
//             }

//             // Send the vector of d1+r1 to party 0
//             let cloned_d1r1_vec = d1r1_vec.clone();
//             session
//                 .network_session
//                 .send_prev(u32::new_network_vec(cloned_d1r1_vec))
//                 .await?;

//             // Receive d2+r2 from party 0
//             let d2r2_vec = match session.network_session.receive_prev().await {
//                 Ok(v) => u32::into_vec(v),
//                 Err(e) => Err(eyre!("FSS: Party 1 cannot receive d2+r2 from party 0: {e}")),
//             }?;

//             // Receive batch_size number of fss keys from dealer
//             let k_fss_1_vec = match session.network_session.receive_next().await {
//                 Ok(v) => u32::into_vec(v),
//                 Err(e) => Err(eyre!("Party 1 cannot receive my fss key from dealer {e}")),
//             }?;

//             // Set up the function for FSS
//             // we need this below to handle signed numbers, if input is unsigned no need to add N/2
//             let n_falf_u32 = 1u32 << 31;
//             let n_half = InG::from(n_falf_u32);
//             // make the interval so that we return 1 when MSB == 1
//             // this is (our number + n/2 ) % n, modulo is handled by U32Group
//             let p = InG::from(1u32 << 31) + n_half;
//             let q = InG::from(u32::MAX) + n_half; // modulo is handled by U32Group
//             let keys: Vec<[u8; 16]> = vec![[0u8; 16]; 4];
//             let mut f_x_1_bits = Vec::with_capacity(batch_size); // store all the eval results

//             // Deserialize each to find original IcShare and call eval
//             // Deserialize each to find original IcShare and call eval
//             let key_words_fss_1: Vec<u32> = RingElement::<u32>::convert_vec(k_fss_1_vec); //need to un-flatten key vector
//             let mut offset: usize = 0;
//             for i in 0..batch_size {
//                 // // Need to "unflatten" to get batch_size number of fss keys
//                 let curr_key_byte_len = 1 + (key_words_fss_1[offset] as usize + 3) / 4; // offset index has the byte length, then find total u32s for this key

//                 // Get current key
//                 let k_fss_1_icshare: IcShare =
//                     IcShare::deserialize(&key_words_fss_1[offset..offset + curr_key_byte_len])?;
//                 offset += curr_key_byte_len; //update offset to point to next cell that contains size of next key

//                 // reconstruct the input d+r [recall d0=x.b] for each x[i]
//                 let d_plus_r: RingElement<u32> =
//                     d1r1_vec[i] + d2r2_vec[i] + x[i].b + RingElement(n_falf_u32);
//                 // this should be wrapping addition, implemented by RingElement

//                 // Now we're ready to call eval
//                 let prg =
//                     Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&std::array::from_fn(|i| &keys[i]));
//                 let icf = Icf::new(p, q, prg);
//                 //Call eval & convert from from ByteGroup<16> to RingElement<u128>
//                 let temp_eval = RingElement::<u128>(u128::from_le_bytes(
//                     icf.eval(
//                         true,
//                         &k_fss_1_icshare,
//                         fss_rs::group::int::U32Group(d_plus_r.0),
//                     )
//                     .0,
//                 ));

//                 // Add the respective r_prime and add to the vector of results and take only the LSB
//                 // Make them RingElements so it's easy to send to network
//                 f_x_1_bits.push(RingElement(Bit::new(
//                     ((temp_eval ^ r_prime_keys[i]).0 & 1) != 0,
//                 )));
//             }

//             // Prepare them in a vector to send to dealer and next party
//             let f_1_res_network: Vec<NetworkValue> = f_x_1_bits
//                 .iter()
//                 .copied()
//                 .map(NetworkValue::RingElementBit)
//                 .collect();

//             let cloned_f_1_res_network = f_1_res_network.clone();
//             session // send to party 0
//                 .network_session
//                 .send_prev(NetworkValue::vec_to_network(cloned_f_1_res_network))
//                 .await?;

//             session // send to the dealer (party 2)
//                 .network_session
//                 .send_next(NetworkValue::vec_to_network(f_1_res_network))
//                 .await?;

//             // Receive Bits of share of party 0 --> this is a vec of network values
//             let f_x_0_bits_net = match session.network_session.receive_prev().await {
//                 Ok(v) => NetworkValue::vec_from_network(v),
//                 Err(e) => return Err(eyre!("Party 0 cannot receive bit shares from party 1: {e}")),
//             }?;

//             // Convert Vec<NetworkValue> to Vec<RingElement<Bit>>
//             let f_x_0_bits: Vec<RingElement<Bit>> = f_x_0_bits_net
//                 .into_iter()
//                 .map(|nv| match nv {
//                     NetworkValue::RingElementBit(b) => Ok(b),
//                     other => Err(eyre!("expected RingElementBit, got {:?}", other)),
//                 })
//                 .collect::<Result<_, _>>()?;

//             // Return a vector of Share<Bit> where the a is from f_x_0_bits
//             // and the b is from f_x_1_bits
//             let shares: Vec<Share<Bit>> = f_x_0_bits
//                 .into_iter()
//                 .zip(f_x_1_bits)
//                 .map(|(a, b)| Share { a, b })
//                 .collect();
//             Ok(shares)
//         }
//         2 => {
//             let batch_size = x.len();

//             // Setting up the Interval Containment function
//             // we need this to handle signed numbers, if input is unsigned no need to add N/2
//             let n_half = InG::from(1u32 << 31);
//             let keys: Vec<[u8; 16]> = vec![[0u8; 16]; 4];

//             // make the interval so that we return 1 when MSB == 1
//             // this is (our number + n/2 ) % n, modulo is handled by U32Group
//             let p = InG::from(1u32 << 31) + n_half;
//             let q = InG::from(u32::MAX) + n_half; // modulo is handled by U32Group
//                                                   // println!("Interval is p={p:?}, q={q:?}");

//             let mut k_fss_0_vec_flat = Vec::with_capacity(batch_size); // to store the fss keys
//             let mut k_fss_1_vec_flat = Vec::with_capacity(batch_size);
//             for _i in 0..batch_size {
//                 // Draw r1 + r2 (aka r_in)
//                 let (_r2, _r1) = session.prf.gen_rands::<RingElement<u32>>().clone();
//                 let r2 = RingElement(0);
//                 let r1 = RingElement(0);

//                 let r1_plus_r2_u32: u32 = (r1 + r2).convert();
//                 // Defining the function f using r_in
//                 let f = IntvFn {
//                     r_in: InG::from(r1_plus_r2_u32), //rin = r1+r2
//                     r_out: OutG::from(0u128),        // rout=0
//                 };
//                 // now we can call gen to generate the FSS keys for each party
//                 let prg =
//                     Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&std::array::from_fn(|i| &keys[i]));
//                 let icf = Icf::new(p, q, prg);
//                 let (k_fss_0_pre_ser, k_fss_1_pre_ser): (IcShare, IcShare) = {
//                     let mut rng = rand::thread_rng();
//                     icf.gen(f, &mut rng)
//                 };

//                 let temp_key0 = k_fss_0_pre_ser.serialize()?;
//                 k_fss_0_vec_flat.extend(RingElement::<u32>::convert_vec_rev(temp_key0.clone()));

//                 let temp_key1 = k_fss_1_pre_ser.serialize()?;
//                 k_fss_1_vec_flat.extend(RingElement::<u32>::convert_vec_rev(temp_key1.clone()));
//             }

//             // Send the flattened FSS keys to parties 0 and 1, so they can do Eval
//             session
//                 .network_session
//                 .send_next(NetworkInt::new_network_vec(k_fss_0_vec_flat))
//                 .await?; //next is party 0

//             session
//                 .network_session
//                 .send_prev(NetworkInt::new_network_vec(k_fss_1_vec_flat))
//                 .await?; //previous is party 1

//             // Receive bit of share from party 0
//             let f_x_0_bits_net = match session.network_session.receive_next().await {
//                 Ok(v) => NetworkValue::vec_from_network(v),
//                 Err(e) => return Err(eyre!("Party 0 cannot receive bit shares from party 1: {e}")),
//             }?;

//             // Convert Vec<NetworkValue> to Vec<RingElement<Bit>>
//             let f_x_0_bits: Vec<RingElement<Bit>> = f_x_0_bits_net
//                 .into_iter()
//                 .map(|nv| match nv {
//                     NetworkValue::RingElementBit(b) => Ok(b),
//                     other => Err(eyre!("expected RingElementBit, got {:?}", other)),
//                 })
//                 .collect::<Result<_, _>>()?;

//             // Receive Bits of share of party 1 --> this is a vec of network values
//             let f_x_1_bits_net = match session.network_session.receive_prev().await {
//                 Ok(v) => NetworkValue::vec_from_network(v),
//                 Err(e) => return Err(eyre!("Party 0 cannot receive bit shares from party 1: {e}")),
//             }?;

//             // Convert Vec<NetworkValue> to Vec<RingElement<Bit>>
//             let f_x_1_bits: Vec<RingElement<Bit>> = f_x_1_bits_net
//                 .into_iter()
//                 .map(|nv| match nv {
//                     NetworkValue::RingElementBit(b) => Ok(b),
//                     other => Err(eyre!("expected RingElementBit, got {:?}", other)),
//                 })
//                 .collect::<Result<_, _>>()?;

//             // Return a vector of Share<Bit> where the a is from f_x_0_bits
//             // and the b is from f_x_1_bits
//             let shares: Vec<Share<Bit>> = f_x_0_bits
//                 .into_iter()
//                 .zip(f_x_1_bits)
//                 .map(|(a, b)| Share { a, b })
//                 .collect();
//             Ok(shares)
//         }
//         _ => {
//             // this is not a valid party number
//             Err(eyre!("Party no is invalid for FSS."))
//         }
//     }
// }

/// u16 version of the function above
/// just to check
/// Batched version of the function above
/// Instead of handling one request at a time, get a batch of size ???
/// Main differences: each party
async fn add_3_get_msb_fss_batch_u16(
    session: &mut Session,
    x: &[Share<u16>],
) -> Result<Vec<Share<Bit>>, Error>
where
    Standard: Distribution<u32>,
{
    // Input is Share {a,b}, in the notation below we have:
    // Party0: a=d0, b=d2
    // Party1: a=d1, b=d0
    // Party2: a=d2, b=d1

    // Get party number
    let role = session.own_role().index();
    // Depending on the role, do different stuff
    match role {
        0 => {
            //Generate all r2 prf key, keep the r0 keys for later
            let batch_size = x.len();
            // println!("party 0: Batch size is {}", batch_size);
            let mut r_prime_keys = Vec::with_capacity(batch_size);
            let mut d2r2_vec = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let (r_prime_temp, _) = session.prf.gen_rands::<RingElement<u16>>().clone(); //_ is r2
                r_prime_keys.push(RingElement::<u128>(u128::from(r_prime_temp.0))); //convet to this for later
                d2r2_vec.push(x[i].b + RingElement(0)); // change this to take the second thing gen_rands returns
            }

            // Send the vector of d2+r2 to party 1
            let clone_d2r2_vec = d2r2_vec.clone();
            session
                .network_session
                .send_next(u16::new_network_vec(clone_d2r2_vec))
                .await?;

            // Receive d1+r1 from party 1
            let d1r1 = match session.network_session.receive_next().await {
                Ok(v) => u16::into_vec(v),
                Err(e) => Err(eyre!("FSS: Party 0 cannot receive d1+r1 from party 1: {e}")),
            }?;

            // Receive batch_size number of fss keys from dealer, the key is an ICShare that is serialized to u32
            let k_fss_0_vec = match session.network_session.receive_prev().await {
                Ok(v) => u32::into_vec(v),
                Err(e) => Err(eyre!("Party 0 cannot receive my fss key from dealer {e}")),
            }?;

            // Set up the function for FSS
            // we need this below to handle signed numbers, if input is unsigned no need to add N/2
            let n_half_u32 = 1u16 << 15;
            let n_half = InG::from(n_half_u32);
            // make the interval so that we return 1 when MSB == 1
            // this is (our number + n/2 ) % n, modulo is handled by U16Group
            let p = InG::from(1u16 << 15) + n_half;
            let q = InG::from(u16::MAX) + n_half; // modulo is handled by U16Group
            let keys: Vec<[u8; 16]> = vec![[0u8; 16]; 4]; // !!! do math again for this? is this correct size?
            let mut f_x_0_bits = Vec::with_capacity(batch_size); // store all the eval results

            // Deserialize each to find original IcShare and call eval
            let key_words_fss_0: Vec<u32> = RingElement::<u32>::convert_vec(k_fss_0_vec); //need to un-flatten key vector
            let mut offset: usize = 0;
            for i in 0..batch_size {
                // Need to "unflatten" to get batch_size number of fss keys
                let curr_key_byte_len = 1 + (key_words_fss_0[offset] as usize + 3) / 4; // offset index has the byte length, then find total u32s for this key

                // Get current key (ICShare)
                let k_fss_0_icshare: IcShare =
                    IcShare::deserialize(&key_words_fss_0[offset..offset + curr_key_byte_len])?;
                offset += curr_key_byte_len; //update offset to point to next cell that contains size of next key

                // reconstruct the input d+r [recall x.a=d0] for each x[i]
                let d_plus_r: RingElement<u16> =
                    d1r1[i] + d2r2_vec[i] + x[i].a + RingElement(n_half_u32);
                // this should be wrapping addition, implemented by RingElement

                // Now we're ready to call eval
                let prg_start = std::time::Instant::now();
                let prg =
                    Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&std::array::from_fn(|i| &keys[i]));
                let prg_elapsed_micros = (prg_start.elapsed().as_secs_f64() * 1_000_000.0) as u64;
                record_prg_keygen_micros(prg_elapsed_micros);

                let icf = Icf::new(p, q, prg);
                //Call eval & convert from from ByteGroup<16> to RingElement<u128>
                let icf_start = std::time::Instant::now();
                let eval_result = icf.eval(
                    false,
                    &k_fss_0_icshare,
                    fss_rs::group::int::U16Group(d_plus_r.0),
                );
                let icf_elapsed_micros = (icf_start.elapsed().as_secs_f64() * 1_000_000.0) as u64;
                record_icf_eval_micros(icf_elapsed_micros);

                let temp_eval = RingElement::<u128>(u128::from_le_bytes(eval_result.0));

                // Add the respective r_prime and add to the vector of results and take only the LSB
                // Make them RingElements so it's easy to send to network
                f_x_0_bits.push(RingElement(Bit::new(
                    ((temp_eval ^ r_prime_keys[i]).0 & 1) != 0,
                )));
            }

            // Prepare them in a vector to send to dealer and next party
            let f_0_res_network: Vec<NetworkValue> = f_x_0_bits
                .iter()
                .copied()
                .map(NetworkValue::RingElementBit)
                .collect();

            let cloned_f_0_res_network = f_0_res_network.clone();
            let net_start = std::time::Instant::now();
            let bytes_next =
                (cloned_f_0_res_network.len() * std::mem::size_of::<NetworkValue>()) as u64;
            session // send to party 1
                .network_session
                .send_next(NetworkValue::vec_to_network(cloned_f_0_res_network))
                .await?;

            let bytes_prev = (f_0_res_network.len() * std::mem::size_of::<NetworkValue>()) as u64;
            session // send to the dealer (party 2)
                .network_session
                .send_prev(NetworkValue::vec_to_network(f_0_res_network))
                .await?;

            // Receive Bits of share of party 1 --> this is a vec of network values
            let f_x_1_bits_net = match session.network_session.receive_next().await {
                Ok(v) => NetworkValue::vec_from_network(v),
                Err(e) => return Err(eyre!("Party 0 cannot receive bit shares from party 1: {e}")),
            }?;
            let bytes_recv = (f_x_1_bits_net.len() * std::mem::size_of::<NetworkValue>()) as u64;
            record_network_bytes_sent(bytes_recv);
            let net_elapsed_micros = (net_start.elapsed().as_secs_f64() * 1_000_000.0) as u64;
            record_network_micros(net_elapsed_micros);
            record_network_bytes_sent(bytes_next + bytes_prev);

            // Convert Vec<NetworkValue> to Vec<RingElement<Bit>>
            let f_x_1_bits: Vec<RingElement<Bit>> = f_x_1_bits_net
                .into_iter()
                .map(|nv| match nv {
                    NetworkValue::RingElementBit(b) => Ok(b),
                    other => Err(eyre!("expected RingElementBit, got {:?}", other)),
                })
                .collect::<Result<_, _>>()?;

            // Return a vector of Share<Bit> where the a is from f_x_0_bits
            // and the b is from f_x_1_bits
            let shares: Vec<Share<Bit>> = f_x_0_bits
                .into_iter()
                .zip(f_x_1_bits)
                .map(|(a, b)| Share { a, b })
                .collect();
            Ok(shares)
        }
        1 => {
            let batch_size = x.len();
            let mut r_prime_keys = Vec::with_capacity(batch_size);
            let mut d1r1_vec = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let (_, r_prime_temp) = session.prf.gen_rands::<RingElement<u16>>().clone(); //_ is r1
                r_prime_keys.push(RingElement::<u128>(u128::from(r_prime_temp.0))); //convet to this for later
                d1r1_vec.push(x[i].a + RingElement(0)); // change this to take the first thing gen_rands returns
            }

            // Send the vector of d1+r1 to party 0
            let cloned_d1r1_vec = d1r1_vec.clone();
            session
                .network_session
                .send_prev(u16::new_network_vec(cloned_d1r1_vec))
                .await?;

            // Receive d2+r2 from party 0
            let d2r2_vec = match session.network_session.receive_prev().await {
                Ok(v) => u16::into_vec(v),
                Err(e) => Err(eyre!("FSS: Party 1 cannot receive d2+r2 from party 0: {e}")),
            }?;

            // Receive batch_size number of fss keys from dealer
            let k_fss_1_vec = match session.network_session.receive_next().await {
                Ok(v) => u32::into_vec(v),
                Err(e) => Err(eyre!("Party 1 cannot receive my fss key from dealer {e}")),
            }?;

            // Set up the function for FSS
            // we need this below to handle signed numbers, if input is unsigned no need to add N/2
            let n_half_u32 = 1u16 << 15;
            let n_half = InG::from(n_half_u32);
            // make the interval so that we return 1 when MSB == 1
            // this is (our number + n/2 ) % n, modulo is handled by U16Group
            let p = InG::from(1u16 << 15) + n_half;
            let q = InG::from(u16::MAX) + n_half; // modulo is handled by U16Group
            let keys: Vec<[u8; 16]> = vec![[0u8; 16]; 4];
            let mut f_x_1_bits = Vec::with_capacity(batch_size); // store all the eval results

            // Deserialize each to find original IcShare and call eval
            // Deserialize each to find original IcShare and call eval
            let key_words_fss_1: Vec<u32> = RingElement::<u32>::convert_vec(k_fss_1_vec); //need to un-flatten key vector
            let mut offset: usize = 0;
            for i in 0..batch_size {
                // // Need to "unflatten" to get batch_size number of fss keys
                let curr_key_byte_len = 1 + (key_words_fss_1[offset] as usize + 3) / 4; // offset index has the byte length, then find total u32s for this key

                // Get current key
                let k_fss_1_icshare: IcShare =
                    IcShare::deserialize(&key_words_fss_1[offset..offset + curr_key_byte_len])?;
                offset += curr_key_byte_len; //update offset to point to next cell that contains size of next key

                // reconstruct the input d+r [recall d0=x.b] for each x[i]
                let d_plus_r: RingElement<u16> =
                    d1r1_vec[i] + d2r2_vec[i] + x[i].b + RingElement(n_half_u32);
                // this should be wrapping addition, implemented by RingElement

                // Now we're ready to call eval
                let prg_start = std::time::Instant::now();
                let prg =
                    Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&std::array::from_fn(|i| &keys[i]));
                let prg_elapsed_micros = (prg_start.elapsed().as_secs_f64() * 1_000_000.0) as u64;
                record_prg_keygen_micros(prg_elapsed_micros);

                let icf = Icf::new(p, q, prg);
                //Call eval & convert from from ByteGroup<16> to RingElement<u128>
                let icf_start = std::time::Instant::now();
                let eval_result = icf.eval(
                    true,
                    &k_fss_1_icshare,
                    fss_rs::group::int::U16Group(d_plus_r.0),
                );
                let icf_elapsed_micros = (icf_start.elapsed().as_secs_f64() * 1_000_000.0) as u64;
                record_icf_eval_micros(icf_elapsed_micros);

                let temp_eval = RingElement::<u128>(u128::from_le_bytes(eval_result.0));

                // Add the respective r_prime and add to the vector of results and take only the LSB
                // Make them RingElements so it's easy to send to network
                f_x_1_bits.push(RingElement(Bit::new(
                    ((temp_eval ^ r_prime_keys[i]).0 & 1) != 0,
                )));
            }

            // Prepare them in a vector to send to dealer and next party
            let f_1_res_network: Vec<NetworkValue> = f_x_1_bits
                .iter()
                .copied()
                .map(NetworkValue::RingElementBit)
                .collect();

            let cloned_f_1_res_network = f_1_res_network.clone();
            let net_start = std::time::Instant::now();
            let bytes_prev =
                (cloned_f_1_res_network.len() * std::mem::size_of::<NetworkValue>()) as u64;
            session // send to party 0
                .network_session
                .send_prev(NetworkValue::vec_to_network(cloned_f_1_res_network))
                .await?;

            let bytes_next = (f_1_res_network.len() * std::mem::size_of::<NetworkValue>()) as u64;
            session // send to the dealer (party 2)
                .network_session
                .send_next(NetworkValue::vec_to_network(f_1_res_network))
                .await?;

            // Receive Bits of share of party 0 --> this is a vec of network values
            let f_x_0_bits_net = match session.network_session.receive_prev().await {
                Ok(v) => NetworkValue::vec_from_network(v),
                Err(e) => return Err(eyre!("Party 0 cannot receive bit shares from party 1: {e}")),
            }?;
            let bytes_recv = (f_x_0_bits_net.len() * std::mem::size_of::<NetworkValue>()) as u64;
            record_network_bytes_sent(bytes_recv);
            let net_elapsed_micros = (net_start.elapsed().as_secs_f64() * 1_000_000.0) as u64;
            record_network_micros(net_elapsed_micros);
            record_network_bytes_sent(bytes_prev + bytes_next);

            // Convert Vec<NetworkValue> to Vec<RingElement<Bit>>
            let f_x_0_bits: Vec<RingElement<Bit>> = f_x_0_bits_net
                .into_iter()
                .map(|nv| match nv {
                    NetworkValue::RingElementBit(b) => Ok(b),
                    other => Err(eyre!("expected RingElementBit, got {:?}", other)),
                })
                .collect::<Result<_, _>>()?;

            // Return a vector of Share<Bit> where the a is from f_x_0_bits
            // and the b is from f_x_1_bits
            let shares: Vec<Share<Bit>> = f_x_0_bits
                .into_iter()
                .zip(f_x_1_bits)
                .map(|(a, b)| Share { a, b })
                .collect();
            Ok(shares)
        }
        2 => {
            let batch_size = x.len();

            // Setting up the Interval Containment function
            // we need this to handle signed numbers, if input is unsigned no need to add N/2
            let n_half = InG::from(1u16 << 15);
            let keys: Vec<[u8; 16]> = vec![[0u8; 16]; 4];

            // make the interval so that we return 1 when MSB == 1
            // this is (our number + n/2 ) % n, modulo is handled by U16Group
            let p = InG::from(1u16 << 15) + n_half;
            let q = InG::from(u16::MAX) + n_half; // modulo is handled by U32Group

            let mut k_fss_0_vec_flat = Vec::with_capacity(batch_size); // to store the fss keys
            let mut k_fss_1_vec_flat = Vec::with_capacity(batch_size);
            for _i in 0..batch_size {
                // Draw r1 + r2 (aka r_in)
                let (_r2, _r1) = session.prf.gen_rands::<RingElement<u16>>().clone();
                let r2 = RingElement(0);
                let r1 = RingElement(0);

                let r1_plus_r2_u32: u16 = (r1 + r2).convert();
                // Defining the function f using r_in
                let f = IntvFn {
                    r_in: InG::from(r1_plus_r2_u32), //rin = r1+r2
                    r_out: OutG::from(0u128),        // rout=0
                };
                // now we can call gen to generate the FSS keys for each party
                let prg =
                    Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&std::array::from_fn(|i| &keys[i]));
                let icf = Icf::new(p, q, prg);
                let (k_fss_0_pre_ser, k_fss_1_pre_ser): (IcShare, IcShare) = {
                    let mut rng = rand::thread_rng();
                    icf.gen(f, &mut rng)
                };

                // Serialize the ICShare into u32 (no need to use u16 here)
                let temp_key0 = k_fss_0_pre_ser.serialize()?;
                k_fss_0_vec_flat.extend(RingElement::<u32>::convert_vec_rev(temp_key0.clone()));

                let temp_key1 = k_fss_1_pre_ser.serialize()?;
                k_fss_1_vec_flat.extend(RingElement::<u32>::convert_vec_rev(temp_key1.clone()));
            }

            // Send the flattened FSS keys to parties 0 and 1, so they can do Eval
            let net_start = std::time::Instant::now();
            let bytes_next = (k_fss_0_vec_flat.len() * std::mem::size_of::<u32>()) as u64;
            session
                .network_session
                .send_next(NetworkInt::new_network_vec(k_fss_0_vec_flat))
                .await?; //next is party 0

            let bytes_prev = (k_fss_1_vec_flat.len() * std::mem::size_of::<u32>()) as u64;
            session
                .network_session
                .send_prev(NetworkInt::new_network_vec(k_fss_1_vec_flat))
                .await?; //previous is party 1

            // Receive bit of share from party 0
            let f_x_0_bits_net = match session.network_session.receive_next().await {
                Ok(v) => NetworkValue::vec_from_network(v),
                Err(e) => return Err(eyre!("Party 0 cannot receive bit shares from party 1: {e}")),
            }?;
            let bytes_recv_0 = (f_x_0_bits_net.len() * std::mem::size_of::<NetworkValue>()) as u64;
            record_network_bytes_sent(bytes_recv_0);

            // Convert Vec<NetworkValue> to Vec<RingElement<Bit>>
            let f_x_0_bits: Vec<RingElement<Bit>> = f_x_0_bits_net
                .into_iter()
                .map(|nv| match nv {
                    NetworkValue::RingElementBit(b) => Ok(b),
                    other => Err(eyre!("expected RingElementBit, got {:?}", other)),
                })
                .collect::<Result<_, _>>()?;

            // Receive Bits of share of party 1 --> this is a vec of network values
            let f_x_1_bits_net = match session.network_session.receive_prev().await {
                Ok(v) => NetworkValue::vec_from_network(v),
                Err(e) => return Err(eyre!("Party 0 cannot receive bit shares from party 1: {e}")),
            }?;
            let bytes_recv_1 = (f_x_1_bits_net.len() * std::mem::size_of::<NetworkValue>()) as u64;
            record_network_bytes_sent(bytes_recv_1);
            let net_elapsed_micros = (net_start.elapsed().as_secs_f64() * 1_000_000.0) as u64;
            record_network_micros(net_elapsed_micros);
            record_network_bytes_sent(bytes_next + bytes_prev);

            // Convert Vec<NetworkValue> to Vec<RingElement<Bit>>
            let f_x_1_bits: Vec<RingElement<Bit>> = f_x_1_bits_net
                .into_iter()
                .map(|nv| match nv {
                    NetworkValue::RingElementBit(b) => Ok(b),
                    other => Err(eyre!("expected RingElementBit, got {:?}", other)),
                })
                .collect::<Result<_, _>>()?;

            // Return a vector of Share<Bit> where the a is from f_x_0_bits
            // and the b is from f_x_1_bits
            let shares: Vec<Share<Bit>> = f_x_0_bits
                .into_iter()
                .zip(f_x_1_bits)
                .map(|(a, b)| Share { a, b })
                .collect();
            Ok(shares)
        }
        _ => {
            // this is not a valid party number
            Err(eyre!("Party no is invalid for FSS."))
        }
    }
}

async fn add_3_get_msb_fss_batch_parallel_u16(
    session: &mut Session,
    x: &[Share<u16>],
) -> Result<Vec<Share<Bit>>, Error>
where
    Standard: Distribution<u32>,
{
    // Input is Share {a,b}, in the notation below we have:
    // Party0: a=d0, b=d2
    // Party1: a=d1, b=d0
    // Party2: a=d2, b=d1

    // Get party number
    let role = session.own_role().index();
    // Depending on the role, do different stuff
    match role {
        0 => {
            //Generate all r2 prf key, keep the r0 keys for later
            let batch_size = x.len();
            // println!("party 0: Batch size is {}", batch_size);
            let mut r_prime_keys = Vec::with_capacity(batch_size);
            let mut d2r2_vec = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let (r_prime_temp, _) = session.prf.gen_rands::<RingElement<u16>>().clone(); //_ is r2
                r_prime_keys.push(RingElement::<u128>(u128::from(r_prime_temp.0))); //convet to this for later
                d2r2_vec.push(x[i].b + RingElement(0)); // change this to take the second thing gen_rands returns
            }

            // Send the vector of d2+r2 to party 1
            let clone_d2r2_vec = d2r2_vec.clone();
            session
                .network_session
                .send_next(u16::new_network_vec(clone_d2r2_vec))
                .await?;

            // Receive d1+r1 from party 1
            let d1r1 = match session.network_session.receive_next().await {
                Ok(v) => u16::into_vec(v),
                Err(e) => Err(eyre!("FSS: Party 0 cannot receive d1+r1 from party 1: {e}")),
            }?;

            // Receive batch_size number of fss keys from dealer, the key is an ICShare that is serialized to u32
            let k_fss_0_vec = match session.network_session.receive_prev().await {
                Ok(v) => u32::into_vec(v),
                Err(e) => Err(eyre!("Party 0 cannot receive my fss key from dealer {e}")),
            }?;

            // Set up the function for FSS
            // we need this below to handle signed numbers, if input is unsigned no need to add N/2
            let n_half_u32 = 1u16 << 15;
            let n_half = InG::from(n_half_u32);
            // make the interval so that we return 1 when MSB == 1
            // this is (our number + n/2 ) % n, modulo is handled by U16Group
            let p = InG::from(1u16 << 15) + n_half;
            let q = InG::from(u16::MAX) + n_half; // modulo is handled by U16Group
            let keys: Vec<[u8; 16]> = vec![[0u8; 16]; 4]; // !!! do math again for this? is this correct size?
            let mut f_x_0_bits = Vec::with_capacity(batch_size); // store all the eval results

            // Deserialize each to find original IcShare and call eval
            let key_words_fss_0: Vec<u32> = RingElement::<u32>::convert_vec(k_fss_0_vec); //need to un-flatten key vector
            let mut offset: usize = 0;
            for i in 0..batch_size {
                // Need to "unflatten" to get batch_size number of fss keys
                let curr_key_byte_len = 1 + (key_words_fss_0[offset] as usize + 3) / 4; // offset index has the byte length, then find total u32s for this key

                // Get current key (ICShare)
                let k_fss_0_icshare: IcShare =
                    IcShare::deserialize(&key_words_fss_0[offset..offset + curr_key_byte_len])?;
                offset += curr_key_byte_len; //update offset to point to next cell that contains size of next key

                // reconstruct the input d+r [recall x.a=d0] for each x[i]
                let d_plus_r: RingElement<u16> =
                    d1r1[i] + d2r2_vec[i] + x[i].a + RingElement(n_half_u32);
                // this should be wrapping addition, implemented by RingElement

                // Now we're ready to call eval
                let prg =
                    Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&std::array::from_fn(|i| &keys[i]));
                let icf = Icf::new(p, q, prg);
                //Call eval & convert from from ByteGroup<16> to RingElement<u128>
                let temp_eval = RingElement::<u128>(u128::from_le_bytes(
                    icf.eval(
                        false,
                        &k_fss_0_icshare,
                        fss_rs::group::int::U16Group(d_plus_r.0),
                    )
                    .0,
                ));

                // Add the respective r_prime and add to the vector of results and take only the LSB
                // Make them RingElements so it's easy to send to network
                f_x_0_bits.push(RingElement(Bit::new(
                    ((temp_eval ^ r_prime_keys[i]).0 & 1) != 0,
                )));
            }

            // Prepare them in a vector to send to dealer and next party
            let f_0_res_network: Vec<NetworkValue> = f_x_0_bits
                .iter()
                .copied()
                .map(NetworkValue::RingElementBit)
                .collect();

            let cloned_f_0_res_network = f_0_res_network.clone();
            session // send to party 1
                .network_session
                .send_next(NetworkValue::vec_to_network(cloned_f_0_res_network))
                .await?;

            session // send to the dealer (party 2)
                .network_session
                .send_prev(NetworkValue::vec_to_network(f_0_res_network))
                .await?;

            // Receive Bits of share of party 1 --> this is a vec of network values
            let f_x_1_bits_net = match session.network_session.receive_next().await {
                Ok(v) => NetworkValue::vec_from_network(v),
                Err(e) => return Err(eyre!("Party 0 cannot receive bit shares from party 1: {e}")),
            }?;

            // Convert Vec<NetworkValue> to Vec<RingElement<Bit>>
            let f_x_1_bits: Vec<RingElement<Bit>> = f_x_1_bits_net
                .into_iter()
                .map(|nv| match nv {
                    NetworkValue::RingElementBit(b) => Ok(b),
                    other => Err(eyre!("expected RingElementBit, got {:?}", other)),
                })
                .collect::<Result<_, _>>()?;

            // Return a vector of Share<Bit> where the a is from f_x_0_bits
            // and the b is from f_x_1_bits
            let shares: Vec<Share<Bit>> = f_x_0_bits
                .into_iter()
                .zip(f_x_1_bits)
                .map(|(a, b)| Share { a, b })
                .collect();
            Ok(shares)
        }
        1 => {
            let batch_size = x.len();
            let mut r_prime_keys = Vec::with_capacity(batch_size);
            let mut d1r1_vec = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let (_, r_prime_temp) = session.prf.gen_rands::<RingElement<u16>>().clone(); //_ is r1
                r_prime_keys.push(RingElement::<u128>(u128::from(r_prime_temp.0))); //convet to this for later
                d1r1_vec.push(x[i].a + RingElement(0)); // change this to take the first thing gen_rands returns
            }

            // Send the vector of d1+r1 to party 0
            let cloned_d1r1_vec = d1r1_vec.clone();
            session
                .network_session
                .send_prev(u16::new_network_vec(cloned_d1r1_vec))
                .await?;

            // Receive d2+r2 from party 0
            let d2r2_vec = match session.network_session.receive_prev().await {
                Ok(v) => u16::into_vec(v),
                Err(e) => Err(eyre!("FSS: Party 1 cannot receive d2+r2 from party 0: {e}")),
            }?;

            // Receive batch_size number of fss keys from dealer
            let k_fss_1_vec = match session.network_session.receive_next().await {
                Ok(v) => u32::into_vec(v),
                Err(e) => Err(eyre!("Party 1 cannot receive my fss key from dealer {e}")),
            }?;

            // Set up the function for FSS
            // we need this below to handle signed numbers, if input is unsigned no need to add N/2
            let n_half_u32 = 1u16 << 15;
            let n_half = InG::from(n_half_u32);
            // make the interval so that we return 1 when MSB == 1
            // this is (our number + n/2 ) % n, modulo is handled by U16Group
            let p = InG::from(1u16 << 15) + n_half;
            let q = InG::from(u16::MAX) + n_half; // modulo is handled by U16Group
            let keys: Vec<[u8; 16]> = vec![[0u8; 16]; 4];
            let mut f_x_1_bits = Vec::with_capacity(batch_size); // store all the eval results

            // Deserialize each to find original IcShare and call eval
            // Deserialize each to find original IcShare and call eval
            let key_words_fss_1: Vec<u32> = RingElement::<u32>::convert_vec(k_fss_1_vec); //need to un-flatten key vector
            let mut offset: usize = 0;
            for i in 0..batch_size {
                // // Need to "unflatten" to get batch_size number of fss keys
                let curr_key_byte_len = 1 + (key_words_fss_1[offset] as usize + 3) / 4; // offset index has the byte length, then find total u32s for this key

                // Get current key
                let k_fss_1_icshare: IcShare =
                    IcShare::deserialize(&key_words_fss_1[offset..offset + curr_key_byte_len])?;
                offset += curr_key_byte_len; //update offset to point to next cell that contains size of next key

                // reconstruct the input d+r [recall d0=x.b] for each x[i]
                let d_plus_r: RingElement<u16> =
                    d1r1_vec[i] + d2r2_vec[i] + x[i].b + RingElement(n_half_u32);
                // this should be wrapping addition, implemented by RingElement

                // Now we're ready to call eval
                let prg =
                    Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&std::array::from_fn(|i| &keys[i]));
                let icf = Icf::new(p, q, prg);
                //Call eval & convert from from ByteGroup<16> to RingElement<u128>
                let temp_eval = RingElement::<u128>(u128::from_le_bytes(
                    icf.eval(
                        true,
                        &k_fss_1_icshare,
                        fss_rs::group::int::U16Group(d_plus_r.0),
                    )
                    .0,
                ));

                // Add the respective r_prime and add to the vector of results and take only the LSB
                // Make them RingElements so it's easy to send to network
                f_x_1_bits.push(RingElement(Bit::new(
                    ((temp_eval ^ r_prime_keys[i]).0 & 1) != 0,
                )));
            }

            // Prepare them in a vector to send to dealer and next party
            let f_1_res_network: Vec<NetworkValue> = f_x_1_bits
                .iter()
                .copied()
                .map(NetworkValue::RingElementBit)
                .collect();

            let cloned_f_1_res_network = f_1_res_network.clone();
            session // send to party 0
                .network_session
                .send_prev(NetworkValue::vec_to_network(cloned_f_1_res_network))
                .await?;

            session // send to the dealer (party 2)
                .network_session
                .send_next(NetworkValue::vec_to_network(f_1_res_network))
                .await?;

            // Receive Bits of share of party 0 --> this is a vec of network values
            let f_x_0_bits_net = match session.network_session.receive_prev().await {
                Ok(v) => NetworkValue::vec_from_network(v),
                Err(e) => return Err(eyre!("Party 0 cannot receive bit shares from party 1: {e}")),
            }?;

            // Convert Vec<NetworkValue> to Vec<RingElement<Bit>>
            let f_x_0_bits: Vec<RingElement<Bit>> = f_x_0_bits_net
                .into_iter()
                .map(|nv| match nv {
                    NetworkValue::RingElementBit(b) => Ok(b),
                    other => Err(eyre!("expected RingElementBit, got {:?}", other)),
                })
                .collect::<Result<_, _>>()?;

            // Return a vector of Share<Bit> where the a is from f_x_0_bits
            // and the b is from f_x_1_bits
            let shares: Vec<Share<Bit>> = f_x_0_bits
                .into_iter()
                .zip(f_x_1_bits)
                .map(|(a, b)| Share { a, b })
                .collect();
            Ok(shares)
        }
        2 => {
            let batch_size = x.len();

            // Setting up the Interval Containment function
            // we need this to handle signed numbers, if input is unsigned no need to add N/2
            let n_half = InG::from(1u16 << 15);
            let keys: Vec<[u8; 16]> = vec![[0u8; 16]; 4];

            // make the interval so that we return 1 when MSB == 1
            // this is (our number + n/2 ) % n, modulo is handled by U16Group
            let p = InG::from(1u16 << 15) + n_half;
            let q = InG::from(u16::MAX) + n_half; // modulo is handled by U32Group

            // Draw r1 + r2 (aka r_in) before doing parallelization

            let mut r1_keys = vec![RingElement(0u16); batch_size];
            let mut r2_keys = vec![RingElement(0u16); batch_size];
            for j in 0..batch_size {
                let (_r2, _r1) = session.prf.gen_rands::<RingElement<u16>>().clone();
                r2_keys[j] = RingElement(0);
                r1_keys[j] = RingElement(0);
            }

            // Do the main loop in parallel, collecting per-key serialized results
            use rayon::prelude::*;
            let serialized_pairs: Vec<(Vec<RingElement<u32>>, Vec<RingElement<u32>>)> = (0
                ..batch_size)
                .into_par_iter()
                .map(|i| {
                    let r1_plus_r2_u32: u16 = (r1_keys[i] + r2_keys[i]).convert();
                    // Defining the function f using r_in
                    let f = IntvFn {
                        r_in: InG::from(r1_plus_r2_u32), //rin = r1+r2
                        r_out: OutG::from(0u128),        // rout=0
                    };
                    // now we can call gen to generate the FSS keys for each party
                    let prg =
                        Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&std::array::from_fn(|i| {
                            &keys[i]
                        }));
                    let icf = Icf::new(p, q, prg);
                    let (k_fss_0_pre_ser, k_fss_1_pre_ser): (IcShare, IcShare) = {
                        let mut rng = rand::thread_rng();
                        let icf_keygen_start = std::time::Instant::now();
                        let result = icf.gen(f, &mut rng);
                        let icf_keygen_elapsed_micros =
                            (icf_keygen_start.elapsed().as_secs_f64() * 1_000_000.0) as u64;
                        record_icf_keygen_micros(icf_keygen_elapsed_micros);
                        result
                    };

                    // Serialize the ICShare into u32
                    let temp_key0 = k_fss_0_pre_ser.serialize()?;
                    let v0 = RingElement::<u32>::convert_vec_rev(temp_key0.clone());

                    let temp_key1 = k_fss_1_pre_ser.serialize()?;
                    let v1 = RingElement::<u32>::convert_vec_rev(temp_key1.clone());

                    Ok((v0, v1))
                })
                .collect::<Result<Vec<(Vec<RingElement<u32>>, Vec<RingElement<u32>>)>, Error>>()?;

            // Flatten the per-key results into final flat vectors serially
            let mut k_fss_0_vec_flat = Vec::with_capacity(batch_size);
            let mut k_fss_1_vec_flat = Vec::with_capacity(batch_size);
            for (v0, v1) in serialized_pairs {
                k_fss_0_vec_flat.extend(v0);
                k_fss_1_vec_flat.extend(v1);
            }

            // Send the flattened FSS keys to parties 0 and 1, so they can do Eval
            session
                .network_session
                .send_next(NetworkInt::new_network_vec(k_fss_0_vec_flat))
                .await?; //next is party 0

            session
                .network_session
                .send_prev(NetworkInt::new_network_vec(k_fss_1_vec_flat))
                .await?; //previous is party 1

            // Receive bit of share from party 0
            let f_x_0_bits_net = match session.network_session.receive_next().await {
                Ok(v) => NetworkValue::vec_from_network(v),
                Err(e) => return Err(eyre!("Party 0 cannot receive bit shares from party 1: {e}")),
            }?;

            // Convert Vec<NetworkValue> to Vec<RingElement<Bit>>
            let f_x_0_bits: Vec<RingElement<Bit>> = f_x_0_bits_net
                .into_iter()
                .map(|nv| match nv {
                    NetworkValue::RingElementBit(b) => Ok(b),
                    other => Err(eyre!("expected RingElementBit, got {:?}", other)),
                })
                .collect::<Result<_, _>>()?;

            // Receive Bits of share of party 1 --> this is a vec of network values
            let f_x_1_bits_net = match session.network_session.receive_prev().await {
                Ok(v) => NetworkValue::vec_from_network(v),
                Err(e) => return Err(eyre!("Party 0 cannot receive bit shares from party 1: {e}")),
            }?;

            // Convert Vec<NetworkValue> to Vec<RingElement<Bit>>
            let f_x_1_bits: Vec<RingElement<Bit>> = f_x_1_bits_net
                .into_iter()
                .map(|nv| match nv {
                    NetworkValue::RingElementBit(b) => Ok(b),
                    other => Err(eyre!("expected RingElementBit, got {:?}", other)),
                })
                .collect::<Result<_, _>>()?;

            // Return a vector of Share<Bit> where the a is from f_x_0_bits
            // and the b is from f_x_1_bits
            let shares: Vec<Share<Bit>> = f_x_0_bits
                .into_iter()
                .zip(f_x_1_bits)
                .map(|(a, b)| Share { a, b })
                .collect();
            Ok(shares)
        }
        _ => {
            // this is not a valid party number
            Err(eyre!("Party no is invalid for FSS."))
        }
    }
}
/// Returns the MSB of the sum of three 32-bit integers using the binary parallel prefix adder tree.
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
    binary_add_3_get_msb(session, x1, x2, x3).await
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
pub(crate) async fn extract_msb_u32_batch_fss(
    session: &mut Session,
    x: &[Share<u32>],
) -> Result<Vec<Share<Bit>>> {
    // FSS: loop over get_msb_fss for all relevant entries of x, and collect results
    // open_bin later will take care of XOR-ing the msb shares from each party
    // Commented below is previous version without sending batches to add_3_get...
    // let mut vec_of_msb_shares: Vec<Share<Bit>> = Vec::new();
    // for x_ in x {
    //     vec_of_msb_shares.push(add_3_get_msb_fss(session, x_).await?);
    // }

    let batch_size: usize = 64;

    let mut vec_of_msb_shares: Vec<Share<Bit>> = Vec::with_capacity(x.len());
    for batch in x.chunks(batch_size) {
        let batch_out = add_3_get_msb_fss_batch(session, batch).await?;
        vec_of_msb_shares.extend(batch_out);
    }
    Ok(vec_of_msb_shares)
}

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

async fn extract_msb_u16(session: &mut Session, x_: VecShare<u16>) -> Result<VecShare<u64>, Error> {
    let x = x_.transpose_pack_u64();
    extract_msb::<u64>(session, x).await
}

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

// same as above but with fss instead
#[instrument(level = "trace", target = "searcher::network", skip_all)]
pub(crate) async fn extract_msb_u16_batch_fss(
    session: &mut Session,
    x: &[Share<u16>],
) -> Result<Vec<Share<Bit>>> {
    // FSS: loop over get_msb_fss for all relevant entries of x, and collect results
    // open_bin later will take care of XOR-ing the msb shares from each party
    // Commented below is previous version without sending batches to add_3_get...
    // let mut vec_of_msb_shares: Vec<Share<Bit>> = Vec::new();
    // for x_ in x {
    //     vec_of_msb_shares.push(add_3_get_msb_fss(session, x_).await?);
    // }

    let batch_size: usize = 512;

    let mut vec_of_msb_shares: Vec<Share<Bit>> = Vec::with_capacity(x.len());
    for batch in x.chunks(batch_size) {
        tracing::debug!("Inside extract_msb_u16_fss, batch size: {}", batch.len());
        // let batch_out = add_3_get_msb_fss_batch_parallel_u16(session, batch).await?;
        let batch_out = add_3_get_msb_fss_batch_u16(session, batch).await?;
        vec_of_msb_shares.extend(batch_out);
    }
    Ok(vec_of_msb_shares)
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

/// This open bin does NOT communicate
/// Assume each party has the same 2 shares (i.e.
/// a = the LSB of (fss_0+r')
/// b = the LSB of (fss_1-r')
/// For whoever called open_bit_fss, for each of our Share<Bit> we just XOR each of the two shares and return
#[instrument(level = "trace", target = "searcher::network", skip_all)]
pub(crate) async fn open_bin_fss(
    _session: &mut Session,
    shares: &[Share<Bit>],
) -> Result<Vec<Bit>> {
    let mut res = Vec::new();
    for sh in shares {
        res.push(sh.a.0 ^ sh.b.0);
    }
    Ok(res)
}

/// Same as open_bin BUT with u128 instead of Bit
/// Send my share b to the next party (so they can reconstruct)
/// Receiving their share b, now I can reconstruct the msb
/// XOR my 2 shares with the one I received from the prev party, return the MSB
/// we assume the parties have RSS shares of the MSB, and the shares are u128
#[instrument(level = "trace", target = "searcher::network", skip_all)]
#[allow(dead_code)]
pub(crate) async fn open_bin_fss_from_rss(
    session: &mut Session,
    shares: &[Share<u128>],
) -> Result<Vec<Bit>> {
    let network = &mut session.network_session;

    //Pack each Share<u128> in shares into a vec[u64_low, u64_high]
    // If there's more than one, the vector will look like this:
    // [u64_low_1, u64_high_1, u64_low_2, u64_high_2...] so the size should be an even number
    // where u64_low_1 is the low 64 bits of the first Share<u128> etc
    let mut message = Vec::new();
    for sh in shares {
        let low = (sh.b.0 & 0xFFFF_FFFF_FFFF_FFFF) as u64; //isolate 64 LSBs
        let high = (sh.b.0 >> 64) as u64; // isolate 64 MSBs
        message.push(RingElement::<u64>(low));
        message.push(RingElement::<u64>(high)); // add both low and high to the message
    }

    network
        .send_next(NetworkInt::new_network_vec(message))
        .await?;

    // Receiving `b` from previous party
    // First get the message from the network, we need to unpack this to Vec<RingElement<u128>>
    let other_shares = match network.receive_prev().await {
        Ok(v) => u64::into_vec(v),
        Err(e) => Err(eyre!(
            "[open_bin_fss] Cannot receive shares from previous party {}",
            e
        )),
    }?;

    let mut i = 0;
    let mut b_from_previous = Vec::new();
    while i < other_shares.len() {
        let low_bits = other_shares[i].0 as u128;
        let high_bits = other_shares[i + 1].0 as u128;
        b_from_previous.push(RingElement::<u128>((high_bits << 64) | low_bits));
        i += 2; //we look at pairs, so we need to jump by 2 each time
    }

    // XOR shares with the received shares, this should be a vector of u32 that are Bits (i.e. either all 0 or only LSB=1)
    // we convert them to Bit to keep the same return type for the function
    izip!(shares.iter(), b_from_previous.iter())
        .map(|(s, prev_b)| to_bit_from_u128(s.a ^ s.b ^ prev_b))
        .collect::<Result<Vec<_>>>()
}

fn to_bit_from_u32(msb_xored: RingElement<u32>) -> Result<Bit, Error> {
    if (msb_xored.is_zero()) {
        Ok(Bit::new(false))
    } else if (msb_xored.is_one()) {
        Ok(Bit::new(true))
    } else {
        Err(eyre!("expected a Bit (0/1), got {}", msb_xored))
    }
}

fn to_bit_from_u128(msb_xored: RingElement<u128>) -> Result<Bit, Error> {
    if (msb_xored.is_zero()) {
        Ok(Bit::new(false))
    } else if (msb_xored.is_one()) {
        Ok(Bit::new(true))
    } else {
        Err(eyre!("expected a Bit (0/1), got {}", msb_xored))
    }
}

/// Format timing value in appropriate units (µs, ms, or s).
fn format_time_micros(micros: f64) -> String {
    if micros >= 1_000_000.0 {
        format!("{:.2} s", micros / 1_000_000.0)
    } else if micros >= 1_000.0 {
        format!("{:.2} ms", micros / 1_000.0)
    } else {
        format!("{:.2} µs", micros)
    }
}

/// Format bytes in appropriate units (B, MB, or GB).
fn format_bytes(bytes: u64) -> String {
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const KB: f64 = 1024.0;

    let bytes_f = bytes as f64;
    if bytes_f >= GB {
        format!("{:.2} GB", bytes_f / GB)
    } else if bytes_f >= MB {
        format!("{:.2} MB", bytes_f / MB)
    } else if bytes_f >= KB {
        format!("{:.2} KB", bytes_f / KB)
    } else {
        format!("{} B", bytes)
    }
}

/// Print a summary of timing statistics for the FSS operations.
/// Displays average time per operation and operation count for:
/// - PRG key generation
/// - ICF evaluation and keygen
/// - Network operations
pub fn print_fss_timing_summary() {
    tracing::info!("=== FSS Timing Summary ===");

    // PRG Key Generation
    let prg_count = FSS_PRG_KEYGEN_COUNT.load(Ordering::Relaxed);
    if prg_count > 0 {
        let prg_sum = FSS_PRG_KEYGEN_SUM_MICROS.load(Ordering::Relaxed) as f64;
        let prg_avg = prg_sum / prg_count as f64;
        tracing::info!(
            "  PRG Key Generation: {} ops, avg = {}",
            prg_count,
            format_time_micros(prg_avg)
        );
    }

    // ICF Evaluation
    let icf_count = FSS_ICF_EVAL_COUNT.load(Ordering::Relaxed);
    if icf_count > 0 {
        let icf_sum = FSS_ICF_EVAL_SUM_MICROS.load(Ordering::Relaxed) as f64;
        let icf_avg = icf_sum / icf_count as f64;
        tracing::info!(
            "  ICF Evaluation: {} ops, avg = {}",
            icf_count,
            format_time_micros(icf_avg)
        );
    }

    // ICF Keygen
    let icf_keygen_count = FSS_ICF_KEYGEN_COUNT.load(Ordering::Relaxed);
    if icf_keygen_count > 0 {
        let icf_keygen_sum = FSS_ICF_KEYGEN_SUM_MICROS.load(Ordering::Relaxed) as f64;
        let icf_keygen_avg = icf_keygen_sum / icf_keygen_count as f64;
        tracing::info!(
            "  ICF Keygen: {} ops, avg = {}",
            icf_keygen_count,
            format_time_micros(icf_keygen_avg)
        );
    }

    // Network Operations
    let net_count = FSS_NETWORK_COUNT.load(Ordering::Relaxed);
    if net_count > 0 {
        let net_sum = FSS_NETWORK_SUM_MICROS.load(Ordering::Relaxed) as f64;
        let net_avg = net_sum / net_count as f64;
        tracing::info!(
            "  Network Operations: {} ops, avg = {}",
            net_count,
            format_time_micros(net_avg)
        );
    }

    // Network Bytes Sent
    let net_bytes = FSS_NETWORK_BYTES_SENT.load(Ordering::Relaxed);
    if net_bytes > 0 {
        tracing::info!("  Network Bytes Sent: {}", format_bytes(net_bytes));
    }
}

#[cfg(test)]
fn prepare_shares_correct_msbs(total_shares: usize) -> (Vec<Vec<Share<u32>>>, Vec<Bit>) {
    // Generate random and fixed values to check their msbs
    let mut rng = AesRng::seed_from_u64(0xC0FFEE);
    let mut shares_list: Vec<u32> = vec![
        2048,
        0u32,
        1u32,
        3424991880,
        u32::MAX,
        (1u32 << 31) - 1, // 0x7FFF_FFFF  MSB=0
        1u32 << 31,       // 0x8000_0000  MSB=1
        (1u32 << 31) + 1,
    ];
    // Add also 20 more random ones
    for _ in 0..(total_shares - 8) {
        shares_list.push(rng.gen::<u32>());
    }

    // Split the values among 3 parties
    let [party_0_shares, party_1_shares, party_2_shares] = make_replicated_shares_u32(&shares_list);
    let party_i_shares: Vec<Vec<Share<u32>>> = vec![party_0_shares, party_1_shares, party_2_shares];

    // Calculate correct MSBs
    let correct_msbs: Vec<Bit> = shares_list
        .clone()
        .into_iter()
        .map(|x| Bit::new((x >> 31) != 0))
        .collect();

    (party_i_shares, correct_msbs)
}
#[cfg(test)]
/// Just splitting vals into shares, to test the extract_msb_u32 functions
fn make_replicated_shares_u32(vals: &[u32]) -> [Vec<Share<u32>>; 3] {
    let mut p0 = Vec::with_capacity(vals.len());
    let mut p1 = Vec::with_capacity(vals.len());
    let mut p2 = Vec::with_capacity(vals.len());

    for &v in vals {
        let d0: u32 = thread_rng().gen();
        let d1: u32 = thread_rng().gen();
        let d2: u32 = v.wrapping_sub(d0).wrapping_sub(d1);

        // Party 0: (d0, d2)
        p0.push(Share {
            a: RingElement(d0),
            b: RingElement(d2),
        });
        // Party 1: (d1, d0)
        p1.push(Share {
            a: RingElement(d1),
            b: RingElement(d0),
        });
        // Party 2: (d2, d1)
        p2.push(Share {
            a: RingElement(d2),
            b: RingElement(d1),
        });
    }
    [p0, p1, p2]
}
#[cfg(test)]
mod tests_fss {
    use super::*;
    use crate::{
        execution::local::{generate_local_identities, LocalRuntime},
        shares::share,
    };

    use futures::{future::join_all, stream::Zip};
    #[tokio::test]
    async fn unit_test_add_3_fss_simple() -> Result<(), Error> {
        let total_shares = 10000;

        let (party_i_shares, correct_msbs) = prepare_shares_correct_msbs(total_shares);

        // Create dummy sessions --> creates 3 dummy sessions, Alice (0), Bob(1), Charlie (2)
        let sessions = LocalRuntime::mock_sessions_with_channel().await.unwrap();

        let mut fut = Vec::with_capacity(3);
        for party_no in 0..3 {
            // for each party, ger their session
            let sess_i = sessions[party_no].clone();
            let shares_i = party_i_shares[party_no].clone();

            fut.push(async move {
                let mut results: Vec<Share<Bit>> = Vec::new();
                for share_i in &shares_i {
                    // for each of the party's shares, call add_3_get_msb_fss
                    //is this &shares_i?

                    let mut sess_i_mutex = sess_i.lock().await; // get next session
                    results.push(add_3_get_msb_fss(&mut sess_i_mutex, &share_i).await?);
                }
                Ok(results) as Result<Vec<Share<Bit>>, Error>
            });
        }

        // wait the futures (otherwise they'll do nothing)
        let mut joined: Vec<Result<Vec<Share<Bit>>, Error>> = join_all(fut).await;

        let output0 = joined.remove(0)?;
        let output1 = joined.remove(0)?;
        let output2 = joined.remove(0)?;

        // Everyone is done now, we need to cpmpare the results with our correct_msb vector
        // let mut disagreement_count = 0;
        for i in 0..total_shares {
            // Check that everyone has the correct shares
            assert_eq!(output0[i].a, output1[i].a);
            assert_eq!(output2[i].a, output1[i].a);
            assert_eq!(output0[i].b, output1[i].b);
            assert_eq!(output2[i].b, output1[i].b);

            // Reconstruct the MSB for i'th share and check them against the correct one
            let curr_msb = output0[i].a ^ output0[i].b; // this is 1 if number is positive (i.e. MSB==0)

            assert!(curr_msb.0 == correct_msbs[i]);
        }
        // PASS: all numbers are either 0 or 1, AND correct results
        // FAIL: else
        Ok(())
    }

    #[tokio::test]
    async fn unit_test_add_3_fss_batch() -> Result<(), Error> {
        let total_shares = 1000;
        let batch_size = 16;

        // let total_shares: usize = std::env::var("TOTAL_SHARES")
        //     .ok()
        //     .and_then(|s| s.parse().ok())
        //     .unwrap();

        // let batch_size: usize = std::env::var("BATCH_SIZE")
        //     .ok()
        //     .and_then(|s| s.parse().ok())
        //     .unwrap();
        let t0 = std::time::Instant::now();
        let (party_i_shares, correct_msbs) = prepare_shares_correct_msbs(total_shares);

        // Create dummy sessions --> creates 3 dummy sessions, Alice (0), Bob(1), Charlie (2)
        let sessions = LocalRuntime::mock_sessions_with_channel().await.unwrap();

        let mut fut = Vec::with_capacity(3);
        for party_no in 0..3 {
            // for each party, ger their session
            let sess_i = sessions[party_no].clone();
            let shares_i = party_i_shares[party_no].clone();

            fut.push(async move {
                let mut results: Vec<Share<Bit>> = Vec::new();
                for share_i in shares_i.chunks(batch_size) {
                    let mut sess_i_mutex = sess_i.lock().await; // get next session
                    results.extend(add_3_get_msb_fss_batch(&mut sess_i_mutex, share_i).await?);
                }
                Ok(results) as Result<Vec<Share<Bit>>, Error>
            });
        }

        // // wait the futures (otherwise they'll do nothing)
        let mut joined: Vec<Result<Vec<Share<Bit>>, Error>> = join_all(fut).await;
        // eprintln!(
        //     "TIMING ({}, {}) = {:?}",
        //     total_shares,
        //     batch_size,
        //     t0.elapsed()
        // );

        let output0 = joined.remove(0)?;
        let output1 = joined.remove(0)?;
        let output2 = joined.remove(0)?;

        // // Everyone is done now, we need to cpmpare the results with our correct_msb vector
        // // let mut disagreement_count = 0;
        for i in 0..total_shares {
            // Check that everyone has the correct shares
            assert_eq!(output0[i].a, output1[i].a);
            assert_eq!(output2[i].a, output1[i].a);
            assert_eq!(output0[i].b, output1[i].b);
            assert_eq!(output2[i].b, output1[i].b);

            // Reconstruct the MSB for i'th share and check them against the correct one
            let curr_msb = output0[i].a ^ output0[i].b; // this is 1 if number is positive (i.e. MSB==0)

            assert!(curr_msb.0 == correct_msbs[i]);
        }
        // // PASS: all numbers are either 0 or 1, AND correct results
        // // FAIL: else
        Ok(())
    }

    #[tokio::test]
    async fn unit_test_extract_msb_batch_fss() -> Result<(), Error> {
        let total_shares = 100;

        let (party_i_shares, correct_msbs) = prepare_shares_correct_msbs(total_shares);

        // Create sessions, call oer party
        let sessions = LocalRuntime::mock_sessions_with_channel().await.unwrap();

        let mut fut = Vec::with_capacity(3);
        for party_no in 0..3 {
            // for each party, ger their session
            let sess_i = sessions[party_no].clone();
            let shares_i = party_i_shares[party_no].clone();

            fut.push(async move {
                // for each of the party's vector of shares, call extract_msb_u32_batch_fss
                let mut sess_i_mutex = sess_i.lock().await; // get next session
                let results: Vec<Share<Bit>> =
                    extract_msb_u32_batch_fss(&mut sess_i_mutex, &shares_i).await?;
                Ok(results) as Result<Vec<Share<Bit>>, Error>
            });
        }

        let mut joined: Vec<Result<Vec<Share<Bit>>, Error>> = join_all(fut).await;

        let output0 = joined.remove(0)?;
        let output1 = joined.remove(0)?;
        let output2 = joined.remove(0)?;

        // Everyone is done now, we need to cpmpare the results with our correct_msb vector
        // let mut disagreement_count = 0;
        for i in 0..total_shares {
            // Reconstruct the MSB for i'th share and check them against the correct one
            let curr_msb = output0[i].a ^ output0[i].b; // this is 1 if number is positive (i.e. MSB==0)
                                                        // println!("Reconstructed MSB is {:?}", curr_msb);

            assert_eq!(curr_msb.0, correct_msbs[i]);

            // Check that everyone has the correct shares (whouls all be equal)
            assert_eq!(output0[i], output1[i]);
            assert_eq!(output2[i], output1[i]);
        }

        Ok(())
    }

    #[tokio::test]
    async fn unit_test_all_new_add_3_fss_pipeline() -> Result<(), Error> {
        let total_shares = 1000; //_000;

        let (party_i_shares, correct_msbs) = prepare_shares_correct_msbs(total_shares);

        // Create sessions, call per party
        let sessions = LocalRuntime::mock_sessions_with_channel().await.unwrap();

        let mut fut = Vec::with_capacity(3);
        for party_no in 0..3 {
            // for each party, get their session
            let sess_i = sessions[party_no].clone();
            let shares_i = party_i_shares[party_no].clone();

            fut.push(async move {
                // for each of the party's vector of shares, call extract_msb_u32_batch_fss
                let mut sess_i_mutex = sess_i.lock().await; // get next session
                let results: Vec<Share<Bit>> =
                    extract_msb_u32_batch_fss(&mut sess_i_mutex, &shares_i).await?;
                let msbs = open_bin_fss(&mut sess_i_mutex, &results).await?;
                Ok(msbs) as Result<Vec<Bit>, Error>
            });
        }

        let mut jobs = join_all(fut).await;
        let output0 = jobs.remove(0)?;
        let output1 = jobs.remove(0)?;
        let output2 = jobs.remove(0)?;

        //Check the msb vector that open_bin returned
        for i in 0..total_shares {
            // Everyone agrees on the opened bit
            assert_eq!(
                output0[i], output1[i],
                "MSB mismatch (0 vs 1) at index {}",
                i
            );
            assert_eq!(
                output2[i], output1[i],
                "MSB mismatch (2 vs 1) at index {}",
                i
            );
            // Agreement witht he correct bit also
            assert_eq!(output0[i], correct_msbs[i], "MSB mismatch at index {}", i);
        }
        Ok(())
    }
}
