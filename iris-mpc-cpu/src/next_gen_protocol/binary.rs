use crate::{
    execution::{
        player::Role,
        session::{Session, SessionHandles},
    },
    next_gen_network::value::NetworkValue,
    shares::{
        bit::Bit,
        int_ring::IntRing2k,
        ring_impl::RingElement,
        share::Share,
        vecshare::{SliceShare, VecShare},
    },
};
use eyre::{eyre, Error};
use num_traits::{One, Zero};
use rand::{distributions::Standard, prelude::Distribution};

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

    match session.own_role()?.zero_based() {
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
        _ => unimplemented!(),
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
    let mut prf = session.prf_as_mut();

    let mut shares_a = Vec::with_capacity(a.len());
    for (a_, b_) in a.iter().zip(b.iter()) {
        let rand = prf.gen_binary_zero_share::<u64>();
        let mut c = a_ & b_;
        c ^= rand;
        shares_a.push(c);
    }

    let next_party = session.next_identity()?;
    let network = session.network().clone();
    let sid = session.session_id();
    let message = shares_a.clone();
    let _ = tokio::spawn(async move {
        let _ = network
            .send(
                NetworkValue::VecRing64(message).to_network(),
                &next_party,
                &sid,
            )
            .await;
    })
    .await;
    Ok(shares_a)
}

pub(crate) async fn and_many_receive(
    session: &mut Session,
) -> Result<Vec<RingElement<u64>>, Error> {
    let network = session.network().clone();
    let sid = session.session_id();
    let prev_party = session.prev_identity()?;

    let shares_b = tokio::spawn(async move {
        let serialized_other_share = network.receive(&prev_party, &sid).await;
        match NetworkValue::from_network(serialized_other_share) {
            Ok(NetworkValue::VecRing64(message)) => Ok(message),
            _ => Err(eyre!("Error in receiving in and_many operation")),
        }
    })
    .await??;
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

pub(crate) async fn transposed_pack_and(
    session: &mut Session,
    x1: Vec<VecShare<u64>>,
    x2: Vec<VecShare<u64>>,
) -> Result<Vec<VecShare<u64>>, Error>
where
    Standard: Distribution<u64>,
{
    // TODO(Dragos) this could probably be parallelized even more.
    let mut res = Vec::with_capacity(x1.len());
    for (x1, x2) in x1.iter().zip(x2.iter()) {
        let shares_a = and_many_send(session, x1.as_slice(), x2.as_slice()).await?;
        let shares_b = and_many_receive(session).await?;
        res.push(VecShare::from_ab(shares_a, shares_b))
    }
    Ok(res)
}

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

    // First full adder (carry is 0)
    let mut c = and_many(session, a[1].as_slice(), b[0].as_slice()).await?;

    // For last round
    let mut b_msb = b.pop().expect("Enough elements present");

    // 2 -> k
    for (a_, b_) in a.iter_mut().skip(2).zip(b.iter_mut().skip(1)) {
        *a_ ^= c.as_slice();
        *b_ ^= c.as_slice();
        let tmp_c = and_many(session, a_.as_slice(), b_.as_slice()).await?;
        c ^= tmp_c;
    }

    // Finally, last bit of a is 0
    let res2 = and_many(session, b_msb.as_slice(), c.as_slice()).await?;
    b_msb ^= c;

    // Extract bits for outputs
    let mut res1 = b_msb.convert_to_bits();
    res1.truncate(truncate_len);
    let mut res2 = res2.convert_to_bits();
    res2.truncate(truncate_len);

    Ok((res1, res2))
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
            let (x1__, x2__, x3__) = a2b_pre(&session, x__)?;
            x1_.push(x1__);
            x2_.push(x2__);
            x3_.push(x3__);
        }
        x1.push(x1_);
        x2.push(x2_);
        x3.push(x3_);
    }

    let (mut b1, b2) = binary_add_3_get_two_carries(session, x1, x2, x3, len).await?;

    unimplemented!()
}
