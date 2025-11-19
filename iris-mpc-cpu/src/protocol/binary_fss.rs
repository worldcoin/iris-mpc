// same imports as in binary.rs

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
use std::{cell::RefCell, ops::SubAssign};
use tracing::{instrument, trace_span, Instrument};

use fss_rs::icf::{IcShare, Icf, InG, IntvFn, OutG};
use fss_rs::prg::Aes128MatyasMeyerOseasPrg;

// Evaluation (P0/P1) and generation (P2) becomes parallel under `parallel-msb` if batch.len > parallel_threshold
// Here r_2 = r_1 = 0, so d2r2=d2 and d1r1=d1
pub(crate) async fn add_3_get_msb_fss_batch_parallel_threshold_timers(
    session: &mut Session,
    batch: &[Share<u32>],
    parallel_threshold: usize,
) -> Result<Vec<Share<Bit>>, Error> {
    use eyre::eyre;
    use fss_rs::icf::{IcShare, Icf, InG, IntvFn, OutG};
    use fss_rs::prg::Aes128MatyasMeyerOseasPrg;
    use rand::thread_rng;

    let role = session.own_role().index();
    let n = batch.len();
    let bucket_bound = 150;

    #[inline]
    fn re_vec_to_u32(v: Vec<RingElement<u32>>) -> Vec<u32> {
        RingElement::<u32>::convert_vec(v)
    }
    #[inline]
    fn u32_to_re_vec(v: Vec<u32>) -> Vec<RingElement<u32>> {
        RingElement::<u32>::convert_vec_rev(v)
    }

    // We test y ∈ [0, 2^31-1] where y = x + r_in + 2^31 (mod 2^32)
    let p = InG::from(0u32);
    //let q = InG::from((1u32 << 31) - 1);
    let q = InG::from(1u32 << 31); // [0, 2^31)
    let n_half_u32: u32 = 1u32 << 31;

    match role {
        // =======================
        // Party 0 (Evaluator)
        // =======================
        0 => {
            // metrics: measure the network for share reconstruction
            let _tt_net_recv = crate::perf_scoped_for_party!(
                "fss.network.start_recv_keylen",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Receive lengths and concatenated key blob from Dealer (P2/prev)
            let lens_re = session.network_session.receive_prev().await.map_err(|e| {
                eyre!("Party 0 cannot receive FSS key lengths batch from dealer: {e}")
            })?;

            // metrics: stop timer
            drop(_tt_net_recv);

            // metrics: measure the network for share reconstruction
            let _tt_net_recv = crate::perf_scoped_for_party!(
                "fss.network.start_recv_keys",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            let key_blob_re = session
                .network_session
                .receive_prev()
                .await
                .map_err(|e| eyre!("Party 0 cannot receive FSS key batch from dealer: {e}"))?;

            // metrics: stop timer
            drop(_tt_net_recv);

            let lens_p0 = re_vec_to_u32(u32::into_vec(lens_re)?);
            let key_blob = re_vec_to_u32(u32::into_vec(key_blob_re)?);

            // Deserialize my keys (batched)
            let mut off = 0usize;
            let mut my_keys = Vec::with_capacity(n);
            for &l in &lens_p0 {
                let len = l as usize;
                my_keys.push(IcShare::deserialize(&key_blob[off..off + len])?);
                off += len;
            }

            // Precompute offsets so we can (optionally) re-deserialize per-index in parallel
            let lens_usize: Vec<usize> = lens_p0.iter().map(|&x| x as usize).collect();
            let mut offs: Vec<usize> = Vec::with_capacity(n);
            {
                let mut acc = 0usize;
                for &len in &lens_usize {
                    offs.push(acc);
                    acc += len;
                }
            }

            // Build a single PRG/ICF for eval (used in the sequential path)
            let seed = [[0u8; 16]; 4];
            let prg = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                &seed[0], &seed[1], &seed[2], &seed[3],
            ]);
            let icf = Icf::new(p, q, prg);

            // ===== batched masked-share exchange =====
            // Send all (d2+r2) to NEXT; receive all (d1+r1) from NEXT

            //metrics: measure the network for share reconstruction
            let _tt_net_recon = crate::perf_scoped_for_party!(
                "fss.network.recon.send",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            let send_d2r2: Vec<RingElement<u32>> = batch.iter().map(|x| x.b).collect();
            session
                .network_session
                .send_next(NetworkInt::new_network_vec(send_d2r2))
                .await?;

            // metrics: stop timer here for network reconstruction
            drop(_tt_net_recon);

            //metrics: measure the network for share reconstruction
            let _tt_net_recon = crate::perf_scoped_for_party!(
                "fss.network.recon.recv",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            let d1r1_msg = session
                .network_session
                .receive_next()
                .await
                .map_err(|e| eyre!("FSS: Party 0 cannot receive d1+r1 vector from P1: {e}"))?;
            let d1r1_vec: Vec<RingElement<u32>> = u32::into_vec(d1r1_msg)?;

            // metrics: stop timer here for network reconstruction
            drop(_tt_net_recon);

            // ===== Evaluate all indices and collect bit shares =====
            // Parallel when feature enabled and n >= threshold; otherwise sequential.
            #[cfg(feature = "parallel-msb")]
            let (bit0s_ringbit, bit0s_u32) = {
                if n >= parallel_threshold {
                    use rayon::prelude::*;
                    let eval_pairs: Vec<(RingElement<Bit>, RingElement<u32>)> = (0..n)
                        .into_par_iter()
                        .map(|i| {
                            let x = &batch[i]; // borrow, do not move
                            let y = x.a + d1r1_vec[i] + x.b + RingElement(n_half_u32);

                            // Rebuild ICF locally per task to avoid shared state
                            let seed = [[0u8; 16]; 4];
                            let prg_i = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                                &seed[0], &seed[1], &seed[2], &seed[3],
                            ]);
                            let icf_i = Icf::new(p, q, prg_i);

                            // Re-deserialize key locally to avoid Sync bounds on IcShare
                            let start = offs[i];
                            let len = lens_usize[i];
                            let key_i = IcShare::deserialize(&key_blob[start..start + len])
                                .expect("deserialize IcShare (P0)");

                            // Evaluate ICF; OutG is 16 bytes BE; take LSB as bit share
                            let f0 = icf_i.eval(false, &key_i, fss_rs::group::int::U32Group(y.0));
                            let f0_u128 = u128::from_le_bytes(f0.0);
                            let b = (f0_u128 & 1) != 0;
                            (
                                RingElement(Bit::new(b)),
                                RingElement(if b { 1u32 } else { 0u32 }),
                            )
                        })
                        .collect();
                    eval_pairs.into_iter().unzip()
                } else {
                    let mut bit0s_ringbit: Vec<RingElement<Bit>> = Vec::with_capacity(n);
                    let mut bit0s_u32: Vec<RingElement<u32>> = Vec::with_capacity(n);
                    for i in 0..n {
                        let x = &batch[i];
                        let y = x.a + d1r1_vec[i] + x.b + RingElement(n_half_u32);
                        let f0 = icf.eval(false, &my_keys[i], fss_rs::group::int::U32Group(y.0));
                        //let f0_u128 = u128::from_be_bytes(f0.0);
                        let f0_u128 = u128::from_le_bytes(f0.0);
                        let b = (f0_u128 & 1) != 0;
                        //let b = fss_out_bit(&f0.0);
                        bit0s_ringbit.push(RingElement(Bit::new(b)));
                        bit0s_u32.push(RingElement(if b { 1u32 } else { 0u32 }));
                    }
                    (bit0s_ringbit, bit0s_u32)
                }
            };

            #[cfg(not(feature = "parallel-msb"))]
            let (bit0s_ringbit, bit0s_u32) = {
                let _tt = crate::perf_scoped_for_party!(
                    "fss.add3.non-parallel",
                    role,
                    n,            // bucket on the items this block processes
                    bucket_bound  // your desired bucket cap
                );

                let mut bit0s_ringbit: Vec<RingElement<Bit>> = Vec::with_capacity(n);
                let mut bit0s_u32: Vec<RingElement<u32>> = Vec::with_capacity(n);
                for i in 0..n {
                    let x = &batch[i];
                    let y = x.a + d1r1_vec[i] + x.b + RingElement(n_half_u32);

                    crate::perf_time_let_for_party!(
                        "fss.add3.icf.eval",
                        role,
                        n,
                        bucket_bound,
                            let f0 = icf.eval(false, &my_keys[i], fss_rs::group::int::U32Group(y.0)) //;
                    );

                    let f0_u128 = u128::from_le_bytes(f0.0);
                    let b = (f0_u128 & 1) != 0;
                    bit0s_ringbit.push(RingElement(Bit::new(b)));
                    bit0s_u32.push(RingElement(if b { 1u32 } else { 0u32 }));
                }
                (bit0s_ringbit, bit0s_u32)
            };

            // ===== batched bit-share exchange =====
            // Send our bits to BOTH neighbors as a single vector of u32 (0/1)

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.send_prev",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session
                .network_session
                .send_prev(NetworkInt::new_network_vec(bit0s_u32.clone()))
                .await?;

            // metrics: stop timer
            drop(_tt_net);

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.send_next",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session
                .network_session
                .send_next(NetworkInt::new_network_vec(bit0s_u32))
                .await?;

            // metrics: stop timer
            drop(_tt_net);

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.recv_to_eval",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Receive P1's bits (as u32 0/1 vector) from NEXT
            let p1_bits_msg = session
                .network_session
                .receive_next()
                .await
                .map_err(|e| eyre!("Party 0 cannot receive bit vector from P1: {e}"))?;

            // metrics: stop timer
            drop(_tt_net);

            let p1_bits_u32: Vec<RingElement<u32>> = u32::into_vec(p1_bits_msg)?;
            let bit1s_ringbit: Vec<RingElement<Bit>> = p1_bits_u32
                .into_iter()
                .map(|re| RingElement(Bit::new(re.0 != 0)))
                .collect();

            // Assemble output shares
            let out: Vec<Share<Bit>> = bit0s_ringbit
                .into_iter()
                .zip(bit1s_ringbit.into_iter())
                .map(|(b0, b1)| Share::new(b0, b1))
                .collect();

            Ok(out)
        }

        // =======================
        // Party 1 (Evaluator)
        // =======================
        1 => {
            // metrics: measure the network for share reconstruction
            let _tt_net_recv = crate::perf_scoped_for_party!(
                "fss.network.start_recv_keylen",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Receive lengths and concatenated key blob from Dealer (P2/next)
            let lens_re = session.network_session.receive_next().await.map_err(|e| {
                eyre!("Party 1 cannot receive FSS key lengths batch from dealer: {e}")
            })?;

            // metrics: stop timer
            drop(_tt_net_recv);

            // metrics: measure the network for share reconstruction
            let _tt_net_recv = crate::perf_scoped_for_party!(
                "fss.network.start_recv_keys",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );
            let key_blob_re = session
                .network_session
                .receive_next()
                .await
                .map_err(|e| eyre!("Party 1 cannot receive FSS key batch from dealer: {e}"))?;

            // metrics: stop timer
            drop(_tt_net_recv);

            let lens_p1 = re_vec_to_u32(u32::into_vec(lens_re)?);
            let key_blob = re_vec_to_u32(u32::into_vec(key_blob_re)?);

            // Deserialize my keys (batched)
            let mut off = 0usize;
            let mut my_keys = Vec::with_capacity(n);
            for &l in &lens_p1 {
                let len = l as usize;
                my_keys.push(IcShare::deserialize(&key_blob[off..off + len])?);
                off += len;
            }

            // Precompute offsets for optional per-iteration re-deserialization
            let lens_usize: Vec<usize> = lens_p1.iter().map(|&x| x as usize).collect();
            let mut offs: Vec<usize> = Vec::with_capacity(n);
            {
                let mut acc = 0usize;
                for &len in &lens_usize {
                    offs.push(acc);
                    acc += len;
                }
            }

            // Build a single PRG/ICF for eval (used in the sequential path)
            let seed = [[0u8; 16]; 4];
            let prg = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                &seed[0], &seed[1], &seed[2], &seed[3],
            ]);
            let icf = Icf::new(p, q, prg);

            // ===== batched masked-share exchange =====
            // Send all (d1+r1) to PREV; receive all (d2+r2) from PREV

            // metrics: measure the network for share reconstruction
            let _tt_net_recon = crate::perf_scoped_for_party!(
                "fss.network.recon.send",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            let send_d1r1: Vec<RingElement<u32>> = batch.iter().map(|x| x.a).collect();
            session
                .network_session
                .send_prev(NetworkInt::new_network_vec(send_d1r1))
                .await?;

            // metrics: stop timer here for network reconstruction
            drop(_tt_net_recon);

            // metrics: measure the network for share reconstruction
            let _tt_net_recon = crate::perf_scoped_for_party!(
                "fss.network.recon.recv",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            let d2r2_msg = session
                .network_session
                .receive_prev()
                .await
                .map_err(|e| eyre!("FSS: Party 1 cannot receive d2+r2 vector from P0: {e}"))?;
            let d2r2_vec: Vec<RingElement<u32>> = u32::into_vec(d2r2_msg)?;

            // metrics: stop timer here for network reconstruction
            drop(_tt_net_recon);

            // ===== Evaluate all indices and collect bit shares =====
            #[cfg(feature = "parallel-msb")]
            let (bit1s_ringbit, bit1s_u32) = {
                if n >= parallel_threshold {
                    use rayon::prelude::*;
                    let eval_pairs: Vec<(RingElement<Bit>, RingElement<u32>)> = (0..n)
                        .into_par_iter()
                        .map(|i| {
                            let x = &batch[i]; // borrow, do not move
                            let y = x.a + d2r2_vec[i] + x.b + RingElement(n_half_u32);

                            // Rebuild ICF locally
                            let seed = [[0u8; 16]; 4];
                            let prg_i = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                                &seed[0], &seed[1], &seed[2], &seed[3],
                            ]);
                            let icf_i = Icf::new(p, q, prg_i);

                            // Re-deserialize key locally
                            let start = offs[i];
                            let len = lens_usize[i];
                            let key_i = IcShare::deserialize(&key_blob[start..start + len])
                                .expect("deserialize IcShare (P1)");

                            // Evaluate ICF; OutG is 16 bytes BE; take LSB as bit share
                            let f1 = icf_i.eval(true, &key_i, fss_rs::group::int::U32Group(y.0));
                            let f1_u128 = u128::from_le_bytes(f1.0);
                            let b = (f1_u128 & 1) != 0;
                            (
                                RingElement(Bit::new(b)),
                                RingElement(if b { 1u32 } else { 0u32 }),
                            )
                        })
                        .collect();
                    eval_pairs.into_iter().unzip()
                } else {
                    let mut bit1s_ringbit: Vec<RingElement<Bit>> = Vec::with_capacity(n);
                    let mut bit1s_u32: Vec<RingElement<u32>> = Vec::with_capacity(n);
                    for i in 0..n {
                        let x = &batch[i];
                        let y = x.a + d2r2_vec[i] + x.b + RingElement(n_half_u32);
                        let f1 = icf.eval(true, &my_keys[i], fss_rs::group::int::U32Group(y.0));
                        let f1_u128 = u128::from_le_bytes(f1.0);
                        let b = (f1_u128 & 1) != 0;
                        bit1s_ringbit.push(RingElement(Bit::new(b)));
                        bit1s_u32.push(RingElement(if b { 1u32 } else { 0u32 }));
                    }
                    (bit1s_ringbit, bit1s_u32)
                }
            };

            #[cfg(not(feature = "parallel-msb"))]
            let (bit1s_ringbit, bit1s_u32) = {
                let _tt = crate::perf_scoped_for_party!(
                    "fss.add3.non-parallel",
                    role,
                    n,            // bucket on the items this block processes
                    bucket_bound  // your desired bucket cap
                );

                let mut bit1s_ringbit: Vec<RingElement<Bit>> = Vec::with_capacity(n);
                let mut bit1s_u32: Vec<RingElement<u32>> = Vec::with_capacity(n);
                for i in 0..n {
                    let x = &batch[i];
                    let y = x.a + d2r2_vec[i] + x.b + RingElement(n_half_u32);

                    crate::perf_time_let_for_party!(
                        "fss.add3.icf.eval",
                        role,
                        n,
                        bucket_bound,
                            let f1 = icf.eval(true, &my_keys[i], fss_rs::group::int::U32Group(y.0)) //;
                    );

                    let f1_u128: u128 = u128::from_le_bytes(f1.0);
                    let b = (f1_u128 & 1) != 0;
                    bit1s_ringbit.push(RingElement(Bit::new(b)));
                    bit1s_u32.push(RingElement(if b { 1u32 } else { 0u32 }));
                }
                (bit1s_ringbit, bit1s_u32)
            };

            // ===== batched bit-share exchange =====
            // Send our bits to BOTH neighbors as a single vector of u32 (0/1)

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.send_next",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session
                .network_session
                .send_next(NetworkInt::new_network_vec(bit1s_u32.clone()))
                .await?;

            // metrics: stop timer
            drop(_tt_net);

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.send_prev",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session
                .network_session
                .send_prev(NetworkInt::new_network_vec(bit1s_u32))
                .await?;

            // metrics: stop time
            drop(_tt_net);

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.recv_to_eval",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Receive P0's bits (as u32 0/1 vector) from PREV
            let p0_bits_msg = session
                .network_session
                .receive_prev()
                .await
                .map_err(|e| eyre!("Party 1 cannot receive bit vector from P0: {e}"))?;

            // metrics: stop timer
            drop(_tt_net);

            let p0_bits_u32: Vec<RingElement<u32>> = u32::into_vec(p0_bits_msg)?;
            let bit0s_ringbit: Vec<RingElement<Bit>> = p0_bits_u32
                .into_iter()
                .map(|re| RingElement(Bit::new(re.0 != 0)))
                .collect();

            // Assemble output shares
            let out: Vec<Share<Bit>> = bit0s_ringbit
                .into_iter()
                .zip(bit1s_ringbit.into_iter())
                .map(|(b0, b1)| Share::new(b0, b1))
                .collect();

            Ok(out)
        }

        // =======================
        // Party 2 (Dealer)
        // =======================
        2 => {
            // // Build r_in (input masks) using the session PRF; must match evaluators' PRF consumption
            // let mut r_in_list: Vec<u32> = Vec::with_capacity(n);
            // for _ in 0..n {
            //     let (r_next, r_prev) = session.prf.gen_rands::<RingElement<u32>>().clone();
            //     let r_in = (r_next + r_prev).convert();
            //     r_in_list.push(r_in);
            // }

            // This variant assumes r1 = r2 = 0 on evaluators, so use r_in = 0 here as well.
            // Otherwise keys would be generated for x + r_in while evaluators feed x.
            let r_in_list: Vec<u32> = vec![0u32; n];

            // Fresh ICF.Gen per index, r_out = 0; adaptive parallelism by threshold
            // First collect key pairs, then pre-size and flatten to avoid reallocs.
            let pairs: Vec<(Vec<u32>, Vec<u32>)> = {
                #[cfg(feature = "parallel-msb")]
                {
                    if n >= parallel_threshold {
                        use rayon::prelude::*;
                        r_in_list
                            .par_iter()
                            .map(|&r_in_u32| {
                                let seed = [[0u8; 16]; 4];
                                let prg_i = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                                    &seed[0], &seed[1], &seed[2], &seed[3],
                                ]);
                                let icf = Icf::new(p, q, prg_i);
                                let mut rng = thread_rng();
                                let f = IntvFn {
                                    //r_in: InG::from(r_in_u32),
                                    r_in: InG::from(0u32),
                                    r_out: OutG::from(0u128),
                                };
                                let (k0, k1) = icf.gen(f, &mut rng);
                                (k0.serialize().unwrap(), k1.serialize().unwrap())
                            })
                            .collect()
                    } else {
                        let mut tmp = Vec::with_capacity(n);
                        for &r_in_u32 in &r_in_list {
                            let seed = [[0u8; 16]; 4];
                            let prg_i = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                                &seed[0], &seed[1], &seed[2], &seed[3],
                            ]);
                            let icf = Icf::new(p, q, prg_i);
                            let mut rng = thread_rng();
                            let f = IntvFn {
                                //r_in: InG::from(r_in_u32),
                                r_in: InG::from(0u32),
                                r_out: OutG::from(0u128),
                            };
                            let (k0, k1) = icf.gen(f, &mut rng);
                            tmp.push((k0.serialize()?, k1.serialize()?));
                        }
                        tmp
                    }
                }
                #[cfg(not(feature = "parallel-msb"))]
                {
                    //metrics: measure the genkeys time
                    let _tt = crate::perf_scoped_for_party!(
                        "fss.dealer.genkeys",
                        role,
                        n,            // bucket on the items this block processes
                        bucket_bound  // your desired bucket cap
                    );

                    let mut tmp = Vec::with_capacity(n);
                    for &r_in_u32 in &r_in_list {
                        let seed = [[0u8; 16]; 4];
                        let prg_i = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                            &seed[0], &seed[1], &seed[2], &seed[3],
                        ]);
                        let icf = Icf::new(p, q, prg_i);
                        let mut rng = thread_rng();
                        let f = IntvFn {
                            //r_in: InG::from(r_in_u32),
                            r_in: InG::from(0u32),
                            r_out: OutG::from(0u128),
                        };
                        let (k0, k1) = icf.gen(f, &mut rng);
                        tmp.push((k0.serialize()?, k1.serialize()?));
                    }
                    tmp
                }
            };

            // Flatten and send to P0 (prev) and P1 (next)
            let lens_p0: Vec<u32> = pairs.iter().map(|(k0, _)| k0.len() as u32).collect();
            let lens_p1: Vec<u32> = pairs.iter().map(|(_, k1)| k1.len() as u32).collect();
            let key_blob_for_p0: Vec<u32> = pairs.iter().flat_map(|(k0, _)| k0.clone()).collect();
            let key_blob_for_p1: Vec<u32> = pairs.iter().flat_map(|(_, k1)| k1.clone()).collect();

            //metrics: measure the network time
            let _tt_net0a = crate::perf_scoped_for_party!(
                "fss.network.dealer.send_P0a",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session
                .network_session
                .send_prev(NetworkInt::new_network_vec(u32_to_re_vec(lens_p0.clone())))
                .await?;

            // drop
            drop(_tt_net0a);

            //metrics: measure the network time
            let _tt_net0b = crate::perf_scoped_for_party!(
                "fss.network.dealer.send_P0b",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session
                .network_session
                .send_prev(NetworkInt::new_network_vec(u32_to_re_vec(key_blob_for_p0)))
                .await?;

            // drop
            drop(_tt_net0b);

            //metrics: measure the network time
            let _tt_net1a = crate::perf_scoped_for_party!(
                "fss.network.dealer.send_P1a",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session
                .network_session
                .send_next(NetworkInt::new_network_vec(u32_to_re_vec(lens_p1.clone())))
                .await?;

            // drop
            drop(_tt_net1a);

            //metrics: measure the network time
            let _tt_net1b = crate::perf_scoped_for_party!(
                "fss.network.dealer.send_P1b",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session
                .network_session
                .send_next(NetworkInt::new_network_vec(u32_to_re_vec(key_blob_for_p1)))
                .await?;

            //drop
            drop(_tt_net1b);

            // Dealer (P2)
            // Receive from PREV => P1; from NEXT => P0

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.dealer.recv_P1",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            let p1_bits_msg = session
                .network_session
                .receive_prev()
                .await
                .map_err(|e| eyre!("Dealer cannot receive bit vector from P1: {e}"))?;

            // drop timer
            drop(_tt_net);

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.dealer.recv_P0",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            let p0_bits_msg = session
                .network_session
                .receive_next()
                .await
                .map_err(|e| eyre!("Dealer cannot receive bit vector from P0: {e}"))?;

            // drop timer
            drop(_tt_net);

            let p0_bits_u32: Vec<RingElement<u32>> = u32::into_vec(p0_bits_msg)?;
            let p1_bits_u32: Vec<RingElement<u32>> = u32::into_vec(p1_bits_msg)?;

            // Keep the (P0_bits, P1_bits) ordering to match P0/P1
            let out: Vec<Share<Bit>> = p0_bits_u32
                .into_iter()
                .zip(p1_bits_u32.into_iter())
                .map(|(b0u32, b1u32)| {
                    let b0 = RingElement(Bit::new(b0u32.0 != 0));
                    let b1 = RingElement(Bit::new(b1u32.0 != 0));
                    Share::new(b0, b1)
                })
                .collect();

            Ok(out)
        }

        _ => Err(eyre!("invalid role index {}", role).into()),
    }
}

// Evaluation (P0/P1) and generation (P2) becomes parallel under `parallel-msb` if batch.len > parallel_threshold
// Here r_2 = r_1 = 0, so d2r2=d2 and d1r1=d1
// This is the version without timers, provided for better readability
pub(crate) async fn add_3_get_msb_fss_batch_parallel_threshold(
    session: &mut Session,
    batch: &[Share<u32>],
    parallel_threshold: usize,
) -> Result<Vec<Share<Bit>>, Error> {
    use eyre::eyre;
    use fss_rs::icf::{IcShare, Icf, InG, IntvFn, OutG};
    use fss_rs::prg::Aes128MatyasMeyerOseasPrg;
    use rand::thread_rng;

    let role = session.own_role().index();
    let n = batch.len();

    #[inline]
    fn re_vec_to_u32(v: Vec<RingElement<u32>>) -> Vec<u32> {
        RingElement::<u32>::convert_vec(v)
    }
    #[inline]
    fn u32_to_re_vec(v: Vec<u32>) -> Vec<RingElement<u32>> {
        RingElement::<u32>::convert_vec_rev(v)
    }

    // We test y ∈ [0, 2^31-1] where y = x + r_in + 2^31 (mod 2^32)
    let p = InG::from(0u32);
    //let q = InG::from((1u32 << 31) - 1);
    let q = InG::from(1u32 << 31); // [0, 2^31)
    let n_half_u32: u32 = 1u32 << 31;

    match role {
        // =======================
        // Party 0 (Evaluator)
        // =======================
        0 => {
            // Receive lengths and concatenated key blob from Dealer (P2/prev)
            let lens_re = session.network_session.receive_prev().await.map_err(|e| {
                eyre!("Party 0 cannot receive FSS key lengths batch from dealer: {e}")
            })?;

            let key_blob_re = session
                .network_session
                .receive_prev()
                .await
                .map_err(|e| eyre!("Party 0 cannot receive FSS key batch from dealer: {e}"))?;

            let lens_p0 = re_vec_to_u32(u32::into_vec(lens_re)?);
            let key_blob = re_vec_to_u32(u32::into_vec(key_blob_re)?);

            // Deserialize my keys (batched)
            let mut off = 0usize;
            let mut my_keys = Vec::with_capacity(n);
            for &l in &lens_p0 {
                let len = l as usize;
                my_keys.push(IcShare::deserialize(&key_blob[off..off + len])?);
                off += len;
            }

            // Precompute offsets so we can (optionally) re-deserialize per-index in parallel
            let lens_usize: Vec<usize> = lens_p0.iter().map(|&x| x as usize).collect();
            let mut offs: Vec<usize> = Vec::with_capacity(n);
            {
                let mut acc = 0usize;
                for &len in &lens_usize {
                    offs.push(acc);
                    acc += len;
                }
            }

            // Build a single PRG/ICF for eval (used in the sequential path)
            let seed = [[0u8; 16]; 4];
            let prg = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                &seed[0], &seed[1], &seed[2], &seed[3],
            ]);
            let icf = Icf::new(p, q, prg);

            // ===== batched masked-share exchange =====
            // Send all (d2+r2) to NEXT; receive all (d1+r1) from NEXT

            let send_d2r2: Vec<RingElement<u32>> = batch.iter().map(|x| x.b).collect();
            session
                .network_session
                .send_next(NetworkInt::new_network_vec(send_d2r2))
                .await?;

            let d1r1_msg = session
                .network_session
                .receive_next()
                .await
                .map_err(|e| eyre!("FSS: Party 0 cannot receive d1+r1 vector from P1: {e}"))?;
            let d1r1_vec: Vec<RingElement<u32>> = u32::into_vec(d1r1_msg)?;

            // ===== Evaluate all indices and collect bit shares =====
            // Parallel when feature enabled and n >= threshold; otherwise sequential.
            #[cfg(feature = "parallel-msb")]
            let (bit0s_ringbit, bit0s_u32) = {
                if n >= parallel_threshold {
                    use rayon::prelude::*;
                    let eval_pairs: Vec<(RingElement<Bit>, RingElement<u32>)> = (0..n)
                        .into_par_iter()
                        .map(|i| {
                            let x = &batch[i]; // borrow, do not move
                            let y = x.a + d1r1_vec[i] + x.b + RingElement(n_half_u32);

                            // Rebuild ICF locally per task to avoid shared state
                            let seed = [[0u8; 16]; 4];
                            let prg_i = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                                &seed[0], &seed[1], &seed[2], &seed[3],
                            ]);
                            let icf_i = Icf::new(p, q, prg_i);

                            // Re-deserialize key locally to avoid Sync bounds on IcShare
                            let start = offs[i];
                            let len = lens_usize[i];
                            let key_i = IcShare::deserialize(&key_blob[start..start + len])
                                .expect("deserialize IcShare (P0)");

                            // Evaluate ICF; OutG is 16 bytes BE; take LSB as bit share
                            let f0 = icf_i.eval(false, &key_i, fss_rs::group::int::U32Group(y.0));
                            let f0_u128 = u128::from_le_bytes(f0.0);
                            let b = (f0_u128 & 1) != 0;
                            (
                                RingElement(Bit::new(b)),
                                RingElement(if b { 1u32 } else { 0u32 }),
                            )
                        })
                        .collect();
                    eval_pairs.into_iter().unzip()
                } else {
                    let mut bit0s_ringbit: Vec<RingElement<Bit>> = Vec::with_capacity(n);
                    let mut bit0s_u32: Vec<RingElement<u32>> = Vec::with_capacity(n);
                    for i in 0..n {
                        let x = &batch[i];
                        let y = x.a + d1r1_vec[i] + x.b + RingElement(n_half_u32);
                        let f0 = icf.eval(false, &my_keys[i], fss_rs::group::int::U32Group(y.0));
                        //let f0_u128 = u128::from_be_bytes(f0.0);
                        let f0_u128 = u128::from_le_bytes(f0.0);
                        let b = (f0_u128 & 1) != 0;
                        //let b = fss_out_bit(&f0.0);
                        bit0s_ringbit.push(RingElement(Bit::new(b)));
                        bit0s_u32.push(RingElement(if b { 1u32 } else { 0u32 }));
                    }
                    (bit0s_ringbit, bit0s_u32)
                }
            };

            #[cfg(not(feature = "parallel-msb"))]
            let (bit0s_ringbit, bit0s_u32) = {
                let mut bit0s_ringbit: Vec<RingElement<Bit>> = Vec::with_capacity(n);
                let mut bit0s_u32: Vec<RingElement<u32>> = Vec::with_capacity(n);
                for i in 0..n {
                    let x = &batch[i];
                    let y = x.a + d1r1_vec[i] + x.b + RingElement(n_half_u32);
                    let f0 = icf.eval(false, &my_keys[i], fss_rs::group::int::U32Group(y.0));
                    let f0_u128 = u128::from_le_bytes(f0.0);
                    let b = (f0_u128 & 1) != 0;
                    bit0s_ringbit.push(RingElement(Bit::new(b)));
                    bit0s_u32.push(RingElement(if b { 1u32 } else { 0u32 }));
                }
                (bit0s_ringbit, bit0s_u32)
            };

            // ===== batched bit-share exchange =====
            // Send our bits to BOTH neighbors as a single vector of u32 (0/1)
            session
                .network_session
                .send_prev(NetworkInt::new_network_vec(bit0s_u32.clone()))
                .await?;
            session
                .network_session
                .send_next(NetworkInt::new_network_vec(bit0s_u32))
                .await?;
            // Receive P1's bits (as u32 0/1 vector) from NEXT
            let p1_bits_msg = session
                .network_session
                .receive_next()
                .await
                .map_err(|e| eyre!("Party 0 cannot receive bit vector from P1: {e}"))?;

            let p1_bits_u32: Vec<RingElement<u32>> = u32::into_vec(p1_bits_msg)?;
            let bit1s_ringbit: Vec<RingElement<Bit>> = p1_bits_u32
                .into_iter()
                .map(|re| RingElement(Bit::new(re.0 != 0)))
                .collect();

            // Assemble output shares
            let out: Vec<Share<Bit>> = bit0s_ringbit
                .into_iter()
                .zip(bit1s_ringbit.into_iter())
                .map(|(b0, b1)| Share::new(b0, b1))
                .collect();

            Ok(out)
        }

        // =======================
        // Party 1 (Evaluator)
        // =======================
        1 => {
            // Receive lengths and concatenated key blob from Dealer (P2/next)
            let lens_re = session.network_session.receive_next().await.map_err(|e| {
                eyre!("Party 1 cannot receive FSS key lengths batch from dealer: {e}")
            })?;
            let key_blob_re = session
                .network_session
                .receive_next()
                .await
                .map_err(|e| eyre!("Party 1 cannot receive FSS key batch from dealer: {e}"))?;

            let lens_p1 = re_vec_to_u32(u32::into_vec(lens_re)?);
            let key_blob = re_vec_to_u32(u32::into_vec(key_blob_re)?);

            // Deserialize my keys (batched)
            let mut off = 0usize;
            let mut my_keys = Vec::with_capacity(n);
            for &l in &lens_p1 {
                let len = l as usize;
                my_keys.push(IcShare::deserialize(&key_blob[off..off + len])?);
                off += len;
            }

            // Precompute offsets for optional per-iteration re-deserialization
            let lens_usize: Vec<usize> = lens_p1.iter().map(|&x| x as usize).collect();
            let mut offs: Vec<usize> = Vec::with_capacity(n);
            {
                let mut acc = 0usize;
                for &len in &lens_usize {
                    offs.push(acc);
                    acc += len;
                }
            }

            // Build a single PRG/ICF for eval (used in the sequential path)
            let seed = [[0u8; 16]; 4];
            let prg = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                &seed[0], &seed[1], &seed[2], &seed[3],
            ]);
            let icf = Icf::new(p, q, prg);

            // ===== batched masked-share exchange =====
            // Send all (d1+r1) to PREV; receive all (d2+r2) from PREV
            let send_d1r1: Vec<RingElement<u32>> = batch.iter().map(|x| x.a).collect();
            session
                .network_session
                .send_prev(NetworkInt::new_network_vec(send_d1r1))
                .await?;

            let d2r2_msg = session
                .network_session
                .receive_prev()
                .await
                .map_err(|e| eyre!("FSS: Party 1 cannot receive d2+r2 vector from P0: {e}"))?;
            let d2r2_vec: Vec<RingElement<u32>> = u32::into_vec(d2r2_msg)?;

            // ===== Evaluate all indices and collect bit shares =====
            #[cfg(feature = "parallel-msb")]
            let (bit1s_ringbit, bit1s_u32) = {
                if n >= parallel_threshold {
                    use rayon::prelude::*;
                    let eval_pairs: Vec<(RingElement<Bit>, RingElement<u32>)> = (0..n)
                        .into_par_iter()
                        .map(|i| {
                            let x = &batch[i]; // borrow, do not move
                            let y = x.a + d2r2_vec[i] + x.b + RingElement(n_half_u32);

                            // Rebuild ICF locally
                            let seed = [[0u8; 16]; 4];
                            let prg_i = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                                &seed[0], &seed[1], &seed[2], &seed[3],
                            ]);
                            let icf_i = Icf::new(p, q, prg_i);

                            // Re-deserialize key locally
                            let start = offs[i];
                            let len = lens_usize[i];
                            let key_i = IcShare::deserialize(&key_blob[start..start + len])
                                .expect("deserialize IcShare (P1)");

                            // Evaluate ICF; OutG is 16 bytes BE; take LSB as bit share
                            let f1 = icf_i.eval(true, &key_i, fss_rs::group::int::U32Group(y.0));
                            let f1_u128 = u128::from_le_bytes(f1.0);
                            let b = (f1_u128 & 1) != 0;
                            (
                                RingElement(Bit::new(b)),
                                RingElement(if b { 1u32 } else { 0u32 }),
                            )
                        })
                        .collect();
                    eval_pairs.into_iter().unzip()
                } else {
                    let mut bit1s_ringbit: Vec<RingElement<Bit>> = Vec::with_capacity(n);
                    let mut bit1s_u32: Vec<RingElement<u32>> = Vec::with_capacity(n);
                    for i in 0..n {
                        let x = &batch[i];
                        let y = x.a + d2r2_vec[i] + x.b + RingElement(n_half_u32);
                        let f1 = icf.eval(true, &my_keys[i], fss_rs::group::int::U32Group(y.0));
                        let f1_u128 = u128::from_le_bytes(f1.0);
                        let b = (f1_u128 & 1) != 0;
                        bit1s_ringbit.push(RingElement(Bit::new(b)));
                        bit1s_u32.push(RingElement(if b { 1u32 } else { 0u32 }));
                    }
                    (bit1s_ringbit, bit1s_u32)
                }
            };

            #[cfg(not(feature = "parallel-msb"))]
            let (bit1s_ringbit, bit1s_u32) = {
                let mut bit1s_ringbit: Vec<RingElement<Bit>> = Vec::with_capacity(n);
                let mut bit1s_u32: Vec<RingElement<u32>> = Vec::with_capacity(n);
                for i in 0..n {
                    let x = &batch[i];
                    let y = x.a + d2r2_vec[i] + x.b + RingElement(n_half_u32);
                    let f1 = icf.eval(true, &my_keys[i], fss_rs::group::int::U32Group(y.0));
                    let f1_u128: u128 = u128::from_le_bytes(f1.0);
                    let b = (f1_u128 & 1) != 0;
                    bit1s_ringbit.push(RingElement(Bit::new(b)));
                    bit1s_u32.push(RingElement(if b { 1u32 } else { 0u32 }));
                }
                (bit1s_ringbit, bit1s_u32)
            };

            // ===== batched bit-share exchange =====
            // Send our bits to BOTH neighbors as a single vector of u32 (0/1)
            session
                .network_session
                .send_next(NetworkInt::new_network_vec(bit1s_u32.clone()))
                .await?;
            session
                .network_session
                .send_prev(NetworkInt::new_network_vec(bit1s_u32))
                .await?;

            // Receive P0's bits (as u32 0/1 vector) from PREV
            let p0_bits_msg = session
                .network_session
                .receive_prev()
                .await
                .map_err(|e| eyre!("Party 1 cannot receive bit vector from P0: {e}"))?;

            let p0_bits_u32: Vec<RingElement<u32>> = u32::into_vec(p0_bits_msg)?;
            let bit0s_ringbit: Vec<RingElement<Bit>> = p0_bits_u32
                .into_iter()
                .map(|re| RingElement(Bit::new(re.0 != 0)))
                .collect();

            // Assemble output shares
            let out: Vec<Share<Bit>> = bit0s_ringbit
                .into_iter()
                .zip(bit1s_ringbit.into_iter())
                .map(|(b0, b1)| Share::new(b0, b1))
                .collect();

            Ok(out)
        }

        // =======================
        // Party 2 (Dealer)
        // =======================
        2 => {
            // // Build r_in (input masks) using the session PRF; must match evaluators' PRF consumption
            // let mut r_in_list: Vec<u32> = Vec::with_capacity(n);
            // for _ in 0..n {
            //     let (r_next, r_prev) = session.prf.gen_rands::<RingElement<u32>>().clone();
            //     let r_in = (r_next + r_prev).convert();
            //     r_in_list.push(r_in);
            // }

            // This variant assumes r1 = r2 = 0 on evaluators, so use r_in = 0 here as well.
            // Otherwise keys would be generated for x + r_in while evaluators feed x.
            let r_in_list: Vec<u32> = vec![0u32; n];

            // Fresh ICF.Gen per index, r_out = 0; adaptive parallelism by threshold
            // First collect key pairs, then pre-size and flatten to avoid reallocs.
            let pairs: Vec<(Vec<u32>, Vec<u32>)> = {
                #[cfg(feature = "parallel-msb")]
                {
                    if n >= parallel_threshold {
                        use rayon::prelude::*;
                        r_in_list
                            .par_iter()
                            .map(|&r_in_u32| {
                                let seed = [[0u8; 16]; 4];
                                let prg_i = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                                    &seed[0], &seed[1], &seed[2], &seed[3],
                                ]);
                                let icf = Icf::new(p, q, prg_i);
                                let mut rng = thread_rng();
                                let f = IntvFn {
                                    //r_in: InG::from(r_in_u32),
                                    r_in: InG::from(0u32),
                                    r_out: OutG::from(0u128),
                                };
                                let (k0, k1) = icf.gen(f, &mut rng);
                                (k0.serialize().unwrap(), k1.serialize().unwrap())
                            })
                            .collect()
                    } else {
                        let mut tmp = Vec::with_capacity(n);
                        for &r_in_u32 in &r_in_list {
                            let seed = [[0u8; 16]; 4];
                            let prg_i = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                                &seed[0], &seed[1], &seed[2], &seed[3],
                            ]);
                            let icf = Icf::new(p, q, prg_i);
                            let mut rng = thread_rng();
                            let f = IntvFn {
                                //r_in: InG::from(r_in_u32),
                                r_in: InG::from(0u32),
                                r_out: OutG::from(0u128),
                            };
                            let (k0, k1) = icf.gen(f, &mut rng);
                            tmp.push((k0.serialize()?, k1.serialize()?));
                        }
                        tmp
                    }
                }
                #[cfg(not(feature = "parallel-msb"))]
                {
                    let mut tmp = Vec::with_capacity(n);
                    for &r_in_u32 in &r_in_list {
                        let seed = [[0u8; 16]; 4];
                        let prg_i = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                            &seed[0], &seed[1], &seed[2], &seed[3],
                        ]);
                        let icf = Icf::new(p, q, prg_i);
                        let mut rng = thread_rng();
                        let f = IntvFn {
                            //r_in: InG::from(r_in_u32),
                            r_in: InG::from(0u32),
                            r_out: OutG::from(0u128),
                        };
                        let (k0, k1) = icf.gen(f, &mut rng);
                        tmp.push((k0.serialize()?, k1.serialize()?));
                    }
                    tmp
                }
            };

            // Flatten and send to P0 (prev) and P1 (next)
            let lens_p0: Vec<u32> = pairs.iter().map(|(k0, _)| k0.len() as u32).collect();
            let lens_p1: Vec<u32> = pairs.iter().map(|(_, k1)| k1.len() as u32).collect();
            let key_blob_for_p0: Vec<u32> = pairs.iter().flat_map(|(k0, _)| k0.clone()).collect();
            let key_blob_for_p1: Vec<u32> = pairs.iter().flat_map(|(_, k1)| k1.clone()).collect();

            session
                .network_session
                .send_prev(NetworkInt::new_network_vec(u32_to_re_vec(lens_p0.clone())))
                .await?;
            session
                .network_session
                .send_prev(NetworkInt::new_network_vec(u32_to_re_vec(key_blob_for_p0)))
                .await?;
            session
                .network_session
                .send_next(NetworkInt::new_network_vec(u32_to_re_vec(lens_p1.clone())))
                .await?;
            session
                .network_session
                .send_next(NetworkInt::new_network_vec(u32_to_re_vec(key_blob_for_p1)))
                .await?;

            // Dealer (P2)
            // Receive from PREV => P1; from NEXT => P0
            let p1_bits_msg = session
                .network_session
                .receive_prev()
                .await
                .map_err(|e| eyre!("Dealer cannot receive bit vector from P1: {e}"))?;
            let p0_bits_msg = session
                .network_session
                .receive_next()
                .await
                .map_err(|e| eyre!("Dealer cannot receive bit vector from P0: {e}"))?;

            let p0_bits_u32: Vec<RingElement<u32>> = u32::into_vec(p0_bits_msg)?;
            let p1_bits_u32: Vec<RingElement<u32>> = u32::into_vec(p1_bits_msg)?;

            // Keep the (P0_bits, P1_bits) ordering to match P0/P1
            let out: Vec<Share<Bit>> = p0_bits_u32
                .into_iter()
                .zip(p1_bits_u32.into_iter())
                .map(|(b0u32, b1u32)| {
                    let b0 = RingElement(Bit::new(b0u32.0 != 0));
                    let b1 = RingElement(Bit::new(b1u32.0 != 0));
                    Share::new(b0, b1)
                })
                .collect();

            Ok(out)
        }

        _ => Err(eyre!("invalid role index {}", role).into()),
    }
}

/// Batched version of the function above
/// Instead of handling one request at a time, get a batch of size ???
/// Main differences: each party
pub(crate) async fn add_3_get_msb_fss_batch_timers(
    session: &mut Session,
    x: &[Share<u32>],
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
    let n = x.len();
    let batch_size = x.len();
    let bucket_bound = 150;

    // Depending on the role, do different stuff
    match role {
        0 => {
            //Generate all r2 prf key, keep the r0 keys for later
            // println!("party 0: Batch size is {}", batch_size);
            let mut r_prime_keys = Vec::with_capacity(batch_size);
            let mut d2r2_vec = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let (r_prime_temp, _) = session.prf.gen_rands::<RingElement<u32>>().clone(); //_ is r2
                r_prime_keys.push(RingElement::<u128>(u128::from(r_prime_temp.0))); //convet to this for later
                d2r2_vec.push(x[i].b + RingElement(0)); // change this to take the second thing gen_rands returns
            }

            // Send the vector of d2+r2 to party 1
            let clone_d2r2_vec = d2r2_vec.clone();

            //metrics: measure the network for share reconstruction
            let _tt_net_recon = crate::perf_scoped_for_party!(
                "fss.network.recon.send",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session
                .network_session
                .send_next(u32::new_network_vec(clone_d2r2_vec))
                .await?;

            // drop timer
            drop(_tt_net_recon);

            //metrics: measure the network for share reconstruction
            let _tt_net_recon = crate::perf_scoped_for_party!(
                "fss.network.recon.recv",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Receive d1+r1 from party 1
            let d1r1 = match session.network_session.receive_next().await {
                Ok(v) => u32::into_vec(v),
                Err(e) => Err(eyre!("FSS: Party 0 cannot receive d1+r1 from party 1: {e}")),
            }?;

            // drop timer
            drop(_tt_net_recon);

            // metrics: measure the network for share reconstruction
            let _tt_net_recv = crate::perf_scoped_for_party!(
                "fss.network.start_recv_keys",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Receive batch_size number of fss keys from dealer
            let k_fss_0_vec = match session.network_session.receive_prev().await {
                Ok(v) => u32::into_vec(v),
                Err(e) => Err(eyre!("Party 0 cannot receive my fss key from dealer {e}")),
            }?;

            // drop timer
            drop(_tt_net_recv);

            // Set up the function for FSS
            // we need this below to handle signed numbers, if input is unsigned no need to add N/2
            let n_falf_u32 = 1u32 << 31;
            let n_half = InG::from(n_falf_u32);
            // make the interval so that we return 1 when MSB == 1
            // this is (our number + n/2 ) % n, modulo is handled by U32Group
            let p = InG::from(1u32 << 31) + n_half;
            let q = InG::from(u32::MAX) + n_half; // modulo is handled by U32Group
            let keys: Vec<[u8; 16]> = vec![[0u8; 16]; 4];
            let mut f_x_0_bits = Vec::with_capacity(batch_size); // store all the eval results

            // Deserialize each to find original IcShare and call eval
            let key_words_fss_0: Vec<u32> = RingElement::<u32>::convert_vec(k_fss_0_vec); //need to un-flatten key vector
            let mut offset: usize = 0;
            for i in 0..batch_size {
                // // Need to "unflatten" to get batch_size number of fss keys
                let curr_key_byte_len = 1 + (key_words_fss_0[offset] as usize + 3) / 4; // offset index has the byte length, then find total u32s for this key

                // Get current key
                let k_fss_0_icshare: IcShare =
                    IcShare::deserialize(&key_words_fss_0[offset..offset + curr_key_byte_len])?;
                offset += curr_key_byte_len; //update offset to point to next cell that contains size of next key

                // reconstruct the input d+r [recall x.a=d0] for each x[i]
                let d_plus_r: RingElement<u32> =
                    d1r1[i] + d2r2_vec[i] + x[i].a + RingElement(n_falf_u32);
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
                        fss_rs::group::int::U32Group(d_plus_r.0),
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

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.send_next",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session // send to party 1
                .network_session
                .send_next(NetworkValue::vec_to_network(cloned_f_0_res_network))
                .await?;

            // drop timer
            drop(_tt_net);

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.send_prev",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session // send to the dealer (party 2)
                .network_session
                .send_prev(NetworkValue::vec_to_network(f_0_res_network))
                .await?;

            // drop timer
            drop(_tt_net);

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.recv",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Receive Bits of share of party 1 --> this is a vec of network values
            let f_x_1_bits_net = match session.network_session.receive_next().await {
                Ok(v) => NetworkValue::vec_from_network(v),
                Err(e) => return Err(eyre!("Party 0 cannot receive bit shares from party 1: {e}")),
            }?;

            // drop timer
            drop(_tt_net);

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
            // eprintln!("party 1: Batch size is {}", batch_size);
            std::io::stderr().flush().ok();
            let mut r_prime_keys = Vec::with_capacity(batch_size);
            let mut d1r1_vec = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let (_, r_prime_temp) = session.prf.gen_rands::<RingElement<u32>>().clone(); //_ is r1
                r_prime_keys.push(RingElement::<u128>(u128::from(r_prime_temp.0))); //convet to this for later
                d1r1_vec.push(x[i].a + RingElement(0)); // change this to take the first thing gen_rands returns
            }

            // Send the vector of d1+r1 to party 0
            let cloned_d1r1_vec = d1r1_vec.clone();

            //metrics: measure the network for share reconstruction
            let _tt_net_recon = crate::perf_scoped_for_party!(
                "fss.network.recon.send",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session
                .network_session
                .send_prev(u32::new_network_vec(cloned_d1r1_vec))
                .await?;

            // drop timer
            drop(_tt_net_recon);

            //metrics: measure the network for share reconstruction
            let _tt_net_recon = crate::perf_scoped_for_party!(
                "fss.network.recon.recv",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Receive d2+r2 from party 0
            let d2r2_vec = match session.network_session.receive_prev().await {
                Ok(v) => u32::into_vec(v),
                Err(e) => Err(eyre!("FSS: Party 1 cannot receive d2+r2 from party 0: {e}")),
            }?;

            // drop timer
            drop(_tt_net_recon);

            // metrics: measure the network for share reconstruction
            let _tt_net_recv = crate::perf_scoped_for_party!(
                "fss.network.start_recv_keys",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Receive batch_size number of fss keys from dealer
            let k_fss_1_vec = match session.network_session.receive_next().await {
                Ok(v) => u32::into_vec(v),
                Err(e) => Err(eyre!("Party 1 cannot receive my fss key from dealer {e}")),
            }?;

            // drop timer
            drop(_tt_net_recv);

            // Set up the function for FSS
            // we need this below to handle signed numbers, if input is unsigned no need to add N/2
            let n_falf_u32 = 1u32 << 31;
            let n_half = InG::from(n_falf_u32);
            // make the interval so that we return 1 when MSB == 1
            // this is (our number + n/2 ) % n, modulo is handled by U32Group
            let p = InG::from(1u32 << 31) + n_half;
            let q = InG::from(u32::MAX) + n_half; // modulo is handled by U32Group
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
                let d_plus_r: RingElement<u32> =
                    d1r1_vec[i] + d2r2_vec[i] + x[i].b + RingElement(n_falf_u32);
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
                        fss_rs::group::int::U32Group(d_plus_r.0),
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

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.send_prev",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session // send to party 0
                .network_session
                .send_prev(NetworkValue::vec_to_network(cloned_f_1_res_network))
                .await?;

            // drop timer
            drop(_tt_net);

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.send_next",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session // send to the dealer (party 2)
                .network_session
                .send_next(NetworkValue::vec_to_network(f_1_res_network))
                .await?;

            // drop timer
            drop(_tt_net);

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.recv",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Receive Bits of share of party 0 --> this is a vec of network values
            let f_x_0_bits_net = match session.network_session.receive_prev().await {
                Ok(v) => NetworkValue::vec_from_network(v),
                Err(e) => return Err(eyre!("Party 0 cannot receive bit shares from party 1: {e}")),
            }?;

            // drop timer
            drop(_tt_net);

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
            // Setting up the Interval Containment function
            // we need this to handle signed numbers, if input is unsigned no need to add N/2
            let n_half = InG::from(1u32 << 31);
            let keys: Vec<[u8; 16]> = vec![[0u8; 16]; 4];

            // make the interval so that we return 1 when MSB == 1
            // this is (our number + n/2 ) % n, modulo is handled by U32Group
            let p = InG::from(1u32 << 31) + n_half;
            let q = InG::from(u32::MAX) + n_half; // modulo is handled by U32Group
                                                  // println!("Interval is p={p:?}, q={q:?}");

            let mut k_fss_0_vec_flat = Vec::with_capacity(batch_size); // to store the fss keys
            let mut k_fss_1_vec_flat = Vec::with_capacity(batch_size);

            //metrics: measure the genkeys time
            let _tt = crate::perf_scoped_for_party!(
                "fss.dealer.genkeys",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            for _i in 0..batch_size {
                // Draw r1 + r2 (aka r_in)
                let (_r2, _r1) = session.prf.gen_rands::<RingElement<u32>>().clone();
                let r2 = RingElement(0);
                let r1 = RingElement(0);

                let r1_plus_r2_u32: u32 = (r1 + r2).convert();
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

                let temp_key0 = k_fss_0_pre_ser.serialize()?;
                k_fss_0_vec_flat.extend(RingElement::<u32>::convert_vec_rev(temp_key0.clone()));

                let temp_key1 = k_fss_1_pre_ser.serialize()?;
                k_fss_1_vec_flat.extend(RingElement::<u32>::convert_vec_rev(temp_key1.clone()));
            }

            // drop timer
            drop(_tt);

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.dealer.send_P0",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Send the flattened FSS keys to parties 0 and 1, so they can do Eval
            session
                .network_session
                .send_next(NetworkInt::new_network_vec(k_fss_0_vec_flat))
                .await?; //next is party 0

            // drop timer
            drop(_tt_net);

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.dealer.send_P1",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session
                .network_session
                .send_prev(NetworkInt::new_network_vec(k_fss_1_vec_flat))
                .await?; //previous is party 1

            // drop timer
            drop(_tt_net);

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.dealer.recv_P0",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Receive bit of share from party 0
            let f_x_0_bits_net = match session.network_session.receive_next().await {
                Ok(v) => NetworkValue::vec_from_network(v),
                Err(e) => return Err(eyre!("Party 0 cannot receive bit shares from party 1: {e}")),
            }?;

            // drop timer
            drop(_tt_net);

            // Convert Vec<NetworkValue> to Vec<RingElement<Bit>>
            let f_x_0_bits: Vec<RingElement<Bit>> = f_x_0_bits_net
                .into_iter()
                .map(|nv| match nv {
                    NetworkValue::RingElementBit(b) => Ok(b),
                    other => Err(eyre!("expected RingElementBit, got {:?}", other)),
                })
                .collect::<Result<_, _>>()?;

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.dealer.recv_P1",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            // Receive Bits of share of party 1 --> this is a vec of network values
            let f_x_1_bits_net = match session.network_session.receive_prev().await {
                Ok(v) => NetworkValue::vec_from_network(v),
                Err(e) => return Err(eyre!("Party 0 cannot receive bit shares from party 1: {e}")),
            }?;

            // drop timer
            drop(_tt_net);

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
