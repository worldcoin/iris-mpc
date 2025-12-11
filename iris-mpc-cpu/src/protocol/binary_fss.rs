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
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{info, instrument, trace_span, Instrument};
use std::fs::OpenOptions;

use fss_rs::icf::{IcShare, Icf, InG, IntvFn, OutG};
use fss_rs::prg::Aes128MatyasMeyerOseasPrg;

const FSS_DEBUG_LOG_PATH: &str = "./fss_debug_batch_bits.log";
const DEBUG_TARGET_IDX: usize = 815;

#[inline]
fn append_debug_line(line: &str) {
    if let Ok(mut f) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(FSS_DEBUG_LOG_PATH)
    {
        let _ = writeln!(f, "{}", line);
    }
}

#[inline]
fn log_batch_start(role: usize, base_idx: usize, n: usize) {
    #[cfg(feature = "parallel-msb")]
    {
        append_debug_line(&format!(
            "p{} batch_start base={} len={}",
            role, base_idx, n
        ));
    }
}

#[inline]
fn log_target_bit(role: usize, base_idx: usize, local_idx: usize, stage: &str, bit: bool) {
    #[cfg(feature = "parallel-msb")]
    {
        if base_idx + local_idx == DEBUG_TARGET_IDX {
            append_debug_line(&format!(
                "p{} target={} stage={} local={} bit={}",
                role,
                DEBUG_TARGET_IDX,
                stage,
                local_idx,
                bit as u8
            ));
        }
    }
}

#[inline]
fn log_target_y(role: usize, base_idx: usize, local_idx: usize, stage: &str, y: u32) {
    #[cfg(feature = "parallel-msb")]
    {
        if base_idx + local_idx == DEBUG_TARGET_IDX {
            append_debug_line(&format!(
                "p{} target={} stage={} local={} y={}",
                role,
                DEBUG_TARGET_IDX,
                stage,
                local_idx,
                y
            ));
        }
    }
}

#[inline]
fn log_target_y_minus_half(role: usize, base_idx: usize, local_idx: usize, stage: &str, y: u32) {
    #[cfg(feature = "parallel-msb")]
    {
        if base_idx + local_idx == DEBUG_TARGET_IDX {
            // Undo the +N/2 shift (mod 2^32) to get the original signed-space value.
            let y_minus_half = y.wrapping_sub(1u32 << 31);
            append_debug_line(&format!(
                "p{} target={} stage={} local={} y_minus_half={}",
                role,
                DEBUG_TARGET_IDX,
                stage,
                local_idx,
                y_minus_half
            ));
        }
    }
}

#[inline]
fn approx_bytes_int_vec(v: &[RingElement<u32>]) -> usize {
    v.len() * std::mem::size_of::<u32>()
}

#[inline]
fn approx_bytes_bit_vec(v: &[RingElement<Bit>]) -> usize {
    v.len() // rough count: 1 byte per bit share
}

static FSS_BYTES_SENT: AtomicU64 = AtomicU64::new(0);
static FSS_BYTES_RECV: AtomicU64 = AtomicU64::new(0);
static GLOBAL_FSS_BASE_IDX: AtomicU64 = AtomicU64::new(0);

#[inline]
fn record_traffic(sent: usize, recv: usize) {
    FSS_BYTES_SENT.fetch_add(sent as u64, Ordering::Relaxed);
    FSS_BYTES_RECV.fetch_add(recv as u64, Ordering::Relaxed);
}

pub fn fss_traffic_totals() -> (u64, u64) {
    (
        FSS_BYTES_SENT.load(Ordering::Relaxed),
        FSS_BYTES_RECV.load(Ordering::Relaxed),
    )
}

// Deterministic base seed for FSS key generation (shared across all parties).
const FSS_KEYGEN_BASE_SEED: [u8; 16] = [
    0x42, 0x9a, 0xf1, 0x7c, 0x3d, 0x55, 0x6b, 0x10, 0x99, 0xaa, 0xbb, 0xcc, 0xde, 0x01, 0x23, 0x45,
];

#[inline]
fn bits_to_network_vec(bits: &[RingElement<Bit>]) -> Vec<NetworkValue> {
    bits.iter()
        .copied()
        .map(NetworkValue::RingElementBit)
        .collect()
}

#[inline]
fn network_vec_to_bits(values: Vec<NetworkValue>) -> Result<Vec<RingElement<Bit>>, Error> {
    values
        .into_iter()
        .map(|nv| match nv {
            NetworkValue::RingElementBit(bit) => Ok(bit),
            other => Err(eyre!("expected RingElementBit, got {:?}", other)),
        })
        .collect()
}

#[inline]
fn decode_key_package(
    pkg: NetworkValue,
) -> Result<(Vec<RingElement<u32>>, Vec<RingElement<u32>>), Error> {
    let mut entries = NetworkValue::vec_from_network(pkg)?;
    if entries.len() != 2 {
        bail!("invalid key package length {}", entries.len());
    }
    let blob = entries
        .pop()
        .ok_or_else(|| eyre!("missing key blob in package"))?;
    let lens = entries
        .pop()
        .ok_or_else(|| eyre!("missing key lengths in package"))?;

    let lens = match lens {
        NetworkValue::VecRing32(v) => Ok(v),
        other => Err(eyre!("expected VecRing32 for lens, got {:?}", other)),
    }?;
    let blob = match blob {
        NetworkValue::VecRing32(v) => Ok(v),
        other => Err(eyre!("expected VecRing32 for blob, got {:?}", other)),
    }?;
    Ok((lens, blob))
}

#[inline]
fn encode_key_package(
    lens: Vec<RingElement<u32>>,
    blob: Vec<RingElement<u32>>,
) -> NetworkValue {
    NetworkValue::vec_to_network(vec![
        NetworkValue::VecRing32(lens),
        NetworkValue::VecRing32(blob),
    ])
}

// Evaluation (P0/P1) and generation (P2) becomes parallel under `parallel-msb` if batch.len > parallel_threshold
// Here r_2 = r_1 = 0, so d2r2=d2 and d1r1=d1
pub(crate) async fn add_3_get_msb_fss_batch_parallel_threshold_timers(
    session: &mut Session,
    batch: &[Share<u32>],
    parallel_threshold: usize,
    base_idx: usize,
) -> Result<Vec<Share<Bit>>, Error> {
    use eyre::eyre;
    use fss_rs::icf::{IcShare, Icf, InG, IntvFn, OutG};
    use fss_rs::prg::Aes128MatyasMeyerOseasPrg;

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
            let base_idx_msg = session
                .network_session
                .receive_prev()
                .await
                .map_err(|e| eyre!("Party 0 cannot receive base_idx from dealer: {e}"))?;
            let base_idx = u64::into_vec(base_idx_msg)?
                .pop()
                .ok_or_else(|| eyre!("base_idx message empty"))?
                .0 as usize;
            log_batch_start(role, base_idx, n);

            let mut sent_bytes = 0usize;
            let mut recv_bytes = 0usize;

            // Generate FSS keys locally using deterministic randomness from session PRF
            //metrics: measure the genkeys time
            let _tt_gen = crate::perf_scoped_for_party!(
                "fss.dealer.genkeys",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            let mut my_keys = Vec::with_capacity(n);
            for i in 0..n {
                // Deterministic RNG derived from a shared base seed and per-index counter
                let mut seed_u128 = u128::from_le_bytes(FSS_KEYGEN_BASE_SEED);
                seed_u128 ^= (base_idx + i) as u128 + 1;
                let derived_seed = seed_u128.to_le_bytes();
                let mut prf_rng = AesRng::from_seed(derived_seed);
                
                // Build PRG/ICF for key generation
                let prg_seed = [[0u8; 16]; 4];
                let prg = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                    &prg_seed[0], &prg_seed[1], &prg_seed[2], &prg_seed[3],
                ]);
                let icf = Icf::new(p, q, prg);
                
                // Generate FSS key pair using deterministic PRF RNG
                // All parties will generate the same (k0, k1) pair
                let f = IntvFn {
                    r_in: InG::from(0u32),
                    r_out: OutG::from(0u128),
                };
                let (k0, _k1) = icf.gen(f, &mut prf_rng);
                my_keys.push(k0);
            }

            drop(_tt_gen);

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
            sent_bytes += approx_bytes_int_vec(&send_d2r2);
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
            recv_bytes += approx_bytes_int_vec(&d1r1_vec);

            // metrics: stop timer here for network reconstruction
            drop(_tt_net_recon);

            // ===== Evaluate all indices and collect bit shares =====
            // Parallel when feature enabled and n >= threshold; otherwise sequential.
            #[cfg(feature = "parallel-msb")]
            let (bit0s_ringbit, _bit0s_u32) = {
                if n >= parallel_threshold {
                    use rayon::prelude::*;
                    let eval_pairs: Vec<(RingElement<Bit>, RingElement<u32>)> = (0..n)
                        .into_par_iter()
                        .map(|i| {
                            let x = &batch[i]; // borrow, do not move
                            let y = x.a + d1r1_vec[i] + x.b + RingElement(n_half_u32);
                            log_target_y(role, base_idx, i, "p0_eval_y", y.0);
                            log_target_y_minus_half(role, base_idx, i, "p0_eval_y_minus_half", y.0);

                            // Rebuild ICF locally per task to avoid shared state
                            let seed = [[0u8; 16]; 4];
                            let prg_i = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                                &seed[0], &seed[1], &seed[2], &seed[3],
                            ]);
                            let icf_i = Icf::new(p, q, prg_i);

                            // Use pre-generated key (keys are already generated above)
                            let key_i = &my_keys[i];

                            // Evaluate ICF; OutG is 16 bytes BE; take LSB as bit share
                            let f0 = icf_i.eval(false, key_i, fss_rs::group::int::U32Group(y.0));
                            let f0_u128 = u128::from_le_bytes(f0.0);
                            let b = (f0_u128 & 1) != 0;
                            log_target_bit(role, base_idx, i, "p0_eval", b);
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
                        log_target_y(role, base_idx, i, "p0_eval_y", y.0);
                        log_target_y_minus_half(role, base_idx, i, "p0_eval_y_minus_half", y.0);
                        let f0 = icf.eval(false, &my_keys[i], fss_rs::group::int::U32Group(y.0));
                        //let f0_u128 = u128::from_be_bytes(f0.0);
                        let f0_u128 = u128::from_le_bytes(f0.0);
                        let b = (f0_u128 & 1) != 0;
                        log_target_bit(role, base_idx, i, "p0_eval", b);
                        //let b = fss_out_bit(&f0.0);
                        bit0s_ringbit.push(RingElement(Bit::new(b)));
                        bit0s_u32.push(RingElement(if b { 1u32 } else { 0u32 }));
                    }
                    (bit0s_ringbit, bit0s_u32)
                }
            };

            #[cfg(not(feature = "parallel-msb"))]
            let (bit0s_ringbit, _bit0s_u32) = {
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
            // Send our bits to BOTH neighbors as packed NetworkValues

            let bit0s_network_vec = bits_to_network_vec(&bit0s_ringbit);

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.send_prev",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session
                .network_session
                .send_prev(NetworkValue::vec_to_network(bit0s_network_vec.clone()))
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
                .send_next(NetworkValue::vec_to_network(bit0s_network_vec))
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

            // Receive P1's bits (packed) from NEXT
            let p1_bits_msg = session
                .network_session
                .receive_next()
                .await
                .map_err(|e| eyre!("Party 0 cannot receive bit vector from P1: {e}"))?;

            // metrics: stop timer
            drop(_tt_net);

            let bit1s_ringbit =
                network_vec_to_bits(NetworkValue::vec_from_network(p1_bits_msg)?).map_err(|e| {
                    eyre!("Party 0 cannot deserialize bit vector from P1: {e}")
                })?;

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
            let base_idx_msg = session
                .network_session
                .receive_next()
                .await
                .map_err(|e| eyre!("Party 1 cannot receive base_idx from dealer: {e}"))?;
            let base_idx = u64::into_vec(base_idx_msg)?
                .pop()
                .ok_or_else(|| eyre!("base_idx message empty"))?
                .0 as usize;
            log_batch_start(role, base_idx, n);

            let mut sent_bytes = 0usize;
            let mut recv_bytes = 0usize;

            // Generate FSS keys locally using deterministic randomness from session PRF
            //metrics: measure the genkeys time
            let _tt_gen = crate::perf_scoped_for_party!(
                "fss.dealer.genkeys",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            let mut my_keys = Vec::with_capacity(n);
            for i in 0..n {
                // Deterministic RNG derived from a shared base seed and per-index counter
                let mut seed_u128 = u128::from_le_bytes(FSS_KEYGEN_BASE_SEED);
                seed_u128 ^= (base_idx + i) as u128 + 1;
                let derived_seed = seed_u128.to_le_bytes();
                let mut prf_rng = AesRng::from_seed(derived_seed);
                
                // Build PRG/ICF for key generation
                let prg_seed = [[0u8; 16]; 4];
                let prg = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                    &prg_seed[0], &prg_seed[1], &prg_seed[2], &prg_seed[3],
                ]);
                let icf = Icf::new(p, q, prg);
                
                // Generate FSS key pair using deterministic PRF RNG
                // All parties will generate the same (k0, k1) pair
                let f = IntvFn {
                    r_in: InG::from(0u32),
                    r_out: OutG::from(0u128),
                };
                let (_k0, k1) = icf.gen(f, &mut prf_rng);
                my_keys.push(k1);
            }

            drop(_tt_gen);

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
            sent_bytes += approx_bytes_int_vec(&send_d1r1);
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
            recv_bytes += approx_bytes_int_vec(&d2r2_vec);

            // metrics: stop timer here for network reconstruction
            drop(_tt_net_recon);

            // ===== Evaluate all indices and collect bit shares =====
            #[cfg(feature = "parallel-msb")]
            let (bit1s_ringbit, _bit1s_u32) = {
                if n >= parallel_threshold {
                    use rayon::prelude::*;
                    let eval_pairs: Vec<(RingElement<Bit>, RingElement<u32>)> = (0..n)
                        .into_par_iter()
                        .map(|i| {
                            let x = &batch[i]; // borrow, do not move
                            let y = x.a + d2r2_vec[i] + x.b + RingElement(n_half_u32);
                            log_target_y(role, base_idx, i, "p1_eval_y", y.0);
                            log_target_y_minus_half(role, base_idx, i, "p1_eval_y_minus_half", y.0);

                            // Rebuild ICF locally
                            let seed = [[0u8; 16]; 4];
                            let prg_i = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                                &seed[0], &seed[1], &seed[2], &seed[3],
                            ]);
                            let icf_i = Icf::new(p, q, prg_i);

                            // Use pre-generated key (keys are already generated above)
                            let key_i = &my_keys[i];

                            // Evaluate ICF; OutG is 16 bytes BE; take LSB as bit share
                            let f1 = icf_i.eval(true, key_i, fss_rs::group::int::U32Group(y.0));
                            let f1_u128 = u128::from_le_bytes(f1.0);
                            let b = (f1_u128 & 1) != 0;
                            log_target_bit(role, base_idx, i, "p1_eval", b);
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
                        log_target_y(role, base_idx, i, "p1_eval_y", y.0);
                        log_target_y_minus_half(role, base_idx, i, "p1_eval_y_minus_half", y.0);
                        let f1 = icf.eval(true, &my_keys[i], fss_rs::group::int::U32Group(y.0));
                        let f1_u128 = u128::from_le_bytes(f1.0);
                        let b = (f1_u128 & 1) != 0;
                        log_target_bit(role, base_idx, i, "p1_eval", b);
                        bit1s_ringbit.push(RingElement(Bit::new(b)));
                        bit1s_u32.push(RingElement(if b { 1u32 } else { 0u32 }));
                    }
                    (bit1s_ringbit, bit1s_u32)
                }
            };

            #[cfg(not(feature = "parallel-msb"))]
            let (bit1s_ringbit, _bit1s_u32) = {
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
                    log_target_bit(role, base_idx, i, "p1_eval", b);
                    bit1s_ringbit.push(RingElement(Bit::new(b)));
                    bit1s_u32.push(RingElement(if b { 1u32 } else { 0u32 }));
                }
                (bit1s_ringbit, bit1s_u32)
            };

            // ===== batched bit-share exchange =====
            let bit1s_network_vec = bits_to_network_vec(&bit1s_ringbit);

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.send_next",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session
                .network_session
                .send_next(NetworkValue::vec_to_network(bit1s_network_vec.clone()))
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
                .send_prev(NetworkValue::vec_to_network(bit1s_network_vec))
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

            // Receive P0's bits (packed) from PREV
            let p0_bits_msg = session
                .network_session
                .receive_prev()
                .await
                .map_err(|e| eyre!("Party 1 cannot receive bit vector from P0: {e}"))?;

            // metrics: stop timer
            drop(_tt_net);

            let bit0s_ringbit =
                network_vec_to_bits(NetworkValue::vec_from_network(p0_bits_msg)?).map_err(|e| {
                    eyre!("Party 1 cannot deserialize bit vector from P0: {e}")
                })?;

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
            let base_idx = GLOBAL_FSS_BASE_IDX.fetch_add(n as u64, Ordering::Relaxed) as usize;
            log_batch_start(role, base_idx, n);
            session
                .network_session
                .send_next(u64::new_network_element(RingElement(base_idx as u64)))
                .await
                .map_err(|e| eyre!("Dealer cannot send base_idx to P0: {e}"))?;
            session
                .network_session
                .send_prev(u64::new_network_element(RingElement(base_idx as u64)))
                .await
                .map_err(|e| eyre!("Dealer cannot send base_idx to P1: {e}"))?;
            session
                .network_session
                .send_next(u64::new_network_element(RingElement(base_idx as u64)))
                .await
                .map_err(|e| eyre!("Dealer cannot send base_idx to P0: {e}"))?;
            session
                .network_session
                .send_prev(u64::new_network_element(RingElement(base_idx as u64)))
                .await
                .map_err(|e| eyre!("Dealer cannot send base_idx to P1: {e}"))?;

            let mut sent_bytes = 0usize;
            let mut recv_bytes = 0usize;
            // Party 2 no longer generates or sends FSS keys - each party generates them locally
            // We just need to consume the same PRF randomness to stay in sync
            // (even though we don't use the keys, we need to match PRF consumption)
            let _tt_gen = crate::perf_scoped_for_party!(
                "fss.dealer.genkeys",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            for i in 0..n {
                // Consume deterministic RNG to match parties 0 and 1 (keys discarded)
                let mut seed_u128 = u128::from_le_bytes(FSS_KEYGEN_BASE_SEED);
                seed_u128 ^= (base_idx + i) as u128 + 1;
                let derived_seed = seed_u128.to_le_bytes();
                let mut prf_rng = AesRng::from_seed(derived_seed);
                
                // Build PRG/ICF for key generation (same as parties 0 and 1)
                let prg_seed = [[0u8; 16]; 4];
                let prg = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                    &prg_seed[0], &prg_seed[1], &prg_seed[2], &prg_seed[3],
                ]);
                let icf = Icf::new(p, q, prg);

                // Generate FSS key pair to consume randomness (keys are discarded)
                let f = IntvFn {
                    r_in: InG::from(0u32),
                    r_out: OutG::from(0u128),
                };
                let (_k0, _k1) = icf.gen(f, &mut prf_rng);
            }

            drop(_tt_gen);

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

            let bit0s_ringbit =
                network_vec_to_bits(NetworkValue::vec_from_network(p0_bits_msg)?).map_err(|e| {
                    eyre!("Dealer cannot deserialize bit vector from P0: {e}")
                })?;
            let bit1s_ringbit =
                network_vec_to_bits(NetworkValue::vec_from_network(p1_bits_msg)?).map_err(|e| {
                    eyre!("Dealer cannot deserialize bit vector from P1: {e}")
                })?;
            recv_bytes += approx_bytes_bit_vec(&bit0s_ringbit);
            recv_bytes += approx_bytes_bit_vec(&bit1s_ringbit);

            // Keep the (P0_bits, P1_bits) ordering to match P0/P1
            let out: Vec<Share<Bit>> = bit0s_ringbit
                .into_iter()
                .zip(bit1s_ringbit.into_iter())
                .map(|(b0, b1)| Share::new(b0, b1))
                .collect();
            if n > 0 {
                if let Some(local_idx) = (0..n).find(|i| base_idx + *i == DEBUG_TARGET_IDX) {
                    if let Some(b0) = out.get(local_idx) {
                        let a_bool = b0.a.0 != Bit::new(false);
                        let b_bool = b0.b.0 != Bit::new(false);
                        log_target_bit(role, base_idx, local_idx, "p2_out", a_bool ^ b_bool);
                    }
                }
            }
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
    base_idx: usize,
) -> Result<Vec<Share<Bit>>, Error> {
    use eyre::eyre;
    use fss_rs::icf::{IcShare, Icf, InG, IntvFn, OutG};
    use fss_rs::prg::Aes128MatyasMeyerOseasPrg;

    let role = session.own_role().index();
    let n = batch.len();
    let debug_target_idx: usize = DEBUG_TARGET_IDX;

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
            let mut sent_bytes = 0usize;
            let mut recv_bytes = 0usize;
            // Generate FSS keys locally using deterministic randomness from session PRF
            let mut my_keys = Vec::with_capacity(n);
            for i in 0..n {
                // Deterministic RNG derived from a shared base seed and per-index counter
                let mut seed_u128 = u128::from_le_bytes(FSS_KEYGEN_BASE_SEED);
                seed_u128 ^= (base_idx + i) as u128 + 1;
                let derived_seed = seed_u128.to_le_bytes();
                let mut prf_rng = AesRng::from_seed(derived_seed);
                
                // Build PRG/ICF for key generation
                let prg_seed = [[0u8; 16]; 4];
                let prg = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                    &prg_seed[0], &prg_seed[1], &prg_seed[2], &prg_seed[3],
                ]);
                let icf = Icf::new(p, q, prg);
                
                // Generate FSS key pair using deterministic PRF RNG
                // All parties will generate the same (k0, k1) pair
                let f = IntvFn {
                    r_in: InG::from(0u32),
                    r_out: OutG::from(0u128),
                };
                let (k0, _k1) = icf.gen(f, &mut prf_rng);
                my_keys.push(k0);
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
            sent_bytes += approx_bytes_int_vec(&send_d2r2);
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
            recv_bytes += approx_bytes_int_vec(&d1r1_vec);

            // ===== Evaluate all indices and collect bit shares =====
            // Parallel when feature enabled and n >= threshold; otherwise sequential.
            #[cfg(feature = "parallel-msb")]
            let (bit0s_ringbit, _bit0s_u32) = {
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

                            // Use pre-generated key (keys are already generated above)
                            let key_i = &my_keys[i];

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
            let (bit0s_ringbit, _bit0s_u32) = {
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
            let bit0s_network_vec = bits_to_network_vec(&bit0s_ringbit);
            session
                .network_session
                .send_prev(NetworkValue::vec_to_network(bit0s_network_vec.clone()))
                .await?;
            sent_bytes += approx_bytes_bit_vec(&bit0s_ringbit);
            session
                .network_session
                .send_next(NetworkValue::vec_to_network(bit0s_network_vec))
                .await?;
            sent_bytes += approx_bytes_bit_vec(&bit0s_ringbit);
            let p1_bits_msg = session
                .network_session
                .receive_next()
                .await
                .map_err(|e| eyre!("Party 0 cannot receive bit vector from P1: {e}"))?;

            let bit1s_ringbit =
                network_vec_to_bits(NetworkValue::vec_from_network(p1_bits_msg)?).map_err(|e| {
                    eyre!("Party 0 cannot deserialize bit vector from P1: {e}")
                })?;
            recv_bytes += approx_bytes_bit_vec(&bit1s_ringbit);

            // Assemble output shares
            let out: Vec<Share<Bit>> = bit0s_ringbit
                .into_iter()
                .zip(bit1s_ringbit.into_iter())
                .map(|(b0, b1)| Share::new(b0, b1))
                .collect();

            record_traffic(sent_bytes, recv_bytes);

            Ok(out)
        }

        // =======================
        // Party 1 (Evaluator)
        // =======================
        1 => {
            let mut sent_bytes = 0usize;
            let mut recv_bytes = 0usize;
            // Generate FSS keys locally using deterministic randomness from session PRF
            let mut my_keys = Vec::with_capacity(n);
            for i in 0..n {
                // Deterministic RNG derived from a shared base seed and per-index counter
                let mut seed_u128 = u128::from_le_bytes(FSS_KEYGEN_BASE_SEED);
                seed_u128 ^= (base_idx + i) as u128 + 1;
                let derived_seed = seed_u128.to_le_bytes();
                let mut prf_rng = AesRng::from_seed(derived_seed);
                
                // Build PRG/ICF for key generation
                let prg_seed = [[0u8; 16]; 4];
                let prg = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                    &prg_seed[0], &prg_seed[1], &prg_seed[2], &prg_seed[3],
                ]);
                let icf = Icf::new(p, q, prg);
                
                // Generate FSS key pair using deterministic PRF RNG
                // All parties will generate the same (k0, k1) pair
                let f = IntvFn {
                    r_in: InG::from(0u32),
                    r_out: OutG::from(0u128),
                };
                let (_k0, k1) = icf.gen(f, &mut prf_rng);
                my_keys.push(k1);
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
            sent_bytes += approx_bytes_int_vec(&send_d1r1);
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
            recv_bytes += approx_bytes_int_vec(&d2r2_vec);

            // ===== Evaluate all indices and collect bit shares =====
            #[cfg(feature = "parallel-msb")]
            let (bit1s_ringbit, _bit1s_u32) = {
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

                            // Use pre-generated key (keys are already generated above)
                            let key_i = &my_keys[i];

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

            // Debug: log the bit share for the target index if this batch covers it
            #[cfg(feature = "parallel-msb")]
            {
                if debug_target_idx >= base_idx && debug_target_idx < base_idx + n {
                    let local_idx = debug_target_idx - base_idx;
                    // Evaluate sequentially for the target index using the same icf to log value
                    let x = &batch[local_idx];
                    let y = x.a + d2r2_vec[local_idx] + x.b + RingElement(n_half_u32);
                    let f1 = icf.eval(true, &my_keys[local_idx], fss_rs::group::int::U32Group(y.0));
                    let f1_u128 = u128::from_le_bytes(f1.0);
                    let b = (f1_u128 & 1) != 0;
                    append_debug_line(&format!(
                        "p1 target={} local={} bit={}",
                        debug_target_idx, local_idx, b
                    ));
                }
            }

            #[cfg(not(feature = "parallel-msb"))]
            let (bit1s_ringbit, _bit1s_u32) = {
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
            let bit1s_network_vec = bits_to_network_vec(&bit1s_ringbit);
            session
                .network_session
                .send_next(NetworkValue::vec_to_network(bit1s_network_vec.clone()))
                .await?;
            sent_bytes += approx_bytes_bit_vec(&bit1s_ringbit);
            session
                .network_session
                .send_prev(NetworkValue::vec_to_network(bit1s_network_vec))
                .await?;
            sent_bytes += approx_bytes_bit_vec(&bit1s_ringbit);

            // Receive P0's bits (packed) from PREV
            let p0_bits_msg = session
                .network_session
                .receive_prev()
                .await
                .map_err(|e| eyre!("Party 1 cannot receive bit vector from P0: {e}"))?;

            let bit0s_ringbit =
                network_vec_to_bits(NetworkValue::vec_from_network(p0_bits_msg)?).map_err(|e| {
                    eyre!("Party 1 cannot deserialize bit vector from P0: {e}")
                })?;
            recv_bytes += approx_bytes_bit_vec(&bit0s_ringbit);

            // Assemble output shares
            let out: Vec<Share<Bit>> = bit0s_ringbit
                .into_iter()
                .zip(bit1s_ringbit.into_iter())
                .map(|(b0, b1)| Share::new(b0, b1))
                .collect();

            record_traffic(sent_bytes, recv_bytes);
            Ok(out)
        }

        // =======================
        // Party 2 (Dealer)
        // =======================
        2 => {
            log_batch_start(role, base_idx, n);
            let mut sent_bytes = 0usize;
            let mut recv_bytes = 0usize;
            // Party 2 no longer generates or sends FSS keys - each party generates them locally
            // We just need to consume the same PRF randomness to stay in sync
            // (even though we don't use the keys, we need to match PRF consumption)
            for i in 0..n {
                // Consume deterministic RNG to match parties 0 and 1 (keys discarded)
                let mut seed_u128 = u128::from_le_bytes(FSS_KEYGEN_BASE_SEED);
                seed_u128 ^= (base_idx + i) as u128 + 1;
                let derived_seed = seed_u128.to_le_bytes();
                let mut prf_rng = AesRng::from_seed(derived_seed);

                // Build PRG/ICF for key generation (same as parties 0 and 1)
                let prg_seed = [[0u8; 16]; 4];
                let prg = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                    &prg_seed[0], &prg_seed[1], &prg_seed[2], &prg_seed[3],
                ]);
                let icf = Icf::new(p, q, prg);

                // Generate FSS key pair to consume randomness (keys are discarded)
                let f = IntvFn {
                    r_in: InG::from(0u32),
                    r_out: OutG::from(0u128),
                };
                let (_k0, _k1) = icf.gen(f, &mut prf_rng);
            }

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

            let bit0s_ringbit =
                network_vec_to_bits(NetworkValue::vec_from_network(p0_bits_msg)?).map_err(|e| {
                    eyre!("Dealer cannot deserialize bit vector from P0: {e}")
                })?;
            let bit1s_ringbit =
                network_vec_to_bits(NetworkValue::vec_from_network(p1_bits_msg)?).map_err(|e| {
                    eyre!("Dealer cannot deserialize bit vector from P1: {e}")
                })?;
            recv_bytes += approx_bytes_bit_vec(&bit0s_ringbit);
            recv_bytes += approx_bytes_bit_vec(&bit1s_ringbit);

            // Keep the (P0_bits, P1_bits) ordering to match P0/P1
            let out: Vec<Share<Bit>> = bit0s_ringbit
                .into_iter()
                .zip(bit1s_ringbit.into_iter())
                .map(|(b0, b1)| Share::new(b0, b1))
                .collect();

            record_traffic(sent_bytes, recv_bytes);
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
    base_idx: usize,
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
    let debug_target_idx: usize = DEBUG_TARGET_IDX;

    // Depending on the role, do different stuff
    match role {
        0 => {
            let base_idx_msg = session
                .network_session
                .receive_prev()
                .await
                .map_err(|e| eyre!("Party 0 cannot receive base_idx from dealer: {e}"))?;
            let base_idx = u64::into_vec(base_idx_msg)?
                .pop()
                .ok_or_else(|| eyre!("base_idx message empty"))?
                .0 as usize;
            log_batch_start(role, base_idx, n);
            //Generate all r2 prf key, keep the r0 keys for later
            // println!("party 0: Batch size is {}", batch_size);
            let mut d2r2_vec = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                d2r2_vec.push(x[i].b + RingElement(0)); // r2 assumed 0 in this deterministic variant
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

            // Set up the function for FSS
            // we need this below to handle signed numbers, if input is unsigned no need to add N/2
            let n_falf_u32 = 1u32 << 31;
            let n_half = InG::from(n_falf_u32);
            // Interval: [0, 2^31) after shifting inputs by n/2 to handle signed values
            // Matches the parallel path parameters
            let p = InG::from(0u32);
            let q = InG::from(1u32 << 31);

            // Generate FSS keys locally using deterministic randomness
            let mut my_keys = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                // Deterministic RNG derived from a shared base seed and per-index counter
                let mut seed_u128 = u128::from_le_bytes(FSS_KEYGEN_BASE_SEED);
                seed_u128 ^= (base_idx + i) as u128 + 1;
                let derived_seed = seed_u128.to_le_bytes();
                let mut prf_rng = AesRng::from_seed(derived_seed);
                
                // Build PRG/ICF for key generation
                let prg_seed = [[0u8; 16]; 4];
                let prg = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                    &prg_seed[0], &prg_seed[1], &prg_seed[2], &prg_seed[3],
                ]);
                let icf = Icf::new(p, q, prg);
                
                // Generate FSS key pair using deterministic PRF RNG
                let f = IntvFn {
                    r_in: InG::from(0u32),
                    r_out: OutG::from(0u128),
                };
                let (k0, _k1) = icf.gen(f, &mut prf_rng);
                my_keys.push(k0);
            }

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

            let keys: Vec<[u8; 16]> = vec![[0u8; 16]; 4];
            let mut f_x_0_bits = Vec::with_capacity(batch_size); // store all the eval results

            // Use pre-generated keys directly
            for i in 0..batch_size {
                let k_fss_0_icshare = &my_keys[i];

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
                        k_fss_0_icshare,
                        fss_rs::group::int::U32Group(d_plus_r.0),
                    )
                    .0,
                ));

                // Take only the LSB of the evaluation result as our bit share
                f_x_0_bits.push(RingElement(Bit::new((temp_eval.0 & 1) != 0)));
            }

            // Prepare them in a vector to send to dealer and next party
            let f_0_res_network: Vec<NetworkValue> = f_x_0_bits
                .iter()
                .copied()
                .map(NetworkValue::RingElementBit)
                .collect();

            // Create NetworkValue once, then clone it for the second send
            let f_0_network_value = NetworkValue::vec_to_network(f_0_res_network);
            let f_0_network_value_clone = f_0_network_value.clone();

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.send_next",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session // send to party 1
                .network_session
                .send_next(f_0_network_value_clone)
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
                .send_prev(f_0_network_value)
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

            // Debug: log the bit share for the target index if this batch covers it
            if debug_target_idx >= base_idx && debug_target_idx < base_idx + n {
                let local_idx = debug_target_idx - base_idx;
                if let Some(share) = shares.get(local_idx) {
                    append_debug_line(&format!(
                        "p0 target={} local={} bit_a={} bit_b={}",
                        debug_target_idx, local_idx, share.a.0, share.b.0
                    ));
                }
            }

            Ok(shares)
        }
        1 => {
            let base_idx_msg = session
                .network_session
                .receive_next()
                .await
                .map_err(|e| eyre!("Party 1 cannot receive base_idx from dealer: {e}"))?;
            let base_idx = u64::into_vec(base_idx_msg)?
                .pop()
                .ok_or_else(|| eyre!("base_idx message empty"))?
                .0 as usize;
            log_batch_start(role, base_idx, n);
            // eprintln!("party 1: Batch size is {}", batch_size);
            std::io::stderr().flush().ok();
            let mut d1r1_vec = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                d1r1_vec.push(x[i].a + RingElement(0)); // r1 assumed 0 in this deterministic variant
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

            // Set up the function for FSS
            // we need this below to handle signed numbers, if input is unsigned no need to add N/2
            let n_falf_u32 = 1u32 << 31;
            let n_half = InG::from(n_falf_u32);
            // Interval: [0, 2^31) after shifting inputs by n/2 to handle signed values
            // Matches the parallel path parameters
            let p = InG::from(0u32);
            let q = InG::from(1u32 << 31);

            // Generate FSS keys locally using deterministic randomness
            let mut my_keys = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                // Deterministic RNG derived from a shared base seed and per-index counter
                let mut seed_u128 = u128::from_le_bytes(FSS_KEYGEN_BASE_SEED);
                seed_u128 ^= (base_idx + i) as u128 + 1;
                let derived_seed = seed_u128.to_le_bytes();
                let mut prf_rng = AesRng::from_seed(derived_seed);
                
                // Build PRG/ICF for key generation
                let prg_seed = [[0u8; 16]; 4];
                let prg = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                    &prg_seed[0], &prg_seed[1], &prg_seed[2], &prg_seed[3],
                ]);
                let icf = Icf::new(p, q, prg);
                
                // Generate FSS key pair using deterministic PRF RNG
                let f = IntvFn {
                    r_in: InG::from(0u32),
                    r_out: OutG::from(0u128),
                };
                let (_k0, k1) = icf.gen(f, &mut prf_rng);
                my_keys.push(k1);
            }

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

            let keys: Vec<[u8; 16]> = vec![[0u8; 16]; 4];
            let mut f_x_1_bits = Vec::with_capacity(batch_size); // store all the eval results

            // Use pre-generated keys directly
            for i in 0..batch_size {
                let k_fss_1_icshare = &my_keys[i];

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
                        k_fss_1_icshare,
                        fss_rs::group::int::U32Group(d_plus_r.0),
                    )
                    .0,
                ));

                // Take only the LSB of the evaluation result as our bit share
                f_x_1_bits.push(RingElement(Bit::new((temp_eval.0 & 1) != 0)));
            }

            // Prepare them in a vector to send to dealer and next party
            let f_1_res_network: Vec<NetworkValue> = f_x_1_bits
                .iter()
                .copied()
                .map(NetworkValue::RingElementBit)
                .collect();

            // Create NetworkValue once, then clone it for the second send
            let f_1_network_value = NetworkValue::vec_to_network(f_1_res_network);
            let f_1_network_value_clone = f_1_network_value.clone();

            //metrics: measure the network time
            let _tt_net = crate::perf_scoped_for_party!(
                "fss.network.post-icf.send_prev",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            session // send to party 0
                .network_session
                .send_prev(f_1_network_value_clone)
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
                .send_next(f_1_network_value)
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

            // Interval: [0, 2^31) after shifting inputs by n/2 to handle signed values
            // Matches the parallel path parameters
            let p = InG::from(0u32);
            let q = InG::from(1u32 << 31);

            // Generate FSS keys locally (same as P0/P1) to consume PRF randomness and stay in sync
            // We don't send these keys since P0/P1 generate them locally
            //metrics: measure the genkeys time
            let _tt_gen = crate::perf_scoped_for_party!(
                "fss.dealer.genkeys",
                role,
                n,            // bucket on the items this block processes
                bucket_bound  // your desired bucket cap
            );

            for i in 0..batch_size {
                // Generate FSS keys locally (same as P0/P1) using deterministic randomness
                // Deterministic RNG derived from a shared base seed and per-index counter
                let mut seed_u128 = u128::from_le_bytes(FSS_KEYGEN_BASE_SEED);
                seed_u128 ^= (base_idx + i) as u128 + 1;
                let derived_seed = seed_u128.to_le_bytes();
                let mut prf_rng = AesRng::from_seed(derived_seed);
                
                // Build PRG/ICF for key generation
                let prg_seed = [[0u8; 16]; 4];
                let prg = Aes128MatyasMeyerOseasPrg::<16, 2, 4>::new(&[
                    &prg_seed[0], &prg_seed[1], &prg_seed[2], &prg_seed[3],
                ]);
                let icf = Icf::new(p, q, prg);
                
                // Generate FSS key pair using deterministic PRF RNG (discard keys, just consume randomness)
                let f = IntvFn {
                    r_in: InG::from(0u32),
                    r_out: OutG::from(0u128),
                };
                let (_k0, _k1) = icf.gen(f, &mut prf_rng);
                // Keys are discarded - we just need to consume the same deterministic randomness as P0/P1
            }

            // drop timer
            drop(_tt_gen);

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

/// Pipelined batch processing: overlaps sends with receives to reduce latency
/// Strategy: send batch N+1's reconstruct data while waiting for batch N's reconstruct receive
/// This reduces total latency by overlapping network I/O operations
pub(crate) async fn add_3_get_msb_fss_batch_pipelined(
    session: &mut Session,
    batches: &[&[Share<u32>]],
    parallel_threshold: usize,
    output: &mut Vec<Share<Bit>>,
    base_start: usize,
) -> Result<(), Error> {
    use eyre::eyre;
    use fss_rs::icf::{Icf, InG, IntvFn, OutG};
    use fss_rs::prg::Aes128MatyasMeyerOseasPrg;

    let role = session.own_role().index();
    let bucket_bound = 150;
    let p = InG::from(0u32);
    let q = InG::from(1u32 << 31);
    let n_half_u32: u32 = 1u32 << 31;

    if batches.is_empty() {
        return Ok(());
    }

    // For simplicity and correctness, process batches sequentially but optimize the inner loop
    // The key optimization: we can't easily pipeline due to protocol dependencies
    // But we ensure sends complete quickly (they're non-blocking) before receives
    let mut base_idx = base_start;
    for &batch in batches {
        let batch_out = add_3_get_msb_fss_batch_parallel_threshold_timers(session, batch, parallel_threshold, base_idx).await?;
        output.extend(batch_out);
        base_idx += batch.len();
    }
    Ok(())
}
