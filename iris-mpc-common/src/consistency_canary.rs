//! Private full-database consistency check for rerandomized iris shares.
//!
//! Each party compresses its complete two-eye database to a small additive
//! syndrome share. The syndrome is never opened. Instead, a replicated Boolean
//! MPC circuit adds the three shares modulo `2^16`, OR-reduces all result bits,
//! and opens only the final one-bit "nonzero" result.

use std::io::Read;

use ampc_actor_utils::{execution::session::NetworkSession, network::mpc::NetworkValue};
use eyre::{bail, ensure, Result};
use rand::RngCore;

use crate::galois::degree4::{basis::Monomial, GaloisRingElement};
use crate::{IRIS_CODE_LENGTH, MASK_CODE_LENGTH};

type Ring = GaloisRingElement<Monomial>;

/// Ring positions in one two-eye row, plus a public constant sentinel which
/// makes a missing all-zero row detectable.
pub const CANARY_LANES: usize = (2 * IRIS_CODE_LENGTH + 2 * MASK_CODE_LENGTH) / 4 + 1;

/// Under the BLAKE3-XOF-as-PRF assumption, a worst-case nonzero error is
/// annihilated with probability at most `2^-4` per repetition in GR(2^16, 4),
/// so 12 independent repetitions bound the miss probability by `2^-48`.
pub const DEFAULT_CANARY_REPETITIONS: usize = 12;

pub const STARTUP_CANARY_CONTEXT: [u8; 32] = [0; 32];

const CHALLENGE_DOMAIN: &str = "iris-mpc/consistency-canary/challenge/v2";
const WEIGHT_DOMAIN: &str = "iris-mpc/consistency-canary/row-lane-weights/v2";
const BOOLEAN_SESSION_DOMAIN: &str = "iris-mpc/consistency-canary/boolean-session/v1";

const FRAME_MAGIC: &[u8; 8] = b"ICANMPC1";
const SESSION_TAG_LEN: usize = 16;
const FRAME_HEADER_LEN: usize = FRAME_MAGIC.len() + SESSION_TAG_LEN + 1 + 4 + 4;
const PHASE_ARITHMETIC_MASK: u8 = 1;
const PHASE_MASKED_SHARE: u8 = 2;
const PHASE_AND_RANDOMNESS: u8 = 3;
const PHASE_AND_SHARE: u8 = 4;
const PHASE_FINAL_OPEN: u8 = 5;

pub fn fresh_challenge_contribution() -> [u8; 32] {
    let mut contribution = [0; 32];
    rand::rngs::OsRng.fill_bytes(&mut contribution);
    contribution
}

pub fn derive_challenge(
    context: [u8; 32],
    contributions: impl IntoIterator<Item = [u8; 32]>,
) -> [u8; 32] {
    let combined = contributions.into_iter().fold([0u8; 32], |mut sum, part| {
        for (destination, source) in sum.iter_mut().zip(part) {
            *destination ^= source;
        }
        sum
    });
    let mut hasher = blake3::Hasher::new_derive_key(CHALLENGE_DOMAIN);
    hasher.update(&context);
    hasher.update(&combined);
    *hasher.finalize().as_bytes()
}

pub fn syndrome_const(party_id: usize) -> Ring {
    let x = |party: usize| Ring::EXCEPTIONAL_SEQUENCE[(party % 3) + 1];
    x(party_id + 1) - x(party_id + 2)
}

fn next_weight(reader: &mut blake3::OutputReader) -> Ring {
    let mut bytes = [0; 8];
    reader
        .read_exact(&mut bytes)
        .expect("BLAKE3 XOF is unbounded");
    Ring::from_coefs([
        u16::from_le_bytes([bytes[0], bytes[1]]),
        u16::from_le_bytes([bytes[2], bytes[3]]),
        u16::from_le_bytes([bytes[4], bytes[5]]),
        u16::from_le_bytes([bytes[6], bytes[7]]),
    ])
}

/// Streaming random linear compression to one Galois-ring element per
/// repetition. One row XOF supplies independent outputs in canonical
/// `(lane, repetition)` order, thereby binding every weight to both indices.
pub struct CanaryAccumulator {
    party_id: usize,
    repetitions: usize,
    challenge: [u8; 32],
    values: Vec<Ring>,
    rows: u64,
}

impl CanaryAccumulator {
    pub fn new(party_id: usize, repetitions: usize, challenge: [u8; 32]) -> Self {
        assert!(party_id < 3);
        assert!(repetitions > 0);
        Self {
            party_id,
            repetitions,
            challenge,
            values: vec![Ring::ZERO; repetitions],
            rows: 0,
        }
    }

    pub fn accumulate(
        &mut self,
        row_id: u32,
        left_code: &[u16],
        left_mask: &[u16],
        right_code: &[u16],
        right_mask: &[u16],
    ) -> Result<()> {
        ensure!(row_id > 0, "canary row IDs must be positive");
        ensure!(
            left_code.len() == IRIS_CODE_LENGTH
                && right_code.len() == IRIS_CODE_LENGTH
                && left_mask.len() == MASK_CODE_LENGTH
                && right_mask.len() == MASK_CODE_LENGTH,
            "canary row {row_id} has invalid share lengths"
        );

        let mut hasher = blake3::Hasher::new_derive_key(WEIGHT_DOMAIN);
        hasher.update(&self.challenge);
        hasher.update(&row_id.to_le_bytes());
        hasher.update(&(self.repetitions as u64).to_le_bytes());
        hasher.update(&(CANARY_LANES as u64).to_le_bytes());
        let mut weights = hasher.finalize_xof();
        let mut lane = 0usize;
        for coefficients in [left_code, left_mask, right_code, right_mask] {
            for chunk in coefficients.chunks_exact(4) {
                let value = Ring::from_coefs(chunk.try_into().expect("four coefficients"));
                for accumulator in &mut self.values {
                    *accumulator = *accumulator + next_weight(&mut weights) * value;
                }
                lane += 1;
            }
        }
        debug_assert_eq!(lane, CANARY_LANES - 1);
        let sentinel = Ring::from_coefs([1, 0, 0, 0]);
        for accumulator in &mut self.values {
            *accumulator = *accumulator + next_weight(&mut weights) * sentinel;
        }
        self.rows += 1;
        Ok(())
    }

    pub fn merge(&mut self, other: Self) -> Result<()> {
        ensure!(
            self.party_id == other.party_id
                && self.repetitions == other.repetitions
                && self.challenge == other.challenge,
            "cannot merge canary accumulators from different checks"
        );
        ensure!(self.values.len() == other.values.len());
        for (left, right) in self.values.iter_mut().zip(other.values) {
            *left = *left + right;
        }
        self.rows = self
            .rows
            .checked_add(other.rows)
            .ok_or_else(|| eyre::eyre!("canary row count overflow"))?;
        Ok(())
    }

    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// Four `u16` coefficients per repetition.
    pub fn into_syndrome_share(self) -> Vec<u16> {
        let coefficient = syndrome_const(self.party_id);
        self.values
            .into_iter()
            .flat_map(|value| (coefficient * value).coefs)
            .collect()
    }
}

/// Backend-specific private point-to-point transport. `exchange_next` sends to
/// the next party and receives from the previous; `exchange_previous` does the
/// reverse. Implementations must preserve message boundaries.
#[allow(async_fn_in_trait)]
pub trait BooleanMpcTransport {
    fn party_id(&self) -> usize;
    async fn exchange_next(&mut self, message: Vec<u8>) -> Result<Vec<u8>>;
    async fn exchange_previous(&mut self, message: Vec<u8>) -> Result<Vec<u8>>;
}

/// Adapter for the private point-to-point sessions used by the CPU service and
/// standalone maintenance protocols. The underlying network preserves message
/// boundaries and is configured with mutual TLS in production.
pub struct NetworkSessionBooleanTransport<'a> {
    party_id: usize,
    session: &'a mut NetworkSession,
}

impl<'a> NetworkSessionBooleanTransport<'a> {
    pub fn new(party_id: usize, session: &'a mut NetworkSession) -> Self {
        Self { party_id, session }
    }
}

impl BooleanMpcTransport for NetworkSessionBooleanTransport<'_> {
    fn party_id(&self) -> usize {
        self.party_id
    }

    async fn exchange_next(&mut self, message: Vec<u8>) -> Result<Vec<u8>> {
        self.session.send_next(NetworkValue::Bytes(message)).await?;
        match self.session.receive_prev().await? {
            NetworkValue::Bytes(message) => Ok(message),
            _ => bail!("canary Boolean MPC received an unexpected message"),
        }
    }

    async fn exchange_previous(&mut self, message: Vec<u8>) -> Result<Vec<u8>> {
        self.session.send_prev(NetworkValue::Bytes(message)).await?;
        match self.session.receive_next().await? {
            NetworkValue::Bytes(message) => Ok(message),
            _ => bail!("canary Boolean MPC received an unexpected message"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ReplicatedBit {
    local: u8,
    previous: u8,
}

impl ReplicatedBit {
    fn xor(self, other: Self) -> Self {
        Self {
            local: self.local ^ other.local,
            previous: self.previous ^ other.previous,
        }
    }
}

type ReplicatedWord = [ReplicatedBit; 16];

fn session_tag(challenge: [u8; 32], repetitions: usize, words: usize) -> [u8; SESSION_TAG_LEN] {
    let mut hasher = blake3::Hasher::new_derive_key(BOOLEAN_SESSION_DOMAIN);
    hasher.update(&challenge);
    hasher.update(&(repetitions as u64).to_le_bytes());
    hasher.update(&(words as u64).to_le_bytes());
    hasher.finalize().as_bytes()[..SESSION_TAG_LEN]
        .try_into()
        .expect("fixed length")
}

fn encode_frame(tag: [u8; SESSION_TAG_LEN], phase: u8, step: u32, payload: &[u8]) -> Vec<u8> {
    let payload_len = u32::try_from(payload.len()).expect("canary MPC frame is too large");
    let mut frame = Vec::with_capacity(FRAME_HEADER_LEN + payload.len());
    frame.extend_from_slice(FRAME_MAGIC);
    frame.extend_from_slice(&tag);
    frame.push(phase);
    frame.extend_from_slice(&step.to_le_bytes());
    frame.extend_from_slice(&payload_len.to_le_bytes());
    frame.extend_from_slice(payload);
    frame
}

fn decode_frame(
    frame: Vec<u8>,
    tag: [u8; SESSION_TAG_LEN],
    phase: u8,
    step: u32,
    expected_payload_len: usize,
) -> Result<Vec<u8>> {
    ensure!(
        frame.len() >= FRAME_HEADER_LEN,
        "canary MPC frame is truncated"
    );
    ensure!(
        &frame[..FRAME_MAGIC.len()] == FRAME_MAGIC,
        "canary MPC frame domain mismatch"
    );
    let tag_start = FRAME_MAGIC.len();
    ensure!(
        frame[tag_start..tag_start + SESSION_TAG_LEN] == tag,
        "canary MPC session mismatch"
    );
    let phase_offset = tag_start + SESSION_TAG_LEN;
    ensure!(frame[phase_offset] == phase, "canary MPC phase mismatch");
    ensure!(
        u32::from_le_bytes(frame[phase_offset + 1..phase_offset + 5].try_into()?) == step,
        "canary MPC step mismatch"
    );
    let declared =
        u32::from_le_bytes(frame[phase_offset + 5..FRAME_HEADER_LEN].try_into()?) as usize;
    ensure!(
        declared == expected_payload_len && frame.len() == FRAME_HEADER_LEN + declared,
        "canary MPC payload length mismatch"
    );
    Ok(frame[FRAME_HEADER_LEN..].to_vec())
}

fn encode_u16(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn decode_u16(bytes: &[u8], expected: usize) -> Result<Vec<u16>> {
    ensure!(
        bytes.len() == expected * 2,
        "canary MPC word payload length mismatch"
    );
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

struct BooleanCircuit<'a, T> {
    transport: &'a mut T,
    tag: [u8; SESSION_TAG_LEN],
    local_randomness: Vec<u8>,
    previous_randomness: Vec<u8>,
    random_cursor: usize,
    and_step: u32,
}

impl<'a, T: BooleanMpcTransport> BooleanCircuit<'a, T> {
    async fn and_batch(
        &mut self,
        gates: &[(ReplicatedBit, ReplicatedBit)],
    ) -> Result<Vec<ReplicatedBit>> {
        ensure!(
            self.random_cursor + gates.len() <= self.local_randomness.len(),
            "canary MPC exhausted AND randomness"
        );
        let mut local = Vec::with_capacity(gates.len());
        for (offset, (left, right)) in gates.iter().enumerate() {
            let index = self.random_cursor + offset;
            local.push(
                (left.local & right.local)
                    ^ (left.local & right.previous)
                    ^ (left.previous & right.local)
                    ^ self.local_randomness[index]
                    ^ self.previous_randomness[index],
            );
        }
        self.random_cursor += gates.len();
        let frame = encode_frame(self.tag, PHASE_AND_SHARE, self.and_step, &local);
        let previous = self.transport.exchange_next(frame).await?;
        let previous = decode_frame(
            previous,
            self.tag,
            PHASE_AND_SHARE,
            self.and_step,
            gates.len(),
        )?;
        ensure!(
            previous.iter().all(|bit| *bit <= 1),
            "canary MPC received a non-bit AND share"
        );
        self.and_step = self
            .and_step
            .checked_add(1)
            .ok_or_else(|| eyre::eyre!("canary MPC step overflow"))?;
        Ok(local
            .into_iter()
            .zip(previous)
            .map(|(local, previous)| ReplicatedBit { local, previous })
            .collect())
    }

    async fn add_words(
        &mut self,
        left: &[ReplicatedWord],
        right: &[ReplicatedWord],
    ) -> Result<Vec<ReplicatedWord>> {
        ensure!(left.len() == right.len(), "canary MPC add length mismatch");
        let mut output = vec![[ReplicatedBit::default(); 16]; left.len()];
        let mut carry = vec![ReplicatedBit::default(); left.len()];
        for bit in 0..16 {
            let propagate = left
                .iter()
                .zip(right)
                .map(|(left, right)| left[bit].xor(right[bit]))
                .collect::<Vec<_>>();
            for word in 0..left.len() {
                output[word][bit] = propagate[word].xor(carry[word]);
            }
            let mut gates = Vec::with_capacity(left.len() * 2);
            for word in 0..left.len() {
                gates.push((left[word][bit], right[word][bit]));
                gates.push((carry[word], propagate[word]));
            }
            let products = self.and_batch(&gates).await?;
            for word in 0..left.len() {
                carry[word] = products[2 * word].xor(products[2 * word + 1]);
            }
        }
        Ok(output)
    }

    async fn or_reduce(&mut self, mut bits: Vec<ReplicatedBit>) -> Result<ReplicatedBit> {
        ensure!(!bits.is_empty(), "canary MPC cannot OR an empty vector");
        while bits.len() > 1 {
            let pairs = bits.len() / 2;
            let gates = (0..pairs)
                .map(|pair| (bits[2 * pair], bits[2 * pair + 1]))
                .collect::<Vec<_>>();
            let products = self.and_batch(&gates).await?;
            let mut next = Vec::with_capacity(bits.len().div_ceil(2));
            for pair in 0..pairs {
                next.push(bits[2 * pair].xor(bits[2 * pair + 1]).xor(products[pair]));
            }
            if bits.len() % 2 == 1 {
                next.push(*bits.last().expect("nonempty"));
            }
            bits = next;
        }
        Ok(bits[0])
    }
}

fn source_words(
    source_party: usize,
    party_id: usize,
    local_words: &[u16],
    previous_words: &[u16],
) -> Vec<ReplicatedWord> {
    let previous_party = (party_id + 2) % 3;
    local_words
        .iter()
        .zip(previous_words)
        .map(|(local, previous)| {
            std::array::from_fn(|bit| ReplicatedBit {
                local: u8::from(source_party == party_id) * ((local >> bit) as u8 & 1),
                previous: u8::from(source_party == previous_party) * ((previous >> bit) as u8 & 1),
            })
        })
        .collect()
}

fn circuit_dimensions(repetitions: usize) -> Result<(usize, usize)> {
    ensure!(repetitions > 0, "canary MPC needs at least one repetition");
    let words = repetitions
        .checked_mul(4)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| eyre::eyre!("canary MPC word count overflow"))?;
    let word_bits = words
        .checked_mul(16)
        .ok_or_else(|| eyre::eyre!("canary MPC bit count overflow"))?;
    let and_gates = words
        .checked_mul(64)
        .and_then(|value| value.checked_add(word_bits - 1))
        .ok_or_else(|| eyre::eyre!("canary MPC gate count overflow"))?;
    Ok((words, and_gates))
}

/// Largest private frame used by the Boolean circuit. GPU allocates both
/// point-to-point buffers before publishing readiness.
pub fn boolean_mpc_max_frame_len(repetitions: usize) -> Result<usize> {
    let (_, and_gates) = circuit_dimensions(repetitions)?;
    FRAME_HEADER_LEN
        .checked_add(and_gates)
        .ok_or_else(|| eyre::eyre!("canary MPC frame length overflow"))
}

/// Privately test whether any compressed syndrome coefficient is nonzero.
/// `syndrome_share` contains four `u16` coefficients per repetition followed
/// by one local scan-failure word. The only reconstructed value is the returned
/// Boolean.
pub async fn private_any_nonzero<T: BooleanMpcTransport>(
    syndrome_share: Vec<u16>,
    repetitions: usize,
    challenge: [u8; 32],
    transport: &mut T,
) -> Result<bool> {
    let (words, total_and_gates) = circuit_dimensions(repetitions)?;
    ensure!(
        syndrome_share.len() == words,
        "canary MPC syndrome length mismatch"
    );
    let party_id = transport.party_id();
    ensure!(party_id < 3, "canary MPC party ID must be in 0..3");
    let tag = session_tag(challenge, repetitions, words);

    let mut mask_bytes = vec![0; words * 2];
    rand::rngs::OsRng.fill_bytes(&mut mask_bytes);
    let local_masks = decode_u16(&mask_bytes, words)?;
    let previous_masks = transport
        .exchange_next(encode_frame(tag, PHASE_ARITHMETIC_MASK, 0, &mask_bytes))
        .await?;
    let previous_masks = decode_frame(previous_masks, tag, PHASE_ARITHMETIC_MASK, 0, words * 2)?;
    let previous_masks = decode_u16(&previous_masks, words)?;
    let masked = syndrome_share
        .into_iter()
        .zip(local_masks)
        .zip(previous_masks)
        .map(|((share, local), previous)| share.wrapping_add(local).wrapping_sub(previous))
        .collect::<Vec<_>>();

    let masked_payload = encode_u16(&masked);
    let previous_masked = transport
        .exchange_next(encode_frame(tag, PHASE_MASKED_SHARE, 0, &masked_payload))
        .await?;
    let previous_masked = decode_frame(previous_masked, tag, PHASE_MASKED_SHARE, 0, words * 2)?;
    let previous_masked = decode_u16(&previous_masked, words)?;

    let source = |party| source_words(party, party_id, &masked, &previous_masked);
    let source0 = source(0);
    let source1 = source(1);
    let source2 = source(2);

    let mut local_randomness = vec![0; total_and_gates];
    rand::rngs::OsRng.fill_bytes(&mut local_randomness);
    for bit in &mut local_randomness {
        *bit &= 1;
    }
    let previous_randomness = transport
        .exchange_next(encode_frame(
            tag,
            PHASE_AND_RANDOMNESS,
            0,
            &local_randomness,
        ))
        .await?;
    let previous_randomness = decode_frame(
        previous_randomness,
        tag,
        PHASE_AND_RANDOMNESS,
        0,
        total_and_gates,
    )?;
    ensure!(
        previous_randomness.iter().all(|bit| *bit <= 1),
        "canary MPC received non-bit AND randomness"
    );

    let mut circuit = BooleanCircuit {
        transport,
        tag,
        local_randomness,
        previous_randomness,
        random_cursor: 0,
        and_step: 0,
    };
    let first_sum = circuit.add_words(&source0, &source1).await?;
    let sum = circuit.add_words(&first_sum, &source2).await?;
    let nonzero = circuit
        .or_reduce(sum.into_iter().flatten().collect())
        .await?;
    ensure!(
        circuit.random_cursor == total_and_gates,
        "canary MPC did not consume its exact AND budget"
    );

    // Send the local component to the previous party and receive the missing
    // next component. This is the protocol's only reconstruction.
    let opened = circuit
        .transport
        .exchange_previous(encode_frame(tag, PHASE_FINAL_OPEN, 0, &[nonzero.local]))
        .await?;
    let opened = decode_frame(opened, tag, PHASE_FINAL_OPEN, 0, 1)?;
    ensure!(opened[0] <= 1, "canary MPC final opening is not a bit");
    Ok(nonzero.local ^ nonzero.previous ^ opened[0] != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};
    use tokio::sync::Notify;

    struct LocalNetwork {
        queues: Vec<Vec<Mutex<VecDeque<Vec<u8>>>>>,
        notify: Vec<Vec<Notify>>,
        open_payload_lengths: Mutex<Vec<usize>>,
    }

    impl LocalNetwork {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                queues: (0..3)
                    .map(|_| (0..3).map(|_| Mutex::new(VecDeque::new())).collect())
                    .collect(),
                notify: (0..3)
                    .map(|_| (0..3).map(|_| Notify::new()).collect())
                    .collect(),
                open_payload_lengths: Mutex::new(Vec::new()),
            })
        }

        fn send(&self, from: usize, to: usize, message: Vec<u8>) {
            if message.get(FRAME_MAGIC.len() + SESSION_TAG_LEN) == Some(&PHASE_FINAL_OPEN) {
                self.open_payload_lengths
                    .lock()
                    .unwrap()
                    .push(message.len() - FRAME_HEADER_LEN);
            }
            self.queues[from][to].lock().unwrap().push_back(message);
            self.notify[from][to].notify_one();
        }

        async fn receive(&self, from: usize, to: usize) -> Vec<u8> {
            loop {
                let notified = self.notify[from][to].notified();
                if let Some(message) = self.queues[from][to].lock().unwrap().pop_front() {
                    return message;
                }
                notified.await;
            }
        }
    }

    struct LocalTransport {
        party_id: usize,
        network: Arc<LocalNetwork>,
    }

    impl BooleanMpcTransport for LocalTransport {
        fn party_id(&self) -> usize {
            self.party_id
        }

        async fn exchange_next(&mut self, message: Vec<u8>) -> Result<Vec<u8>> {
            let next = (self.party_id + 1) % 3;
            let previous = (self.party_id + 2) % 3;
            self.network.send(self.party_id, next, message);
            Ok(self.network.receive(previous, self.party_id).await)
        }

        async fn exchange_previous(&mut self, message: Vec<u8>) -> Result<Vec<u8>> {
            let next = (self.party_id + 1) % 3;
            let previous = (self.party_id + 2) % 3;
            self.network.send(self.party_id, previous, message);
            Ok(self.network.receive(next, self.party_id).await)
        }
    }

    async fn run_private_test(
        shares: [Vec<u16>; 3],
        repetitions: usize,
    ) -> (Vec<bool>, Arc<LocalNetwork>) {
        let network = LocalNetwork::new();
        let run = |party_id: usize| {
            let network = network.clone();
            let share = shares[party_id].clone();
            async move {
                let mut transport = LocalTransport { party_id, network };
                private_any_nonzero(share, repetitions, [7; 32], &mut transport)
                    .await
                    .unwrap()
            }
        };
        let (a, b, c) = tokio::join!(run(0), run(1), run(2));
        (vec![a, b, c], network)
    }

    #[tokio::test]
    async fn boolean_mpc_opens_only_one_zero_test_bit() {
        let mut shares = [vec![0; 5], vec![0; 5], vec![0; 5]];
        shares[0][0] = 10;
        shares[1][0] = 20;
        shares[2][0] = u16::MAX - 29;
        shares[0][1] = u16::MAX;
        shares[1][1] = 1;
        let (result, network) = run_private_test(shares, 1).await;
        assert_eq!(result, vec![false; 3]);
        assert_eq!(*network.open_payload_lengths.lock().unwrap(), vec![1; 3]);
    }

    #[tokio::test]
    async fn boolean_mpc_detects_nonzero_without_opening_a_word() {
        let mut shares = [vec![0; 5], vec![0; 5], vec![0; 5]];
        shares[1][3] = 1;
        let (result, network) = run_private_test(shares, 1).await;
        assert_eq!(result, vec![true; 3]);
        assert_eq!(*network.open_payload_lengths.lock().unwrap(), vec![1; 3]);
    }

    #[test]
    fn inventory_sentinel_detects_a_missing_all_zero_row() {
        const REPETITIONS: usize = 3;
        let zero_code = vec![0; IRIS_CODE_LENGTH];
        let zero_mask = vec![0; MASK_CODE_LENGTH];
        let mut shares = Vec::new();
        for party in 0..3 {
            let mut accumulator = CanaryAccumulator::new(party, REPETITIONS, [9; 32]);
            accumulator
                .accumulate(1, &zero_code, &zero_mask, &zero_code, &zero_mask)
                .unwrap();
            if party != 2 {
                accumulator
                    .accumulate(2, &zero_code, &zero_mask, &zero_code, &zero_mask)
                    .unwrap();
            }
            shares.push(accumulator.into_syndrome_share());
        }
        assert!((0..shares[0].len()).any(|index| {
            shares
                .iter()
                .map(|share| share[index])
                .fold(0u16, u16::wrapping_add)
                != 0
        }));
    }
}
