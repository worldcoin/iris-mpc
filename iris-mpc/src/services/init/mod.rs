use eyre::{ensure, eyre, Result};
use hkdf::Hkdf;
use iris_mpc_common::{
    config::Config,
    helpers::{kms_dh::derive_shared_secret, sync::SyncState},
};
use sha2::Sha256;

/// Number of MPC parties. The seed transcript always carries exactly one
/// contribution per party, so that all three nodes hash the same bytes.
pub const N_PARTIES: usize = 3;

/// Public, fixed HKDF salt for the ChaCha seed derivation. The salt need not be
/// secret (HKDF is keyed by the ECDH secret passed as IKM); it only provides
/// domain separation from other uses of the same shared secret.
const CHACHA_SEED_KDF_SALT: &[u8] = b"iris-mpc-gpu/chacha-seed-salt";

/// Domain separator prefixed to the HKDF `info` string.
///
/// Bump the version suffix on any change to the derivation. Parties running
/// different versions must derive visibly different seeds and fail, rather than
/// half-agreeing on a seed derived two different ways.
const CHACHA_SEED_KDF_INFO: &[u8] = b"iris-mpc-gpu/chacha-seed/v2";

/// Assemble the per-startup seed transcript from this party's own contribution
/// and the ones the peers published in their `SyncState`.
///
/// `other_states` is what `get_others_sync_state` returns: the two peers, this
/// party excluded. The contributions are sorted rather than slotted by party id,
/// so the transcript does not depend on the order peers come back in — every
/// node sees the same three values and therefore hashes the same bytes.
pub fn seed_transcript_nonces(
    party_id: usize,
    my_dh_nonce: [u8; 32],
    other_states: &[SyncState],
) -> Result<[[u8; 32]; N_PARTIES]> {
    ensure!(party_id < N_PARTIES, "party_id {party_id} out of range");
    ensure!(
        other_states.len() == N_PARTIES - 1,
        "expected {} peer sync states, got {}",
        N_PARTIES - 1,
        other_states.len()
    );

    // Best-effort labelling for diagnostics only: `get_others_sync_state`
    // returns peers in ascending party-id order. Nothing below depends on this
    // being right, so a change on that side degrades an error message rather
    // than the derivation.
    let peer_ids: Vec<usize> = (0..N_PARTIES).filter(|id| *id != party_id).collect();

    let mut nonces = [[0u8; 32]; N_PARTIES];
    nonces[0] = my_dh_nonce;
    for ((slot, state), peer_id) in nonces[1..].iter_mut().zip(other_states).zip(peer_ids) {
        // Seeds have to match across all parties, so a peer that publishes no
        // nonce cannot reproduce ours. Refuse to start rather than run the
        // protocol with mismatched keystreams.
        *slot = state.dh_nonce.ok_or_else(|| {
            eyre!(
                "party {peer_id} published no startup DH nonce; it is running a build that \
                 predates per-startup seed derivation. All parties must be upgraded together."
            )
        })?;
    }

    ensure!(
        nonces.iter().all(|nonce| nonce != &[0u8; 32]),
        "a party published an all-zero startup DH nonce"
    );

    nonces.sort_unstable();
    Ok(nonces)
}

/// Derive this party's two pairwise ChaCha seeds.
///
/// The long-term input is the raw ECDH secret KMS derives from the two parties'
/// static `KEY_AGREEMENT` key pairs. That value is the same on every run, so on
/// its own it produces the same keystream after every restart — which matters
/// because the keystream is used as a one-time pad on the NCCL wire. The
/// per-startup `dh_nonces` — one 32-byte contribution per party, exchanged over
/// the startup-sync endpoint and assembled by [`seed_transcript_nonces`] — are
/// mixed in as HKDF `info` to make each run's seeds independent.
///
/// The nonces are public. Security rests on the ECDH secret keying the HKDF; the
/// nonces only supply freshness, and this party's own contribution is enough to
/// guarantee that its own keystream is fresh regardless of what the peers send.
pub async fn initialize_chacha_seeds(
    config: &Config,
    dh_nonces: &[[u8; 32]; N_PARTIES],
) -> Result<([u32; 8], [u32; 8])> {
    // Init RNGs
    let own_key_arn = config
        .kms_key_arns
        .0
        .get(config.party_id)
        .expect("Expected value not found in kms_key_arns");
    let dh_pairs = match config.party_id {
        0 => (1usize, 2usize),
        1 => (2usize, 0usize),
        2 => (0usize, 1usize),
        _ => unimplemented!(),
    };

    let dh_pair_0: &str = config
        .kms_key_arns
        .0
        .get(dh_pairs.0)
        .expect("Expected value not found in kms_key_arns");
    let dh_pair_1: &str = config
        .kms_key_arns
        .0
        .get(dh_pairs.1)
        .expect("Expected value not found in kms_key_arns");

    // Identical on all three nodes, so both members of a pair feed the same
    // bytes into the KDF and land on the same seed.
    let transcript = dh_nonces.concat();

    // To be used only for e2e testing where we use localstack. There's a bug in
    // localstack's implementation of `derive_shared_secret`. See: https://github.com/localstack/localstack/pull/12071
    let chacha_seeds: ([u32; 8], [u32; 8]) = if config.fixed_shared_secrets {
        ([0u32; 8], [0u32; 8])
    } else {
        (
            derive_chacha_seed(
                &derive_shared_secret(own_key_arn, dh_pair_0).await?,
                config.party_id,
                dh_pairs.0,
                &transcript,
            )?,
            derive_chacha_seed(
                &derive_shared_secret(own_key_arn, dh_pair_1).await?,
                config.party_id,
                dh_pairs.1,
                &transcript,
            )?,
        )
    };

    Ok(chacha_seeds)
}

/// HKDF-SHA256 the raw pairwise ECDH secret into a ChaCha seed, bound to the
/// startup transcript and to the (unordered) pair of parties that share it.
///
/// The pair ids are sorted so that both members of a pair produce the same
/// binding: party 0's "next" seed is party 1's "prev" seed.
fn derive_chacha_seed(
    shared_secret: &[u8; 32],
    my_party_id: usize,
    peer_party_id: usize,
    transcript: &[u8],
) -> Result<[u32; 8]> {
    let pair = [
        my_party_id.min(peer_party_id) as u8,
        my_party_id.max(peer_party_id) as u8,
    ];

    let hkdf = Hkdf::<Sha256>::new(Some(CHACHA_SEED_KDF_SALT), shared_secret);
    let mut seed = [0u8; 32];
    hkdf.expand_multi_info(&[CHACHA_SEED_KDF_INFO, &pair, transcript], &mut seed)
        .map_err(|e| eyre!("HKDF expansion for ChaCha seed failed: {e}"))?;

    Ok(bytemuck::cast(seed))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET_01: [u8; 32] = [1u8; 32];
    const SECRET_02: [u8; 32] = [2u8; 32];

    fn nonces(tag: u8) -> [[u8; 32]; N_PARTIES] {
        [
            [tag; 32],
            [tag.wrapping_add(1); 32],
            [tag.wrapping_add(2); 32],
        ]
    }

    /// Both members of a pair must land on the same seed: party 0 derives it as
    /// its "next" partner, party 1 as its "prev" partner.
    #[test]
    fn pair_members_agree() {
        let transcript = nonces(7).concat();
        let from_0 = derive_chacha_seed(&SECRET_01, 0, 1, &transcript).unwrap();
        let from_1 = derive_chacha_seed(&SECRET_01, 1, 0, &transcript).unwrap();
        assert_eq!(from_0, from_1);
    }

    /// The whole point of the change: a different startup transcript must give a
    /// different seed even though the ECDH secret is unchanged.
    #[test]
    fn transcript_changes_seed() {
        let a = derive_chacha_seed(&SECRET_01, 0, 1, &nonces(7).concat()).unwrap();
        let b = derive_chacha_seed(&SECRET_01, 0, 1, &nonces(8).concat()).unwrap();
        assert_ne!(a, b);
    }

    /// A single party changing its own contribution is enough to refresh the
    /// seed, so a party is never at the mercy of a peer replaying a stale nonce.
    #[test]
    fn own_contribution_alone_refreshes_seed() {
        let mut first = nonces(7);
        let mut second = first;
        second[0] = [0xab; 32];

        let a = derive_chacha_seed(&SECRET_01, 0, 1, &first.concat()).unwrap();
        let b = derive_chacha_seed(&SECRET_01, 0, 1, &second.concat()).unwrap();
        assert_ne!(a, b);

        // ... and likewise for a contribution from a party outside this pair.
        first[2] = [0xcd; 32];
        let c = derive_chacha_seed(&SECRET_01, 0, 1, &first.concat()).unwrap();
        assert_ne!(a, c);
    }

    /// Distinct pairwise secrets must not collapse onto the same seed.
    #[test]
    fn distinct_pairs_give_distinct_seeds() {
        let transcript = nonces(7).concat();
        let s01 = derive_chacha_seed(&SECRET_01, 0, 1, &transcript).unwrap();
        let s02 = derive_chacha_seed(&SECRET_02, 0, 2, &transcript).unwrap();
        assert_ne!(s01, s02);
    }

    fn peer_state(dh_nonce: Option<[u8; 32]>) -> SyncState {
        SyncState {
            db_len: 0,
            modifications: vec![],
            next_sns_sequence_num: None,
            common_config: Default::default(),
            graph_mutation_bytes: vec![],
            max_persisted_sequence_number: None,
            dh_nonce,
        }
    }

    /// All three parties must land on the same transcript no matter which slot
    /// each contribution arrived in.
    #[test]
    fn transcript_is_identical_on_every_party() {
        let [n0, n1, n2] = nonces(7);

        let from_0 =
            seed_transcript_nonces(0, n0, &[peer_state(Some(n1)), peer_state(Some(n2))]).unwrap();
        let from_1 =
            seed_transcript_nonces(1, n1, &[peer_state(Some(n0)), peer_state(Some(n2))]).unwrap();
        // Same party, peers delivered in the opposite order.
        let from_2 =
            seed_transcript_nonces(2, n2, &[peer_state(Some(n1)), peer_state(Some(n0))]).unwrap();

        assert_eq!(from_0, from_1);
        assert_eq!(from_0, from_2);
    }

    /// A peer on a build without the field cannot reproduce our seeds, so we
    /// must refuse to start instead of running with mismatched keystreams.
    #[test]
    fn peer_without_nonce_is_rejected() {
        let [n0, n1, _] = nonces(7);
        let err = seed_transcript_nonces(0, n0, &[peer_state(Some(n1)), peer_state(None)])
            .unwrap_err()
            .to_string();
        assert!(err.contains("party 2"), "unexpected error: {err}");
    }

    #[test]
    fn all_zero_and_wrong_peer_count_are_rejected() {
        let [n0, n1, n2] = nonces(7);
        assert!(seed_transcript_nonces(0, n0, &[peer_state(Some(n1))]).is_err());
        assert!(seed_transcript_nonces(
            0,
            n0,
            &[peer_state(Some([0u8; 32])), peer_state(Some(n2))]
        )
        .is_err());
    }
}
