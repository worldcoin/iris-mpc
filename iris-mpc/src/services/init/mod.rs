use eyre::{ensure, eyre, Result};
use hkdf::Hkdf;
use iris_mpc_common::{
    config::Config,
    helpers::{kms_dh::derive_shared_secret, sync::SyncState},
};
use sha2::Sha256;

/// Number of MPC parties; the seed transcript carries one contribution each.
pub const N_PARTIES: usize = 3;

/// Public, fixed HKDF salt for the ChaCha seed derivation (HKDF is keyed by the
/// ECDH secret; the salt only provides domain separation).
const CHACHA_SEED_KDF_SALT: &[u8] = b"iris-mpc-gpu/chacha-seed-salt";

/// Domain separator prefixed to the HKDF `info` string. Bump the version on any
/// change to the derivation so mismatched builds fail instead of half-agreeing.
const CHACHA_SEED_KDF_INFO: &[u8] = b"iris-mpc-gpu/chacha-seed/v2";

/// Assemble the per-startup seed transcript from this party's contribution and
/// the peers' `SyncState`s (as returned by `get_others_sync_state`). The
/// contributions are sorted so every node hashes the same bytes regardless of
/// the order peers come back in.
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

    // Peer ids for diagnostics only; `get_others_sync_state` returns peers in
    // ascending party-id order.
    let peer_ids: Vec<usize> = (0..N_PARTIES).filter(|id| *id != party_id).collect();

    let mut nonces = [[0u8; 32]; N_PARTIES];
    nonces[0] = my_dh_nonce;
    for ((slot, state), peer_id) in nonces[1..].iter_mut().zip(other_states).zip(peer_ids) {
        // A peer without a nonce cannot reproduce our seeds; refuse to start.
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

/// Derive this party's two pairwise ChaCha seeds via HKDF from the static KMS
/// ECDH secrets, mixing in the per-startup `dh_nonces` (assembled by
/// [`seed_transcript_nonces`]) so each run's seeds are fresh — their keystream
/// is used as a one-time pad on the NCCL wire. The nonces are public; security
/// rests on the ECDH secret keying the HKDF, and this party's own contribution
/// alone guarantees its keystream is fresh regardless of what the peers send.
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

    // Identical on all three nodes, so both members of a pair derive the same seed.
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
/// startup transcript and to the pair of parties sharing it (ids sorted, so
/// both members produce the same binding).
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

    /// Both members of a pair must land on the same seed.
    #[test]
    fn pair_members_agree() {
        let transcript = nonces(7).concat();
        let from_0 = derive_chacha_seed(&SECRET_01, 0, 1, &transcript).unwrap();
        let from_1 = derive_chacha_seed(&SECRET_01, 1, 0, &transcript).unwrap();
        assert_eq!(from_0, from_1);
    }

    /// A different startup transcript must give a different seed even though
    /// the ECDH secret is unchanged.
    #[test]
    fn transcript_changes_seed() {
        let a = derive_chacha_seed(&SECRET_01, 0, 1, &nonces(7).concat()).unwrap();
        let b = derive_chacha_seed(&SECRET_01, 0, 1, &nonces(8).concat()).unwrap();
        assert_ne!(a, b);
    }

    /// Changing any single contribution is enough to refresh the seed.
    #[test]
    fn own_contribution_alone_refreshes_seed() {
        let mut first = nonces(7);
        let mut second = first;
        second[0] = [0xab; 32];

        let a = derive_chacha_seed(&SECRET_01, 0, 1, &first.concat()).unwrap();
        let b = derive_chacha_seed(&SECRET_01, 0, 1, &second.concat()).unwrap();
        assert_ne!(a, b);

        // ... including from a party outside this pair.
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

    /// All parties must land on the same transcript regardless of peer order.
    #[test]
    fn transcript_is_identical_on_every_party() {
        let [n0, n1, n2] = nonces(7);

        let from_0 =
            seed_transcript_nonces(0, n0, &[peer_state(Some(n1)), peer_state(Some(n2))]).unwrap();
        let from_1 =
            seed_transcript_nonces(1, n1, &[peer_state(Some(n0)), peer_state(Some(n2))]).unwrap();
        // Peers delivered in the opposite order.
        let from_2 =
            seed_transcript_nonces(2, n2, &[peer_state(Some(n1)), peer_state(Some(n0))]).unwrap();

        assert_eq!(from_0, from_1);
        assert_eq!(from_0, from_2);
    }

    /// A peer on a build without the field must cause startup to fail.
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
