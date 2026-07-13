//! Absolute, self-describing rerandomization offsets.
//!
//! A row at epoch `e` contains `share_i + R_e(id) * x_i`, where `x_i` is the
//! party's public evaluation point and `R_0` is zero.  Retargeting subtracts
//! the source offset and adds the destination offset, so it composes exactly
//! and never changes the shared secret.

use std::io::Read;

use eyre::{ensure, Result};

use crate::galois::degree4::{basis::Monomial, GaloisRingElement};
use crate::{IRIS_CODE_LENGTH, MASK_CODE_LENGTH};

pub type RerandEpoch = i32;
pub type EpochSeed = [u8; 32];

const EPOCH_SEED_DOMAIN: &str = "iris-mpc/rerand-v2/epoch-seed/v1";
const SEED_COMMITMENT_DOMAIN: &str = "iris-mpc/rerand-v2/seed-commitment/v1";
const OFFSET_DOMAIN: &[u8] = b"iris-mpc/rerand-v2/row-offset/v1";

pub fn epoch_seed_from_dh(shared_secret: &[u8]) -> EpochSeed {
    blake3::derive_key(EPOCH_SEED_DOMAIN, shared_secret)
}

pub fn seed_commitment(seed: &EpochSeed) -> [u8; 32] {
    blake3::derive_key(SEED_COMMITMENT_DOMAIN, seed)
}

#[derive(Clone, Copy)]
pub struct EpochKey<'a> {
    pub epoch: RerandEpoch,
    pub seed: Option<&'a EpochSeed>,
}

impl<'a> EpochKey<'a> {
    pub fn new(epoch: RerandEpoch, seed: Option<&'a EpochSeed>) -> Self {
        Self { epoch, seed }
    }

    fn validate(self) -> Result<()> {
        ensure!(self.epoch >= 0, "negative rerandomization epoch");
        ensure!(
            self.epoch == 0 || self.seed.is_some(),
            "missing seed for rerandomization epoch {}",
            self.epoch
        );
        Ok(())
    }
}

struct OffsetStream(Option<blake3::OutputReader>);

impl OffsetStream {
    fn new(key: EpochKey<'_>, serial_id: i64) -> Result<Self> {
        key.validate()?;
        ensure!(serial_id >= 0, "negative serial id");
        if key.epoch == 0 {
            return Ok(Self(None));
        }
        let mut hasher = blake3::Hasher::new_keyed(key.seed.expect("validated"));
        hasher.update(OFFSET_DOMAIN);
        hasher.update(&(key.epoch as u32).to_le_bytes());
        hasher.update(&(serial_id as u64).to_le_bytes());
        Ok(Self(Some(hasher.finalize_xof())))
    }

    fn next(&mut self) -> GaloisRingElement<Monomial> {
        let Some(xof) = &mut self.0 else {
            return GaloisRingElement::ZERO;
        };
        let mut bytes = [0; 8];
        xof.read_exact(&mut bytes).expect("BLAKE3 XOF is unbounded");
        GaloisRingElement::from_coefs([
            u16::from_le_bytes(bytes[0..2].try_into().unwrap()),
            u16::from_le_bytes(bytes[2..4].try_into().unwrap()),
            u16::from_le_bytes(bytes[4..6].try_into().unwrap()),
            u16::from_le_bytes(bytes[6..8].try_into().unwrap()),
        ])
    }
}

#[allow(clippy::too_many_arguments)]
pub fn retarget_shares(
    party_id: usize,
    serial_id: i64,
    from: EpochKey<'_>,
    to: EpochKey<'_>,
    left_code: &mut [u16],
    left_mask: &mut [u16],
    right_code: &mut [u16],
    right_mask: &mut [u16],
) -> Result<()> {
    ensure!(party_id < 3, "party id must be 0, 1, or 2");
    from.validate()?;
    to.validate()?;
    if from.epoch == to.epoch {
        return Ok(());
    }
    ensure!(
        left_code.len() == IRIS_CODE_LENGTH
            && right_code.len() == IRIS_CODE_LENGTH
            && left_mask.len() == MASK_CODE_LENGTH
            && right_mask.len() == MASK_CODE_LENGTH,
        "unexpected share lengths"
    );

    let x = GaloisRingElement::<Monomial>::EXCEPTIONAL_SEQUENCE[party_id + 1];
    let mut from = OffsetStream::new(from, serial_id)?;
    let mut to = OffsetStream::new(to, serial_id)?;
    for coefficients in [left_code, left_mask, right_code, right_mask] {
        for chunk in coefficients.chunks_exact_mut(4) {
            let stored = GaloisRingElement::from_coefs(chunk.try_into().unwrap());
            let updated = stored + (to.next() - from.next()) * x;
            chunk.copy_from_slice(&updated.coefs);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_row() -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
        (
            vec![0; IRIS_CODE_LENGTH],
            vec![0; MASK_CODE_LENGTH],
            vec![0; IRIS_CODE_LENGTH],
            vec![0; MASK_CODE_LENGTH],
        )
    }

    fn retarget(
        row: &mut (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>),
        party: usize,
        from: EpochKey<'_>,
        to: EpochKey<'_>,
    ) {
        retarget_shares(
            party, 42, from, to, &mut row.0, &mut row.1, &mut row.2, &mut row.3,
        )
        .unwrap();
    }

    #[test]
    fn retargets_compose_and_invert() {
        let a = [7; 32];
        let b = [42; 32];
        for party in 0..3 {
            let original = zero_row();
            let mut stepped = original.clone();
            retarget(
                &mut stepped,
                party,
                EpochKey::new(0, None),
                EpochKey::new(1, Some(&a)),
            );
            retarget(
                &mut stepped,
                party,
                EpochKey::new(1, Some(&a)),
                EpochKey::new(2, Some(&b)),
            );
            let mut direct = original.clone();
            retarget(
                &mut direct,
                party,
                EpochKey::new(0, None),
                EpochKey::new(2, Some(&b)),
            );
            assert_eq!(stepped, direct);
            retarget(
                &mut direct,
                party,
                EpochKey::new(2, Some(&b)),
                EpochKey::new(0, None),
            );
            assert_eq!(direct, original);
        }
    }

    #[test]
    fn missing_seed_and_bad_party_fail_closed() {
        let mut row = zero_row();
        assert!(retarget_shares(
            0,
            1,
            EpochKey::new(3, None),
            EpochKey::new(0, None),
            &mut row.0,
            &mut row.1,
            &mut row.2,
            &mut row.3,
        )
        .is_err());
        assert!(retarget_shares(
            3,
            1,
            EpochKey::new(0, None),
            EpochKey::new(0, None),
            &mut row.0,
            &mut row.1,
            &mut row.2,
            &mut row.3,
        )
        .is_err());
    }

    #[test]
    fn derivation_is_deterministic_and_domain_separated() {
        let seed = epoch_seed_from_dh(&[1; 32]);
        assert_eq!(seed, epoch_seed_from_dh(&[1; 32]));
        assert_ne!(seed, epoch_seed_from_dh(&[2; 32]));
        assert_ne!(seed, seed_commitment(&seed));
    }

    #[test]
    fn offset_layout_golden_vector() {
        let seed = [7; 32];
        let mut row = zero_row();
        retarget(
            &mut row,
            1,
            EpochKey::new(0, None),
            EpochKey::new(1, Some(&seed)),
        );
        let mut hash = blake3::Hasher::new();
        for coefficients in [&row.0, &row.1, &row.2, &row.3] {
            for coefficient in coefficients {
                hash.update(&coefficient.to_le_bytes());
            }
        }
        assert_eq!(
            hash.finalize().to_hex().as_str(),
            "97df5e6d70abe2d5aeba3ee4fc05629c9b7493efc5086a8368fab0b800509306"
        );
    }
}
