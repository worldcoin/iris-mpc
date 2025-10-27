use crate::utils::constants::N_PARTIES;
use eyre::Result;
use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    iris_db::{get_dummy_shares_for_deletion, iris::IrisCode},
    job::IrisQueryBatchEntries,
};
use itertools::izip;
use rand::{CryptoRng, Rng};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub type ArcIris = Arc<GaloisRingSharedIris>;

#[derive(PartialEq, Eq, Debug, Clone, Serialize, Deserialize, Hash)]
pub struct GaloisRingSharedIris {
    pub code: GaloisRingIrisCodeShare,
    pub mask: GaloisRingTrimmedMaskCodeShare,
}

// Pair of Iris shares associated with left/right eyes.
pub type GaloisRingSharedIrisPair = (GaloisRingSharedIris, GaloisRingSharedIris);

// Set of pairs of Iris shares associated with left/right eyes.
pub type GaloisRingSharedIrisPairSet = [GaloisRingSharedIrisPair; N_PARTIES];

impl GaloisRingSharedIris {
    /// Empty code and mask share. party_id is 0-based.
    pub fn default_for_party(party_id: usize) -> Self {
        GaloisRingSharedIris {
            code: GaloisRingIrisCodeShare::default_for_party(party_id),
            mask: GaloisRingTrimmedMaskCodeShare::default_for_party(party_id),
        }
    }

    pub fn dummy_for_party(party_id: usize) -> Self {
        let (code, mask) = get_dummy_shares_for_deletion(party_id);
        GaloisRingSharedIris { code, mask }
    }

    pub fn from_batch(batch: IrisQueryBatchEntries) -> Vec<Self> {
        izip!(batch.code, batch.mask)
            .map(|(code, mask)| GaloisRingSharedIris { code, mask })
            .collect()
    }

    pub fn to_batch(shares: &[Self]) -> IrisQueryBatchEntries {
        IrisQueryBatchEntries {
            code: shares.iter().map(|s| s.code.clone()).collect(),
            mask: shares.iter().map(|s| s.mask.clone()).collect(),
        }
    }

    pub fn try_from_buffers(party_id: usize, code: &[u16], mask: &[u16]) -> Result<Arc<Self>> {
        Ok(Arc::new(Self::try_from_buffers_inner(
            party_id, code, mask,
        )?))
    }

    pub fn try_from_buffers_inner(party_id: usize, code: &[u16], mask: &[u16]) -> Result<Self> {
        Ok(GaloisRingSharedIris {
            code: GaloisRingIrisCodeShare::new(code.try_into()?, party_id),
            mask: GaloisRingTrimmedMaskCodeShare::new(mask.try_into()?, party_id),
        })
    }

    /// Generate iris code shares of an input iris code using local randomness, alongside with
    /// its mirrored component
    pub fn generate_shares_locally<R: Rng + CryptoRng>(
        rng: &mut R,
        iris: IrisCode,
    ) -> [GaloisRingSharedIris; 3] {
        let code_shares = GaloisRingIrisCodeShare::encode_iris_code(&iris.code, &iris.mask, rng);
        let mask_shares = GaloisRingIrisCodeShare::encode_mask_code(&iris.mask, rng);
        [
            GaloisRingSharedIris {
                code: code_shares[0].clone(),
                mask: GaloisRingTrimmedMaskCodeShare::from(&mask_shares[0]),
            },
            GaloisRingSharedIris {
                code: code_shares[1].clone(),
                mask: GaloisRingTrimmedMaskCodeShare::from(&mask_shares[1]),
            },
            GaloisRingSharedIris {
                code: code_shares[2].clone(),
                mask: GaloisRingTrimmedMaskCodeShare::from(&mask_shares[2]),
            },
        ]
    }
    /// Generate mirrored iris code shares of an input iris code using local randomness
    pub fn generate_mirrored_shares_locally<R: Rng + CryptoRng>(
        rng: &mut R,
        iris: IrisCode,
    ) -> [GaloisRingSharedIris; 3] {
        let code_shares = GaloisRingIrisCodeShare::encode_iris_code(&iris.code, &iris.mask, rng);
        let mask_shares = GaloisRingIrisCodeShare::encode_mask_code(&iris.mask, rng);
        let code_shares_mirrored = code_shares
            .iter()
            .map(|code| code.mirrored_code())
            .collect::<Vec<_>>();
        let mask_shares_mirrored = mask_shares
            .iter()
            .map(|mask| mask.mirrored_mask())
            .collect::<Vec<_>>();
        [
            GaloisRingSharedIris {
                code: code_shares_mirrored[0].clone(),
                mask: GaloisRingTrimmedMaskCodeShare::from(&mask_shares_mirrored[0]),
            },
            GaloisRingSharedIris {
                code: code_shares_mirrored[1].clone(),
                mask: GaloisRingTrimmedMaskCodeShare::from(&mask_shares_mirrored[1]),
            },
            GaloisRingSharedIris {
                code: code_shares_mirrored[2].clone(),
                mask: GaloisRingTrimmedMaskCodeShare::from(&mask_shares_mirrored[2]),
            },
        ]
    }
}
