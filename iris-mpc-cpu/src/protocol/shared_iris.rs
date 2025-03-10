use eyre::Result;
use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    iris_db::iris::IrisCode,
    job::IrisQueryBatchEntries,
};
use itertools::izip;
use rand::{CryptoRng, Rng};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(PartialEq, Eq, Debug, Clone, Serialize, Deserialize, Hash)]
pub struct GaloisRingSharedIris {
    pub code: GaloisRingIrisCodeShare,
    pub mask: GaloisRingTrimmedMaskCodeShare,
}

impl GaloisRingSharedIris {
    /// Empty code and mask share. party_id is 0-based.
    pub fn default_for_party(party_id: usize) -> Self {
        GaloisRingSharedIris {
            code: GaloisRingIrisCodeShare::default_for_party(party_id),
            mask: GaloisRingTrimmedMaskCodeShare::default_for_party(party_id),
        }
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
        Ok(Arc::new(GaloisRingSharedIris {
            code: GaloisRingIrisCodeShare::new(code.try_into()?, party_id),
            mask: GaloisRingTrimmedMaskCodeShare::new(mask.try_into()?, party_id),
        }))
    }

    /// Generate iris code shares of an input iris code using local randomness
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
}
