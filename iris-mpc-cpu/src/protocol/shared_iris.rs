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
            code: GaloisRingIrisCodeShare::new(party_id, code.try_into()?),
            mask: GaloisRingTrimmedMaskCodeShare::new(party_id, mask.try_into()?),
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

#[cfg(test)]
mod tests {
    use super::*;
    use eyre::Result;
    use iris_mpc_common::{IRIS_CODE_LENGTH, MASK_CODE_LENGTH};
    use rand::thread_rng;

    #[test]
    fn test_generate_shares_locally() -> Result<()> {
        let iris = IrisCode::random_rng(&mut thread_rng());
        let shares = GaloisRingSharedIris::generate_shares_locally(&mut thread_rng(), iris);
        for (party_id, share) in shares.iter().enumerate() {
            share.code.validate_party_id(party_id)?;
            share.mask.validate_party_id(party_id)?;
        }
        Ok(())
    }

    #[test]
    fn test_try_from_buffers() -> Result<()> {
        let party_id = 2;
        let code = vec![123; IRIS_CODE_LENGTH];
        let mask = vec![456; MASK_CODE_LENGTH];
        let share = GaloisRingSharedIris::try_from_buffers(party_id, &code, &mask)?;
        share.code.validate_party_id(party_id)?;
        share.mask.validate_party_id(party_id)?;
        assert_eq!(share.code.coefs[0], 123);
        assert_eq!(share.mask.coefs[0], 456);
        Ok(())
    }
}
