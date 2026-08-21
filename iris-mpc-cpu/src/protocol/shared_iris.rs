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

    /// Produce the mirrored variant of this iris share.
    pub fn mirrored(&self) -> Self {
        GaloisRingSharedIris {
            code: self.code.mirrored_code(),
            mask: self.mask.mirrored(),
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

/// 8-byte-interleaved lo/hi plane representation of an iris share for the
/// UMMLA-based exact-scan kernel: every 8 consecutive u16 coefficients are
/// stored as `[lo0..lo7 | hi0..hi7]`. Same byte count as the u16 form; the
/// original share is reconstructed exactly by [`MixedPlaneIris::to_iris`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixedPlaneIris {
    /// `id` of the code share (party_id + 1), preserved for reconstruction.
    code_id: usize,
    /// `id` of the mask share (party_id + 1), preserved for reconstruction.
    mask_id: usize,
    /// 16 rows x 1600 bytes.
    code: Box<[u8]>,
    /// 8 rows x 1600 bytes.
    mask: Box<[u8]>,
}

fn mix_planes(src: &[u16], dst: &mut [u8]) {
    debug_assert_eq!(src.len() * 2, dst.len());
    for (src8, dst16) in src.chunks_exact(8).zip(dst.chunks_exact_mut(16)) {
        for k in 0..8 {
            dst16[k] = src8[k] as u8;
            dst16[8 + k] = (src8[k] >> 8) as u8;
        }
    }
}

fn unmix_planes(src: &[u8], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len() * 2);
    for (dst8, src16) in dst.chunks_exact_mut(8).zip(src.chunks_exact(16)) {
        for k in 0..8 {
            dst8[k] = src16[k] as u16 | ((src16[8 + k] as u16) << 8);
        }
    }
}

impl MixedPlaneIris {
    pub fn from_iris(iris: &GaloisRingSharedIris) -> Self {
        let mut code = vec![0u8; iris.code.coefs.len() * 2].into_boxed_slice();
        let mut mask = vec![0u8; iris.mask.coefs.len() * 2].into_boxed_slice();
        mix_planes(&iris.code.coefs, &mut code);
        mix_planes(&iris.mask.coefs, &mut mask);
        Self {
            code_id: iris.code.id,
            mask_id: iris.mask.id,
            code,
            mask,
        }
    }

    pub fn to_iris(&self) -> GaloisRingSharedIris {
        let mut iris = GaloisRingSharedIris::default_for_party(0);
        iris.code.id = self.code_id;
        iris.mask.id = self.mask_id;
        unmix_planes(&self.code, &mut iris.code.coefs);
        unmix_planes(&self.mask, &mut iris.mask.coefs);
        iris
    }

    #[inline(always)]
    pub fn code_planes(&self) -> &[u8] {
        &self.code
    }

    #[inline(always)]
    pub fn mask_planes(&self) -> &[u8] {
        &self.mask
    }
}

/// Resident layout of a worker pool's iris store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentLayout {
    /// Plain `ArcIris` values; all access paths borrow them at zero cost.
    U16,
    /// Mixed lo/hi plane values for the UMMLA exact-scan kernel. Non-scan
    /// access paths reconstruct the u16 share on demand.
    MixedPlane,
}

/// Layout to use for exact-scan worker pools on this machine: mixed planes
/// when the UMMLA kernel is available (aarch64 with i8mm), unless disabled
/// via `IRIS_MPC_DISABLE_MIXED_SCAN=1`.
pub fn preferred_scan_layout() -> ResidentLayout {
    #[cfg(target_arch = "aarch64")]
    {
        let disabled = std::env::var("IRIS_MPC_DISABLE_MIXED_SCAN")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if !disabled && std::arch::is_aarch64_feature_detected!("i8mm") {
            return ResidentLayout::MixedPlane;
        }
    }
    ResidentLayout::U16
}

/// An iris share as resident in a worker pool store, in the pool's layout.
#[derive(Debug, Clone)]
pub enum ResidentIris {
    U16(ArcIris),
    Mixed(Arc<MixedPlaneIris>),
}

impl ResidentIris {
    pub fn from_arc(iris: ArcIris, layout: ResidentLayout) -> Self {
        match layout {
            ResidentLayout::U16 => Self::U16(iris),
            ResidentLayout::MixedPlane => Self::Mixed(Arc::new(MixedPlaneIris::from_iris(&iris))),
        }
    }

    /// The u16 form: a cheap handle clone for `U16`, an exact reconstruction
    /// for `Mixed`.
    pub fn to_arc(&self) -> ArcIris {
        match self {
            Self::U16(iris) => iris.clone(),
            Self::Mixed(planes) => Arc::new(planes.to_iris()),
        }
    }

    #[inline(always)]
    pub fn as_mixed(&self) -> Option<&MixedPlaneIris> {
        match self {
            Self::U16(_) => None,
            Self::Mixed(planes) => Some(planes),
        }
    }
}

#[cfg(test)]
mod mixed_plane_tests {
    use super::*;
    use iris_mpc_common::iris_db::iris::IrisCode;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn mixed_plane_round_trip() {
        let mut rng = StdRng::seed_from_u64(7);
        let iris = IrisCode::random_rng(&mut rng);
        for share in GaloisRingSharedIris::generate_shares_locally(&mut rng, iris) {
            let planes = MixedPlaneIris::from_iris(&share);
            assert_eq!(planes.to_iris(), share);
        }
    }
}
