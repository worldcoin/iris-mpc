//! This module has functionality for resharing a secret shared iris code to a
//! new party, producing a valid share for the new party, without leaking
//! information about the individual shares of the sending parties.

use crate::proto::{
    self,
    iris_mpc_reshare::{IrisCodeReShare, IrisCodeReShareRequest},
};
use iris_mpc_common::{
    galois::degree4::{basis::Monomial, GaloisRingElement, ShamirGaloisRingShare},
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    id::PartyID,
    iris_db::shamir_iris::ShamirIris,
    IRIS_CODE_LENGTH, MASK_CODE_LENGTH,
};
use itertools::{izip, Itertools};
use rand::{CryptoRng, Rng, SeedableRng};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use tracing_subscriber::field::RecordFields;

pub struct IrisCodeReshareSenderHelper {
    my_party_id:     usize,
    other_party_id:  usize,
    target_party_id: usize,
    lagrange_helper: GaloisRingElement<Monomial>,
    common_seed:     [u8; 32],
    current_packet:  Option<IrisCodeReShareRequest>,
}

impl IrisCodeReshareSenderHelper {
    pub fn new(
        my_party_id: usize,
        other_party_id: usize,
        target_party_id: usize,
        common_seed: [u8; 32],
    ) -> Self {
        let lagrange_helper = ShamirGaloisRingShare::deg_1_lagrange_poly_at_v(
            my_party_id,
            other_party_id,
            target_party_id,
        );
        Self {
            my_party_id,
            other_party_id,
            target_party_id,
            lagrange_helper,
            common_seed,
            current_packet: None,
        }
    }
    fn reshare_with_random_additive_zero(
        &self,
        share: GaloisRingElement<Monomial>,
        rng: &mut (impl CryptoRng + Rng),
    ) -> GaloisRingElement<Monomial> {
        let random_mask = GaloisRingElement::<Monomial>::random(rng);
        if self.my_party_id < self.other_party_id {
            share + random_mask
        } else {
            share - random_mask
        }
    }

    fn reshare_code(
        &self,
        mut code_share: GaloisRingIrisCodeShare,
        rng: &mut (impl CryptoRng + Rng),
    ) -> Vec<u8> {
        for i in (0..IRIS_CODE_LENGTH).step_by(4) {
            let mut share = GaloisRingElement::<Monomial>::from_coefs([
                code_share.coefs[i],
                code_share.coefs[i + 1],
                code_share.coefs[i + 2],
                code_share.coefs[i + 3],
            ]);
            share = share * self.lagrange_helper;
            share = self.reshare_with_random_additive_zero(share, rng);
            code_share.coefs[i] = share.coefs[0];
            code_share.coefs[i + 1] = share.coefs[1];
            code_share.coefs[i + 2] = share.coefs[2];
            code_share.coefs[i + 3] = share.coefs[3];
        }
        code_share
            .coefs
            .into_iter()
            .map(|x| x.to_le_bytes())
            .flatten()
            .collect()
    }
    fn reshare_mask(
        &self,
        mut mask_share: GaloisRingTrimmedMaskCodeShare,
        rng: &mut (impl CryptoRng + Rng),
    ) -> Vec<u8> {
        for i in (0..MASK_CODE_LENGTH).step_by(4) {
            let mut share = GaloisRingElement::<Monomial>::from_coefs([
                mask_share.coefs[i],
                mask_share.coefs[i + 1],
                mask_share.coefs[i + 2],
                mask_share.coefs[i + 3],
            ]);
            share = share * self.lagrange_helper;
            share = self.reshare_with_random_additive_zero(share, rng);
            mask_share.coefs[i] = share.coefs[0];
            mask_share.coefs[i + 1] = share.coefs[1];
            mask_share.coefs[i + 2] = share.coefs[2];
            mask_share.coefs[i + 3] = share.coefs[3];
        }
        mask_share
            .coefs
            .into_iter()
            .map(|x| x.to_le_bytes())
            .flatten()
            .collect()
    }

    /// Start the production of a new reshare batch request.
    /// The batch will contain reshared iris codes for the given range of
    /// database indices. The start range is inclusive, the end range is
    /// exclusive.
    ///
    /// # Panics
    ///
    /// Panics if this is called while a batch is already being built.
    pub fn start_reshare_batch(&mut self, start_db_index: u64, end_db_index: u64) {
        assert!(
            self.current_packet.is_none(),
            "We expected no batch to be currently being built, but it is..."
        );
        self.current_packet = Some(IrisCodeReShareRequest {
            sender_id:                  self.my_party_id as u64,
            other_id:                   self.other_party_id as u64,
            receiver_id:                self.target_party_id as u64,
            id_range_start_inclusive:   start_db_index,
            id_range_end_non_inclusive: end_db_index,
            iris_code_re_shares:        Vec::new(),
        });
    }

    /// Adds a new iris code to the current reshare batch.
    ///
    /// # Panics
    ///
    /// Panics if this is called without [Self::start_reshare_batch] being
    /// called beforehand.
    /// Panics if this is called with an iris code id that is out of the range
    /// of the current batch.
    pub fn add_reshare_iris_to_batch(
        &mut self,
        iris_code_id: u64,
        code_share: GaloisRingIrisCodeShare,
        mask_share: GaloisRingTrimmedMaskCodeShare,
    ) {
        assert!(
            self.current_packet.is_some(),
            "We expect a batch to be currently being built"
        );
        assert!(
            self.current_packet
                .as_ref()
                .unwrap()
                .id_range_start_inclusive
                <= iris_code_id
                && self
                    .current_packet
                    .as_ref()
                    .unwrap()
                    .id_range_end_non_inclusive
                    > iris_code_id,
            "The iris code id is out of the range of the current batch"
        );
        let mut digest = Sha256::new();
        digest.update(&self.common_seed);
        digest.update(iris_code_id.to_le_bytes());
        let mut rng = rand_chacha::ChaChaRng::from_seed(digest.finalize().into());
        let reshared_code = self.reshare_code(code_share, &mut rng);
        let reshared_mask = self.reshare_mask(mask_share, &mut rng);

        let reshare = IrisCodeReShare {
            iris_code_share: reshared_code,
            mask_share:      reshared_mask,
        };
        self.current_packet
            .as_mut()
            .expect("There is currently a batch being built")
            .iris_code_re_shares
            .push(reshare);
    }

    /// Finalizes the current reshare batch and returns the reshare request.
    ///
    /// # Panics
    ///
    /// Panics if this is called without [Self::start_reshare_batch] being
    /// called beforehand. Also panics if this is called without the correct
    /// number of iris codes being added to the batch.
    pub fn finalize_reshare_batch(&mut self) -> IrisCodeReShareRequest {
        assert!(self.current_packet.is_some(), "No batch to finalize");
        let packet = self.current_packet.take().unwrap();
        assert_eq!(
            packet.iris_code_re_shares.len(),
            (packet.id_range_end_non_inclusive - packet.id_range_start_inclusive) as usize,
            "Expected the correct number of iris codes to be added to the batch"
        );
        packet
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IrisCodeReShareError {
    #[error("Invalid reshare request received: {reason}")]
    InvalidRequest { reason: String },
    #[error(
        "Too many requests received from this party ({party_id}) without matching request from \
         the other party ({other_party_id}"
    )]
    TooManyRequests {
        party_id:       usize,
        other_party_id: usize,
    },
}

#[derive(Debug)]
pub struct IrisCodeReshareReceiverHelper {
    my_party_id:      usize,
    sender1_party_id: usize,
    sender2_party_id: usize,
    max_buffer_size:  usize,
    sender_1_buffer:  VecDeque<IrisCodeReShareRequest>,
    sender_2_buffer:  VecDeque<IrisCodeReShareRequest>,
}

impl IrisCodeReshareReceiverHelper {
    pub fn new(
        my_party_id: usize,
        sender1_party_id: usize,
        sender2_party_id: usize,
        max_buffer_size: usize,
    ) -> Self {
        Self {
            my_party_id,
            sender1_party_id,
            sender2_party_id,
            max_buffer_size,
            sender_1_buffer: VecDeque::new(),
            sender_2_buffer: VecDeque::new(),
        }
    }

    fn check_valid(&self, request: &IrisCodeReShareRequest) -> Result<(), IrisCodeReShareError> {
        if request.sender_id as usize == self.sender1_party_id {
            if request.other_id as usize != self.sender2_party_id {
                return Err(IrisCodeReShareError::InvalidRequest {
                    reason: "Received a request from unexpected set of parties".to_string(),
                });
            }
        } else if request.sender_id as usize == self.sender2_party_id {
            if request.other_id as usize != self.sender1_party_id {
                return Err(IrisCodeReShareError::InvalidRequest {
                    reason: "Received a request from unexpected set of parties".to_string(),
                });
            }
        } else {
            return Err(IrisCodeReShareError::InvalidRequest {
                reason: "Received a request from unexpected set of parties".to_string(),
            });
        }
        if request.receiver_id != self.my_party_id as u64 {
            return Err(IrisCodeReShareError::InvalidRequest {
                reason: "Received a request intended for a different party".to_string(),
            });
        }
        if request.id_range_start_inclusive >= request.id_range_end_non_inclusive {
            return Err(IrisCodeReShareError::InvalidRequest {
                reason: "Invalid range of iris codes in received request".to_string(),
            });
        }
        if request.iris_code_re_shares.len()
            != (request.id_range_end_non_inclusive - request.id_range_start_inclusive) as usize
        {
            return Err(IrisCodeReShareError::InvalidRequest {
                reason: "Invalid number of iris codes in received request".to_string(),
            });
        }

        // Check that the iris code shares are of the correct length
        request.iris_code_re_shares.iter().all(|reshare| {
            reshare.iris_code_share.len() == IRIS_CODE_LENGTH * std::mem::size_of::<u16>()
                && reshare.mask_share.len() == MASK_CODE_LENGTH * std::mem::size_of::<u16>()
        });
        Ok(())
    }

    pub fn add_request_batch(
        &mut self,
        request: IrisCodeReShareRequest,
    ) -> Result<(), IrisCodeReShareError> {
        self.check_valid(&request)?;
        if request.sender_id as usize == self.sender1_party_id {
            if self.sender_1_buffer.len() + 1 >= self.max_buffer_size {
                return Err(IrisCodeReShareError::TooManyRequests {
                    party_id:       self.sender1_party_id,
                    other_party_id: self.sender2_party_id,
                });
            }
            self.sender_1_buffer.push_back(request);
        } else if request.sender_id as usize == self.sender2_party_id {
            if self.sender_2_buffer.len() + 1 >= self.max_buffer_size {
                return Err(IrisCodeReShareError::TooManyRequests {
                    party_id:       self.sender2_party_id,
                    other_party_id: self.sender1_party_id,
                });
            }
            self.sender_2_buffer.push_back(request);
        } else {
            // check valid should have caught this
            unreachable!()
        }

        Ok(())
    }

    fn check_requests_matching(
        &self,
        request1: &IrisCodeReShareRequest,
        request2: &IrisCodeReShareRequest,
    ) -> Result<(), IrisCodeReShareError> {
        if request1.id_range_start_inclusive != request2.id_range_start_inclusive
            || request1.id_range_end_non_inclusive != request2.id_range_end_non_inclusive
        {
            return Err(IrisCodeReShareError::InvalidRequest {
                reason: format!(
                    "Received requests with different iris code ranges: {}-{} from {} and {}-{} \
                     from {}",
                    request1.id_range_start_inclusive,
                    request1.id_range_end_non_inclusive,
                    request1.sender_id,
                    request2.id_range_start_inclusive,
                    request2.id_range_end_non_inclusive,
                    request2.sender_id,
                ),
            });
        }
        Ok(())
    }

    fn reshare_code_batch(
        &self,
        request1: IrisCodeReShareRequest,
        request2: IrisCodeReShareRequest,
    ) -> Result<RecombinedIrisCodeBatch, IrisCodeReShareError> {
        let len = request1.iris_code_re_shares.len();
        let mut code = Vec::with_capacity(len);
        let mut mask = Vec::with_capacity(len);

        for (reshare1, reshare2) in
            izip!(request1.iris_code_re_shares, request2.iris_code_re_shares)
        {
            // build galois shares from the u8 Vecs
            let mut code_share1 = GaloisRingIrisCodeShare {
                id:    self.my_party_id + 1,
                coefs: reshare1
                    .iris_code_share
                    .chunks_exact(std::mem::size_of::<u16>())
                    .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
                    .collect_vec()
                    .try_into()
                    // we checked this beforehand in check_valid
                    .expect("Invalid iris code share length"),
            };
            let mut mask_share1 = GaloisRingTrimmedMaskCodeShare {
                id:    self.my_party_id + 1,
                coefs: reshare1
                    .mask_share
                    .chunks_exact(std::mem::size_of::<u16>())
                    .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
                    // we checked this beforehand in check_valid
                    .collect_vec()
                    .try_into()
                    .expect("Invalid mask share length"),
            };
            let code_share2 = GaloisRingIrisCodeShare {
                id:    self.my_party_id + 1,
                coefs: reshare2
                    .iris_code_share
                    .chunks_exact(std::mem::size_of::<u16>())
                    .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
                    .collect_vec()
                    .try_into()
                    // we checked this beforehand in check_valid
                    .expect("Invalid iris code share length"),
            };
            let mask_share2 = GaloisRingTrimmedMaskCodeShare {
                id:    self.my_party_id + 1,
                coefs: reshare2
                    .mask_share
                    .chunks_exact(std::mem::size_of::<u16>())
                    .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
                    // we checked this beforehand in check_valid
                    .collect_vec()
                    .try_into()
                    .expect("Invalid mask share length"),
            };

            // add them together
            code_share1
                .coefs
                .iter_mut()
                .zip(code_share2.coefs.iter())
                .for_each(|(x, y)| {
                    *x = x.wrapping_add(*y);
                });
            mask_share1
                .coefs
                .iter_mut()
                .zip(mask_share2.coefs.iter())
                .for_each(|(x, y)| {
                    *x = x.wrapping_add(*y);
                });

            code.push(code_share1);
            mask.push(mask_share1);
        }

        Ok(RecombinedIrisCodeBatch {
            range_start_inclusive: request1.id_range_start_inclusive,
            range_end_exclusive:   request1.id_range_end_non_inclusive,
            iris_codes:            code,
            masks:                 mask,
        })
    }

    pub fn try_handle_batch(
        &mut self,
    ) -> Result<Option<RecombinedIrisCodeBatch>, IrisCodeReShareError> {
        if self.sender_1_buffer.is_empty() || self.sender_2_buffer.is_empty() {
            return Ok(None);
        }

        let sender_1_batch = self.sender_1_buffer.pop_front().unwrap();
        let sender_2_batch = self.sender_2_buffer.pop_front().unwrap();

        self.check_requests_matching(&sender_1_batch, &sender_2_batch)?;

        let reshare = self.reshare_code_batch(sender_1_batch, sender_2_batch)?;

        Ok(Some(reshare))
    }
}

/// A batch of recombined iris codes, produced by resharing iris codes from two
/// other parties. This should be inserted into the database.
pub struct RecombinedIrisCodeBatch {
    range_start_inclusive: u64,
    range_end_exclusive:   u64,
    iris_codes:            Vec<GaloisRingIrisCodeShare>,
    masks:                 Vec<GaloisRingTrimmedMaskCodeShare>,
}

#[cfg(test)]
mod tests {
    use super::IrisCodeReshareSenderHelper;
    use crate::reshare::IrisCodeReshareReceiverHelper;
    use iris_mpc_common::{
        galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
        iris_db::db::IrisDB,
    };
    use itertools::MultiUnzip;
    use rand::thread_rng;

    #[test]
    fn test_basic_resharing() {
        const DB_SIZE: usize = 100;

        let db = IrisDB::new_random_rng(DB_SIZE, &mut thread_rng());

        let (party0_db, party1_db, party2_db): (Vec<_>, Vec<_>, Vec<_>) = db
            .db
            .iter()
            .map(|x| {
                let [share0, share1, share2] =
                    GaloisRingIrisCodeShare::encode_iris_code(&x.code, &x.mask, &mut thread_rng());
                let [mask0, mask1, mask2] =
                    GaloisRingIrisCodeShare::encode_mask_code(&x.mask, &mut thread_rng());

                (
                    (share0, GaloisRingTrimmedMaskCodeShare::from(mask0)),
                    (share1, GaloisRingTrimmedMaskCodeShare::from(mask1)),
                    (share2, GaloisRingTrimmedMaskCodeShare::from(mask2)),
                )
            })
            .multiunzip();

        let mut reshare_helper_0_1_2 = IrisCodeReshareSenderHelper::new(0, 1, 2, [0; 32]);
        let mut reshare_helper_1_0_2 = IrisCodeReshareSenderHelper::new(1, 0, 2, [0; 32]);
        let mut reshare_helper_2 = IrisCodeReshareReceiverHelper::new(2, 0, 1, 100);

        reshare_helper_0_1_2.start_reshare_batch(0, DB_SIZE as u64);
        for (idx, (code, mask)) in party0_db.iter().enumerate() {
            reshare_helper_0_1_2.add_reshare_iris_to_batch(idx as u64, code.clone(), mask.clone());
        }
        let reshare_request_0_1_2 = reshare_helper_0_1_2.finalize_reshare_batch();

        reshare_helper_1_0_2.start_reshare_batch(0, DB_SIZE as u64);
        for (idx, (code, mask)) in party1_db.iter().enumerate() {
            reshare_helper_1_0_2.add_reshare_iris_to_batch(idx as u64, code.clone(), mask.clone());
        }
        let reshare_request_1_0_2 = reshare_helper_1_0_2.finalize_reshare_batch();

        reshare_helper_2
            .add_request_batch(reshare_request_0_1_2)
            .unwrap();
        reshare_helper_2
            .add_request_batch(reshare_request_1_0_2)
            .unwrap();

        let reshare_batch = reshare_helper_2.try_handle_batch().unwrap().unwrap();

        for (idx, (code, mask)) in party2_db.iter().enumerate() {
            assert_eq!(code, &reshare_batch.iris_codes[idx]);
            assert_eq!(mask, &reshare_batch.masks[idx]);
        }
    }
}
