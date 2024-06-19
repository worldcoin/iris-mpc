use super::packets::MaskShareMessage;
use crate::{
    setup::id::PartyID,
    upgrade::{packets::TwoToThreeIrisCodeMessage, NewIrisShareSink},
};
use eyre::{bail, Error};
use itertools::izip;

#[derive(Clone)]
pub struct IrisCodeUpgrader<S> {
    party_id:  PartyID,
    iris_sink: S,
}

impl<S: NewIrisShareSink> IrisCodeUpgrader<S> {
    /// Creates a new IrisCodeUpgrader with the given party id.
    pub fn new(party_id: PartyID, iris_sink: S) -> Self {
        Self {
            party_id,
            iris_sink,
        }
    }

    /// Finalizes the upgrade protocol.
    /// Takes 2 [TwoToThreeIrisCodeMessage] from the 3 parties and combines them
    /// into a final shamir share. Also takes a [ShamirSharesMessage] with
    /// the masks. Both the new shared code and mask are then stored in the
    /// [NewIrisShareSink].
    pub fn finalize(
        &self,
        iris_code_share_0: TwoToThreeIrisCodeMessage,
        iris_code_share_1: TwoToThreeIrisCodeMessage,
        mask_share: MaskShareMessage,
    ) -> Result<(), Error> {
        // todo: sanity checks
        let id = iris_code_share_0.id;
        if id != iris_code_share_1.id || id != mask_share.id {
            bail!("received ids do not match");
        }
        let from0 = iris_code_share_0.from;
        let from1 = iris_code_share_1.from;
        if !([0, 1].contains(&from0) && [0, 1].contains(&from1)) || from0 == from1 {
            bail!(
                "received messages from invalid senders: {} and {}",
                from0,
                from1
            );
        }
        if iris_code_share_0.party_id != self.party_id as u8
            || iris_code_share_1.party_id != self.party_id as u8
            || mask_share.party_id != self.party_id as u8
        {
            bail!("received messages for invalid party");
        }
        let mut result = iris_code_share_0.data;
        let part2 = iris_code_share_1.data;
        let mask = mask_share.data;
        for (a, b) in izip!(result.iter_mut(), part2.iter()) {
            *a = a.wrapping_add(*b);
        }

        self.iris_sink.store_code_share(id, &result)?;
        self.iris_sink.store_mask_share(id, &mask)?;
        Ok(())
    }
}
