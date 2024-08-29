use eyre::{bail, Error, Result};
use futures::Stream;
use iris_mpc_common::{id::PartyID, IRIS_CODE_LENGTH, MASK_CODE_LENGTH};
use itertools::izip;
use mpc_uniqueness_check::{bits::Bits, distance::EncodedBits};
use packets::{MaskShareMessage, TwoToThreeIrisCodeMessage};
use std::{
    fs::File,
    future::Future,
    io::{BufWriter, Write},
};

pub mod config;
pub mod db;
pub mod packets;

pub trait OldIrisShareSource {
    /// loads an 1-of-2 additive share of the iris code with id `share_id`
    fn load_code_share(&self, share_id: u64) -> impl Future<Output = eyre::Result<EncodedBits>>;
    /// loads the masks of the iris code with id `share_id`
    fn load_mask(&self, share_id: u64) -> impl Future<Output = eyre::Result<Bits>>;

    /// loads the masks of the iris code with id `share_id`
    fn stream_shares(
        &self,
        share_id_range: std::ops::Range<u64>,
    ) -> eyre::Result<impl Stream<Item = eyre::Result<(u64, EncodedBits)>> + Sized>;

    /// loads the masks of the iris code with id `share_id`
    fn stream_masks(
        &self,
        share_id_range: std::ops::Range<u64>,
    ) -> eyre::Result<impl Stream<Item = eyre::Result<(u64, Bits)>> + Sized>;
}

#[allow(async_fn_in_trait)]
pub trait NewIrisShareSink {
    async fn store_code_mask_share(
        &self,
        share_id: u64,
        code_share: &[u16; IRIS_CODE_LENGTH],
        mask_share: &[u16; MASK_CODE_LENGTH],
    ) -> eyre::Result<()>;
}

#[derive(Debug, Clone)]
pub struct IrisShareTestFileSink {
    path: std::path::PathBuf,
}

impl IrisShareTestFileSink {
    pub fn new(folder: std::path::PathBuf) -> Result<Self> {
        if !folder.is_dir() {
            bail!("{} is not a directory", folder.display());
        }
        Ok(Self { path: folder })
    }
}

impl NewIrisShareSink for IrisShareTestFileSink {
    async fn store_code_mask_share(
        &self,
        share_id: u64,
        code_share: &[u16; IRIS_CODE_LENGTH],
        mask_share: &[u16; MASK_CODE_LENGTH],
    ) -> eyre::Result<()> {
        let mut file = BufWriter::new(File::create(
            self.path.join(format!("code_share_{}", share_id)),
        )?);
        for s in code_share {
            writeln!(file, "{}", s)?;
        }
        file.flush()?;
        let mut file = BufWriter::new(File::create(
            self.path.join(format!("mask_share_{}", share_id)),
        )?);
        for s in mask_share {
            writeln!(file, "{}", s)?;
        }
        file.flush()?;
        Ok(())
    }
}

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
    /// into a final shamir share. Also takes a [MaskShareMessage] with
    /// the masks. Both the new shared code and mask are then stored in the
    /// [NewIrisShareSink].
    pub async fn finalize(
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

        self.iris_sink
            .store_code_mask_share(id, &result, &mask)
            .await?;
        Ok(())
    }
}
