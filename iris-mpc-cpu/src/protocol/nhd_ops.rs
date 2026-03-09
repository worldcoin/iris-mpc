use crate::execution::session::Session;
use crate::protocol::min_round_robin::{min_round_robin_batch_with, CrossCompareFnResult};
use crate::protocol::ops::DistancePair;
use ampc_actor_utils::protocol::nhd_ops::nhd_oblivious_cross_compare;
use ampc_secret_sharing::shares::ring48::Ring48;
pub use ampc_secret_sharing::shares::{
    bit::Bit,
    ring_impl::{RingElement, RingRandFillable},
    share::{DistanceShare, Share},
    vecshare::VecShare,
};
use eyre::Result;

pub const MATCH_THRESHOLD_RATIO: f64 = iris_mpc_common::iris_db::iris::MATCH_THRESHOLD_RATIO;

fn nhd_oblivious_cross_compare_boxed<'a>(
    session: &'a mut Session,
    pairs: &'a [DistancePair<Ring48>],
) -> CrossCompareFnResult<'a> {
    Box::pin(nhd_oblivious_cross_compare(session, pairs))
}

// Round-robin minimum distance selection for NHD.
pub(crate) async fn nhd_min_round_robin_batch(
    session: &mut Session,
    distances: &[DistanceShare<Ring48>],
    batch_size: usize,
) -> Result<Vec<DistanceShare<Ring48>>> {
    min_round_robin_batch_with(
        session,
        distances,
        batch_size,
        nhd_oblivious_cross_compare_boxed,
    )
    .await
}
