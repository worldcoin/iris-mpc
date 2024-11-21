use iris_mpc_common::{IRIS_CODE_LENGTH, MASK_CODE_LENGTH};
use iris_mpc_reshare::IrisCodeReShare;
use prost::Message;

// this is generated code so we skip linting it
#[rustfmt::skip]
#[allow(clippy::all)]
pub mod iris_mpc_reshare;

pub fn get_size_of_reshare_iris_code_share_batch(batch_size: usize) -> usize {
    let dummy = iris_mpc_reshare::IrisCodeReShareRequest {
        sender_id:                  0,
        other_id:                   1,
        receiver_id:                2,
        id_range_start_inclusive:   0,
        id_range_end_non_inclusive: batch_size as i64,
        iris_code_re_shares:        vec![
            IrisCodeReShare {
                left_iris_code_share:  vec![1u8; IRIS_CODE_LENGTH * size_of::<u16>()],
                left_mask_share:       vec![2u8; MASK_CODE_LENGTH * size_of::<u16>()],
                right_iris_code_share: vec![3u8; IRIS_CODE_LENGTH * size_of::<u16>()],
                right_mask_share:      vec![4u8; MASK_CODE_LENGTH * size_of::<u16>()],
            };
            batch_size
        ],
    };

    dummy.encoded_len()
}
