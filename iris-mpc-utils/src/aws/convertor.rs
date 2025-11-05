use uuid::Uuid;

use iris_mpc::client::iris_data::IrisCodePartyShares;
use iris_mpc_common::helpers::smpc_request::IrisCodeSharesJSON;

use crate::types::{NetGaloisRingIrisCodeShare, NetIrisCodeSharesJSON};

fn to_iris_code_shares_json(
    left_shared_code: NetGaloisRingIrisCodeShare,
    left_shared_mask: NetGaloisRingIrisCodeShare,
    right_shared_code: NetGaloisRingIrisCodeShare,
    right_shared_mask: NetGaloisRingIrisCodeShare,
) -> NetIrisCodeSharesJSON {
    [
        IrisCodeSharesJSON {
            iris_version: "1.0".to_string(),
            iris_shares_version: "1.3".to_string(),
            right_iris_code_shares: right_shared_code[0].to_base64(),
            right_mask_code_shares: right_shared_mask[0].to_base64(),
            left_iris_code_shares: left_shared_code[0].to_base64(),
            left_mask_code_shares: left_shared_mask[0].to_base64(),
        },
        IrisCodeSharesJSON {
            iris_version: "1.0".to_string(),
            iris_shares_version: "1.3".to_string(),
            right_iris_code_shares: right_shared_code[1].to_base64(),
            right_mask_code_shares: right_shared_mask[1].to_base64(),
            left_iris_code_shares: left_shared_code[1].to_base64(),
            left_mask_code_shares: left_shared_mask[1].to_base64(),
        },
        IrisCodeSharesJSON {
            iris_version: "1.0".to_string(),
            iris_shares_version: "1.3".to_string(),
            right_iris_code_shares: right_shared_code[2].to_base64(),
            right_mask_code_shares: right_shared_mask[2].to_base64(),
            left_iris_code_shares: left_shared_code[2].to_base64(),
            left_mask_code_shares: left_shared_mask[2].to_base64(),
        },
    ]
}

fn to_iris_code_party_shares(
    left_shared_code: NetGaloisRingIrisCodeShare,
    left_shared_mask: NetGaloisRingIrisCodeShare,
    right_shared_code: NetGaloisRingIrisCodeShare,
    right_shared_mask: NetGaloisRingIrisCodeShare,
) -> IrisCodePartyShares {
    IrisCodePartyShares::new(
        Uuid::new_v4().to_string(),
        to_iris_code_shares_json(
            left_shared_code,
            left_shared_mask,
            right_shared_code,
            right_shared_mask,
        )
        .to_vec(),
    )
}
