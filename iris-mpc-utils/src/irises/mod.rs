mod generator;
pub mod modifications;
pub mod shares;

pub use generator::{
    generate_iris_code_and_mask_shares, generate_iris_code_and_mask_shares_for_both_eyes,
    generate_shared_iris_locally, generate_shared_iris_locally_mirrored,
};
