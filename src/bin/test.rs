use gpu_iris_mpc::helpers::kms_dh::{derive_shared_secret, derive_shared_secret2};

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let shared_secret_one = derive_shared_secret(
        "077788e2-9eeb-4044-859b-34496cfd500b",
        "896353dc-5ea5-42d4-9e4e-f65dd8169dee",
    )
    .await?;
    println!("{:?}", shared_secret_one);

    let shared_secret_two = derive_shared_secret2(
        "077788e2-9eeb-4044-859b-34496cfd500b",
        "896353dc-5ea5-42d4-9e4e-f65dd8169dee",
    )
    .await?;
    println!("{:?}", shared_secret_two);

    Ok(())
}
