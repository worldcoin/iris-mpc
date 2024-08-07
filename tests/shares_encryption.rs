use base64::{engine::general_purpose::STANDARD, Engine};
use gpu_iris_mpc::{
    helpers::{key_pair::SharesEncryptionKeyPair, sqs::SMPCRequest},
    setup::{galois_engine::degree4::GaloisRingIrisCodeShare, iris_db::iris::IrisCode},
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use sodiumoxide::crypto::{
    box_::{curve25519xsalsa20poly1305, PublicKey, SecretKey, Seed},
    sealedbox,
};

const RNG_SEED_SERVER: u64 = 1;
const RNG_SEED_CODES: u64 = 2;
const RNG_SEED_SHARES: u64 = 3;

fn key_pair_from_seed(seed: u64) -> (PublicKey, SecretKey) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut seedbuf = [0u8; 32];
    rng.fill(&mut seedbuf);
    let pk_seed = Seed(seedbuf);
    let (public_key, private_key) = curve25519xsalsa20poly1305::keypair_from_seed(&pk_seed);
    (public_key, private_key)
}

fn generate_iris_shares() -> ([GaloisRingIrisCodeShare; 3], [GaloisRingIrisCodeShare; 3]) {
    let mut rng = StdRng::seed_from_u64(RNG_SEED_CODES);
    let template = IrisCode::random_rng(&mut rng);
    let shared_code = GaloisRingIrisCodeShare::encode_iris_code(
        &template.code,
        &template.mask,
        &mut StdRng::seed_from_u64(RNG_SEED_SHARES),
    );
    let shared_mask = GaloisRingIrisCodeShare::encode_mask_code(
        &template.mask,
        &mut StdRng::seed_from_u64(RNG_SEED_SERVER),
    );
    (shared_code, shared_mask)
}

#[tokio::test]
async fn test_share_encryption_and_decryption() -> eyre::Result<()> {
    let (shares, masks) = generate_iris_shares();

    // Encrypt shares
    for i in 0..3 {
        let (server_pk, server_sk) = key_pair_from_seed(RNG_SEED_SERVER);

        let server_pub_key_str = STANDARD.encode(server_pk);
        let server_priv_key_str = STANDARD.encode(server_sk);

        let server_shares_key_pair = SharesEncryptionKeyPair::from_b64_strings(
            server_pub_key_str.clone(),
            server_priv_key_str.clone(),
        )
        .unwrap();

        let iris_code_coefs = bytemuck::cast_slice(&shares[i].coefs);
        let mask_code_coefs = bytemuck::cast_slice(&masks[i].coefs);
        let encrypted_iris_code =
            STANDARD.encode(sealedbox::seal(iris_code_coefs, &server_shares_key_pair.pk));
        let encrypted_mask_code =
            STANDARD.encode(sealedbox::seal(mask_code_coefs, &server_shares_key_pair.pk));

        let as_smpc_request = SMPCRequest {
            request_id: format!("request_id_{}", i),
            iris_code:  encrypted_iris_code,
            mask_code:  encrypted_mask_code,
        };
        // Decrypt shares
        let decrypted_shares =
            as_smpc_request.get_iris_shares(true, server_shares_key_pair.clone());
        let decrypted_mask = as_smpc_request.get_mask_shares(true, server_shares_key_pair.clone());
        assert_eq!(shares[i].coefs, decrypted_shares);
        assert_eq!(masks[i].coefs, decrypted_mask);
    }

    Ok(())
}
