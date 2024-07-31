use aws_sdk_s3::{config::Region as S3Region, Client as S3Client, Error as S3Error};
use aws_sdk_secretsmanager::{Client as SecretsManagerClient, Error as SecretsManagerError};
use base64::{engine::general_purpose::STANDARD, Engine};
use clap::Parser;
use rand::{rngs::StdRng, Rng, SeedableRng};
use sodiumoxide::crypto::box_::{curve25519xsalsa20poly1305, PublicKey, SecretKey, Seed};

const PUBLIC_KEY_S3_KEY_NAME_PREFIX: &str = "public-key";
const PRIVATE_KEY_SM_KEY_NAME_PREFIX: &str = "private-key";
const PRIVATE_KEY_KMS_KEY_ID: &str = "private-key-kms-key-id";
const REGION: &str = "eu-north-1";

#[derive(Debug, Parser)]
struct Opt {
    #[arg(short, long, env)]
    public_key_s3_bucket_name: String,

    #[arg(short, long, env)]
    node_id: u16,

    #[arg(short, long, env)]
    rng_seed: Option<u64>,

    #[arg(short, long, env)]
    dry_run: Option<bool>,
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();

    let Opt {
        public_key_s3_bucket_name,
        node_id,
        rng_seed,
        dry_run,
    } = Opt::parse();

    let mut rng = if let Some(rng_seed) = rng_seed {
        StdRng::seed_from_u64(rng_seed)
    } else {
        StdRng::from_entropy()
    };

    let mut seedbuf = [0u8; 32];
    rng.fill(&mut seedbuf);
    let pk_seed = Seed(seedbuf);

    let region_provider = S3Region::new(REGION);
    let shared_config = aws_config::from_env().region(region_provider).load().await;

    let s3_client = S3Client::new(&shared_config);
    let sm_client = SecretsManagerClient::new(&shared_config);

    let (public_key, private_key) = generate_key_pairs(pk_seed);
    let pub_key_str = STANDARD.encode(public_key);
    let priv_key_str = STANDARD.encode(private_key.clone());
    let bucket_key_name = format!("{}-{}", PUBLIC_KEY_S3_KEY_NAME_PREFIX, node_id);
    let secret_key_name: String = format!("{}-{}", PRIVATE_KEY_SM_KEY_NAME_PREFIX, node_id);

    if dry_run.unwrap_or(false) {
        println!("Dry run enabled, skipping upload of public key to S3");
        println!("Public key: {}", pub_key_str);
        println!("Private key: {}", priv_key_str);

        let user_pubkey = STANDARD.decode(pub_key_str.as_bytes()).unwrap();
        let decoded_pub_key = PublicKey::from_slice(&user_pubkey).unwrap();

        assert_eq!(public_key, decoded_pub_key);

        let user_privkey = STANDARD.decode(priv_key_str.as_bytes()).unwrap();
        let decoded_priv_key = SecretKey::from_slice(&user_privkey).unwrap();

        assert_eq!(private_key, decoded_priv_key);

        return Ok(());
    }
    upload_public_key_to_s3(
        &s3_client,
        public_key_s3_bucket_name.as_str(),
        bucket_key_name.as_str(),
        pub_key_str.as_str(),
    )
    .await?;

    upload_private_key_to_asm(&sm_client, secret_key_name.as_str(), priv_key_str.as_str()).await?;

    println!("File uploaded successfully!");

    Ok(())
}

async fn upload_private_key_to_asm(
    client: &SecretsManagerClient,
    secret_name: &str,
    content: &str,
) -> Result<(), SecretsManagerError> {
    client
        .create_secret()
        .secret_string(content)
        .name(secret_name)
        .kms_key_id(PRIVATE_KEY_KMS_KEY_ID)
        .send()
        .await?;
    Ok(())
}

async fn upload_public_key_to_s3(
    client: &S3Client,
    bucket: &str,
    key: &str,
    content: &str,
) -> Result<(), S3Error> {
    client
        .put_object()
        .bucket(bucket)
        .key(key)
        .body(content.to_string().into_bytes().into())
        .send()
        .await?;

    Ok(())
}

fn generate_key_pairs(seed: Seed) -> (PublicKey, SecretKey) {
    // Generate an ephemeral secret (private key)
    let (public_key, private_key) = curve25519xsalsa20poly1305::keypair_from_seed(&seed);

    (public_key, private_key)
}

// tests
#[cfg(test)]
mod test {
    use super::*;

    pub fn get_public_key(user_pub_key: &str) -> PublicKey {
        let user_pubkey = STANDARD.decode(user_pub_key.as_bytes()).unwrap();
        match PublicKey::from_slice(&user_pubkey) {
            Some(key) => key,
            None => panic!("Invalid public key"),
        }
    }

    #[test]
    fn test_encode_pk_to_pem() {
        let (public_key, _) = generate_key_pairs(Seed([0u8; 32]));
        let pub_key_str = STANDARD.encode(public_key);
        let decoded_pub_key = get_public_key(&pub_key_str);
        assert_eq!(public_key, decoded_pub_key);
    }
}
