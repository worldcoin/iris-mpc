use aws_config::SdkConfig;
use aws_sdk_s3::{config::Region as S3Region, Client as S3Client};
use aws_sdk_secretsmanager::{Client as SecretsManagerClient, Error as SecretsManagerError};
use base64::{engine::general_purpose::STANDARD, Engine};
use clap::{Parser, Subcommand};
use digest::Digest;
use iris_mpc_common::helpers::aws::{
    create_secret_string_with_asm, download_key_from_asm, download_key_from_s3,
    get_secret_string_from_asm, upload_private_key_to_asm, upload_public_key_to_s3,
};
use sha2::Sha256;
use sodiumoxide::crypto::box_::{PublicKey, SecretKey};

const PUBLIC_KEY_S3_BUCKET_NAME: &str = "wf-smpcv2-stage-public-keys";
const PUBLIC_KEY_S3_KEY_NAME_PREFIX: &str = "public-key";
const REGION: &str = "eu-north-1";

/// A CLI tool for managing private and public keys
#[derive(Debug, Parser)]
#[command(name = "key-manager")]
#[command(about = "Key manager CLI", long_about = None)]
struct KeyManagerCli {
    #[command(subcommand)]
    command: Commands,

    #[arg(
        short, long, env, value_parser = clap::builder::PossibleValuesParser::new(& ["0", "1", "2"])
    )]
    node_id: String,

    #[arg(short, long, env, default_value = "stage")]
    env: String,
}

#[derive(Debug, Subcommand)]
enum Commands {
    #[command(about = "Rotate private and public keys")]
    Rotate {
        #[arg(short, long)]
        public_key_bucket_name: Option<String>,
    },
    #[command(
        about = "Validate private key in key manager against public keys (either provided or in \
                 s3)"
    )]
    Validate {
        #[arg(
            short, long, env, value_parser = clap::builder::PossibleValuesParser::new(& ["AWSCURRENT", "AWSPREVIOUS"])
        )]
        version_stage: String,

        #[arg(short, long, env)]
        b64_pub_key: Option<String>,
    },
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();

    let args = KeyManagerCli::parse();

    let region_provider = S3Region::new(REGION);
    let shared_config = aws_config::from_env().region(region_provider).load().await;

    let bucket_key_name = format!("{}-{}", PUBLIC_KEY_S3_KEY_NAME_PREFIX, args.node_id);
    let private_key_secret_id: String =
        format!("{}/iris-mpc/ecdh-private-key-{}", args.env, args.node_id);

    let private_key_seed_secret_id: String = format!(
        "{}/iris-mpc/ecdh-private-key-seed-{}",
        args.env, args.node_id
    );

    match args.command {
        Commands::Rotate {
            public_key_bucket_name,
        } => {
            rotate_keys(
                &shared_config,
                &bucket_key_name,
                &private_key_secret_id,
                &private_key_seed_secret_id,
                public_key_bucket_name,
            )
            .await?;
        }
        Commands::Validate {
            version_stage,
            b64_pub_key,
        } => {
            validate_keys(
                &shared_config,
                &private_key_secret_id,
                &version_stage,
                b64_pub_key,
                &args.env,
                &args.node_id,
            )
            .await?;
        }
    }
    Ok(())
}

async fn rotate_keys(
    sdk_config: &SdkConfig,
    bucket_key_name: &str,
    private_key_secret_id: &str,
    private_key_seed_secret_id: &str,
    public_key_bucket_name: Option<String>,
) -> eyre::Result<()> {
    let bucket_name = if let Some(bucket_name) = public_key_bucket_name {
        bucket_name
    } else {
        PUBLIC_KEY_S3_BUCKET_NAME.to_string()
    };

    let s3_client = S3Client::new(sdk_config);
    let sm_client = SecretsManagerClient::new(sdk_config);

    let secret_json = get_or_create_secret_string(&sm_client, private_key_seed_secret_id).await?;
    // Extract the secret seed from the stored secret JSON
    let secret_json: serde_json::Value = serde_json::from_str(&secret_json)?;
    let seed_str = secret_json["seed"]
        .as_str()
        .ok_or_else(|| eyre::eyre!("Missing seed in secret"))?;
    let secret_seed = hash_secret_to_seed(seed_str);

    let (public_key, private_key) = generate_key_pairs_from_secret_seed(secret_seed);
    let public_key_str = STANDARD.encode(public_key);
    let private_key_str = STANDARD.encode(private_key.clone());

    match upload_public_key_to_s3(
        &s3_client,
        bucket_name.as_str(),
        bucket_key_name,
        public_key_str.as_str(),
    )
    .await
    {
        Ok(output) => {
            println!("Bucket: {}", bucket_name);
            println!("Key: {}", bucket_key_name);
            println!("ETag: {}", output.e_tag.unwrap());
        }
        Err(e) => {
            eprintln!("Error uploading public key to S3: {:?}", e);
            return Err(eyre::eyre!("Error uploading public key to S3"));
        }
    }

    println!("private key secret id: {}", private_key_secret_id);
    match upload_private_key_to_asm(&sm_client, private_key_secret_id, private_key_str.as_str())
        .await
    {
        Ok(output) => {
            println!("Secret ARN: {}", output.arn.unwrap());
            println!("Secret Name: {}", output.name.unwrap());
            println!("Version ID: {}", output.version_id.unwrap());
        }
        Err(e) => {
            eprintln!("Error uploading private key to Secrets Manager: {:?}", e);
            return Err(eyre::eyre!(
                "Error uploading private key to Secrets Manager"
            ));
        }
    }

    println!("File uploaded successfully!");

    Ok(())
}

async fn validate_keys(
    sdk_config: &SdkConfig,
    secret_id: &str,
    version_stage: &str,
    b64_pub_key: Option<String>,
    env: &str,
    node_id: &str,
) -> eyre::Result<()> {
    let sm_client = SecretsManagerClient::new(sdk_config);

    // Parse user-provided public key, if present
    let pub_key = if let Some(b64_pub_key) = b64_pub_key {
        let user_pubkey = STANDARD.decode(b64_pub_key.as_bytes())?;
        match PublicKey::from_slice(&user_pubkey) {
            Some(key) => key,
            None => panic!("Invalid public key"),
        }
    } else {
        // Otherwise, get the latest one from S3 using HTTPS
        let user_pubkey_string = download_key_from_s3(env, node_id).await?;

        println!("user_pubkey_string: {}", user_pubkey_string);
        let user_pubkey = STANDARD.decode(user_pubkey_string.as_bytes())?;
        match PublicKey::from_slice(&user_pubkey) {
            Some(key) => key,
            None => panic!("Invalid public key"),
        }
    };

    let private_key = download_key_from_asm(&sm_client, secret_id, version_stage).await?;
    let data = private_key.secret_string.unwrap();
    let user_private_key = STANDARD.decode(data.as_bytes())?;
    let decoded_private_key = SecretKey::from_slice(&user_private_key).unwrap();

    assert_eq!(decoded_private_key.public_key(), pub_key);
    Ok(())
}

async fn get_or_create_secret_string(
    sm_client: &SecretsManagerClient,
    private_key_seed_secret_id: &str,
) -> Result<String, SecretsManagerError> {
    // Step 1: Try to retrieve the secret from AWS Secrets Manager
    let secret_string =
        match get_secret_string_from_asm(sm_client, private_key_seed_secret_id).await {
            Ok(output) => {
                if let Some(secret_string) = output.secret_string() {
                    Ok(secret_string.to_string())
                } else {
                    Err("Secret string not found".to_string())
                }
            }
            Err(e) => {
                eprintln!("Error getting secret from SM secret key seed will be created");
                Err(e.to_string())
            }
        };

    if let Ok(secret_string) = secret_string {
        return Ok(secret_string);
    }

    // Step 2: If the secret does not exist, create it
    let new_secret_string =
        create_secret_string_with_asm(sm_client, private_key_seed_secret_id).await?;
    Ok(new_secret_string)
}

// Step 3: Hash the secret to a 32-byte seed
fn hash_secret_to_seed(secret: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(secret);
    let result = hasher.finalize();
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&result[..32]); // Take the first 32 bytes
    seed
}

// Step 4: Generate key pairs using the secret seed
fn generate_key_pairs_from_secret_seed(secret_seed: [u8; 32]) -> (PublicKey, SecretKey) {
    let private_key = SecretKey::from_slice(&secret_seed).unwrap();
    let public_key = private_key.public_key();
    (public_key, private_key)
}

// tests
#[cfg(test)]
mod test {
    use super::*;
    use rand::{distributions::Alphanumeric, thread_rng, Rng};
    use sodiumoxide::crypto::sealedbox;
    use std::{fs::File, io::Read};

    pub fn get_public_key(user_pub_key: &str) -> PublicKey {
        let user_pubkey = STANDARD.decode(user_pub_key.as_bytes()).unwrap();
        match PublicKey::from_slice(&user_pubkey) {
            Some(key) => key,
            None => panic!("Invalid public key"),
        }
    }

    fn generate_large_secret() -> String {
        let mut rng = thread_rng();
        (0..128)
            .map(|_| rng.sample(Alphanumeric) as char)
            .collect::<String>()
    }

    #[test]
    fn test_encode_pk_to_pem() {
        let secret_seed = generate_large_secret();
        let seed = hash_secret_to_seed(&secret_seed);
        let (public_key, _) = generate_key_pairs_from_secret_seed(seed);
        let pub_key_str = STANDARD.encode(public_key);
        let decoded_pub_key = get_public_key(&pub_key_str);
        assert_eq!(public_key, decoded_pub_key);
    }

    #[test]
    fn test_encode_and_decode_shares() {
        let secret_seed = generate_large_secret();
        let seed = hash_secret_to_seed(&secret_seed);
        let (server_public_key, server_private_key) = generate_key_pairs_from_secret_seed(seed);

        let iris_code_file = "./src/bin/data/iris_codes.json";
        let mut file = File::open(iris_code_file).expect("Unable to open file");
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("Unable to read file");

        let client_iris_code_plaintext = STANDARD.encode(contents);
        let ciphertext = sealedbox::seal(client_iris_code_plaintext.as_bytes(), &server_public_key);

        let server_iris_code_plaintext =
            sealedbox::open(&ciphertext, &server_public_key, &server_private_key).unwrap();

        assert_eq!(
            client_iris_code_plaintext.as_bytes(),
            server_iris_code_plaintext.as_slice()
        );
    }
}
