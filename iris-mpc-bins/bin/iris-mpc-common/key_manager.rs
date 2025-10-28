use aws_config::SdkConfig;
use aws_sdk_s3::{
    config::Region as S3Region, operation::put_object::PutObjectOutput, Client as S3Client,
    Error as S3Error,
};
use aws_sdk_secretsmanager::{
    operation::{get_secret_value::GetSecretValueOutput, put_secret_value::PutSecretValueOutput},
    Client as SecretsManagerClient, Error as SecretsManagerError,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use clap::{Parser, Subcommand};
use eyre::Result;
use rand::{thread_rng, Rng};
use reqwest::Client;
use sodiumoxide::crypto::box_::{curve25519xsalsa20poly1305, PublicKey, SecretKey, Seed};

const PUBLIC_KEY_S3_BUCKET_NAME: &str = "wf-smpcv2-stage-public-keys";
const PUBLIC_KEY_S3_KEY_NAME_PREFIX: &str = "public-key";

#[derive(Debug, Parser)] // requires `derive` feature
#[command(name = "key-manager")]
#[command(about = "Key manager CLI", long_about = None)]
struct KeyManagerCli {
    #[command(subcommand)]
    command: Commands,

    #[arg(
        short, long, env, value_parser = clap::builder::PossibleValuesParser::new(& ["0", "1", "2"])
    )]
    node_id: String,

    #[arg(long, env, default_value = "stage")]
    env: String,

    #[arg(short, long, env, default_value = "eu-north-1")]
    region: String,

    #[arg(long, env, default_value = None)]
    endpoint_url: Option<String>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Rotate private and public keys
    #[command(arg_required_else_help = true)]
    Rotate {
        #[arg(short, long, env)]
        dry_run: Option<bool>,

        #[arg(short, long, env)]
        public_key_bucket_name: Option<String>,
    },
    /// Validate private key in key manager against public keys (either provided
    /// or in s3)
    Validate {
        // AWSCURRENT or AWSPREVIOUS or a specific version
        #[arg(
            short, long, env, value_parser = clap::builder::PossibleValuesParser::new(& ["AWSCURRENT", "AWSPREVIOUS"])
        )]
        version_stage: String,

        #[arg(short, long, env)]
        b64_pub_key: Option<String>,

        #[arg(short, long, env)]
        public_key_bucket_name: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = KeyManagerCli::parse();
    let region = args.region;

    let region_provider = S3Region::new(region.clone());
    let shared_config = aws_config::from_env().region(region_provider).load().await;

    let bucket_key_name = format!("{}-{}", PUBLIC_KEY_S3_KEY_NAME_PREFIX, args.node_id);
    let private_key_secret_id: String =
        format!("{}/iris-mpc/ecdh-private-key-{}", args.env, args.node_id);

    match args.command {
        Commands::Rotate {
            dry_run,
            public_key_bucket_name,
        } => {
            rotate_keys(
                &shared_config,
                &bucket_key_name,
                &private_key_secret_id,
                dry_run,
                public_key_bucket_name,
                args.endpoint_url,
            )
            .await?;
        }
        Commands::Validate {
            version_stage,
            b64_pub_key,
            public_key_bucket_name,
        } => {
            validate_keys(
                &shared_config,
                &private_key_secret_id,
                &version_stage,
                b64_pub_key,
                &bucket_key_name,
                public_key_bucket_name,
                region.clone(),
                args.endpoint_url,
            )
            .await?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn validate_keys(
    sdk_config: &SdkConfig,
    secret_id: &str,
    version_stage: &str,
    b64_pub_key: Option<String>,
    bucket_key_name: &str,
    public_key_bucket_name: Option<String>,
    region: String,
    endpoint_url: Option<String>,
) -> Result<()> {
    let mut sm_config_builder = aws_sdk_secretsmanager::config::Builder::from(sdk_config);

    if let Some(endpoint_url) = endpoint_url.as_ref() {
        sm_config_builder = sm_config_builder.endpoint_url(endpoint_url);
    }

    let sm_client = SecretsManagerClient::from_conf(sm_config_builder.build());

    let bucket_name = if let Some(bucket_name) = public_key_bucket_name {
        bucket_name
    } else {
        PUBLIC_KEY_S3_BUCKET_NAME.to_string()
    };
    // Parse user-provided public key, if present
    let pub_key = if let Some(b64_pub_key) = b64_pub_key {
        let user_pubkey = STANDARD.decode(b64_pub_key.as_bytes()).unwrap();
        match PublicKey::from_slice(&user_pubkey) {
            Some(key) => key,
            None => panic!("Invalid public key"),
        }
    } else {
        // Otherwise, get the latest one from S3 using HTTPS
        let user_pubkey_string =
            download_key_from_s3(bucket_name.as_str(), bucket_key_name, region.clone()).await?;
        let user_pubkey = STANDARD.decode(user_pubkey_string.as_bytes()).unwrap();
        match PublicKey::from_slice(&user_pubkey) {
            Some(key) => key,
            None => panic!("Invalid public key"),
        }
    };

    let private_key = download_key_from_asm(&sm_client, secret_id, version_stage).await?;
    let data = private_key.secret_string.unwrap();
    let user_privkey = STANDARD.decode(data.as_bytes()).unwrap();
    let decoded_priv_key = SecretKey::from_slice(&user_privkey).unwrap();

    assert_eq!(decoded_priv_key.public_key(), pub_key);
    Ok(())
}

async fn rotate_keys(
    sdk_config: &SdkConfig,
    bucket_key_name: &str,
    private_key_secret_id: &str,
    dry_run: Option<bool>,
    public_key_bucket_name: Option<String>,
    endpoint_url: Option<String>,
) -> Result<()> {
    let mut rng = thread_rng();

    let bucket_name = if let Some(bucket_name) = public_key_bucket_name {
        bucket_name
    } else {
        PUBLIC_KEY_S3_BUCKET_NAME.to_string()
    };

    let mut seedbuf = [0u8; 32];
    rng.fill(&mut seedbuf);
    let pk_seed = Seed(seedbuf);

    let mut s3_config_builder = aws_sdk_s3::config::Builder::from(sdk_config);
    let mut sm_config_builder = aws_sdk_secretsmanager::config::Builder::from(sdk_config);

    if let Some(endpoint_url) = endpoint_url.as_ref() {
        s3_config_builder = s3_config_builder.endpoint_url(endpoint_url);
        s3_config_builder = s3_config_builder.force_path_style(true);
        sm_config_builder = sm_config_builder.endpoint_url(endpoint_url);
    }

    let s3_client = S3Client::from_conf(s3_config_builder.build());
    let sm_client = SecretsManagerClient::from_conf(sm_config_builder.build());

    let (public_key, private_key) = generate_key_pairs(pk_seed);
    let pub_key_str = STANDARD.encode(public_key);
    let priv_key_str = STANDARD.encode(private_key.clone());

    if dry_run.unwrap_or(false) {
        println!("Dry run enabled, skipping upload of public key to S3");
        println!("Public key: {}", pub_key_str);

        let user_pubkey = STANDARD.decode(pub_key_str.as_bytes()).unwrap();
        let decoded_pub_key = PublicKey::from_slice(&user_pubkey).unwrap();

        assert_eq!(public_key, decoded_pub_key);

        let user_privkey = STANDARD.decode(priv_key_str.as_bytes()).unwrap();
        let decoded_priv_key = SecretKey::from_slice(&user_privkey).unwrap();

        assert_eq!(private_key, decoded_priv_key);

        return Ok(());
    }
    match upload_public_key_to_s3(
        &s3_client,
        bucket_name.as_str(),
        bucket_key_name,
        pub_key_str.as_str(),
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

    match upload_private_key_to_asm(&sm_client, private_key_secret_id, priv_key_str.as_str()).await
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

async fn download_key_from_s3(
    bucket: &str,
    key: &str,
    region: String,
) -> Result<String, reqwest::Error> {
    print!("Downloading key from S3 bucket: {} key: {}", bucket, key);
    let s3_url = format!("https://{}.s3.{}.amazonaws.com/{}", bucket, region, key);
    let client = Client::new();
    let response = client.get(&s3_url).send().await?.text().await?;
    Ok(response)
}

async fn download_key_from_asm(
    client: &SecretsManagerClient,
    secret_id: &str,
    version_stage: &str,
) -> Result<GetSecretValueOutput, SecretsManagerError> {
    Ok(client
        .get_secret_value()
        .secret_id(secret_id)
        .version_stage(version_stage)
        .send()
        .await?)
}

async fn upload_private_key_to_asm(
    client: &SecretsManagerClient,
    secret_id: &str,
    content: &str,
) -> Result<PutSecretValueOutput, SecretsManagerError> {
    Ok(client
        .put_secret_value()
        .secret_string(content)
        .secret_id(secret_id)
        .send()
        .await?)
}

async fn upload_public_key_to_s3(
    client: &S3Client,
    bucket: &str,
    key: &str,
    content: &str,
) -> Result<PutObjectOutput, S3Error> {
    Ok(client
        .put_object()
        .bucket(bucket)
        .key(key)
        .body(content.to_string().into_bytes().into())
        .send()
        .await?)
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
    use sodiumoxide::crypto::sealedbox;
    use std::{fs::File, io::Read};

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

    #[test]
    fn test_encode_and_decode_shares() {
        let (server_public_key, server_private_key) = generate_key_pairs(Seed([0u8; 32]));

        let iris_code_file = "./data/iris_codes.json";
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
