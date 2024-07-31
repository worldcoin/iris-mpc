use clap::Parser;
use sodiumoxide::crypto::box_::{PublicKey, SecretKey, Seed, curve25519xsalsa20poly1305};
use aws_sdk_s3::{Client, Error, config::Region};
use rand::{rngs::StdRng, Rng, SeedableRng};
use base64::{engine::general_purpose::STANDARD, DecodeError, Engine};



const PUBLIC_KEY_S3_KEY_NAME_PREFIX: &str = "public-key";
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


    let region_provider = Region::new(REGION);
    let shared_config = aws_config::from_env().region(region_provider).load().await;
    let client = Client::new(&shared_config);
    let bucket_key_name = format!("{}-{}", PUBLIC_KEY_S3_KEY_NAME_PREFIX, node_id);
    
    let (public_key, _) = generate_key_pairs(pk_seed);
    let pub_key_str = STANDARD.encode(public_key);

    if dry_run.unwrap_or(false) {
        println!("Dry run enabled, skipping upload of public key to S3");
        println!("Public key: {}", pub_key_str);
        return Ok(());
    }
    upload_public_key_to_s3(
        &client, 
        public_key_s3_bucket_name.as_str(), 
        bucket_key_name.as_str(), 
        pub_key_str.as_str()
    ).await?;

    println!("File uploaded successfully!");

    Ok(())
}

async fn upload_public_key_to_s3(client: &Client, bucket: &str, key: &str, content: &str) -> Result<(), Error> {
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

    #[test]
    fn test_encode_pk_to_pem() {
        let (public_key, _) = generate_key_pairs(Seed([0u8; 32]));
        let pub_key_str = STANDARD.encode(public_key);
        let decoded_pub_key = get_public_key(&pub_key_str);
        assert_eq!(public_key, decoded_pub_key);
    }

}


pub fn get_public_key(user_pub_key: &str,) -> PublicKey {
    let user_pubkey = STANDARD.decode(user_pub_key.as_bytes()).unwrap();
    match PublicKey::from_slice(&user_pubkey) {
        Some(key) => key,
        None => panic!("Invalid public key"),
    }
}

