use base64::{engine::general_purpose, Engine};
use iris_mpc_common::{
    galois_engine::degree4::GaloisRingIrisCodeShare, iris_db::iris::IrisCodeArray,
};
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{Read, Write},
};

const RNG_SEED: u64 = 42; // Replace with your seed value

#[derive(Debug, Serialize, Deserialize)]
struct Signup {
    signup_id:              String,
    iris_code_left:         String,
    iris_code_shares_left:  Option<[String; 3]>,
    mask_code_left:         String,
    mask_code_shares_left:  Option<[String; 3]>,
    iris_code_right:        String,
    iris_code_shares_right: Option<[String; 3]>,
    mask_code_right:        String,
    mask_code_shares_right: Option<[String; 3]>,
    matched_with:           Vec<Match>,
    whitelisted:            bool,
    closest_signups:        Vec<Match>,
    zkp:                    String,
    idcomm:                 String,
}

impl Signup {
    fn add_shares(&mut self) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(RNG_SEED);
        let iris_code_left = IrisCodeArray::from_base64(&self.iris_code_left).unwrap();
        let mask_code_left = IrisCodeArray::from_base64(&self.mask_code_left).unwrap();
        let iris_code_right = IrisCodeArray::from_base64(&self.iris_code_right).unwrap();
        let mask_code_right = IrisCodeArray::from_base64(&self.mask_code_right).unwrap();

        // Encode iris code shares left
        let shares_iris_code_left =
            GaloisRingIrisCodeShare::encode_iris_code(&iris_code_left, &mask_code_left, &mut rng);
        self.iris_code_shares_left = Some(get_share_strings(&shares_iris_code_left));

        // Encode iris code shares right
        let shares_iris_code_right =
            GaloisRingIrisCodeShare::encode_iris_code(&iris_code_right, &mask_code_right, &mut rng);
        self.iris_code_shares_right = Some(get_share_strings(&shares_iris_code_right));

        // Encode mask code shares left
        let shares_mask_code_left =
            GaloisRingIrisCodeShare::encode_mask_code(&mask_code_left, &mut rng);
        self.mask_code_shares_left = Some(get_share_strings(&shares_mask_code_left));

        // Encode mask code shares right
        let shares_mask_code_right =
            GaloisRingIrisCodeShare::encode_mask_code(&mask_code_right, &mut rng);
        self.mask_code_shares_right = Some(get_share_strings(&shares_mask_code_right));
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Match {
    signup_id:          String,
    distance_left:      f64,
    distance_right:     f64,
    manipulation_type:  String,
    normalization_used: bool,
}

fn get_share_strings(shares_iris_code_left: &[GaloisRingIrisCodeShare; 3]) -> [String; 3] {
    let mut shares: [String; 3] = Default::default();
    for i in 0..3 {
        shares[i] =
            general_purpose::STANDARD.encode(bytemuck::cast_slice(&shares_iris_code_left[i].coefs));
    }
    return shares;
}

fn read_json_file(file_path: &str) -> serde_json::Result<Vec<Signup>> {
    let mut file = File::open(file_path).expect("File not found");
    let mut data = String::new();
    file.read_to_string(&mut data)
        .expect("Unable to read string");

    let signups: Vec<Signup> = serde_json::from_str(&data)?;
    Ok(signups)
}

fn write_json_file(file_path: &str, signups: &Vec<Signup>) -> std::io::Result<()> {
    let json_data = serde_json::to_string_pretty(signups).expect("Serialization failed");
    let mut file = File::create(file_path)?;
    file.write_all(json_data.as_bytes())?;
    Ok(())
}

fn main() {
    let input_file = "./iris-mpc-common/src/bin/data/ss_e2e.json";
    let output_file = "./iris-mpc-common/src/bin/data/generated_ss_e2e_shares.json";
    let mut signups = read_json_file(input_file).unwrap();
    for signup in &mut signups {
        println!("Calculating shares for signup {:?}", signup.signup_id);
        signup.add_shares();
    }
    write_json_file(
        "./iris-mpc-common/src/bin/data/generated_ss_e2e_shares.json",
        &signups,
    )
    .unwrap();
    println!(
        "Share calculation completed. Results written to {:?}",
        output_file
    );
}
