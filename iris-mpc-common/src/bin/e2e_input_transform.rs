use iris_mpc_common::{
    galois_engine::degree4::GaloisRingIrisCodeShare, iris_db::iris::IrisCodeArray,
};
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{Read, Write},
};
use uuid::Uuid;

const RNG_SEED: u64 = 42; // Replace with your seed value

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Signup {
    signup_id: String,
    // iris_code_left: String,
    iris_code_shares_left: Option<[String; 3]>,
    // mask_code_left: String,
    mask_code_shares_left: Option<[String; 3]>,
    // iris_code_right: String,
    iris_code_shares_right: Option<[String; 3]>,
    // mask_code_right: String,
    mask_code_shares_right: Option<[String; 3]>,
    // matched_with: Vec<Match>,
    // whitelisted: bool,
    // closest_signups: Vec<Match>,
    // zkp: String,
    // idcomm: String,
}

#[derive(Debug, Serialize, Deserialize,Clone)]
struct InversedSignup {
    iris_codes: String,
    mask_codes: String,
}

impl Signup {
    fn add_shares(&mut self, signup_inversed: InversedSignup, signup_inversed_right: InversedSignup) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(RNG_SEED);
        let iris_code_left = IrisCodeArray::from_base64(&signup_inversed.iris_codes).unwrap();
        let mask_code_left = IrisCodeArray::from_base64(&signup_inversed.mask_codes).unwrap();
        let iris_code_right = IrisCodeArray::from_base64(&signup_inversed_right.iris_codes).unwrap();
        let mask_code_right = IrisCodeArray::from_base64(&signup_inversed_right.mask_codes).unwrap();

        // Encode iris code shares left
        let shares_iris_code_left =
            GaloisRingIrisCodeShare::encode_iris_code(&iris_code_left, &mask_code_left, &mut rng);
        self.iris_code_shares_left = Some(shares_iris_code_left.map(|x| x.to_base64()));

        // Encode iris code shares right
        let shares_iris_code_right =
            GaloisRingIrisCodeShare::encode_iris_code(&iris_code_right, &mask_code_right, &mut rng);
        self.iris_code_shares_right = Some(shares_iris_code_right.map(|x| x.to_base64()));

        // Encode mask code shares left
        let shares_mask_code_left =
            GaloisRingIrisCodeShare::encode_mask_code(&mask_code_left, &mut rng);
        self.mask_code_shares_left = Some(shares_mask_code_left.map(|x| x.to_base64()));

        // Encode mask code shares right
        let shares_mask_code_right =
            GaloisRingIrisCodeShare::encode_mask_code(&mask_code_right, &mut rng);
        self.mask_code_shares_right = Some(shares_mask_code_right.map(|x| x.to_base64()));
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Match {
    signup_id: String,
    distance_left: f64,
    distance_right: f64,
    manipulation_type: String,
    normalization_used: bool,
}

fn read_json_file(file_path: &str) -> serde_json::Result<Vec<InversedSignup>> {
    let mut file = File::open(file_path).expect("File not found");
    let mut data = String::new();
    file.read_to_string(&mut data)
        .expect("Unable to read string");

    let signups: Vec<InversedSignup> = serde_json::from_str(&data)?;
    Ok(signups)
}

fn write_json_file(file_path: &str, signups: &Vec<Signup>) -> std::io::Result<()> {
    let json_data = serde_json::to_string_pretty(signups).expect("Serialization failed");
    let mut file = File::create(file_path)?;
    file.write_all(json_data.as_bytes())?;
    Ok(())
}

fn main() {
    let input_file = "./iris-mpc-common/src/bin/data/1001lines.json";
    let output_file = "./iris-mpc-common/src/bin/data/1001lines_shares.json";
    let mut signups = read_json_file(input_file).unwrap();

    let mut output_signups = vec![];
    let mut s = Signup {
        signup_id: Uuid::new_v4().to_string(),
        iris_code_shares_left: None,
        mask_code_shares_left: None,
        iris_code_shares_right: None,
        mask_code_shares_right: None,
    };

    s.add_shares(signups[0].clone(),signups[1].clone());
    output_signups.push(s.clone());


    write_json_file(
        output_file,
        &output_signups,
    )
    .unwrap();
    println!(
        "Share calculation completed. Results written to {:?}",
        output_file
    );
}
