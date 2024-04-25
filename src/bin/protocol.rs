use std::env;

use gpu_iris_mpc::{
    setup::{
        id::PartyID,
        iris_db::{db::IrisDB, iris::IrisCode, shamir_db::ShamirIrisDB, shamir_iris::ShamirIris},
        shamir::Shamir,
    },
    IrisCodeDB,
};

const DB_SIZE: usize = 1000;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let mut rng = rand::thread_rng();
    let args = env::args().collect::<Vec<_>>();
    let party_id: usize = args[1].parse().unwrap();
    let url = args.get(2);

    let db = IrisDB::new_random_rng(DB_SIZE, &mut rng);
    let shamir_db = ShamirIrisDB::share_db(&db, &mut rng);

    let l_coeff = Shamir::my_lagrange_coeff_d2(PartyID::try_from(party_id as u8).unwrap());

    let codes_db = shamir_db[party_id]
        .db
        .iter()
        .flat_map(|entry| entry.code)
        .collect::<Vec<_>>();

    let mut engine = IrisCodeDB::init(party_id, l_coeff, &codes_db, url.clone(), false);


    ShamirIris::share_iris(&IrisCode::random_rng(&mut rng), &mut rng);




    Ok(())
}
