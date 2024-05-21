use super::{db::IrisDB, shamir_iris::ShamirIris};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

#[derive(Default)]
pub struct ShamirIrisDB {
    pub db: Vec<ShamirIris>,
}

impl ShamirIrisDB {
    pub fn new() -> Self {
        Self { db: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.db.is_empty()
    }

    pub fn len(&self) -> usize {
        self.db.len()
    }

    pub fn share_db<R: Rng>(db: &IrisDB, rng: &mut R) -> [Self; 3] {
        let len = db.len();
        let mut db1 = Vec::with_capacity(len);
        let mut db2 = Vec::with_capacity(len);
        let mut db3 = Vec::with_capacity(len);
        for iris in db.db.iter() {
            let [shares1, shares2, shares3] = ShamirIris::share_iris(iris, rng);
            db1.push(shares1);
            db2.push(shares2);
            db3.push(shares3);
        }

        [Self { db: db1 }, Self { db: db2 }, Self { db: db3 }]
    }

    /// Only use for testing
    pub fn share_db_par<R: Rng>(db: &IrisDB, rng: &mut R) -> [Self; 3] {
        // Fork out the rngs to be able to use them concurrently
        let rng_seeds = db.db.iter().map(|_| rng.gen()).collect::<Vec<_>>();

        let (db1, (db2, db3)): (Vec<_>, (Vec<_>, Vec<_>)) = db
            .db
            .par_iter()
            .enumerate()
            .map(|(i, iris)| {
                let mut rng = StdRng::from_seed(rng_seeds[i]);
                let [shares1, shares2, shares3] = ShamirIris::share_iris(iris, &mut rng);
                (shares1, (shares2, shares3))
            })
            .unzip();

        [Self { db: db1 }, Self { db: db2 }, Self { db: db3 }]
    }
}

#[cfg(test)]
mod shamir_db_test {
    use super::*;
    use crate::setup::{
        id::PartyID,
        iris_db::iris::IrisCodeArray,
        shamir::{Shamir, P, P32},
    };

    const TESTRUNS: usize = 5;
    const DB_SIZE: usize = 100;

    #[test]
    fn share_db_reconstruct_test() {
        let mut rng = rand::thread_rng();

        let lagrange = [
            Shamir::my_lagrange_coeff_d1(PartyID::ID0, PartyID::ID1) as u32,
            Shamir::my_lagrange_coeff_d1(PartyID::ID1, PartyID::ID0) as u32,
        ];

        for _ in 0..TESTRUNS {
            let db = IrisDB::new_random_rng(DB_SIZE, &mut rng);
            let shamir_db = ShamirIrisDB::share_db(&db, &mut rng);
            assert_eq!(db.len(), DB_SIZE);
            assert_eq!(shamir_db[0].len(), DB_SIZE);
            assert_eq!(shamir_db[1].len(), DB_SIZE);
            assert_eq!(shamir_db[2].len(), DB_SIZE);

            for i in 0..DB_SIZE {
                for bitindex in 0..IrisCodeArray::IRIS_CODE_SIZE {
                    // mask comparison (only 2 parties required for reconstruction)
                    let mask = db.db[i].mask.get_bit(bitindex);
                    let rec_mask = ((0..2).fold(0u32, |acc, j| {
                        acc + (shamir_db[j].db[i].mask[bitindex] as u32 * lagrange[j]) % P32
                    }) % P32) as u16;
                    assert!(rec_mask == 0 || rec_mask == 1);
                    assert_eq!(rec_mask, mask as u16);

                    // code comparison (only 2 parties required for reconstruction)
                    let code = db.db[i].code.get_bit(bitindex);
                    let rec_code = ((0..2).fold(0u32, |acc, j| {
                        acc + (shamir_db[j].db[i].code[bitindex] as u32 * lagrange[j]) % P32
                    }) % P32) as u16;
                    assert!(rec_code == 0 || rec_code == 1 || rec_code == P - 1);
                    match (code, mask) {
                        (false, false) => assert_eq!(rec_code, 0),
                        (true, false) => assert_eq!(rec_code, 0),
                        (false, true) => assert_eq!(rec_code, 1),
                        (true, true) => assert_eq!(rec_code, P - 1),
                    }
                }
            }
        }
    }
}
