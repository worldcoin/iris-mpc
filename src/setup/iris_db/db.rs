use super::iris::IrisCode;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;

#[derive(Default)]
pub struct IrisDB {
    pub db: Vec<IrisCode>,
}

impl IrisDB {
    pub fn new() -> Self {
        Self { db: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.db.is_empty()
    }

    pub fn len(&self) -> usize {
        self.db.len()
    }

    pub fn add_iris(&mut self, iris: IrisCode) {
        self.db.push(iris);
    }

    pub fn new_random_rng<R: Rng>(size: usize, rng: &mut R) -> Self {
        let mut db = Vec::with_capacity(size);
        for _ in 0..size {
            db.push(IrisCode::random_rng(rng));
        }

        Self { db }
    }

    pub fn new_random_seed(size: usize, rng_seed: u64) -> Self {
        let db = (0..size)
            .into_par_iter()
            .map(|_| {
                let mut rng = StdRng::seed_from_u64(rng_seed);
                IrisCode::random_rng(&mut rng)
            })
            .collect::<Vec<_>>();

        Self { db }
    }

    pub fn iris_in_db(&self, iris: &IrisCode) -> bool {
        self.db.iter().any(|x| iris.is_close(x))
    }
}

#[cfg(test)]
mod iris_test {
    use super::*;

    const TESTRUNS: usize = 5;
    const DB_SIZE: usize = 100;

    #[test]
    fn iris_in_db_test() {
        let mut rng = rand::thread_rng();
        let db = IrisDB::new_random_rng(DB_SIZE, &mut rng);
        for _ in 0..TESTRUNS {
            let iris = IrisCode::random_rng(&mut rng);
            assert_eq!(db.iris_in_db(&iris), db.db.iter().any(|x| iris.is_close(x)));
            let index = rng.gen_range(0..DB_SIZE);
            let iris = db.db[index].get_similar_iris(&mut rng);
            let in_db = db.iris_in_db(&iris);
            assert!(in_db);
            assert_eq!(in_db, db.db.iter().any(|x| iris.is_close(x)));
        }
    }
}
