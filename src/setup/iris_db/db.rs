use super::iris::IrisCode;
use rand::Rng;

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
}
