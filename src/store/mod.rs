pub struct StoredIris {}

impl StoredIris {
    pub fn code(&self) -> &[u16] {
        &[]
    }

    pub fn mask(&self) -> &[u16] {
        &[]
    }
}

pub struct Store {}

impl Store {
    pub fn new() -> Self {
        Store {}
    }

    pub fn iter_irises(&self) -> impl Iterator<Item = &StoredIris> {
        // TODO
        vec![].into_iter()
    }

    pub fn insert_irises(&self, _codes_and_masks: &[(&[u16], &[u16])]) {
        // TODO
    }
}
