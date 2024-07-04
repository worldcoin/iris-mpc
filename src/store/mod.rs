use eyre::Result;

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
    pub async fn new_from_env() -> Result<Self> {
        Ok(Store {})
    }

    pub async fn iter_irises(&self) -> Result<impl Iterator<Item = &StoredIris>> {
        // TODO
        Ok(vec![].into_iter())
    }

    pub async fn insert_irises(&self, _codes_and_masks: &[(&[u16], &[u16])]) -> Result<()> {
        // TODO
        Ok(())
    }
}
