use std::sync::Arc;

use crate::{job::Eye, vector_id::VectorId};

/// A helper trait encapsulating the functionality to add iris codes to some
/// form of in-memory store.
pub trait InMemoryStore {
    /// Adds a single record to the in-memory store.
    /// The position of the record in the in-memory store is identified by the
    /// given index, which is 0-based. This method does not directly increase
    /// the size of the in-memory store, since it can also be used to override
    /// existing records. To increase the size of the in-memory store, the
    /// `increment_db_size` method should be called, first ensuring that the
    /// record is a new addition.
    ///
    /// The *_code slices are expected to be of length `IRIS_CODE_LENGTH` and
    /// the `*_mask` slices are expected of length `MASK_CODE_LENGTH`.
    /// The implementation is allowed to panic if this not the case.
    fn load_single_record_from_db(
        &mut self,
        index: usize,
        vector_id: VectorId,
        left_code: &[u16],
        left_mask: &[u16],
        right_code: &[u16],
        right_mask: &[u16],
    );
    /// Increments the internal size of the in-memory store by 1.
    ///
    /// # Arguments
    /// Index - The index of the record in the in-memory store that was just
    /// added and the reason for the size increase. This can be used internally
    /// to increase the size of a specific part of the in-memory store, if
    /// required.
    fn increment_db_size(&mut self, index: usize);

    /// Reserves capacity for at least `additional` more elements to be
    /// inserted, like Vec::reserve.
    fn reserve(&mut self, _additional: usize) {}

    /// Adds a single record to the in-memory store.
    /// The position of the record in the in-memory store is identified by the
    /// given index, which is 0-based. This method does not directly increase
    /// the size of the in-memory store, since it can also be used to override
    /// existing records. To increase the size of the in-memory store, the
    /// `increment_db_size` method should be called, first ensuring that the
    /// record is a new addition.
    ///
    /// The *_code slices are expected to be of length `IRIS_CODE_LENGTH` and
    /// the `*_mask` slices are expected of length `MASK_CODE_LENGTH`.
    /// The implementation is allowed to panic if this not the case.
    ///
    /// In contrast to the DB version, this method loads the lower and upper u8
    /// of the underlying u16 separately in the `_odd` and `_even` slices,
    /// respectively.
    #[allow(clippy::too_many_arguments)]
    fn load_single_record_from_s3(
        &mut self,
        index: usize,
        vector_id: VectorId,
        left_code_odd: &[u8],
        left_code_even: &[u8],
        right_code_odd: &[u8],
        right_code_even: &[u8],
        left_mask_odd: &[u8],
        left_mask_even: &[u8],
        right_mask_odd: &[u8],
        right_mask_even: &[u8],
    ) {
        // this calculates the inverse mapping of the preprocessing done in share_db
        // https://github.com/worldcoin/iris-mpc/blob/d92f3c394ace6ade8ddb8d574fccb2411b6a3ddd/iris-mpc-gpu/src/dot/share_db.rs#L299-L307
        let map_back_to_u16 = |a0, a1| {
            let mut a = [0u8; 2];
            a[0] = ((a0 as i8 as i32) + 128) as u8;
            a[1] = ((a1 as i8 as i32) + 128) as u8;
            u16::from_le_bytes(a)
        };

        let left_code = left_code_odd
            .iter()
            .zip(left_code_even.iter())
            .map(|(odd, even)| map_back_to_u16(*odd, *even))
            .collect::<Vec<_>>();
        let right_code = right_code_odd
            .iter()
            .zip(right_code_even.iter())
            .map(|(odd, even)| map_back_to_u16(*odd, *even))
            .collect::<Vec<_>>();
        let left_mask = left_mask_odd
            .iter()
            .zip(left_mask_even.iter())
            .map(|(odd, even)| map_back_to_u16(*odd, *even))
            .collect::<Vec<_>>();
        let right_mask = right_mask_odd
            .iter()
            .zip(right_mask_even.iter())
            .map(|(odd, even)| map_back_to_u16(*odd, *even))
            .collect::<Vec<_>>();
        self.load_single_record_from_db(
            index,
            vector_id,
            &left_code,
            &left_mask,
            &right_code,
            &right_mask,
        );
    }

    /// Executes any necessary preprocessing steps on the in-memory store.
    ///
    /// This method should be called ONCE, after all records have been loaded.
    fn preprocess_db(&mut self) {}

    /// Return the current size(s) of the in-memory store for printing in Debug
    /// logs.
    fn current_db_sizes(&self) -> impl std::fmt::Debug;

    /// Initialize the DB with fake data of a given size, data may be random and
    /// unpredictable.
    fn fake_db(&mut self, size: usize);
}

/// A helper trait encapsulating the functionality to load iris codes on demand from some source (DB, File-backed, etc.).
#[async_trait::async_trait]
pub trait OnDemandLoader {
    /// Loads records from the source.
    /// The returned Vec has the form:
    /// `(index, side_code, side_mask)`.
    #[allow(clippy::type_complexity)]
    async fn load_records(
        &self,
        side: Eye,
        indices: &[usize],
    ) -> eyre::Result<Vec<(usize, Vec<u16>, Vec<u16>)>>;
}

pub struct CachingOnDemandLoader {
    inner: Arc<dyn OnDemandLoader + Send + Sync + 'static>,
    cache: quick_cache::sync::Cache<usize, (Vec<u16>, Vec<u16>)>,
    fixed_side: Eye,
}

impl CachingOnDemandLoader {
    pub fn new(
        inner: Arc<dyn OnDemandLoader + Send + Sync + 'static>,
        num_cached_iris_codes: usize,
        fixed_side: Eye,
    ) -> Self {
        let cache = quick_cache::sync::Cache::new(num_cached_iris_codes);
        Self {
            inner,
            cache,
            fixed_side,
        }
    }
}

#[async_trait::async_trait]
impl OnDemandLoader for CachingOnDemandLoader {
    async fn load_records(
        &self,
        side: Eye,
        indices: &[usize],
    ) -> eyre::Result<Vec<(usize, Vec<u16>, Vec<u16>)>> {
        if side != self.fixed_side {
            return Err(eyre::eyre!(
                "Invalid side requested, expected: {:?}",
                self.fixed_side
            ));
        }

        // grab cached iris codes
        let cached: Vec<_> = indices.iter().map(|x| (*x, self.cache.get(x))).collect();
        // grab missing ones
        let missing_indices = cached
            .iter()
            .filter_map(|(i, c)| if c.is_none() { Some(*i) } else { None })
            .collect::<Vec<_>>();

        // for missing ones, load from the inner source
        let loaded = if missing_indices.is_empty() {
            vec![]
        } else {
            self.inner.load_records(side, &missing_indices).await?
        };

        // put collected iris codes into the cache
        for (idx, code, mask) in loaded.iter() {
            self.cache.insert(*idx, (code.clone(), mask.clone()));
        }
        let result = cached
            .into_iter()
            .filter_map(|x| {
                if let Some((code, mask)) = x.1 {
                    Some((x.0, code, mask))
                } else {
                    None
                }
            })
            .chain(loaded.into_iter())
            .collect::<Vec<_>>();

        Ok(result)
    }
}
