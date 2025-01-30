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
        left_code_odd: &[u8],
        left_code_even: &[u8],
        right_code_odd: &[u8],
        right_code_even: &[u8],
        left_mask_odd: &[u8],
        left_mask_even: &[u8],
        right_mask_odd: &[u8],
        right_mask_even: &[u8],
    );
}
