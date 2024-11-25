pub mod hnsw;
pub mod io;
pub mod plaintext_store;

pub fn limited_iterator<I>(iter: I, limit: Option<usize>) -> Box<dyn Iterator<Item = I::Item>>
where
    I: Iterator + 'static,
{
    match limit {
        Some(num) => Box::new(iter.take(num)),
        None => Box::new(iter),
    }
}
