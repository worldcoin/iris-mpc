use bytemuck::cast_slice;
use memmap2::{Mmap, MmapMut};
use std::fs::{File, OpenOptions};

pub fn write_mmap_file(file_path: &str, data: &[u16]) -> eyre::Result<()> {
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(file_path)?;
    file.set_len((data.len() * std::mem::size_of::<u16>()) as u64)?;
    let mut mmap = unsafe { MmapMut::map_mut(&file)? };
    let bytes = cast_slice(data);
    mmap[..bytes.len()].copy_from_slice(bytes);
    mmap.flush()?;
    Ok(())
}

pub fn read_mmap_file(file_path: &str) -> eyre::Result<Vec<u16>> {
    let file = File::open(file_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let data: &[u16] = cast_slice(&mmap);
    Ok(data.to_vec())
}
