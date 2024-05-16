use std::fs::{File, OpenOptions};
use bytemuck::cast_slice;
use memmap2::{Mmap, MmapMut};

pub fn write_mmap_file(file_path: &str, data: &[u16]) -> eyre::Result<()> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(file_path)?;
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