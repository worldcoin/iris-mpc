use bytemuck::cast_slice;
use memmap2::{Mmap, MmapMut};
use std::fs::{File, OpenOptions};

pub fn write_mmap_file(file_path: &str, data: &[u16]) -> eyre::Result<()> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(false) // we truncate by setting the length later
        .read(true)
        .write(true)
        .open(file_path)?;
    file.set_len(std::mem::size_of_val(data) as u64)?;
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
