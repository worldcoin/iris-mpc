//! Intel AMX kernel for the exact linear scan's full-rotation dot products.
//!
//! Per iris row (800 coefficients), the 31 rotation dot products of 16
//! records form one AMX problem: `C[window][record] = sum_t hq[4 window + t] *
//! d[record][t]`, where `hq` is the query row shifted by 15 rotations and
//! extended cyclically, so every rotation window starts at a plain offset. A
//! 16-window A tile is a single `tileloadd` from `hq` with a 4-byte row
//! stride. Products in `Z_{2^16}` are split into bytes, `x * y = ll + 2^8 (lh +
//! hl) mod 2^16`, so two `u32` accumulators (`ll` and the combined cross term)
//! take three `tdpbuud` per tile pair.
//!
//! Database records are stored in AMX VNNI order in groups of 16 records,
//! [`GROUP_BYTES`] per group. Each row's 800 coefficients are padded to 832
//! (13 K-blocks of 64), which is the only memory overhead (4%) over the plain
//! `u16` shares. Groups live in a NUMA-bound [`GroupArena`].

use crate::protocol::shared_iris::GaloisRingSharedIris;
use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    IRIS_CODE_LENGTH, MASK_CODE_LENGTH, ROTATIONS,
};
use std::sync::{RwLock, RwLockReadGuard};

/// Coefficients per iris row; rotations shift cyclically within a row.
pub const ROW_COEFS: usize = 800;
pub const CODE_ROWS: usize = IRIS_CODE_LENGTH / ROW_COEFS;
pub const MASK_ROWS: usize = MASK_CODE_LENGTH / ROW_COEFS;
pub const ROWS: usize = CODE_ROWS + MASK_ROWS;
/// Records per group (the columns of one B tile).
pub const GROUP: usize = 16;
/// K-blocks of 64 coefficients per row, 832 coefficients with padding.
const KB: usize = ROW_COEFS.div_ceil(64);
const TILE: usize = 1024;
/// A query row is `[low bytes | high bytes]` of the shifted, extended row.
const Q_ROW_BYTES: usize = 2 * TILE;
/// A group row is `KB x [low-byte tile | high-byte tile]`.
const DB_ROW_BYTES: usize = KB * 2 * TILE;
/// Bytes of one group of [`GROUP`] records.
pub const GROUP_BYTES: usize = ROWS * DB_ROW_BYTES;
/// Rotation windows per query tile pair; window 31 is padding.
const WINDOWS: usize = 32;
/// Window `s` evaluates rotation `CENTER_WINDOW - s`.
const CENTER_WINDOW: usize = ROTATIONS - 1;
/// Coefficient shift of the query row: window 0 is the most-shifted rotation.
const QUERY_SHIFT: usize = 4 * (ROTATIONS / 2);
/// Records per arena segment. One segment is the unit of NUMA placement.
pub const SEGMENT_RECORDS: usize = 4096;
const SEGMENT_GROUPS: usize = SEGMENT_RECORDS / GROUP;
const SEGMENT_BYTES: usize = SEGMENT_GROUPS * GROUP_BYTES;
/// Database prefetch distance of the kernel, in bytes of the group stream.
#[cfg(target_arch = "x86_64")]
const PREFETCH_BYTES: usize = 12 * 1024;

const _: () = {
    assert!(CODE_ROWS * ROW_COEFS == IRIS_CODE_LENGTH);
    assert!(MASK_ROWS * ROW_COEFS == MASK_CODE_LENGTH);
    assert!(ROTATIONS == WINDOWS - 1);
    // Every window of the last K-block stays inside the extended query row.
    assert!(4 * (WINDOWS - 1) + 64 * KB <= TILE);
    // Groups start on page boundaries inside a segment.
    assert!(GROUP_BYTES.is_multiple_of(4096));
};

/// Byte offset of coefficient `coef` (low byte) of `row` for lane `lane`.
/// Tile row `kk` of K-block `kb` holds coefficients `64 kb + 4 kk + j` of the
/// 16 records, 4 bytes per record (VNNI order).
#[inline(always)]
fn group_offset(row: usize, coef: usize, lane: usize) -> usize {
    let (kb, within) = (coef / 64, coef % 64);
    row * DB_ROW_BYTES + kb * 2 * TILE + (within / 4) * 64 + lane * 4 + within % 4
}

fn row_coefs(iris: &GaloisRingSharedIris, row: usize) -> &[u16] {
    if row < CODE_ROWS {
        &iris.code.coefs[row * ROW_COEFS..(row + 1) * ROW_COEFS]
    } else {
        let row = row - CODE_ROWS;
        &iris.mask.coefs[row * ROW_COEFS..(row + 1) * ROW_COEFS]
    }
}

/// Write `iris` into lane `lane` of a group. Padding coefficients are never
/// written and must stay zero.
pub fn write_record(group: &mut [u8], lane: usize, iris: &GaloisRingSharedIris) {
    assert_eq!(group.len(), GROUP_BYTES);
    assert!(lane < GROUP);
    let at = 4 * lane;
    for (row, row_bytes) in group.chunks_exact_mut(DB_ROW_BYTES).enumerate() {
        let coefs = row_coefs(iris, row);
        for (block, tiles) in coefs.chunks(64).zip(row_bytes.chunks_exact_mut(2 * TILE)) {
            let (low, high) = tiles.split_at_mut(TILE);
            // Tile row `kk` of a K-block holds coefficients `4 kk..4 kk + 4`
            // of every lane, 4 bytes per lane.
            for ((quad, low), high) in block
                .chunks_exact(4)
                .zip(low.chunks_exact_mut(64))
                .zip(high.chunks_exact_mut(64))
            {
                for (j, &value) in quad.iter().enumerate() {
                    low[at + j] = value as u8;
                    high[at + j] = (value >> 8) as u8;
                }
            }
        }
    }
}

/// Read lane `lane` of a group back as a share with the given share id.
pub fn read_record(group: &[u8], lane: usize, share_id: usize) -> GaloisRingSharedIris {
    debug_assert_eq!(group.len(), GROUP_BYTES);
    let mut code = [0_u16; IRIS_CODE_LENGTH];
    let mut mask = [0_u16; MASK_CODE_LENGTH];
    for row in 0..ROWS {
        let out = if row < CODE_ROWS {
            &mut code[row * ROW_COEFS..(row + 1) * ROW_COEFS]
        } else {
            let row = row - CODE_ROWS;
            &mut mask[row * ROW_COEFS..(row + 1) * ROW_COEFS]
        };
        for (quad, values) in out.chunks_exact_mut(4).enumerate() {
            let low = group_offset(row, 4 * quad, lane);
            for (j, value) in values.iter_mut().enumerate() {
                *value = u16::from(group[low + j]) | (u16::from(group[low + TILE + j]) << 8);
            }
        }
    }
    GaloisRingSharedIris {
        code: GaloisRingIrisCodeShare {
            id: share_id,
            coefs: code,
        },
        mask: GaloisRingTrimmedMaskCodeShare {
            id: share_id,
            coefs: mask,
        },
    }
}

#[repr(C, align(64))]
struct QueryBytes([u8; ROWS * Q_ROW_BYTES]);

/// A preprocessed query in AMX A-tile layout: per row, the low and high bytes
/// of `hq[c] = q[(c - QUERY_SHIFT) mod ROW_COEFS]` for `c < 1024`, so that
/// window `s` (rotation `30 - s`) starts at byte `4 s`.
pub struct AmxQuery {
    bytes: Box<QueryBytes>,
}

impl std::fmt::Debug for AmxQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmxQuery").finish_non_exhaustive()
    }
}

impl AmxQuery {
    /// Pack a preprocessed center-rotation query.
    pub fn new(query: &GaloisRingSharedIris) -> Self {
        let mut bytes = Box::new(QueryBytes([0; ROWS * Q_ROW_BYTES]));
        for row in 0..ROWS {
            let coefs = row_coefs(query, row);
            let out = &mut bytes.0[row * Q_ROW_BYTES..(row + 1) * Q_ROW_BYTES];
            for c in 0..TILE {
                let value = coefs[(c + ROW_COEFS - QUERY_SHIFT) % ROW_COEFS];
                out[c] = value as u8;
                out[TILE + c] = (value >> 8) as u8;
            }
        }
        Self { bytes }
    }

    fn row(&self, row: usize) -> &[u8] {
        &self.bytes.0[row * Q_ROW_BYTES..(row + 1) * Q_ROW_BYTES]
    }

    /// The rows from `row` onward, for kernels that read several rows.
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    fn rows_from(&self, row: usize) -> &[u8] {
        &self.bytes.0[row * Q_ROW_BYTES..]
    }
}

/// The four C tiles of one component: `[ll | mid]` for windows 0..16, then
/// for windows 16..32, each 16 windows x 16 lanes of `u32`.
#[repr(C, align(64))]
pub struct Accumulators(pub [u32; 4 * 256]);

impl Default for Accumulators {
    fn default() -> Self {
        Self([0; 4 * 256])
    }
}

impl Accumulators {
    /// Scatter one component's 31 rotations of the lanes selected by
    /// `outputs` into `(code, mask)`-interleaved records: `out[(record * 31 +
    /// rotation) * 2 + component]`, where `outputs[lane]` is the record index
    /// or `u32::MAX` for an unused lane.
    #[inline]
    pub fn scatter(&self, component: usize, outputs: &[u32; GROUP], out: &mut [u16]) {
        // Combine the byte-limb accumulators one 16-lane tile row at a time
        // (vectorizable), then transpose windows into per-record rotations.
        let mut values = [[0_u16; GROUP]; WINDOWS];
        for (window, row) in values.iter_mut().enumerate() {
            let (half, tile_row) = (window / 16, window % 16);
            let ll = &self.0[(2 * half) * 256 + tile_row * 16..][..GROUP];
            let mid = &self.0[(2 * half + 1) * 256 + tile_row * 16..][..GROUP];
            for ((value, &ll), &mid) in row.iter_mut().zip(ll).zip(mid) {
                *value = ll.wrapping_add(mid << 8) as u16;
            }
        }
        for (lane, &record) in outputs.iter().enumerate() {
            if record == u32::MAX {
                continue;
            }
            // This component's 31 values of the record, every other element.
            let base = record as usize * ROTATIONS * 2 + component;
            for (rotation, out) in out[base..base + 2 * ROTATIONS - 1]
                .iter_mut()
                .step_by(2)
                .enumerate()
            {
                *out = values[CENTER_WINDOW - rotation][lane];
            }
        }
    }
}

/// Scalar evaluation of the AMX layout: `[rotation][code, mask]` of one lane.
/// A reference for tests and for hosts without AMX.
pub fn scan_lane_reference(query: &AmxQuery, group: &[u8], lane: usize) -> [[u16; 2]; ROTATIONS] {
    let mut out = [[0_u16; 2]; ROTATIONS];
    for (rotation, pair) in out.iter_mut().enumerate() {
        let window = CENTER_WINDOW - rotation;
        for row in 0..ROWS {
            let component = usize::from(row >= CODE_ROWS);
            let q = query.row(row);
            let mut sum = 0_u16;
            for coef in 0..ROW_COEFS {
                let qv =
                    u16::from(q[4 * window + coef]) | (u16::from(q[TILE + 4 * window + coef]) << 8);
                let offset = group_offset(row, coef, lane);
                let dv = u16::from(group[offset]) | (u16::from(group[offset + TILE]) << 8);
                sum = sum.wrapping_add(qv.wrapping_mul(dv));
            }
            pair[component] = pair[component].wrapping_add(sum);
        }
    }
    out
}

/// Whether this host can run the AMX kernel: AMX-TILE and AMX-INT8 are
/// present and the kernel grants tile-data permission. The result is cached;
/// `IRIS_MPC_DISABLE_AMX_SCAN=1` turns the kernel off.
pub fn amx_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let disabled = std::env::var("IRIS_MPC_DISABLE_AMX_SCAN")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        !disabled && kernel::detect()
    })
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod kernel {
    use super::*;
    use std::arch::asm;

    pub(super) fn detect() -> bool {
        // CPUID.(EAX=7, ECX=0):EDX bit 24 is AMX-TILE, bit 25 AMX-INT8.
        let leaf = std::arch::x86_64::__cpuid_count(7, 0);
        let has_amx = leaf.edx & (1 << 24) != 0 && leaf.edx & (1 << 25) != 0;
        has_amx && request_permission()
    }

    /// Ask the kernel for the AMX tile-data state (process-wide).
    pub(super) fn request_permission() -> bool {
        const ARCH_REQ_XCOMP_PERM: libc::c_long = 0x1023;
        const XFEATURE_XTILEDATA: libc::c_long = 18;
        // SAFETY: a plain syscall without pointer arguments.
        unsafe {
            libc::syscall(
                libc::SYS_arch_prctl,
                ARCH_REQ_XCOMP_PERM,
                XFEATURE_XTILEDATA,
            ) == 0
        }
    }

    #[repr(C, align(64))]
    struct TileConfig {
        palette: u8,
        start_row: u8,
        reserved: [u8; 14],
        colsb: [u16; 16],
        rows: [u8; 16],
    }

    /// Configure all eight tiles as 16 rows x 64 bytes on this thread.
    ///
    /// # Safety
    /// [`amx_available`] must have returned true.
    pub unsafe fn configure_tiles() {
        let mut config = TileConfig {
            palette: 1,
            start_row: 0,
            reserved: [0; 14],
            colsb: [0; 16],
            rows: [0; 16],
        };
        for tile in 0..8 {
            config.colsb[tile] = 64;
            config.rows[tile] = 16;
        }
        asm!("ldtilecfg [{}]", in(reg) &config, options(nostack, readonly));
    }

    /// Accumulate `rows` rows of one component of one query against one
    /// group and store the four C tiles to `acc`.
    ///
    /// Tiles: 0/1 are `ll`/`mid` of windows 0..16, 2/3 of windows 16..32; 4 and
    /// 7 hold the lower- and upper-window query tile of the low or high byte
    /// plane, 5 and 6 the database low- and high-byte tiles. The upper-window
    /// tile of K-block `k` equals the lower-window tile of K-block `k + 1`, so
    /// it stays in its register and only two query tiles (whose rows straddle
    /// cache lines) are loaded per K-block.
    ///
    /// # Safety
    /// Tiles must be configured on this thread; `query` must point to `rows`
    /// query rows and `group` to `rows` group rows.
    #[inline(never)]
    pub unsafe fn component_tiles(
        query: *const u8,
        group: *const u8,
        rows: usize,
        acc: &mut Accumulators,
    ) {
        asm!(
            "tilezero tmm0", "tilezero tmm1", "tilezero tmm2", "tilezero tmm3",
            "2:",
            "mov {qa}, {qrow}",
            "tileloadd tmm4, [{qa} + {s4}]",
            "tileloadd tmm7, [{qa} + {s4} + 1024]",
            "mov {kb:e}, 13",
            "3:",
            "lea {p}, [{db} + {pf}]",
            "prefetcht0 [{p}]", "prefetcht0 [{p} + 64]", "prefetcht0 [{p} + 128]",
            "prefetcht0 [{p} + 192]", "prefetcht0 [{p} + 256]", "prefetcht0 [{p} + 320]",
            "prefetcht0 [{p} + 384]", "prefetcht0 [{p} + 448]", "prefetcht0 [{p} + 512]",
            "prefetcht0 [{p} + 576]", "prefetcht0 [{p} + 640]", "prefetcht0 [{p} + 704]",
            "prefetcht0 [{p} + 768]", "prefetcht0 [{p} + 832]", "prefetcht0 [{p} + 896]",
            "prefetcht0 [{p} + 960]", "prefetcht0 [{p} + 1024]", "prefetcht0 [{p} + 1088]",
            "prefetcht0 [{p} + 1152]", "prefetcht0 [{p} + 1216]", "prefetcht0 [{p} + 1280]",
            "prefetcht0 [{p} + 1344]", "prefetcht0 [{p} + 1408]", "prefetcht0 [{p} + 1472]",
            "prefetcht0 [{p} + 1536]", "prefetcht0 [{p} + 1600]", "prefetcht0 [{p} + 1664]",
            "prefetcht0 [{p} + 1728]", "prefetcht0 [{p} + 1792]", "prefetcht0 [{p} + 1856]",
            "prefetcht0 [{p} + 1920]", "prefetcht0 [{p} + 1984]",
            "tileloadd tmm5, [{db} + {s64}]",
            "tileloadd tmm6, [{db} + {s64} + 1024]",
            "tdpbuud tmm1, tmm7, tmm5",
            "tileloadd tmm7, [{qa} + {s4} + 1088]",
            "tdpbuud tmm0, tmm4, tmm5",
            "tdpbuud tmm1, tmm4, tmm6",
            "tileloadd tmm4, [{qa} + {s4} + 64]",
            "tdpbuud tmm3, tmm7, tmm5",
            "tdpbuud tmm2, tmm4, tmm5",
            "tdpbuud tmm3, tmm4, tmm6",
            "add {db}, 2048",
            "add {qa}, 64",
            "dec {kb:e}",
            "jnz 3b",
            "add {qrow}, 2048",
            "dec {rows}",
            "jnz 2b",
            "tilestored [{acc} + {s64}], tmm0",
            "tilestored [{acc} + {s64} + 1024], tmm1",
            "tilestored [{acc} + {s64} + 2048], tmm2",
            "tilestored [{acc} + {s64} + 3072], tmm3",
            qrow = inout(reg) query => _,
            db = inout(reg) group => _,
            rows = inout(reg) rows => _,
            qa = out(reg) _,
            kb = out(reg) _,
            p = out(reg) _,
            pf = in(reg) PREFETCH_BYTES,
            s64 = in(reg) 64usize,
            s4 = in(reg) 4usize,
            acc = in(reg) acc.0.as_mut_ptr(),
            options(nostack),
        );
    }
}

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
mod kernel {
    pub(super) fn detect() -> bool {
        false
    }
}

/// Per-thread AMX state for [`scan_group`].
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
pub struct AmxThread {
    acc: Box<Accumulators>,
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
impl AmxThread {
    /// Configure the tiles of the calling thread. Returns `None` without AMX.
    pub fn new() -> Option<Self> {
        if !amx_available() {
            return None;
        }
        // SAFETY: AMX is available and permission was granted to the process.
        unsafe { kernel::configure_tiles() };
        Some(Self {
            acc: Box::default(),
        })
    }

    /// Evaluate all 31 rotations of one query against one group and hand the
    /// code, then the mask, accumulators to `emit`.
    ///
    /// The group must be a full [`GROUP_BYTES`] slice of the arena.
    pub fn scan_group(
        &mut self,
        query: &AmxQuery,
        group: &[u8],
        mut emit: impl FnMut(usize, &Accumulators),
    ) {
        assert_eq!(group.len(), GROUP_BYTES);
        for (component, row0, rows) in [(0, 0, CODE_ROWS), (1, CODE_ROWS, MASK_ROWS)] {
            // SAFETY: tiles were configured on this thread in `new`; the query
            // has ROWS rows of Q_ROW_BYTES and the group ROWS rows of
            // DB_ROW_BYTES, and the kernel reads `rows` rows from `row0`.
            // Prefetches past the group are hints and never fault.
            unsafe {
                kernel::component_tiles(
                    query.rows_from(row0).as_ptr(),
                    group.as_ptr().add(row0 * DB_ROW_BYTES),
                    rows,
                    &mut self.acc,
                );
            }
            emit(component, &self.acc);
        }
    }
}

/// An anonymous memory mapping of one arena segment, with one lock per group.
///
/// A group's bytes are only accessed under its lock: scans read a group
/// under a read guard, record writes take the write guard, so records of
/// different groups are written concurrently.
struct Segment {
    ptr: *mut u8,
    locks: Box<[RwLock<()>]>,
}

impl Segment {
    /// Map a segment and prefer placing its pages on NUMA node `node`.
    /// Aborts like any allocation failure if the mapping fails.
    fn new(node: Option<usize>) -> Self {
        // SAFETY: an anonymous private mapping; the result is checked below.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                SEGMENT_BYTES,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            tracing::error!(
                "failed to map an iris arena segment: {}",
                std::io::Error::last_os_error()
            );
            std::alloc::handle_alloc_error(
                std::alloc::Layout::from_size_align(SEGMENT_BYTES, 4096)
                    .expect("the segment layout is valid"),
            );
        }
        #[cfg(target_os = "linux")]
        {
            // Best effort: huge pages cut the TLB misses of the streaming
            // scan, and binding places the pages before first touch.
            // SAFETY: advice and policy on the mapping created above.
            unsafe {
                libc::madvise(ptr, SEGMENT_BYTES, libc::MADV_HUGEPAGE);
                if let Some(node) = node.filter(|&node| node < 64) {
                    const MPOL_PREFERRED: libc::c_long = 1;
                    let nodemask: libc::c_ulong = 1 << node;
                    libc::syscall(
                        libc::SYS_mbind,
                        ptr,
                        SEGMENT_BYTES,
                        MPOL_PREFERRED,
                        &nodemask as *const libc::c_ulong,
                        64 as libc::c_ulong,
                        0 as libc::c_uint,
                    );
                }
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = node;
        Self {
            ptr: ptr.cast(),
            locks: (0..SEGMENT_GROUPS).map(|_| RwLock::new(())).collect(),
        }
    }

    fn read_group(&self, index: usize) -> GroupRef<'_> {
        let guard = self.locks[index].read().expect("arena group lock poisoned");
        // SAFETY: the mapping spans SEGMENT_GROUPS groups and lives as long
        // as self; the read guard excludes writers of this group.
        let bytes =
            unsafe { std::slice::from_raw_parts(self.ptr.add(index * GROUP_BYTES), GROUP_BYTES) };
        GroupRef {
            _guard: guard,
            bytes,
        }
    }

    fn write_group(&self, index: usize, write: impl FnOnce(&mut [u8])) {
        let _guard = self.locks[index]
            .write()
            .expect("arena group lock poisoned");
        // SAFETY: as in `read_group`; the write guard makes this the only
        // reference to the group's bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(self.ptr.add(index * GROUP_BYTES), GROUP_BYTES)
        };
        write(bytes);
    }
}

impl Drop for Segment {
    fn drop(&mut self) {
        // SAFETY: mapped in `new` with this length.
        unsafe { libc::munmap(self.ptr.cast(), SEGMENT_BYTES) };
    }
}

// SAFETY: a segment owns its mapping, and every access to a group's bytes
// holds that group's lock.
unsafe impl Send for Segment {}
// SAFETY: as above.
unsafe impl Sync for Segment {}

/// Records in AMX group layout, indexed by `serial_id - 1`, in segments of
/// [`SEGMENT_RECORDS`] that are placed round-robin on the NUMA nodes.
///
/// Each group has its own lock (see [`Segment`]), and the segment list is
/// only locked exclusively to append a segment, so records of different
/// groups are written in parallel. A slot holds the latest record written to
/// it: callers order record updates with the scans that must not observe
/// them (the service applies a batch's mutations after its searches).
pub struct GroupArena {
    segments: RwLock<Vec<Segment>>,
    share_id: usize,
    /// NUMA node ids, in the order of [`GroupArena::node_of_group`].
    numa_nodes: Vec<usize>,
}

impl std::fmt::Debug for GroupArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupArena")
            .field("share_id", &self.share_id)
            .field("numa_nodes", &self.numa_nodes)
            .finish_non_exhaustive()
    }
}

impl GroupArena {
    /// An empty arena for the shares of party `party_id` (0-based), spread
    /// round-robin over the NUMA nodes with ids `numa_nodes`. With fewer than
    /// two nodes the pages are not bound.
    pub fn new(party_id: usize, numa_nodes: Vec<usize>) -> Self {
        Self {
            segments: RwLock::new(Vec::new()),
            share_id: party_id + 1,
            numa_nodes,
        }
    }

    /// Position in the NUMA node list of the node holding group `group`.
    pub fn node_of_group(&self, group: usize) -> usize {
        (group / SEGMENT_GROUPS) % self.numa_nodes()
    }

    /// Number of NUMA nodes the arena is spread over (at least 1).
    pub fn numa_nodes(&self) -> usize {
        self.numa_nodes.len().max(1)
    }

    /// Write the record at `index`, growing the arena as needed.
    pub fn write(&self, index: usize, iris: &GaloisRingSharedIris) {
        let segment = index / SEGMENT_RECORDS;
        self.grow_to(segment + 1);
        let within = index % SEGMENT_RECORDS;
        let segments = self.segments.read().expect("arena lock poisoned");
        segments[segment].write_group(within / GROUP, |group| {
            write_record(group, within % GROUP, iris)
        });
    }

    /// Append segments until there are at least `count`. Segments are mapped
    /// outside the lock; one that lost a race to another writer is dropped.
    fn grow_to(&self, count: usize) {
        loop {
            let len = self.segments.read().expect("arena lock poisoned").len();
            if len >= count {
                return;
            }
            let node =
                (self.numa_nodes.len() > 1).then(|| self.numa_nodes[len % self.numa_nodes.len()]);
            let segment = Segment::new(node);
            let mut segments = self.segments.write().expect("arena lock poisoned");
            if segments.len() == len {
                segments.push(segment);
            }
        }
    }

    /// Read the record at `index`, or `None` if it was never allocated.
    pub fn read(&self, index: usize) -> Option<GaloisRingSharedIris> {
        let groups = self.groups();
        let group = groups.group(index / GROUP)?;
        Some(read_record(&group, index % GROUP, self.share_id))
    }

    /// Access to the groups for a scan. The arena cannot grow while it is
    /// held, but records can still be written.
    pub fn groups(&self) -> ArenaGroups<'_> {
        ArenaGroups(self.segments.read().expect("arena lock poisoned"))
    }
}

/// Access to the arena's groups; see [`GroupArena::groups`].
pub struct ArenaGroups<'a>(RwLockReadGuard<'a, Vec<Segment>>);

impl ArenaGroups<'_> {
    /// The read-locked bytes of group `group`, or `None` if it was never
    /// allocated.
    pub fn group(&self, group: usize) -> Option<GroupRef<'_>> {
        self.0
            .get(group / SEGMENT_GROUPS)
            .map(|segment| segment.read_group(group % SEGMENT_GROUPS))
    }
}

/// The bytes of one group, read-locked while this is held.
pub struct GroupRef<'a> {
    _guard: RwLockReadGuard<'a, ()>,
    bytes: &'a [u8],
}

impl std::ops::Deref for GroupRef<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::ops::rotation_aware_pairwise_distance_rowmajor_trimmed;
    use aes_prng::AesRng;
    use iris_mpc_common::iris_db::iris::IrisCode;
    use rand::{Rng, SeedableRng};
    use std::sync::Arc;

    fn random_share(rng: &mut AesRng) -> GaloisRingSharedIris {
        let mut iris = GaloisRingSharedIris::default_for_party(1);
        rng.fill(&mut iris.code.coefs[..]);
        rng.fill(&mut iris.mask.coefs[..]);
        iris
    }

    #[test]
    fn records_round_trip_through_the_group_layout() {
        let mut rng = AesRng::seed_from_u64(1);
        let mut group = vec![0_u8; GROUP_BYTES];
        let records = (0..GROUP)
            .map(|_| random_share(&mut rng))
            .collect::<Vec<_>>();
        for (lane, record) in records.iter().enumerate() {
            write_record(&mut group, lane, record);
        }
        for (lane, record) in records.iter().enumerate() {
            assert_eq!(&read_record(&group, lane, 2), record);
        }
    }

    #[test]
    fn arena_round_trips_across_segments() {
        let mut rng = AesRng::seed_from_u64(2);
        let arena = GroupArena::new(1, vec![0, 1]);
        let indices = [
            0,
            15,
            16,
            SEGMENT_RECORDS - 1,
            SEGMENT_RECORDS,
            3 * SEGMENT_RECORDS + 5,
        ];
        let records = indices.map(|_| random_share(&mut rng));
        for (&index, record) in indices.iter().zip(&records) {
            arena.write(index, record);
        }
        for (&index, record) in indices.iter().zip(&records) {
            assert_eq!(arena.read(index).as_ref(), Some(record));
        }
        assert!(arena.read(5 * SEGMENT_RECORDS).is_none());
        assert_eq!(arena.node_of_group(SEGMENT_GROUPS), 1);
    }

    /// The group layout and the shifted query evaluate exactly the production
    /// rotation definition.
    #[test]
    fn layout_matches_rowmajor_kernel() {
        let mut rng = AesRng::seed_from_u64(3);
        let query = {
            let iris = IrisCode::random_rng(&mut rng);
            let shares = GaloisRingSharedIris::generate_shares_locally(&mut rng, iris);
            let mut query = shares[1].clone();
            query.code.preprocess_iris_code_query_share();
            query.mask.preprocess_mask_code_query_share();
            Arc::new(query)
        };
        let records = (0..3)
            .map(|_| Arc::new(random_share(&mut rng)))
            .collect::<Vec<_>>();
        let expected = rotation_aware_pairwise_distance_rowmajor_trimmed::<ROTATIONS, _>(
            &query,
            records.iter().map(Some),
        );
        let packed = AmxQuery::new(&query);
        let mut group = vec![0_u8; GROUP_BYTES];
        let lanes = [0, 7, 15];
        for (&lane, record) in lanes.iter().zip(&records) {
            write_record(&mut group, lane, record);
        }
        for (index, &lane) in lanes.iter().enumerate() {
            let got = scan_lane_reference(&packed, &group, lane);
            for (rotation, pair) in got.iter().enumerate() {
                let base = (index * ROTATIONS + rotation) * 2;
                assert_eq!(
                    pair[0], expected[base].0,
                    "record {index} rotation {rotation}"
                );
                assert_eq!(
                    pair[1],
                    expected[base + 1].0,
                    "record {index} rotation {rotation}"
                );
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    #[test]
    fn amx_kernel_matches_reference() {
        let Some(mut thread) = AmxThread::new() else {
            eprintln!("AMX not available; skipping");
            return;
        };
        let mut rng = AesRng::seed_from_u64(4);
        let query = AmxQuery::new(&random_share(&mut rng));
        let mut group = vec![0_u8; GROUP_BYTES];
        for lane in 0..GROUP {
            write_record(&mut group, lane, &random_share(&mut rng));
        }
        let outputs: [u32; GROUP] = std::array::from_fn(|lane| lane as u32);
        let mut out = vec![0_u16; GROUP * ROTATIONS * 2];
        thread.scan_group(&query, &group, |component, acc| {
            acc.scatter(component, &outputs, &mut out)
        });
        for lane in 0..GROUP {
            let expected = scan_lane_reference(&query, &group, lane);
            for (rotation, pair) in expected.iter().enumerate() {
                let base = (lane * ROTATIONS + rotation) * 2;
                assert_eq!(
                    [out[base], out[base + 1]],
                    *pair,
                    "lane {lane} rotation {rotation}"
                );
            }
        }
    }
}
