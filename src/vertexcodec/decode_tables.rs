//! Shared SIMD vertex-decode tables (upstream `kDecodeBytesGroupShuffle`,
//! `kDecodeBytesGroupCount`, `lanes`), used by the SSE, NEON and WASM kernels.
//! Upstream builds these at static-init time (`decodeBytesGroupBuildTables`);
//! const-evaluated tables are behavior-identical (remesher precedent).

// 0 for 1-bit, 1 for 2-bit, 2 for 4-bit, 3 for 8-bit, and 4 for 0-bit as it
// makes some of the uses easier (only the x86 kernels consult this)
#[cfg(target_arch = "x86_64")]
pub(super) const HBTN: [u32; 9] = [4, 1, 2, 3, 4, 0, 1, 2, 3];

// mask out one bit per group that is set if all group bits were 1; only the
// x86 kernels consult this (NEON/WASM derive the count inline or via table)
#[cfg(target_arch = "x86_64")]
pub(super) const LANES: [u64; 9] = [
    0,
    0x5555_5555,
    0x1111_1111_1111_1111,
    0,
    0,
    0xffff,
    0x5555_5555,
    0x1111_1111_1111_1111,
    0,
];

#[cfg(not(all(target_arch = "x86_64", feature = "simd-avx512")))]
const fn build_group_tables() -> ([[u8; 8]; 256], [u8; 256]) {
    let mut shuffle = [[0u8; 8]; 256];
    let mut count = [0u8; 256];
    let mut mask = 0usize;
    while mask < 256 {
        let mut sh = [0u8; 8];
        let mut cnt = 0u8;
        let mut i = 0usize;
        while i < 8 {
            let maski = ((mask >> i) & 1) as u8;
            sh[i] = if maski != 0 { cnt } else { 0x80 };
            cnt += maski;
            i += 1;
        }
        shuffle[mask] = sh;
        count[mask] = cnt;
        mask += 1;
    }
    (shuffle, count)
}

// the x86 AVX512 kernel derives the shuffle mask in-register and consumes
// neither table; every other kernel uses at least the shuffle half
#[cfg(not(all(target_arch = "x86_64", feature = "simd-avx512")))]
static GROUP_TABLES: ([[u8; 8]; 256], [u8; 256]) = build_group_tables();

// the x86 AVX512 kernel derives the shuffle mask in-register and never reads
// this table; every other kernel does
#[cfg(not(all(target_arch = "x86_64", feature = "simd-avx512")))]
pub(super) static GROUP_SHUFFLE: [[u8; 8]; 256] = GROUP_TABLES.0;

// only the NEON/WASM kernels read the count table; the SSE kernel derives the
// count in-register via psadbw
#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
pub(super) static GROUP_COUNT: [u8; 256] = GROUP_TABLES.1;
