//! Vertex buffer codec.
//!
//! Upstream: `vertexcodec.cpp` (`meshopt_encodeVertexBuffer`,
//! `meshopt_encodeVertexBufferBound`, `meshopt_encodeVertexBufferLevel`,
//! `meshopt_decodeVertexBuffer`, `meshopt_decodeVertexVersion`).
//!
//! Scalar kernels only (SRS §6.1); the SIMD decode variants produce
//! byte-identical output upstream, so the wire format is unaffected. Encoders
//! emit wire version 1 only ([ADR-0009](../docs/adr/0009-encoder-version-hardcoded.md));
//! decoders accept v0 and v1. Fixtures pin the scalar reference profile
//! ([ADR-0010](../docs/adr/0010-scalar-profile-fixtures.md)).

mod bitstream;
mod decode;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod decode_simd;
mod encode;

pub use decode::{decode_vertex_buffer, decode_vertex_version};
pub use encode::{encode_vertex_buffer, encode_vertex_buffer_bound, encode_vertex_buffer_level};

pub(crate) const VERTEX_HEADER: u8 = 0xa0;
pub(crate) const DECODE_VERTEX_VERSION: i32 = 1;

pub(crate) const VERTEX_BLOCK_SIZE_BYTES: usize = 8192;
pub(crate) const VERTEX_BLOCK_MAX_SIZE: usize = 256;
pub(crate) const BYTE_GROUP_SIZE: usize = 16;
pub(crate) const BYTE_GROUP_DECODE_LIMIT: usize = 24;
pub(crate) const TAIL_MIN_SIZE_V0: usize = 32;
pub(crate) const TAIL_MIN_SIZE_V1: usize = 24;

#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
pub(crate) const K_BITS_V0: [usize; 4] = [0, 2, 4, 8];
pub(crate) const K_BITS_V1: [usize; 5] = [0, 1, 2, 4, 8];

const ENCODE_DEFAULT_LEVEL: i32 = 2;

pub(crate) fn get_vertex_block_size(vertex_size: usize) -> usize {
    // make sure the entire block fits into the scratch buffer and is aligned to byte group size
    // note: the block size is implicitly part of the format, so we can't change it without breaking compatibility
    let result = (VERTEX_BLOCK_SIZE_BYTES / vertex_size) & !(BYTE_GROUP_SIZE - 1);

    result.min(VERTEX_BLOCK_MAX_SIZE)
}

// C rotates by a variable amount; the masked shift mirrors the C rotation idiom
pub(crate) fn rotate(v: u32, r: u32) -> u32 {
    (v << r) | (v >> ((32 - r) & 31))
}

pub(crate) fn zigzag8(v: u8) -> u8 {
    (0u8.wrapping_sub(v >> 7)) ^ (v << 1)
}

pub(crate) fn zigzag16(v: u16) -> u16 {
    (0u16.wrapping_sub(v >> 15)) ^ (v << 1)
}

#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
pub(crate) fn unzigzag8(v: u8) -> u8 {
    (0u8.wrapping_sub(v & 1)) ^ (v >> 1)
}

#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
pub(crate) fn unzigzag16(v: u16) -> u16 {
    (0u16.wrapping_sub(v & 1)) ^ (v >> 1)
}
