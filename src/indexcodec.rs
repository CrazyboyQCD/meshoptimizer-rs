//! Index buffer and index sequence codecs.
//!
//! Upstream: `indexcodec.cpp` (`meshopt_encodeIndexBuffer`,
//! `meshopt_encodeIndexBufferBound`, `meshopt_decodeIndexBuffer`,
//! `meshopt_decodeIndexVersion`, `meshopt_encodeIndexSequence`,
//! `meshopt_encodeIndexSequenceBound`, `meshopt_decodeIndexSequence`).
//!
//! Encoders emit wire version 1 only ([ADR-0009](../docs/adr/0009-encoder-version-hardcoded.md));
//! decoders accept every version upstream produces (v0 and v1).

mod decode;
mod encode;

pub use decode::{decode_index_buffer, decode_index_sequence, decode_index_version};
pub use encode::{
    encode_index_buffer, encode_index_buffer_bound, encode_index_sequence,
    encode_index_sequence_bound,
};

const INDEX_HEADER: u8 = 0xe0;
const SEQUENCE_HEADER: u8 = 0xd0;
const DECODE_INDEX_VERSION: i32 = 1;
