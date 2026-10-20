//! Execution-level differential harness for the wasm32 SIMD128 vertex codec.
//!
//! Built as a cdylib for `wasm32-unknown-unknown` with
//! `RUSTFLAGS="-C target-feature=+simd128"` and driven by the wasmtime CLI
//! (`wasmtime run --invoke run <module>`): the `run` export returns the number
//! of failed differential checks, so CI asserts the printed value is `0`.
//! The case-count pins guard against fixture drift silently emptying the
//! run (a parse break must fail, not pass vacuously).
//!
//! This harness exists because the wasm32 CI job previously only ran
//! `cargo check`; the SIMD128 decode kernel therefore never executed, and the
//! same FIXD data-flow bug that the aarch64 golden run caught in the NEON
//! kernel shipped unnoticed in the wasm kernel too (fixed in `5866293`).

use rmeshopt::{
    decode_vertex_buffer, decode_vertex_version, encode_vertex_buffer_bound,
    encode_vertex_buffer_level,
};

// Fixture pins: committed golden fixtures must keep these exact sizes.
const _VC_DEC_LEN: usize = 211914;
const _VC_HOSTILE_LEN: usize = 86773;
const _VC_ENC_LEN: usize = 209261;

// Case pins: a record-count change means fixture drift, not a free pass.
const GOLDEN_CASES: u32 = 18;
const HOSTILE_CASES: u32 = 229;
const ROUNDTRIP_CASES: u32 = 18;

#[unsafe(no_mangle)]
pub extern "C" fn run() -> u32 {
    golden_failures() + hostile_failures() + roundtrip_failures()
}

#[unsafe(no_mangle)]
pub extern "C" fn case_count() -> u32 {
    GOLDEN_CASES + HOSTILE_CASES + ROUNDTRIP_CASES
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Cursor { data, pos: 0 }
    }

    fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        s
    }

    fn u32(&mut self) -> u32 {
        let mut a = [0u8; 4];
        a.copy_from_slice(self.take(4));
        u32::from_le_bytes(a)
    }

    fn i32(&mut self) -> i32 {
        self.u32() as i32
    }

    fn bytes(&mut self, n: usize) -> &'a [u8] {
        self.take(n)
    }
}

fn counted(cases: u32, expected: u32, fails: u32) -> u32 {
    fails + cases.abs_diff(expected)
}

fn golden_failures() -> u32 {
    let data: &[u8] = include_bytes!("../../../tests/data/vc_dec.bin");
    debug_assert_eq!(data.len(), _VC_DEC_LEN);
    let mut c = Cursor::new(data);
    let mut cases = 0u32;
    let mut fails = 0u32;

    while c.pos < data.len() {
        cases += 1;
        let vertex_count = c.u32() as usize;
        let vertex_size = c.u32() as usize;
        let buffer_size = c.u32() as usize;
        let expected_version = c.i32();
        let stream = c.bytes(buffer_size).to_vec();
        let result = c.i32();
        let expected = if result == 0 {
            c.bytes(vertex_count * vertex_size).to_vec()
        } else {
            Vec::new()
        };

        if decode_vertex_version(&stream) != expected_version {
            fails += 1;
            continue;
        }
        let mut destination = vec![0u8; vertex_count * vertex_size];
        let ok = decode_vertex_buffer(&mut destination, vertex_count, vertex_size, &stream);
        if ok != (result == 0) {
            fails += 1;
            continue;
        }
        if ok && destination != expected {
            fails += 1;
        }
    }
    counted(cases, GOLDEN_CASES, fails)
}

fn hostile_failures() -> u32 {
    let data: &[u8] = include_bytes!("../../../tests/data/vc_hostile.bin");
    debug_assert_eq!(data.len(), _VC_HOSTILE_LEN);
    let mut c = Cursor::new(data);
    let mut cases = 0u32;
    let mut fails = 0u32;

    while c.pos < data.len() {
        cases += 1;
        let buffer_size = c.u32() as usize;
        let vertex_count = c.u32() as usize;
        let vertex_size = c.u32() as usize;
        let stream = c.bytes(buffer_size).to_vec();
        let result = c.i32();
        let expected = if result == 0 {
            c.bytes(vertex_count * vertex_size).to_vec()
        } else {
            Vec::new()
        };

        let mut destination = vec![0u8; vertex_count * vertex_size];
        let ok = decode_vertex_buffer(&mut destination, vertex_count, vertex_size, &stream);
        if ok != (result == 0) {
            fails += 1;
            continue;
        }
        if ok && destination != expected {
            fails += 1;
        }
    }
    counted(cases, HOSTILE_CASES, fails)
}

fn roundtrip_failures() -> u32 {
    let data: &[u8] = include_bytes!("../../../tests/data/vc_enc.bin");
    debug_assert_eq!(data.len(), _VC_ENC_LEN);
    let mut c = Cursor::new(data);
    let mut cases = 0u32;
    let mut fails = 0u32;

    while c.pos < data.len() {
        cases += 1;
        let vertex_count = c.u32() as usize;
        let vertex_size = c.u32() as usize;
        let level = c.i32();
        let vertices = c.bytes(vertex_count * vertex_size).to_vec();
        let encoded_size = c.u32() as usize;
        c.bytes(encoded_size);

        let mut buffer = vec![0u8; encode_vertex_buffer_bound(vertex_count, vertex_size)];
        let size =
            encode_vertex_buffer_level(&mut buffer, &vertices, vertex_count, vertex_size, level);
        if size == 0 {
            fails += 1;
            continue;
        }
        let mut decoded = vec![0u8; vertex_count * vertex_size];
        let ok = decode_vertex_buffer(
            &mut decoded,
            vertex_count,
            vertex_size,
            &buffer[..size],
        );
        if !ok || decoded != vertices {
            fails += 1;
        }
    }
    counted(cases, ROUNDTRIP_CASES, fails)
}
