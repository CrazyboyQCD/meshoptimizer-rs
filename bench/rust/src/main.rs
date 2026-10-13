//! Rust-side benchmark driver for the speed/memory comparison against upstream
//! meshoptimizer (docs/adr/0011). Standalone binary crate; mirrors bench/cpp/driver.cpp
//! op-for-op so both sides measure bit-identical inputs (verified via corpus pins) and
//! hash outputs the same way.
//!
//! Modes (mode <pins.txt> <arg> [op...]):
//!   gen    <pins.txt> <expected-dir>   generate pins (+ expected blobs for float-tolerant ops)
//!   verify <pins.txt> <expected-dir>   run each op once, check input/output pins
//!   speed  <pins.txt> <out.json>       timing loop, allocation counting disabled
//!   mem    <pins.txt> <out.json>       single call per op, allocation counting enabled
//!
//! The counting GlobalAlloc below is the one place this harness touches `unsafe`
//! (implementing the GlobalAlloc trait requires it); the library itself remains
//! `#![forbid(unsafe_code)]`.

use meshoptimizer_rs as meshopt;

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

// --- allocation counting ------------------------------------------------------

static TRACKING: AtomicBool = AtomicBool::new(false);
static ALLOC_CUR: AtomicU64 = AtomicU64::new(0);
static ALLOC_PEAK: AtomicU64 = AtomicU64::new(0);
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);

struct CountingAlloc;

// Harness-only `unsafe`: required by the GlobalAlloc trait surface; the wrapper
// forwards to `System` unchanged, counters are updated on the side.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if TRACKING.load(Ordering::Relaxed) && !ptr.is_null() {
            let size = layout.size() as u64;
            let cur = ALLOC_CUR.fetch_add(size, Ordering::Relaxed) + size;
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_PEAK.fetch_max(cur, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if TRACKING.load(Ordering::Relaxed) {
            ALLOC_CUR.fetch_sub(layout.size() as u64, Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAlloc = CountingAlloc;

fn alloc_reset() {
    ALLOC_CUR.store(0, Ordering::Relaxed);
    ALLOC_PEAK.store(0, Ordering::Relaxed);
    ALLOC_COUNT.store(0, Ordering::Relaxed);
}

// --- deterministic input generation (byte-parity with bench/cpp/driver.cpp) ----

static LCG_STATE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0x12345678);

fn lcg_reset() {
    LCG_STATE.store(0x12345678, Ordering::Relaxed);
}

fn lcg() -> u32 {
    let s = LCG_STATE.load(Ordering::Relaxed);
    let next = s.wrapping_mul(1664525).wrapping_add(1013904223);
    LCG_STATE.store(next, Ordering::Relaxed);
    next
}

fn lcg_bytes(n: usize) -> Vec<u8> {
    lcg_reset();
    (0..n).map(|_| (lcg() & 0xff) as u8).collect()
}

const LGW: u32 = 316; // large tier: 99856 vertices
const LGH: u32 = 316;
const MGW: u32 = 128; // mid tier: 16384 vertices
const MGH: u32 = 128;

struct Grid {
    v: usize,
    ic: usize,
    pos: Vec<f32>,
    idx: Vec<u32>,
}

fn make_grid(gw: u32, gh: u32, hole: u32) -> Grid {
    lcg_reset();
    let v = (gw * gh) as usize;
    let mut pos = vec![0f32; v * 3];
    for (i, chunk) in pos.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let x = i as u32 % gw;
        let y = i as u32 / gw;
        chunk[0] = x as f32 + (lcg() % 8) as f32 / 32.0;
        chunk[1] = y as f32 + (lcg() % 8) as f32 / 32.0;
        chunk[2] = ((i as u32 * 7 + y * 13) % 5) as f32 / 4.0;
    }
    let mut idx = Vec::new();
    for y in 0..gh - 1 {
        for x in 0..gw - 1 {
            if hole == 1 && x == gw / 2 {
                continue;
            }
            if hole == 2 && x == gw / 4 {
                continue;
            }
            let v0 = y * gw + x;
            let (v1, v2, v3) = (v0 + 1, v0 + gw, v0 + gw + 1);
            idx.extend_from_slice(&[v0, v1, v2, v1, v3, v2]);
        }
    }
    Grid {
        v,
        ic: idx.len(),
        pos,
        idx,
    }
}

fn smooth_attrs(gw: u32, gh: u32, count: usize) -> Vec<f32> {
    let v = (gw * gh) as usize;
    let mut a = vec![0f32; v * count];
    for i in 0..v {
        for k in 0..count {
            a[i * count + k] =
                ((i as u32 % gw + (i as u32 / gw) * 2) as f32 / 16.0) + (k as f32 / 8.0);
        }
    }
    a
}

fn grid_uvs(gw: u32, gh: u32) -> Vec<f32> {
    let v = (gw * gh) as usize;
    let mut t = vec![0f32; v * 2];
    for i in 0..v {
        t[i * 2] = (i as u32 % gw) as f32 / (gw - 1) as f32;
        t[i * 2 + 1] = (i as u32 / gw) as f32 / (gh - 1) as f32;
    }
    t
}

fn const_normals(v: usize) -> Vec<f32> {
    let mut n = vec![0f32; v * 3];
    for chunk in n.as_chunks_mut::<3>().0 {
        chunk[2] = 1.0;
    }
    n
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn die(msg: &str) -> ! {
    eprintln!("{}", msg);
    std::process::exit(1);
}

// --- hashing (byte-parity: lengths as u64 LE, FNV-1a 64) ------------------------

#[derive(Default)]
struct Fnv {
    h: u64,
}

impl Fnv {
    fn new() -> Self {
        Fnv {
            h: 0xcbf29ce484222325,
        }
    }
    fn bytes(&mut self, data: &[u8]) {
        for &b in data {
            self.h ^= b as u64;
            self.h = self.h.wrapping_mul(0x100000001b3);
        }
    }
    fn buf(&mut self, data: &[u8]) {
        self.bytes(&(data.len() as u64).to_le_bytes());
        self.bytes(data);
    }
    fn u64v(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
}

// --- benchmark op table ----------------------------------------------------------

struct Prep {
    pos: Vec<f32>,
    posb: Vec<u8>,
    idx: Vec<u32>,
    v: usize,
    ic: usize,
    attrs: Vec<f32>,
    uvs: Vec<f32>,
    nrm: Vec<f32>,
    bytes: Vec<u8>,
    enc: Vec<u8>,
    outb: Vec<u8>,
    outi: Vec<u32>,
    remap: Vec<u32>,
    counts: Vec<u32>,
    part: Vec<u32>,
    work_i: Vec<u32>,
    work_b: Vec<u8>,
    outf: Vec<f32>,
    meshlets: Vec<meshopt::Meshlet>,
    mverts: Vec<u32>,
    mtris: Vec<u8>,
    ret: usize,
    ret2: usize,
    target: usize,
    total: usize,
    cc: usize,
    error: f32,
    trace: Vec<Vec<u8>>,
    in_hash: u64,
}

impl Prep {
    fn new() -> Self {
        Prep {
            pos: Vec::new(),
            posb: Vec::new(),
            idx: Vec::new(),
            v: 0,
            ic: 0,
            attrs: Vec::new(),
            uvs: Vec::new(),
            nrm: Vec::new(),
            bytes: Vec::new(),
            enc: Vec::new(),
            outb: Vec::new(),
            outi: Vec::new(),
            remap: Vec::new(),
            counts: Vec::new(),
            part: Vec::new(),
            work_i: Vec::new(),
            work_b: Vec::new(),
            outf: Vec::new(),
            meshlets: Vec::new(),
            mverts: Vec::new(),
            mtris: Vec::new(),
            ret: 0,
            ret2: 0,
            target: 0,
            total: 0,
            cc: 0,
            error: 0.0,
            trace: Vec::new(),
            in_hash: 0,
        }
    }

    fn trace_bytes(&mut self, data: Vec<u8>) {
        self.trace.push(data);
    }

    fn seal_inputs(&mut self) {
        let mut h = Fnv::new();
        for chunk in &self.trace {
            h.buf(chunk);
        }
        self.in_hash = h.h;
        self.trace.clear();
    }

    fn load_grid(&mut self, hole: u32) {
        let g = make_grid(LGW, LGH, hole);
        self.posb = f32_bytes(&g.pos);
        self.pos = g.pos;
        self.idx = g.idx;
        self.v = g.v;
        self.ic = g.ic;
        self.trace_bytes(self.posb.clone());
        let mut ib = Vec::with_capacity(self.idx.len() * 4);
        for &i in &self.idx {
            ib.extend_from_slice(&i.to_le_bytes());
        }
        self.trace_bytes(ib);
    }

    fn load_grid_m(&mut self) {
        let g = make_grid(MGW, MGH, 0);
        self.posb = f32_bytes(&g.pos);
        self.pos = g.pos;
        self.idx = g.idx;
        self.v = g.v;
        self.ic = g.ic;
        self.trace_bytes(self.posb.clone());
        let mut ib = Vec::with_capacity(self.idx.len() * 4);
        for &i in &self.idx {
            ib.extend_from_slice(&i.to_le_bytes());
        }
        self.trace_bytes(ib);
    }

    fn make_box(&mut self, sub: u32, extent: f32) {
        const FACE_N: [[f32; 3]; 6] = [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
        const FACE_U: [[f32; 3]; 6] = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ];
        const FACE_V: [[f32; 3]; 6] = [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ];
        let vs = sub + 1;
        let h = extent / 2.0;
        self.pos.clear();
        self.idx.clear();
        for f in 0..6usize {
            let base = (self.pos.len() / 3) as u32;
            for b in 0..vs {
                for a in 0..vs {
                    let fa = a as f32 / sub as f32 * 2.0 - 1.0;
                    let fb = b as f32 / sub as f32 * 2.0 - 1.0;
                    for k in 0..3usize {
                        self.pos.push((FACE_N[f][k] + FACE_U[f][k] * fa + FACE_V[f][k] * fb) * h);
                    }
                }
            }
            for b in 0..vs - 1 {
                for a in 0..vs - 1 {
                    let v00 = base + b * vs + a;
                    let (v10, v01, v11) = (v00 + 1, v00 + vs, v00 + vs + 1);
                    self.idx.extend_from_slice(&[v00, v10, v01, v10, v11, v01]);
                }
            }
        }
        self.v = self.pos.len() / 3;
        self.ic = self.idx.len();
    }
}

struct Op {
    name: &'static str,
    tol: bool,
    prepare: fn(&mut Prep),
    once: fn(&mut Prep) -> f64,
    hash_out: fn(&mut Prep, &mut Fnv, Option<&mut Vec<u8>>),
}

fn now_ms() -> f64 {
    Instant::now().elapsed().as_secs_f64() * 1000.0
}

// --- codec ops (tier L) -----------------------------------------------------------

fn prep_vc_encode(p: &mut Prep) {
    let g = make_grid(LGW, LGH, 0);
    p.posb = f32_bytes(&g.pos);
    p.pos = g.pos;
    p.v = g.v;
    p.trace_bytes(p.posb.clone());
    p.outb = vec![0u8; meshopt::encode_vertex_buffer_bound(p.v, 12)];
}

fn once_vc_encode(p: &mut Prep) -> f64 {
    let t0 = now_ms();
    p.ret = meshopt::encode_vertex_buffer(&mut p.outb, &p.posb, p.v, 12);
    now_ms() - t0
}

fn hash_vc_encode(p: &mut Prep, h: &mut Fnv, blob: Option<&mut Vec<u8>>) {
    h.buf(&p.outb[..p.ret]);
    if let Some(blob) = blob {
        *blob = p.outb[..p.ret].to_vec();
    }
}

fn prep_vc_decode(p: &mut Prep) {
    let g = make_grid(LGW, LGH, 0);
    p.posb = f32_bytes(&g.pos);
    p.v = g.v;
    let mut buf = vec![0u8; meshopt::encode_vertex_buffer_bound(p.v, 12)];
    let n = meshopt::encode_vertex_buffer(&mut buf, &p.posb, p.v, 12);
    buf.truncate(n);
    p.trace_bytes(buf.clone());
    p.enc = buf;
    p.outb = vec![0u8; p.v * 12];
}

fn once_vc_decode(p: &mut Prep) -> f64 {
    let t0 = now_ms();
    let ok = meshopt::decode_vertex_buffer(&mut p.outb, p.v, 12, &p.enc);
    let dt = now_ms() - t0;
    if !ok {
        die("op vc_decode failed");
    }
    dt
}

fn hash_vc_decode(p: &mut Prep, h: &mut Fnv, blob: Option<&mut Vec<u8>>) {
    h.buf(&p.outb);
    if let Some(blob) = blob {
        *blob = p.outb.clone();
    }
}

fn prep_ic_encode(p: &mut Prep) {
    let g = make_grid(LGW, LGH, 0);
    p.posb = f32_bytes(&g.pos);
    p.idx = g.idx;
    p.v = g.v;
    p.ic = g.ic;
    p.trace_bytes(p.posb.clone());
    let mut ib = Vec::with_capacity(p.idx.len() * 4);
    for &i in &p.idx {
        ib.extend_from_slice(&i.to_le_bytes());
    }
    p.trace_bytes(ib);
    p.outb = vec![0u8; meshopt::encode_index_buffer_bound(p.ic, p.v)];
}

fn once_ic_encode(p: &mut Prep) -> f64 {
    let t0 = now_ms();
    p.ret = meshopt::encode_index_buffer(&mut p.outb, &p.idx);
    now_ms() - t0
}

fn hash_ic_encode(p: &mut Prep, h: &mut Fnv, blob: Option<&mut Vec<u8>>) {
    h.buf(&p.outb[..p.ret]);
    if let Some(blob) = blob {
        *blob = p.outb[..p.ret].to_vec();
    }
}

fn prep_ic_decode(p: &mut Prep) {
    let g = make_grid(LGW, LGH, 0);
    p.idx = g.idx;
    p.v = g.v;
    p.ic = g.ic;
    let mut buf = vec![0u8; meshopt::encode_index_buffer_bound(p.ic, p.v)];
    let n = meshopt::encode_index_buffer(&mut buf, &p.idx);
    buf.truncate(n);
    p.trace_bytes(buf.clone());
    p.enc = buf;
    p.outb = vec![0u8; p.ic * 4];
}

fn once_ic_decode(p: &mut Prep) -> f64 {
    let t0 = now_ms();
    let ok = meshopt::decode_index_buffer(&mut p.outb, p.ic, 4, &p.enc);
    let dt = now_ms() - t0;
    if !ok {
        die("op ic_decode failed");
    }
    dt
}

fn hash_ic_decode(p: &mut Prep, h: &mut Fnv, blob: Option<&mut Vec<u8>>) {
    h.buf(&p.outb);
    if let Some(blob) = blob {
        *blob = p.outb.clone();
    }
}

// --- filter ops (tier L, in-place: every call restores the pristine input first,
// --- with the restore copy outside the measured window) -----------------------------

fn prep_vf_oct(p: &mut Prep) {
    p.work_b = lcg_bytes((LGW * LGH) as usize * 8);
    p.bytes = vec![0u8; p.work_b.len()];
    p.v = (LGW * LGH) as usize;
    p.trace_bytes(p.work_b.clone());
}

fn once_vf_oct(p: &mut Prep) -> f64 {
    p.bytes.copy_from_slice(&p.work_b);
    let t0 = now_ms();
    meshopt::decode_filter_oct(&mut p.bytes, p.v, 8);
    now_ms() - t0
}

fn hash_vf_oct(p: &mut Prep, h: &mut Fnv, blob: Option<&mut Vec<u8>>) {
    h.buf(&p.bytes);
    if let Some(blob) = blob {
        *blob = p.bytes.clone();
    }
}

fn prep_vf_quat(p: &mut Prep) {
    p.work_b = lcg_bytes((LGW * LGH) as usize * 8);
    p.bytes = vec![0u8; p.work_b.len()];
    p.v = (LGW * LGH) as usize;
    p.trace_bytes(p.work_b.clone());
}

fn once_vf_quat(p: &mut Prep) -> f64 {
    p.bytes.copy_from_slice(&p.work_b);
    let t0 = now_ms();
    meshopt::decode_filter_quat(&mut p.bytes, p.v, 8);
    now_ms() - t0
}

fn prep_vf_exp(p: &mut Prep) {
    // same byte stream as the C++ lcg_words generator: LE bytes of each u32 sample
    lcg_reset();
    p.work_b = Vec::with_capacity((LGW * LGH) as usize * 12);
    for _ in 0..(LGW * LGH) as usize * 3 {
        p.work_b.extend_from_slice(&lcg().to_le_bytes());
    }
    p.bytes = vec![0u8; p.work_b.len()];
    p.v = (LGW * LGH) as usize;
    p.trace_bytes(p.work_b.clone());
}

fn once_vf_exp(p: &mut Prep) -> f64 {
    p.bytes.copy_from_slice(&p.work_b);
    let t0 = now_ms();
    meshopt::decode_filter_exp(&mut p.bytes, p.v, 12);
    now_ms() - t0
}

fn hash_vf_exp(p: &mut Prep, h: &mut Fnv, blob: Option<&mut Vec<u8>>) {
    h.buf(&p.bytes);
    if let Some(blob) = blob {
        *blob = p.bytes.clone();
    }
}

// --- geometry ops (tier L) ---------------------------------------------------------

fn prep_ig_remap(p: &mut Prep) {
    p.load_grid(0);
    p.remap = vec![0u32; p.v];
}

fn once_ig_remap(p: &mut Prep) -> f64 {
    let t0 = now_ms();
    p.ret = meshopt::generate_vertex_remap(&mut p.remap, Some(&p.idx), &p.posb, p.v, 12);
    now_ms() - t0
}

fn hash_remap(p: &mut Prep, h: &mut Fnv, blob: Option<&mut Vec<u8>>) {
    let mut ib = Vec::with_capacity(p.remap.len() * 4);
    for &i in &p.remap {
        ib.extend_from_slice(&i.to_le_bytes());
    }
    h.buf(&ib);
    let _ = blob;
}

fn prep_vcache(p: &mut Prep) {
    p.load_grid(0);
    p.outi = vec![0u32; p.ic];
}

fn once_vcache(p: &mut Prep) -> f64 {
    let t0 = now_ms();
    meshopt::optimize_vertex_cache(&mut p.outi, &p.idx, p.v);
    now_ms() - t0
}

fn hash_out_i(p: &mut Prep, h: &mut Fnv, blob: Option<&mut Vec<u8>>) {
    let n = if p.ret != 0 { p.ret } else { p.ic };
    let mut ib = Vec::with_capacity(n * 4);
    for &i in &p.outi[..n] {
        ib.extend_from_slice(&i.to_le_bytes());
    }
    h.buf(&ib);
    let _ = blob;
}

fn once_overdraw(p: &mut Prep) -> f64 {
    let t0 = now_ms();
    meshopt::optimize_overdraw(&mut p.outi, &p.idx, &p.pos, p.v, 12, 1.05);
    now_ms() - t0
}

fn prep_vfetch(p: &mut Prep) {
    p.load_grid(0);
    p.work_i = p.idx.clone();
    p.outb = vec![0u8; p.v * 12];
}

fn once_vfetch(p: &mut Prep) -> f64 {
    p.work_i.copy_from_slice(&p.idx);
    let t0 = now_ms();
    p.ret = meshopt::optimize_vertex_fetch(&mut p.outb, &mut p.work_i, &p.posb, p.v, 12);
    now_ms() - t0
}

fn hash_vfetch(p: &mut Prep, h: &mut Fnv, blob: Option<&mut Vec<u8>>) {
    h.buf(&p.outb);
    let mut ib = Vec::with_capacity(p.work_i.len() * 4);
    for &i in &p.work_i {
        ib.extend_from_slice(&i.to_le_bytes());
    }
    h.buf(&ib);
    let _ = blob;
}

fn once_stripify(p: &mut Prep) -> f64 {
    let t0 = now_ms();
    p.ret = meshopt::stripify(&mut p.outi, &p.idx, p.v, 0xffff);
    now_ms() - t0
}

fn prep_simplify(p: &mut Prep) {
    p.load_grid(0);
    p.target = p.ic / 3 / 2 * 3;
    p.outi = vec![0u32; p.ic];
}

fn once_simplify(p: &mut Prep) -> f64 {
    let t0 = now_ms();
    p.ret = meshopt::simplify(
        &mut p.outi,
        &p.idx,
        &p.pos,
        p.v,
        12,
        p.target,
        1e-2,
        0,
        Some(&mut p.error),
    );
    now_ms() - t0
}

fn hash_simplify(p: &mut Prep, h: &mut Fnv, blob: Option<&mut Vec<u8>>) {
    let mut ib = Vec::with_capacity(p.ret * 4);
    for &i in &p.outi[..p.ret] {
        ib.extend_from_slice(&i.to_le_bytes());
    }
    h.buf(&ib);
    h.bytes(&p.error.to_le_bytes());
    let _ = blob;
}

fn prep_simplify_attrs(p: &mut Prep) {
    p.load_grid(0);
    p.attrs = smooth_attrs(LGW, LGH, 2);
    p.trace_bytes(f32_bytes(&p.attrs));
    p.target = p.ic / 3 / 2 * 3;
    p.outi = vec![0u32; p.ic];
}

fn once_simplify_attrs(p: &mut Prep) -> f64 {
    let weights = [1.0f32, 1.0];
    let t0 = now_ms();
    p.ret = meshopt::simplify_with_attributes(
        &mut p.outi,
        &p.idx,
        &p.pos,
        p.v,
        12,
        &p.attrs,
        8,
        &weights,
        None,
        p.target,
        1e-2,
        0,
        Some(&mut p.error),
    );
    now_ms() - t0
}

fn once_simplify_sloppy(p: &mut Prep) -> f64 {
    let t0 = now_ms();
    p.ret = meshopt::simplify_sloppy(
        &mut p.outi,
        &p.idx,
        &p.pos,
        p.v,
        12,
        None,
        p.target,
        1.0,
        Some(&mut p.error),
    );
    now_ms() - t0
}

fn prep_prune(p: &mut Prep) {
    p.load_grid(2);
    p.outi = vec![0u32; p.ic];
}

fn once_prune(p: &mut Prep) -> f64 {
    let t0 = now_ms();
    p.ret = meshopt::simplify_prune(&mut p.outi, &p.idx, &p.pos, p.v, 12, 1.0);
    now_ms() - t0
}

fn prep_points(p: &mut Prep) {
    let g = make_grid(LGW, LGH, 0);
    p.posb = f32_bytes(&g.pos);
    p.pos = g.pos;
    p.v = g.v;
    p.trace_bytes(p.posb.clone());
    p.outi = vec![0u32; p.v];
}

fn once_points(p: &mut Prep) -> f64 {
    let t0 = now_ms();
    p.ret = meshopt::simplify_points(&mut p.outi, &p.pos, p.v, 12, None, 0, 0.0, p.v / 4);
    now_ms() - t0
}

fn once_spatial(p: &mut Prep) -> f64 {
    let t0 = now_ms();
    meshopt::spatial_sort_remap(&mut p.remap, &p.pos, 12);
    now_ms() - t0
}

fn prep_clusterize(p: &mut Prep) {
    p.load_grid(0);
    let bound = meshopt::build_meshlets_bound(p.ic, 64, 124);
    p.meshlets = vec![
        meshopt::Meshlet {
            vertex_offset: 0,
            triangle_offset: 0,
            vertex_count: 0,
            triangle_count: 0,
        };
        bound
    ];
    p.mverts = vec![0u32; p.ic];
    p.mtris = vec![0u8; p.ic];
}

fn once_clusterize(p: &mut Prep) -> f64 {
    let t0 = now_ms();
    p.ret = meshopt::build_meshlets(
        &mut p.meshlets,
        &mut p.mverts,
        &mut p.mtris,
        &p.idx,
        &p.pos,
        p.v,
        12,
        64,
        124,
        0.0,
    );
    now_ms() - t0
}

fn hash_clusterize(p: &mut Prep, h: &mut Fnv, blob: Option<&mut Vec<u8>>) {
    h.u64v(p.ret as u64);
    for m in &p.meshlets[..p.ret] {
        h.u64v(m.vertex_offset as u64);
        h.u64v(m.triangle_offset as u64);
        h.u64v(m.vertex_count as u64);
        h.u64v(m.triangle_count as u64);
    }
    let mut vb = Vec::with_capacity(p.ic * 4);
    for &i in &p.mverts {
        vb.extend_from_slice(&i.to_le_bytes());
    }
    h.buf(&vb);
    h.buf(&p.mtris);
    let _ = blob;
}

fn prep_partition(p: &mut Prep) {
    p.load_grid(0);
    let chunk = 124usize * 3;
    p.cc = p.ic / chunk;
    p.total = p.cc * chunk;
    p.counts = vec![chunk as u32; p.cc];
    p.part = vec![0u32; p.cc];
    let mut cb = Vec::with_capacity(p.cc * 4);
    for &c in &p.counts {
        cb.extend_from_slice(&c.to_le_bytes());
    }
    p.trace_bytes(cb);
}

fn once_partition(p: &mut Prep) -> f64 {
    let t0 = now_ms();
    p.ret = meshopt::partition_clusters(
        &mut p.part,
        &p.idx[..p.total],
        &p.counts,
        Some(&p.pos),
        p.v,
        12,
        8,
    );
    now_ms() - t0
}

fn hash_partition(p: &mut Prep, h: &mut Fnv, blob: Option<&mut Vec<u8>>) {
    let mut pb = Vec::with_capacity(p.cc * 4);
    for &x in &p.part {
        pb.extend_from_slice(&x.to_le_bytes());
    }
    h.buf(&pb);
    h.u64v(p.ret as u64);
    let _ = blob;
}

// --- mid-tier ops ----------------------------------------------------------------

fn prep_tangents(p: &mut Prep) {
    p.load_grid_m();
    p.uvs = grid_uvs(MGW, MGH);
    p.nrm = const_normals(p.v);
    p.trace_bytes(f32_bytes(&p.uvs));
    p.trace_bytes(f32_bytes(&p.nrm));
    p.outf = vec![0f32; p.ic * 4]; // per-corner output (index_count*4, meshoptimizer.h L949)
}

fn once_tangents(p: &mut Prep) -> f64 {
    let t0 = now_ms();
    meshopt::generate_tangents(
        &mut p.outf,
        Some(&p.idx),
        p.ic,
        &p.pos,
        p.v,
        12,
        &p.nrm,
        12,
        &p.uvs,
        8,
        0,
    );
    now_ms() - t0
}

fn hash_out_f(p: &mut Prep, h: &mut Fnv, blob: Option<&mut Vec<u8>>) {
    let fb = f32_bytes(&p.outf);
    h.buf(&fb);
    if let Some(blob) = blob {
        *blob = fb;
    }
}

fn prep_normals(p: &mut Prep) {
    p.load_grid_m();
    p.outf = vec![0f32; p.ic * 3]; // per-corner output (index_count*3, meshoptimizer.h L964)
}

fn once_normals(p: &mut Prep) -> f64 {
    let t0 = now_ms();
    meshopt::generate_normals(&mut p.outf, Some(&p.idx), p.ic, &p.pos, p.v, 12, core::f32::consts::PI, 0.0);
    now_ms() - t0
}

fn prep_remesh(p: &mut Prep) {
    p.make_box(32, 32.0);
    p.trace_bytes(f32_bytes(&p.pos));
    let mut ib = Vec::with_capacity(p.idx.len() * 4);
    for &i in &p.idx {
        ib.extend_from_slice(&i.to_le_bytes());
    }
    p.trace_bytes(ib);
    p.ret = meshopt::remesh(None, &p.idx, &p.pos, p.v, 12, 64, 0);
    p.outf = vec![0f32; p.ret * 9];
}

fn once_remesh(p: &mut Prep) -> f64 {
    let t0 = now_ms();
    let n = meshopt::remesh(None, &p.idx, &p.pos, p.v, 12, 64, 0);
    if n > p.ret {
        die(&format!("op remesh count grew: {} > {}", n, p.ret));
    }
    p.ret2 = meshopt::remesh(Some(&mut p.outf), &p.idx, &p.pos, p.v, 12, 64, 0);
    now_ms() - t0
}

fn hash_remesh(p: &mut Prep, h: &mut Fnv, blob: Option<&mut Vec<u8>>) {
    h.u64v(p.ret2 as u64);
    let fb = f32_bytes(&p.outf[..p.ret2 * 9]);
    h.buf(&fb);
    if let Some(blob) = blob {
        *blob = (p.ret2 as u64).to_le_bytes().into_iter().collect();
        blob.extend_from_slice(&fb);
    }
}

static OPS: &[Op] = &[
    Op { name: "vc_encode", tol: false, prepare: prep_vc_encode, once: once_vc_encode, hash_out: hash_vc_encode },
    Op { name: "vc_decode", tol: false, prepare: prep_vc_decode, once: once_vc_decode, hash_out: hash_vc_decode },
    Op { name: "ic_encode", tol: false, prepare: prep_ic_encode, once: once_ic_encode, hash_out: hash_ic_encode },
    Op { name: "ic_decode", tol: false, prepare: prep_ic_decode, once: once_ic_decode, hash_out: hash_ic_decode },
    Op { name: "vf_oct", tol: false, prepare: prep_vf_oct, once: once_vf_oct, hash_out: hash_vf_oct },
    Op { name: "vf_quat", tol: false, prepare: prep_vf_quat, once: once_vf_quat, hash_out: hash_vf_oct },
    Op { name: "vf_exp", tol: false, prepare: prep_vf_exp, once: once_vf_exp, hash_out: hash_vf_exp },
    Op { name: "ig_remap", tol: false, prepare: prep_ig_remap, once: once_ig_remap, hash_out: hash_remap },
    Op { name: "vcache", tol: false, prepare: prep_vcache, once: once_vcache, hash_out: hash_out_i },
    Op { name: "overdraw", tol: false, prepare: prep_vcache, once: once_overdraw, hash_out: hash_out_i },
    Op { name: "vfetch", tol: false, prepare: prep_vfetch, once: once_vfetch, hash_out: hash_vfetch },
    Op { name: "stripify", tol: false, prepare: prep_vcache, once: once_stripify, hash_out: hash_out_i },
    Op { name: "simplify", tol: false, prepare: prep_simplify, once: once_simplify, hash_out: hash_simplify },
    Op { name: "simplify_attrs", tol: false, prepare: prep_simplify_attrs, once: once_simplify_attrs, hash_out: hash_simplify },
    Op { name: "simplify_sloppy", tol: false, prepare: prep_simplify, once: once_simplify_sloppy, hash_out: hash_simplify },
    Op { name: "prune", tol: false, prepare: prep_prune, once: once_prune, hash_out: hash_out_i },
    Op { name: "points", tol: false, prepare: prep_points, once: once_points, hash_out: hash_out_i },
    Op { name: "spatial", tol: false, prepare: prep_ig_remap, once: once_spatial, hash_out: hash_remap },
    Op { name: "clusterize", tol: false, prepare: prep_clusterize, once: once_clusterize, hash_out: hash_clusterize },
    Op { name: "partition", tol: false, prepare: prep_partition, once: once_partition, hash_out: hash_partition },
    Op { name: "tangents", tol: true, prepare: prep_tangents, once: once_tangents, hash_out: hash_out_f },
    Op { name: "normals", tol: true, prepare: prep_normals, once: once_normals, hash_out: hash_out_f },
    Op { name: "remesh", tol: true, prepare: prep_remesh, once: once_remesh, hash_out: hash_remesh },
];

fn select_ops(names: &[String]) -> Vec<&'static Op> {
    if names.is_empty() {
        return OPS.iter().collect();
    }
    names
        .iter()
        .map(|n| {
            OPS.iter()
                .find(|op| op.name == n.as_str())
                .unwrap_or_else(|| die(&format!("unknown op: {}", n)))
        })
        .collect()
}

// --- pins ------------------------------------------------------------------------

fn load_pins(path: &str) -> HashMap<String, u64> {
    let data = fs::read_to_string(path).unwrap_or_else(|_| die(&format!("cannot load pins: {}", path)));
    let mut pins = HashMap::new();
    for line in data.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split('|').collect();
        if parts.len() != 3 {
            continue;
        }
        let v = u64::from_str_radix(parts[2], 16)
            .unwrap_or_else(|_| die(&format!("bad pin line: {}", line)));
        pins.insert(format!("{}|{}", parts[0], parts[1]), v);
    }
    pins
}

// --- float-tolerant comparison (<=1 ULP, tdd.md section 7) -------------------------

fn ordered_bits(bits: u32) -> i64 {
    let m = (bits & 0x7fff_ffff) as i64;
    if bits >> 31 == 1 {
        -m
    } else {
        m
    }
}

fn f32_close(a: u32, b: u32) -> bool {
    if a == b {
        return true;
    }
    let fa = f32::from_bits(a);
    let fb = f32::from_bits(b);
    if fa.is_nan() && fb.is_nan() {
        return true;
    }
    (ordered_bits(a) - ordered_bits(b)).abs() <= 1
}

fn compare_blob(expected: &[u8], actual: &[u8], op: &str) {
    if expected.len() != actual.len() {
        die(&format!(
            "op {}: blob size mismatch: expected {}, got {}",
            op,
            expected.len(),
            actual.len()
        ));
    }
    for off in (0..actual.len()).step_by(4) {
        let a = u32::from_le_bytes([
            actual[off],
            actual[off + 1],
            actual[off + 2],
            actual[off + 3],
        ]);
        let b = u32::from_le_bytes([
            expected[off],
            expected[off + 1],
            expected[off + 2],
            expected[off + 3],
        ]);
        if !f32_close(a, b) {
            die(&format!(
                "op {}: blob word mismatch at offset {}: {:08x} vs {:08x}",
                op,
                off,
                a,
                b
            ));
        }
    }
}

// --- run modes ---------------------------------------------------------------------

struct Result {
    name: String,
    iters: usize,
    median: f64,
    p95: f64,
    minv: f64,
    peak: i64,
    count: i64,
    net: i64,
    in_hash: u64,
    out_hash: u64,
}

fn run_op(op: &Op, p: &mut Prep, blob: &mut Vec<u8>) -> u64 {
    blob.clear();
    (op.once)(p);
    let mut h = Fnv::new();
    if op.tol {
        (op.hash_out)(p, &mut h, Some(blob));
    } else {
        (op.hash_out)(p, &mut h, None);
    }
    h.h
}

fn write_results(path: &str, mode: &str, opt: &str, results: &[Result]) {
    let cpu = std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown".into());
    let rustc = std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".into());
    let mut s = String::new();
    let _ = writeln!(
        s,
        "{{\"lang\":\"rust\",\"compiler\":\"{}\",\"opt\":\"{}\",\"mode\":\"{}\",\"cpu\":\"{}\",\"no_simd\":true,",
        rustc.replace('"', ""), opt, mode, cpu
    );
    let _ = writeln!(s, "\"results\":[");
    for (i, r) in results.iter().enumerate() {
        let _ = writeln!(
            s,
            "{{\"op\":\"{}\",\"iters\":{},\"median_ms\":{:.4},\"p95_ms\":{:.4},\"min_ms\":{:.4},\"peak_alloc\":{},\"alloc_count\":{},\"net_alloc\":{},\"in_hash\":\"{:016x}\",\"out_hash\":\"{:016x}\"}}{}",
            r.name,
            r.iters,
            r.median,
            r.p95,
            r.minv,
            r.peak,
            r.count,
            r.net,
            r.in_hash,
            r.out_hash,
            if i + 1 < results.len() { "," } else { "" }
        );
    }
    let _ = writeln!(s, "]}}");
    fs::write(path, s).unwrap_or_else(|_| die(&format!("cannot write results: {}", path)));
    println!("results written to {}", path);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        die("usage: bench_rs <gen|verify|speed|mem> <pins.txt> <expected-dir|out.json> [op...]");
    }
    let mode = args[1].clone();
    let pins_path = args[2].clone();
    let arg3 = args[3].clone();

    let pins = if mode != "gen" {
        load_pins(&pins_path)
    } else {
        HashMap::new()
    };

    let ops = select_ops(&args[4..]);

    if mode == "gen" {
        fs::create_dir_all(arg3.as_str())
            .unwrap_or_else(|_| die(&format!("cannot create expected dir: {}", arg3)));
        let mut pins_out = String::from(
            "# bench corpus pins (FNV-1a64): <in|out>|<op>|<hash>; regenerate with bench/cpp/build.cmd + gen\n",
        );
        for op in &ops {
            let mut p = Prep::new();
            (op.prepare)(&mut p);
            p.seal_inputs();
            let mut blob = Vec::new();
            let oh = run_op(op, &mut p, &mut blob);
            let _ = writeln!(pins_out, "in|{}|{:016x}", op.name, p.in_hash);
            let _ = writeln!(pins_out, "out|{}|{:016x}", op.name, oh);
            if op.tol {
                let path = Path::new(&arg3).join(format!("{}.bin", op.name));
                fs::write(&path, &blob)
                    .unwrap_or_else(|_| die(&format!("cannot write blob: {}", path.display())));
            }
            println!("gen {:<16} in={:016x} out={:016x}", op.name, p.in_hash, oh);
        }
        fs::write(&pins_path, pins_out)
            .unwrap_or_else(|_| die(&format!("cannot write pins: {}", pins_path)));
        println!("pins written to {}", pins_path);
        return;
    }

    let mut results: Vec<Result> = Vec::new();

    for op in &ops {
        let mut p = Prep::new(); // fresh state per op: return values must not leak across ops
        (op.prepare)(&mut p);
        p.seal_inputs();

        match pins.get(&format!("in|{}", op.name)) {
            Some(&h) if h == p.in_hash => {}
            _ => die(&format!("input pin mismatch for {} (corpus diverged)", op.name)),
        }

        let mut blob: Vec<u8> = Vec::new();
        let mut r = Result {
            name: op.name.to_string(),
            iters: 0,
            median: 0.0,
            p95: 0.0,
            minv: 0.0,
            peak: 0,
            count: 0,
            net: 0,
            in_hash: p.in_hash,
            out_hash: 0,
        };

        match mode.as_str() {
            "verify" => {
                r.out_hash = run_op(op, &mut p, &mut blob);
            }
            "speed" => {
                for _ in 0..3 {
                    (op.once)(&mut p);
                }
                let est = (op.once)(&mut p);
                let n: usize = if est > 250.0 {
                    10
                } else if est > 25.0 {
                    20
                } else if est > 2.5 {
                    100
                } else {
                    500
                };
                let mut ts: Vec<f64> = Vec::with_capacity(n);
                for _ in 0..n {
                    let t0 = now_ms();
                    (op.once)(&mut p);
                    ts.push(now_ms() - t0);
                }
                ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                r.iters = n;
                r.median = ts[n / 2];
                r.p95 = ts[usize::min(n * 95 / 100, n - 1)];
                r.minv = ts[0];
                r.out_hash = run_op(op, &mut p, &mut blob);
            }
            "mem" => {
                TRACKING.store(true, Ordering::Relaxed);
                alloc_reset();
                (op.once)(&mut p);
                TRACKING.store(false, Ordering::Relaxed);
                r.peak = ALLOC_PEAK.load(Ordering::Relaxed) as i64;
                r.count = ALLOC_COUNT.load(Ordering::Relaxed) as i64;
                r.net = ALLOC_CUR.load(Ordering::Relaxed) as i64;
                if r.net != 0 {
                    die(&format!("op {} leaks {} bytes", op.name, r.net));
                }
                let mut h = Fnv::new();
                (op.hash_out)(&mut p, &mut h, None);
                r.out_hash = h.h;
            }
            _ => die(&format!("unknown mode: {}", mode)),
        }

        if op.tol {
            if mode == "verify" {
                let path = Path::new(&arg3).join(format!("{}.bin", op.name));
                let expected = fs::read(&path)
                    .unwrap_or_else(|_| die(&format!("cannot read blob: {}", path.display())));
                compare_blob(&expected, &blob, op.name);
            }
        } else {
            match pins.get(&format!("out|{}", op.name)) {
                Some(&h) if h == r.out_hash => {}
                _ => die(&format!("output pin mismatch for {}", op.name)),
            }
        }
        println!("{:<6} {:<16} ok", mode, op.name);
        results.push(r);
    }

    if mode == "speed" || mode == "mem" {
        let opt = std::env::var("BENCH_OPT").unwrap_or_else(|_| "unknown".into());
        write_results(&arg3, &mode, &opt, &results);
    }
}
