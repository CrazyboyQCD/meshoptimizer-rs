// C++-side benchmark driver for the speed/memory comparison against the Rust port
// (docs/adr/0011). Compiled only at benchmark time against upstream meshoptimizer at
// the pinned revision (docs/adr/0001: v1.3 = 9e1f07b159d3cb777f1c67ed31fc11fd117986f4),
// with clang (LLVM backend parity with rustc) and MESHOPTIMIZER_NO_SIMD (scalar
// profile, docs/adr/0010). Never part of the crate build.
//
// Build:  build.cmd <meshoptimizer-checkout> [O2|O3]
//
// Modes (mode <pins.txt> <arg> [op...]):
//   gen    <pins.txt> <expected-dir>   generate pins (+ expected blobs for float-tolerant ops)
//   verify <pins.txt> <expected-dir>   run each op once, check input/output pins
//   speed  <pins.txt> <out.json>       timing loop, allocation counting disabled
//   mem    <pins.txt> <out.json>       single call per op, allocation counting enabled
//
// Corpus: every op builds its inputs deterministically from the fixed-seed LCG
// (0x12345678, reset per case) with the same generators as
// tests/fixtures/gen_driver.cpp. Input bytes are FNV-1a-hashed into pins.txt at gen
// time and asserted in every other mode, so both drivers are guaranteed to measure
// bit-identical inputs. Outputs are hashed the same way; ops flagged float-tolerant
// (acos/log2/sin/cos paths, tdd.md section 7) are verified against expected blobs
// with a <=1 ULP allowance instead of exact hashes.
//
// Speed runs keep the counting allocator in pass-through mode (one untaken branch
// over malloc); mem mode records current/peak/count per single call and never
// produces timing claims.

#include "meshoptimizer.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <direct.h>
#include <string>
#include <unordered_map>
#include <vector>

// --- deterministic input generation -----------------------------------------

static uint32_t lcg_state = 0x12345678u;

static void lcg_reset()
{
	lcg_state = 0x12345678u;
}

static uint32_t lcg()
{
	lcg_state = lcg_state * 1664525u + 1013904223u;
	return lcg_state;
}

static std::vector<unsigned char> lcg_bytes(size_t n)
{
	lcg_reset();
	std::vector<unsigned char> v(n);
	for (size_t i = 0; i < n; ++i)
		v[i] = static_cast<unsigned char>(lcg() & 0xff);
	return v;
}

static std::vector<unsigned> lcg_words(size_t n)
{
	lcg_reset();
	std::vector<unsigned> v(n);
	for (size_t i = 0; i < n; ++i)
		v[i] = lcg();
	return v;
}

// Grid mesh, byte-identical generator to gen_driver.cpp make_grid_mesh (12-byte
// stride positions, jittered, optional column hole splitting the surface).
static const unsigned LGW = 316, LGH = 316; // large tier: 99856 vertices
static const unsigned MGW = 128, MGH = 128; // mid tier: 16384 vertices

struct Grid
{
	unsigned V;
	unsigned ic;
	std::vector<float> pos;
	std::vector<unsigned> idx;
};

static Grid make_grid(unsigned gw, unsigned gh, unsigned hole)
{
	lcg_reset();
	Grid g;
	g.V = gw * gh;
	g.pos.resize(size_t(g.V) * 3);
	for (unsigned i = 0; i < g.V; ++i)
	{
		unsigned y = i / gw;
		g.pos[size_t(i) * 3 + 0] = float(i % gw) + float(lcg() % 8) / 32.f;
		g.pos[size_t(i) * 3 + 1] = float(y) + float(lcg() % 8) / 32.f;
		g.pos[size_t(i) * 3 + 2] = float((i * 7 + y * 13) % 5) / 4.f;
	}
	for (unsigned y = 0; y + 1 < gh; ++y)
		for (unsigned x = 0; x + 1 < gw; ++x)
		{
			if (hole == 1 && x == gw / 2)
				continue;
			if (hole == 2 && x == gw / 4)
				continue;
			unsigned v0 = y * gw + x;
			unsigned v1 = v0 + 1, v2 = v0 + gw, v3 = v0 + gw + 1;
			g.idx.push_back(v0), g.idx.push_back(v1), g.idx.push_back(v2);
			g.idx.push_back(v1), g.idx.push_back(v3), g.idx.push_back(v2);
		}
	g.ic = unsigned(g.idx.size());
	return g;
}

// Position-derived attributes, byte-identical to gen_driver.cpp
// sp_fill_smooth_attributes (attr_stride = count * 4 bytes).
static std::vector<float> smooth_attrs(unsigned gw, unsigned gh, unsigned count)
{
	unsigned V = gw * gh;
	std::vector<float> a(size_t(V) * count);
	for (unsigned i = 0; i < V; ++i)
		for (unsigned k = 0; k < count; ++k)
			a[size_t(i) * count + k] = float((i % gw) + (i / gw) * 2) / 16.f + float(k) / 8.f;
	return a;
}

static std::vector<float> grid_uvs(unsigned gw, unsigned gh)
{
	unsigned V = gw * gh;
	std::vector<float> t(size_t(V) * 2);
	for (unsigned i = 0; i < V; ++i)
	{
		t[size_t(i) * 2 + 0] = float(i % gw) / float(gw - 1);
		t[size_t(i) * 2 + 1] = float(i / gw) / float(gh - 1);
	}
	return t;
}

static std::vector<float> const_normals(unsigned V)
{
	std::vector<float> n(size_t(V) * 3, 0.f);
	for (unsigned i = 0; i < V; ++i)
		n[size_t(i) * 3 + 2] = 1.f;
	return n;
}

// Closed axis-aligned box shell, per-face outward-wound grids (subdivided sub x sub
// quads per face). Solid-compatible input for the remesher.

// --- allocation counting -----------------------------------------------------

static long long g_alloc_cur = 0;
static long long g_alloc_peak = 0;
static long long g_alloc_count = 0;
static bool g_alloc_track = false;

static void* MESHOPTIMIZER_ALLOC_CALLCONV bench_allocate(size_t s)
{
	unsigned char* raw = static_cast<unsigned char*>(malloc(s + 16));
	if (!raw)
		return NULL;
	memcpy(raw, &s, sizeof(s));
	if (g_alloc_track)
	{
		g_alloc_cur += static_cast<long long>(s);
		g_alloc_count++;
		if (g_alloc_cur > g_alloc_peak)
			g_alloc_peak = g_alloc_cur;
	}
	return raw + 16;
}

static void MESHOPTIMIZER_ALLOC_CALLCONV bench_deallocate(void* q)
{
	if (!q)
		return;
	unsigned char* raw = static_cast<unsigned char*>(q) - 16;
	size_t s;
	memcpy(&s, raw, sizeof(s));
	if (g_alloc_track)
		g_alloc_cur -= static_cast<long long>(s);
	free(raw);
}

static void alloc_reset()
{
	g_alloc_cur = 0;
	g_alloc_peak = 0;
	g_alloc_count = 0;
}

// --- hashing ------------------------------------------------------------------

struct Fnv
{
	unsigned long long h = 0xcbf29ce484222325ull;

	void bytes(const void* p, size_t n)
	{
		const unsigned char* b = static_cast<const unsigned char*>(p);
		for (size_t i = 0; i < n; ++i)
		{
			h ^= b[i];
			h *= 0x100000001b3ull;
		}
	}
	void len(size_t n)
	{
		bytes(&n, sizeof(n));
	}
	void buf(const void* p, size_t n)
	{
		len(n);
		bytes(p, n);
	}
	void u64v(unsigned long long v)
	{
		bytes(&v, 8);
	}
};

static char hex16(unsigned long long v, char out[17])
{
	static const char* d = "0123456789abcdef";
	for (int i = 15; i >= 0; --i)
		out[i] = d[(v >> ((15 - i) * 4)) & 0xf];
	out[16] = 0;
	return *out;
}

// --- timing -------------------------------------------------------------------

static double now_ms()
{
	using namespace std::chrono;
	return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// --- benchmark op table ---------------------------------------------------------

struct Prep
{
	std::vector<float> pos;
	std::vector<unsigned> idx;
	unsigned V = 0;
	unsigned ic = 0;
	std::vector<float> attrs;
	std::vector<float> uvs;
	std::vector<float> nrm;
	std::vector<unsigned char> bytes; // filter scratch (filtered in place)
	std::vector<unsigned> words;      // exp filter input
	std::vector<unsigned char> enc;   // encoded input for decode ops
	std::vector<unsigned char> out_b;
	std::vector<unsigned> out_i;
	std::vector<unsigned> remap;
	std::vector<unsigned> counts; // partition cluster sizes
	std::vector<unsigned> part;   // partition output
	std::vector<unsigned> work_i; // mutable index copy (vfetch)
	std::vector<unsigned char> work_b; // pristine input copy (in-place filters)
	std::vector<float> out_f;
	std::vector<meshopt_Meshlet> meshlets;
	std::vector<unsigned> mverts;
	std::vector<unsigned char> mtris;
	size_t ret = 0;    // primary return (bytes written / index count / count pass)
	size_t ret2 = 0;   // secondary return (remesh emit pass)
	size_t target = 0; // simplifier target index count
	size_t total = 0;  // partition total index count
	unsigned cc = 0;   // partition cluster count
	float error = 0.f; // simplifier result_error
	std::vector<std::pair<const void*, size_t>> trace;
	unsigned long long in_hash = 0;

	void trace_input(const void* p, size_t n)
	{
		trace.emplace_back(p, n);
	}
	void seal_inputs()
	{
		Fnv h;
		for (size_t i = 0; i < trace.size(); ++i)
			h.buf(trace[i].first, trace[i].second);
		in_hash = h.h;
		trace.clear();
	}
};

static void make_box(Prep& p, unsigned sub, float extent)
{
	static const float face_n[6][3] = {{0, 0, 1}, {0, 0, -1}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}};
	static const float face_u[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 0, -1}, {0, 0, 1}, {1, 0, 0}, {1, 0, 0}};
	static const float face_v[6][3] = {{0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
	unsigned vs = sub + 1;
	float h = extent / 2.f;
	p.pos.clear();
	p.idx.clear();
	for (unsigned f = 0; f < 6; ++f)
	{
		unsigned base = unsigned(p.pos.size() / 3);
		for (unsigned b = 0; b < vs; ++b)
			for (unsigned a = 0; a < vs; ++a)
			{
				float fa = float(a) / float(sub) * 2.f - 1.f;
				float fb = float(b) / float(sub) * 2.f - 1.f;
				for (int k = 0; k < 3; ++k)
					p.pos.push_back((face_n[f][k] + face_u[f][k] * fa + face_v[f][k] * fb) * h);
			}
		for (unsigned b = 0; b + 1 < vs; ++b)
			for (unsigned a = 0; a + 1 < vs; ++a)
			{
				unsigned v00 = base + b * vs + a, v10 = v00 + 1, v01 = v00 + vs, v11 = v01 + 1;
				p.idx.push_back(v00), p.idx.push_back(v10), p.idx.push_back(v01);
				p.idx.push_back(v10), p.idx.push_back(v11), p.idx.push_back(v01);
			}
	}
	p.V = unsigned(p.pos.size() / 3);
	p.ic = unsigned(p.idx.size());
}

#define STR2(x) #x
#define STR(x) STR2(x)
#ifndef OPT_LEVEL
#define OPT_LEVEL O2
#endif

struct Op
{
	const char* name;
	bool tol; // float-tolerant verification (expected blob + <=1 ULP, verify mode)
	void (*prepare)(Prep& p);
	double (*once)(Prep& p);
	void (*hash_out)(Prep& p, Fnv& h, std::vector<unsigned char>* blob);
};

// --- codec ops (tier L) ---------------------------------------------------------

static void prep_vc_encode(Prep& p)
{
	Grid g = make_grid(LGW, LGH, 0);
	p.pos = std::move(g.pos);
	p.V = g.V;
	p.trace_input(p.pos.data(), p.pos.size() * 4);
	p.out_b.resize(meshopt_encodeVertexBufferBound(p.V, 12));
}

static double once_vc_encode(Prep& p)
{
	double t0 = now_ms();
	p.ret = meshopt_encodeVertexBuffer(p.out_b.data(), p.out_b.size(), p.pos.data(), p.V, 12);
	return now_ms() - t0;
}

static void hash_vc_encode(Prep& p, Fnv& h, std::vector<unsigned char>* blob)
{
	h.len(p.ret);
	h.bytes(p.out_b.data(), p.ret);
	if (blob)
		blob->assign(p.out_b.begin(), p.out_b.begin() + ptrdiff_t(p.ret));
}

static void prep_vc_decode(Prep& p)
{
	Grid g = make_grid(LGW, LGH, 0);
	p.pos = std::move(g.pos);
	p.V = g.V;
	p.enc.resize(meshopt_encodeVertexBufferBound(p.V, 12));
	size_t n = meshopt_encodeVertexBuffer(p.enc.data(), p.enc.size(), p.pos.data(), p.V, 12);
	p.enc.resize(n);
	p.trace_input(p.enc.data(), p.enc.size());
	p.out_b.assign(size_t(p.V) * 12, 0);
}

static double once_vc_decode(Prep& p)
{
	double t0 = now_ms();
	p.ret = size_t(meshopt_decodeVertexBuffer(p.out_b.data(), p.V, 12, p.enc.data(), p.enc.size()));
	return now_ms() - t0;
}

static void hash_vc_decode(Prep& p, Fnv& h, std::vector<unsigned char>* blob)
{
	if (p.ret != 0)
	{
		fprintf(stderr, "op vc_decode failed: %zu\n", p.ret);
		exit(1);
	}
	h.len(p.out_b.size());
	h.bytes(p.out_b.data(), p.out_b.size());
	if (blob)
		*blob = p.out_b;
}

static void prep_ic_encode(Prep& p)
{
	Grid g = make_grid(LGW, LGH, 0);
	p.pos = std::move(g.pos);
	p.idx = std::move(g.idx);
	p.V = g.V;
	p.ic = g.ic;
	p.trace_input(p.pos.data(), p.pos.size() * 4);
	p.trace_input(p.idx.data(), p.idx.size() * 4);
	p.out_b.resize(meshopt_encodeIndexBufferBound(p.ic, p.V));
}

static double once_ic_encode(Prep& p)
{
	double t0 = now_ms();
	p.ret = meshopt_encodeIndexBuffer(p.out_b.data(), p.out_b.size(), p.idx.data(), p.ic);
	return now_ms() - t0;
}

static void hash_ic_encode(Prep& p, Fnv& h, std::vector<unsigned char>* blob)
{
	h.len(p.ret);
	h.bytes(p.out_b.data(), p.ret);
	if (blob)
		blob->assign(p.out_b.begin(), p.out_b.begin() + ptrdiff_t(p.ret));
}

static void prep_ic_decode(Prep& p)
{
	Grid g = make_grid(LGW, LGH, 0);
	p.pos = std::move(g.pos);
	p.idx = std::move(g.idx);
	p.V = g.V;
	p.ic = g.ic;
	p.enc.resize(meshopt_encodeIndexBufferBound(p.ic, p.V));
	size_t n = meshopt_encodeIndexBuffer(p.enc.data(), p.enc.size(), p.idx.data(), p.ic);
	p.enc.resize(n);
	p.trace_input(p.enc.data(), p.enc.size());
	p.out_i.assign(p.ic, 0);
}

static double once_ic_decode(Prep& p)
{
	double t0 = now_ms();
	p.ret = size_t(meshopt_decodeIndexBuffer(p.out_i.data(), p.ic, 4, p.enc.data(), p.enc.size()));
	return now_ms() - t0;
}

static void hash_ic_decode(Prep& p, Fnv& h, std::vector<unsigned char>* blob)
{
	if (p.ret != 0)
	{
		fprintf(stderr, "op ic_decode failed: %zu\n", p.ret);
		exit(1);
	}
	h.len(p.out_i.size() * 4);
	h.bytes(p.out_i.data(), p.out_i.size() * 4);
	if (blob)
	{
		blob->resize(p.out_i.size() * 4);
		memcpy(blob->data(), p.out_i.data(), blob->size());
	}
}

// --- filter ops (tier L, in-place: every call restores the pristine input first,
// --- with the restore copy outside the measured window) -----------------------------

static void prep_vf_oct(Prep& p)
{
	p.work_b = lcg_bytes(size_t(LGW * LGH) * 8);
	p.bytes.assign(p.work_b.size(), 0);
	p.V = LGW * LGH;
	p.trace_input(p.work_b.data(), p.work_b.size());
}

static double once_vf_oct(Prep& p)
{
	memcpy(p.bytes.data(), p.work_b.data(), p.work_b.size());
	double t0 = now_ms();
	meshopt_decodeFilterOct(p.bytes.data(), p.V, 8);
	return now_ms() - t0;
}

static void hash_vf_oct(Prep& p, Fnv& h, std::vector<unsigned char>* blob)
{
	h.len(p.bytes.size());
	h.bytes(p.bytes.data(), p.bytes.size());
	if (blob)
		*blob = p.bytes;
}

static void prep_vf_quat(Prep& p)
{
	p.work_b = lcg_bytes(size_t(LGW * LGH) * 8);
	p.bytes.assign(p.work_b.size(), 0);
	p.V = LGW * LGH;
	p.trace_input(p.work_b.data(), p.work_b.size());
}

static double once_vf_quat(Prep& p)
{
	memcpy(p.bytes.data(), p.work_b.data(), p.work_b.size());
	double t0 = now_ms();
	meshopt_decodeFilterQuat(p.bytes.data(), p.V, 8);
	return now_ms() - t0;
}

static void prep_vf_exp(Prep& p)
{
	p.work_i = lcg_words(size_t(LGW * LGH) * 3);
	p.words.assign(p.work_i.size(), 0);
	p.V = LGW * LGH;
	p.trace_input(p.work_i.data(), p.work_i.size() * 4);
}

static double once_vf_exp(Prep& p)
{
	memcpy(p.words.data(), p.work_i.data(), p.work_i.size() * 4);
	double t0 = now_ms();
	meshopt_decodeFilterExp(p.words.data(), p.V, 12);
	return now_ms() - t0;
}

static void hash_vf_exp(Prep& p, Fnv& h, std::vector<unsigned char>* blob)
{
	h.len(p.words.size() * 4);
	h.bytes(p.words.data(), p.words.size() * 4);
	if (blob)
	{
		blob->resize(p.words.size() * 4);
		memcpy(blob->data(), p.words.data(), blob->size());
	}
}

// --- geometry ops (tier L) ------------------------------------------------------

static void load_grid(Prep& p, unsigned hole)
{
	Grid g = make_grid(LGW, LGH, hole);
	p.pos = std::move(g.pos);
	p.idx = std::move(g.idx);
	p.V = g.V;
	p.ic = g.ic;
	p.trace_input(p.pos.data(), p.pos.size() * 4);
	p.trace_input(p.idx.data(), p.idx.size() * 4);
}

static void prep_ig_remap(Prep& p)
{
	load_grid(p, 0);
	p.remap.assign(p.V, 0);
}

static double once_ig_remap(Prep& p)
{
	double t0 = now_ms();
	p.ret = meshopt_generateVertexRemap(p.remap.data(), p.idx.data(), p.ic, p.pos.data(), p.V, 12);
	return now_ms() - t0;
}

static void hash_remap(Prep& p, Fnv& h, std::vector<unsigned char>* blob)
{
	(void)p;
	(void)blob;
	h.len(p.remap.size() * 4);
	h.bytes(p.remap.data(), p.remap.size() * 4);
}

static void prep_vcache(Prep& p)
{
	load_grid(p, 0);
	p.out_i.assign(p.ic, 0);
}

static double once_vcache(Prep& p)
{
	double t0 = now_ms();
	meshopt_optimizeVertexCache(p.out_i.data(), p.idx.data(), p.ic, p.V);
	return now_ms() - t0;
}

static void hash_out_i(Prep& p, Fnv& h, std::vector<unsigned char>* blob)
{
	(void)blob;
	size_t n = (p.ret ? p.ret : p.ic) * 4;
	h.len(n);
	h.bytes(p.out_i.data(), n);
}

static double once_overdraw(Prep& p)
{
	double t0 = now_ms();
	meshopt_optimizeOverdraw(p.out_i.data(), p.idx.data(), p.ic, p.pos.data(), p.V, 12, 1.05f);
	return now_ms() - t0;
}

static void prep_vfetch(Prep& p)
{
	load_grid(p, 0);
	p.work_i = p.idx;
	p.out_b.assign(size_t(p.V) * 12, 0);
}

static double once_vfetch(Prep& p)
{
	memcpy(p.work_i.data(), p.idx.data(), size_t(p.ic) * 4);
	double t0 = now_ms();
	p.ret = meshopt_optimizeVertexFetch(p.out_b.data(), p.work_i.data(), p.ic, p.pos.data(), p.V, 12);
	return now_ms() - t0;
}

static void hash_vfetch(Prep& p, Fnv& h, std::vector<unsigned char>* blob)
{
	(void)blob;
	h.len(p.out_b.size());
	h.bytes(p.out_b.data(), p.out_b.size());
	h.len(p.work_i.size() * 4);
	h.bytes(p.work_i.data(), p.work_i.size() * 4);
}

static double once_stripify(Prep& p)
{
	double t0 = now_ms();
	p.ret = meshopt_stripify(p.out_i.data(), p.idx.data(), p.ic, p.V, 0xffffu);
	return now_ms() - t0;
}

static void prep_simplify(Prep& p)
{
	load_grid(p, 0);
	p.target = p.ic / 3 / 2 * 3;
	p.out_i.assign(p.ic, 0);
}

static double once_simplify(Prep& p)
{
	double t0 = now_ms();
	p.ret = meshopt_simplify(p.out_i.data(), p.idx.data(), p.ic, p.pos.data(), p.V, 12, p.target, 1e-2f, 0, &p.error);
	return now_ms() - t0;
}

static void hash_simplify(Prep& p, Fnv& h, std::vector<unsigned char>* blob)
{
	(void)blob;
	h.len(p.ret * 4);
	h.bytes(p.out_i.data(), p.ret * 4);
	h.bytes(&p.error, 4);
}

static void prep_simplify_attrs(Prep& p)
{
	load_grid(p, 0);
	p.attrs = smooth_attrs(LGW, LGH, 2);
	p.trace_input(p.attrs.data(), p.attrs.size() * 4);
	p.target = p.ic / 3 / 2 * 3;
	p.out_i.assign(p.ic, 0);
}

static double once_simplify_attrs(Prep& p)
{
	static const float kW[2] = {1.f, 1.f};
	double t0 = now_ms();
	p.ret = meshopt_simplifyWithAttributes(p.out_i.data(), p.idx.data(), p.ic, p.pos.data(), p.V, 12,
	    p.attrs.data(), 8, kW, 2, NULL, p.target, 1e-2f, 0, &p.error);
	return now_ms() - t0;
}

static double once_simplify_sloppy(Prep& p)
{
	double t0 = now_ms();
	p.ret = meshopt_simplifySloppy(p.out_i.data(), p.idx.data(), p.ic, p.pos.data(), p.V, 12, NULL, p.target, 1.f, &p.error);
	return now_ms() - t0;
}

static void prep_prune(Prep& p)
{
	load_grid(p, 2);
	p.out_i.assign(p.ic, 0);
}

static double once_prune(Prep& p)
{
	double t0 = now_ms();
	p.ret = meshopt_simplifyPrune(p.out_i.data(), p.idx.data(), p.ic, p.pos.data(), p.V, 12, 1.f);
	return now_ms() - t0;
}

static void prep_points(Prep& p)
{
	Grid g = make_grid(LGW, LGH, 0);
	p.pos = std::move(g.pos);
	p.V = g.V;
	p.trace_input(p.pos.data(), p.pos.size() * 4);
	p.out_i.assign(p.V, 0);
}

static double once_points(Prep& p)
{
	double t0 = now_ms();
	p.ret = meshopt_simplifyPoints(p.out_i.data(), p.pos.data(), p.V, 12, NULL, 0, 0.f, p.V / 4);
	return now_ms() - t0;
}

static double once_spatial(Prep& p)
{
	double t0 = now_ms();
	meshopt_spatialSortRemap(p.remap.data(), p.pos.data(), p.V, 12);
	return now_ms() - t0;
}

static void prep_clusterize(Prep& p)
{
	load_grid(p, 0);
	p.meshlets.resize(meshopt_buildMeshletsBound(p.ic, 64, 124));
	p.mverts.assign(p.ic, 0);
	p.mtris.assign(p.ic, 0);
}

static double once_clusterize(Prep& p)
{
	double t0 = now_ms();
	p.ret = meshopt_buildMeshlets(p.meshlets.data(), p.mverts.data(), p.mtris.data(), p.idx.data(), p.ic,
	    p.pos.data(), p.V, 12, 64, 124, 0.f);
	return now_ms() - t0;
}

static void hash_clusterize(Prep& p, Fnv& h, std::vector<unsigned char>* blob)
{
	(void)blob;
	h.u64v(p.ret);
	for (size_t i = 0; i < p.ret; ++i)
	{
		const meshopt_Meshlet& m = p.meshlets[i];
		h.u64v(m.vertex_offset);
		h.u64v(m.triangle_offset);
		h.u64v(m.vertex_count);
		h.u64v(m.triangle_count);
	}
	h.len(size_t(p.ic) * 4);
	h.bytes(p.mverts.data(), size_t(p.ic) * 4);
	h.len(p.ic);
	h.bytes(p.mtris.data(), p.ic);
}

static void prep_partition(Prep& p)
{
	load_grid(p, 0);
	const size_t chunk = 124 * 3;
	p.cc = unsigned(p.ic / chunk);
	p.total = size_t(p.cc) * chunk;
	p.counts.assign(p.cc, unsigned(chunk));
	p.part.assign(p.cc, 0);
	p.trace_input(p.counts.data(), p.counts.size() * 4);
}

static double once_partition(Prep& p)
{
	double t0 = now_ms();
	p.ret = meshopt_partitionClusters(p.part.data(), p.idx.data(), p.total, p.counts.data(), p.cc,
	    p.pos.data(), p.V, 12, 8);
	return now_ms() - t0;
}

static void hash_partition(Prep& p, Fnv& h, std::vector<unsigned char>* blob)
{
	(void)blob;
	h.len(size_t(p.cc) * 4);
	h.bytes(p.part.data(), size_t(p.cc) * 4);
	h.u64v(p.ret);
}

// --- mid-tier ops ----------------------------------------------------------------

static void load_grid_m(Prep& p)
{
	Grid g = make_grid(MGW, MGH, 0);
	p.pos = std::move(g.pos);
	p.idx = std::move(g.idx);
	p.V = g.V;
	p.ic = g.ic;
	p.trace_input(p.pos.data(), p.pos.size() * 4);
	p.trace_input(p.idx.data(), p.idx.size() * 4);
}

static void prep_tangents(Prep& p)
{
	load_grid_m(p);
	p.uvs = grid_uvs(MGW, MGH);
	p.nrm = const_normals(p.V);
	p.trace_input(p.uvs.data(), p.uvs.size() * 4);
	p.trace_input(p.nrm.data(), p.nrm.size() * 4);
	p.out_f.resize(size_t(p.ic) * 4); // per-corner output (index_count*4, meshoptimizer.h L949)
}

static double once_tangents(Prep& p)
{
	double t0 = now_ms();
	meshopt_generateTangents(p.out_f.data(), p.idx.data(), p.ic, p.pos.data(), p.V, 12, p.nrm.data(), 12,
	    p.uvs.data(), 8, 0);
	return now_ms() - t0;
}

static void hash_out_f(Prep& p, Fnv& h, std::vector<unsigned char>* blob)
{
	size_t n = p.out_f.size() * 4;
	h.len(n);
	h.bytes(p.out_f.data(), n);
	if (blob)
	{
		blob->resize(n);
		memcpy(blob->data(), p.out_f.data(), n);
	}
}

static void prep_normals(Prep& p)
{
	load_grid_m(p);
	p.out_f.resize(size_t(p.ic) * 3); // per-corner output (index_count*3, meshoptimizer.h L964)
}

static double once_normals(Prep& p)
{
	double t0 = now_ms();
	meshopt_generateNormals(p.out_f.data(), p.idx.data(), p.ic, p.pos.data(), p.V, 12, 3.14159265f, 0.f);
	return now_ms() - t0;
}

static void prep_remesh(Prep& p)
{
	make_box(p, 32, 32.f);
	p.trace_input(p.pos.data(), p.pos.size() * 4);
	p.trace_input(p.idx.data(), p.idx.size() * 4);
	p.ret = meshopt_remesh(NULL, 0, p.idx.data(), p.ic, p.pos.data(), p.V, 12, 64, 0);
	p.out_f.resize(p.ret * 9);
}

static double once_remesh(Prep& p)
{
	double t0 = now_ms();
	size_t n = meshopt_remesh(NULL, 0, p.idx.data(), p.ic, p.pos.data(), p.V, 12, 64, 0);
	if (n > p.ret)
	{
		fprintf(stderr, "op remesh count grew: %zu > %zu\n", n, p.ret);
		exit(1);
	}
	p.ret2 = meshopt_remesh(p.out_f.data(), p.ret, p.idx.data(), p.ic, p.pos.data(), p.V, 12, 64, 0);
	return now_ms() - t0;
}

static void hash_remesh(Prep& p, Fnv& h, std::vector<unsigned char>* blob)
{
	h.u64v(p.ret2);
	size_t n = p.ret2 * 9 * 4;
	h.len(n);
	h.bytes(p.out_f.data(), n);
	if (blob)
	{
		blob->resize(8 + n);
		memcpy(blob->data(), &p.ret2, 8);
		memcpy(blob->data() + 8, p.out_f.data(), n);
	}
}

static const Op kOps[] = {
    {"vc_encode", false, prep_vc_encode, once_vc_encode, hash_vc_encode},
    {"vc_decode", false, prep_vc_decode, once_vc_decode, hash_vc_decode},
    {"ic_encode", false, prep_ic_encode, once_ic_encode, hash_ic_encode},
    {"ic_decode", false, prep_ic_decode, once_ic_decode, hash_ic_decode},
    {"vf_oct", false, prep_vf_oct, once_vf_oct, hash_vf_oct},
    {"vf_quat", false, prep_vf_quat, once_vf_quat, hash_vf_oct},
    {"vf_exp", false, prep_vf_exp, once_vf_exp, hash_vf_exp},
    {"ig_remap", false, prep_ig_remap, once_ig_remap, hash_remap},
    {"vcache", false, prep_vcache, once_vcache, hash_out_i},
    {"overdraw", false, prep_vcache, once_overdraw, hash_out_i},
    {"vfetch", false, prep_vfetch, once_vfetch, hash_vfetch},
    {"stripify", false, prep_vcache, once_stripify, hash_out_i},
    {"simplify", false, prep_simplify, once_simplify, hash_simplify},
    {"simplify_attrs", false, prep_simplify_attrs, once_simplify_attrs, hash_simplify},
    {"simplify_sloppy", false, prep_simplify, once_simplify_sloppy, hash_simplify},
    {"prune", false, prep_prune, once_prune, hash_out_i},
    {"points", false, prep_points, once_points, hash_out_i},
    {"spatial", false, prep_ig_remap, once_spatial, hash_remap},
    {"clusterize", false, prep_clusterize, once_clusterize, hash_clusterize},
    {"partition", false, prep_partition, once_partition, hash_partition},
    {"tangents", true, prep_tangents, once_tangents, hash_out_f},
    {"normals", true, prep_normals, once_normals, hash_out_f},
    {"remesh", true, prep_remesh, once_remesh, hash_remesh},
};

static const size_t kOpCount = sizeof(kOps) / sizeof(kOps[0]);

static bool select_ops(char** names, int count, std::vector<const Op*>& sel)
{
	for (int i = 0; i < count; ++i)
	{
		bool found = false;
		for (size_t j = 0; j < kOpCount; ++j)
			if (strcmp(names[i], kOps[j].name) == 0)
			{
				sel.push_back(&kOps[j]);
				found = true;
				break;
			}
		if (!found)
		{
			fprintf(stderr, "unknown op: %s\n", names[i]);
			return false;
		}
	}
	if (sel.empty())
		for (size_t j = 0; j < kOpCount; ++j)
			sel.push_back(&kOps[j]);
	return true;
}

// --- pins ------------------------------------------------------------------------

typedef std::unordered_map<std::string, unsigned long long> PinMap;

static bool load_pins(const char* path, PinMap& pins)
{
	FILE* f = fopen(path, "rb");
	if (!f)
		return false;
	char line[256];
	while (fgets(line, sizeof(line), f))
	{
		if (line[0] == '#' || line[0] == '\n')
			continue;
		char* bar1 = strchr(line, '|');
		char* bar2 = bar1 ? strchr(bar1 + 1, '|') : NULL;
		if (!bar2)
			continue;
		*bar1 = 0;
		*bar2 = 0;
		char* nl = strchr(bar2 + 1, '\n');
		if (nl)
			*nl = 0;
		pins[std::string(line) + "|" + std::string(bar1 + 1)] = strtoull(bar2 + 1, NULL, 16);
	}
	fclose(f);
	return true;
}

// --- float-tolerant comparison (<=1 ULP, tdd.md section 7) -------------------------

static long long ordered_bits(float v)
{
	int32_t b;
	memcpy(&b, &v, 4);
	int32_t m = b & 0x7fffffff;
	return b < 0 ? -static_cast<long long>(m) : static_cast<long long>(m);
}

static bool f32_close(unsigned int a, unsigned int b)
{
	if (a == b)
		return true;
	float fa, fb;
	memcpy(&fa, &a, 4);
	memcpy(&fb, &b, 4);
	if (std::isnan(fa) && std::isnan(fb))
		return true;
	return llabs(ordered_bits(fa) - ordered_bits(fb)) <= 1;
}

static bool compare_blob(const std::vector<unsigned char>& expected, const std::vector<unsigned char>& actual)
{
	if (expected.size() != actual.size())
	{
		fprintf(stderr, "blob size mismatch: expected %zu, got %zu\n", expected.size(), actual.size());
		return false;
	}
	for (size_t off = 0; off + 4 <= actual.size(); off += 4)
	{
		unsigned int a, b;
		memcpy(&a, actual.data() + off, 4);
		memcpy(&b, expected.data() + off, 4);
		if (!f32_close(a, b))
		{
			fprintf(stderr, "blob word mismatch at offset %zu: %08x vs %08x\n", off, a, b);
			return false;
		}
	}
	return true;
}

static bool read_file(const char* path, std::vector<unsigned char>& data)
{
	FILE* f = fopen(path, "rb");
	if (!f)
		return false;
	fseek(f, 0, SEEK_END);
	long n = ftell(f);
	fseek(f, 0, SEEK_SET);
	data.resize(size_t(n));
	size_t rd = fread(data.data(), 1, data.size(), f);
	fclose(f);
	return rd == data.size();
}

// --- run modes ---------------------------------------------------------------------

struct Result
{
	std::string name;
	int iters = 0;
	double median = 0, p95 = 0, minv = 0;
	long long peak = 0, count = 0, net = 0;
	unsigned long long in_hash = 0, out_hash = 0;
};

// Runs the op once and computes its output hash (collecting the expected blob for
// float-tolerant ops when requested). Returns the output hash.
static unsigned long long run_op(const Op& op, Prep& p, std::vector<unsigned char>& blob)
{
	blob.clear();
	op.once(p);
	Fnv h;
	op.hash_out(p, h, op.tol ? &blob : NULL);
	return h.h;
}

int main(int argc, char** argv)
{
	if (argc < 4)
	{
		fprintf(stderr, "usage: %s <gen|verify|speed|mem> <pins.txt> <expected-dir|out.json> [op...]\n", argv[0]);
		return 1;
	}
	meshopt_setAllocator(bench_allocate, bench_deallocate);

	std::string mode = argv[1];
	const char* pins_path = argv[2];
	const char* arg3 = argv[3];

	PinMap pins;
	if (mode != "gen" && !load_pins(pins_path, pins))
	{
		fprintf(stderr, "cannot load pins: %s\n", pins_path);
		return 1;
	}

	std::vector<const Op*> ops;
	if (!select_ops(argv + 4, argc - 4, ops))
		return 1;

	if (mode == "gen")
	{
		if (_mkdir(arg3) != 0 && errno != EEXIST)
		{
			fprintf(stderr, "cannot create expected dir: %s\n", arg3);
			return 1;
		}
		FILE* pf = fopen(pins_path, "wb");
		if (!pf)
		{
			fprintf(stderr, "cannot write pins: %s\n", pins_path);
			return 1;
		}
		fprintf(pf, "# bench corpus pins (FNV-1a64): <in|out>|<op>|<hash>; regenerate with bench/cpp/build.cmd + gen\n");
		for (size_t i = 0; i < ops.size(); ++i)
		{
			const Op& op = *ops[i];
			Prep p;
			op.prepare(p);
			p.seal_inputs();
			std::vector<unsigned char> blob;
			unsigned long long oh = run_op(op, p, blob);
			char hex[17];
			hex16(p.in_hash, hex);
			fprintf(pf, "in|%s|%s\n", op.name, hex);
			hex16(oh, hex);
			fprintf(pf, "out|%s|%s\n", op.name, hex);
			if (op.tol)
			{
				std::string path = std::string(arg3) + "/" + op.name + ".bin";
				FILE* bf = fopen(path.c_str(), "wb");
				if (!bf || fwrite(blob.data(), 1, blob.size(), bf) != blob.size())
				{
					fprintf(stderr, "cannot write blob: %s\n", path.c_str());
					return 1;
				}
				fclose(bf);
			}
			printf("gen %-16s in=%016llx out=%016llx\n", op.name, p.in_hash, oh);
		}
		fclose(pf);
		printf("pins written to %s\n", pins_path);
		return 0;
	}

	Prep p;
	std::vector<unsigned char> blob;
	std::vector<Result> results;

	for (size_t i = 0; i < ops.size(); ++i)
	{
		const Op& op = *ops[i];
		p = Prep(); // fresh state per op: return values must not leak across ops
		op.prepare(p);
		p.seal_inputs();

		PinMap::const_iterator it = pins.find(std::string("in|") + op.name);
		if (it == pins.end() || it->second != p.in_hash)
		{
			fprintf(stderr, "input pin mismatch for %s (corpus diverged)\n", op.name);
			return 1;
		}

		Result r;
		r.name = op.name;
		r.in_hash = p.in_hash;

		if (mode == "verify")
		{
			r.out_hash = run_op(op, p, blob);
		}
		else if (mode == "speed")
		{
			for (int w = 0; w < 3; ++w)
				op.once(p);
			double est = op.once(p);
			int n = est > 250.0 ? 10 : est > 25.0 ? 20 : est > 2.5 ? 100 : 500;
			std::vector<double> ts;
			ts.resize(size_t(n));
			for (int k = 0; k < n; ++k)
			{
				double t0 = now_ms();
				op.once(p);
				ts[size_t(k)] = now_ms() - t0;
			}
			std::sort(ts.begin(), ts.end());
			r.iters = n;
			r.median = ts[size_t(n) / 2];
			r.p95 = ts[size_t(n * 95 / 100 < size_t(n) ? n * 95 / 100 : n - 1)];
			r.minv = ts[0];
			r.out_hash = run_op(op, p, blob);
		}
		else if (mode == "mem")
		{
			g_alloc_track = true;
			alloc_reset();
			op.once(p);
			g_alloc_track = false;
			r.peak = g_alloc_peak;
			r.count = g_alloc_count;
			r.net = g_alloc_cur;
			if (r.net != 0)
			{
				fprintf(stderr, "op %s leaks %lld bytes\n", op.name, r.net);
				return 1;
			}
			Fnv h;
			op.hash_out(p, h, NULL);
			r.out_hash = h.h;
		}
		else
		{
			fprintf(stderr, "unknown mode: %s\n", mode.c_str());
			return 1;
		}

		if (op.tol)
		{
			if (mode == "verify")
			{
				std::string path = std::string(arg3) + "/" + op.name + ".bin";
				std::vector<unsigned char> expected;
				if (!read_file(path.c_str(), expected) || !compare_blob(expected, blob))
				{
					fprintf(stderr, "output verification failed for %s\n", op.name);
					return 1;
				}
			}
		}
		else
		{
			PinMap::const_iterator ot = pins.find(std::string("out|") + op.name);
			if (ot == pins.end() || ot->second != r.out_hash)
			{
				fprintf(stderr, "output pin mismatch for %s\n", op.name);
				return 1;
			}
		}
		results.push_back(r);
		printf("%-6s %-16s ok\n", mode.c_str(), op.name);
	}

	if (mode == "speed" || mode == "mem")
	{
		FILE* f = fopen(arg3, "wb");
		if (!f)
		{
			fprintf(stderr, "cannot write results: %s\n", arg3);
			return 1;
		}
		const char* cpu = getenv("PROCESSOR_IDENTIFIER");
		fprintf(f, "{\"lang\":\"cpp\",\"compiler\":\"%s\",\"opt\":\"%s\",\"mode\":\"%s\",\"cpu\":\"%s\",\"no_simd\":true,\n",
		    __clang_version__, STR(OPT_LEVEL), mode.c_str(), cpu ? cpu : "unknown");
		fprintf(f, "\"results\":[\n");
		for (size_t i = 0; i < results.size(); ++i)
		{
			const Result& r = results[i];
			fprintf(f, "{\"op\":\"%s\",\"iters\":%d,\"median_ms\":%.4f,\"p95_ms\":%.4f,\"min_ms\":%.4f,"
			    "\"peak_alloc\":%lld,\"alloc_count\":%lld,\"net_alloc\":%lld,\"in_hash\":\"%016llx\",\"out_hash\":\"%016llx\"}%s\n",
			    r.name.c_str(), r.iters, r.median, r.p95, r.minv, r.peak, r.count, r.net, r.in_hash, r.out_hash,
			    i + 1 < results.size() ? "," : "");
		}
		fprintf(f, "]}\n");
		fclose(f);
		printf("results written to %s\n", arg3);
	}
	return 0;
}
