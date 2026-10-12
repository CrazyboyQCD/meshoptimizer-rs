// Golden-fixture generator for the Rust port. Compiled only at fixture-regeneration
// time against upstream meshoptimizer at the pinned revision (docs/adr/0001:
// v1.3 = 9e1f07b159d3cb777f1c67ed31fc11fd117986f4); never part of the crate build.
//
// Usage: gen_driver <command> <output.bin>
//
// All values little-endian, tightly packed. Input sets are deterministic: fixed
// edge cases plus a fixed-seed LCG stream (reset at the start of every command,
// so commands are independent). Regenerate via tests/fixtures/generate.cmd.
//
// Quantization commands:
//   quant-half    repeat: u32 in_bits, u16 out
//   dequant-half  repeat: u16 in, u32 out_bits            (full 0x0000..0xFFFF sweep)
//   quant-float   repeat: u32 in_bits, u16 n, u32 out_bits
//   pos-exp       repeat: f32 minv[3], f32 maxv[3], i32 min_exp, i32 max_bits, i32 result
//
// Index generation commands (u32 has_indices: 1 = index bytes follow, 0 = identity
// iteration over 0..vertex_count where the API allows it):
//   ig-remap        hdr: u32 V, u32 vertex_size, u32 index_count, u32 has_indices
//                   in: vertices V*vertex_size bytes, [indices index_count u32]
//                   out: remap V u32, result u32
//   ig-remap-multi  hdr: u32 V, u32 stream_count, u32 index_count, u32 has_indices
//                   in: per stream: u32 size, u32 stride, buffer V*stride bytes
//                       [indices index_count u32]
//                   out: remap V u32, result u32
//   ig-remap-custom hdr: u32 V, u32 stride, u32 index_count, u32 has_indices
//                   in: positions V*stride bytes (f32 x,y,z), [indices index_count u32]
//                   out: remap V u32, result u32        (NULL callback)
//   ig-remap-vb     hdr: u32 V, u32 vertex_size, u32 index_count
//                   in: vertices V*vertex_size bytes, indices index_count u32
//                   out: remap V u32, destination V*vertex_size bytes
//   ig-remap-ib     hdr: u32 V, u32 index_count, u32 has_indices
//                   in: remap V u32 (computed via generateVertexRemap on this mesh),
//                       [indices index_count u32]
//                   out: destination index_count u32
//   ig-filter-ib    hdr: u32 V, u32 vertex_size, u32 stride, u32 index_count
//                   in: vertices V*stride bytes, indices index_count u32
//                   out: destination index_count u32, written u32
//   ig-filter-multi hdr: u32 V, u32 stream_count, u32 index_count
//                   in: per stream: u32 size, u32 stride, buffer V*stride bytes
//                       indices index_count u32
//                   out: destination index_count u32, written u32
//   ig-shadow       hdr: u32 V, u32 vertex_size, u32 stride, u32 index_count
//                   in: vertices V*stride bytes, indices index_count u32
//                   out: destination index_count u32
//   ig-shadow-multi hdr: u32 V, u32 stream_count, u32 index_count
//                   in: per stream: u32 size, u32 stride, buffer V*stride bytes
//                       indices index_count u32
//                   out: destination index_count u32
//   ig-pos-remap    hdr: u32 V, u32 stride
//                   in: positions V*stride bytes (f32 x,y/z, includes -0.0)
//                   out: destination V u32
//   ig-adjacency    hdr: u32 V, u32 stride, u32 index_count
//                   in: positions V*stride bytes, indices index_count u32
//                   out: destination 2*index_count u32
//   ig-tess         hdr: u32 V, u32 stride, u32 index_count
//                   in: positions V*stride bytes, indices index_count u32
//                   out: destination 4*index_count u32
//   ig-provoking    hdr: u32 V, u32 index_count
//                   in: indices index_count u32
//                   out: destination index_count u32, reorder_offset u32,
//                        reorder reorder_offset u32
//
// Index codec commands (FR-05; encoders emit v1, v0 decode cases drive the
// upstream version setter generator-side):
//   ic-enc-ib       repeat: hdr u32 index_count, u32 vertex_count
//                   in: indices index_count u32
//                   out: u32 encoded_size, encoded bytes
//   ic-enc-seq      repeat: hdr u32 index_count, u32 vertex_count
//                   in: indices index_count u32
//                   out: u32 encoded_size, encoded bytes
//   ic-dec-ib       repeat: hdr u32 index_count, u32 index_size (2|4),
//                                u32 buffer_size, u32 expected_version
//                   in: encoded bytes
//                   out: i32 result, [decoded index_count*index_size bytes when result==0]
//   ic-dec-seq      repeat: hdr u32 index_count, u32 index_size (2|4),
//                                u32 buffer_size, u32 expected_version
//                   in: encoded bytes
//                   out: i32 result, [decoded index_count*index_size bytes when result==0]
//   ic-hostile-ib   repeat: hdr u32 buffer_size, u32 index_count, u32 index_size
//                   in: mutated encoded bytes
//                   out: i32 result, [decoded index_count*index_size bytes when result==0]
//   ic-hostile-seq  repeat: hdr u32 buffer_size, u32 index_count, u32 index_size
//                   in: mutated encoded bytes
//                   out: i32 result, [decoded index_count*index_size bytes when result==0]
//
// Vertex codec commands (FR-07; patterns: 0=random, 1=ramp, 2=constant, 3=sparse):
//   vc-enc          repeat: hdr u32 vertex_count, u32 vertex_size, u32 level
//                   in: vertices vertex_count*vertex_size bytes
//                   out: u32 encoded_size, encoded bytes
//   vc-dec          repeat: hdr u32 vertex_count, u32 vertex_size, u32 buffer_size,
//                                u32 expected_version
//                   in: encoded bytes
//                   out: i32 result, [decoded vertex_count*vertex_size bytes when result==0]
//   vc-hostile      repeat: hdr u32 buffer_size, u32 vertex_count, u32 vertex_size
//                   in: mutated encoded bytes
//                   out: i32 result, [decoded vertex_count*vertex_size bytes when result==0]
//
// Vertex filter commands (FR-08; scalar profile, ADR-0010):
//   vf-dec-oct8     repeat: u32 count, in count*4 bytes, out count*4 bytes
//   vf-dec-oct16    repeat: u32 count, in count*8 bytes, out count*8 bytes
//   vf-dec-quat     repeat: u32 count, in count*8 bytes, out count*8 bytes
//   vf-dec-exp      repeat: u32 word_count, in word_count*4 bytes, out word_count*4 bytes
//   vf-dec-color8   repeat: u32 count, in count*4 bytes, out count*4 bytes
//   vf-dec-color16  repeat: u32 count, in count*8 bytes, out count*8 bytes
//   vf-enc-oct      repeat: u32 stride, u32 bits, u32 count,
//                            in count*4 f32, out count*stride bytes
//   vf-enc-quat     repeat: u32 bits, u32 count, in count*4 f32, out count*8 bytes
//   vf-enc-exp      repeat: u32 stride_float, u32 bits, u32 mode, u32 count,
//                            in count*stride_float f32, out count*stride_float u32
//   vf-enc-color    repeat: u32 stride, u32 bits, u32 count,
//                            in count*4 f32, out count*stride bytes
//
// Meshlet/bounds commands (P1.5; bounding-sphere kernel via computeSphereBounds):
//   cb-sphere       repeat: hdr u32 count, u32 stride, u32 has_radii, u32 radii_stride
//                   in: positions count*stride bytes (f32 x,y,z), [radii count*radii_stride bytes]
//                   out: f32 center[3], f32 radius
//
// Meshlet codec commands (FR-06; scalar profile):
//   mc-enc          repeat: hdr u32 vertex_count, u32 triangle_count
//                   in: vertices vertex_count u32, triangles triangle_count*3 u8
//                   out: u32 encoded_size, encoded bytes
//   mc-dec          repeat: hdr u32 vertex_count, u32 vertex_size, u32 triangle_count,
//                                u32 triangle_size, u32 buffer_size
//                   in: encoded bytes
//                   out: i32 result, [vertices vertex_count*vertex_size bytes,
//                        triangles triangle_count*triangle_size bytes when result==0]
//   mc-dec-raw      repeat: hdr u32 vertex_count, u32 triangle_count, u32 buffer_size
//                   in: encoded bytes
//                   out: i32 result, [vertices align4(vertex_count)*4 u32,
//                        triangles align2(triangle_count)*4 u32 when result==0]
//                   (raw decoding writes the rounded-up element counts)
//   mc-hostile      repeat: hdr u32 buffer_size, u32 vertex_count, u32 triangle_count,
//                                u32 raw
//                   in: mutated encoded bytes
//                   out: i32 result, [vertices vertex_count*(raw?4:4) bytes,
//                        triangles triangle_count*(raw?4:3) bytes when result==0]
//
// Clusterizer commands (FR-12; bounds serialized as f32 center[3], f32 radius,
// f32 cone_apex[3], f32 cone_axis[3], f32 cone_cutoff, i8 cone_axis_s8[k] and
// i8 cone_cutoff_s8 each widened to u32 little-endian):
//   cb-cluster      repeat: hdr u32 index_count, u32 vertex_count, u32 stride
//                   in: positions vertex_count*stride bytes, indices index_count u32
//                   out: bounds
//   cb-meshlet      repeat: hdr u32 vertex_count, u32 triangle_count,
//                                u32 vertex_total, u32 stride
//                   in: meshlet_vertices vertex_total u32 (first vertex_count used),
//                       meshlet_triangles triangle_count*3 u8,
//                       positions vertex_total*stride bytes
//                   out: bounds
//   ms-optimize     repeat: hdr u32 vertex_count, u32 triangle_count, u32 level
//                   in: vertices vertex_count u32, triangles triangle_count*3 u8
//                   out: vertices vertex_count u32, triangles triangle_count*3 u8
//   ms-extract      repeat: hdr u32 index_count, u32 unique
//                   in: indices index_count u32
//                   out: vertices unique u32, triangles index_count u8
//   ms-build        repeat: hdr u32 index_count, u32 vertex_count, u32 stride,
//                                u32 variant, u32 max_vertices, u32 min_triangles,
//                                u32 max_triangles, u32 weight_bits, f32 weight
//                   in: positions vertex_count*stride bytes, indices index_count u32
//                   out: u32 meshlet_count, per meshlet: u32 vertex_offset,
//                        u32 triangle_offset, u32 vertex_count, u32 triangle_count,
//                        vertices index_count u32, triangles index_count u8

#include "meshoptimizer.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

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

static float lcg_float()
{
	uint32_t bits = lcg();
	float v;
	memcpy(&v, &bits, 4);
	return v;
}

static void put_u16(FILE* f, uint16_t v)
{
	unsigned char b[2] = {unsigned char(v & 0xff), unsigned char(v >> 8)};
	fwrite(b, 1, 2, f);
}

static void put_u32(FILE* f, uint32_t v)
{
	unsigned char b[4] = {unsigned char(v & 0xff), unsigned char((v >> 8) & 0xff), unsigned char((v >> 16) & 0xff), unsigned char(v >> 24)};
	fwrite(b, 1, 4, f);
}

static void put_i32(FILE* f, int32_t v)
{
	put_u32(f, uint32_t(v));
}

static void put_f32(FILE* f, float v)
{
	uint32_t bits;
	memcpy(&bits, &v, 4);
	put_u32(f, bits);
}

static void put_bytes(FILE* f, const void* p, size_t n)
{
	fwrite(p, 1, n, f);
}

// --- quantization -----------------------------------------------------------

static void gen_quant_half(FILE* f)
{
	static const uint32_t kEdge[] = {
	    0x00000000, 0x80000000, 0x00000001, 0x00000002, 0x007FFFFF, 0x00800000,
	    0x00800001, 0x33800000, 0x33800001, 0x387FFFFF, 0x38800000, 0x38800001,
	    0x3F800000, 0xBF800000, 0x3F7FFFFF, 0x3FFFFFFF, 0x40000000, 0x47000000,
	    0x477FE000, 0x477FE001, 0x477FFFFF, 0x47800000, 0x7F7FFFFF, 0x7F800000,
	    0xFF800000, 0x7F800001, 0x7FA00000, 0x7FC00000, 0x7FFFFFFF, 0xFFC00000,
	    0xFFFFFFFF, 0x4B400001, 0x4B7FFFFF, 0x4B3FFFFF,
	};
	const size_t kEdgeCount = sizeof(kEdge) / sizeof(kEdge[0]);
	const size_t kLcgCount = 8192;

	for (size_t i = 0; i < kEdgeCount; ++i)
	{
		float v;
		memcpy(&v, &kEdge[i], 4);
		put_u32(f, kEdge[i]);
		put_u16(f, meshopt_quantizeHalf(v));
	}

	for (size_t i = 0; i < kLcgCount; ++i)
	{
		float v = lcg_float();
		uint32_t bits;
		memcpy(&bits, &v, 4);
		put_u32(f, bits);
		put_u16(f, meshopt_quantizeHalf(v));
	}
}

static void gen_dequant_half(FILE* f)
{
	for (uint32_t h = 0; h <= 0xffff; ++h)
	{
		put_u16(f, uint16_t(h));
		put_f32(f, meshopt_dequantizeHalf(uint16_t(h)));
	}
}

static void gen_quant_float(FILE* f)
{
	static const uint32_t kEdge[] = {
	    0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x3F000000, 0xBF000000,
	    0x3FFFFFFF, 0x40000000, 0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000,
	    0x7F800001, 0x007FFFFF, 0x00800000, 0x80000001,
	};
	const size_t kEdgeCount = sizeof(kEdge) / sizeof(kEdge[0]);
	const size_t kLcgCount = 2048;
	static const int kLcgN[] = {0, 1, 7, 15, 22, 23};

	for (size_t i = 0; i < kEdgeCount; ++i)
	{
		float v;
		memcpy(&v, &kEdge[i], 4);
		for (int n = 0; n <= 23; ++n)
		{
			put_u32(f, kEdge[i]);
			put_u16(f, uint16_t(n));
			put_f32(f, meshopt_quantizeFloat(v, n));
		}
	}

	for (size_t i = 0; i < kLcgCount; ++i)
	{
		float v = lcg_float();
		uint32_t bits;
		memcpy(&bits, &v, 4);
		int n = kLcgN[lcg() % 6];
		put_u32(f, bits);
		put_u16(f, uint16_t(n));
		put_f32(f, meshopt_quantizeFloat(v, n));
	}
}

static void put_pos_exp_case(FILE* f, const float minv[3], const float maxv[3], int min_exp, int max_bits)
{
	for (int k = 0; k < 3; ++k)
		put_f32(f, minv[k]);
	for (int k = 0; k < 3; ++k)
		put_f32(f, maxv[k]);
	put_i32(f, min_exp);
	put_i32(f, max_bits);
	put_i32(f, meshopt_computePositionExponent(minv, maxv, min_exp, max_bits));
}

static void gen_pos_exp(FILE* f)
{
	{
		static const float z[3] = {0, 0, 0};
		const float unit[3] = {1, 1, 1};
		const float centered[3] = {-1, -1, -1};
		const float offsetlo[3] = {1024.5f, 1024.5f, 1024.5f};
		const float offsethi[3] = {2048.5f, 2048.5f, 2048.5f};
		const float tiny[3] = {1e-6f, 1e-6f, 1e-6f};
		const float point[3] = {3.5f, -2.25f, 7.125f};
		const float mixedlo[3] = {-1234.5f, 0.001f, -77.f};
		const float mixedhi[3] = {2345.75f, 99.9f, 1e5f};
		const float eights[3] = {-8.f, -8.f, -8.f};
		const float eightsu[3] = {8.f, 8.f, 8.f};

		put_pos_exp_case(f, z, unit, -15, 14);
		put_pos_exp_case(f, centered, unit, -15, 14);
		put_pos_exp_case(f, offsetlo, offsethi, -15, 14);
		put_pos_exp_case(f, z, tiny, -24, 24);
		put_pos_exp_case(f, point, point, -15, 16);
		put_pos_exp_case(f, z, z, -126, 2);
		put_pos_exp_case(f, mixedlo, mixedhi, -15, 14);
		put_pos_exp_case(f, eights, eightsu, -10, 10);
	}

	static const int kMinExp[] = {-15, -24, -126, 0, -8};
	static const int kMaxBits[] = {2, 8, 14, 16, 24};

	for (size_t i = 0; i < 256; ++i)
	{
		float minv[3], maxv[3];
		for (int k = 0; k < 3; ++k)
		{
			minv[k] = std::ldexp(float(lcg() % 1000) / 32.f, -int(lcg() % 12));
			maxv[k] = minv[k] + std::ldexp(float(lcg() % 2048) / 32.f, -int(lcg() % 12));
		}
		int min_exp = kMinExp[lcg() % 5];
		int max_bits = kMaxBits[lcg() % 5];
		put_pos_exp_case(f, minv, maxv, min_exp, max_bits);
	}
}

// --- index generation -------------------------------------------------------

struct Mesh
{
	unsigned V;
	unsigned index_count;
	std::vector<unsigned char> vertices; // V * stride (vertex data commands)
	std::vector<unsigned> indices;       // index_count entries
	bool has_indices;
};

// Byte-oriented vertex mesh with injected exact duplicates and degenerate triangles.
static void make_mesh(Mesh& m, unsigned V, unsigned T, unsigned vertex_size, unsigned stride, bool with_indices)
{
	m.V = V;
	m.index_count = with_indices ? T * 3 : V;
	m.has_indices = with_indices;
	m.vertices.assign(size_t(V) * stride, 0);

	for (unsigned i = 0; i < V; ++i)
		for (unsigned b = 0; b < vertex_size; ++b)
			m.vertices[size_t(i) * stride + b] = unsigned char(lcg() & 0xff);

	// inject exact byte duplicates (wedges)
	for (unsigned i = 8; i < V; i += 8)
		memcpy(&m.vertices[size_t(i) * stride], &m.vertices[size_t(i / 2) * stride], vertex_size);

	m.indices.resize(m.index_count);
	for (size_t i = 0; i < m.index_count; ++i)
		m.indices[i] = lcg() % V;

	// inject degenerate triangles
	for (unsigned t = 0; t + 4 <= T; t += 4)
	{
		m.indices[size_t(t) * 3 + 1] = m.indices[size_t(t) * 3 + 0];
		m.indices[size_t(t) * 3 + 2] = m.indices[size_t(t) * 3 + 0];
	}
}

// Position mesh (f32 x,y,z at byte stride) with injected -0.0 and exact duplicates.
static void make_positions(Mesh& m, unsigned V, unsigned T, unsigned stride, bool with_indices)
{
	m.V = V;
	m.index_count = with_indices ? T * 3 : V;
	m.has_indices = with_indices;
	m.vertices.assign(size_t(V) * stride, 0);

	for (unsigned i = 0; i < V; ++i)
	{
		unsigned char* v = &m.vertices[size_t(i) * stride];
		float p[3];
		for (int k = 0; k < 3; ++k)
		{
			p[k] = float(lcg() % 64) / 8.f;
			if (lcg() % 16 == 0)
				p[k] = -0.0f;
		}
		memcpy(v, p, 12);
	}

	// inject exact duplicates
	for (unsigned i = 6; i < V; i += 6)
		memcpy(&m.vertices[size_t(i) * stride], &m.vertices[size_t(i / 2) * stride], 12);

	m.indices.resize(m.index_count);
	for (size_t i = 0; i < m.index_count; ++i)
		m.indices[i] = lcg() % V;

	for (unsigned t = 0; t + 4 <= T; t += 4)
	{
		m.indices[size_t(t) * 3 + 1] = m.indices[size_t(t) * 3 + 0];
		m.indices[size_t(t) * 3 + 2] = m.indices[size_t(t) * 3 + 0];
	}
}

static void put_indices(FILE* f, const Mesh& m)
{
	if (m.has_indices)
		for (size_t i = 0; i < m.index_count; ++i)
			put_u32(f, m.indices[i]);
	else
		for (unsigned i = 0; i < m.index_count; ++i)
			put_u32(f, i);
}

static void put_mesh_header(FILE* f, const Mesh& m, unsigned size_field)
{
	put_u32(f, m.V);
	put_u32(f, size_field);
	put_u32(f, m.index_count);
	put_u32(f, m.has_indices ? 1u : 0u);
}

static void put_streams_header(FILE* f, unsigned V, unsigned stream_count, unsigned index_count, bool has_indices)
{
	put_u32(f, V);
	put_u32(f, stream_count);
	put_u32(f, index_count);
	put_u32(f, has_indices ? 1u : 0u);
}

static void gen_ig_remap(FILE* f)
{
	struct Cfg
	{
		unsigned V, T, vertex_size, stride;
		bool with_indices;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, 16, 16, true},
	    {64, 100, 12, 12, true},
	    {1, 1, 4, 4, true},
	    {257, 300, 8, 8, true},
	    {64, 0, 16, 16, false},
	    {1000, 1500, 16, 16, true},
	    {64, 100, 20, 24, true},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_mesh(m, c.V, c.T, c.vertex_size, c.stride, c.with_indices);
		put_mesh_header(f, m, c.vertex_size);
		// the remap call uses stride == vertex_size, so trailing padding bytes in
		// the stride do not participate; record only the significant prefix
		put_bytes(f, m.vertices.data(), size_t(m.V) * c.vertex_size);
		put_indices(f, m);

		std::vector<unsigned> remap(m.V);
		size_t result = meshopt_generateVertexRemap(remap.data(), m.has_indices ? m.indices.data() : NULL, m.index_count, m.vertices.data(), m.V, c.vertex_size);
		for (unsigned v : remap)
			put_u32(f, v);
		put_u32(f, unsigned(result));
	}
}

static void gen_ig_remap_multi(FILE* f)
{
	struct SCfg
	{
		unsigned size, stride;
	};
	struct Cfg
	{
		unsigned V, T;
		bool with_indices;
		SCfg streams[3];
		unsigned stream_count;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, true, {{12, 12}}, 1},
	    {64, 100, true, {{12, 16}, {4, 4}}, 2},
	    {257, 300, true, {{16, 16}, {12, 12}, {4, 8}}, 3},
	    {64, 0, false, {{12, 16}, {4, 4}}, 2},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		m.V = c.V;
		m.index_count = c.with_indices ? c.T * 3 : c.V;
		m.has_indices = c.with_indices;

		// each stream gets its own buffer; duplicates are injected across all streams
		std::vector<std::vector<unsigned char> > buffers(c.stream_count);
		for (unsigned s = 0; s < c.stream_count; ++s)
		{
			buffers[s].assign(size_t(c.V) * c.streams[s].stride, 0);
			for (unsigned i = 0; i < c.V; ++i)
				for (unsigned b = 0; b < c.streams[s].size; ++b)
					buffers[s][size_t(i) * c.streams[s].stride + b] = unsigned char(lcg() & 0xff);
		}
		for (unsigned i = 8; i < c.V; i += 8)
			for (unsigned s = 0; s < c.stream_count; ++s)
				memcpy(&buffers[s][size_t(i) * c.streams[s].stride], &buffers[s][size_t(i / 2) * c.streams[s].stride], c.streams[s].size);

		m.indices.resize(m.index_count);
		for (size_t i = 0; i < m.index_count; ++i)
			m.indices[i] = lcg() % c.V;

		put_streams_header(f, c.V, c.stream_count, m.index_count, c.with_indices);
		for (unsigned s = 0; s < c.stream_count; ++s)
		{
			put_u32(f, c.streams[s].size);
			put_u32(f, c.streams[s].stride);
			put_bytes(f, buffers[s].data(), buffers[s].size());
		}
		put_indices(f, m);

		std::vector<meshopt_Stream> streams(c.stream_count);
		for (unsigned s = 0; s < c.stream_count; ++s)
		{
			streams[s].data = buffers[s].data();
			streams[s].size = c.streams[s].size;
			streams[s].stride = c.streams[s].stride;
		}

		std::vector<unsigned> remap(c.V);
		size_t result = meshopt_generateVertexRemapMulti(remap.data(), m.has_indices ? m.indices.data() : NULL, m.index_count, c.V, streams.data(), c.stream_count);
		for (unsigned v : remap)
			put_u32(f, v);
		put_u32(f, unsigned(result));
	}
}

static void gen_ig_remap_custom(FILE* f)
{
	struct Cfg
	{
		unsigned V, T, stride;
		bool with_indices;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, 12, true},
	    {97, 150, 16, true},
	    {64, 0, 12, false},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_positions(m, c.V, c.T, c.stride, c.with_indices);
		put_mesh_header(f, m, c.stride);
		put_bytes(f, m.vertices.data(), m.vertices.size());
		put_indices(f, m);

		std::vector<unsigned> remap(m.V);
		size_t result = meshopt_generateVertexRemapCustom(remap.data(), m.has_indices ? m.indices.data() : NULL, m.index_count, reinterpret_cast<const float*>(m.vertices.data()), m.V, c.stride, NULL, NULL);
		for (unsigned v : remap)
			put_u32(f, v);
		put_u32(f, unsigned(result));
	}
}

static void gen_ig_remap_vb(FILE* f)
{
	struct Cfg
	{
		unsigned V, T, vertex_size, stride;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, 4, 4},
	    {64, 100, 8, 8},
	    {64, 100, 12, 12},
	    {64, 100, 16, 16},
	    {64, 100, 20, 20},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_mesh(m, c.V, c.T, c.vertex_size, c.stride, true);
		put_u32(f, m.V);
		put_u32(f, c.vertex_size);
		put_u32(f, m.index_count);
		put_bytes(f, m.vertices.data(), m.vertices.size());
		put_indices(f, m);

		std::vector<unsigned> remap(m.V);
		meshopt_generateVertexRemap(remap.data(), m.indices.data(), m.index_count, m.vertices.data(), m.V, c.vertex_size);
		for (unsigned v : remap)
			put_u32(f, v);

		std::vector<unsigned char> out(m.vertices.size(), 0xcc);
		meshopt_remapVertexBuffer(out.data(), m.vertices.data(), m.V, c.vertex_size, remap.data());
		put_bytes(f, out.data(), out.size());
	}
}

static void gen_ig_remap_ib(FILE* f)
{
	struct Cfg
	{
		unsigned V, T;
		bool with_indices;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, true},
	    {257, 300, true},
	    {64, 0, false},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_mesh(m, c.V, c.T, 16, 16, c.with_indices);
		put_u32(f, c.V);
		put_u32(f, c.with_indices ? c.T * 3 : c.V);
		put_u32(f, c.with_indices ? 1u : 0u);

		std::vector<unsigned> remap(m.V);
		meshopt_generateVertexRemap(remap.data(), m.has_indices ? m.indices.data() : NULL, m.index_count, m.vertices.data(), m.V, 16);
		for (unsigned v : remap)
			put_u32(f, v);

		if (c.with_indices)
		{
			put_indices(f, m);

			std::vector<unsigned> out(m.indices.size());
			meshopt_remapIndexBuffer(out.data(), m.indices.data(), m.indices.size(), remap.data());
			for (unsigned v : out)
				put_u32(f, v);
		}
		else
		{
			std::vector<unsigned> out(c.V);
			meshopt_remapIndexBuffer(out.data(), NULL, c.V, remap.data());
			for (unsigned v : out)
				put_u32(f, v);
		}
	}
}

static void put_filter_case(FILE* f, const Mesh& m, unsigned vertex_size, unsigned stride, const meshopt_Stream* streams, unsigned stream_count)
{
	std::vector<unsigned> out(m.index_count, 0xee);
	size_t written = streams
	    ? meshopt_filterIndexBufferMulti(out.data(), m.indices.data(), m.index_count, m.V, streams, stream_count)
	    : meshopt_filterIndexBuffer(out.data(), m.indices.data(), m.index_count, m.vertices.data(), m.V, vertex_size, stride);
	for (unsigned v : out)
		put_u32(f, v);
	put_u32(f, unsigned(written));
}

static void gen_ig_filter_ib(FILE* f)
{
	struct Cfg
	{
		unsigned V, T, vertex_size, stride;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, 16, 16},
	    {257, 300, 12, 12},
	    {16, 12, 16, 20},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_mesh(m, c.V, c.T, c.vertex_size, c.stride, true);
		put_u32(f, m.V);
		put_u32(f, c.vertex_size);
		put_u32(f, c.stride);
		put_u32(f, m.index_count);
		put_bytes(f, m.vertices.data(), m.vertices.size());
		put_indices(f, m);
		put_filter_case(f, m, c.vertex_size, c.stride, NULL, 0);
	}
}

static void gen_ig_filter_multi(FILE* f)
{
	struct Cfg
	{
		unsigned V, T;
		unsigned size0, stride0, size1, stride1;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, 12, 16, 4, 4},
	    {97, 150, 16, 16, 12, 12},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		m.V = c.V;
		m.index_count = c.T * 3;
		m.has_indices = true;

		std::vector<unsigned char> b0(size_t(c.V) * c.stride0), b1(size_t(c.V) * c.stride1);
		for (unsigned i = 0; i < c.V; ++i)
		{
			for (unsigned b = 0; b < c.size0; ++b)
				b0[size_t(i) * c.stride0 + b] = unsigned char(lcg() & 0xff);
			for (unsigned b = 0; b < c.size1; ++b)
				b1[size_t(i) * c.stride1 + b] = unsigned char(lcg() & 0xff);
		}
		for (unsigned i = 8; i < c.V; i += 8)
		{
			memcpy(&b0[size_t(i) * c.stride0], &b0[size_t(i / 2) * c.stride0], c.size0);
			memcpy(&b1[size_t(i) * c.stride1], &b1[size_t(i / 2) * c.stride1], c.size1);
		}

		m.indices.resize(m.index_count);
		for (size_t i = 0; i < m.index_count; ++i)
			m.indices[i] = lcg() % c.V;

		meshopt_Stream streams[2] = {
		    {b0.data(), c.size0, c.stride0},
		    {b1.data(), c.size1, c.stride1},
		};

		put_streams_header(f, c.V, 2, m.index_count, 1);
		put_u32(f, c.size0);
		put_u32(f, c.stride0);
		put_bytes(f, b0.data(), b0.size());
		put_u32(f, c.size1);
		put_u32(f, c.stride1);
		put_bytes(f, b1.data(), b1.size());
		put_indices(f, m);
		put_filter_case(f, m, 0, 0, streams, 2);
	}
}

static void gen_ig_shadow(FILE* f)
{
	struct Cfg
	{
		unsigned V, T, vertex_size, stride;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, 16, 16},
	    {257, 300, 12, 16},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_mesh(m, c.V, c.T, c.vertex_size, c.stride, true);
		put_u32(f, m.V);
		put_u32(f, c.vertex_size);
		put_u32(f, c.stride);
		put_u32(f, m.index_count);
		put_bytes(f, m.vertices.data(), m.vertices.size());
		put_indices(f, m);

		std::vector<unsigned> out(m.index_count);
		meshopt_generateShadowIndexBuffer(out.data(), m.indices.data(), m.index_count, m.vertices.data(), m.V, c.vertex_size, c.stride);
		for (unsigned v : out)
			put_u32(f, v);
	}
}

static void gen_ig_shadow_multi(FILE* f)
{
	struct Cfg
	{
		unsigned V, T;
	};
	static const Cfg kCfgs[] = {{64, 100}, {97, 150}};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		m.V = c.V;
		m.index_count = c.T * 3;
		m.has_indices = true;

		std::vector<unsigned char> b0(size_t(c.V) * 16), b1(size_t(c.V) * 12);
		for (unsigned i = 0; i < c.V; ++i)
		{
			for (unsigned b = 0; b < 12; ++b)
				b0[size_t(i) * 16 + b] = unsigned char(lcg() & 0xff);
			for (unsigned b = 0; b < 12; ++b)
				b1[size_t(i) * 12 + b] = unsigned char(lcg() & 0xff);
		}
		for (unsigned i = 8; i < c.V; i += 8)
		{
			memcpy(&b0[size_t(i) * 16], &b0[size_t(i / 2) * 16], 12);
			memcpy(&b1[size_t(i) * 12], &b1[size_t(i / 2) * 12], 12);
		}

		m.indices.resize(m.index_count);
		for (size_t i = 0; i < m.index_count; ++i)
			m.indices[i] = lcg() % c.V;

		meshopt_Stream streams[2] = {
		    {b0.data(), 12, 16},
		    {b1.data(), 12, 12},
		};

		put_streams_header(f, c.V, 2, m.index_count, 1);
		put_u32(f, 12);
		put_u32(f, 16);
		put_bytes(f, b0.data(), b0.size());
		put_u32(f, 12);
		put_u32(f, 12);
		put_bytes(f, b1.data(), b1.size());
		put_indices(f, m);

		std::vector<unsigned> out(m.index_count);
		meshopt_generateShadowIndexBufferMulti(out.data(), m.indices.data(), m.index_count, c.V, streams, 2);
		for (unsigned v : out)
			put_u32(f, v);
	}
}

static void gen_ig_pos_remap(FILE* f)
{
	struct Cfg
	{
		unsigned V, stride;
	};
	static const Cfg kCfgs[] = {{64, 12}, {97, 16}};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_positions(m, c.V, 0, c.stride, false);
		put_u32(f, m.V);
		put_u32(f, c.stride);
		put_bytes(f, m.vertices.data(), m.vertices.size());

		std::vector<unsigned> out(m.V);
		meshopt_generatePositionRemap(out.data(), reinterpret_cast<const float*>(m.vertices.data()), m.V, c.stride);
		for (unsigned v : out)
			put_u32(f, v);
	}
}

static void put_positions_indices_header(FILE* f, const Mesh& m, unsigned stride)
{
	put_u32(f, m.V);
	put_u32(f, stride);
	put_u32(f, m.index_count);
	put_bytes(f, m.vertices.data(), m.vertices.size());
	put_indices(f, m);
}

static void gen_ig_adjacency(FILE* f)
{
	struct Cfg
	{
		unsigned V, T, stride;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, 12},
	    {97, 150, 16},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_positions(m, c.V, c.T, c.stride, true);
		// crafted shared-edge pair at the front (requires V >= 4)
		m.indices[0] = 0, m.indices[1] = 1, m.indices[2] = 2;
		m.indices[3] = 2, m.indices[4] = 1, m.indices[5] = 3;

		put_positions_indices_header(f, m, c.stride);

		std::vector<unsigned> out(size_t(m.index_count) * 2);
		meshopt_generateAdjacencyIndexBuffer(out.data(), m.indices.data(), m.index_count, reinterpret_cast<const float*>(m.vertices.data()), m.V, c.stride);
		for (unsigned v : out)
			put_u32(f, v);
	}
}

static void gen_ig_tess(FILE* f)
{
	struct Cfg
	{
		unsigned V, T, stride;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, 12},
	    {97, 150, 16},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_positions(m, c.V, c.T, c.stride, true);
		m.indices[0] = 0, m.indices[1] = 1, m.indices[2] = 2;
		m.indices[3] = 2, m.indices[4] = 1, m.indices[5] = 3;

		put_positions_indices_header(f, m, c.stride);

		std::vector<unsigned> out(size_t(m.index_count) * 4);
		meshopt_generateTessellationIndexBuffer(out.data(), m.indices.data(), m.index_count, reinterpret_cast<const float*>(m.vertices.data()), m.V, c.stride);
		for (unsigned v : out)
			put_u32(f, v);
	}
}

static void gen_ig_provoking(FILE* f)
{
	struct Cfg
	{
		unsigned V, T;
	};
	static const Cfg kCfgs[] = {{32, 60}, {64, 100}};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_mesh(m, c.V, c.T, 4, 4, true);
		put_u32(f, m.V);
		put_u32(f, m.index_count);
		put_indices(f, m);

		std::vector<unsigned> out(m.index_count);
		std::vector<unsigned> reorder(m.V + c.T + 1);
		size_t result = meshopt_generateProvokingIndexBuffer(out.data(), reorder.data(), m.indices.data(), m.index_count, m.V);
		for (unsigned v : out)
			put_u32(f, v);
		put_u32(f, unsigned(result));
		for (size_t i = 0; i < result; ++i)
			put_u32(f, reorder[i]);
	}
}

// --- analysis / optimizers (P1.2) -------------------------------------------

static void put_mesh_with_indices(FILE* f, const Mesh& m, unsigned vertex_size)
{
	put_u32(f, m.V);
	put_u32(f, vertex_size);
	put_u32(f, m.index_count);
	put_bytes(f, m.vertices.data(), size_t(m.V) * vertex_size);
	put_indices(f, m);
}

static void gen_an_vcache(FILE* f)
{
	static const unsigned kCfgs[][3] = {
	    {16, 0, 0}, {14, 0, 0}, {32, 0, 0}, {16, 8, 0}, {16, 0, 4}, {16, 32, 8}, {12, 8, 2},
	};

	for (const unsigned* c : kCfgs)
	{
		Mesh m;
		make_mesh(m, 64, 100, 16, 16, true);
		put_u32(f, m.index_count);
		put_u32(f, m.V);
		put_u32(f, c[0]);
		put_u32(f, c[1]);
		put_u32(f, c[2]);
		put_indices(f, m);

		meshopt_VertexCacheStatistics st = meshopt_analyzeVertexCache(m.indices.data(), m.index_count, m.V, c[0], c[1], c[2]);
		put_u32(f, st.vertices_transformed);
		put_u32(f, st.warps_executed);
		put_f32(f, st.acmr);
		put_f32(f, st.atvr);
	}
}

static void gen_an_vfetch(FILE* f)
{
	struct Cfg
	{
		unsigned V, T, vertex_size;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, 16},
	    {257, 300, 12},
	    {16, 12, 20},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_mesh(m, c.V, c.T, c.vertex_size, c.vertex_size, true);
		put_u32(f, m.index_count);
		put_u32(f, m.V);
		put_u32(f, c.vertex_size);
		put_indices(f, m);

		meshopt_VertexFetchStatistics st = meshopt_analyzeVertexFetch(m.indices.data(), m.index_count, m.V, c.vertex_size);
		put_u32(f, st.bytes_fetched);
		put_f32(f, st.overfetch);
	}
}

static void gen_vf_remap(FILE* f)
{
	struct Cfg
	{
		unsigned V, T;
	};
	static const Cfg kCfgs[] = {{64, 100}, {257, 300}};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_mesh(m, c.V, c.T, 16, 16, true);
		put_u32(f, m.index_count);
		put_u32(f, m.V);
		put_indices(f, m);

		std::vector<unsigned> remap(m.V);
		size_t result = meshopt_optimizeVertexFetchRemap(remap.data(), m.indices.data(), m.index_count, m.V);
		for (unsigned v : remap)
			put_u32(f, v);
		put_u32(f, unsigned(result));
	}
}

static void gen_vf_fetch(FILE* f)
{
	struct Cfg
	{
		unsigned V, T, vertex_size;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, 16},
	    {257, 300, 12},
	    {16, 12, 20},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_mesh(m, c.V, c.T, c.vertex_size, c.vertex_size, true);
		put_u32(f, m.index_count);
		put_u32(f, m.V);
		put_u32(f, c.vertex_size);
		put_bytes(f, m.vertices.data(), size_t(m.V) * c.vertex_size);
		put_indices(f, m);

		std::vector<unsigned char> destination(size_t(m.V) * c.vertex_size, 0xcc);
		std::vector<unsigned> indices(m.indices);
		size_t result = meshopt_optimizeVertexFetch(destination.data(), indices.data(), m.index_count, m.vertices.data(), m.V, c.vertex_size);
		for (unsigned v : indices)
			put_u32(f, v);
		put_bytes(f, destination.data(), destination.size());
		put_u32(f, unsigned(result));
	}
}

static void gen_strip(FILE* f)
{
	struct Cfg
	{
		unsigned V, T;
		unsigned restart;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, 0},
	    {257, 300, 0xffffffffu},
	    {16, 12, 0xffff},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_mesh(m, c.V, c.T, 16, 16, true);
		put_u32(f, m.index_count);
		put_u32(f, m.V);
		put_u32(f, c.restart);
		put_indices(f, m);

		std::vector<unsigned> strip((m.index_count / 3) * 5);
		size_t result = meshopt_stripify(strip.data(), m.indices.data(), m.index_count, m.V, c.restart);
		put_u32(f, unsigned(result));
		for (size_t i = 0; i < result; ++i)
			put_u32(f, strip[i]);
	}
}

static void gen_unstrip(FILE* f)
{
	struct Cfg
	{
		unsigned V, T;
		unsigned restart;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, 0},
	    {257, 300, 0xffffffffu},
	    {16, 12, 0xffff},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_mesh(m, c.V, c.T, 16, 16, true);

		std::vector<unsigned> strip((m.index_count / 3) * 5);
		size_t strip_size = meshopt_stripify(strip.data(), m.indices.data(), m.index_count, m.V, c.restart);

		put_u32(f, unsigned(strip_size));
		put_u32(f, c.restart);
		for (size_t i = 0; i < strip_size; ++i)
			put_u32(f, strip[i]);

		std::vector<unsigned> tris(strip_size * 3, 0xee);
		size_t result = meshopt_unstripify(tris.data(), strip.data(), strip_size, c.restart);
		put_u32(f, unsigned(result));
		for (size_t i = 0; i < result; ++i)
			put_u32(f, tris[i]);
	}
}

static void put_vcache_output(FILE* f, const Mesh& m, int mode, unsigned cache_size)
{
	std::vector<unsigned> out(m.index_count, 0xee);
	if (mode == 0)
		meshopt_optimizeVertexCache(out.data(), m.indices.data(), m.index_count, m.V);
	else if (mode == 1)
		meshopt_optimizeVertexCacheStrip(out.data(), m.indices.data(), m.index_count, m.V);
	else
		meshopt_optimizeVertexCacheFifo(out.data(), m.indices.data(), m.index_count, m.V, cache_size);
	for (unsigned v : out)
		put_u32(f, v);
}

static void gen_vc_table(FILE* f)
{
	struct Cfg
	{
		unsigned V, T;
	};
	static const Cfg kCfgs[] = {{64, 100}, {257, 300}, {16, 12}, {1, 1}};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_mesh(m, c.V, c.T, 16, 16, true);
		put_u32(f, m.index_count);
		put_u32(f, m.V);
		put_indices(f, m);
		put_vcache_output(f, m, 0, 0);
	}
}

static void gen_vc_strip(FILE* f)
{
	struct Cfg
	{
		unsigned V, T;
	};
	static const Cfg kCfgs[] = {{64, 100}, {257, 300}, {16, 12}, {1, 1}};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_mesh(m, c.V, c.T, 16, 16, true);
		put_u32(f, m.index_count);
		put_u32(f, m.V);
		put_indices(f, m);
		put_vcache_output(f, m, 1, 0);
	}
}

static void gen_vc_fifo(FILE* f)
{
	struct Cfg
	{
		unsigned V, T, cache_size;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, 16},
	    {257, 300, 8},
	    {64, 100, 6},
	    {16, 12, 12},
	    {1, 1, 3},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_mesh(m, c.V, c.T, 16, 16, true);
		put_u32(f, m.index_count);
		put_u32(f, m.V);
		put_u32(f, c.cache_size);
		put_indices(f, m);
		put_vcache_output(f, m, 2, c.cache_size);
	}
}

// --- P1.3: rasterizer / overdraw / spatial / partition ----------------------

static void make_position_mesh(Mesh& m, unsigned V, unsigned T, unsigned stride)
{
	make_positions(m, V, T, stride, true);
}

static void put_position_mesh(FILE* f, const Mesh& m, unsigned stride)
{
	put_u32(f, m.V);
	put_u32(f, stride);
	put_u32(f, m.index_count);
	put_bytes(f, m.vertices.data(), m.vertices.size());
	put_indices(f, m);
}

static void gen_an_overdraw(FILE* f)
{
	struct Cfg
	{
		unsigned V, T, stride;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, 12},
	    {97, 150, 16},
	    {16, 12, 12},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_position_mesh(m, c.V, c.T, c.stride);
		put_position_mesh(f, m, c.stride);

		meshopt_OverdrawStatistics st = meshopt_analyzeOverdraw(m.indices.data(), m.index_count, reinterpret_cast<const float*>(m.vertices.data()), m.V, c.stride);
		put_u32(f, st.pixels_covered);
		put_u32(f, st.pixels_shaded);
		put_f32(f, st.overdraw);
	}
}

static void gen_an_coverage(FILE* f)
{
	struct Cfg
	{
		unsigned V, T, stride;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, 12},
	    {97, 150, 16},
	    {16, 12, 12},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_position_mesh(m, c.V, c.T, c.stride);
		put_position_mesh(f, m, c.stride);

		meshopt_CoverageStatistics st = meshopt_analyzeCoverage(m.indices.data(), m.index_count, reinterpret_cast<const float*>(m.vertices.data()), m.V, c.stride);
		for (int k = 0; k < 3; ++k)
			put_f32(f, st.coverage[k]);
		put_f32(f, st.extent);
	}
}

static void gen_od(FILE* f)
{
	struct Cfg
	{
		unsigned V, T, stride;
		float threshold;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, 12, 1.0f},
	    {97, 150, 16, 0.75f},
	    {16, 12, 12, 1.0f},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_position_mesh(m, c.V, c.T, c.stride);
		put_position_mesh(f, m, c.stride);
		put_f32(f, c.threshold);

		std::vector<unsigned> out(m.index_count, 0xee);
		meshopt_optimizeOverdraw(out.data(), m.indices.data(), m.index_count, reinterpret_cast<const float*>(m.vertices.data()), m.V, c.stride, c.threshold);
		for (unsigned v : out)
			put_u32(f, v);
	}
}

static void gen_sp_remap(FILE* f)
{
	struct Cfg
	{
		unsigned V, stride;
	};
	static const Cfg kCfgs[] = {{64, 12}, {257, 16}, {1000, 12}};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_positions(m, c.V, 0, c.stride, false);
		put_u32(f, m.V);
		put_u32(f, c.stride);
		put_bytes(f, m.vertices.data(), m.vertices.size());

		std::vector<unsigned> out(m.V);
		meshopt_spatialSortRemap(out.data(), reinterpret_cast<const float*>(m.vertices.data()), m.V, c.stride);
		for (unsigned v : out)
			put_u32(f, v);
	}
}

static void gen_sp_tri(FILE* f)
{
	struct Cfg
	{
		unsigned V, T, stride;
	};
	static const Cfg kCfgs[] = {{64, 100, 12}, {97, 150, 16}};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_position_mesh(m, c.V, c.T, c.stride);
		put_position_mesh(f, m, c.stride);

		std::vector<unsigned> out(m.index_count, 0xee);
		meshopt_spatialSortTriangles(out.data(), m.indices.data(), m.index_count, reinterpret_cast<const float*>(m.vertices.data()), m.V, c.stride);
		for (unsigned v : out)
			put_u32(f, v);
	}
}

static void gen_sp_cluster(FILE* f)
{
	struct Cfg
	{
		unsigned V, stride, cluster_size;
	};
	static const Cfg kCfgs[] = {{64, 12, 4}, {257, 16, 8}, {128, 12, 16}};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_positions(m, c.V, 0, c.stride, false);
		put_u32(f, m.V);
		put_u32(f, c.stride);
		put_u32(f, c.cluster_size);
		put_bytes(f, m.vertices.data(), m.vertices.size());

		std::vector<unsigned> out(m.V);
		meshopt_spatialClusterPoints(out.data(), reinterpret_cast<const float*>(m.vertices.data()), m.V, c.stride, c.cluster_size);
		for (unsigned v : out)
			put_u32(f, v);
	}
}

static void gen_pt(FILE* f)
{
	struct Cfg
	{
		unsigned V, T, stride, tris_per_cluster, target;
		bool with_positions;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, 12, 3, 4, true},
	    {64, 100, 12, 3, 8, true},
	    {97, 150, 16, 5, 4, true},
	    {64, 100, 12, 3, 4, false},
	    {16, 12, 12, 1, 2, true},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_position_mesh(m, c.V, c.T, c.stride);

		size_t cluster_count = m.index_count / 3 / c.tris_per_cluster;
		std::vector<unsigned> counts(cluster_count);
		std::vector<unsigned> offsets(cluster_count + 1);
		for (size_t i = 0; i < cluster_count; ++i)
			counts[i] = c.tris_per_cluster * 3;
		unsigned offset = 0;
		for (size_t i = 0; i < cluster_count; ++i)
		{
			offsets[i] = offset;
			offset += counts[i];
		}
		offsets[cluster_count] = offset;

		put_u32(f, unsigned(offset));
		put_u32(f, unsigned(cluster_count));
		put_u32(f, m.V);
		put_u32(f, c.stride);
		put_u32(f, c.target);
		put_u32(f, c.with_positions ? 1u : 0u);
		for (size_t i = 0; i < cluster_count; ++i)
			put_u32(f, counts[i]);
		for (size_t i = 0; i < offset; ++i)
			put_u32(f, m.indices[i]);
		if (c.with_positions)
			put_bytes(f, m.vertices.data(), m.vertices.size());

		std::vector<unsigned> out(cluster_count, 0xee);
		size_t result = meshopt_partitionClusters(out.data(), m.indices.data(), offset, counts.data(), cluster_count, c.with_positions ? reinterpret_cast<const float*>(m.vertices.data()) : NULL, m.V, c.stride, c.target);
		for (size_t i = 0; i < cluster_count; ++i)
			put_u32(f, out[i]);
		put_u32(f, unsigned(result));
	}
}

static void gen_od_debug(FILE* f)
{
	// removed: called upstream-static internals not visible at the pin; never compiled
	(void)f;
}

// --- dispatch ---------------------------------------------------------------

// --- index codec (FR-05) ----------------------------------------------------

static void ic_put_enc_ib(FILE* f, const std::vector<unsigned>& indices, unsigned vertex_count)
{
	put_u32(f, unsigned(indices.size()));
	put_u32(f, vertex_count);
	for (unsigned v : indices)
		put_u32(f, v);

	std::vector<unsigned char> buffer(meshopt_encodeIndexBufferBound(indices.size(), vertex_count));
	size_t size = meshopt_encodeIndexBuffer(buffer.data(), buffer.size(), indices.data(), indices.size());
	if (size == 0)
	{
		fprintf(stderr, "encodeIndexBuffer failed (n=%u)\n", unsigned(indices.size()));
		exit(1);
	}
	put_u32(f, unsigned(size));
	put_bytes(f, buffer.data(), size);
}

static void ic_put_enc_seq(FILE* f, const std::vector<unsigned>& indices, unsigned vertex_count)
{
	put_u32(f, unsigned(indices.size()));
	put_u32(f, vertex_count);
	for (unsigned v : indices)
		put_u32(f, v);

	std::vector<unsigned char> buffer(meshopt_encodeIndexSequenceBound(indices.size(), vertex_count));
	size_t size = meshopt_encodeIndexSequence(buffer.data(), buffer.size(), indices.data(), indices.size());
	if (size == 0)
	{
		fprintf(stderr, "encodeIndexSequence failed (n=%u)\n", unsigned(indices.size()));
		exit(1);
	}
	put_u32(f, unsigned(size));
	put_bytes(f, buffer.data(), size);
}

// triangle list [a, a+1, a+2], [a+3, a+4, a+5], ...
static void ic_tri_range(std::vector<unsigned>& v, unsigned start, unsigned triangles)
{
	for (unsigned t = 0; t < triangles; ++t)
		for (unsigned k = 0; k < 3; ++k)
			v.push_back(start + t * 3 + k);
}

// strip-order triangles sharing edges: (0,1,2) (2,1,3) (2,3,4) (4,3,5) ...
static void ic_tri_strip(std::vector<unsigned>& v, unsigned start, unsigned triangles)
{
	unsigned a = start, b = start + 1, c = start + 2;
	for (unsigned t = 0; t < triangles; ++t)
	{
		v.push_back(a);
		v.push_back(b);
		v.push_back(c);
		unsigned n = c + 1;
		if (t % 2 == 0)
		{
			a = c;
		}
		else
		{
			b = c;
		}
		c = n;
	}
}

// Encodes a corpus into a buffer and records a decode case (result + decoded bytes).
static void ic_put_dec_case(FILE* f, const std::vector<unsigned>& indices, unsigned vertex_count, unsigned index_size, int version)
{
	meshopt_encodeIndexVersion(version);
	std::vector<unsigned char> buffer(meshopt_encodeIndexBufferBound(indices.size(), vertex_count));
	size_t size = meshopt_encodeIndexBuffer(buffer.data(), buffer.size(), indices.data(), indices.size());
	meshopt_encodeIndexVersion(1);
	if (size == 0)
	{
		fprintf(stderr, "encodeIndexBuffer failed (v%d, n=%u)\n", version, unsigned(indices.size()));
		exit(1);
	}

	put_u32(f, unsigned(indices.size()));
	put_u32(f, index_size);
	put_u32(f, unsigned(size));
	put_u32(f, unsigned(version));
	put_bytes(f, buffer.data(), size);

	std::vector<unsigned char> dest(size_t(indices.size()) * index_size);
	int result = meshopt_decodeIndexBuffer(dest.data(), indices.size(), index_size, buffer.data(), size);
	put_i32(f, result);
	if (result == 0)
		put_bytes(f, dest.data(), dest.size());
}

static void ic_put_dec_seq_case(FILE* f, const std::vector<unsigned>& indices, unsigned vertex_count, unsigned index_size, int version)
{
	meshopt_encodeIndexVersion(version);
	std::vector<unsigned char> buffer(meshopt_encodeIndexSequenceBound(indices.size(), vertex_count));
	size_t size = meshopt_encodeIndexSequence(buffer.data(), buffer.size(), indices.data(), indices.size());
	meshopt_encodeIndexVersion(1);
	if (size == 0)
	{
		fprintf(stderr, "encodeIndexSequence failed (v%d, n=%u)\n", version, unsigned(indices.size()));
		exit(1);
	}

	put_u32(f, unsigned(indices.size()));
	put_u32(f, index_size);
	put_u32(f, unsigned(size));
	put_u32(f, unsigned(version));
	put_bytes(f, buffer.data(), size);

	std::vector<unsigned char> dest(size_t(indices.size()) * index_size);
	int result = meshopt_decodeIndexSequence(dest.data(), indices.size(), index_size, buffer.data(), size);
	put_i32(f, result);
	if (result == 0)
		put_bytes(f, dest.data(), dest.size());
}

static void ic_put_hostile_case(FILE* f, const std::vector<unsigned char>& buffer, unsigned index_count, unsigned index_size, bool sequence)
{
	put_u32(f, unsigned(buffer.size()));
	put_u32(f, index_count);
	put_u32(f, index_size);
	put_bytes(f, buffer.data(), buffer.size());

	std::vector<unsigned char> dest(size_t(index_count) * index_size);
	int result = sequence ? meshopt_decodeIndexSequence(dest.data(), index_count, index_size, buffer.data(), buffer.size())
	                      : meshopt_decodeIndexBuffer(dest.data(), index_count, index_size, buffer.data(), buffer.size());
	put_i32(f, result);
	if (result == 0)
		put_bytes(f, dest.data(), dest.size());
}

static void gen_ic_enc_ib(FILE* f)
{
	// empty
	ic_put_enc_ib(f, std::vector<unsigned>(), 1);

	// single triangle 0/1/2
	{
		std::vector<unsigned> v;
		ic_tri_range(v, 0, 1);
		ic_put_enc_ib(f, v, 3);
	}

	// sequential non-overlapping triangles
	{
		std::vector<unsigned> v;
		ic_tri_range(v, 0, 100);
		ic_put_enc_ib(f, v, 300);
	}

	// strip-like sequences (v1 last-1/last+1 codes)
	{
		std::vector<unsigned> v;
		ic_tri_strip(v, 0, 64);
		ic_put_enc_ib(f, v, 70);
	}

	// repeated triangle: exercises reset code (v1) and codeaux 00 entry
	{
		std::vector<unsigned> v;
		for (unsigned t = 0; t < 40; ++t)
		{
			v.push_back(0);
			v.push_back(1);
			v.push_back(2);
		}
		ic_put_enc_ib(f, v, 3);
	}

	// repeated triangle with a gap so next > 0 when 0/1/2 reappears
	{
		std::vector<unsigned> v;
		ic_tri_range(v, 0, 10);
		for (unsigned t = 0; t < 10; ++t)
		{
			v.push_back(0);
			v.push_back(1);
			v.push_back(2);
		}
		ic_put_enc_ib(f, v, 30);
	}

	// large indices: multi-byte varints
	{
		std::vector<unsigned> v;
		unsigned base = 0xfffffff0u;
		for (unsigned t = 0; t < 16; ++t)
		{
			v.push_back(base + t * 3);
			v.push_back(base + t * 3 + 1);
			v.push_back(base + t * 3 + 2);
		}
		ic_put_enc_ib(f, v, base + 48);
	}

	// random meshes, raw and vcache-optimized, with degenerate triangles
	for (int variant = 0; variant < 2; ++variant)
	{
		Mesh m;
		make_mesh(m, 64, 100, 12, 12, true);
		if (variant == 1)
		{
			std::vector<unsigned> opt(m.index_count);
			meshopt_optimizeVertexCache(opt.data(), m.indices.data(), m.index_count, m.V);
			m.indices = opt;
		}
		ic_put_enc_ib(f, m.indices, m.V);
	}

	// all-degenerate triangles (same vertex three times)
	{
		std::vector<unsigned> v;
		for (unsigned t = 0; t < 32; ++t)
		{
			unsigned k = t * 7 % 64;
			v.push_back(k);
			v.push_back(k);
			v.push_back(k);
		}
		ic_put_enc_ib(f, v, 64);
	}
}

static void gen_ic_enc_seq(FILE* f)
{
	// empty
	ic_put_enc_seq(f, std::vector<unsigned>(), 1);

	// single
	{
		std::vector<unsigned> v;
		v.push_back(42);
		ic_put_enc_seq(f, v, 43);
	}

	// monotonic small-delta sequence
	{
		std::vector<unsigned> v;
		for (unsigned i = 0; i < 256; ++i)
			v.push_back(i);
		ic_put_enc_seq(f, v, 256);
	}

	// alternating far-apart values: baseline switching heuristic
	{
		std::vector<unsigned> v;
		for (unsigned i = 0; i < 128; ++i)
		{
			v.push_back(i);
			v.push_back(0xffff0000u + i);
		}
		ic_put_enc_seq(f, v, 0xffff0080u);
	}

	// random sequence
	{
		std::vector<unsigned> v;
		for (unsigned i = 0; i < 300; ++i)
			v.push_back(lcg() % 1000);
		ic_put_enc_seq(f, v, 1000);
	}

	// big values with big deltas; kept within a 2^30 window so encoded
	// deltas stay below 2^31 (the varint loses the zigzag sign bit beyond
	// that, so full-range random values do not round-trip upstream either)
	{
		std::vector<unsigned> v;
		for (unsigned i = 0; i < 64; ++i)
			v.push_back(lcg() % 0x40000000u);
		ic_put_enc_seq(f, v, 0x40000000u);
	}
}

static void gen_ic_dec_ib(FILE* f)
{
	// corpora shared with the encode fixture; v0 and v1 decode, index_size 2 and 4
	struct Corpus
	{
		std::vector<unsigned> indices;
		unsigned vertex_count;
	};
	std::vector<Corpus> corpora;

	{
		Corpus c;
		c.vertex_count = 1;
		corpora.push_back(c);
	}
	{
		Corpus c;
		ic_tri_range(c.indices, 0, 1);
		c.vertex_count = 3;
		corpora.push_back(c);
	}
	{
		Corpus c;
		ic_tri_range(c.indices, 0, 64);
		c.vertex_count = 192;
		corpora.push_back(c);
	}
	{
		Corpus c;
		ic_tri_strip(c.indices, 0, 64);
		c.vertex_count = 70;
		corpora.push_back(c);
	}
	{
		Corpus c;
		for (unsigned t = 0; t < 10; ++t)
			ic_tri_range(c.indices, 0, 10);
		c.vertex_count = 30;
		corpora.push_back(c);
	}
	{
		Corpus c;
		for (unsigned t = 0; t < 40; ++t)
		{
			c.indices.push_back(0);
			c.indices.push_back(1);
			c.indices.push_back(2);
		}
		c.vertex_count = 3;
		corpora.push_back(c);
	}
	{
		// wrapping multi-byte varint deltas
		Corpus c;
		unsigned base = 0xfffffff0u;
		for (unsigned t = 0; t < 16; ++t)
		{
			c.indices.push_back(base + t * 3);
			c.indices.push_back(base + t * 3 + 1);
			c.indices.push_back(base + t * 3 + 2);
		}
		c.vertex_count = base + 48;
		corpora.push_back(c);
	}
	{
		Mesh m;
		make_mesh(m, 64, 100, 12, 12, true);
		std::vector<unsigned> opt(m.index_count);
		meshopt_optimizeVertexCache(opt.data(), m.indices.data(), m.index_count, m.V);
		Corpus c;
		c.indices = opt;
		c.vertex_count = m.V;
		corpora.push_back(c);
	}

	for (const Corpus& v : corpora)
	{
		ic_put_dec_case(f, v.indices, v.vertex_count, 4, 1);
		ic_put_dec_case(f, v.indices, v.vertex_count, 4, 0);
		ic_put_dec_case(f, v.indices, v.vertex_count, 2, 1);
	}
}

static void gen_ic_dec_seq(FILE* f)
{
	std::vector<std::vector<unsigned> > corpora;

	{
		std::vector<unsigned> v;
		for (unsigned i = 0; i < 256; ++i)
			v.push_back(i);
		corpora.push_back(v);
	}
	{
		std::vector<unsigned> v;
		for (unsigned i = 0; i < 128; ++i)
		{
			v.push_back(i);
			v.push_back(0xffff0000u + i);
		}
		corpora.push_back(v);
	}
	{
		std::vector<unsigned> v;
		for (unsigned i = 0; i < 300; ++i)
			v.push_back(lcg() % 1000);
		corpora.push_back(v);
	}

	for (const std::vector<unsigned>& v : corpora)
	{
		ic_put_dec_seq_case(f, v, 1000, 4, 1);
		ic_put_dec_seq_case(f, v, 1000, 4, 0);
		ic_put_dec_seq_case(f, v, 1000, 2, 1);
	}
}

static void gen_ic_hostile_ib(FILE* f)
{
	struct Base
	{
		std::vector<unsigned char> enc;
		unsigned index_count;
		unsigned vertex_count;
	};
	Base bases[2];

	{
		std::vector<unsigned> v;
		ic_tri_strip(v, 0, 8);
		std::vector<unsigned char> buffer(meshopt_encodeIndexBufferBound(v.size(), 16));
		size_t size = meshopt_encodeIndexBuffer(buffer.data(), buffer.size(), v.data(), v.size());
		bases[0].enc.assign(buffer.begin(), buffer.begin() + size);
		bases[0].index_count = unsigned(v.size());
		bases[0].vertex_count = 16;
	}
	{
		Mesh m;
		make_mesh(m, 64, 100, 12, 12, true);
		std::vector<unsigned> opt(m.index_count);
		meshopt_optimizeVertexCache(opt.data(), m.indices.data(), m.index_count, m.V);
		std::vector<unsigned char> buffer(meshopt_encodeIndexBufferBound(opt.size(), m.V));
		size_t size = meshopt_encodeIndexBuffer(buffer.data(), buffer.size(), opt.data(), opt.size());
		bases[1].enc.assign(buffer.begin(), buffer.begin() + size);
		bases[1].index_count = unsigned(opt.size());
		bases[1].vertex_count = m.V;
	}

	for (const Base& b : bases)
	{
		unsigned size = unsigned(b.enc.size());

		// truncations at varied prefix lengths
		unsigned lengths[] = {0, 1, 2, 17, 18, size / 2, size - 1, size};
		for (unsigned L : lengths)
		{
			if (L > size)
				continue;
			std::vector<unsigned char> prefix(b.enc.begin(), b.enc.begin() + L);
			ic_put_hostile_case(f, prefix, b.index_count, 4, false);
		}

		// header mutations: bad version nibbles, wrong magic, sequence header
		unsigned char headers[] = {0xe2, 0xef, 0xf1, 0x01, 0xd1, 0x60};
		for (unsigned char hb : headers)
		{
			std::vector<unsigned char> copy = b.enc;
			copy[0] = hb;
			ic_put_hostile_case(f, copy, b.index_count, 4, false);
		}

		// body flips: first code byte, code area, data area, codeaux table
		unsigned offsets[] = {1, 1 + size / 6, size / 2, size - 16, size - 5};
		for (unsigned off : offsets)
		{
			if (off >= size)
				continue;
			std::vector<unsigned char> copy = b.enc;
			copy[off] ^= 0xff;
			ic_put_hostile_case(f, copy, b.index_count, 4, false);
			copy[off] ^= 0x01;
			ic_put_hostile_case(f, copy, b.index_count, 4, false);
		}

		// valid stream decoded as u16 (indices < 65536)
		ic_put_hostile_case(f, b.enc, b.index_count, 2, false);
	}
}

static void gen_ic_hostile_seq(FILE* f)
{
	std::vector<unsigned> base_indices;
	for (unsigned i = 0; i < 128; ++i)
	{
		base_indices.push_back(i);
		base_indices.push_back(0xffff0000u + i);
	}

	std::vector<unsigned char> buffer(meshopt_encodeIndexSequenceBound(base_indices.size(), 0xffff0080u));
	size_t size = meshopt_encodeIndexSequence(buffer.data(), buffer.size(), base_indices.data(), base_indices.size());
	std::vector<unsigned char> enc(buffer.begin(), buffer.begin() + size);

	unsigned lengths[] = {0, 1, 2, 4, 5, unsigned(enc.size()) / 2, unsigned(enc.size()) - 1, unsigned(enc.size())};
	for (unsigned L : lengths)
	{
		if (L > enc.size())
			continue;
		std::vector<unsigned char> prefix(enc.begin(), enc.begin() + L);
		ic_put_hostile_case(f, prefix, unsigned(base_indices.size()), 4, true);
	}

	unsigned char headers[] = {0xd2, 0xdf, 0xe1, 0x01, 0xc1};
	for (unsigned char hb : headers)
	{
		std::vector<unsigned char> copy = enc;
		copy[0] = hb;
		ic_put_hostile_case(f, copy, unsigned(base_indices.size()), 4, true);
	}

	unsigned offsets[] = {1, unsigned(enc.size() / 2), unsigned(enc.size() - 5)};
	for (unsigned off : offsets)
	{
		std::vector<unsigned char> copy = enc;
		copy[off] ^= 0xff;
		ic_put_hostile_case(f, copy, unsigned(base_indices.size()), 4, true);
		copy[off] ^= 0x01;
		ic_put_hostile_case(f, copy, unsigned(base_indices.size()), 4, true);
	}
}

// --- vertex codec (FR-07) ---------------------------------------------------

enum VertexPattern
{
	VP_RANDOM = 0,
	VP_RAMP = 1,
	VP_CONSTANT = 2,
	VP_SPARSE = 3,
};

static void make_vertex_block(std::vector<unsigned char>& data, unsigned count, unsigned vertex_size, int pattern)
{
	data.assign(size_t(count) * vertex_size, 0);

	for (unsigned i = 0; i < count; ++i)
	{
		for (unsigned k = 0; k < vertex_size; k += 4)
		{
			unsigned component;
			switch (pattern)
			{
			case VP_RAMP:
				component = (i * 3 + k * 5) & 0x3f;
				break;
			case VP_CONSTANT:
				component = 0x5a;
				break;
			case VP_SPARSE:
				component = (i % 7 == 0) ? lcg() : 0;
				break;
			default:
				component = lcg();
				break;
			}
			data[size_t(i) * vertex_size + k + 0] = unsigned char(component & 0xff);
			data[size_t(i) * vertex_size + k + 1] = unsigned char((component >> 8) & 0xff);
			data[size_t(i) * vertex_size + k + 2] = unsigned char((component >> 16) & 0xff);
			data[size_t(i) * vertex_size + k + 3] = unsigned char((component >> 24) & 0xff);
		}
	}
}

static size_t vc_encode(std::vector<unsigned char>& buffer, const std::vector<unsigned char>& data, unsigned count, unsigned vertex_size, int level, int version)
{
	meshopt_encodeVertexVersion(version);
	buffer.resize(meshopt_encodeVertexBufferBound(count, vertex_size));
	size_t size = meshopt_encodeVertexBufferLevel(buffer.data(), buffer.size(), data.data(), count, vertex_size, level, -1);
	meshopt_encodeVertexVersion(1);
	if (size == 0)
	{
		fprintf(stderr, "encodeVertexBuffer failed (n=%u, s=%u, l=%d, v=%d)\n", count, vertex_size, level, version);
		exit(1);
	}
	return size;
}

static void vc_put_enc(FILE* f, const std::vector<unsigned char>& data, unsigned count, unsigned vertex_size, int level, int version)
{
	put_u32(f, count);
	put_u32(f, vertex_size);
	put_u32(f, unsigned(level));
	put_bytes(f, data.data(), size_t(count) * vertex_size);

	std::vector<unsigned char> buffer;
	size_t size = vc_encode(buffer, data, count, vertex_size, level, version);
	put_u32(f, unsigned(size));
	put_bytes(f, buffer.data(), size);
}

static void vc_put_dec(FILE* f, const std::vector<unsigned char>& data, unsigned count, unsigned vertex_size, int level, int version)
{
	std::vector<unsigned char> buffer;
	size_t size = vc_encode(buffer, data, count, vertex_size, level, version);

	put_u32(f, count);
	put_u32(f, vertex_size);
	put_u32(f, unsigned(size));
	put_u32(f, unsigned(version));
	put_bytes(f, buffer.data(), size);

	std::vector<unsigned char> dest(size_t(count) * vertex_size);
	int result = meshopt_decodeVertexBuffer(dest.data(), count, vertex_size, buffer.data(), size);
	put_i32(f, result);
	if (result == 0)
		put_bytes(f, dest.data(), dest.size());
}

static void vc_put_hostile(FILE* f, const std::vector<unsigned char>& buffer, unsigned count, unsigned vertex_size)
{
	put_u32(f, unsigned(buffer.size()));
	put_u32(f, count);
	put_u32(f, vertex_size);
	put_bytes(f, buffer.data(), buffer.size());

	std::vector<unsigned char> dest(size_t(count) * vertex_size);
	int result = meshopt_decodeVertexBuffer(dest.data(), count, vertex_size, buffer.data(), buffer.size());
	put_i32(f, result);
	if (result == 0)
		put_bytes(f, dest.data(), dest.size());
}

static void gen_vc_enc(FILE* f)
{
	struct Cfg
	{
		int pattern;
		unsigned count, vertex_size, level;
	};
	static const Cfg kCfgs[] = {
	    {VP_RANDOM, 1000, 16, 2},  {VP_RANDOM, 257, 12, 2},   {VP_RANDOM, 16, 4, 2},
	    {VP_RANDOM, 1, 8, 2},      {VP_RANDOM, 0, 12, 2},     {VP_RANDOM, 255, 20, 2},
	    {VP_RAMP, 1000, 16, 2},    {VP_RAMP, 1000, 12, 0},    {VP_RAMP, 1000, 12, 1},
	    {VP_RAMP, 1000, 12, 3},    {VP_CONSTANT, 1000, 16, 2}, {VP_CONSTANT, 16, 4, 1},
	    {VP_SPARSE, 1000, 16, 2},  {VP_SPARSE, 255, 8, 3},    {VP_RANDOM, 70, 256, 2},
	    {VP_RAMP, 64, 24, 3},      {VP_RANDOM, 64, 32, 3},    {VP_CONSTANT, 257, 64, 3},
	};

	for (const Cfg& c : kCfgs)
	{
		// reset per case so the same (pattern, count, size) corpus is identical
		// across the enc/dec/hostile commands
		lcg_reset();
		std::vector<unsigned char> data;
		make_vertex_block(data, c.count, c.vertex_size, c.pattern);
		vc_put_enc(f, data, c.count, c.vertex_size, c.level, 1);
	}
}

static void gen_vc_dec(FILE* f)
{
	struct Cfg
	{
		int pattern;
		unsigned count, vertex_size, level;
	};
	static const Cfg kCfgs[] = {
	    {VP_RANDOM, 1000, 16, 2},  {VP_RAMP, 1000, 12, 2},  {VP_CONSTANT, 1000, 16, 2},
	    {VP_SPARSE, 1000, 16, 2},  {VP_RANDOM, 16, 4, 2},   {VP_RANDOM, 1, 8, 2},
	    {VP_RANDOM, 0, 12, 2},     {VP_RAMP, 1000, 12, 0},  {VP_RAMP, 1000, 12, 3},
	    {VP_SPARSE, 255, 8, 3},    {VP_RANDOM, 257, 20, 1}, {VP_RANDOM, 70, 256, 2},
	};

	for (const Cfg& c : kCfgs)
	{
		lcg_reset();
		std::vector<unsigned char> data;
		make_vertex_block(data, c.count, c.vertex_size, c.pattern);
		vc_put_dec(f, data, c.count, c.vertex_size, c.level, 1);
	}

	// v0 decode cases (driven through the upstream version setter)
	static const Cfg kV0[] = {
	    {VP_RAMP, 1000, 12, 2},   {VP_RANDOM, 1000, 16, 2}, {VP_CONSTANT, 16, 4, 2},
	    {VP_SPARSE, 255, 8, 2},   {VP_RANDOM, 1, 8, 2},     {VP_RANDOM, 0, 12, 2},
	};
	for (const Cfg& c : kV0)
	{
		lcg_reset();
		std::vector<unsigned char> data;
		make_vertex_block(data, c.count, c.vertex_size, c.pattern);
		vc_put_dec(f, data, c.count, c.vertex_size, c.level, 0);
	}
}

static void gen_vc_hostile(FILE* f)
{
	std::vector<unsigned char> data;
	make_vertex_block(data, 100, 12, VP_RAMP);

	std::vector<unsigned char> buffer;
	size_t size = vc_encode(buffer, data, 100, 12, 2, 1);
	unsigned full = unsigned(size);

	// truncations
	unsigned lengths[] = {0, 1, 2, 24, 25, full / 2, full - 1, full};
	for (unsigned L : lengths)
	{
		if (L > full)
			continue;
		std::vector<unsigned char> prefix(buffer.begin(), buffer.begin() + L);
		vc_put_hostile(f, prefix, 100, 12);
	}

	// header mutations
	unsigned char headers[] = {0xa2, 0xaf, 0x90, 0x01, 0xe1};
	for (unsigned char hb : headers)
	{
		std::vector<unsigned char> copy = buffer;
		copy[0] = hb;
		vc_put_hostile(f, copy, 100, 12);
	}

	// body flips: control byte, byte-group header, group data, literal area
	unsigned offsets[] = {1, 2, full / 2, full - 40, full - 25};
	for (unsigned off : offsets)
	{
		if (off >= full)
			continue;
		std::vector<unsigned char> copy = buffer;
		copy[off] ^= 0xff;
		vc_put_hostile(f, copy, 100, 12);
		copy[off] ^= 0x01;
		vc_put_hostile(f, copy, 100, 12);
	}

	// tail flips: first vertex bytes and channel bytes (vsize=12: tail is
	// 9 bytes of padding + 12 bytes first_vertex + 3 channel bytes)
	unsigned tail_offsets[] = {full - 12, full - 8, full - 5, full - 2};
	for (unsigned off : tail_offsets)
	{
		if (off >= full)
			continue;
		std::vector<unsigned char> copy = buffer;
		copy[off] ^= 0xff;
		vc_put_hostile(f, copy, 100, 12);
	}
}

// --- vertex filters (FR-08) -------------------------------------------------

static void put_u16_into(std::vector<unsigned char>& v, size_t index, unsigned x, unsigned y, unsigned z, unsigned w)
{
	size_t off = index * 8;
	v[off + 0] = unsigned char(x & 0xff);
	v[off + 1] = unsigned char(x >> 8);
	v[off + 2] = unsigned char(y & 0xff);
	v[off + 3] = unsigned char(y >> 8);
	v[off + 4] = unsigned char(z & 0xff);
	v[off + 5] = unsigned char(z >> 8);
	v[off + 6] = unsigned char(w & 0xff);
	v[off + 7] = unsigned char(w >> 8);
}

static void put_filter_case(FILE* f, unsigned count, const std::vector<unsigned char>& in, const std::vector<unsigned char>& out)
{
	put_u32(f, count);
	put_bytes(f, in.data(), in.size());
	put_bytes(f, out.data(), out.size());
}

static void gen_vf_dec_oct8(FILE* f)
{
	struct Cfg
	{
		unsigned count;
		bool edges;
	};
	static const Cfg kCfgs[] = {{1, false}, {3, false}, {4, true}, {5, false}, {16, false}, {63, false}, {100, false}};

	for (const Cfg& c : kCfgs)
	{
		std::vector<unsigned char> in(size_t(c.count) * 4);
		if (c.edges)
		{
			// extreme x/y, z sentinel (255 encodes 1.f) and small z
			const unsigned char e[4] = {0x80, 0x80, 0xff, 0};
			const unsigned char e2[4] = {0x7f, 0x7f, 0xff, 0};
			const unsigned char e3[4] = {0xff, 0x00, 0x00, 0};
			const unsigned char e4[4] = {0x00, 0xff, 0x01, 0};
			memcpy(&in[0 * 4], e, 4);
			memcpy(&in[1 * 4], e2, 4);
			memcpy(&in[2 * 4], e3, 4);
			memcpy(&in[3 * 4], e4, 4);
		}
		else
		{
			for (unsigned i = 0; i < c.count; ++i)
			{
				in[size_t(i) * 4 + 0] = unsigned char(lcg() & 0xff);
				in[size_t(i) * 4 + 1] = unsigned char(lcg() & 0xff);
				in[size_t(i) * 4 + 2] = unsigned char((lcg() % 4 == 0) ? 255 : (lcg() & 0xff));
				in[size_t(i) * 4 + 3] = unsigned char(lcg() & 0xff);
			}
		}

		std::vector<unsigned char> out = in;
		meshopt_decodeFilterOct(out.data(), c.count, 4);
		put_filter_case(f, c.count, in, out);
	}
}

static void gen_vf_dec_oct16(FILE* f)
{
	for (unsigned count : {1u, 3u, 4u, 5u, 16u, 63u})	{
		std::vector<unsigned char> in(size_t(count) * 8);
		for (unsigned i = 0; i < count; ++i)
		{
			unsigned x = lcg() & 0xffff;
			unsigned y = lcg() & 0xffff;
			unsigned z = lcg() & 0x7fff;
			unsigned w = lcg() & 0xffff;
			if (i == 0)
			{
				x = 0x8000;
				y = 0x7fff;
				z = 0x7fff;
			}
			if (i == 1)
			{
				x = 0xffff;
				y = 0x0001;
				z = 0x0001;
			}
			put_u16_into(in, i, x, y, z, w);
		}

		std::vector<unsigned char> out = in;
		meshopt_decodeFilterOct(out.data(), count, 8);
		put_filter_case(f, count, in, out);
	}
}

static void gen_vf_dec_quat(FILE* f)
{
	for (unsigned count : {1u, 3u, 4u, 7u, 16u, 64u})
	{
		std::vector<unsigned char> in(size_t(count) * 8);
		for (unsigned i = 0; i < count; ++i)
		{
			unsigned x = lcg() & 0xffff;
			unsigned y = lcg() & 0xffff;
			unsigned z = lcg() & 0xffff;
			unsigned w = lcg() & 0xffff;
			if (i == 0)
			{
				x = 0x8000;
				y = 0x7fff;
				z = 0x0001;
				w = 0x0000; // sf = 3
			}
			if (i == 1)
			{
				x = 0x4000;
				y = 0xc000;
				z = 0x2000;
				w = 0x00fc; // sf = 0xff, qc = 0
			}
			put_u16_into(in, i, x, y, z, w);
		}

		std::vector<unsigned char> out = in;
		meshopt_decodeFilterQuat(out.data(), count, 8);
		put_filter_case(f, count, in, out);
	}
}

static void gen_vf_dec_exp(FILE* f)
{
	for (unsigned count : {1u, 3u, 4u, 16u, 64u})
	{
		std::vector<unsigned char> in(size_t(count) * 4);
		for (unsigned i = 0; i < count; ++i)
		{
			unsigned v = lcg();
			if (i % 8 == 0)
				v = (lcg() & 0xff) << 24; // zero mantissa
			if (i % 8 == 1)
				v = ((lcg() % 2 ? 0u : 255u) << 24) | (lcg() & 0xffffff); // extreme exponent
			unsigned char* p = &in[size_t(i) * 4];
			p[0] = unsigned char(v & 0xff);
			p[1] = unsigned char((v >> 8) & 0xff);
			p[2] = unsigned char((v >> 16) & 0xff);
			p[3] = unsigned char((v >> 24) & 0xff);
		}

		std::vector<unsigned char> out = in;
		meshopt_decodeFilterExp(out.data(), count, 4);
		put_filter_case(f, count, in, out);
	}
}

static void gen_vf_dec_color8(FILE* f)
{
	for (unsigned count : {1u, 3u, 4u, 5u, 16u, 64u})
	{
		std::vector<unsigned char> in(size_t(count) * 4);
		for (unsigned i = 0; i < count; ++i)
		{
			in[size_t(i) * 4 + 0] = unsigned char(lcg() & 0xff);       // y
			in[size_t(i) * 4 + 1] = unsigned char(lcg() & 0xff);       // co (i8)
			in[size_t(i) * 4 + 2] = unsigned char(lcg() & 0xff);       // cg (i8)
			in[size_t(i) * 4 + 3] = unsigned char((lcg() % 4 == 0) ? (lcg() & 0xff) : ((lcg() & 0x7f) | 0x80)); // a
		}

		std::vector<unsigned char> out = in;
		meshopt_decodeFilterColor(out.data(), count, 4);
		put_filter_case(f, count, in, out);
	}
}

static void gen_vf_dec_color16(FILE* f)
{
	for (unsigned count : {1u, 3u, 4u, 16u, 64u})
	{
		std::vector<unsigned char> in(size_t(count) * 8);
		for (unsigned i = 0; i < count; ++i)
		{
			unsigned y = lcg() & 0xffff;
			unsigned co = lcg() & 0xffff;
			unsigned cg = lcg() & 0xffff;
			unsigned a = (lcg() % 4 == 0) ? (lcg() & 0xffff) : ((lcg() & 0x7fff) | 0x8000);
			if (i == 0)
			{
				y = 0x8000;
				co = 0x7fff;
				cg = 0x8000;
				a = 0xffff;
			}
			put_u16_into(in, i, y, co, cg, a);
		}

		std::vector<unsigned char> out = in;
		meshopt_decodeFilterColor(out.data(), count, 8);
		put_filter_case(f, count, in, out);
	}
}

static void put_f32_vec(FILE* f, const std::vector<float>& v)
{
	for (float x : v)
		put_f32(f, x);
}

static void gen_vf_enc_oct(FILE* f)
{
	struct Cfg
	{
		unsigned stride, bits;
	};
	static const Cfg kCfgs[] = {{4, 2}, {4, 4}, {4, 8}, {8, 4}, {8, 8}, {8, 12}, {8, 16}};

	for (const Cfg& c : kCfgs)
	{
		unsigned count = 64;
		std::vector<float> data(size_t(count) * 4);
		for (unsigned i = 0; i < count; ++i)
		{
			float x = lcg_float(), y = lcg_float(), z = lcg_float();
			float l = sqrtf(x * x + y * y + z * z);
			if (l == 0.f)
				l = 1.f;
			data[size_t(i) * 4 + 0] = x / l;
			data[size_t(i) * 4 + 1] = y / l;
			data[size_t(i) * 4 + 2] = z / l;
			data[size_t(i) * 4 + 3] = lcg_float() * 0.001f;
		}
		// edges: axes, zero vector, negative, w sentinel
		const float e[4] = {1, 0, 0, 1};
		const float e2[4] = {0, -1, 0, -1};
		const float e3[4] = {0, 0, 0, 0};
		const float e4[4] = {0.57735026f, 0.57735026f, 0.57735026f, 0.25f};
		memcpy(&data[0], e, 16);
		memcpy(&data[4], e2, 16);
		memcpy(&data[8], e3, 16);
		memcpy(&data[12], e4, 16);

		put_u32(f, c.stride);
		put_u32(f, c.bits);
		put_u32(f, count);
		put_f32_vec(f, data);

		std::vector<unsigned char> dest(size_t(count) * c.stride);
		meshopt_encodeFilterOct(dest.data(), count, c.stride, c.bits, data.data());
		put_bytes(f, dest.data(), dest.size());
	}
}

static void gen_vf_enc_quat(FILE* f)
{
	for (unsigned bits : {4u, 8u, 12u, 16u})
	{
		unsigned count = 64;
		std::vector<float> data(size_t(count) * 4);
		for (unsigned i = 0; i < count; ++i)
		{
			float x = lcg_float(), y = lcg_float(), z = lcg_float(), w = lcg_float();
			float l = sqrtf(x * x + y * y + z * z + w * w);
			if (l == 0.f)
				l = 1.f;
			data[size_t(i) * 4 + 0] = x / l;
			data[size_t(i) * 4 + 1] = y / l;
			data[size_t(i) * 4 + 2] = z / l;
			data[size_t(i) * 4 + 3] = w / l;
			if (i % 8 == 0)
				for (int k = 0; k < 4; ++k)
					data[size_t(i) * 4 + k] = -data[size_t(i) * 4 + k]; // double-cover
		}
		// edges: identity, single components, zero quaternion
		const float e[4] = {0, 0, 0, 1};
		const float e2[4] = {1, 0, 0, 0};
		const float e3[4] = {0, 0, 1, 0};
		const float e4[4] = {0, 0, 0, 0};
		memcpy(&data[0], e, 16);
		memcpy(&data[4], e2, 16);
		memcpy(&data[8], e3, 16);
		memcpy(&data[12], e4, 16);

		put_u32(f, bits);
		put_u32(f, count);
		put_f32_vec(f, data);

		std::vector<unsigned char> dest(size_t(count) * 8);
		meshopt_encodeFilterQuat(dest.data(), count, 8, bits, data.data());
		put_bytes(f, dest.data(), dest.size());
	}
}

static void gen_vf_enc_exp(FILE* f)
{
	struct Cfg
	{
		unsigned stride_float, bits, mode;
	};
	static const Cfg kCfgs[] = {
	    {1, 8, 0}, {1, 24, 0}, {3, 8, 0}, {4, 8, 0}, {4, 16, 0}, {4, 1, 0}, {4, 24, 0},
	    {1, 8, 1}, {4, 8, 1}, {4, 24, 1},
	    {1, 8, 2}, {4, 8, 2}, {4, 24, 2},
	    {1, 8, 3}, {4, 8, 3}, {4, 24, 3},
	};

	for (const Cfg& c : kCfgs)
	{
		unsigned count = 64;
		std::vector<float> data(size_t(count) * c.stride_float);
		for (unsigned i = 0; i < count; ++i)
			for (unsigned j = 0; j < c.stride_float; ++j)
			{
				float v;
				switch ((i + j) % 8)
				{
				case 0:
					v = 0.f;
					break;
				case 1:
					v = -0.f;
					break;
				case 2:
					v = 1e-30f;
					break;
				case 3:
					v = 1e30f;
					break;
				case 4:
					v = -lcg_float();
					break;
				case 5:
					v = float(int(lcg() % 2001) - 1000) / 64.f;
					break;
				case 6:
					v = 1.17549435e-38f; // smallest normal
					break;
				default:
					v = lcg_float();
					break;
				}
				data[size_t(i) * c.stride_float + j] = v;
			}

		put_u32(f, c.stride_float);
		put_u32(f, c.bits);
		put_u32(f, c.mode);
		put_u32(f, count);
		put_f32_vec(f, data);

		std::vector<unsigned> dest(size_t(count) * c.stride_float);
		meshopt_encodeFilterExp(dest.data(), count, c.stride_float * 4, c.bits, data.data(), meshopt_EncodeExpMode(c.mode));
		put_bytes(f, dest.data(), dest.size() * 4);
	}
}

static void gen_vf_enc_color(FILE* f)
{
	struct Cfg
	{
		unsigned stride, bits;
	};
	static const Cfg kCfgs[] = {{4, 2}, {4, 4}, {4, 8}, {8, 8}, {8, 12}, {8, 16}};

	for (const Cfg& c : kCfgs)
	{
		unsigned count = 64;
		std::vector<float> data(size_t(count) * 4);
		for (unsigned i = 0; i < count; ++i)
		{
			for (int k = 0; k < 4; ++k)
				data[size_t(i) * 4 + k] = float(lcg() % 1001) / 1000.f;
		}
		// edges: black, white, primary/secondary colors, out-of-range (clamped)
		const float e[4] = {0, 0, 0, 1};
		const float e2[4] = {1, 1, 1, 0.5f};
		const float e3[4] = {1, 0, 0, 1};
		const float e4[4] = {0, 1, 1, 0};
		const float e5[4] = {-1, 2, 0.5f, 0.25f};
		memcpy(&data[0], e, 16);
		memcpy(&data[4], e2, 16);
		memcpy(&data[8], e3, 16);
		memcpy(&data[12], e4, 16);
		memcpy(&data[16], e5, 16);

		put_u32(f, c.stride);
		put_u32(f, c.bits);
		put_u32(f, count);
		put_f32_vec(f, data);

		std::vector<unsigned char> dest(size_t(count) * c.stride);
		meshopt_encodeFilterColor(dest.data(), count, c.stride, c.bits, data.data());
		put_bytes(f, dest.data(), dest.size());
	}
}

// --- meshlets & bounds (P1.5) ------------------------------------------------

static void gen_cb_sphere(FILE* f)
{
	struct Cfg
	{
		unsigned V, stride, radii_stride;
		bool radii;
	};
	static const Cfg kCfgs[] = {
	    {1, 12, 0, false},
	    {64, 12, 0, false},
	    {257, 16, 0, false},
	    {64, 12, 4, true},
	    {257, 16, 8, true},
	    {1000, 12, 0, false},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_positions(m, c.V, 0, c.stride, false);

		std::vector<float> radii;
		if (c.radii)
		{
			radii.assign(size_t(c.V) * (c.radii_stride / 4), 0.f);
			for (unsigned i = 0; i < c.V; ++i)
				radii[size_t(i) * (c.radii_stride / 4)] = float(lcg() % 64) / 8.f - 1.f;
		}

		put_u32(f, c.V);
		put_u32(f, c.stride);
		put_u32(f, c.radii ? 1u : 0u);
		put_u32(f, c.radii_stride);
		put_bytes(f, m.vertices.data(), m.vertices.size());
		for (float r : radii)
			put_f32(f, r);

		meshopt_Bounds bounds = meshopt_computeSphereBounds(reinterpret_cast<const float*>(m.vertices.data()), c.V, c.stride, c.radii ? radii.data() : NULL, c.radii ? c.radii_stride : 0);
		for (int k = 0; k < 3; ++k)
			put_f32(f, bounds.center[k]);
		put_f32(f, bounds.radius);
	}

	// all-identical points: zero-diameter sphere (paxisd == 0 path)
	{
		const unsigned V = 8, stride = 12;
		std::vector<unsigned char> points(size_t(V) * stride, 0);
		const float p[3] = {1.5f, -2.25f, 3.75f};
		for (unsigned i = 0; i < V; ++i)
			memcpy(&points[size_t(i) * stride], p, 12);

		put_u32(f, V);
		put_u32(f, stride);
		put_u32(f, 0);
		put_u32(f, 0);
		put_bytes(f, points.data(), points.size());

		meshopt_Bounds bounds = meshopt_computeSphereBounds(reinterpret_cast<const float*>(points.data()), V, stride, NULL, 0);
		for (int k = 0; k < 3; ++k)
			put_f32(f, bounds.center[k]);
		put_f32(f, bounds.radius);
	}
}

// --- meshlet codec (FR-06) ---------------------------------------------------

struct MeshletCorpus
{
	std::vector<unsigned> vertices;
	std::vector<unsigned char> triangles;
};

static MeshletCorpus mc_make_meshlet(const std::vector<unsigned>& indices)
{
	MeshletCorpus m;
	m.vertices.resize(256);
	m.triangles.resize(indices.size());
	size_t unique = meshopt_extractMeshletIndices(m.vertices.data(), m.triangles.data(), indices.data(), indices.size());
	m.vertices.resize(unique);
	return m;
}

static size_t mc_encode(std::vector<unsigned char>& buffer, const std::vector<unsigned>& vertices, const std::vector<unsigned char>& triangles)
{
	buffer.resize(meshopt_encodeMeshletBound(256, 512));
	size_t size = meshopt_encodeMeshlet(buffer.data(), buffer.size(), vertices.data(), vertices.size(), triangles.data(), triangles.size() / 3);
	if (size == 0)
	{
		fprintf(stderr, "encodeMeshlet failed (v=%u, t=%u)\n", unsigned(vertices.size()), unsigned(triangles.size() / 3));
		exit(1);
	}
	return size;
}

static void mc_put_enc(FILE* f, const std::vector<unsigned>& vertices, const std::vector<unsigned char>& triangles)
{
	put_u32(f, unsigned(vertices.size()));
	put_u32(f, unsigned(triangles.size() / 3));
	for (unsigned v : vertices)
		put_u32(f, v);
	put_bytes(f, triangles.data(), triangles.size());

	std::vector<unsigned char> buffer;
	size_t size = mc_encode(buffer, vertices, triangles);
	put_u32(f, unsigned(size));
	put_bytes(f, buffer.data(), size);
}

static void mc_put_enc_indices(FILE* f, const std::vector<unsigned>& indices)
{
	MeshletCorpus m = mc_make_meshlet(indices);
	mc_put_enc(f, m.vertices, m.triangles);
}

static void mc_put_dec(FILE* f, const std::vector<unsigned>& vertices, const std::vector<unsigned char>& triangles, unsigned vertex_size, unsigned triangle_size)
{
	std::vector<unsigned char> buffer;
	size_t size = mc_encode(buffer, vertices, triangles);

	put_u32(f, unsigned(vertices.size()));
	put_u32(f, vertex_size);
	put_u32(f, unsigned(triangles.size() / 3));
	put_u32(f, triangle_size);
	put_u32(f, unsigned(size));
	put_bytes(f, buffer.data(), size);

	std::vector<unsigned char> vdest(size_t(vertices.size()) * vertex_size);
	std::vector<unsigned char> tdest(size_t(triangles.size() / 3) * triangle_size);
	int result = meshopt_decodeMeshlet(vdest.data(), vertices.size(), vertex_size, tdest.data(), triangles.size() / 3, triangle_size, buffer.data(), size);
	put_i32(f, result);
	if (result == 0)
	{
		put_bytes(f, vdest.data(), vdest.size());
		put_bytes(f, tdest.data(), tdest.size());
	}
}

static void mc_put_dec_raw(FILE* f, const std::vector<unsigned>& vertices, const std::vector<unsigned char>& triangles)
{
	std::vector<unsigned char> buffer;
	size_t size = mc_encode(buffer, vertices, triangles);

	put_u32(f, unsigned(vertices.size()));
	put_u32(f, unsigned(triangles.size() / 3));
	put_u32(f, unsigned(size));
	put_bytes(f, buffer.data(), size);

	std::vector<unsigned> vdest((vertices.size() + 3) & ~3u);
	std::vector<unsigned> tdest((triangles.size() / 3 + 1) & ~1u);
	int result = meshopt_decodeMeshletRaw(vdest.data(), vertices.size(), tdest.data(), triangles.size() / 3, buffer.data(), size);
	put_i32(f, result);
	if (result == 0)
	{
		for (unsigned v : vdest)
			put_u32(f, v);
		for (unsigned t : tdest)
			put_u32(f, t);
	}
}

static void mc_put_hostile(FILE* f, const std::vector<unsigned char>& buffer, unsigned vertex_count, unsigned triangle_count, bool raw)
{
	put_u32(f, unsigned(buffer.size()));
	put_u32(f, vertex_count);
	put_u32(f, triangle_count);
	put_u32(f, raw ? 1u : 0u);
	put_bytes(f, buffer.data(), buffer.size());

	unsigned vertex_size = raw ? 4 : 4;
	unsigned triangle_size = raw ? 4 : 3;

	std::vector<unsigned char> vdest(size_t(vertex_count) * vertex_size);
	std::vector<unsigned char> tdest(size_t(triangle_count) * triangle_size);
	int result = raw ? meshopt_decodeMeshletRaw(reinterpret_cast<unsigned*>(vdest.data()), vertex_count, reinterpret_cast<unsigned*>(tdest.data()), triangle_count, buffer.data(), buffer.size())
	                 : meshopt_decodeMeshlet(vdest.data(), vertex_count, vertex_size, tdest.data(), triangle_count, triangle_size, buffer.data(), buffer.size());
	put_i32(f, result);
	if (result == 0)
	{
		put_bytes(f, vdest.data(), vdest.size());
		put_bytes(f, tdest.data(), tdest.size());
	}
}

static void mc_hostile_rounds(FILE* f, const std::vector<unsigned>& vertices, const std::vector<unsigned char>& triangles)
{
	std::vector<unsigned char> buffer;
	size_t size = mc_encode(buffer, vertices, triangles);
	unsigned full = unsigned(size);
	unsigned vertex_count = unsigned(vertices.size());
	unsigned triangle_count = unsigned(triangles.size() / 3);

	// truncations
	unsigned lengths[] = {0, 1, 2, full / 2, full - 1, full};
	for (unsigned L : lengths)
	{
		std::vector<unsigned char> prefix(buffer.begin(), buffer.begin() + L);
		mc_put_hostile(f, prefix, vertex_count, triangle_count, false);
	}

	// blind flips across data/extra area, gap area, ctrl area and codes area
	size_t codes_size = (triangle_count + 1) / 2;
	size_t ctrl_size = (vertex_count + 3) / 4;
	unsigned offsets[] = {1, full / 4, full - unsigned(codes_size), full - unsigned(codes_size + ctrl_size), full - unsigned(codes_size + ctrl_size) / 2};
	for (unsigned off : offsets)
	{
		if (off >= full)
			continue;
		std::vector<unsigned char> copy = buffer;
		copy[off] ^= 0xff;
		mc_put_hostile(f, copy, vertex_count, triangle_count, false);
		copy[off] ^= 0x01;
		mc_put_hostile(f, copy, vertex_count, triangle_count, false);
	}

	// raw decoding of the same mutated streams
	{
		std::vector<unsigned char> copy = buffer;
		copy[full / 3] ^= 0xff;
		mc_put_hostile(f, copy, vertex_count, triangle_count, true);
	}

	// valid stream decoded with a mismatched triangle count
	mc_put_hostile(f, buffer, vertex_count, triangle_count < 256 ? triangle_count + 1 : triangle_count - 1, false);
}

static void gen_mc_enc(FILE* f)
{
	// single triangle
	{
		std::vector<unsigned> v;
		ic_tri_range(v, 0, 1);
		mc_put_enc_indices(f, v);
	}

	// strip-like connectivity (heavy edge reuse)
	{
		std::vector<unsigned> v;
		ic_tri_strip(v, 0, 100);
		mc_put_enc_indices(f, v);
	}

	// sequential triangles filling a 255-vertex meshlet
	{
		std::vector<unsigned> v;
		ic_tri_range(v, 0, 85);
		mc_put_enc_indices(f, v);
	}

	// repeated triangle
	{
		std::vector<unsigned> v;
		for (unsigned t = 0; t < 40; ++t)
		{
			v.push_back(0);
			v.push_back(1);
			v.push_back(2);
		}
		mc_put_enc_indices(f, v);
	}

	// random mesh, raw and vcache-optimized
	for (int variant = 0; variant < 2; ++variant)
	{
		Mesh m;
		make_mesh(m, 64, 100, 12, 12, true);
		std::vector<unsigned> indices = m.indices;
		if (variant == 1)
		{
			std::vector<unsigned> opt(indices.size());
			meshopt_optimizeVertexCache(opt.data(), indices.data(), indices.size(), m.V);
			indices = opt;
		}
		mc_put_enc_indices(f, indices);
	}

	// meshlet-optimized variants (levels 1 and 3)
	for (int level = 1; level <= 3; level += 2)
	{
		Mesh m;
		make_mesh(m, 64, 100, 12, 12, true);
		MeshletCorpus c = mc_make_meshlet(m.indices);
		meshopt_optimizeMeshletLevel(c.vertices.data(), c.vertices.size(), c.triangles.data(), c.triangles.size() / 3, level);
		mc_put_enc(f, c.vertices, c.triangles);
	}

	// vertex-free encoding: triangle data only (decodable with vertex_count == 0)
	{
		std::vector<unsigned> vertices;
		std::vector<unsigned char> triangles;
		for (unsigned t = 0; t < 40; ++t)
		{
			triangles.push_back(0);
			triangles.push_back(1);
			triangles.push_back(2);
		}
		mc_put_enc(f, vertices, triangles);
	}
}

static std::vector<MeshletCorpus> mc_dec_corpora()
{
	std::vector<MeshletCorpus> corpora;

	{
		std::vector<unsigned> v;
		ic_tri_range(v, 0, 1);
		corpora.push_back(mc_make_meshlet(v));
	}
	{
		std::vector<unsigned> v;
		ic_tri_strip(v, 0, 100);
		corpora.push_back(mc_make_meshlet(v));
	}
	{
		std::vector<unsigned> v;
		ic_tri_range(v, 0, 85);
		corpora.push_back(mc_make_meshlet(v));
	}
	{
		Mesh m;
		make_mesh(m, 64, 100, 12, 12, true);
		std::vector<unsigned> opt(m.index_count);
		meshopt_optimizeVertexCache(opt.data(), m.indices.data(), m.index_count, m.V);
		MeshletCorpus c = mc_make_meshlet(opt);
		meshopt_optimizeMeshletLevel(c.vertices.data(), c.vertices.size(), c.triangles.data(), c.triangles.size() / 3, 3);
		corpora.push_back(c);
	}
	{
		// vertex-free meshlet
		MeshletCorpus c;
		for (unsigned t = 0; t < 40; ++t)
		{
			c.triangles.push_back(0);
			c.triangles.push_back(1);
			c.triangles.push_back(2);
		}
		corpora.push_back(c);
	}

	return corpora;
}

static void gen_mc_dec(FILE* f)
{
	for (const MeshletCorpus& c : mc_dec_corpora())
	{
		mc_put_dec(f, c.vertices, c.triangles, 4, 3);
		mc_put_dec(f, c.vertices, c.triangles, 2, 3);
		mc_put_dec(f, c.vertices, c.triangles, 4, 4);
		mc_put_dec(f, c.vertices, c.triangles, 2, 4);
	}
}

static void gen_mc_dec_raw(FILE* f)
{
	for (const MeshletCorpus& c : mc_dec_corpora())
		mc_put_dec_raw(f, c.vertices, c.triangles);
}

static void gen_mc_hostile(FILE* f)
{
	{
		std::vector<unsigned> v;
		ic_tri_strip(v, 0, 32);
		MeshletCorpus c = mc_make_meshlet(v);
		mc_hostile_rounds(f, c.vertices, c.triangles);
	}
	{
		Mesh m;
		make_mesh(m, 64, 100, 12, 12, true);
		std::vector<unsigned> opt(m.index_count);
		meshopt_optimizeVertexCache(opt.data(), m.indices.data(), m.index_count, m.V);
		MeshletCorpus c = mc_make_meshlet(opt);
		meshopt_optimizeMeshletLevel(c.vertices.data(), c.vertices.size(), c.triangles.data(), c.triangles.size() / 3, 1);
		mc_hostile_rounds(f, c.vertices, c.triangles);
	}
}

// --- clusterizer (FR-12) -----------------------------------------------------

static void put_bounds(FILE* f, const meshopt_Bounds& bounds)
{
	for (int k = 0; k < 3; ++k)
		put_f32(f, bounds.center[k]);
	put_f32(f, bounds.radius);
	for (int k = 0; k < 3; ++k)
		put_f32(f, bounds.cone_apex[k]);
	for (int k = 0; k < 3; ++k)
		put_f32(f, bounds.cone_axis[k]);
	put_f32(f, bounds.cone_cutoff);
	for (int k = 0; k < 3; ++k)
		put_u32(f, uint32_t(int8_t(bounds.cone_axis_s8[k])));
	put_u32(f, uint32_t(int8_t(bounds.cone_cutoff_s8)));
}

static void cb_put_cluster(FILE* f, const Mesh& m, unsigned stride)
{
	put_u32(f, m.index_count);
	put_u32(f, m.V);
	put_u32(f, stride);
	put_bytes(f, m.vertices.data(), m.vertices.size());
	put_indices(f, m);

	meshopt_Bounds bounds = meshopt_computeClusterBounds(m.has_indices ? m.indices.data() : NULL, m.index_count, reinterpret_cast<const float*>(m.vertices.data()), m.V, stride);
	put_bounds(f, bounds);
}

static void gen_cb_cluster(FILE* f)
{
	struct Cfg
	{
		unsigned V, T, stride;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, 12},
	    {97, 150, 16},
	    {16, 12, 12},
	    {257, 300, 12},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_position_mesh(m, c.V, c.T, c.stride);
		cb_put_cluster(f, m, c.stride);
	}

	// vcache-optimized indices (smoother cones)
	{
		Mesh m;
		make_position_mesh(m, 64, 100, 12);
		std::vector<unsigned> opt(m.index_count);
		meshopt_optimizeVertexCache(opt.data(), m.indices.data(), m.index_count, m.V);
		m.indices = opt;
		cb_put_cluster(f, m, 12);
	}

	// all-degenerate cluster (no valid triangles => trivial reject)
	{
		Mesh m;
		make_position_mesh(m, 32, 20, 12);
		for (unsigned t = 0; t < 20; ++t)
		{
			m.indices[t * 3 + 1] = m.indices[t * 3 + 0];
			m.indices[t * 3 + 2] = m.indices[t * 3 + 0];
		}
		cb_put_cluster(f, m, 12);
	}

	// single triangle
	{
		Mesh m;
		make_position_mesh(m, 8, 1, 12);
		m.indices[0] = 0, m.indices[1] = 1, m.indices[2] = 2;
		cb_put_cluster(f, m, 12);
	}

	// coplanar fan around a center vertex (open cone)
	{
		Mesh m;
		make_position_mesh(m, 16, 12, 12);
		for (unsigned t = 0; t < 12; ++t)
		{
			m.indices[t * 3 + 0] = 0;
			m.indices[t * 3 + 1] = 1 + t;
			m.indices[t * 3 + 2] = 2 + t;
		}
		cb_put_cluster(f, m, 12);
	}
}

static void gen_cb_meshlet(FILE* f)
{
	struct Cfg
	{
		unsigned V, T, stride;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, 12},
	    {97, 150, 16},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_position_mesh(m, c.V, c.T, c.stride);
		MeshletCorpus mc = mc_make_meshlet(m.indices);

		put_u32(f, unsigned(mc.vertices.size()));
		put_u32(f, unsigned(mc.triangles.size() / 3));
		put_u32(f, m.V);
		put_u32(f, c.stride);
		for (unsigned v : mc.vertices)
			put_u32(f, v);
		put_bytes(f, mc.triangles.data(), mc.triangles.size());
		put_bytes(f, m.vertices.data(), m.vertices.size());

		meshopt_Bounds bounds = meshopt_computeMeshletBounds(mc.vertices.data(), mc.triangles.data(), mc.triangles.size() / 3, reinterpret_cast<const float*>(m.vertices.data()), m.V, c.stride);
		put_bounds(f, bounds);
	}

	// optimized meshlet (level 3) bounds
	{
		Mesh m;
		make_position_mesh(m, 64, 100, 12);
		MeshletCorpus mc = mc_make_meshlet(m.indices);
		meshopt_optimizeMeshletLevel(mc.vertices.data(), mc.vertices.size(), mc.triangles.data(), mc.triangles.size() / 3, 3);

		put_u32(f, unsigned(mc.vertices.size()));
		put_u32(f, unsigned(mc.triangles.size() / 3));
		put_u32(f, m.V);
		put_u32(f, 12);
		for (unsigned v : mc.vertices)
			put_u32(f, v);
		put_bytes(f, mc.triangles.data(), mc.triangles.size());
		put_bytes(f, m.vertices.data(), m.vertices.size());

		meshopt_Bounds bounds = meshopt_computeMeshletBounds(mc.vertices.data(), mc.triangles.data(), mc.triangles.size() / 3, reinterpret_cast<const float*>(m.vertices.data()), m.V, 12);
		put_bounds(f, bounds);
	}
}

static void gen_ms_optimize(FILE* f)
{
	struct Cfg
	{
		unsigned V, T, level;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, 0}, {64, 100, 1}, {64, 100, 3}, {64, 100, 9},
	    {16, 12, 0}, {16, 12, 2}, {1, 1, 0},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_mesh(m, c.V, c.T, 12, 12, true);
		MeshletCorpus mc = mc_make_meshlet(m.indices);

		put_u32(f, unsigned(mc.vertices.size()));
		put_u32(f, unsigned(mc.triangles.size() / 3));
		put_u32(f, c.level);
		for (unsigned v : mc.vertices)
			put_u32(f, v);
		put_bytes(f, mc.triangles.data(), mc.triangles.size());

		if (c.level == 0)
			meshopt_optimizeMeshlet(mc.vertices.data(), mc.triangles.data(), mc.triangles.size() / 3, mc.vertices.size());
		else
			meshopt_optimizeMeshletLevel(mc.vertices.data(), mc.vertices.size(), mc.triangles.data(), mc.triangles.size() / 3, int(c.level));

		for (unsigned v : mc.vertices)
			put_u32(f, v);
		put_bytes(f, mc.triangles.data(), mc.triangles.size());
	}
}

static void ms_put_extract(FILE* f, const std::vector<unsigned>& indices)
{
	std::vector<unsigned> vertices(256);
	std::vector<unsigned char> triangles(indices.size());
	size_t unique = meshopt_extractMeshletIndices(vertices.data(), triangles.data(), indices.data(), indices.size());

	put_u32(f, unsigned(indices.size()));
	put_u32(f, unsigned(unique));
	for (unsigned v : indices)
		put_u32(f, v);
	for (size_t i = 0; i < unique; ++i)
		put_u32(f, vertices[i]);
	put_bytes(f, triangles.data(), triangles.size());
}

static void gen_ms_extract(FILE* f)
{
	{
		std::vector<unsigned> v;
		ic_tri_range(v, 0, 1);
		ms_put_extract(f, v);
	}
	{
		std::vector<unsigned> v;
		ic_tri_strip(v, 0, 100);
		ms_put_extract(f, v);
	}
	{
		Mesh m;
		make_mesh(m, 64, 100, 12, 12, true);
		ms_put_extract(f, m.indices);
	}
	{
		// repeated triangle: heavy cache hits
		std::vector<unsigned> v;
		for (unsigned t = 0; t < 40; ++t)
		{
			v.push_back(0);
			v.push_back(1);
			v.push_back(2);
		}
		ms_put_extract(f, v);
	}
}

static void ms_put_build(FILE* f, const Mesh& m, unsigned stride, unsigned variant, unsigned max_vertices, unsigned min_triangles, unsigned max_triangles, float weight)
{
	put_u32(f, m.index_count);
	put_u32(f, m.V);
	put_u32(f, stride);
	put_u32(f, variant);
	put_u32(f, max_vertices);
	put_u32(f, min_triangles);
	put_u32(f, max_triangles);
	put_f32(f, weight);
	put_bytes(f, m.vertices.data(), m.vertices.size());
	put_indices(f, m);

	std::vector<meshopt_Meshlet> meshlets(meshopt_buildMeshletsBound(m.index_count, max_vertices, min_triangles));
	std::vector<unsigned> meshlet_vertices(m.index_count);
	std::vector<unsigned char> meshlet_triangles(m.index_count);

	size_t meshlet_count;
	switch (variant)
	{
	case 0:
		meshlet_count = meshopt_buildMeshlets(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), m.has_indices ? m.indices.data() : NULL, m.index_count, reinterpret_cast<const float*>(m.vertices.data()), m.V, stride, max_vertices, max_triangles, weight);
		break;
	case 1:
		meshlet_count = meshopt_buildMeshletsScan(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), m.has_indices ? m.indices.data() : NULL, m.index_count, m.V, max_vertices, max_triangles);
		break;
	case 2:
		meshlet_count = meshopt_buildMeshletsFlex(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), m.has_indices ? m.indices.data() : NULL, m.index_count, reinterpret_cast<const float*>(m.vertices.data()), m.V, stride, max_vertices, min_triangles, max_triangles, weight, 1.0f);
		break;
	default:
		meshlet_count = meshopt_buildMeshletsSpatial(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), m.has_indices ? m.indices.data() : NULL, m.index_count, reinterpret_cast<const float*>(m.vertices.data()), m.V, stride, max_vertices, min_triangles, max_triangles, weight);
		break;
	}

	put_u32(f, unsigned(meshlet_count));
	for (size_t i = 0; i < meshlet_count; ++i)
	{
		put_u32(f, meshlets[i].vertex_offset);
		put_u32(f, meshlets[i].triangle_offset);
		put_u32(f, meshlets[i].vertex_count);
		put_u32(f, meshlets[i].triangle_count);
	}
	for (unsigned v : meshlet_vertices)
		put_u32(f, v);
	put_bytes(f, meshlet_triangles.data(), meshlet_triangles.size());
}

static void gen_ms_build(FILE* f)
{
	struct Cfg
	{
		unsigned V, T, stride, variant, max_vertices, min_triangles, max_triangles;
		float weight;
	};
	static const Cfg kCfgs[] = {
	    {64, 100, 12, 0, 64, 32, 32, 0.0f},
	    {64, 100, 12, 0, 64, 32, 32, 1.0f},
	    {97, 150, 16, 0, 32, 16, 32, 0.5f},
	    {257, 300, 12, 0, 256, 256, 512, 0.0f},
	    {64, 100, 12, 1, 64, 32, 32, 0.0f},
	    {16, 12, 12, 1, 8, 4, 8, 0.0f},
	    {64, 100, 12, 2, 64, 16, 32, 0.5f},
	    {97, 150, 16, 2, 32, 8, 64, 0.0f},
	    {64, 100, 12, 3, 64, 32, 32, 0.0f},
	    {257, 300, 12, 3, 64, 32, 64, 1.0f},
	    {16, 12, 12, 3, 16, 8, 16, 0.5f},
	};

	for (const Cfg& c : kCfgs)
	{
		Mesh m;
		make_position_mesh(m, c.V, c.T, c.stride);
		ms_put_build(f, m, c.stride, c.variant, c.max_vertices, c.min_triangles, c.max_triangles, c.weight);
	}
}

int main(int argc, char** argv)
{
	if (argc != 3)
	{
		fprintf(stderr, "usage: gen_driver <command> <output.bin>\n");
		return 1;
	}

	FILE* f = fopen(argv[2], "wb");
	if (!f)
	{
		fprintf(stderr, "cannot open %s\n", argv[2]);
		return 1;
	}

	const char* cmd = argv[1];

	lcg_reset();

	void (*gen)(FILE*) = NULL;
	if (strcmp(cmd, "quant-half") == 0)
		gen = gen_quant_half;
	else if (strcmp(cmd, "dequant-half") == 0)
		gen = gen_dequant_half;
	else if (strcmp(cmd, "quant-float") == 0)
		gen = gen_quant_float;
	else if (strcmp(cmd, "pos-exp") == 0)
		gen = gen_pos_exp;
	else if (strcmp(cmd, "ig-remap") == 0)
		gen = gen_ig_remap;
	else if (strcmp(cmd, "ig-remap-multi") == 0)
		gen = gen_ig_remap_multi;
	else if (strcmp(cmd, "ig-remap-custom") == 0)
		gen = gen_ig_remap_custom;
	else if (strcmp(cmd, "ig-remap-vb") == 0)
		gen = gen_ig_remap_vb;
	else if (strcmp(cmd, "ig-remap-ib") == 0)
		gen = gen_ig_remap_ib;
	else if (strcmp(cmd, "ig-filter-ib") == 0)
		gen = gen_ig_filter_ib;
	else if (strcmp(cmd, "ig-filter-multi") == 0)
		gen = gen_ig_filter_multi;
	else if (strcmp(cmd, "ig-shadow") == 0)
		gen = gen_ig_shadow;
	else if (strcmp(cmd, "ig-shadow-multi") == 0)
		gen = gen_ig_shadow_multi;
	else if (strcmp(cmd, "ig-pos-remap") == 0)
		gen = gen_ig_pos_remap;
	else if (strcmp(cmd, "ig-adjacency") == 0)
		gen = gen_ig_adjacency;
	else if (strcmp(cmd, "ig-tess") == 0)
		gen = gen_ig_tess;
	else if (strcmp(cmd, "ig-provoking") == 0)
		gen = gen_ig_provoking;
	else if (strcmp(cmd, "an-vcache") == 0)
		gen = gen_an_vcache;
	else if (strcmp(cmd, "an-vfetch") == 0)
		gen = gen_an_vfetch;
	else if (strcmp(cmd, "vf-remap") == 0)
		gen = gen_vf_remap;
	else if (strcmp(cmd, "vf-fetch") == 0)
		gen = gen_vf_fetch;
	else if (strcmp(cmd, "strip") == 0)
		gen = gen_strip;
	else if (strcmp(cmd, "unstrip") == 0)
		gen = gen_unstrip;
	else if (strcmp(cmd, "vc-cache") == 0)
		gen = gen_vc_table;
	else if (strcmp(cmd, "vc-strip") == 0)
		gen = gen_vc_strip;
	else if (strcmp(cmd, "vc-fifo") == 0)
		gen = gen_vc_fifo;
	else if (strcmp(cmd, "an-overdraw") == 0)
		gen = gen_an_overdraw;
	else if (strcmp(cmd, "an-coverage") == 0)
		gen = gen_an_coverage;
	else if (strcmp(cmd, "od") == 0)
		gen = gen_od;
	else if (strcmp(cmd, "sp-remap") == 0)
		gen = gen_sp_remap;
	else if (strcmp(cmd, "sp-tri") == 0)
		gen = gen_sp_tri;
	else if (strcmp(cmd, "sp-cluster") == 0)
		gen = gen_sp_cluster;
	else if (strcmp(cmd, "pt") == 0)
		gen = gen_pt;
	else if (strcmp(cmd, "od-debug") == 0)
		gen = gen_od_debug;
	else if (strcmp(cmd, "ic-enc-ib") == 0)
		gen = gen_ic_enc_ib;
	else if (strcmp(cmd, "ic-enc-seq") == 0)
		gen = gen_ic_enc_seq;
	else if (strcmp(cmd, "ic-dec-ib") == 0)
		gen = gen_ic_dec_ib;
	else if (strcmp(cmd, "ic-dec-seq") == 0)
		gen = gen_ic_dec_seq;
	else if (strcmp(cmd, "ic-hostile-ib") == 0)
		gen = gen_ic_hostile_ib;
	else if (strcmp(cmd, "ic-hostile-seq") == 0)
		gen = gen_ic_hostile_seq;
	else if (strcmp(cmd, "vc-enc") == 0)
		gen = gen_vc_enc;
	else if (strcmp(cmd, "vc-dec") == 0)
		gen = gen_vc_dec;
	else if (strcmp(cmd, "vc-hostile") == 0)
		gen = gen_vc_hostile;
	else if (strcmp(cmd, "vf-dec-oct8") == 0)
		gen = gen_vf_dec_oct8;
	else if (strcmp(cmd, "vf-dec-oct16") == 0)
		gen = gen_vf_dec_oct16;
	else if (strcmp(cmd, "vf-dec-quat") == 0)
		gen = gen_vf_dec_quat;
	else if (strcmp(cmd, "vf-dec-exp") == 0)
		gen = gen_vf_dec_exp;
	else if (strcmp(cmd, "vf-dec-color8") == 0)
		gen = gen_vf_dec_color8;
	else if (strcmp(cmd, "vf-dec-color16") == 0)
		gen = gen_vf_dec_color16;
	else if (strcmp(cmd, "vf-enc-oct") == 0)
		gen = gen_vf_enc_oct;
	else if (strcmp(cmd, "vf-enc-quat") == 0)
		gen = gen_vf_enc_quat;
	else if (strcmp(cmd, "vf-enc-exp") == 0)
		gen = gen_vf_enc_exp;
	else if (strcmp(cmd, "vf-enc-color") == 0)
		gen = gen_vf_enc_color;
	else if (strcmp(cmd, "cb-sphere") == 0)
		gen = gen_cb_sphere;
	else if (strcmp(cmd, "mc-enc") == 0)
		gen = gen_mc_enc;
	else if (strcmp(cmd, "mc-dec") == 0)
		gen = gen_mc_dec;
	else if (strcmp(cmd, "mc-dec-raw") == 0)
		gen = gen_mc_dec_raw;
	else if (strcmp(cmd, "mc-hostile") == 0)
		gen = gen_mc_hostile;
	else if (strcmp(cmd, "cb-cluster") == 0)
		gen = gen_cb_cluster;
	else if (strcmp(cmd, "cb-meshlet") == 0)
		gen = gen_cb_meshlet;
	else if (strcmp(cmd, "ms-optimize") == 0)
		gen = gen_ms_optimize;
	else if (strcmp(cmd, "ms-extract") == 0)
		gen = gen_ms_extract;
	else if (strcmp(cmd, "ms-build") == 0)
		gen = gen_ms_build;

	if (!gen)
	{
		fprintf(stderr, "unknown command %s\n", cmd);
		fclose(f);
		return 1;
	}

	gen(f);
	fclose(f);
	return 0;
}
