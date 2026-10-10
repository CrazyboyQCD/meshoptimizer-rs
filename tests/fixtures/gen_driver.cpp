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
	// diagnostic: dump hard/soft cluster boundaries for the first overdraw case
	Mesh m;
	make_position_mesh(m, 64, 100, 12);

	unsigned cache_size = 16;
	std::vector<unsigned> cache_timestamps(m.V);
	std::vector<unsigned> hard(m.index_count / 3);
	size_t hard_count = meshopt::generateHardBoundaries(hard.data(), m.indices.data(), m.index_count, m.V, cache_size, cache_timestamps.data());
	std::vector<unsigned> soft(m.index_count / 3 + 1);
	size_t soft_count = meshopt::generateSoftBoundaries(soft.data(), m.indices.data(), m.index_count, m.V, hard.data(), hard_count, cache_size, 1.0f, cache_timestamps.data());

	put_u32(f, unsigned(hard_count));
	for (size_t i = 0; i < hard_count; ++i)
		put_u32(f, hard[i]);
	put_u32(f, unsigned(soft_count));
	for (size_t i = 0; i < soft_count; ++i)
		put_u32(f, soft[i]);
}

// --- dispatch ---------------------------------------------------------------

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
