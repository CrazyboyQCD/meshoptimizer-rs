#!/bin/sh
# Linux twin of build.cmd: builds the C++ benchmark driver with clang against
# upstream meshoptimizer at the pinned revision (docs/adr/0001).
# MESHOPTIMIZER_NO_SIMD pins the scalar reference profile (docs/adr/0010).
#
# Usage: build.sh <path-to-meshoptimizer-checkout> [O2|O3] [simd] [avx512]
# Output: bench_cpp_<opt>[_simd] in this directory.
# "simd" drops MESHOPTIMIZER_NO_SIMD (upstream's shipped runtime-dispatched
# SIMD paths) and adds -DBENCH_SIMD (timing-only mode in the driver, mirroring
# build.cmd). "avx512" additionally enables upstream's experimental AVX512
# vertex-decode kernel: its compile gate reads compiler predefines
# (__AVX512VBMI2__ && __AVX512VBMI__ && __AVX512VL__, vertexcodec.cpp top
# guard), so only vertexcodec.cpp is compiled with the matching -m flags and
# linked as an object; every other TU stays at the baseline ISA. Upstream's
# own cpuid runtime check still guards the kernel at execution time.
set -e
if [ -z "$1" ]; then
    echo "usage: $0 <path-to-meshoptimizer-checkout> [O2|O3] [simd] [avx512]" >&2
    exit 1
fi
CXX="${CXX:-clang++}"
OPT="${2:-O2}"

CPP="$(cd "$(dirname "$0")" && pwd)"
UP="$(cd "$1" && pwd)"

PROFILE_DEF="-DMESHOPTIMIZER_NO_SIMD"
EXTRA_DEFS=""
SUFFIX=""
VC_INPUT="$UP/src/vertexcodec.cpp"
if [ "$3" = "simd" ]; then
    PROFILE_DEF=""
    EXTRA_DEFS="-DBENCH_SIMD"
    SUFFIX="_simd"
fi

if [ "$4" = "avx512" ]; then
    if [ "$3" != "simd" ]; then
        echo "avx512 requires the simd argument" >&2
        exit 1
    fi
    "$CXX" "-$OPT" -DNDEBUG -DBENCH_SIMD "-DOPT_LEVEL=$OPT" -std=c++17 \
        -mavx512vbmi2 -mavx512vbmi -mavx512vl -I "$UP/src" \
        -c "$UP/src/vertexcodec.cpp" -o "$CPP/vertexcodec_avx.o"
    VC_INPUT="$CPP/vertexcodec_avx.o"
fi

"$CXX" "-$OPT" -DNDEBUG $PROFILE_DEF $EXTRA_DEFS "-DOPT_LEVEL=$OPT" -std=c++17 \
    -I "$UP/src" \
    "$CPP/driver.cpp" \
    "$UP/src/quantization.cpp" \
    "$UP/src/indexgenerator.cpp" \
    "$UP/src/indexanalyzer.cpp" \
    "$UP/src/vfetchoptimizer.cpp" \
    "$UP/src/stripifier.cpp" \
    "$UP/src/vcacheoptimizer.cpp" \
    "$UP/src/rasterizer.cpp" \
    "$UP/src/overdrawoptimizer.cpp" \
    "$UP/src/spatialorder.cpp" \
    "$UP/src/partition.cpp" \
    "$UP/src/indexcodec.cpp" \
    "$VC_INPUT" \
    "$UP/src/vertexfilter.cpp" \
    "$UP/src/meshletutils.cpp" \
    "$UP/src/meshletcodec.cpp" \
    "$UP/src/clusterizer.cpp" \
    "$UP/src/simplifier.cpp" \
    "$UP/src/tangentspace.cpp" \
    "$UP/src/opacitymap.cpp" \
    "$UP/src/remesher.cpp" \
    "$UP/src/allocator.cpp" \
    -o "$CPP/bench_cpp_${OPT}${SUFFIX}"
echo "built $CPP/bench_cpp_${OPT}${SUFFIX}"
