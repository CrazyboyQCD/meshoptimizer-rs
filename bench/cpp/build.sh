#!/bin/sh
# Linux twin of build.cmd: builds the C++ benchmark driver with clang against
# upstream meshoptimizer at the pinned revision (docs/adr/0001).
# MESHOPTIMIZER_NO_SIMD pins the scalar reference profile (docs/adr/0010).
#
# Usage: build.sh <path-to-meshoptimizer-checkout> [O2|O3] [simd] [avx512]
# Output: bench_cpp_<opt>[_simd] in this directory.
# "simd" drops MESHOPTIMIZER_NO_SIMD (upstream's shipped runtime-dispatched
# SIMD paths) and adds -DBENCH_SIMD (timing-only mode in the driver, mirroring
# build.cmd). "avx512" additionally compiles upstream's AVX512-VBMI2 vertex
# decode kernel (-DSIMD_AVX compile gate); upstream still runtime-checks, so
# the binary stays safe on non-AVX512 hardware.
set -e
if [ -z "$1" ]; then
    echo "usage: $0 <path-to-meshoptimizer-checkout> [O2|O3] [simd] [avx512]" >&2
    exit 1
fi
CXX="${CXX:-clang++}"
OPT="${2:-O2}"

PROFILE_DEF="-DMESHOPTIMIZER_NO_SIMD"
EXTRA_DEFS=""
SUFFIX=""
if [ "$3" = "simd" ]; then
    PROFILE_DEF=""
    EXTRA_DEFS="-DBENCH_SIMD"
    SUFFIX="_simd"
fi
if [ "$4" = "avx512" ]; then
    EXTRA_DEFS="$EXTRA_DEFS -DSIMD_AVX"
fi

CPP="$(cd "$(dirname "$0")" && pwd)"
UP="$(cd "$1" && pwd)"

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
    "$UP/src/vertexcodec.cpp" \
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
