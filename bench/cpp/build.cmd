@echo off
rem Builds the C++ benchmark driver with clang (LLVM backend parity with rustc, ADR-0011)
rem against upstream meshoptimizer at the pinned revision (docs\adr\0001).
rem MESHOPTIMIZER_NO_SIMD pins the scalar reference profile (docs\adr\0010).
rem
rem Usage: build.cmd <path-to-meshoptimizer-checkout> [O2|O3]
rem Output: bench_cpp_<opt>.exe in this directory.
setlocal
if "%~1"=="" (
	echo usage: %~nx0 ^<path-to-meshoptimizer-checkout^> [O2^|O3]
	exit /b 1
)
set "OPT=%~2"
if "%OPT%"=="" set "OPT=O2"

set "CPP=%~dp0"
set "UP=%~f1"

rem -fuse-ld=lld: link with LLVM's lld-link so no MSVC vcvars environment is needed;
rem the linker does not affect the benchmarked code (clang codegen is identical).
clang++ -%OPT% -DNDEBUG -DMESHOPTIMIZER_NO_SIMD "-DOPT_LEVEL=%OPT%" -std=c++17 -fuse-ld=lld -I "%UP%\src" "%CPP%driver.cpp" "%UP%\src\quantization.cpp" "%UP%\src\indexgenerator.cpp" "%UP%\src\indexanalyzer.cpp" "%UP%\src\vfetchoptimizer.cpp" "%UP%\src\stripifier.cpp" "%UP%\src\vcacheoptimizer.cpp" "%UP%\src\rasterizer.cpp" "%UP%\src\overdrawoptimizer.cpp" "%UP%\src\spatialorder.cpp" "%UP%\src\partition.cpp" "%UP%\src\indexcodec.cpp" "%UP%\src\vertexcodec.cpp" "%UP%\src\vertexfilter.cpp" "%UP%\src\meshletutils.cpp" "%UP%\src\meshletcodec.cpp" "%UP%\src\clusterizer.cpp" "%UP%\src\simplifier.cpp" "%UP%\src\tangentspace.cpp" "%UP%\src\opacitymap.cpp" "%UP%\src\remesher.cpp" "%UP%\src\allocator.cpp" -o "%CPP%bench_cpp_%OPT%.exe"
if errorlevel 1 (
	echo clang build failed [%OPT%]
	exit /b 1
)
echo built %CPP%bench_cpp_%OPT%.exe
