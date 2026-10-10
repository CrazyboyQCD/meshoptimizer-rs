@echo off
rem Regenerates tests\data\*.bin against upstream meshoptimizer at the pinned revision
rem (docs\adr\0001: tag v1.3 = 9e1f07b159d3cb777f1c67ed31fc11fd117986f4).
rem
rem Usage: generate.cmd <path-to-meshoptimizer-checkout>
rem Requires MSVC Build Tools; override the vcvars path via MSVC_VCVARS.
rem Fixture record formats are documented in gen_driver.cpp. Outputs are
rem little-endian and deterministic (fixed edge lists + fixed-seed LCG).
setlocal
if "%~1"=="" (
	echo usage: %~nx0 ^<path-to-meshoptimizer-checkout^>
	exit /b 1
)
if defined MSVC_VCVARS (set "VCVARS=%MSVC_VCVARS%") else (set "VCVARS=D:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat")

set "FIX=%~dp0"
set "UP=%~f1"

call "%VCVARS%" >nul 2>&1
if errorlevel 1 (
	echo vcvars failed: "%VCVARS%"
	exit /b 1
)

cd /d "%FIX%"
cl /nologo /O2 /W4 /I "%UP%\src" gen_driver.cpp "%UP%\src\quantization.cpp" "%UP%\src\indexgenerator.cpp" "%UP%\src\indexanalyzer.cpp" "%UP%\src\vfetchoptimizer.cpp" "%UP%\src\stripifier.cpp" "%UP%\src\vcacheoptimizer.cpp" "%UP%\src\rasterizer.cpp" "%UP%\src\overdrawoptimizer.cpp" "%UP%\src\spatialorder.cpp" "%UP%\src\partition.cpp" /Fe:gen_driver.exe
if errorlevel 1 exit /b 1

if not exist "%FIX%..\data" mkdir "%FIX%..\data"
gen_driver.exe quant-half "%FIX%..\data\quantization_half.bin" || exit /b 1
gen_driver.exe dequant-half "%FIX%..\data\quantization_dehalf.bin" || exit /b 1
gen_driver.exe quant-float "%FIX%..\data\quantization_float.bin" || exit /b 1
gen_driver.exe pos-exp "%FIX%..\data\quantization_posexp.bin" || exit /b 1
gen_driver.exe ig-remap "%FIX%..\data\indexgen_remap.bin" || exit /b 1
gen_driver.exe ig-remap-multi "%FIX%..\data\indexgen_remap_multi.bin" || exit /b 1
gen_driver.exe ig-remap-custom "%FIX%..\data\indexgen_remap_custom.bin" || exit /b 1
gen_driver.exe ig-remap-vb "%FIX%..\data\indexgen_remap_vb.bin" || exit /b 1
gen_driver.exe ig-remap-ib "%FIX%..\data\indexgen_remap_ib.bin" || exit /b 1
gen_driver.exe ig-filter-ib "%FIX%..\data\indexgen_filter_ib.bin" || exit /b 1
gen_driver.exe ig-filter-multi "%FIX%..\data\indexgen_filter_multi.bin" || exit /b 1
gen_driver.exe ig-shadow "%FIX%..\data\indexgen_shadow.bin" || exit /b 1
gen_driver.exe ig-shadow-multi "%FIX%..\data\indexgen_shadow_multi.bin" || exit /b 1
gen_driver.exe ig-pos-remap "%FIX%..\data\indexgen_pos_remap.bin" || exit /b 1
gen_driver.exe ig-adjacency "%FIX%..\data\indexgen_adjacency.bin" || exit /b 1
gen_driver.exe ig-tess "%FIX%..\data\indexgen_tess.bin" || exit /b 1
gen_driver.exe ig-provoking "%FIX%..\data\indexgen_provoking.bin" || exit /b 1
gen_driver.exe an-vcache "%FIX%..\data\analyze_vcache.bin" || exit /b 1
gen_driver.exe an-vfetch "%FIX%..\data\analyze_vfetch.bin" || exit /b 1
gen_driver.exe vf-remap "%FIX%..\data\vfetch_remap.bin" || exit /b 1
gen_driver.exe vf-fetch "%FIX%..\data\vfetch_fetch.bin" || exit /b 1
gen_driver.exe strip "%FIX%..\data\stripify.bin" || exit /b 1
gen_driver.exe unstrip "%FIX%..\data\unstripify.bin" || exit /b 1
gen_driver.exe vc-cache "%FIX%..\data\vcache.bin" || exit /b 1
gen_driver.exe vc-strip "%FIX%..\data\vcache_strip.bin" || exit /b 1
gen_driver.exe vc-fifo "%FIX%..\data\vcache_fifo.bin" || exit /b 1
gen_driver.exe an-overdraw "%FIX%..\data\analyze_overdraw.bin" || exit /b 1
gen_driver.exe an-coverage "%FIX%..\data\analyze_coverage.bin" || exit /b 1
gen_driver.exe od "%FIX%..\data\overdraw.bin" || exit /b 1
gen_driver.exe sp-remap "%FIX%..\data\spatial_remap.bin" || exit /b 1
gen_driver.exe sp-tri "%FIX%..\data\spatial_tri.bin" || exit /b 1
gen_driver.exe sp-cluster "%FIX%..\data\spatial_cluster.bin" || exit /b 1
gen_driver.exe pt "%FIX%..\data\partition.bin" || exit /b 1

echo fixtures regenerated under tests\data\ from upstream at "%UP%"
