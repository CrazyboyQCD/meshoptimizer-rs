@echo off
rem Regenerates tests\data\*.bin against upstream meshoptimizer at the pinned revision
rem (docs\adr\0001: tag v1.3 = 9e1f07b159d3cb777f1c67ed31fc11fd117986f4).
rem
rem Usage: generate.cmd <path-to-meshoptimizer-checkout>
rem Requires MSVC Build Tools; override the vcvars path via MSVC_VCVARS.
rem Fixture record formats are documented in gen_driver.cpp. Outputs are
rem little-endian and deterministic (fixed edge lists + fixed-seed LCG).
rem MESHOPTIMIZER_NO_SIMD pins the scalar reference profile (docs\adr\0010).
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
cl /nologo /O2 /W4 /DMESHOPTIMIZER_NO_SIMD /I "%UP%\src" gen_driver.cpp "%UP%\src\quantization.cpp" "%UP%\src\indexgenerator.cpp" "%UP%\src\indexanalyzer.cpp" "%UP%\src\vfetchoptimizer.cpp" "%UP%\src\stripifier.cpp" "%UP%\src\vcacheoptimizer.cpp" "%UP%\src\rasterizer.cpp" "%UP%\src\overdrawoptimizer.cpp" "%UP%\src\spatialorder.cpp" "%UP%\src\partition.cpp" "%UP%\src\indexcodec.cpp" "%UP%\src\vertexcodec.cpp" "%UP%\src\vertexfilter.cpp" "%UP%\src\meshletutils.cpp" "%UP%\src\meshletcodec.cpp" "%UP%\src\clusterizer.cpp" "%UP%\src\simplifier.cpp" "%UP%\src\tangentspace.cpp" "%UP%\src\opacitymap.cpp" /Fe:gen_driver.exe
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
gen_driver.exe ic-enc-ib "%FIX%..\data\ic_enc_ib.bin" || exit /b 1
gen_driver.exe ic-enc-seq "%FIX%..\data\ic_enc_seq.bin" || exit /b 1
gen_driver.exe ic-dec-ib "%FIX%..\data\ic_dec_ib.bin" || exit /b 1
gen_driver.exe ic-dec-seq "%FIX%..\data\ic_dec_seq.bin" || exit /b 1
gen_driver.exe ic-hostile-ib "%FIX%..\data\ic_hostile_ib.bin" || exit /b 1
gen_driver.exe ic-hostile-seq "%FIX%..\data\ic_hostile_seq.bin" || exit /b 1
gen_driver.exe vc-enc "%FIX%..\data\vc_enc.bin" || exit /b 1
gen_driver.exe vc-dec "%FIX%..\data\vc_dec.bin" || exit /b 1
gen_driver.exe vc-hostile "%FIX%..\data\vc_hostile.bin" || exit /b 1
gen_driver.exe vf-dec-oct8 "%FIX%..\data\vf_dec_oct8.bin" || exit /b 1
gen_driver.exe vf-dec-oct16 "%FIX%..\data\vf_dec_oct16.bin" || exit /b 1
gen_driver.exe vf-dec-quat "%FIX%..\data\vf_dec_quat.bin" || exit /b 1
gen_driver.exe vf-dec-exp "%FIX%..\data\vf_dec_exp.bin" || exit /b 1
gen_driver.exe vf-dec-color8 "%FIX%..\data\vf_dec_color8.bin" || exit /b 1
gen_driver.exe vf-dec-color16 "%FIX%..\data\vf_dec_color16.bin" || exit /b 1
gen_driver.exe vf-enc-oct "%FIX%..\data\vf_enc_oct.bin" || exit /b 1
gen_driver.exe vf-enc-quat "%FIX%..\data\vf_enc_quat.bin" || exit /b 1
gen_driver.exe vf-enc-exp "%FIX%..\data\vf_enc_exp.bin" || exit /b 1
gen_driver.exe vf-enc-color "%FIX%..\data\vf_enc_color.bin" || exit /b 1
gen_driver.exe cb-sphere "%FIX%..\data\clusterbounds_sphere.bin" || exit /b 1
gen_driver.exe mc-enc "%FIX%..\data\mc_enc.bin" || exit /b 1
gen_driver.exe mc-dec "%FIX%..\data\mc_dec.bin" || exit /b 1
gen_driver.exe mc-dec-raw "%FIX%..\data\mc_dec_raw.bin" || exit /b 1
gen_driver.exe mc-hostile "%FIX%..\data\mc_hostile.bin" || exit /b 1
gen_driver.exe cb-cluster "%FIX%..\data\cb_cluster.bin" || exit /b 1
gen_driver.exe cb-meshlet "%FIX%..\data\cb_meshlet.bin" || exit /b 1
gen_driver.exe ms-optimize "%FIX%..\data\ms_optimize.bin" || exit /b 1
gen_driver.exe ms-extract "%FIX%..\data\ms_extract.bin" || exit /b 1
gen_driver.exe ms-build "%FIX%..\data\ms_build.bin" || exit /b 1
gen_driver.exe sp-simplify "%FIX%..\data\sp_simplify.bin" || exit /b 1
gen_driver.exe sp-attrs "%FIX%..\data\sp_attrs.bin" || exit /b 1
gen_driver.exe sp-update "%FIX%..\data\sp_update.bin" || exit /b 1
gen_driver.exe sp-sloppy "%FIX%..\data\sp_sloppy.bin" || exit /b 1
gen_driver.exe sp-prune "%FIX%..\data\sp_prune.bin" || exit /b 1
gen_driver.exe sp-points "%FIX%..\data\sp_points.bin" || exit /b 1
gen_driver.exe sp-scale "%FIX%..\data\sp_scale.bin" || exit /b 1
gen_driver.exe tan-tangents "%FIX%..\data\tan_tangents.bin" || exit /b 1
gen_driver.exe tan-normals "%FIX%..\data\tan_normals.bin" || exit /b 1
gen_driver.exe om-entrysize "%FIX%..\data\om_entrysize.bin" || exit /b 1
gen_driver.exe om-rasterize "%FIX%..\data\om_rasterize.bin" || exit /b 1
gen_driver.exe om-measure "%FIX%..\data\om_measure.bin" || exit /b 1
gen_driver.exe om-compact "%FIX%..\data\om_compact.bin" || exit /b 1

echo fixtures regenerated under tests\data\ from upstream at "%UP%"
