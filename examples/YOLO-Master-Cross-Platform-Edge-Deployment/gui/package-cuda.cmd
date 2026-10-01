@echo off
setlocal EnableDelayedExpansion
rem ============================================================================
rem  Package the CUDA build of the YOLO-Master Windows Runner.
rem
rem  Unlike package.cmd (lean, ~150MB, GPU via ncnn-Vulkan / MNN-OpenCL), this
rem  bundles ONNX Runtime's CUDA execution provider plus the cuDNN and CUDA
rem  runtime DLLs it dlopens, so ONNX runs on the GPU with NOTHING installed on
rem  the target machine except an NVIDIA driver. That costs GIGABYTES - expect
rem  roughly 2GB. Ship this alongside the lean bundle, not instead of it.
rem
rem  Usage:  package-cuda.cmd [version] [full]
rem            package-cuda.cmd 1.0.0          minimal CUDA set (recommended)
rem            package-cuda.cmd 1.0.0 full     + cufft/curand/cusparse/nvrtc
rem
rem  'full' is a fallback: if the minimal bundle fails on a clean machine with
rem  "failed to load shared library", rebuild with 'full' to catch a lazily
rem  loaded library a conv detector normally never touches.
rem
rem  Requires: cmake + MSVC on PATH (VS Dev Prompt), the *GPU* ONNX Runtime
rem  package, and matching CUDA + cuDNN installs.
rem ============================================================================

set "VERSION=%~1"
if "%VERSION%"=="" set "VERSION=1.1.0"
set "FULL=%~2"

rem SDK paths come from your own sdk-paths.cmd (see sdk-paths.example.cmd),
rem or from environment variables of the same names. ONNX_ROOT must be the *GPU*
rem ONNX Runtime package here, and CUDNN_DIR / CUDA_BIN_DIR must be set.
if exist "%~dp0sdk-paths.cmd" call "%~dp0sdk-paths.cmd"
if "%MODELS_DIR%"=="" set "MODELS_DIR=..\models"
if "%OPENCV_DIR%"=="" ( echo ERROR: OPENCV_DIR not set ^(copy sdk-paths.example.cmd to sdk-paths.cmd^) & exit /b 1 )

set "ROOT=%~dp0"
set "BUILD=%ROOT%build-cuda"
set "NAME=YOLO-Master-Windows-CUDA-%VERSION%"
set "STAGE=%ROOT%dist-cuda\%NAME%"

rem ---- sanity checks: fail early with a clear reason -------------------------
if not exist "%ONNX_ROOT%/lib/onnxruntime_providers_cuda.dll" (
  echo ERROR: %ONNX_ROOT% has no onnxruntime_providers_cuda.dll
  echo        Point ONNX_ROOT at the *GPU* ONNX Runtime package.
  goto :fail
)
if not exist "%CUDNN_DIR%/cudnn64_9.dll" (
  echo ERROR: no cudnn64_9.dll in %CUDNN_DIR%
  echo        Install cuDNN 9.x for CUDA 12 and set CUDNN_DIR to its bin\12.x folder.
  goto :fail
)
if not exist "%CUDA_BIN_DIR%/cudart64_12.dll" (
  echo ERROR: no cudart64_12.dll in %CUDA_BIN_DIR%
  echo        Set CUDA_BIN_DIR to the CUDA 12.x toolkit bin folder.
  goto :fail
)

echo == 1/5  CUDA release build ==
if exist "%BUILD%" rmdir /s /q "%BUILD%"
rem CMake bundles cuDNN + CUDA runtime when these two are set (see gui/CMakeLists.txt);
rem it already skips cudnn_adv* (RNN/attention only, ~200MB a detector never loads).
cmake -S "%ROOT%." -B "%BUILD%" -A x64 ^
  "-DONNXRUNTIME_ROOT=%ONNX_ROOT%" "-DNCNN_ROOT=%NCNN_ROOT%" ^
  "-DMNN_ROOT=%MNN_ROOT%" "-DOpenCV_DIR=%OPENCV_DIR%" ^
  "-DCUDNN_DIR=%CUDNN_DIR%" "-DCUDA_BIN_DIR=%CUDA_BIN_DIR%"
if errorlevel 1 goto :fail
cmake --build "%BUILD%" --config Release -j
if errorlevel 1 goto :fail

echo == 2/5  stage Release (exe + DLLs + assets) ==
if exist "%ROOT%dist-cuda" rmdir /s /q "%ROOT%dist-cuda"
mkdir "%STAGE%"
xcopy "%BUILD%\Release\*" "%STAGE%\" /E /I /Y >nul
if errorlevel 1 goto :fail
rem drop build artifacts only -- KEEP every CUDA/cuDNN dll here, unlike package.cmd
del /f /q /s "%STAGE%\*.exp" "%STAGE%\*.lib" "%STAGE%\*.pdb" >nul 2>&1

if /i "%FULL%"=="full" (
  echo    + full CUDA set requested, copying extras
  for %%X in (cufft64_*.dll curand64_*.dll cusparse64_*.dll nvrtc64_*.dll nvrtc-builtins64_*.dll) do (
    copy /y "%CUDA_BIN_DIR%\%%X" "%STAGE%\" >nul 2>&1
  )
)

echo == 3/5  verify the CUDA dependency chain is present ==
set "MISSING="
for %%R in (onnxruntime.dll onnxruntime_providers_shared.dll onnxruntime_providers_cuda.dll ^
            cudart64_12.dll cublas64_12.dll cublasLt64_12.dll ^
            cudnn64_9.dll cudnn_graph64_9.dll cudnn_ops64_9.dll cudnn_cnn64_9.dll) do (
  if not exist "%STAGE%\%%R" set "MISSING=!MISSING! %%R"
)
if not "!MISSING!"=="" (
  echo ERROR: these required DLLs did not make it into the bundle:!MISSING!
  echo        Check CUDNN_DIR / CUDA_BIN_DIR, then re-run.
  goto :fail
)
echo    all required CUDA/cuDNN DLLs present

echo == 4/5  bundle default models ==
mkdir "%STAGE%\models" 2>nul
for %%M in (v0.1-seg-n.onnx v0.1-seg-n.mnn v0.1-seg-n.metadata.yaml) do (
  if exist "%ROOT%%MODELS_DIR%\%%M" copy /y "%ROOT%%MODELS_DIR%\%%M" "%STAGE%\models\" >nul
)
if exist "%ROOT%%MODELS_DIR%\v0.1-seg-n_ncnn" (
  xcopy "%ROOT%%MODELS_DIR%\v0.1-seg-n_ncnn" "%STAGE%\models\v0.1-seg-n_ncnn\" /E /I /Y >nul
)
if not exist "%STAGE%\models\v0.1-seg-n.onnx" (
  echo ERROR: default model models\v0.1-seg-n.onnx missing ^(looked in %MODELS_DIR%^)
  goto :fail
)
if exist "%ROOT%DISTRIBUTION-CUDA.md" (
  copy /y "%ROOT%DISTRIBUTION-CUDA.md" "%STAGE%\README.txt" >nul
) else (
  copy /y "%ROOT%DISTRIBUTION.md" "%STAGE%\README.txt" >nul
)

echo == 5/5  zip (this is big, give it a minute) ==
pushd "%ROOT%dist-cuda"
if exist "%NAME%.zip" del /q "%NAME%.zip"
tar -a -c -f "%NAME%.zip" "%NAME%"
popd
if not exist "%ROOT%dist-cuda\%NAME%.zip" goto :fail

echo.
echo Done.
echo   folder: %STAGE%
echo   zip:    %ROOT%dist-cuda\%NAME%.zip
for %%F in ("%ROOT%dist-cuda\%NAME%.zip") do echo   zip size: %%~zF bytes
echo.
echo   largest files in the bundle:
powershell -NoProfile -Command "Get-ChildItem '%STAGE%' -File | Sort-Object Length -Descending | Select-Object -First 12 | ForEach-Object { '{0,10:N1} MB  {1}' -f ($_.Length/1MB), $_.Name }" 2>nul
exit /b 0

:fail
echo.
echo PACKAGING FAILED.
exit /b 1
