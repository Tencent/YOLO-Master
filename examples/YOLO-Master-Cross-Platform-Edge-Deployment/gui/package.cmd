@echo off
setlocal EnableDelayedExpansion
rem ============================================================================
rem  Package the YOLO-Master Windows Runner into a self-contained zip.
rem  Batch equivalent of package.ps1 - use this when PowerShell execution policy
rem  blocks .ps1 files. Needs: cmake + MSVC on PATH (run from a VS Dev Prompt).
rem
rem  Usage:  package.cmd [version]
rem          package.cmd 1.0.0
rem
rem  Edit the SDK paths below to match your machine. IMPORTANT: ONNX_ROOT must be
rem  the *CPU* ONNX Runtime -- the gpu package would add ~600MB of CUDA provider.
rem ============================================================================

set "VERSION=%~1"
if "%VERSION%"=="" set "VERSION=1.1.0"

rem SDK paths come from your own sdk-paths.cmd (see sdk-paths.example.cmd),
rem or from environment variables of the same names.
if exist "%~dp0sdk-paths.cmd" call "%~dp0sdk-paths.cmd"
if "%MODELS_DIR%"=="" set "MODELS_DIR=..\models"
if "%OPENCV_DIR%"=="" ( echo ERROR: OPENCV_DIR not set ^(copy sdk-paths.example.cmd to sdk-paths.cmd^) & exit /b 1 )
rem NOTE: for this lean bundle ONNX_ROOT should be the *CPU* ONNX Runtime.
rem Override just for this run:  set "ONNX_ROOT=C:/dev/onnxruntime-win-x64-1.18.1"
if exist "%ONNX_ROOT%\lib\onnxruntime_providers_cuda.dll" (
  echo [warn] ONNX_ROOT looks like the GPU package; the CUDA provider will be
  echo        stripped from this bundle. Use package-cuda.cmd for a CUDA build.
)

set "ROOT=%~dp0"
set "BUILD=%ROOT%build-dist"
set "NAME=YOLO-Master-Windows-%VERSION%"
set "STAGE=%ROOT%dist\%NAME%"

echo == 1/5  clean CPU-ORT release build ==
if exist "%BUILD%" rmdir /s /q "%BUILD%"
cmake -S "%ROOT%." -B "%BUILD%" -A x64 ^
  "-DONNXRUNTIME_ROOT=%ONNX_ROOT%" "-DNCNN_ROOT=%NCNN_ROOT%" ^
  "-DMNN_ROOT=%MNN_ROOT%" "-DOpenCV_DIR=%OPENCV_DIR%"
if errorlevel 1 goto :fail
cmake --build "%BUILD%" --config Release -j
if errorlevel 1 goto :fail

echo == 2/5  stage Release (exe + DLLs + assets) ==
if exist "%ROOT%dist" rmdir /s /q "%ROOT%dist"
mkdir "%STAGE%"
xcopy "%BUILD%\Release\*" "%STAGE%\" /E /I /Y >nul
if errorlevel 1 goto :fail

echo == 3/5  strip stray CUDA / build artifacts ==
for %%P in (onnxruntime_providers_cuda*.dll onnxruntime_providers_tensorrt*.dll ^
            cudnn*.dll cublas*.dll cufft*.dll cudart*.dll curand*.dll cusparse*.dll ^
            *.exp *.lib *.pdb) do (
  del /f /q /s "%STAGE%\%%P" >nul 2>&1
)
rem del silently skips read-only files without /f (SDK archives often extract
rem read-only); a surviving CUDA provider adds ~120 MB compressed, so verify.
rem single-line ifs + exact name: parenthesized if+goto misparses in some shells.
if exist "%STAGE%\onnxruntime_providers_cuda.dll" attrib -r -s -h "%STAGE%\onnxruntime_providers_cuda.dll"
if exist "%STAGE%\onnxruntime_providers_cuda.dll" del /f /q "%STAGE%\onnxruntime_providers_cuda.dll"
if exist "%STAGE%\onnxruntime_providers_cuda.dll" echo ERROR: CUDA provider survived the strip - lean bundle must not ship it
if exist "%STAGE%\onnxruntime_providers_cuda.dll" goto :fail

echo == 4/5  bundle default models ==
mkdir "%STAGE%\models" 2>nul
for %%M in (v0.1-seg-n.onnx v0.1-seg-n.mnn v0.1-seg-n.metadata.yaml) do (
  if exist "%ROOT%%MODELS_DIR%\%%M" (
    copy /y "%ROOT%%MODELS_DIR%\%%M" "%STAGE%\models\" >nul
  ) else (
    echo   [warn] model not found, skipping: %%M
  )
)
for %%D in (v0.1-seg-n_ncnn) do (
  if exist "%ROOT%%MODELS_DIR%\%%D" xcopy "%ROOT%%MODELS_DIR%\%%D" "%STAGE%\models\%%D\" /E /I /Y >nul
)
rem the app auto-loads models\v0.1-seg-n.onnx -- never ship a bundle without it
if not exist "%STAGE%\models\v0.1-seg-n.onnx" (
  echo ERROR: default model models\v0.1-seg-n.onnx missing ^(looked in %MODELS_DIR%^)
  goto :fail
)
copy /y "%ROOT%DISTRIBUTION.md" "%STAGE%\README.txt" >nul

echo == 5/5  zip ==
rem tar.exe (bsdtar) ships with Windows 10 1803+ and writes zip via -a
pushd "%ROOT%dist"
if exist "%NAME%.zip" del /q "%NAME%.zip"
tar -a -c -f "%NAME%.zip" "%NAME%"
popd
if not exist "%ROOT%dist\%NAME%.zip" goto :fail

echo.
echo Done.
echo   folder: %STAGE%
echo   zip:    %ROOT%dist\%NAME%.zip
for %%F in ("%ROOT%dist\%NAME%.zip") do echo   size:   %%~zF bytes
exit /b 0

:fail
echo.
echo PACKAGING FAILED.
exit /b 1
