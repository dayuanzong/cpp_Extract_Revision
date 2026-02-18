@echo off
setlocal
set "ROOT_DIR=%~dp0"
set "BUILD_DIR=%ROOT_DIR%build"
set "BIN_DIR=%ROOT_DIR%..\bin"

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"

:: Clean old artifacts from bin directory
echo Cleaning old artifacts from bin...
if exist "%BIN_DIR%\FaceExtractor.exe" del /Q "%BIN_DIR%\FaceExtractor.exe"
if exist "%BIN_DIR%\FaceExtractorDLL.dll" del /Q "%BIN_DIR%\FaceExtractorDLL.dll"
if exist "%BIN_DIR%\opencv_world*.dll" del /Q "%BIN_DIR%\opencv_world*.dll"
if exist "%BIN_DIR%\onnxruntime*.dll" del /Q "%BIN_DIR%\onnxruntime*.dll"

pushd "%BUILD_DIR%"
cmake -G "Visual Studio 16 2019" -A x64 "%ROOT_DIR%"
if errorlevel 1 exit /b 1
cmake --build . --config Release
if errorlevel 1 exit /b 1

:: Copy artifacts to bin
echo Copying artifacts to bin...
copy /Y "Release\FaceExtractor.exe" "%BIN_DIR%\"
copy /Y "Release\FaceExtractorDLL.dll" "%BIN_DIR%\"
copy /Y "Release\opencv_world455.dll" "%BIN_DIR%\"
copy /Y "Release\onnxruntime.dll" "%BIN_DIR%\"
copy /Y "Release\onnxruntime_providers_cuda.dll" "%BIN_DIR%\"
copy /Y "Release\onnxruntime_providers_shared.dll" "%BIN_DIR%\"
if exist "Release\onnxruntime_providers_tensorrt.dll" copy /Y "Release\onnxruntime_providers_tensorrt.dll" "%BIN_DIR%\"

popd
echo.
echo ========================================
echo Build complete!
echo All DLLs are in: %BIN_DIR%
echo ========================================
endlocal
