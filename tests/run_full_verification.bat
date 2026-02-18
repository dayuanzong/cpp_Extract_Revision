@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
cd ..
set "ROOT_DIR=%CD%"

echo ========================================
echo 完整验证流程
echo ========================================
echo.

echo [1/6] 检查DLL路径配置...
cd /d "%ROOT_DIR%"
python "%SCRIPT_DIR%check_dll_paths.py"
if errorlevel 1 (
    echo.
    echo 配置检查失败！请查看上述错误信息。
    pause
    exit /b 1
)
echo.

echo [2/6] 重新构建C++代码...
cd /d "%ROOT_DIR%\core"
call build_cpp.bat
if errorlevel 1 (
    echo.
    echo 构建失败！请检查编译错误。
    pause
    exit /b 1
)
echo.

echo [3/6] 再次检查DLL路径...
cd /d "%ROOT_DIR%"
python "%SCRIPT_DIR%check_dll_paths.py"
if errorlevel 1 (
    echo.
    echo DLL检查失败！
    pause
    exit /b 1
)
echo.

echo [4/6] 验证DLL加载...
cd /d "%ROOT_DIR%"
python "%SCRIPT_DIR%verify_fix.py"
if errorlevel 1 (
    echo.
    echo DLL加载验证失败！
    pause
    exit /b 1
)
echo.

echo [5/6] 测试面部比例一致性...
cd /d "%ROOT_DIR%"
python "%SCRIPT_DIR%test_face_proportion.py"
echo.

echo [6/6] 验证完成！
echo.
echo ========================================
echo 所有验证步骤已完成
echo ========================================
echo.
echo 生成的文件：
echo   - verify_output.jpg （DLL加载测试输出）
echo   - test_output_*.jpg （面部比例测试输出）
echo.
echo 请查看这些图片以进行视觉验证。
echo.
cd /d "%SCRIPT_DIR%"
pause
endlocal
