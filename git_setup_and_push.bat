@echo off
echo ========================================
echo Git Setup and Push Script
echo ========================================
echo.

REM Initialize git if not already initialized
if not exist .git (
    echo Initializing git repository...
    git init
    echo.
)

REM Configure git (update with your info if needed)
echo Configuring git...
git config user.name "dayuanzong"
git config user.email "your-email@example.com"
echo.

REM Add remote if not exists
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo Adding remote repository...
    git remote add origin https://github.com/dayuanzong/cpp_Extract_Revision.git
    echo.
) else (
    echo Remote already exists, updating URL...
    git remote set-url origin https://github.com/dayuanzong/cpp_Extract_Revision.git
    echo.
)

REM Create .gitignore if not exists
if not exist .gitignore (
    echo Creating .gitignore...
    (
        echo # Build outputs
        echo bin/
        echo core/build/
        echo *.exe
        echo *.dll
        echo *.lib
        echo *.pdb
        echo *.exp
        echo.
        echo # Visual Studio
        echo .vs/
        echo *.user
        echo *.suo
        echo *.sln.docstates
        echo.
        echo # Python
        echo __pycache__/
        echo *.py[cod]
        echo *$py.class
        echo *.so
        echo .Python
        echo venv/
        echo env/
        echo.
        echo # Test outputs
        echo tests/*.jpg
        echo tests/*.png
        echo tests/*.json
        echo tests/test_images/
        echo.
        echo # Data
        echo data/
        echo temp/
        echo.
        echo # IDE
        echo .vscode/
        echo .idea/
        echo.
        echo # OS
        echo .DS_Store
        echo Thumbs.db
        echo desktop.ini
    ) > .gitignore
    echo.
)

REM Show status
echo Current git status:
git status
echo.

REM Add all files
echo Adding files to git...
git add .
echo.

REM Show what will be committed
echo Files to be committed:
git status --short
echo.

REM Commit
echo.
set /p commit_msg="Enter commit message (or press Enter for default): "
if "%commit_msg%"=="" set commit_msg=feat: Add 1k3d68 optimization framework with model type detection and configuration management

echo.
echo Committing with message: %commit_msg%
git commit -m "%commit_msg%"
echo.

REM Push
echo ========================================
echo Ready to push to GitHub
echo ========================================
echo.
echo Repository: https://github.com/dayuanzong/cpp_Extract_Revision.git
echo Branch: main
echo.
echo IMPORTANT: Make sure you are logged in to GitHub in your browser
echo.
pause

echo Pushing to GitHub...
git branch -M main
git push -u origin main

if errorlevel 1 (
    echo.
    echo ========================================
    echo Push failed!
    echo ========================================
    echo.
    echo This might be because:
    echo 1. You need to authenticate with GitHub
    echo 2. The repository doesn't exist yet
    echo 3. You don't have permission
    echo.
    echo Please:
    echo 1. Go to https://github.com/dayuanzong/cpp_Extract_Revision
    echo 2. Create the repository if it doesn't exist
    echo 3. Make sure you're logged in
    echo 4. Run this script again
    echo.
) else (
    echo.
    echo ========================================
    echo Success!
    echo ========================================
    echo.
    echo Code has been pushed to:
    echo https://github.com/dayuanzong/cpp_Extract_Revision
    echo.
)

pause
