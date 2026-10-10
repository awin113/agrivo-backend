@echo off
setlocal

echo ========================================================
echo   AGRIVO BACKEND - Hugging Face Spaces Deployment Helper
echo ========================================================
echo.

set /p HF_REPO="Enter your Hugging Face Space Git URL (e.g., https://huggingface.co/spaces/YOUR_USERNAME/agrivo-api): "

if "%HF_REPO%"=="" (
    echo Error: Repository URL cannot be empty.
    pause
    exit /b 1
)

echo.
echo Adding remote origin...
git remote remove origin 2>nul
git remote add origin %HF_REPO%

echo.
echo Renaming current branch to main...
git branch -M main

echo.
echo Pushing code and model to Hugging Face (via Git LFS)...
echo (If prompted, enter your Hugging Face username and your Access Token as the password)
git push -u origin main --force

echo.
echo ========================================================
echo Deployment push finished!
echo Check your Hugging Face Space page to view the build logs.
echo Once running, your base URL will be:
echo https://<YOUR_USERNAME>-agrivo-api.hf.space
echo ========================================================
pause
