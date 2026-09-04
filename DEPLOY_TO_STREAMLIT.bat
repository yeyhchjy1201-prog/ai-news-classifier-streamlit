@echo off
REM One-click public deployment entry point for this standalone package.
REM Usage: double-click this file, or run:
REM   DEPLOY_TO_STREAMLIT.bat -GitHubRemote "https://github.com/ACCOUNT/REPOSITORY.git"
setlocal EnableExtensions DisableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "PS_SCRIPT=%SCRIPT_DIR%DEPLOY_TO_STREAMLIT.ps1"
set "PS_EXE="

REM Prefer Windows PowerShell, but accept PowerShell 7 where needed.
if exist "%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe" set "PS_EXE=%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
if not defined PS_EXE if exist "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" set "PS_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not defined PS_EXE for /f "usebackq delims=" %%P in (`where powershell.exe 2^>nul`) do if not defined PS_EXE set "PS_EXE=%%P"
if not defined PS_EXE for /f "usebackq delims=" %%P in (`where pwsh.exe 2^>nul`) do if not defined PS_EXE set "PS_EXE=%%P"

if not exist "%PS_SCRIPT%" (
    echo [ERROR] DEPLOY_TO_STREAMLIT.ps1 was not found beside this batch file.
    pause
    exit /b 1
)

if not defined PS_EXE (
    echo [ERROR] PowerShell 5.1 or PowerShell 7 was not found on this computer.
    pause
    exit /b 1
)

REM ExecutionPolicy Bypass applies only to this one PowerShell process.
"%PS_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo The public-deployment preparation did not complete.
)

if /I not "%AI_NEWS_NO_PAUSE%"=="1" pause
endlocal & exit /b %EXIT_CODE%
