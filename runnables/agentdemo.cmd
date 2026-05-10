@echo off
setlocal

REM Go to the project root: SampleApp1
pushd "%~dp0.." || exit /b 1

REM Run the agent demo through Poetry when available, with local Python fallbacks.
where poetry >nul 2>nul
if not errorlevel 1 (
    poetry run python "src\agentdemo.py" %*
    goto done
)

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" --version >nul 2>nul
    if not errorlevel 1 (
        ".venv\Scripts\python.exe" "src\agentdemo.py" %*
        goto done
    )
)

where py >nul 2>nul
if not errorlevel 1 (
    py --version >nul 2>nul
    if not errorlevel 1 (
        py "src\agentdemo.py" %*
        goto done
    )
)

where python >nul 2>nul
if not errorlevel 1 (
    python --version >nul 2>nul
    if not errorlevel 1 (
        python "src\agentdemo.py" %*
        goto done
    )
)

echo ERROR: Could not find Poetry or a usable Python interpreter.
set EXIT_CODE=1
goto cleanup

:done
set EXIT_CODE=%ERRORLEVEL%

:cleanup
REM Return to the original folder
popd

exit /b %EXIT_CODE%
