@echo off

REM Go to the project root: SampleApp1
pushd "%~dp0.."

REM Run the Python CLI through Poetry from the project root
poetry run python "src\llmdemo.py" %*

REM Return to the original folder
popd