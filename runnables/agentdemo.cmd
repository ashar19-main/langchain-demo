@echo off

REM Go to the project root: SampleApp1
pushd "%~dp0.."

REM Run the agent demo through Poetry
poetry run python "src\agentdemo.py" %*

REM Return to the original folder
popd