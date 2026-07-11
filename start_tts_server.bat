@echo off
REM Mornye TTS server - keep this window open while using the Stop-hook voice.
set PYTHONUTF8=1
"%~dp0GPT-SoVITS-v2pro-20250604\runtime\python.exe" -s "%~dp0scripts\tts_server.py"
pause
