@echo off
rem Fanfare Studio: start the local voice server (stop it with Ctrl+C or by closing this window).
rem Options are passed on, e.g.  start-studio.bat --port 5124
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo Fanfare Studio needs "uv" ^(free^). Install it once, then start this file again:
  echo   https://docs.astral.sh/uv/getting-started/installation/
  echo Fanfare Studio braucht "uv" ^(kostenlos^). Einmal installieren, dann diese Datei erneut starten:
  echo   https://docs.astral.sh/uv/getting-started/installation/
  pause
  exit /b 1
)
echo Starting Fanfare Studio. The first start downloads about 1.6 GB of voice model from Hugging Face.
echo Fanfare Studio startet. Der erste Start laedt etwa 1,6 GB Stimmenmodell von Hugging Face.
uv run --frozen --no-dev --extra onnx --extra cpu fanfare-studio %*
pause
