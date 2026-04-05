@echo off
cd /d %~dp0
python gerar_html.py
if %errorlevel% neq 0 (
  echo.
  echo Falha ao atualizar o painel.
  pause
  exit /b 1
)
for /f %%i in ('powershell -NoProfile -Command "(Test-NetConnection -ComputerName 127.0.0.1 -Port 8765 -WarningAction SilentlyContinue).TcpTestSucceeded"') do set AI_SERVER_UP=%%i
if /I not "%AI_SERVER_UP%"=="True" (
  start "Portal AI" /min cmd /c "cd /d %~dp0 && python portal_ai_server.py"
  timeout /t 2 /nobreak >nul
)
for /f %%i in ('powershell -NoProfile -Command "(Test-NetConnection -ComputerName 127.0.0.1 -Port 8000 -WarningAction SilentlyContinue).TcpTestSucceeded"') do set INDEX_SERVER_UP=%%i
if /I not "%INDEX_SERVER_UP%"=="True" (
  start "Portal Index" /min cmd /c "cd /d %~dp0 && python -m http.server 8000 --bind 0.0.0.0"
  timeout /t 2 /nobreak >nul
)
start "" "http://127.0.0.1:8000/index.html"
exit /b 0
