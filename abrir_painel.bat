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
for /f %%i in ('powershell -NoProfile -Command "(Test-NetConnection -ComputerName 127.0.0.1 -Port 8503 -WarningAction SilentlyContinue).TcpTestSucceeded"') do set APP_SERVER_UP=%%i
if /I not "%APP_SERVER_UP%"=="True" (
  start "Portal App" /min cmd /c "cd /d %~dp0 && streamlit run app.py --server.port 8503 --server.headless true"
  timeout /t 2 /nobreak >nul
)
start "" "http://127.0.0.1:8503/?view=app"
exit /b 0
