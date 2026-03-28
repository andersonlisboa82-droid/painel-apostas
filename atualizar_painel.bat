@echo off
cd /d %~dp0
python gerar_html.py
if %errorlevel% neq 0 (
  echo.
  echo Falha ao atualizar o painel.
  pause
  exit /b 1
)
echo.
echo Painel atualizado com sucesso: index.html
pause
