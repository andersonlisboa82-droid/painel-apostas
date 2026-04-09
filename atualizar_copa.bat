@echo off
cd /d %~dp0
echo.
echo ============================================================
echo    PAINEL COPA DO MUNDO 2026 - ATUALIZANDO...
echo    Buscando resultados no betexplorer.com...
echo ============================================================
echo.

python gerar_copa_mundo_html.py

if %errorlevel% neq 0 (
  echo.
  echo [ERRO] Falha ao atualizar o painel da Copa do Mundo.
  echo Verifique sua conexao com a internet e tente novamente.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo    Painel atualizado com sucesso: copa_do_mundo.html
echo ============================================================
echo.

:: Abre o HTML atualizado no navegador padrao
start "" "%~dp0copa_do_mundo.html"

pause
