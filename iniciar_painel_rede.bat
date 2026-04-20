@echo off
cd /d %~dp0
streamlit run app.py --server.port 8503 --server.address 0.0.0.0 --server.headless true
