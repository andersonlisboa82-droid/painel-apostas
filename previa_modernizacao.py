import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Configurações da Página
st.set_page_config(page_title="Preview Football Data Desk", layout="wide")

# CSS Customizado para Modernização (Sem mexer no Back-end)
st.markdown("""
<style>
    /* Estilo Geral */
    .main { background-color: #f8f9fa; }
    
    /* Top Bar Moderna */
    .topbar-modern {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Cards de Navegação (Carrossel Simulado) */
    .nav-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid #eee;
        cursor: pointer;
        min-width: 150px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .nav-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        border-color: #007bff;
    }
    .nav-icon { font-size: 2rem; margin-bottom: 10px; }
    
    /* Cards de Métricas */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        border-left: 5px solid #00d1b2;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #1a1a2e; }
    .metric-label { color: #666; font-size: 0.9rem; text-transform: uppercase; }

    /* Modal Styling */
    .explainer-box {
        background: #f1f3f5;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1a1a2e;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER MODERNO ---
st.markdown("""
<div class="topbar-modern">
    <div>
        <h2 style='margin:0; color:#4cc9f0;'>Football Data Desk</h2>
        <span style='opacity:0.8; font-size:0.9rem;'>Sistemas Inteligentes de Análise Esportiva</span>
    </div>
    <div style='text-align:right;'>
        <div style='font-weight:bold;'>🟢 Sistema Online</div>
        <div style='font-size:0.8rem;'>Última Calibragem: Hoje, 14:30</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- NAVEGAÇÃO EM CARROSSEL (SIMULADO) ---
st.subheader("Explore os Módulos")
cols = st.columns(5)
menu_items = [
    {"icon": "🏆", "label": "Copa 2026"},
    {"icon": "🛡️", "label": "Jogos Seguros"},
    {"icon": "🤖", "label": "IA Analista"},
    {"icon": "📊", "label": "Painel Modelo"},
    {"icon": "⚽", "label": "Resultados"}
]

for i, item in enumerate(menu_items):
    with cols[i]:
        st.markdown(f"""
        <div class="nav-card">
            <div class="nav-icon">{item['icon']}</div>
            <div style='font-weight:bold; color:#1a1a2e;'>{item['label']}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"Abrir {item['label']}", key=f"btn_{i}", use_container_width=True):
            st.toast(f"Navegando para {item['label']}...")

st.divider()

# --- ÁREA CENTRAL (CARDS INTERATIVOS) ---
st.subheader("Visão Executiva")
c1, c2, c3, c4 = st.columns(4)

metrics = [
    {"label": "ROI Médio", "value": "+12.4%", "color": "#00d1b2"},
    {"label": "Acerto Modelo", "value": "68.2%", "color": "#4361ee"},
    {"label": "Jogos Analisados", "value": "1,240", "color": "#7209b7"},
    {"label": "Value Bets", "value": "14", "color": "#f72585"}
]

for i, m in enumerate(metrics):
    with [c1, c2, c3, c4][i]:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {m['color']}">
            <div class="metric-label">{m['label']}</div>
            <div class="metric-value">{m['value']}</div>
            <div style='color:green; font-size:0.8rem;'>↑ 2.1% esta semana</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# --- CENTRAL DE COMANDOS (MODAL) ---
@st.dialog("⚙️ Central de Configurações e Calibragem")
def open_config():
    st.write("Ajuste os parâmetros do modelo e veja o impacto em tempo real.")
    
    tabs = st.tabs(["🎚️ Ajustes", "📖 Como Funciona", "📝 Histórico"])
    
    with tabs[0]:
        st.slider("Probabilidade Mínima", 0.40, 0.90, 0.55)
        st.slider("Fator de Risco (Kelly)", 0.1, 1.0, 0.25)
        if st.button("Aplicar Calibragem", type="primary", use_container_width=True):
            st.success("✅ Modelo Calibrado! Resumo: ROI Estimado subiu 0.5% | Filtros mais rigorosos aplicados.")
            st.info("**Alteração:** Probabilidade mínima alterada de 0.50 para 0.55.")
    
    with tabs[1]:
        st.markdown("""
        <div class="explainer-box">
            <strong>O que é este modelo?</strong><br>
            Nosso sistema utiliza uma rede Bayesiana ponderada por 4 fatores principais:
            1. <b>Força Ofensiva:</b> Gols esperados (xG) nos últimos 10 jogos.<br>
            2. <b>Solidez Defensiva:</b> Eficiência em interceptações e gols sofridos.<br>
            3. <b>Momento:</b> Sequência de vitórias e desfalques atuais.<br>
            4. <b>Priors:</b> Histórico histórico entre as seleções.
        </div>
        """, unsafe_allow_html=True)
        st.warning("⚠️ **Dica:** Para mercados de 'Over Gols', reduza a Probabilidade Mínima para 0.50.")

    with tabs[2]:
        st.write("Últimas alterações:")
        st.caption("- 11/04: Aumentado peso de 'Mandante' para Brasileirão.")
        st.caption("- 10/04: Atualização de odds via Scraper automatizada.")

# BOTÃO QUE CHAMA O MODAL
st.write("### Controle do Sistema")
if st.button("🛠️ Abrir Central de Comandos", use_container_width=True):
    open_config()

st.info("💡 **Dica:** Esta é uma prévia do novo design. O Back-end não foi alterado.")
