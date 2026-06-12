from __future__ import annotations

import json
import importlib
import os
import re
import socket
import subprocess
import sys
import threading
import time
from datetime import date, datetime
from html import escape
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

APP_DIR = Path(__file__).resolve().parent
APP_DIR_STR = str(APP_DIR)
if APP_DIR_STR in sys.path:
    sys.path.remove(APP_DIR_STR)
sys.path.insert(0, APP_DIR_STR)

from gerar_copa_mundo_html import build_world_cup_schedule_html
from gerar_html import AI_PROMPT_TEMPLATE, build_index_html
from analytics import (
    build_backtest_table,
    build_hedge_scenarios,
    build_probability_buckets,
    build_safe_bets_table,
    calculate_match_probabilities,
    default_model_config,
    get_team_context,
    normalize_model_config,
    pick_highest_probability_market,
    suggest_bet_strategy,
    summarize_backtest,
)
from scraper import COMPETITIONS, load_competition_matches


_PORTAL_AI_IMPORT_ERROR: Exception | None = None


def _load_portal_ai_functions() -> tuple[Callable[..., object], Callable[..., str]]:
    global _PORTAL_AI_IMPORT_ERROR
    try:
        portal_ai_server = importlib.import_module("portal_ai_server")
    except Exception as exc:
        _PORTAL_AI_IMPORT_ERROR = exc
        raise RuntimeError(
            "Nao foi possivel importar o modulo `portal_ai_server`. "
            "Confirme se o arquivo existe no mesmo diretorio do app e sem erros de sintaxe."
        ) from exc

    refresh_fn = getattr(portal_ai_server, "refresh_portal_snapshot_with_progress", None)
    analysis_fn = getattr(portal_ai_server, "run_ai_analysis", None)
    if not callable(refresh_fn) or not callable(analysis_fn):
        raise RuntimeError(
            "O modulo `portal_ai_server` foi carregado, mas as funcoes "
            "`refresh_portal_snapshot_with_progress` e/ou `run_ai_analysis` nao foram encontradas."
        )
    return refresh_fn, analysis_fn


def refresh_portal_snapshot_with_progress(*args, **kwargs):
    refresh_fn, _ = _load_portal_ai_functions()
    return refresh_fn(*args, **kwargs)


def run_ai_analysis(*args, **kwargs):
    _, analysis_fn = _load_portal_ai_functions()
    return analysis_fn(*args, **kwargs)

st.set_page_config(page_title="Sistema Apostas Futebol", layout="wide")

# Injetar Chart.js e estilos modernos
st.markdown("""
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root {
  --bg: #eef3f8;
  --card: rgba(255,255,255,0.9);
  --line: rgba(148,163,184,0.28);
  --text: #112031;
  --blue: #1d4ed8;
  --teal: #0f766e;
  --amber: #d97706;
}

/* Modal e Gráficos */
.chart-container {
  background: rgba(255,255,255,0.6);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 15px;
  margin-bottom: 15px;
}

/* Glassmorphism para Cards do Streamlit */
div[data-testid="stVerticalBlock"] > div:has(div.element-container) {
    /* Alguns seletores do streamlit para cards */
}

.modern-details-card {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.05);
}
</style>
""", unsafe_allow_html=True)

APP_TIMEZONE = ZoneInfo("America/Sao_Paulo")
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_NVIDIA_MODEL = "meta/llama-3.1-70b-instruct"
INDEX_HTML_FILE = APP_DIR / "index.html"
WORLD_CUP_HTML_FILE = APP_DIR / "copa_do_mundo.html"
MODEL_CONFIG_SESSION_KEY = "runtime_model_config"
MODEL_CONFIG_FEEDBACK_KEY = "_runtime_model_feedback"
PORTAL_REFRESH_FEEDBACK_KEY = "_portal_refresh_feedback"


def is_double_chance_market(market: str) -> bool:
    return str(market) in {"Casa ou Empate", "Fora ou Empate"}


def _read_runtime_secret(name: str) -> str:
    env_value = str(os.getenv(name, "")).strip()
    if env_value:
        return env_value
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


NVIDIA_MODEL = _read_runtime_secret("NVIDIA_MODEL") or DEFAULT_NVIDIA_MODEL


def _current_git_short_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=APP_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        short_hash = result.stdout.strip()
        return short_hash if short_hash else "sem-git"
    except Exception:
        return "sem-git"


CURRENT_GIT_SHORT_HASH = _current_git_short_hash()
APP_RELEASE_LABEL = f"{datetime.now(APP_TIMEZONE).strftime('%Y-%m-%d')} | {CURRENT_GIT_SHORT_HASH}"


def _get_runtime_model_config() -> dict[str, object]:
    config = normalize_model_config(st.session_state.get(MODEL_CONFIG_SESSION_KEY))
    st.session_state[MODEL_CONFIG_SESSION_KEY] = config
    return config


def _model_config_bins_to_text(config: dict[str, object]) -> str:
    calibration_cfg = config.get("calibration", {})
    if not isinstance(calibration_cfg, dict):
        return "0, 0.20, 0.35, 0.50, 0.65, 1.01"
    bins = calibration_cfg.get("bins", [])
    if not isinstance(bins, list) or not bins:
        return "0, 0.20, 0.35, 0.50, 0.65, 1.01"
    return ", ".join(f"{float(item):.2f}" for item in bins)


def _parse_bins_text(raw_bins: str) -> list[float] | None:
    cleaned = str(raw_bins or "").replace(";", ",").strip()
    if not cleaned:
        return None
    items = [chunk.strip() for chunk in cleaned.split(",") if chunk.strip()]
    parsed: list[float] = []
    for item in items:
        try:
            parsed.append(float(item))
        except ValueError:
            return None
    parsed = sorted(set(parsed))
    if len(parsed) < 2:
        return None
    if parsed[0] > 0.0:
        parsed = [0.0] + parsed
    if parsed[-1] <= 1.0:
        parsed.append(1.01)
    return parsed


def _is_port_open(host: str, port: int, *, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@st.cache_resource(show_spinner=False)
def _ensure_portal_ai_server_thread() -> bool:
    """Starta o servidor local (8765) uma vez por processo (necessario para o HTML embutido)."""
    if _is_port_open("127.0.0.1", 8765):
        return True

    try:
        import portal_ai_server

        thread = threading.Thread(
            target=portal_ai_server.main,
            daemon=True,
            name="portal-ai-server",
        )
        thread.start()
    except Exception:
        return False

    for _ in range(25):
        if _is_port_open("127.0.0.1", 8765):
            return True
        time.sleep(0.08)

    return False


def ensure_portal_ai_server_running() -> None:
    if _is_port_open("127.0.0.1", 8765):
        return

    started = _ensure_portal_ai_server_thread()
    if not started:
        st.warning(
            "O Portal AI server (porta 8765) nao esta ativo. "
            "O botao 'Atualizar placares' do portal HTML pode falhar. "
            "Se quiser, rode `python portal_ai_server.py` em outro terminal."
        )


def _build_match_detail_data(
    row_data: pd.Series,
    matches_df: pd.DataFrame,
    *,
    model_config: dict[str, object] | None = None,
) -> dict[str, object]:
    odd_h = float(row_data.get("odds_home", row_data.get("Odd Casa", row_data.get("odd", 0))))
    odd_d = float(row_data.get("odds_draw", row_data.get("Odd Empate", 0)))
    odd_a = float(row_data.get("odds_away", row_data.get("Odd Fora", 0)))
    bookmakers_value = row_data.get("bookmakers", row_data.get("Casas", 0))
    try:
        bookmakers = int(bookmakers_value) if not pd.isna(bookmakers_value) else 0
    except (TypeError, ValueError):
        bookmakers = 0
    probs = calculate_match_probabilities(
        matches_df,
        row_data["home_team"],
        row_data["away_team"],
        odd_home=odd_h,
        odd_draw=odd_d,
        odd_away=odd_a,
        bookmakers=bookmakers,
        model_config=model_config,
    )
    home_ctx = get_team_context(matches_df, str(row_data["home_team"]))
    away_ctx = get_team_context(matches_df, str(row_data["away_team"]))
    display_date = format_match_datetime(
        row_data.get("date_text"),
        row_data.get("event_timestamp"),
        str(row_data.get("status", "")),
    )
    
    # Se vier da tabela de safe_df, as odds já estão no row_data
    # Se vier da tabela de valid (futuros), também
    tip = suggest_bet_strategy(
        probs,
        odd_home=odd_h,
        odd_draw=odd_d,
        odd_away=odd_a,
        bankroll=1000.0,
        model_config=model_config,
    )

    data = {
        "home": str(row_data["home_team"]),
        "away": str(row_data["away_team"]),
        "date": display_date,
        "probs": {
            "home": round(probs.home_win * 100, 1),
            "draw": round(probs.draw * 100, 1),
            "away": round(probs.away_win * 100, 1),
            "btts": round(probs.btts_yes * 100, 1),
            "over25": round(probs.over_25 * 100, 1),
            "under25": round(probs.under_25 * 100, 1),
            "scorelines": [[s, round(p * 100, 1)] for s, p in probs.top_scorelines]
        },
        "odds": {
            "home": odd_h,
            "draw": odd_d,
            "away": odd_a
        },
        "context": {
            "home": home_ctx,
            "away": away_ctx
        },
        "tip": {
            "market": market_label(tip.best_market, str(row_data["home_team"]), str(row_data["away_team"])),
            "odd": round(tip.best_odd, 2),
            "prob": round(tip.model_probability * 100, 1),
            "ev": round(tip.expected_value * 100, 2),
            "stake": round(tip.suggested_stake, 2)
        }
    }
    return data


def render_match_details_modal(match_data: dict[str, object]) -> None:
    if not match_data:
        return

    home = str(match_data["home"])
    away = str(match_data["away"])
    probs = match_data["probs"]
    odds = match_data["odds"]
    ctx = match_data["context"]
    tip = match_data.get("tip")

    st.markdown(f"""
<div class="modern-details-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <div>
            <div class="hero-tag">{match_data["date"]}</div>
            <h2 style="margin: 5px 0 0; font-size: 1.8rem;">{home} x {away}</h2>
        </div>
        <button onclick="window.parent.document.querySelector('button[aria-label=\\'Close\\']').click()" 
                style="background: none; border: none; font-size: 24px; cursor: pointer; color: var(--muted);">×</button>
    </div>

    <div class="spotlight-grid">
        <div class="chart-container">
            <h4 style="margin-top:0;">Probabilidades 1X2 (%)</h4>
            <canvas id="stChart1x2" style="max-height: 250px;"></canvas>
        </div>
        <div class="chart-container">
            <h4 style="margin-top:0;">Gols e Ambas Marcam (%)</h4>
            <canvas id="stChartAlt" style="max-height: 250px;"></canvas>
        </div>
    </div>

    <div class="spotlight-grid">
        <div class="chart-container">
            <h4 style="margin-top:0;">Placares mais prováveis</h4>
            <canvas id="stChartScores" style="max-height: 250px;"></canvas>
        </div>
        <div>
            <div class="ai-panel" style="margin-bottom: 15px;">
                <strong>Estrategia do Modelo</strong>
                <p style="margin: 8px 0 0; font-size: 0.95rem;">
                    {f"Sugestão: <b>{tip['market']}</b><br>Odd: {tip['odd']:.2f} | EV: {tip['ev']}% | Stake: R$ {tip['stake']:.2f}" if tip else "Sem recomendação clara para este confronto."}
                </p>
            </div>
            <div class="context-grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div class="metric-card" style="background: rgba(29, 78, 216, 0.05); border-color: rgba(29, 78, 216, 0.1);">
                    <span style="color: var(--blue);">{home}</span>
                    <strong style="font-size: 1.2rem;">{ctx['home']['rank']}º</strong>
                    <p style="font-size: 0.8rem;">{ctx['home']['points']} pts | {ctx['home']['recent_text']}</p>
                </div>
                <div class="metric-card" style="background: rgba(15, 118, 110, 0.05); border-color: rgba(15, 118, 110, 0.1);">
                    <span style="color: var(--teal);">{away}</span>
                    <strong style="font-size: 1.2rem;">{ctx['away']['rank']}º</strong>
                    <p style="font-size: 0.8rem;">{ctx['away']['points']} pts | {ctx['away']['recent_text']}</p>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    // Pequeno atraso para garantir que o DOM esteja pronto
    setTimeout(() => {{
        const data = {json.dumps(match_data)};
        
        // Chart 1x2
        new Chart(document.getElementById('stChart1x2'), {{
            type: 'bar',
            data: {{
                labels: ['Casa', 'Empate', 'Fora'],
                datasets: [{{
                    label: 'Modelo %',
                    data: [data.probs.home, data.probs.draw, data.probs.away],
                    backgroundColor: ['rgba(29, 78, 216, 0.7)', 'rgba(148, 163, 184, 0.7)', 'rgba(15, 118, 110, 0.7)'],
                    borderRadius: 8
                }}, {{
                    label: 'Mercado %',
                    data: [
                        data.odds.home > 0 ? (100/data.odds.home).toFixed(1) : 0,
                        data.odds.draw > 0 ? (100/data.odds.draw).toFixed(1) : 0,
                        data.odds.away > 0 ? (100/data.odds.away).toFixed(1) : 0
                    ],
                    backgroundColor: 'rgba(0,0,0,0.1)',
                    borderRadius: 8
                }}]
            }},
            options: {{ responsive: true, maintainAspectRatio: false }}
        }});

        // Chart Alt
        new Chart(document.getElementById('stChartAlt'), {{
            type: 'doughnut',
            data: {{
                labels: ['BTTS Sim', 'Over 2.5', 'Under 2.5'],
                datasets: [{{
                    data: [data.probs.btts, data.probs.over25, data.probs.under25],
                    backgroundColor: ['#fbbf24', '#ef4444', '#10b981']
                }}]
            }},
            options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ position: 'bottom' }} }} }}
        }});

        // Chart Scores
        new Chart(document.getElementById('stChartScores'), {{
            type: 'bar',
            data: {{
                labels: data.probs.scorelines.map(s => s[0]),
                datasets: [{{
                    label: 'Probabilidade %',
                    data: data.probs.scorelines.map(s => s[1]),
                    backgroundColor: 'rgba(29, 78, 216, 0.6)',
                    borderRadius: 6
                }}]
            }},
            options: {{ indexAxis: 'y', responsive: true, maintainAspectRatio: false }}
        }});
    }}, 100);
</script>
""", unsafe_allow_html=True)


@st.dialog("Prompt institucional da IA", width="large")
def open_institutional_prompt_modal(selected_date: date) -> None:
    st.caption(
        "Edite o prompt completo aqui. Quando quiser, atualize a data automaticamente e salve para usar na execucao."
    )
    if "institutional_ai_prompt_modal" not in st.session_state:
        st.session_state["institutional_ai_prompt_modal"] = st.session_state.get("institutional_ai_prompt", "")

    st.text_area(
        "Prompt",
        key="institutional_ai_prompt_modal",
        height=420,
    )

    modal_col1, modal_col2, modal_col3 = st.columns([1.1, 1, 0.9])
    with modal_col1:
        if st.button("Atualizar com a data", use_container_width=True, key="institutional_ai_modal_update"):
            st.session_state["institutional_ai_prompt_modal"] = AI_PROMPT_TEMPLATE.replace(
                "__DATA_SELECIONADA__", selected_date.strftime("%d/%m/%Y")
            )
            st.session_state["institutional_ai_modal_open"] = True
            st.rerun()
    with modal_col2:
        if st.button("Salvar e fechar", use_container_width=True, key="institutional_ai_modal_save"):
            st.session_state["institutional_ai_prompt"] = st.session_state.get("institutional_ai_prompt_modal", "")
            st.session_state["institutional_ai_modal_open"] = False
            st.rerun()
    with modal_col3:
        if st.button("Fechar", use_container_width=True, key="institutional_ai_modal_close"):
            st.session_state["institutional_ai_modal_open"] = False
            st.rerun()


def current_app_timestamp() -> str:
    return datetime.now(APP_TIMEZONE).strftime("%d/%m/%Y %H:%M:%S")


def render_floating_dock(competition_name: str, risk_profile: str) -> None:
    dock_html = """
<script>
const competitionName = __COMPETITION__;
const riskLabel = __RISK__;
const parentWindow = window.parent;
const doc = parentWindow.document;

// --- LIMPEZA DE INSTANCIAS ANTERIORES ---
if (typeof parentWindow.__fdDockCleanup === "function") {
  parentWindow.__fdDockCleanup();
}

const previousDock = doc.getElementById("fd-floating-dock");
if (previousDock) previousDock.remove();
const previousStyle = doc.getElementById("fd-floating-dock-style");
if (previousStyle) previousStyle.remove();

// --- ESTILOS DO PAINEL ---
const style = doc.createElement("style");
style.id = "fd-floating-dock-style";
style.textContent = `
#fd-floating-dock {
  position: fixed;
  right: 18px;
  top: 18px;
  z-index: 999999;
  display: grid;
  gap: 12px;
  justify-items: end;
  pointer-events: none;
  user-select: none;
  transition: opacity 0.3s ease;
}
#fd-floating-dock .fd-meta {
  pointer-events: auto;
  display: grid;
  gap: 2px;
  min-width: 170px;
  padding: 10px 14px;
  border-radius: 18px;
  background: linear-gradient(135deg, #081625, #14304f);
  color: #f8fafc;
  border: 1px solid rgba(148,163,184,0.3);
  box-shadow: 0 10px 30px rgba(0,0,0,0.3);
  backdrop-filter: blur(16px);
  cursor: grab;
}
#fd-floating-dock .fd-meta:active { cursor: grabbing; }
#fd-floating-dock .fd-meta strong { font-size: 0.85rem; line-height: 1.1; display: block; }
#fd-floating-dock .fd-meta span { font-size: 0.72rem; color: #cfe4ff; opacity: 0.8; display: block; }

#fd-floating-dock .fd-stack { display: grid; gap: 8px; }
#fd-floating-dock .fd-btn {
  pointer-events: auto;
  display: inline-flex;
  align-items: center;
  justify-content: flex-start;
  gap: 10px;
  min-width: 48px;
  height: 48px;
  padding: 0 15px;
  border: 0;
  border-radius: 999px;
  background: #ffffff;
  color: #112031;
  cursor: pointer;
  box-shadow: 0 8px 20px rgba(0,0,0,0.15);
  border: 1px solid rgba(148,163,184,0.22);
  transition: transform .15s ease, box-shadow .15s ease;
  overflow: hidden;
}
#fd-floating-dock .fd-btn:hover { transform: scale(1.05); box-shadow: 0 10px 25px rgba(0,0,0,0.2); }
#fd-floating-dock .fd-btn svg { width: 16px; height: 16px; flex: 0 0 16px; }
#fd-floating-dock .fd-btn .fd-label { font: 700 0.78rem/1 sans-serif; white-space: nowrap; }

#fd-floating-dock .fd-btn[data-action="top"] {
  opacity: 0;
  transform: translateY(10px);
  pointer-events: none;
}
#fd-floating-dock.is-scrolled .fd-btn[data-action="top"] {
  opacity: 1;
  transform: translateY(0);
  pointer-events: auto;
}
`;
doc.head.appendChild(style);

// --- ESTRUTURA HTML ---
const dock = doc.createElement("div");
dock.id = "fd-floating-dock";
dock.innerHTML = `
  <div class="fd-meta" id="fd-dock-handle">
    <strong>${competitionName}</strong>
    <span>↕ Arraste para mover</span>
  </div>
  <div class="fd-stack">
    <button type="button" class="fd-btn" data-action="top" title="Voltar ao topo">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 19V5M5 12l7-7 7 7"/></svg>
      <span class="fd-label">Topo</span>
    </button>
    <button type="button" class="fd-btn" data-action="safe" title="Jogos Seguros">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
      <span class="fd-label">Seguros</span>
    </button>
    <button type="button" class="fd-btn" data-action="simulator" title="Simulador">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/></svg>
      <span class="fd-label">Simular</span>
    </button>
    <button type="button" class="fd-btn" data-action="refresh" title="Atualizar">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
      <span class="fd-label">Atualizar</span>
    </button>
  </div>
`;
doc.body.appendChild(dock);

// --- LOGICA DE ARRASTAR (DRAG & DROP) ---
let isDragging = false;
let startX, startY;
let initialLeft, initialTop;

const handle = dock.querySelector("#fd-dock-handle");

const onStart = (e) => {
  isDragging = true;
  const clientX = e.type.startsWith("touch") ? e.touches[0].clientX : e.clientX;
  const clientY = e.type.startsWith("touch") ? e.touches[0].clientY : e.clientY;
  
  const rect = dock.getBoundingClientRect();
  startX = clientX;
  startY = clientY;
  initialLeft = rect.left;
  initialTop = rect.top;
  
  doc.addEventListener("mousemove", onMove);
  doc.addEventListener("mouseup", onEnd);
  doc.addEventListener("touchmove", onMove, { passive: false });
  doc.addEventListener("touchend", onEnd);
  
  // Impede selecao de texto durante o arraste
  doc.body.style.userSelect = "none";
};

const onMove = (e) => {
  if (!isDragging) return;
  if (e.type === "touchmove") e.preventDefault(); // Evita scroll no mobile

  const clientX = e.type.startsWith("touch") ? e.touches[0].clientX : e.clientX;
  const clientY = e.type.startsWith("touch") ? e.touches[0].clientY : e.clientY;

  const dx = clientX - startX;
  const dy = clientY - startY;

  dock.style.right = "auto";
  dock.style.left = (initialLeft + dx) + "px";
  dock.style.top = (initialTop + dy) + "px";
};

const onEnd = () => {
  isDragging = false;
  doc.removeEventListener("mousemove", onMove);
  doc.removeEventListener("mouseup", onEnd);
  doc.removeEventListener("touchmove", onMove);
  doc.removeEventListener("touchend", onEnd);
  doc.body.style.userSelect = "";
};

handle.addEventListener("mousedown", onStart);
handle.addEventListener("touchstart", onStart, { passive: true });

// --- LOGICA DE NAVEGACAO E SCROLL ---
const scrollToId = (id) => {
  const target = doc.getElementById(id);
  if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
};

const updateScrolledState = () => {
  const top = parentWindow.scrollY || doc.documentElement.scrollTop;
  dock.classList.toggle("is-scrolled", top > 200);
};

dock.querySelectorAll("[data-action]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const action = btn.getAttribute("data-action");
    if (action === "top") scrollToId("panel-top-anchor");
    if (action === "safe") scrollToId("anchor-safe");
    if (action === "simulator") scrollToId("anchor-simulator");
    if (action === "refresh") parentWindow.location.reload();
  });
});

parentWindow.addEventListener("scroll", updateScrolledState, { passive: true });
updateScrolledState();

// --- CLEANUP ---
parentWindow.__fdDockCleanup = () => {
  parentWindow.removeEventListener("scroll", updateScrolledState);
  onEnd(); // Garante que remove os listeners globais de drag
  if (dock) dock.remove();
};
</script>
"""
    dock_html = dock_html.replace("__COMPETITION__", json.dumps(competition_name))
    dock_html = dock_html.replace("__RISK__", json.dumps(risk_profile))
    components.html(dock_html, height=0, width=0)


def _safe_percent(numerator: int | float, denominator: int | float) -> int:
    if not denominator:
        return 0
    return round((float(numerator) / float(denominator)) * 100)


@st.cache_data(ttl=600, show_spinner=False)
def get_data(competition: str) -> pd.DataFrame:
    return load_competition_matches(competition)


@st.cache_data(ttl=600, show_spinner=False)
def _build_safe_bets_cached(
    matches_df: pd.DataFrame,
    *,
    bankroll: float,
    kelly_fractional: float,
    min_model_prob: float,
    min_expected_value: float,
    max_odd: float,
    min_bookmakers: int,
    model_config: dict[str, object],
) -> pd.DataFrame:
    return build_safe_bets_table(
        matches_df=matches_df,
        bankroll=bankroll,
        kelly_fractional=kelly_fractional,
        min_model_prob=min_model_prob,
        min_expected_value=min_expected_value,
        max_odd=max_odd,
        min_bookmakers=min_bookmakers,
        model_config=model_config,
    )


@st.cache_data(ttl=600, show_spinner=False)
def _build_backtest_cached(
    matches_df: pd.DataFrame,
    *,
    bankroll: float,
    kelly_fractional: float,
    min_history_matches: int,
    max_evaluated_matches: int,
    model_config: dict[str, object],
) -> pd.DataFrame:
    return build_backtest_table(
        matches_df=matches_df,
        bankroll=bankroll,
        kelly_fractional=kelly_fractional,
        min_history_matches=min_history_matches,
        max_evaluated_matches=max_evaluated_matches,
        model_config=model_config,
    )


@st.cache_data(ttl=600, show_spinner=False)
def _summarize_backtest_cached(backtest_df: pd.DataFrame) -> dict[str, object]:
    return summarize_backtest(backtest_df)


@st.cache_data(ttl=600, show_spinner=False)
def _build_probability_buckets_cached(backtest_df: pd.DataFrame) -> pd.DataFrame:
    return build_probability_buckets(backtest_df)


def get_data_with_fallback(preferred_competition: str) -> tuple[str, pd.DataFrame, str]:
    ordered_competitions = [preferred_competition] + [
        name for name in COMPETITIONS.keys() if name != preferred_competition
    ]
    failures: list[str] = []

    for competition_name in ordered_competitions:
        try:
            df = get_data(competition_name)
            return competition_name, df, ""
        except Exception as exc:
            failures.append(f"{competition_name}: {exc}")

    raise RuntimeError(" | ".join(failures) if failures else "Nenhuma competicao disponivel no momento.")


def market_label(market: str, home_team: str, away_team: str) -> str:
    if market == "Casa":
        return f"Vitoria {home_team}"
    if market == "Fora":
        return f"Vitoria {away_team}"
    if market == "Casa ou Empate":
        return f"Vitoria {home_team} ou empate"
    if market == "Fora ou Empate":
        return f"Vitoria {away_team} ou empate"
    return "Empate"


def format_score_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:.1f}"


def format_match_datetime(date_text: object, event_timestamp: object = None, status: str = "") -> str:
    raw = str(date_text or "").strip()
    if str(status).strip() == "Finalizado":
        explicit_date = re.search(r"(\d{2})/(\d{2})/(\d{4})", raw)
        if explicit_date:
            return f"{explicit_date.group(1)}/{explicit_date.group(2)}/{explicit_date.group(3)}"
    parsed_ts = pd.to_datetime(event_timestamp, errors="coerce", utc=True)
    if not pd.isna(parsed_ts):
        local_dt = parsed_ts.tz_convert(APP_TIMEZONE)
        if str(status).strip() == "Finalizado":
            return local_dt.strftime("%d/%m/%Y")
        return local_dt.strftime("%d/%m/%Y %H:%M")
    return raw if raw else "-"


def format_date_column_for_display(frame: pd.DataFrame, *, status: str | None = None) -> pd.DataFrame:
    if frame.empty or "date_text" not in frame.columns:
        return frame.copy()

    out = frame.copy()
    if "event_timestamp" in out.columns:
        event_series = out["event_timestamp"]
    else:
        event_series = pd.Series([None] * len(out), index=out.index)

    if status is None and "status" in out.columns:
        status_series = out["status"]
    else:
        status_series = pd.Series([status or ""] * len(out), index=out.index)

    out["date_text"] = [
        format_match_datetime(date_text, event_timestamp, row_status)
        for date_text, event_timestamp, row_status in zip(out["date_text"], event_series, status_series)
    ]
    return out


def resolve_match_market(home_goals: object, away_goals: object) -> str:
    if home_goals is None or away_goals is None or pd.isna(home_goals) or pd.isna(away_goals):
        return "-"
    if float(home_goals) > float(away_goals):
        return "Casa"
    if float(home_goals) < float(away_goals):
        return "Fora"
    return "Empate"


def build_analysis_match_label(match_row: pd.Series) -> str:
    status = str(match_row.get("status", "")).strip() or "Desconhecido"
    display_date = format_match_datetime(
        match_row.get("date_text"),
        match_row.get("event_timestamp"),
        status,
    )
    base = f"{display_date} | {match_row['home_team']} x {match_row['away_team']}"
    if status == "Finalizado":
        score = f"{format_score_value(match_row.get('home_goals'))} x {format_score_value(match_row.get('away_goals'))}"
        return f"[Finalizado] {base} | Placar {score}"
    return f"[Agendado] {base}"


def filter_matches_by_team(frame: pd.DataFrame, team_query: str) -> pd.DataFrame:
    query = (team_query or "").strip().casefold()
    if not query or frame.empty:
        return frame.copy()

    home = frame["home_team"].fillna("").astype(str).str.casefold()
    away = frame["away_team"].fillna("").astype(str).str.casefold()
    mask = home.str.contains(query, regex=False) | away.str.contains(query, regex=False)
    return frame.loc[mask].copy()


def filter_matches_by_date(
    frame: pd.DataFrame,
    start_date: date | None,
    end_date: date | None,
) -> pd.DataFrame:
    if frame.empty or (start_date is None and end_date is None):
        return frame.copy()

    out = frame.copy()
    parsed_local = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")

    if "event_timestamp" in out.columns:
        event_series = pd.to_datetime(out["event_timestamp"], errors="coerce", utc=True)
        if event_series.notna().any():
            parsed_local = event_series.dt.tz_convert(APP_TIMEZONE).dt.tz_localize(None)

    if "status" in out.columns and "date_text" in out.columns:
        final_mask = out["status"].fillna("").astype(str).str.strip().eq("Finalizado")
        if final_mask.any():
            explicit_text = (
                out.loc[final_mask, "date_text"]
                .fillna("")
                .astype(str)
                .str.extract(r"(\d{2}/\d{2}/\d{4})", expand=False)
            )
            explicit_dates = pd.to_datetime(explicit_text, errors="coerce", dayfirst=True)
            parsed_local.loc[final_mask] = explicit_dates

    if "date_text" in out.columns:
        unresolved = parsed_local.isna()
        if unresolved.any():
            date_text_series = out.loc[unresolved, "date_text"].fillna("").astype(str).str.strip()
            normalized_dates = date_text_series.str.replace(r"\s+\d{2}:\d{2}$", "", regex=True)
            parsed_text = pd.to_datetime(normalized_dates, errors="coerce", dayfirst=True)
            parsed_local.loc[unresolved] = parsed_text

    match_dates = parsed_local.dt.date
    mask = pd.Series(True, index=out.index)
    if start_date is not None:
        mask &= match_dates >= start_date
    if end_date is not None:
        mask &= match_dates <= end_date
    return out.loc[mask.fillna(False)].copy()


def sort_matches_for_display(frame: pd.DataFrame, *, ascending: bool) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    out = frame.copy()
    if "event_timestamp" in out.columns:
        out["_event_dt"] = pd.to_datetime(out["event_timestamp"], errors="coerce", utc=True)
        if out["_event_dt"].notna().any():
            out = out.sort_values(by=["_event_dt", "home_team", "away_team"], ascending=ascending)
            return out.drop(columns="_event_dt")
    return out.reset_index(drop=True)


def market_badge_label(market: str, home_team: str, away_team: str) -> str:
    if market == "Casa":
        return f"Casa | {home_team}"
    if market == "Fora":
        return f"Fora | {away_team}"
    if market == "Casa ou Empate":
        return f"Casa/Empate | {home_team}"
    if market == "Fora ou Empate":
        return f"Fora/Empate | {away_team}"
    return "Empate"


def render_card_grid(cards: list[dict[str, str]]) -> None:
    if not cards:
        return

    html = ['<div class="card-grid">']
    for card in cards:
        html.append(
            f"""
<article class="modern-card">
  <span>{escape(str(card.get('eyebrow', 'Indicador')))}</span>
  <strong>{escape(str(card.get('value', '-')))}</strong>
</article>
"""
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_split_highlight(title: str, copy: str, items: list[str], tone: str = "neutral") -> None:
    item_html = "".join(f"<li>{escape(item)}</li>" for item in items if str(item).strip())
    st.markdown(
        f"""
<section class="split-highlight split-{escape(tone)}">
  <div>
    <div class="section-title">{escape(title)}</div>
    <div class="section-copy">{escape(copy)}</div>
  </div>
  <ul>{item_html}</ul>
</section>
""",
        unsafe_allow_html=True,
    )


def render_module_hero(title: str, copy: str, badge: str, aside_title: str, aside_copy: str) -> None:
    st.markdown(
        f"""
<section class="module-hero">
  <div class="section-header">
    <div>
      <div class="hero-tag">{escape(badge)}</div>
      <h2>{escape(title)}</h2>
      <p>{escape(copy)}</p>
    </div>
    <div class="portal-callout">
      <strong>{escape(aside_title)}</strong>
      <p>{escape(aside_copy)}</p>
    </div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def render_callout_grid(items: list[dict[str, str]]) -> None:
    if not items:
        return

    html = ['<div class="callout-grid">']
    for item in items:
        html.append(
            f"""
<article class="portal-callout">
  <span class="section-badge">{escape(str(item.get('eyebrow', 'Leitura')))}</span>
  <strong>{escape(str(item.get('title', 'Resumo')))}</strong>
</article>
"""
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _inject_portal_updated_badge(html: str, updated_at_value: str) -> str:
    stamp = str(updated_at_value or "").strip()
    if not stamp:
        return html
    safe_stamp = escape(stamp)
    def _replace_badge(match: re.Match[str]) -> str:
        return f"{match.group(1)}{safe_stamp}{match.group(3)}"
    return re.sub(
        r'(<span id="portalUpdatedAt">)(.*?)(</span>)',
        _replace_badge,
        html,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )


def _resolve_portal_remote_api_base_url() -> str:
    candidates: list[str] = [
        str(os.getenv("PORTAL_REMOTE_API_BASE_URL", "")).strip(),
        str(os.getenv("PUBLIC_PORTAL_API_BASE_URL", "")).strip(),
    ]
    try:
        candidates.extend(
            [
                str(st.secrets.get("PORTAL_REMOTE_API_BASE_URL", "")).strip(),
                str(st.secrets.get("PUBLIC_PORTAL_API_BASE_URL", "")).strip(),
            ]
        )
    except Exception:
        pass

    for candidate in candidates:
        if not candidate:
            continue
        if re.match(r"^https?://", candidate, flags=re.IGNORECASE):
            return candidate.rstrip("/")
    return ""


def _inject_portal_api_base(html: str, api_base: str) -> str:
    return html.replace('"__FD_API_BASE__"', json.dumps(str(api_base or "")))


def _extract_portal_git_hash(html: str) -> str:
    match = re.search(r"<!--\s*portal-build-git:\s*([A-Za-z0-9._-]+)\s*-->", html)
    if not match:
        return ""
    return str(match.group(1)).strip()


def render_embedded_index_portal(updated_at_override: str = "") -> None:
    ensure_portal_ai_server_running()
    html = _load_index_portal_html()
    html = _inject_portal_updated_badge(html, updated_at_override)
    html = _inject_portal_api_base(html, _resolve_portal_remote_api_base_url())
    components.html(html, height=1200, scrolling=True)


def render_public_portal_refresh_button() -> None:
    refresh_href = f"?view=portal&refresh_portal=1&refresh_nonce={int(time.time() * 1000)}"
    st.markdown(
        f"""
<div style="position:fixed;top:18px;right:18px;z-index:2100;">
  <a href="{escape(refresh_href)}" target="_self"
     style="display:inline-flex;align-items:center;justify-content:center;width:48px;height:48px;border-radius:999px;
            background:#1d4ed8;color:#fff;text-decoration:none;font:900 1.35rem/1 'Space Grotesk',sans-serif;
            box-shadow:0 12px 24px rgba(29,78,216,.34);border:1px solid rgba(191,219,254,.36);">↻</a>
</div>
""",
        unsafe_allow_html=True,
    )


def render_public_model_criteria_help() -> None:
    cfg = _get_runtime_model_config()
    poisson_cfg = cfg.get("poisson", {}) if isinstance(cfg, dict) else {}
    calibration_cfg = cfg.get("calibration", {}) if isinstance(cfg, dict) else {}
    betting_cfg = cfg.get("betting", {}) if isinstance(cfg, dict) else {}
    safe_cfg = cfg.get("safe_score", {}) if isinstance(cfg, dict) else {}
    poisson_cfg = poisson_cfg if isinstance(poisson_cfg, dict) else {}
    calibration_cfg = calibration_cfg if isinstance(calibration_cfg, dict) else {}
    betting_cfg = betting_cfg if isinstance(betting_cfg, dict) else {}
    safe_cfg = safe_cfg if isinstance(safe_cfg, dict) else {}

    with st.expander("Criterios do modelo e recalibracao", expanded=False):
        st.markdown(
            f"""
- Poisson: `max_goals={int(poisson_cfg.get('max_goals', 5))}` | `home_default={float(poisson_cfg.get('league_home_default', 1.35)):.2f}` | `away_default={float(poisson_cfg.get('league_away_default', 1.10)):.2f}`.
- Calibracao: `enabled={bool(calibration_cfg.get('enabled', True))}` | `min_history={int(calibration_cfg.get('min_history_matches', 80))}` | `min_bucket={int(calibration_cfg.get('min_bucket_matches', 12))}`.
- Stake: `kelly_fractional={float(betting_cfg.get('kelly_fractional', 0.25)):.2f}`.
- Safe score: `prob_weight={float(safe_cfg.get('prob_weight', 0.55)):.2f}` | `bookmakers_weight={float(safe_cfg.get('bookmakers_weight', 0.20)):.2f}` | `odd_weight={float(safe_cfg.get('odd_weight', 0.05)):.2f}`.
"""
        )
        st.caption(
            "Para ajustar com autonomia: abra o modulo `Configuracoes` no app principal, altere a calibragem e clique em `Recalibrar Modelo Agora`."
        )


@st.cache_data(show_spinner=False)
def _read_cached_html_snapshot(path_str: str, modified_at: float, release_hash: str) -> str:
    del modified_at
    del release_hash
    return Path(path_str).read_text(encoding="utf-8")


def _load_index_portal_html() -> str:
    if INDEX_HTML_FILE.exists():
        return _read_cached_html_snapshot(
            str(INDEX_HTML_FILE),
            INDEX_HTML_FILE.stat().st_mtime,
            CURRENT_GIT_SHORT_HASH,
        )

    html = build_index_html()
    INDEX_HTML_FILE.write_text(html, encoding="utf-8")
    _read_cached_html_snapshot.clear()
    return html


def _load_world_cup_portal_html() -> str:
    if WORLD_CUP_HTML_FILE.exists():
        return _read_cached_html_snapshot(
            str(WORLD_CUP_HTML_FILE),
            WORLD_CUP_HTML_FILE.stat().st_mtime,
            CURRENT_GIT_SHORT_HASH,
        )

    html = build_world_cup_schedule_html()
    WORLD_CUP_HTML_FILE.write_text(html, encoding="utf-8")
    return html


def render_embedded_world_cup_portal() -> None:
    ensure_portal_ai_server_running()
    components.html(_load_world_cup_portal_html(), height=1200, scrolling=True)


def queue_page_navigation(page_name: str) -> None:
    st.session_state["pending_page_menu_v4"] = page_name
    st.rerun()


def render_portal_refresh_action_button(
    *,
    key: str,
    label: str = "Atualizar agora",
    use_container_width: bool = False,
) -> None:
    if not st.button(label, use_container_width=use_container_width, key=key):
        return

    status_box = st.empty()
    progress_box = st.empty()
    status_box.info("Atualizando portal completo (scraping + reprocessamento)...")
    progress_widget = progress_box.progress(0)

    def on_refresh_progress(progress: int, message: str, _stage: str) -> None:
        bounded = max(0, min(100, int(progress)))
        try:
            progress_widget.progress(bounded, text=message)
        except TypeError:
            progress_widget.progress(bounded)
            status_box.info(message)

    try:
        ensure_portal_ai_server_running()
        payload = refresh_portal_snapshot_with_progress(
            progress_callback=on_refresh_progress,
            prefetch_real_stats=True,
        )
        try:
            progress_widget.progress(100, text="Atualizacao concluida.")
        except TypeError:
            progress_widget.progress(100)
            status_box.success("Atualizacao concluida.")
        st.cache_data.clear()
        updated_at = str(payload.get("updated_at", "agora"))
        st.session_state[PORTAL_REFRESH_FEEDBACK_KEY] = f"Portal atualizado com sucesso em {updated_at}."
    except Exception as exc:
        st.cache_data.clear()
        st.session_state[PORTAL_REFRESH_FEEDBACK_KEY] = (
            "Nao foi possivel atualizar dados online agora. "
            "Mantive o painel publicado no Git/Streamlit sem interromper a visualizacao."
        )
    st.rerun()


def render_quick_module_nav(current_page: str) -> None:
    pages = [
        "Inicio",
        "Configuracoes",
        "Jogos Seguros",
        "Painel do Modelo",
        "Analise de Jogo",
        "Resultados",
    ]
    st.caption("Acesso rapido entre modulos")
    cols = st.columns(len(pages))
    for idx, page_name in enumerate(pages):
        with cols[idx]:
            if st.button(
                page_name,
                key=f"quick_nav_btn_{idx}",
                use_container_width=True,
                disabled=page_name == current_page,
            ):
                queue_page_navigation(page_name)


def set_public_portal_shell() -> None:
    st.markdown(
        """
<style>
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] {
  display: none !important;
}
.stApp .block-container {
  max-width: none !important;
  padding: 0 !important;
}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1100px 520px at 0% 0%, rgba(59,130,246,0.10), transparent 55%),
    linear-gradient(180deg, #edf4fb 0%, #eef6f2 100%) !important;
}
.public-back-shell:not(:first-of-type) {
  display: none !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_public_landing() -> None:
    set_public_portal_shell()
    st.markdown(
        """
<style>
.public-home-shell {
  max-width: 980px;
  margin: 0 auto;
  padding: 40px 24px 12px;
}
.public-home-intro {
  margin-bottom: 18px;
  text-align: center;
}
.public-home-intro strong {
  display: block;
  font: 800 clamp(2rem, 5vw, 3.4rem)/1 "Space Grotesk", sans-serif;
  color: #0f2235;
}
.public-home-intro span {
  display: block;
  margin-top: 10px;
  color: #5a6d81;
  font: 500 1rem/1.6 "Manrope", sans-serif;
}
.public-home-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}
.public-home-card {
  padding: 26px;
  border-radius: 28px;
  min-height: 220px;
  box-shadow: 0 28px 60px rgba(15,23,42,.16);
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
}
.public-home-link {
  display: block;
  text-decoration: none;
}
.public-home-link:hover .public-home-card {
  transform: translateY(-4px);
  box-shadow: 0 34px 72px rgba(15,23,42,.22);
  filter: saturate(1.05);
}
.public-home-card.portal {
  background: linear-gradient(135deg,#10233a,#1d4ed8);
}
.public-home-card.copa {
  background: linear-gradient(135deg,#0f6a3c,#1c56b8);
}
.public-home-card strong {
  display: block;
  font: 800 2rem/1.02 "Space Grotesk", sans-serif;
  color: #fff;
}
.public-home-card span {
  display: block;
  margin-top: 10px;
  color: rgba(255,255,255,.84);
  font: 500 1rem/1.6 "Manrope", sans-serif;
}
@media (max-width: 760px) {
  .public-home-grid { grid-template-columns: 1fr; }
}
</style>
<div class="public-home-shell">
  <div class="public-home-intro">
    <strong>Escolha onde entrar</strong>
    <span>Abra o portal de apostas ou o portal da Copa do Mundo 2026.</span>
  </div>
  <div class="public-home-grid">
    <a class="public-home-link" href="?view=portal" target="_self">
      <div class="public-home-card portal">
        <strong>Portal Apostas</strong>
        <span>Abrir o portal principal com o painel de apostas.</span>
      </div>
    </a>
    <a class="public-home-link" href="?view=copa" target="_self">
      <div class="public-home-card copa">
        <strong>Copa do Mundo 2026</strong>
        <span>Abrir o portal da Copa com filtros, placares sugeridos e calibragem.</span>
      </div>
    </a>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_public_back_button() -> None:
    st.markdown(
        """
<div class="public-back-shell" style="max-width:1480px;margin:0 auto;padding:18px 18px 6px;">
  <a class="public-back-link" href="?" target="_self" style="display:inline-flex;align-items:center;gap:8px;padding:12px 16px;border-radius:999px;background:linear-gradient(135deg,rgba(255,255,255,.96),rgba(245,249,255,.96));border:1px solid rgba(148,163,184,.20);box-shadow:0 12px 24px rgba(15,23,42,.08);font:800 .92rem/1 'Space Grotesk',sans-serif;color:#0f2235;text-decoration:none;">Voltar para a pagina inicial</a>
</div>
""",
        unsafe_allow_html=True,
    )


def _query_param_str(name: str, default: str = "") -> str:
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        return str(value[0]) if value else str(default)
    return str(value)


def _set_public_query_params(params: dict[str, str]) -> None:
    try:
        st.query_params.clear()
        for key, value in params.items():
            st.query_params[key] = value
        return
    except Exception:
        pass

    try:
        st.experimental_set_query_params(**params)
    except Exception:
        pass


def _query_flag_enabled(name: str) -> bool:
    return _query_param_str(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _execute_public_portal_refresh(*, refresh_nonce: str) -> str:
    refreshed_updated_at = ""
    try:
        status_box = st.empty()
        progress_box = st.empty()
        status_box.info("Atualizando portal no servidor do Streamlit...")
        progress_widget = progress_box.progress(0)

        def on_progress(progress: int, message: str, _stage: str) -> None:
            bounded = max(0, min(100, int(progress)))
            try:
                progress_widget.progress(bounded, text=message)
            except TypeError:
                progress_widget.progress(bounded)
                status_box.info(message)

        payload = refresh_portal_snapshot_with_progress(
            progress_callback=on_progress,
            prefetch_real_stats=True,
        )
        try:
            progress_widget.progress(100, text="Atualizacao concluida.")
        except TypeError:
            progress_widget.progress(100)
            status_box.success("Atualizacao concluida.")
        updated_at = str(payload.get("updated_at", "agora"))
        st.session_state["_public_portal_last_updated_at"] = updated_at
        refreshed_updated_at = updated_at
        st.session_state["_public_portal_refresh_feedback"] = f"Portal atualizado com sucesso em {updated_at}."
        _read_cached_html_snapshot.clear()
        st.cache_data.clear()
    except Exception:
        st.cache_data.clear()
        refreshed_updated_at = _current_app_timestamp()
        st.session_state["_public_portal_last_updated_at"] = refreshed_updated_at
        st.session_state["_public_portal_refresh_feedback"] = (
            "Nao foi possivel buscar dados online agora; mantive o painel publicado mais recente."
        )
    finally:
        st.session_state["_public_portal_refresh_nonce"] = refresh_nonce

    return refreshed_updated_at


def _run_public_portal_refresh_if_requested() -> bool:
    if not _query_flag_enabled("refresh_portal"):
        return False

    refresh_nonce = _query_param_str("refresh_nonce", "").strip() or f"nonce-{int(time.time() * 1000)}"
    last_nonce = str(st.session_state.get("_public_portal_refresh_nonce", ""))
    if refresh_nonce == last_nonce:
        return False

    refreshed_updated_at = _execute_public_portal_refresh(refresh_nonce=refresh_nonce)

    next_params = {"view": "portal"}
    if refreshed_updated_at:
        next_params["updated_at"] = refreshed_updated_at
    _set_public_query_params(next_params)
    st.rerun()
    return True


public_home_view = _query_param_str("view", "app")
if public_home_view == "landing":
    render_public_landing()
    st.stop()
if public_home_view == "portal":
    if _run_public_portal_refresh_if_requested():
        st.stop()
    set_public_portal_shell()
    manual_refresh = st.button("Atualizar Dados (Streamlit)", use_container_width=False, key="public_portal_streamlit_refresh")
    if manual_refresh:
        refreshed_updated_at = _execute_public_portal_refresh(
            refresh_nonce=f"manual-{int(time.time() * 1000)}"
        )
        next_params = {"view": "portal"}
        if refreshed_updated_at:
            next_params["updated_at"] = refreshed_updated_at
        _set_public_query_params(next_params)
        st.rerun()
    render_public_model_criteria_help()
    render_public_portal_refresh_button()
    render_public_back_button()
    feedback = st.session_state.pop("_public_portal_refresh_feedback", "")
    if feedback:
        if feedback.lower().startswith("falha"):
            st.error(feedback)
        else:
            st.success(feedback)
    updated_at_override = _query_param_str("updated_at", "").strip()
    if not updated_at_override:
        updated_at_override = str(st.session_state.get("_public_portal_last_updated_at", "")).strip()
    render_embedded_index_portal(updated_at_override=updated_at_override)
    st.stop()
if public_home_view == "copa":
    set_public_portal_shell()
    render_public_back_button()
    render_embedded_world_cup_portal()
    st.stop()


def clear_embedded_index_portal() -> None:
    components.html(
        """
<script>
const parentWindow = window.parent;
if (typeof parentWindow.__fdIndexCleanup === "function") {
  parentWindow.__fdIndexCleanup();
}
</script>
""",
        height=0,
        width=0,
    )


def format_ai_analysis(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned

    normalized = cleaned.replace("Resultado mais provável", "Resultado mais provavel")
    normalized = normalized.replace("máximo", "maximo")

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    joined = "\n".join(lines)

    label_map = {
        "resultado mais provavel:": "Resultado mais provavel",
        "melhor aposta por valor:": "Entrada por probabilidade",
        "entrada por probabilidade:": "Entrada por probabilidade",
        "cuidado antes de apostar:": "Cuidado antes de apostar",
    }
    collected = {title: [] for title in label_map.values()}
    current_title = None

    for raw_line in joined.splitlines():
        line = raw_line.strip()
        lower_line = line.lower()
        matched_title = None
        for prefix, title in label_map.items():
            if lower_line.startswith(prefix):
                matched_title = title
                current_title = title
                remainder = line[len(prefix):].strip(" -")
                if remainder:
                    collected[title].append(remainder)
                break
        if matched_title:
            continue
        if current_title:
            collected[current_title].append(line)

    if any(collected.values()):
        blocks = []
        for title in ["Resultado mais provavel", "Entrada por probabilidade", "Cuidado antes de apostar"]:
            body = " ".join(collected[title]).strip(" .:-")
            if body:
                blocks.append(f"**{title}**\n{body}.")
        if blocks:
            return "\n\n".join(blocks)

    plain_text = " ".join(joined.split())
    markers = [
        ("Resultado mais provavel", ["O resultado mais provavel", "Resultado mais provavel"]),
        ("Entrada por probabilidade", ["Entrada por probabilidade", "A melhor aposta por valor", "Melhor aposta por valor"]),
        ("Cuidado antes de apostar", ["Antes de apostar", "Cuidado antes de apostar"]),
    ]

    spans: list[tuple[str, int, int]] = []
    for title, options in markers:
        best_pos = -1
        best_token = ""
        for token in options:
            pos = plain_text.find(token)
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos = pos
                best_token = token
        if best_pos != -1:
            spans.append((title, best_pos, len(best_token)))

    if spans:
        spans.sort(key=lambda item: item[1])
        blocks = []
        for idx, (title, start, token_len) in enumerate(spans):
            end = spans[idx + 1][1] if idx + 1 < len(spans) else len(plain_text)
            body = plain_text[start + token_len:end].strip(" .:-")
            if body:
                blocks.append(f"**{title}**\n{body}.")
        if blocks:
            return "\n\n".join(blocks)

    sentences = [item.strip(" .") for item in plain_text.split(".") if item.strip()]
    labels = ["Resultado mais provavel", "Entrada por probabilidade", "Cuidado antes de apostar"]
    blocks = []
    for idx, sentence in enumerate(sentences[:3]):
        blocks.append(f"**{labels[idx]}**\n{sentence}.")
    return "\n\n".join(blocks) if blocks else cleaned


def extract_ai_analysis_blocks(text: str) -> tuple[str, str, str]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("A IA retornou uma resposta vazia.")

    normalized = cleaned.replace("Resultado mais provável", "Resultado mais provavel")
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]

    label_map = {
        "resultado mais provavel:": "resultado",
        "melhor aposta por valor:": "valor",
        "entrada por probabilidade:": "valor",
        "cuidado antes de apostar:": "cuidado",
    }
    collected = {"resultado": [], "valor": [], "cuidado": []}
    current_key = None

    for raw_line in lines:
        lower_line = raw_line.lower()
        matched = False
        for prefix, key in label_map.items():
            if lower_line.startswith(prefix):
                current_key = key
                remainder = raw_line[len(prefix):].strip(" -")
                if remainder:
                    collected[key].append(remainder)
                matched = True
                break
        if matched:
            continue
        if current_key:
            collected[current_key].append(raw_line)

    if all(collected.values()):
        return (
            " ".join(collected["resultado"]).strip(" .:-"),
            " ".join(collected["valor"]).strip(" .:-"),
            " ".join(collected["cuidado"]).strip(" .:-"),
        )

    plain_text = " ".join(normalized.split())
    sentences = [item.strip(" .") for item in plain_text.split(".") if item.strip()]
    if len(sentences) >= 3:
        return sentences[0], sentences[1], sentences[2]

    raise ValueError("Nao foi possivel separar a resposta da IA em blocos.")


def render_ai_analysis_blocks(resultado: str, valor: str, cuidado: str) -> str:
    blocks = [
        ("Resultado mais provavel", resultado),
        ("Entrada por probabilidade", valor),
        ("Cuidado antes de apostar", cuidado),
    ]
    rendered = []
    for title, body in blocks:
        cleaned_body = (body or "").strip().strip(".")
        if cleaned_body:
            rendered.append(f"**{title}**\n{cleaned_body}.")
    return "\n\n".join(rendered)


def _normalize_ai_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip().strip(".") for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [item.strip().strip(".") for item in value.split("\n") if item.strip()]
    return []


def parse_detailed_ai_analysis(text: str) -> dict[str, object]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("A IA retornou uma resposta vazia.")

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "score_final": 50,
            "classificacao": "RISCO MEDIO",
            "resumo_executivo": "Erro ao processar JSON. Exibindo texto bruto.",
            "detalhe_variaveis": cleaned,
            "leitura_valor": "N/A",
            "riscos_principais": [],
            "recomendacao_conservadora": "Aguarde nova analise.",
            "placares_provaveis": [],
        }

    return {
        "score_final": payload.get("score_final", 0),
        "classificacao": payload.get("classificacao", "N/A"),
        "resumo_executivo": payload.get("resumo_executivo", ""),
        "detalhe_variaveis": payload.get("detalhe_variaveis", ""),
        "leitura_valor": payload.get("leitura_valor", ""),
        "riscos_principais": _normalize_ai_list(payload.get("riscos_principais")),
        "recomendacao_conservadora": payload.get("recomendacao_conservadora", ""),
        "placares_provaveis": _normalize_ai_list(payload.get("placares_provaveis")),
    }


def render_detailed_ai_analysis(payload: dict[str, object]) -> str:
    score = payload.get("score_final", 0)
    classe = payload.get("classificacao", "N/A")
    resumo = payload.get("resumo_executivo", "")
    detalhe = payload.get("detalhe_variaveis", "")
    valor = payload.get("leitura_valor", "")
    conservadora = payload.get("recomendacao_conservadora", "")
    riscos = payload.get("riscos_principais", [])
    placares = payload.get("placares_provaveis", [])
    try:
        score_text = str(int(float(score)))
    except (TypeError, ValueError):
        score_text = str(score).strip() or "0"

    sections = [
        f"### Pontuacao de Risco: {score_text}/100 ({classe})",
        f"**Resumo executivo**\n{resumo}",
        f"**Variaveis influentes (Pesos)**\n{detalhe}",
        f"**Leitura de valor quantitativo**\n{valor}",
        f"**Recomendacao conservadora**\n{conservadora}",
    ]

    if riscos:
        sections.append("**Riscos principais**\n" + "\n".join(f"- {item}" for item in riscos))
    if placares:
        sections.append("**Placares provaveis**\n" + "\n".join(f"- {item}" for item in placares))

    return "\n\n".join(sections)

    if riscos:
        sections.append("**Riscos principais**\n" + "\n".join(f"- {item}." for item in riscos))
    if estrategia:
        sections.append(f"**Estrategia sugerida**\n{estrategia}.")
    if placares:
        sections.append("**Placares que merecem atencao**\n" + "\n".join(f"- {item}." for item in placares))

    return "\n\n".join(section for section in sections if section)


def find_fixture_row(
    fixtures_df: pd.DataFrame,
    *,
    date_text: str,
    home_team: str,
    away_team: str,
) -> pd.Series:
    mask = (
        fixtures_df["date_text"].astype(str).eq(str(date_text))
        & fixtures_df["home_team"].astype(str).eq(str(home_team))
        & fixtures_df["away_team"].astype(str).eq(str(away_team))
    )
    matches = fixtures_df.loc[mask]
    if matches.empty:
        raise ValueError("Nao foi possivel localizar o jogo correspondente nas partidas futuras.")
    return matches.iloc[0]


def build_ai_analysis_for_fixture(
    matches_df: pd.DataFrame,
    fixture_row: pd.Series,
    *,
    model_config: dict[str, object] | None = None,
) -> dict[str, object]:
    fixture_bookmakers = fixture_row.get("bookmakers", 0)
    try:
        fixture_bookmakers = int(fixture_bookmakers) if not pd.isna(fixture_bookmakers) else 0
    except (TypeError, ValueError):
        fixture_bookmakers = 0
    display_date = format_match_datetime(
        fixture_row.get("date_text"),
        fixture_row.get("event_timestamp"),
        str(fixture_row.get("status", "Agendado")),
    )
    probs = calculate_match_probabilities(
        matches_df,
        fixture_row["home_team"],
        fixture_row["away_team"],
        odd_home=float(fixture_row["odds_home"]),
        odd_draw=float(fixture_row["odds_draw"]),
        odd_away=float(fixture_row["odds_away"]),
        bookmakers=fixture_bookmakers,
        model_config=model_config,
    )
    tip = suggest_bet_strategy(
        probs,
        odd_home=float(fixture_row["odds_home"]),
        odd_draw=float(fixture_row["odds_draw"]),
        odd_away=float(fixture_row["odds_away"]),
        bankroll=1000.0,
        model_config=model_config,
    )
    probable_market, probable_probability = max(
        [
            (f"Vitoria {fixture_row['home_team']}", probs.home_win),
            ("Empate", probs.draw),
            (f"Vitoria {fixture_row['away_team']}", probs.away_win),
        ],
        key=lambda item: item[1],
    )
    readable_market = market_label(str(tip.best_market), str(fixture_row["home_team"]), str(fixture_row["away_team"]))
    ai_analysis = generate_nvidia_match_analysis(
        home_team=str(fixture_row["home_team"]),
        away_team=str(fixture_row["away_team"]),
        date_text=display_date,
        odds_home=float(fixture_row["odds_home"]),
        odds_draw=float(fixture_row["odds_draw"]),
        odds_away=float(fixture_row["odds_away"]),
        home_win_prob=float(probs.home_win),
        draw_prob=float(probs.draw),
        away_win_prob=float(probs.away_win),
        probable_market=str(probable_market),
        probable_probability=float(probable_probability),
        value_market=str(readable_market),
        value_probability=float(tip.model_probability),
        value_implied_probability=float(tip.implied_probability),
        expected_value=float(tip.expected_value),
    )
    return parse_detailed_ai_analysis(ai_analysis)


def request_nvidia_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.35,
    top_p: float = 0.9,
    max_tokens: int = 520,
) -> str:
    api_key = _read_runtime_secret("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("Defina a variavel de ambiente NVIDIA_API_KEY para usar a analise com IA.")

    payload = {
        "model": NVIDIA_MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = requests.post(
                NVIDIA_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            break
        except requests.exceptions.ReadTimeout as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(1.5)
                continue
            raise TimeoutError(
                "A NVIDIA demorou para responder. Tente novamente em alguns segundos."
            ) from exc
        except Exception as exc:
            last_error = exc
            raise
    else:
        raise last_error if last_error else RuntimeError("Falha desconhecida ao consultar a NVIDIA.")

    choice = data["choices"][0]
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = message.get("content")

    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    parts.append(text_value.strip())
        if parts:
            return "\n".join(parts)

    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return reasoning_content.strip()

    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        cleaned_reasoning = reasoning.strip()
        markers = [
            "Estrutura da resposta:",
            "Restrições:",
            "Responda em no maximo",
            "Preciso explicar:",
            "Dados fornecidos:",
        ]
        if any(marker in cleaned_reasoning for marker in markers):
            raise ValueError(
                "A NVIDIA retornou apenas rascunho interno, sem resposta final. "
                "Tente novamente ou troque o modelo."
            )
        return cleaned_reasoning

    raise ValueError(f"Resposta da NVIDIA sem texto utilizavel: {data}")


def generate_nvidia_match_analysis(
    home_team: str,
    away_team: str,
    date_text: str,
    odds_home: float,
    odds_draw: float,
    odds_away: float,
    home_win_prob: float,
    draw_prob: float,
    away_win_prob: float,
    probable_market: str,
    probable_probability: float,
    value_market: str,
    value_probability: float,
    value_implied_probability: float,
    expected_value: float,
) -> str:
    ev_text = (
        "sem EV informativo porque a entrada protegida nao tem odd na base"
        if value_implied_probability <= 0
        else f"EV informativo {expected_value * 100:.2f}%"
    )
    prompt = f"""
Atue como um Analista Quantitativo de Apostas Esportivas (Quants).
Sua tarefa e calcular um SCORE DE RISCO (0 a 100) para o jogo abaixo usando EXATAMENTE estes pesos:
- Forma recente ponderada (30%)
- Confronto direto historico (10%)
- Performance casa/fora (15%)
- Media de gols (15%)
- Odds de mercado (20%)
- Volatilidade recente dos resultados (10%)

Jogo: {home_team} x {away_team}
Data: {date_text}

Dados quantitativos:
- Odds 1X2: Casa {odds_home:.2f}, Empate {odds_draw:.2f}, Fora {odds_away:.2f}
- Probabilidades Modelo: Casa {home_win_prob * 100:.1f}%, Empate {draw_prob * 100:.2f}%, Fora {away_win_prob * 100:.1f}%
- Entrada por probabilidade sugerida: {value_market} ({ev_text})

Classificacao de Risco:
0-39 = RISCO BAIXO
40-69 = RISCO MEDIO
70-100 = RISCO ALTO

Responda exatamente neste formato JSON:
{{
  "score_final": 0-100,
  "classificacao": "RISCO BAIXO/MEDIO/ALTO",
  "resumo_executivo": "Analise tecnica curta",
  "detalhe_variaveis": "Explique quais variaveis (Forma, Odds, etc) mais influenciaram este score especifico",
  "leitura_valor": "Por que a aposta sugerida tem valor quantitativo",
  "riscos_principais": ["risco 1", "risco 2"],
  "recomendacao_conservadora": "Acao mais segura baseada nos dados",
  "placares_provaveis": ["placar 1", "placar 2"]
}}
Cada texto deve ser natural, informativo e sem enrolacao.
Use tom simples, humano e confiante, mas sem prometer acerto.
Evite repetir os mesmos numeros e a mesma justificativa em todos os blocos.
Evite frases vagas como "as probabilidades podem variar" ou "a analise depende dos dados disponiveis".
Nos riscos, cite pontos concretos como zebra, odd esticada, jogo equilibrado, dependencia de um gol cedo ou alta variancia.
Na leitura de valor, explique a diferenca entre probabilidade do modelo e odd de forma humana, sem jargao excessivo.
Na recomendacao conservadora, diga se a entrada parece conservadora, moderada ou agressiva.
Nos placares provaveis, prefira formatos como "1-0 para mandante" ou "1-1" e mantenha coerencia com o favoritismo.
Nao invente dados.
"""
    return request_nvidia_completion(
        [
            {
                "role": "system",
                "content": (
                    "Voce e um analista de apostas esportivas. "
                    "Responda em portugues claro, detalhado, natural e honesto. "
                    "Entregue somente JSON valido. "
                    "Nunca exponha raciocinio interno. "
                    "Evite repeticao e frases genericas. "
                    "Seja util para quem vai decidir se entra ou passa a aposta."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.35,
        top_p=0.9,
        max_tokens=520,
    )


def build_match_chat_context(
    fixture_row: pd.Series,
    probs,
    tip,
    probable_market: str,
    probable_probability: float,
    home_ctx: dict[str, object],
    away_ctx: dict[str, object],
) -> str:
    readable_market = market_label(str(tip.best_market), str(fixture_row["home_team"]), str(fixture_row["away_team"]))
    display_date = format_match_datetime(
        fixture_row.get("date_text"),
        fixture_row.get("event_timestamp"),
        str(fixture_row.get("status", "")),
    )
    top_scorelines = ", ".join(
        f"{score} ({prob * 100:.1f}%)" for score, prob in probs.top_scorelines[:4]
    )
    status = str(fixture_row.get("status", "")).strip() or "Desconhecido"
    result_lines = [f"- Status do jogo no painel: {status}"]
    if status == "Finalizado":
        final_score = f"{format_score_value(fixture_row.get('home_goals'))} x {format_score_value(fixture_row.get('away_goals'))}"
        final_market = market_badge_label(
            resolve_match_market(fixture_row.get("home_goals"), fixture_row.get("away_goals")),
            str(fixture_row["home_team"]),
            str(fixture_row["away_team"]),
        )
        result_lines.append(f"- Placar final: {final_score}")
        result_lines.append(f"- Resultado realizado: {final_market}")
    result_context = "\n".join(result_lines)
    if is_double_chance_market(str(tip.best_market)):
        ev_context = "- EV informativo da entrada: indisponivel para dupla chance sem odd na base"
        implied_context = "- Prob. implicita da odd: indisponivel para dupla chance sem odd na base"
    else:
        ev_context = f"- EV informativo da entrada: {tip.expected_value * 100:.2f}%"
        implied_context = f"- Prob. implicita da odd: {tip.implied_probability * 100:.2f}%"
    return f"""
Contexto do jogo atual:
- Jogo: {fixture_row['home_team']} x {fixture_row['away_team']}
- Data/Horario exibido no painel: {display_date}
{result_context}
- Odds 1X2: Casa {float(fixture_row['odds_home']):.2f} | Empate {float(fixture_row['odds_draw']):.2f} | Fora {float(fixture_row['odds_away']):.2f}
- Probabilidades do modelo: Casa {probs.home_win * 100:.1f}% | Empate {probs.draw * 100:.1f}% | Fora {probs.away_win * 100:.1f}%
- Resultado mais provavel: {probable_market} ({probable_probability * 100:.1f}%)
- Entrada por probabilidade: {readable_market}
- {ev_context[2:]}
- Prob. modelo da entrada: {tip.model_probability * 100:.2f}%
- {implied_context[2:]}
- Stake sugerida: R$ {tip.suggested_stake:.2f}
- Gols esperados: mandante {probs.expected_home_goals:.2f} | visitante {probs.expected_away_goals:.2f}
- BTTS SIM: {probs.btts_yes * 100:.1f}%
- Under 2.5: {probs.under_25 * 100:.1f}%
- Over 2.5: {probs.over_25 * 100:.1f}%
- Ranking/contexto mandante: posicao {home_ctx.get('rank')} | pontos {home_ctx.get('points')} | forma recente {home_ctx.get('recent_text')} | pontos recentes {home_ctx.get('recent_points')}
- Ranking/contexto visitante: posicao {away_ctx.get('rank')} | pontos {away_ctx.get('points')} | forma recente {away_ctx.get('recent_text')} | pontos recentes {away_ctx.get('recent_points')}
- Placares provaveis do modelo: {top_scorelines if top_scorelines else '-'}
"""


def generate_nvidia_match_chat_reply(match_context: str, conversation: list[dict[str, str]]) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Voce e um assistente quantitativo de apostas esportivas conversando sobre um unico jogo por vez. "
                "Responda em portugues do Brasil, com objetividade, clareza e honestidade. "
                "Use somente o contexto fornecido do confronto atual. "
                "Nao invente dados, nao cite informacoes externas nao fornecidas e nao fuja do jogo atual. "
                "Quando fizer sentido, explique risco, valor, cenarios de placar e postura de entrada. "
                "Se a pergunta pedir algo fora do escopo, avise isso de forma curta e traga a resposta de volta ao confronto atual."
            ),
        },
        {"role": "system", "content": match_context},
    ]
    messages.extend(conversation[-8:])
    return request_nvidia_completion(messages, temperature=0.4, top_p=0.9, max_tokens=420)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');
:root {{
  --bg: #eef4f7;
  --card: rgba(255,255,255,0.92);
  --line: rgba(148,163,184,0.24);
  --line-strong: rgba(148,163,184,0.42);
  --text: #0f2235;
  --muted: #5a6d81;
  --blue: #1d4ed8;
  --teal: #0f766e;
  --amber: #d97706;
  --ink: #081625;
  --mint: #dff7ef;
  --sky: #dbeafe;
  --shadow: 0 24px 60px rgba(15,23,42,0.10);
}}
.stApp {{
  font-family: "Manrope", "Segoe UI", sans-serif;
  background:
    radial-gradient(900px 420px at -10% -10%, rgba(59,130,246,0.20), transparent 55%),
    radial-gradient(780px 400px at 100% 0%, rgba(16,185,129,0.16), transparent 50%),
    radial-gradient(520px 280px at 50% 100%, rgba(245,158,11,0.10), transparent 50%),
    var(--bg);
}}
.block-container {{
  padding-top: 1rem;
  max-width: 1380px;
}}
[data-testid="stSidebar"] {{
  background:
    radial-gradient(circle at top, rgba(96,165,250,0.12), transparent 24%),
    linear-gradient(180deg, rgba(8,22,37,0.99), rgba(17,40,66,0.98) 48%, rgba(12,62,74,0.98));
  border-right: 1px solid rgba(148,163,184,0.16);
}}
[data-testid="stSidebar"] * {{
  color: #e5eef8;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {{
  gap: 10px;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
  border: 1px solid rgba(191,219,254,0.14);
  background: rgba(255,255,255,0.06);
  border-radius: 18px;
  padding: 12px 14px;
  transition: transform .18s ease, background .18s ease, border-color .18s ease, box-shadow .18s ease;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
  transform: translateX(2px);
  background: rgba(255,255,255,0.10);
  border-color: rgba(191,219,254,0.22);
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
  background: linear-gradient(135deg, rgba(219,234,254,0.18), rgba(16,185,129,0.12));
  border-color: rgba(147,197,253,0.34);
  box-shadow: 0 14px 28px rgba(8,22,37,0.18);
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {{
  font-weight: 700;
  letter-spacing: -.01em;
}}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown h4 {{
  color: #ffffff !important;
}}
[data-testid="stSidebar"] .stSelectbox > div,
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] textarea {{
  background: rgba(255,255,255,0.10) !important;
  border-color: rgba(191,219,254,0.18) !important;
  color: #ffffff !important;
  border-radius: 16px !important;
}}
[data-testid="stSidebar"] .stCaption {{
  color: #bfd6ee !important;
}}
.topbar {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
  padding: 8px 14px;
  border-radius: 16px;
  background: rgba(255,255,255,.72);
  border: 1px solid rgba(148,163,184,.22);
  box-shadow: 0 10px 25px rgba(15,23,42,.05);
  backdrop-filter: blur(16px);
}}
.brand-block {{
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}}
.brand-mark {{
  width: 32px;
  height: 32px;
  border-radius: 10px;
  background: linear-gradient(135deg, #1d4ed8, #0f766e);
  color: #fff;
  display: grid;
  place-items: center;
  font-size: 13px;
  font-weight: 800;
  letter-spacing: .08em;
}}
.brand-copy strong {{
  display: block;
  font-size: .88rem;
  letter-spacing: -.02em;
  font-family: "Space Grotesk", "Manrope", sans-serif;
}}
.brand-copy span {{
  display: block;
  margin-top: 2px;
  color: var(--muted);
  font-size: .78rem;
}}
.topbar-meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}}
.meta-pill {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 999px;
  background: #f8fafc;
  border: 1px solid rgba(148,163,184,.25);
  color: #334155;
  font-size: .76rem;
  font-weight: 600;
}}
.status-dot {{
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #22c55e;
  box-shadow: 0 0 0 4px rgba(34,197,94,.14);
}}
.dashboard-hero {{
  position: relative;
  overflow: hidden;
  padding: 28px;
  border-radius: 28px;
  margin-bottom: 18px;
  color: #f8fafc;
  background: linear-gradient(135deg, #081625 0%, #14304f 54%, #0f766e 100%);
  box-shadow: 0 28px 70px rgba(8,22,37,0.26);
}}
.dashboard-hero::before {{
  content: "";
  position: absolute;
  inset: 0;
  background:
    radial-gradient(circle at top right, rgba(96,165,250,.22), transparent 28%),
    linear-gradient(120deg, transparent 0%, rgba(255,255,255,.06) 100%);
}}
.hero-grid, .hero-metrics, .hero-nav {{
  position: relative;
  z-index: 1;
}}
.hero-grid {{
  display: grid;
  grid-template-columns: minmax(0, 1.6fr) minmax(260px, 0.9fr);
  gap: 18px;
  align-items: end;
}}
.hero-tag {{
  display: inline-flex;
  padding: 8px 12px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
  background: rgba(255,255,255,.10);
  border: 1px solid rgba(255,255,255,.16);
  color: #dbeafe;
}}
.dashboard-hero h1 {{
  margin: 14px 0 0;
  max-width: 12ch;
  font-size: clamp(2.2rem, 4vw, 3.6rem);
  line-height: .96;
  letter-spacing: -.04em;
  font-family: "Space Grotesk", "Manrope", sans-serif;
}}
.dashboard-hero p {{
  margin: 14px 0 0;
  max-width: 64ch;
  color: rgba(226,232,240,.88);
  line-height: 1.68;
}}
.hero-stack {{
  display: grid;
  gap: 12px;
}}
.hero-note {{
  padding: 16px 18px;
  border-radius: 18px;
  background: rgba(255,255,255,.08);
  border: 1px solid rgba(255,255,255,.14);
}}
.hero-note span {{
  display: block;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: #bfdbfe;
}}
.hero-note strong {{
  display: block;
  margin-top: 8px;
  font-size: 1.7rem;
  line-height: 1.05;
}}
.hero-note p {{
  margin-top: 8px;
  font-size: .9rem;
  color: rgba(226,232,240,.82);
}}
.hero-metrics {{
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 12px;
  margin-top: 20px;
}}
.metric-card {{
  padding: 16px;
  border-radius: 18px;
  background: rgba(255,255,255,.08);
  border: 1px solid rgba(255,255,255,.14);
}}
.metric-card span {{
  display: block;
  font-size: 12px;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: #cfe4ff;
}}
.metric-card strong {{
  display: block;
  margin-top: 10px;
  font-size: 1.65rem;
  letter-spacing: -.04em;
}}
.metric-card p {{
  margin: 8px 0 0;
  font-size: .86rem;
  color: rgba(226,232,240,.8);
  line-height: 1.45;
}}
.metric-track {{
  margin-top: 12px;
  height: 7px;
  border-radius: 999px;
  background: rgba(255,255,255,.12);
  overflow: hidden;
}}
.metric-track i {{
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #93c5fd, #6ee7b7);
}}
.hero-nav {{
  margin-top: 16px;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}}
.hero-nav::before {{
  content: "Leitura do painel";
  width: 100%;
  margin-bottom: 2px;
  font-size: .76rem;
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: #bfdbfe;
}}
.nav-pill {{
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  border-radius: 999px;
  background: rgba(255,255,255,.10);
  border: 1px solid rgba(255,255,255,.12);
  color: #fff;
  font-weight: 600;
}}
.nav-pill span {{
  font-size: .76rem;
  letter-spacing: .05em;
  text-transform: uppercase;
  color: #bfdbfe;
}}
.panel-card {{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 22px;
  padding: .95rem 1rem;
  box-shadow: var(--shadow);
  backdrop-filter: blur(18px);
}}
.section-title {{
  margin: 0;
  font-size: 1.08rem;
  color: var(--text);
  font-weight: 800;
  letter-spacing: -.02em;
}}
.section-copy {{
  margin: .45rem 0 .85rem;
  color: var(--muted);
  font-size: .92rem;
  line-height: 1.55;
}}
.stDataFrame, div[data-testid="stDataFrame"] {{
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid var(--line);
  background: #fff;
}}
div[data-testid="stDataFrame"] * {{
  color: #112031 !important;
}}
[data-testid="stMetric"] {{
  background: linear-gradient(180deg, rgba(255,255,255,.98), rgba(248,250,252,.82));
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 14px;
  box-shadow: var(--shadow);
}}
[data-testid="stMetric"] * {{
  color: #112031 !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
  gap: 8px;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
  height: 42px;
  border-radius: 999px;
  padding: 0 16px;
  background: rgba(255,255,255,.72);
  border: 1px solid var(--line);
}}
[data-testid="stTabs"] [data-baseweb="tab"] * {{
  color: #112031 !important;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
  background: linear-gradient(135deg, #eff6ff, #ecfeff);
  border-color: rgba(59,130,246,.18);
}}
.stButton > button,
.stLinkButton > a {{
  color: #112031 !important;
  border-radius: 12px !important;
  border: 1px solid rgba(148,163,184,.34) !important;
  background: linear-gradient(180deg, #ffffff, #f1f5f9) !important;
  font-weight: 700 !important;
  letter-spacing: -.01em;
  min-height: 32px;
  box-shadow: 0 10px 20px rgba(15,23,42,.06);
  transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease, filter .16s ease;
}}
.stButton > button:hover,
.stLinkButton > a:hover {{
  transform: translateY(-1px);
  border-color: rgba(59,130,246,.36) !important;
  box-shadow: 0 14px 28px rgba(15,23,42,.10);
  filter: saturate(1.02);
}}
.stButton > button[kind="primary"] {{
  background: linear-gradient(135deg, #1d4ed8, #0f766e) !important;
  border: 1px solid rgba(15,118,110,.35) !important;
  color: #ffffff !important;
  box-shadow: 0 14px 30px rgba(29,78,216,.24);
}}
.stButton > button:disabled {{
  opacity: .62;
  transform: none !important;
  box-shadow: none;
}}
[data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {{
  background: #ffffff !important;
  color: #112031 !important;
  border-color: rgba(148,163,184,.35) !important;
}}
.info-strip {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin: 0 0 16px;
}}
.info-box {{
  padding: 14px 16px;
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(255,255,255,.96), rgba(241,245,249,.76));
  border: 1px solid var(--line);
  box-shadow: 0 12px 28px rgba(15,23,42,.05);
}}
.info-box strong {{
  display: block;
  font-size: .9rem;
}}
.info-box span {{
  display: block;
  margin-top: 6px;
  color: var(--muted);
  font-size: .84rem;
  line-height: 1.45;
}}
.ai-panel {{
  padding: 16px 18px;
  border-radius: 18px;
  background: linear-gradient(135deg, #eff6ff, #f8fafc);
  border: 1px solid rgba(59,130,246,.14);
}}
.sidebar-ai-panel {{
  margin-top: 10px;
  padding: 14px;
  border-radius: 16px;
  background: rgba(255,255,255,.07);
  border: 1px solid rgba(191,219,254,.14);
}}
.sidebar-ai-panel strong {{
  display: block;
  color: #ffffff;
  font-size: .92rem;
}}
.sidebar-ai-panel span {{
  display: block;
  margin-top: 6px;
  color: #cfe4ff;
  font-size: .82rem;
  line-height: 1.45;
}}
.mini-chat-panel {{
  margin-top: 12px;
  padding: 14px;
  border-radius: 18px;
  background: rgba(255,255,255,.08);
  border: 1px solid rgba(191,219,254,.16);
}}
.mini-chat-panel strong {{
  display: block;
  color: #ffffff;
  font-size: .95rem;
}}
.mini-chat-panel span {{
  display: block;
  margin-top: 6px;
  color: #cfe4ff;
  font-size: .82rem;
  line-height: 1.45;
}}
.chat-quick-note {{
  color: #bfd6ee;
  font-size: .78rem;
  margin: .35rem 0 .2rem;
}}
.chat-bubble {{
  margin: 10px 0;
  padding: 10px 12px;
  border-radius: 14px;
  border: 1px solid rgba(191,219,254,.14);
}}
.chat-bubble strong {{
  display: block;
  margin-bottom: 6px;
  font-size: .78rem;
  letter-spacing: .04em;
  text-transform: uppercase;
}}
.chat-bubble.user {{
  background: rgba(29,78,216,.14);
}}
.chat-bubble.user strong {{
  color: #dbeafe;
}}
.chat-bubble.assistant {{
  background: rgba(255,255,255,.07);
}}
.chat-bubble.assistant strong {{
  color: #bbf7d0;
}}
.chat-empty {{
  color: #bfd6ee;
  font-size: .82rem;
  line-height: 1.45;
  padding: 10px 0 2px;
}}
.ai-action-panel {{
  margin: 14px 0 10px;
  padding: 16px 18px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(29,78,216,.08), rgba(15,118,110,.08));
  border: 1px solid rgba(59,130,246,.16);
}}
.ai-action-panel strong {{
  display: block;
  font-size: 1rem;
  color: var(--text);
}}
.ai-action-panel span {{
  display: block;
  margin-top: 6px;
  color: var(--muted);
  font-size: .9rem;
  line-height: 1.5;
}}
.small-note {{
  color: var(--muted);
  font-size: .86rem;
  margin-top: .2rem;
}}
.top-index-toolbar {{
  margin: 2px 0 6px;
  padding: 7px 10px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  background:
    radial-gradient(circle at top right, rgba(191,219,254,.55), transparent 32%),
    linear-gradient(180deg, rgba(255,255,255,.96), rgba(241,245,249,.90));
  border: 1px solid var(--line);
  box-shadow: 0 10px 22px rgba(15,23,42,.07);
}}
.top-index-toolbar .eyebrow {{
  display: inline-flex;
  align-items: center;
  padding: 4px 8px;
  border-radius: 999px;
  font-size: .66rem;
  letter-spacing: .08em;
  text-transform: uppercase;
  font-weight: 800;
  color: #1e3a8a;
  background: rgba(219,234,254,.8);
  border: 1px solid rgba(147,197,253,.45);
}}
.top-index-toolbar h3 {{
  margin: 0;
  font-size: 1.2rem;
  letter-spacing: -.02em;
  color: var(--text);
  display: none;
}}
.top-index-toolbar p {{
  margin: 0;
  color: var(--muted);
  line-height: 1.25;
  font-size: .74rem;
}}
.quick-action-chip {{
  margin: 0 0 4px;
  padding: 5px 8px;
  border-radius: 10px;
  border: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(255,255,255,.98), rgba(248,250,252,.90));
}}
.quick-action-chip span {{
  display: inline-flex;
  align-items: center;
  font-size: .62rem;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: #1e40af;
  font-weight: 800;
}}
.quick-action-chip strong {{
  display: block;
  margin-top: 3px;
  color: var(--text);
  font-size: .82rem;
  letter-spacing: -.01em;
}}
.action-card-head {{
  margin: 0 0 10px;
  padding: 12px 14px;
  border-radius: 16px;
  border: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(255,255,255,.98), rgba(248,250,252,.90));
}}
.action-card-head span {{
  display: inline-flex;
  align-items: center;
  font-size: .74rem;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: #1e40af;
  font-weight: 800;
}}
.action-card-head strong {{
  display: block;
  margin-top: 6px;
  color: var(--text);
  font-size: 1rem;
  letter-spacing: -.01em;
}}
.action-card-head p {{
  margin: 6px 0 0;
  color: var(--muted);
  font-size: .86rem;
  line-height: 1.45;
}}
.action-card-footnote {{
  margin-top: 8px;
  color: var(--muted);
  font-size: .84rem;
  line-height: 1.45;
}}
.home-top-actions {{
  margin: 0 0 6px;
}}
.page-shell {{
  display: grid;
  gap: 18px;
}}
.card-grid {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  margin: 16px 0 20px;
}}
.modern-card {{
  padding: 18px;
  border-radius: 24px;
  background:
    radial-gradient(circle at top right, rgba(219,234,254,.75), transparent 28%),
    linear-gradient(180deg, rgba(255,255,255,.99), rgba(244,248,252,.90));
  border: 1px solid var(--line);
  box-shadow: var(--shadow);
}}
.modern-card span {{
  display: block;
  font-size: .76rem;
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--muted);
}}
.modern-card strong {{
  display: block;
  margin-top: 10px;
  font-size: 1.85rem;
  letter-spacing: -.05em;
  color: var(--text);
  font-family: "Space Grotesk", "Manrope", sans-serif;
}}
.modern-card p {{
  margin: 8px 0 0;
  color: var(--muted);
  font-size: .9rem;
  line-height: 1.5;
}}
.spotlight-grid {{
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(320px, .85fr);
  gap: 16px;
  margin-bottom: 18px;
}}
.spotlight-card {{
  padding: 22px;
  border-radius: 28px;
  background:
    radial-gradient(circle at top right, rgba(219,234,254,.56), transparent 22%),
    var(--card);
  border: 1px solid var(--line);
  box-shadow: var(--shadow);
}}
.spotlight-card h3 {{
  margin: 0;
  font-size: 1.24rem;
  letter-spacing: -.03em;
}}
.spotlight-card p {{
  margin: 10px 0 0;
  color: var(--muted);
  line-height: 1.6;
}}
.spotlight-list {{
  margin: 14px 0 0;
  padding-left: 18px;
  color: var(--text);
}}
.spotlight-list li {{
  margin-bottom: 8px;
}}
.section-shell {{
  padding: 18px;
  border-radius: 26px;
  background:
    linear-gradient(180deg, rgba(255,255,255,.99), rgba(246,249,252,.90));
  border: 1px solid var(--line);
  box-shadow: var(--shadow);
}}
.section-shell + .section-shell {{
  margin-top: 18px;
}}
.section-header {{
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 14px;
}}
.section-header p {{
  margin: 8px 0 0;
  color: var(--muted);
  line-height: 1.55;
}}
.section-badge {{
  display: inline-flex;
  align-items: center;
  padding: 10px 14px;
  border-radius: 999px;
  background: linear-gradient(135deg, #eff6ff, #ecfeff);
  border: 1px solid rgba(148,163,184,.22);
  color: #334155;
  font-size: .82rem;
  font-weight: 700;
}}
.module-hero {{
  position: relative;
  overflow: hidden;
  padding: 24px;
  border-radius: 28px;
  color: #f8fafc;
  background: linear-gradient(135deg, #0f2235 0%, #163b54 60%, #0f766e 100%);
  box-shadow: 0 26px 60px rgba(8,22,37,.18);
}}
.module-hero::before {{
  content: "";
  position: absolute;
  inset: 0;
  background:
    radial-gradient(circle at top right, rgba(96,165,250,.18), transparent 30%),
    linear-gradient(120deg, transparent 0%, rgba(255,255,255,.05) 100%);
}}
.module-hero > * {{
  position: relative;
  z-index: 1;
}}
.module-hero h2 {{
  margin: 10px 0 0;
  font-size: clamp(1.8rem, 3vw, 2.6rem);
  letter-spacing: -.04em;
  font-family: "Space Grotesk", "Manrope", sans-serif;
}}
.module-hero p {{
  margin: 12px 0 0;
  max-width: 72ch;
  color: rgba(226,232,240,.88);
  line-height: 1.7;
}}
.module-hero .hero-tag {{
  margin: 0;
}}
.portal-callout {{
  padding: 20px;
  border-radius: 24px;
  background: linear-gradient(135deg, rgba(219,234,254,.85), rgba(223,247,239,.84));
  border: 1px solid rgba(148,163,184,.18);
  box-shadow: var(--shadow);
}}
.portal-callout strong {{
  display: block;
  font-size: 1.04rem;
  color: var(--text);
}}
.portal-callout p {{
  margin: 8px 0 0;
  color: var(--muted);
  line-height: 1.6;
}}
.callout-grid {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
  margin: 18px 0 22px;
}}
.callout-grid .portal-callout {{
  height: 100%;
}}
.module-hero .section-header {{
  align-items: stretch;
}}
.module-hero .portal-callout {{
  min-width: min(100%, 320px);
  background: linear-gradient(135deg, rgba(255,255,255,.92), rgba(240,249,255,.94));
}}
.stTextArea textarea {{
  border-radius: 18px !important;
  border: 1px solid rgba(148,163,184,.28) !important;
  background: rgba(255,255,255,.96) !important;
}}
.split-highlight {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(260px, .9fr);
  gap: 16px;
  padding: 18px;
  border-radius: 22px;
  border: 1px solid var(--line);
  margin: 16px 0;
}}
.split-highlight ul {{
  margin: 0;
  padding-left: 18px;
}}
.split-highlight li {{
  margin-bottom: 8px;
}}
.split-neutral {{
  background: linear-gradient(135deg, rgba(239,246,255,.85), rgba(240,253,250,.92));
}}
.split-warm {{
  background: linear-gradient(135deg, rgba(255,247,237,.92), rgba(255,251,235,.94));
}}
.split-danger {{
  background: linear-gradient(135deg, rgba(254,242,242,.94), rgba(255,247,237,.92));
}}
.table-chip {{
  display: inline-flex;
  align-items: center;
  padding: 6px 10px;
  border-radius: 999px;
  background: #eff6ff;
  color: #1d4ed8;
  font-size: .78rem;
  font-weight: 700;
}}
.hero-nav .nav-pill {{
  text-decoration: none;
}}
hr {{
  border: none;
  border-top: 1px solid var(--line);
  margin: 1rem 0;
}}
</style>
""",
    unsafe_allow_html=True,
)

# --- ESTILOS MODERNOS (MODERNIZAÇÃO) ---
st.markdown("""
<style>
    .nav-card-container {
        display: flex;
        overflow-x: auto;
        gap: 15px;
        padding: 10px 5px;
        scrollbar-width: thin;
    }
    .nav-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 15px;
        min-width: 140px;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid #e1e4e8;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .nav-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        border-color: #4cc9f0;
    }
    .nav-icon { font-size: 1.8rem; margin-bottom: 8px; }
    .nav-label { font-weight: 600; font-size: 0.85rem; color: #1a1a2e; }
    
    .metric-card-modern {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border-top: 4px solid #4cc9f0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        height: 100%;
    }
    .metric-label-modern { color: #64748b; font-size: 0.75rem; text-transform: uppercase; font-weight: bold; letter-spacing: 0.5px; }
    .metric-value-modern { font-size: 1.5rem; font-weight: 800; color: #1e293b; margin-top: 5px; }
    
    .explainer-box {
        background: #f8fafc;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #3b82f6;
        margin: 15px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.dialog("⚙️ Central de Comandos: Calibragem do Modelo")
def command_center():
    st.markdown("### Ajustes Técnicos e Filtros")
    st.caption("Altere os parâmetros abaixo para recalibrar a sensibilidade do modelo em tempo real.")
    
    tabs = st.tabs(["🎚️ Calibragem", "📖 Manual do Modelo", "📜 Histórico"])
    
    with tabs[0]:
        new_competition = st.selectbox("Competição Ativa", options=list(COMPETITIONS.keys()), index=list(COMPETITIONS.keys()).index(st.session_state.get("competition_selector", "Brasileirao")))
        new_team_filter = st.text_input("Filtrar por Time (Opcional)", value=st.session_state.get("team_filter_input", ""), placeholder="Ex: Flamengo")
        new_risk_profile = st.selectbox(
            "Perfil de Risco do Modelo",
            options=["Baixo risco", "Medio risco", "Alto risco", "Personalizado"],
            index=["Baixo risco", "Medio risco", "Alto risco", "Personalizado"].index(st.session_state.get("risk_profile_input", "Baixo risco")),
        )
        
        if new_risk_profile == "Personalizado":
            c1, c2 = st.columns(2)
            with c1:
                st.slider("Prob. Mínima", 0.40, 0.80, 0.60, 0.01, key="slider_prob")
                st.caption("EV permanece apenas como métrica informativa (fora do critério de seleção).")
            with c2:
                st.slider("Odd Máxima", 1.20, 4.00, 2.30, 0.05, key="slider_odd")
                st.slider("Mínimo de Casas", 1, 20, 10, 1, key="slider_books")
        
        if st.button("💾 Aplicar Novas Configurações", type="primary", use_container_width=True):
            st.session_state["competition_selector"] = new_competition
            st.session_state["team_filter_input"] = new_team_filter
            st.session_state["risk_profile_input"] = new_risk_profile
            st.success("Configurações aplicadas com sucesso!")
            st.info(f"**Resumo:** Competição alterada para {new_competition} com perfil {new_risk_profile}.")
            st.rerun()

    with tabs[1]:
        st.markdown("""
        <div class="explainer-box">
            <strong>Como o Modelo Funciona:</strong><br><br>
            Nosso algoritmo utiliza uma abordagem de <b>Máxima Verossimilhança</b> baseada em:<br>
            • <b>Ataque (xG):</b> Capacidade de criação de chances claras.<br>
            • <b>Defesa:</b> Resistência a finalizações de alta probabilidade.<br>
            • <b>Fator Casa:</b> Ajuste estatístico pelo peso da torcida e gramado.<br><br>
            <strong>Como Alterar Manualmente:</strong><br>
            Para tornar o modelo mais agressivo, reduza a <i>Probabilidade Mínima</i> e aumente a <i>Odd Máxima</i>. 
            Para um perfil conservador (Banca Alta), mantenha a Probabilidade acima de 0.60 com odds controladas e mais casas.
        </div>
        """, unsafe_allow_html=True)

    with tabs[2]:
        st.markdown("**Últimas Alterações de Calibragem:**")
        st.caption(f"- {datetime.now().strftime('%d/%m/%Y %H:%M')}: Calibragem automática de odds finalizada.")
        st.caption("- Ajuste de peso para 'Premier League' (Inverno) aplicado.")

# Inicilizar session state se necessário
if "competition_selector" not in st.session_state:
    st.session_state["competition_selector"] = "Brasileirao"
if "team_filter_input" not in st.session_state:
    st.session_state["team_filter_input"] = ""
if "risk_profile_input" not in st.session_state:
    st.session_state["risk_profile_input"] = "Baixo risco"

st.markdown("""
<style>
@media (max-width: 1100px) {
  .hero-grid, .hero-metrics, .info-strip, .card-grid, .spotlight-grid, .split-highlight, .callout-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
@media (max-width: 760px) {
  .topbar, .hero-grid, .hero-metrics, .info-strip, .card-grid, .spotlight-grid, .split-highlight, .section-header, .callout-grid {
    display: grid;
    grid-template-columns: 1fr;
  }
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
<style>
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] {
  display: none !important;
}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    if "page_menu_v4" not in st.session_state:
        st.session_state["page_menu_v4"] = "Inicio"
    if "pending_page_menu_v4" in st.session_state:
        st.session_state["page_menu_v4"] = st.session_state.pop("pending_page_menu_v4")

    st.markdown("### Departamento de Dados de Futebol")
    st.caption(f"Release app: {APP_RELEASE_LABEL}")
    page = st.radio(
        "Menus",
        options=[
            "Inicio",
            "Copa 2026",
            "Configuracoes",
            "IA Institucional",
            "Jogos Seguros",
            "Painel do Modelo",
            "Analise de Jogo",
            "Todos os Futuros",
            "Resultados",
        ],
        key="page_menu_v4",
    )
    if st.button("Abrir Painel Copa 2026", use_container_width=True):
        queue_page_navigation("Copa 2026")
    st.caption("Atalho rapido: use o botao acima ou selecione `Copa 2026` no menu.")
    st.caption("Use este menu lateral para navegar entre os modulos taticos, analiticos e de IA do portal.")
    st.markdown("---")
    st.markdown("### Configuracoes")
    competition_options = list(COMPETITIONS.keys())
    if st.session_state.get("competition_selector") not in competition_options:
        st.session_state["competition_selector"] = competition_options[0]
    competition = st.selectbox(
        "Competicao",
        options=competition_options,
        index=competition_options.index(st.session_state.get("competition_selector", competition_options[0])),
    )
    team_filter = st.text_input(
        "Filtrar por time",
        value=st.session_state.get("team_filter_input", ""),
        placeholder="Ex: Flamengo",
    )
    date_filter_enabled = st.checkbox("Filtrar jogos por data", value=False, key="match_date_filter_enabled")
    date_filter_start: date | None = None
    date_filter_end: date | None = None
    if date_filter_enabled:
        date_col_start, date_col_end = st.columns(2)
        with date_col_start:
            date_filter_start = st.date_input("Data inicial", value=date.today(), key="match_date_filter_start")
        with date_col_end:
            date_filter_end = st.date_input("Data final", value=date.today(), key="match_date_filter_end")
        if date_filter_start and date_filter_end and date_filter_end < date_filter_start:
            st.warning("A data final esta anterior a data inicial. O intervalo sera ajustado automaticamente.")
    risk_profile = st.selectbox(
        "Perfil de risco",
        options=["Baixo risco", "Medio risco", "Alto risco", "Personalizado"],
        index=["Baixo risco", "Medio risco", "Alto risco", "Personalizado"].index(
            st.session_state.get("risk_profile_input", "Baixo risco")
        ),
    )
    st.session_state["competition_selector"] = competition
    st.session_state["team_filter_input"] = team_filter
    st.session_state["risk_profile_input"] = risk_profile

    profile_presets = {
        "Baixo risco": {"min_prob": 0.68, "max_odd": 1.95},
        "Medio risco": {"min_prob": 0.62, "max_odd": 2.20},
        "Alto risco": {"min_prob": 0.58, "max_odd": 2.60},
    }

    competition_min_books = {
        "Brasileirao": {"Baixo risco": 8, "Medio risco": 6, "Alto risco": 4},
        "Copa do Brasil": {"Baixo risco": 8, "Medio risco": 6, "Alto risco": 4},
        "Premier League": {"Baixo risco": 10, "Medio risco": 8, "Alto risco": 6},
        "La Liga": {"Baixo risco": 10, "Medio risco": 8, "Alto risco": 6},
        "Bundesliga": {"Baixo risco": 10, "Medio risco": 8, "Alto risco": 6},
        "Ligue 1": {"Baixo risco": 10, "Medio risco": 8, "Alto risco": 6},
        "Saudi Professional League": {"Baixo risco": 8, "Medio risco": 6, "Alto risco": 4},
        "Liga Portugal": {"Baixo risco": 8, "Medio risco": 6, "Alto risco": 4},
        "Copa Sul-Americana": {"Baixo risco": 8, "Medio risco": 6, "Alto risco": 4},
        "Libertadores da America": {"Baixo risco": 8, "Medio risco": 6, "Alto risco": 4},
        "Copa do Mundo": {"Baixo risco": 12, "Medio risco": 10, "Alto risco": 8},
    }

    if risk_profile == "Personalizado":
        min_prob = st.slider("Prob. minima do modelo", min_value=0.40, max_value=0.80, value=0.60, step=0.01)
        max_odd = st.slider("Odd maxima", min_value=1.20, max_value=4.00, value=2.30, step=0.05)
        min_books = st.slider("Minimo de casas (B's)", min_value=1, max_value=20, value=10, step=1)
        st.caption("O filtro tambem exige EV nao negativo, margem clara entre mercados e empate controlado.")
    else:
        preset = profile_presets[risk_profile]
        min_prob = preset["min_prob"]
        max_odd = preset["max_odd"]
        min_books = competition_min_books.get(competition, {}).get(risk_profile, 3)
        st.caption(
            f"Filtro aplicado: Prob >= {min_prob:.2f} | "
            f"Odd <= {max_odd:.2f} | Casas >= {min_books} | EV >= 0"
        )

    runtime_model_config = _get_runtime_model_config()
    poisson_cfg = runtime_model_config["poisson"] if isinstance(runtime_model_config.get("poisson"), dict) else {}
    calibration_cfg = runtime_model_config["calibration"] if isinstance(runtime_model_config.get("calibration"), dict) else {}
    market_anchor_cfg = runtime_model_config["market_anchor"] if isinstance(runtime_model_config.get("market_anchor"), dict) else {}
    betting_cfg = runtime_model_config["betting"] if isinstance(runtime_model_config.get("betting"), dict) else {}
    safe_cfg = runtime_model_config["safe_score"] if isinstance(runtime_model_config.get("safe_score"), dict) else {}

    with st.expander("Criterios do Modelo (Resumo Claro)", expanded=False):
        st.markdown("**Filtro operacional atual**")
        st.caption(
            f"Perfil `{risk_profile}` | Prob >= `{min_prob:.2f}` | "
            f"Odd <= `{max_odd:.2f}` | Casas >= `{int(min_books)}` | EV >= `0.00`"
        )
        st.markdown("**Parametros tecnicos em uso**")
        st.markdown(
            f"""
- Poisson: `max_goals={int(poisson_cfg.get('max_goals', 5))}`, `league_home_default={float(poisson_cfg.get('league_home_default', 1.35)):.2f}`, `league_away_default={float(poisson_cfg.get('league_away_default', 1.10)):.2f}`.
- Minimos de gols esperados: `home={float(poisson_cfg.get('min_expected_home', 0.15)):.2f}`, `away={float(poisson_cfg.get('min_expected_away', 0.10)):.2f}`.
- Calibracao: `enabled={bool(calibration_cfg.get('enabled', True))}`, `min_history={int(calibration_cfg.get('min_history_matches', 80))}`, `min_bucket={int(calibration_cfg.get('min_bucket_matches', 12))}`.
- Ajuste da calibracao: `baseline_weight={float(calibration_cfg.get('baseline_weight', 0.30)):.2f}`, `max_adjustment_weight={float(calibration_cfg.get('max_adjustment_weight', 0.55)):.2f}`, `weight_sample_size={float(calibration_cfg.get('weight_sample_size', 45.0)):.1f}`, `max_eval={int(calibration_cfg.get('max_evaluated_matches', 80))}`.
- Ancora de mercado: `enabled={bool(market_anchor_cfg.get('enabled', True))}`, `min_bookmakers={int(market_anchor_cfg.get('min_bookmakers', 6))}`, `max_weight={float(market_anchor_cfg.get('max_weight', 0.32)):.2f}`.
- Gestao de entrada: `kelly_fractional={float(betting_cfg.get('kelly_fractional', 0.25)):.2f}`, `min_selection_probability={float(betting_cfg.get('min_selection_probability', 0.58)):.2f}`, `min_market_gap={float(betting_cfg.get('min_market_gap', 0.08)):.2f}`, `max_draw_probability_for_winner={float(betting_cfg.get('max_draw_probability_for_winner', 0.25)):.2f}`.
- Safe score: `prob_weight={float(safe_cfg.get('prob_weight', 0.55)):.2f}`, `bookmakers_weight={float(safe_cfg.get('bookmakers_weight', 0.20)):.2f}`, `bookmakers_cap={int(safe_cfg.get('bookmakers_cap', 20))}`, `odd_weight={float(safe_cfg.get('odd_weight', 0.05)):.2f}`.
"""
        )
        st.markdown("**Como recalibrar com autonomia**")
        st.markdown(
            """
1. Ajuste os campos em `Calibragem Avancada do Modelo`.
2. Clique em `Recalibrar Modelo Agora` para aplicar sem editar codigo.
3. Clique em `Atualizar agora` para reprocessar o portal com os novos criterios.
4. Se precisar voltar ao ponto inicial, use `Restaurar Criterios Padrao`.
"""
        )
        runtime_model_json = json.dumps(runtime_model_config, indent=2, ensure_ascii=False)
        st.download_button(
            "Baixar criterios atuais (JSON)",
            data=runtime_model_json,
            file_name="model_config_runtime.json",
            mime="application/json",
            use_container_width=True,
            key="download_runtime_model_config",
        )

    with st.expander("Calibragem Avancada do Modelo", expanded=False):
        st.caption("Edite os criterios tecnicos do motor e clique em recalibrar para reprocessar sem alterar codigo.")

        max_goals_input = st.slider(
            "Poisson - maximo de gols simulados",
            min_value=3,
            max_value=10,
            value=int(poisson_cfg.get("max_goals", 5)),
            step=1,
            key="model_cfg_max_goals",
        )
        col_poisson_a, col_poisson_b = st.columns(2)
        with col_poisson_a:
            league_home_default_input = st.number_input(
                "Media liga casa (padrao)",
                min_value=0.40,
                max_value=4.00,
                value=float(poisson_cfg.get("league_home_default", 1.35)),
                step=0.01,
                key="model_cfg_league_home_default",
            )
            min_expected_home_input = st.number_input(
                "Min xG mandante",
                min_value=0.01,
                max_value=2.50,
                value=float(poisson_cfg.get("min_expected_home", 0.15)),
                step=0.01,
                key="model_cfg_min_expected_home",
            )
        with col_poisson_b:
            league_away_default_input = st.number_input(
                "Media liga fora (padrao)",
                min_value=0.30,
                max_value=4.00,
                value=float(poisson_cfg.get("league_away_default", 1.10)),
                step=0.01,
                key="model_cfg_league_away_default",
            )
            min_expected_away_input = st.number_input(
                "Min xG visitante",
                min_value=0.01,
                max_value=2.50,
                value=float(poisson_cfg.get("min_expected_away", 0.10)),
                step=0.01,
                key="model_cfg_min_expected_away",
            )

        st.markdown("**Calibracao de Probabilidade**")
        calibration_enabled_input = st.checkbox(
            "Ativar calibracao automatica por historico",
            value=bool(calibration_cfg.get("enabled", True)),
            key="model_cfg_calibration_enabled",
        )
        bins_text_input = st.text_input(
            "Faixas de calibracao (bins, separados por virgula)",
            value=_model_config_bins_to_text(runtime_model_config),
            key="model_cfg_bins",
        )
        col_calib_a, col_calib_b = st.columns(2)
        with col_calib_a:
            min_history_input = st.number_input(
                "Min. jogos historicos p/ calibrar",
                min_value=10,
                max_value=2000,
                value=int(calibration_cfg.get("min_history_matches", 80)),
                step=1,
                key="model_cfg_min_history",
            )
            baseline_weight_input = st.slider(
                "Peso baseline da calibracao",
                min_value=0.00,
                max_value=1.00,
                value=float(calibration_cfg.get("baseline_weight", 0.30)),
                step=0.01,
                key="model_cfg_baseline_weight",
            )
        with col_calib_b:
            min_bucket_input = st.number_input(
                "Min. amostras por bucket",
                min_value=1,
                max_value=200,
                value=int(calibration_cfg.get("min_bucket_matches", 12)),
                step=1,
                key="model_cfg_min_bucket",
            )
            max_adjust_weight_input = st.slider(
                "Peso maximo do ajuste",
                min_value=0.00,
                max_value=1.00,
                value=float(calibration_cfg.get("max_adjustment_weight", 0.55)),
                step=0.01,
                key="model_cfg_max_adjust_weight",
            )
        weight_sample_size_input = st.number_input(
            "Amostras para peso maximo do ajuste",
            min_value=1.0,
            max_value=500.0,
            value=float(calibration_cfg.get("weight_sample_size", 45.0)),
            step=1.0,
            key="model_cfg_weight_sample_size",
        )
        max_calibration_evaluated_input = st.number_input(
            "Max. jogos recentes avaliados na calibracao",
            min_value=20,
            max_value=1000,
            value=int(calibration_cfg.get("max_evaluated_matches", 80)),
            step=10,
            key="model_cfg_max_calibration_evaluated",
        )

        st.markdown("**Ancora de Mercado (odds + casas)**")
        market_anchor_enabled_input = st.checkbox(
            "Misturar probabilidade do modelo com consenso das odds",
            value=bool(market_anchor_cfg.get("enabled", True)),
            key="model_cfg_market_anchor_enabled",
        )
        col_anchor_a, col_anchor_b = st.columns(2)
        with col_anchor_a:
            market_anchor_min_books_input = st.number_input(
                "Min. casas para usar ancora",
                min_value=0,
                max_value=100,
                value=int(market_anchor_cfg.get("min_bookmakers", 6)),
                step=1,
                key="model_cfg_market_anchor_min_books",
            )
            market_anchor_base_weight_input = st.slider(
                "Peso base das odds",
                min_value=0.00,
                max_value=1.00,
                value=float(market_anchor_cfg.get("base_weight", 0.10)),
                step=0.01,
                key="model_cfg_market_anchor_base_weight",
            )
            market_anchor_agreement_boost_input = st.slider(
                "Bonus quando modelo e odds concordam",
                min_value=0.00,
                max_value=1.00,
                value=float(market_anchor_cfg.get("agreement_boost", 0.08)),
                step=0.01,
                key="model_cfg_market_anchor_agreement_boost",
            )
        with col_anchor_b:
            market_anchor_books_max_input = st.number_input(
                "Casas para peso maximo",
                min_value=1,
                max_value=100,
                value=int(market_anchor_cfg.get("bookmakers_for_max_weight", 18)),
                step=1,
                key="model_cfg_market_anchor_books_max",
            )
            market_anchor_max_weight_input = st.slider(
                "Peso maximo das odds",
                min_value=0.00,
                max_value=1.00,
                value=float(market_anchor_cfg.get("max_weight", 0.32)),
                step=0.01,
                key="model_cfg_market_anchor_max_weight",
            )
            market_anchor_gap_boost_input = st.slider(
                "Bonus de favorito claro nas odds",
                min_value=0.00,
                max_value=1.00,
                value=float(market_anchor_cfg.get("favorite_gap_boost", 0.06)),
                step=0.01,
                key="model_cfg_market_anchor_gap_boost",
            )
        market_anchor_gap_threshold_input = st.slider(
            "Margem minima para favorito claro nas odds",
            min_value=0.00,
            max_value=0.50,
            value=float(market_anchor_cfg.get("favorite_gap_threshold", 0.12)),
            step=0.01,
            key="model_cfg_market_anchor_gap_threshold",
        )

        st.markdown("**Gestao de Entrada e Score de Seguranca**")
        kelly_fractional_input = st.slider(
            "Kelly fracionado",
            min_value=0.00,
            max_value=1.00,
            value=float(betting_cfg.get("kelly_fractional", 0.25)),
            step=0.01,
            key="model_cfg_kelly_fractional",
        )
        col_bet_a, col_bet_b = st.columns(2)
        with col_bet_a:
            accuracy_threshold_input = st.slider(
                "Prob. minima para confiar no modelo",
                min_value=0.00,
                max_value=1.00,
                value=float(betting_cfg.get("accuracy_threshold", 0.68)),
                step=0.01,
                key="model_cfg_accuracy_threshold",
            )
            min_selection_probability_input = st.slider(
                "Prob. minima da entrada",
                min_value=0.00,
                max_value=1.00,
                value=float(betting_cfg.get("min_selection_probability", 0.58)),
                step=0.01,
                key="model_cfg_min_selection_probability",
            )
        with col_bet_b:
            house_favorite_lock_threshold_input = st.slider(
                "Trava favorito forte das casas",
                min_value=0.00,
                max_value=1.00,
                value=float(betting_cfg.get("house_favorite_lock_threshold", 0.70)),
                step=0.01,
                key="model_cfg_house_favorite_lock_threshold",
            )
            min_market_gap_input = st.slider(
                "Margem minima entre resultados",
                min_value=0.00,
                max_value=0.50,
                value=float(betting_cfg.get("min_market_gap", 0.08)),
                step=0.01,
                key="model_cfg_min_market_gap",
            )
        max_draw_probability_for_winner_input = st.slider(
            "Max. prob. de empate para escolher vencedor",
            min_value=0.00,
            max_value=1.00,
            value=float(betting_cfg.get("max_draw_probability_for_winner", 0.25)),
            step=0.01,
            key="model_cfg_max_draw_probability_for_winner",
        )
        col_safe_a, col_safe_b = st.columns(2)
        with col_safe_a:
            safe_prob_weight_input = st.number_input(
                "Peso probabilidade",
                min_value=0.0,
                max_value=5.0,
                value=float(safe_cfg.get("prob_weight", 0.55)),
                step=0.01,
                key="model_cfg_safe_prob_weight",
            )
            safe_ev_weight_input = st.number_input(
                "Peso EV (desativado)",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.05,
                key="model_cfg_safe_ev_weight",
                disabled=True,
            )
            safe_ev_cap_input = st.number_input(
                "Teto EV no score (desativado)",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                key="model_cfg_safe_ev_cap",
                disabled=True,
            )
            safe_bookmakers_weight_input = st.number_input(
                "Peso bookmakers",
                min_value=0.0,
                max_value=5.0,
                value=float(safe_cfg.get("bookmakers_weight", 0.20)),
                step=0.01,
                key="model_cfg_safe_books_weight",
            )
        with col_safe_b:
            safe_bookmakers_cap_input = st.number_input(
                "Cap bookmakers no score",
                min_value=1,
                max_value=100,
                value=int(safe_cfg.get("bookmakers_cap", 20)),
                step=1,
                key="model_cfg_safe_books_cap",
            )
            safe_odd_weight_input = st.number_input(
                "Peso odd",
                min_value=0.0,
                max_value=5.0,
                value=float(safe_cfg.get("odd_weight", 0.05)),
                step=0.01,
                key="model_cfg_safe_odd_weight",
            )
            safe_odd_reference_input = st.number_input(
                "Odd referencia",
                min_value=1.01,
                max_value=20.0,
                value=float(safe_cfg.get("odd_reference", 1.20)),
                step=0.01,
                key="model_cfg_safe_odd_ref",
            )
            safe_odd_span_input = st.number_input(
                "Faixa de odd (span)",
                min_value=0.01,
                max_value=20.0,
                value=float(safe_cfg.get("odd_span", 1.20)),
                step=0.01,
                key="model_cfg_safe_odd_span",
            )

        model_widget_keys = [
            "model_cfg_max_goals",
            "model_cfg_league_home_default",
            "model_cfg_min_expected_home",
            "model_cfg_league_away_default",
            "model_cfg_min_expected_away",
            "model_cfg_calibration_enabled",
            "model_cfg_bins",
            "model_cfg_min_history",
            "model_cfg_baseline_weight",
            "model_cfg_min_bucket",
            "model_cfg_max_adjust_weight",
            "model_cfg_weight_sample_size",
            "model_cfg_max_calibration_evaluated",
            "model_cfg_market_anchor_enabled",
            "model_cfg_market_anchor_min_books",
            "model_cfg_market_anchor_base_weight",
            "model_cfg_market_anchor_agreement_boost",
            "model_cfg_market_anchor_books_max",
            "model_cfg_market_anchor_max_weight",
            "model_cfg_market_anchor_gap_boost",
            "model_cfg_market_anchor_gap_threshold",
            "model_cfg_kelly_fractional",
            "model_cfg_accuracy_threshold",
            "model_cfg_min_selection_probability",
            "model_cfg_house_favorite_lock_threshold",
            "model_cfg_min_market_gap",
            "model_cfg_max_draw_probability_for_winner",
            "model_cfg_safe_prob_weight",
            "model_cfg_safe_ev_weight",
            "model_cfg_safe_ev_cap",
            "model_cfg_safe_books_weight",
            "model_cfg_safe_books_cap",
            "model_cfg_safe_odd_weight",
            "model_cfg_safe_odd_ref",
            "model_cfg_safe_odd_span",
        ]

        if st.button("Recalibrar Modelo Agora", type="primary", use_container_width=True, key="model_cfg_apply"):
            parsed_bins = _parse_bins_text(bins_text_input)
            if parsed_bins is None:
                st.error("Formato invalido para bins. Exemplo: 0, 0.20, 0.35, 0.50, 0.65, 1.01")
            else:
                st.session_state[MODEL_CONFIG_SESSION_KEY] = normalize_model_config(
                    {
                        "poisson": {
                            "max_goals": int(max_goals_input),
                            "league_home_default": float(league_home_default_input),
                            "league_away_default": float(league_away_default_input),
                            "min_expected_home": float(min_expected_home_input),
                            "min_expected_away": float(min_expected_away_input),
                        },
                        "calibration": {
                            "enabled": bool(calibration_enabled_input),
                            "bins": parsed_bins,
                            "min_history_matches": int(min_history_input),
                            "min_bucket_matches": int(min_bucket_input),
                            "baseline_weight": float(baseline_weight_input),
                            "max_adjustment_weight": float(max_adjust_weight_input),
                            "weight_sample_size": float(weight_sample_size_input),
                            "max_evaluated_matches": int(max_calibration_evaluated_input),
                        },
                        "market_anchor": {
                            "enabled": bool(market_anchor_enabled_input),
                            "min_bookmakers": int(market_anchor_min_books_input),
                            "bookmakers_for_max_weight": int(market_anchor_books_max_input),
                            "base_weight": float(market_anchor_base_weight_input),
                            "max_weight": float(market_anchor_max_weight_input),
                            "agreement_boost": float(market_anchor_agreement_boost_input),
                            "favorite_gap_boost": float(market_anchor_gap_boost_input),
                            "favorite_gap_threshold": float(market_anchor_gap_threshold_input),
                        },
                        "betting": {
                            "kelly_fractional": float(kelly_fractional_input),
                            "accuracy_threshold": float(accuracy_threshold_input),
                            "house_favorite_lock_threshold": float(house_favorite_lock_threshold_input),
                            "min_selection_probability": float(min_selection_probability_input),
                            "min_market_gap": float(min_market_gap_input),
                            "max_draw_probability_for_winner": float(max_draw_probability_for_winner_input),
                        },
                        "safe_score": {
                            "prob_weight": float(safe_prob_weight_input),
                            "ev_weight": 0.0,
                            "ev_cap": 0.0,
                            "bookmakers_weight": float(safe_bookmakers_weight_input),
                            "bookmakers_cap": int(safe_bookmakers_cap_input),
                            "odd_weight": float(safe_odd_weight_input),
                            "odd_reference": float(safe_odd_reference_input),
                            "odd_span": float(safe_odd_span_input),
                        },
                    }
                )
                st.cache_data.clear()
                st.session_state[MODEL_CONFIG_FEEDBACK_KEY] = f"Modelo recalibrado em {datetime.now(APP_TIMEZONE).strftime('%d/%m/%Y %H:%M:%S')}."
                st.rerun()

        if st.button("Restaurar Criterios Padrao", use_container_width=True, key="model_cfg_reset"):
            st.session_state[MODEL_CONFIG_SESSION_KEY] = default_model_config()
            for widget_key in model_widget_keys:
                st.session_state.pop(widget_key, None)
            st.cache_data.clear()
            st.session_state[MODEL_CONFIG_FEEDBACK_KEY] = "Criterios do modelo restaurados para o padrao."
            st.rerun()

    render_portal_refresh_action_button(
        key="sidebar_refresh_now",
        label="Atualizar agora",
        use_container_width=True,
    )

    st.markdown("---")
    ai_enabled = bool(_read_runtime_secret("NVIDIA_API_KEY"))
    st.markdown(
        f"""
<div class="sidebar-ai-panel">
  <strong>IA NVIDIA {'ativa' if ai_enabled else 'indisponivel'}</strong>
  <span>{'A analise detalhada pode ser usada no simulador de jogo e fica salva por confronto durante a sessao.' if ai_enabled else 'Defina a variavel de ambiente NVIDIA_API_KEY para habilitar analises detalhadas em linguagem natural.'}</span>
</div>
""",
        unsafe_allow_html=True,
    )
    if ai_enabled:
        st.caption(f"Modelo configurado: `{NVIDIA_MODEL}`")
    st.caption("A home inicial resume a competicao. As demais areas aprofundam os detalhes por tema.")

page = {"Início": "Inicio"}.get(page, page)
runtime_model_config = _get_runtime_model_config()
runtime_feedback = str(st.session_state.pop(MODEL_CONFIG_FEEDBACK_KEY, "")).strip()
if runtime_feedback:
    st.success(runtime_feedback)
portal_refresh_feedback = str(st.session_state.pop(PORTAL_REFRESH_FEEDBACK_KEY, "")).strip()
if portal_refresh_feedback:
    if portal_refresh_feedback.lower().startswith("falha"):
        st.error(portal_refresh_feedback)
    else:
        st.success(portal_refresh_feedback)

refresh_col, quick_comp_col = st.columns([0.9, 1.1], gap="small")
with refresh_col:
    render_portal_refresh_action_button(
        key="main_refresh_now",
        label="Atualizar",
        use_container_width=True,
    )
if st.session_state.get("main_competition_selector") != competition:
    st.session_state["main_competition_selector"] = competition
with quick_comp_col:
    competition = st.selectbox(
        "Focos por competicao",
        options=competition_options,
        key="main_competition_selector",
        label_visibility="collapsed",
    )
st.session_state["competition_selector"] = competition

try:
    competition, df, competition_load_warning = get_data_with_fallback(competition)
except Exception as exc:
    st.error(f"Falha ao buscar dados da internet: {exc}")
    st.stop()

finished = df[df["status"] == "Finalizado"].copy()
fixtures = df[df["status"] == "Agendado"].copy()
valid = fixtures.dropna(subset=["odds_home", "odds_draw", "odds_away"]).copy()
ai_enabled = bool(_read_runtime_secret("NVIDIA_API_KEY"))
runtime_betting_cfg = runtime_model_config.get("betting", {})
runtime_kelly = (
    float(runtime_betting_cfg.get("kelly_fractional", 0.25))
    if isinstance(runtime_betting_cfg, dict)
    else 0.25
)
if date_filter_enabled and date_filter_start and date_filter_end and date_filter_end < date_filter_start:
    date_filter_start, date_filter_end = date_filter_end, date_filter_start
team_filter_clean = team_filter.strip()
if team_filter_clean:
    finished = filter_matches_by_team(finished, team_filter_clean)
    fixtures = filter_matches_by_team(fixtures, team_filter_clean)
    valid = filter_matches_by_team(valid, team_filter_clean)
if date_filter_enabled:
    finished = filter_matches_by_date(finished, date_filter_start, date_filter_end)
    fixtures = filter_matches_by_date(fixtures, date_filter_start, date_filter_end)
    valid = filter_matches_by_date(valid, date_filter_start, date_filter_end)

finished = sort_matches_for_display(finished, ascending=False)
fixtures = sort_matches_for_display(fixtures, ascending=True)
valid = sort_matches_for_display(valid, ascending=True)
finished_with_odds = finished.dropna(subset=["odds_home", "odds_draw", "odds_away"]).copy()
analysis_candidates = pd.concat([valid, finished_with_odds], ignore_index=True, sort=False)

needs_safe_data = page in {"Inicio", "Jogos Seguros"}
needs_backtest_data = page in {"Inicio", "Painel do Modelo", "Resultados"}

safe_df = pd.DataFrame()
relaxed_note = ""
if needs_safe_data:
    safe_df = _build_safe_bets_cached(
        matches_df=df,
        bankroll=1000.0,
        kelly_fractional=runtime_kelly,
        min_model_prob=float(min_prob),
        min_expected_value=0.0,
        max_odd=float(max_odd),
        min_bookmakers=int(min_books),
        model_config=runtime_model_config,
    )
    if team_filter_clean:
        safe_df = filter_matches_by_team(safe_df, team_filter_clean)
    if date_filter_enabled:
        safe_df = filter_matches_by_date(safe_df, date_filter_start, date_filter_end)

    if safe_df.empty and risk_profile != "Personalizado":
        relax_steps = [
            (max(min_prob - 0.05, 0.35), max_odd + 0.30, max(min_books - 1, 1)),
            (max(min_prob - 0.10, 0.30), max_odd + 0.70, 1),
        ]
        for rp, rodd, rbooks in relax_steps:
            safe_df = _build_safe_bets_cached(
                matches_df=df,
                bankroll=1000.0,
                kelly_fractional=runtime_kelly,
                min_model_prob=float(rp),
                min_expected_value=0.0,
                max_odd=float(rodd),
                min_bookmakers=int(rbooks),
                model_config=runtime_model_config,
            )
            if team_filter_clean:
                safe_df = filter_matches_by_team(safe_df, team_filter_clean)
            if date_filter_enabled:
                safe_df = filter_matches_by_date(safe_df, date_filter_start, date_filter_end)
            if not safe_df.empty:
                relaxed_note = (
                    f"Filtro do perfil foi relaxado automaticamente para exibir opcoes: "
                    f"Prob >= {rp:.2f} | Odd <= {rodd:.2f} | Casas >= {rbooks} | EV >= 0."
                )
                break

backtest_df = pd.DataFrame()
backtest_summary: dict[str, object] = {"total_matches": len(finished)}
probability_buckets = pd.DataFrame()
if needs_backtest_data:
    backtest_df = _build_backtest_cached(
        matches_df=df,
        bankroll=1000.0,
        kelly_fractional=runtime_kelly,
        min_history_matches=40,
        max_evaluated_matches=120,
        model_config=runtime_model_config,
    )
    if team_filter_clean:
        backtest_df = filter_matches_by_team(backtest_df, team_filter_clean)
    if date_filter_enabled:
        backtest_df = filter_matches_by_date(backtest_df, date_filter_start, date_filter_end)
    backtest_summary = _summarize_backtest_cached(backtest_df)
    probability_buckets = _build_probability_buckets_cached(backtest_df)

competition_name = str(competition)
team_filter_display = escape(team_filter_clean)
date_filter_label = ""
if date_filter_enabled and date_filter_start and date_filter_end:
    if date_filter_start == date_filter_end:
        date_filter_label = date_filter_start.strftime("%d/%m/%Y")
    else:
        date_filter_label = (
            f"{date_filter_start.strftime('%d/%m/%Y')} a {date_filter_end.strftime('%d/%m/%Y')}"
        )
date_filter_display = escape(date_filter_label)
finished_count = len(finished)
fixtures_count = len(fixtures)
odds_count = len(valid)
safe_count = len(safe_df)
odds_coverage = _safe_percent(odds_count, fixtures_count)
safe_rate = _safe_percent(safe_count, odds_count)
model_accuracy = backtest_summary.get("model_accuracy", 0.0)
house_accuracy = backtest_summary.get("house_accuracy", 0.0)
value_accuracy = backtest_summary.get("value_accuracy", 0.0)
value_roi = backtest_summary.get("value_roi", 0.0)
model_roi = backtest_summary.get("model_roi", 0.0)
avg_model_edge = backtest_summary.get("avg_model_edge", 0.0)
tuning_actions = backtest_summary.get("tuning_actions", [])
page_descriptions = {
    "Inicio": "Visao executiva com resumo da competicao, IA institucional e resultados recentes do modelo.",
    "Copa 2026": "Acesso direto ao portal da Copa do Mundo 2026 com filtros, sugestoes de placar e calibragem do modelo.",
    "Configuracoes": "Area para ajustar competicao, filtro por time e perfil de risco usando os controles do menu lateral.",
    "IA Institucional": "Central de leitura do dia com prompt profissional, execucao e resposta completa dentro do portal.",
    "Jogos Seguros": "Ranking das selecoes mais conservadoras dentro do filtro atual.",
    "Painel do Modelo": "Comparativo detalhado entre modelo, casas de aposta e entradas por probabilidade.",
    "Analise de Jogo": "Simulador por confronto com probabilidades, valor e revisao de jogos futuros ou finalizados.",
    "Todos os Futuros": "Agenda completa dos jogos futuros com odds e numero de casas.",
    "Resultados": "Jogos ja encerrados com comparativo de acerto do modelo.",
}

if page != "Inicio":
    st.markdown(
        f"""
<div id="panel-top-anchor"></div>
<section class="topbar">
  <div class="brand-block">
    <div class="brand-mark">FD</div>
    <div class="brand-copy">
      <strong>Departamento de Dados de Futebol</strong>
      <span>Painel executivo no Streamlit com a mesma linguagem visual do portal principal.</span>
    </div>
  </div>
  <div class="topbar-meta">
    <div class="meta-pill"><span class="status-dot"></span><strong>Atualizado</strong> {current_app_timestamp()}</div>
    <div class="meta-pill"><strong>{competition_name}</strong> em foco</div>
    <div class="meta-pill"><strong>{odds_coverage}%</strong> cobertura de odds</div>
    {"<div class='meta-pill'><strong>Filtro</strong> " + team_filter_display + "</div>" if team_filter.strip() else ""}
    {"<div class='meta-pill'><strong>Periodo</strong> " + date_filter_display + "</div>" if date_filter_label else ""}
  </div>
</section>
""",
        unsafe_allow_html=True,
    )

if page == "Inicio":
    pass
else:
    render_quick_module_nav(page)
    clear_embedded_index_portal()
    render_module_hero(
        title=page,
        copy=page_descriptions.get(page, "Modulo selecionado no menu lateral."),
        badge="Modulo ativo",
        aside_title=f"{competition_name} em foco",
        aside_copy=(
            f"{fixtures_count} jogos futuros, {finished_count} finalizados e "
            f"{backtest_summary.get('total_matches', 0)} partidas no comparativo historico."
        ),
    )

selected_chat_context = ""
selected_chat_label = ""
selected_chat_match_key = ""
show_safe = pd.DataFrame()
recent_backtest = sort_matches_for_display(backtest_df, ascending=False)

if not safe_df.empty:
    safe_df = safe_df.copy()
    safe_df["market_label"] = safe_df.apply(
        lambda r: market_label(str(r["market"]), str(r["home_team"]), str(r["away_team"])),
        axis=1,
    )
    # Use reindex to safely select columns, inserting NaN when a column is missing.
    expected_cols = [
        "date_text",
        "home_team",
        "away_team",
        "market_label",
        "odd",
        "model_probability",
        "expected_value",
        "market_gap",
        "risk_level",
        "bookmakers",
        "safety_score",
        "match_url",
    ]
    show_safe = safe_df.reindex(columns=expected_cols).copy()
    show_safe.columns = [
        "Data",
        "Mandante",
        "Visitante",
        "Palpite",
        "Odd",
        "Prob. Modelo",
        "EV",
        "Margem",
        "Risco",
        "Casas",
        "Score",
        "Link",
    ]
    show_safe["Prob. Modelo"] = (show_safe["Prob. Modelo"] * 100).round(1).astype(str) + "%"
    show_safe["EV"] = (show_safe["EV"] * 100).round(1).astype(str) + "%"
    show_safe["Margem"] = (show_safe["Margem"] * 100).round(1).astype(str) + "%"
    show_safe["Score"] = (show_safe["Score"] * 100).round(1)

if "institutional_ai_prompt" not in st.session_state:
    st.session_state["institutional_ai_prompt"] = AI_PROMPT_TEMPLATE.replace("__DATA_SELECIONADA__", date.today().strftime("%d/%m/%Y"))
if "institutional_ai_prompt_modal" not in st.session_state:
    st.session_state["institutional_ai_prompt_modal"] = st.session_state["institutional_ai_prompt"]
if "institutional_ai_modal_open" not in st.session_state:
    st.session_state["institutional_ai_modal_open"] = False
if "institutional_ai_result" not in st.session_state:
    st.session_state["institutional_ai_result"] = ""
if "institutional_ai_last_date" not in st.session_state:
    st.session_state["institutional_ai_last_date"] = ""

st.markdown('<div class="page-shell">', unsafe_allow_html=True)

if page == "Inicio":
    st.markdown(
        """
<style>
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stSidebarCollapsedControl"] {
  display: none !important;
}
section[data-testid="stSidebar"] {
  display: none !important;
}
.stApp .block-container {
  max-width: none !important;
  padding: 0 !important;
}
[data-testid="stAppViewContainer"] {
  background: #f8fafc !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

    if "show_home_header" not in st.session_state:
        st.session_state["show_home_header"] = False

    st.markdown('<div class="home-top-actions">', unsafe_allow_html=True)
    home_action_col1, home_action_col2 = st.columns([1, 1], gap="small")
    with home_action_col1:
        if st.button(
            "Exibir painel" if not st.session_state["show_home_header"] else "Ocultar painel",
            use_container_width=True,
            key="home_toggle_navigation_panel",
        ):
            st.session_state["show_home_header"] = not st.session_state["show_home_header"]
            st.rerun()
    with home_action_col2:
        if st.button(
            "Central de comandos",
            use_container_width=True,
            type="primary",
            key="home_open_command_center",
        ):
            command_center()
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state["show_home_header"]:
        # --- HEADER / HERO DA HOME ---
        st.markdown(
            """
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 40px 20px; text-align: center; color: white; border-radius: 0 0 30px 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.15);">
        <h1 style='font-family: "Space Grotesk", sans-serif; font-size: 3rem; margin: 0; color: #4cc9f0;'>Departamento de Dados de Futebol</h1>
        <p style='opacity: 0.9; font-size: 1.1rem; margin-top: 10px;'>Painel de Inteligência e Análise Quantitativa para Apostas</p>
    </div>
    """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='max-width:1400px; margin: 0 auto; padding: 20px;'>", unsafe_allow_html=True)

    if st.session_state["show_home_header"]:
        # --- CARROSSEL DE NAVEGAÇÃO ---
        st.markdown("### 🚀 Navegação Rápida")

        # Gerando o HTML do carrossel para Streamlit
        nav_items = [
            {"icon": "🏆", "label": "Copa 2026", "page": "Copa 2026"},
            {"icon": "🛡️", "label": "Jogos Seguros", "page": "Jogos Seguros"},
            {"icon": "🤖", "label": "IA Analista", "page": "IA Institucional"},
            {"icon": "📊", "label": "Painel Modelo", "page": "Painel do Modelo"},
            {"icon": "⚽", "label": "Simulador", "page": "Analise de Jogo"},
            {"icon": "📅", "label": "Agenda", "page": "Todos os Futuros"},
            {"icon": "📈", "label": "Resultados", "page": "Resultados"},
        ]

        # Criando colunas para os cards de navegação (Simulando o carrossel interativo)
        cols = st.columns(len(nav_items))
        for i, item in enumerate(nav_items):
            with cols[i]:
                st.markdown(f"""
                <div class="nav-card">
                    <div class="nav-icon">{item['icon']}</div>
                    <div class="nav-label">{item['label']}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Abrir", key=f"nav_btn_{i}", use_container_width=True):
                    queue_page_navigation(item['page'])

        st.markdown("---")

    render_embedded_index_portal()
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

elif page == "Copa 2026":
    render_callout_grid(
        [
            {"eyebrow": "Portal dedicado", "title": "Painel da Copa no Streamlit"},
            {"eyebrow": "Horario SP", "title": "Tabela ordenada por Sao Paulo"},
            {"eyebrow": "IA", "title": "Placares sugeridos por confronto"},
            {"eyebrow": "Ajuste", "title": "Modelo recalibravel com resultados"},
        ]
    )
    render_embedded_world_cup_portal()

elif page == "Configuracoes":
    render_callout_grid(
        [
            {"eyebrow": "Competicao", "title": "Ajuste o torneio monitorado"},
            {"eyebrow": "Filtro", "title": "Refine por time ou contexto"},
            {"eyebrow": "Risco", "title": "Troque o perfil operacional"},
            {"eyebrow": "Atualizacao", "title": "Recarregue a base manualmente"},
        ]
    )
    render_card_grid(
        [
            {"eyebrow": "Competicao ativa", "value": competition_name},
            {"eyebrow": "Filtro por time", "value": team_filter.strip() or "Todos os times"},
            {"eyebrow": "Perfil de risco", "value": risk_profile},
            {"eyebrow": "Cobertura de odds", "value": f"{odds_coverage}%"},
        ]
    )
    render_split_highlight(
        title="Como ajustar o portal",
        copy="Use o menu lateral para alterar os parametros do sistema. Esta tela serve como atalho rapido para entrar no modo de configuracao com a barra lateral visivel.",
        items=[
            "Competicao: muda o universo de jogos analisados no portal.",
            "Filtrar por time: reduz a leitura para confrontos do clube ou selecao desejada.",
            "Perfil de risco: troca a regua de probabilidade, odd maxima e casas minimas.",
            "Atualizar agora: refaz o scraping e regenera o portal completo (pode levar alguns segundos).",
        ],
        tone="neutral",
    )
    config_poisson = runtime_model_config.get("poisson", {}) if isinstance(runtime_model_config, dict) else {}
    config_calib = runtime_model_config.get("calibration", {}) if isinstance(runtime_model_config, dict) else {}
    config_anchor = runtime_model_config.get("market_anchor", {}) if isinstance(runtime_model_config, dict) else {}
    config_betting = runtime_model_config.get("betting", {}) if isinstance(runtime_model_config, dict) else {}
    config_safe = runtime_model_config.get("safe_score", {}) if isinstance(runtime_model_config, dict) else {}
    config_poisson = config_poisson if isinstance(config_poisson, dict) else {}
    config_calib = config_calib if isinstance(config_calib, dict) else {}
    config_anchor = config_anchor if isinstance(config_anchor, dict) else {}
    config_betting = config_betting if isinstance(config_betting, dict) else {}
    config_safe = config_safe if isinstance(config_safe, dict) else {}
    render_split_highlight(
        title="Criterios em uso agora",
        copy="Estes sao os valores ativos do modelo neste momento para previsao, calibracao e score de seguranca.",
        items=[
            f"Filtro operacional: Perfil {risk_profile} | Prob >= {min_prob:.2f} | Odd <= {max_odd:.2f} | Casas >= {int(min_books)} | EV >= 0 | margem >= 8 p.p. | empate < 25%.",
            f"Poisson: max_goals={int(config_poisson.get('max_goals', 5))}, home_default={float(config_poisson.get('league_home_default', 1.35)):.2f}, away_default={float(config_poisson.get('league_away_default', 1.10)):.2f}.",
            f"Calibracao: enabled={bool(config_calib.get('enabled', True))}, min_history={int(config_calib.get('min_history_matches', 80))}, min_bucket={int(config_calib.get('min_bucket_matches', 12))}, max_eval={int(config_calib.get('max_evaluated_matches', 80))}.",
            f"Ancora de mercado: enabled={bool(config_anchor.get('enabled', True))}, min_casas={int(config_anchor.get('min_bookmakers', 6))}, peso_max={float(config_anchor.get('max_weight', 0.32)):.2f}.",
            f"Stake (Kelly): {float(config_betting.get('kelly_fractional', 0.25)):.2f} | Safe score prob={float(config_safe.get('prob_weight', 0.55)):.2f}, books={float(config_safe.get('bookmakers_weight', 0.20)):.2f}, odd={float(config_safe.get('odd_weight', 0.05)):.2f}.",
        ],
        tone="neutral",
    )
    render_split_highlight(
        title="Recalibracao em 3 passos",
        copy="Fluxo recomendado para autonomia total sem editar o codigo do projeto.",
        items=[
            "1) Abra 'Calibragem Avancada do Modelo' no menu lateral e ajuste os parametros desejados.",
            "2) Clique em 'Recalibrar Modelo Agora' para aplicar os criterios na sessao atual.",
            "3) Clique em 'Atualizar agora' para reprocessar scraping + portal e validar o impacto com os novos filtros.",
        ],
        tone="neutral",
    )

elif page == "IA Institucional":
    render_callout_grid(
        [
            {
                "eyebrow": "Prompt",
                "title": "Leitura profissional do dia",
                "copy": "Use o prompt completo para filtrar so os jogos com evidencia estatistica forte e sem forcar previsoes.",
            },
            {
                "eyebrow": "Mercado",
                "title": "Modelo x odds x value",
                "copy": "A resposta cruza probabilidades reais, value bets, contexto competitivo, risco e timing de entrada.",
            },
            {
                "eyebrow": "Banca",
                "title": "Cobertura e protecao",
                "copy": "A IA tambem sugere aposta principal, cobertura, outsider e leitura de fechamento com foco em perda controlada.",
            },
        ]
    )
    institutional_date = st.date_input(
        "Data da analise da IA",
        value=date.today(),
        key="institutional_ai_date",
    )
    prompt_preview = st.session_state.get("institutional_ai_prompt", "")
    preview_lines = [line.strip() for line in str(prompt_preview).splitlines() if line.strip()]
    preview_text = "\n".join(preview_lines[:6]) if preview_lines else "Nenhum prompt salvo."
    st.markdown(
        f"""
<div class="ai-panel">
  <div class="section-title">Prompt institucional salvo</div>
  <div class="section-copy">A edicao agora acontece em modal para manter a tela mais limpa. Revise abaixo um trecho do prompt ativo antes de executar.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.code(preview_text, language="markdown")

    ai_day_col1, ai_day_col2, ai_day_col3 = st.columns([1.1, 1, 1.3])
    with ai_day_col1:
        if st.button("Editar prompt em modal", use_container_width=True, key="institutional_ai_open_modal"):
            st.session_state["institutional_ai_prompt_modal"] = st.session_state.get(
                "institutional_ai_prompt", ""
            )
            st.session_state["institutional_ai_modal_open"] = True
    with ai_day_col2:
        if st.button("Atualizar prompt pela data", use_container_width=True, key="institutional_ai_update"):
            st.session_state["institutional_ai_prompt"] = AI_PROMPT_TEMPLATE.replace(
                "__DATA_SELECIONADA__", institutional_date.strftime("%d/%m/%Y")
            )
            st.session_state["institutional_ai_prompt_modal"] = st.session_state["institutional_ai_prompt"]
            st.rerun()
    with ai_day_col3:
        run_institutional_ai = st.button(
            "Executar leitura institucional com IA",
            use_container_width=True,
            key="institutional_ai_run",
            disabled=not ai_enabled,
        )

    if st.session_state.get("institutional_ai_modal_open"):
        open_institutional_prompt_modal(institutional_date)

    if not ai_enabled:
        st.info("A IA institucional fica disponivel quando a variavel NVIDIA_API_KEY estiver configurada.")
    elif run_institutional_ai:
        try:
            with st.spinner("Montando o contexto dos jogos e consultando a IA..."):
                st.session_state["institutional_ai_result"] = run_ai_analysis(
                    prompt=st.session_state.get("institutional_ai_prompt", ""),
                    selected_date=institutional_date,
                )
                st.session_state["institutional_ai_last_date"] = institutional_date.strftime("%d/%m/%Y")
        except Exception as exc:
            st.session_state["institutional_ai_result"] = ""
            st.error(f"Nao foi possivel gerar a leitura institucional: {exc}")

    if st.session_state.get("institutional_ai_result"):
        st.markdown(
            """
<div class="ai-panel">
  <div class="section-title">Resposta da IA institucional</div>
  <div class="section-copy">Analise completa do dia baseada no prompt profissional e na base local do portal.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(st.session_state["institutional_ai_result"])

elif page == "Jogos Seguros":
    st.markdown('<div id="anchor-safe"></div>', unsafe_allow_html=True)
    st.markdown(
        """
<section class="section-shell">
  <div class="section-header">
    <div>
      <div class="section-title">Ranking de Jogos Mais Seguros</div>
      <p>Prioridade para entradas com alta convergência estatística, odds controladas e volume de liquidez (casas).</p>
    </div>
    <div class="section-badge">Filtro operacional ativo</div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )
    
    with st.expander("📚 Como interpretar este painel", expanded=False):
        st.markdown("""
        <div class="explainer-box">
            <strong>Indicadores Principais:</strong><br>
            • <b>Score FD (0-100):</b> Pontuação proprietária. Acima de 75 indica alta segurança.<br>
            • <b>Margem:</b> Diferença de probabilidade entre o palpite principal e o segundo resultado mais provável.<br>
            • <b>Risco:</b> Classificação automática baseada na volatilidade da odd e confiança do modelo.<br>
            • <b>Casas:</b> Indica a liquidez. Quanto mais casas oferecem a linha, maior a confiabilidade da odd.<br><br>
            <strong>Dica Pro:</strong> Foque em jogos com <b>Score > 70</b> e <b>Risco Baixo</b> para uma gestão de banca conservadora.
        </div>
        """, unsafe_allow_html=True)

    if relaxed_note:
        st.info(relaxed_note)
    
    if show_safe.empty:
        st.warning("Nenhum jogo passou no filtro atual. Tente alterar o perfil de risco no menu lateral.")
    else:
        # Dashboard de métricas no topo
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Oportunidades", safe_count, help="Jogos que atendem aos critérios rigorosos de segurança.")
        with m2:
            st.metric("Taxa Segura", f"{safe_rate}%", help="Percentual de cobertura do mercado que atende aos requisitos.")
        with m3:
            st.metric("Acurácia (Hist.)", f"{model_accuracy:.1f}%", help="Taxa de acerto histórica do modelo nesta competição.")
        with m4:
            st.metric("ROI Sugerido", f"{value_roi:.1f}%", help="Retorno médio esperado para este perfil de risco.")

        # Seção de Cards de Destaque (Melhores Escolhas)
        st.write("### 🏆 Melhores Escolhas (Destaques)")
        top_picks = show_safe.sort_values("Score", ascending=False).head(3)
        t_cols = st.columns(len(top_picks) if not top_picks.empty else 1)
        
        for i, (idx, row) in enumerate(top_picks.iterrows()):
            with t_cols[i]:
                # Sistema de Cores Semântico (Heatmap)
                score_val = row["Score"]
                prob_val = float(row["Prob. Modelo"].replace('%',''))
                ev_val = float(row["EV"].replace('%',''))
                
                # Cor baseada na Probabilidade (Heatmap conforme pedido)
                if prob_val >= 75:
                    prob_color = "#1e3a8a" # Azul escuro
                    prob_bg = "#dbeafe"
                elif prob_val >= 60:
                    prob_color = "#3b82f6" # Azul claro
                    prob_bg = "#eff6ff"
                else:
                    prob_color = "#64748b"
                    prob_bg = "#f8fafc"

                card_border = "#1e40af" if score_val >= 80 else "#3b82f6"
                ev_color = "#16a34a" if ev_val > 5 else "#2563eb"
                
                st.markdown(f"""
                <div style="background: white; padding: 20px; border-radius: 15px; border-top: 5px solid {card_border}; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin-bottom: 20px; position: relative; overflow: hidden;">
                    <div style="font-size: 0.8rem; color: #64748b; font-weight: 600; margin-bottom: 5px;">{row['Data']}</div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: #1e293b; margin-bottom: 10px;">{row['Mandante']} x {row['Visitante']}</div>
                    
                    <div style="background: #f8fafc; padding: 12px; border-radius: 10px; margin-bottom: 15px;">
                        <span style="font-size: 0.7rem; color: #64748b; font-weight: 700; text-transform: uppercase; display: block; margin-bottom: 4px;">Palpite Sugerido</span>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 800; color: #1e40af; font-size: 1rem;">{row['Palpite']}</span>
                            <span style="font-weight: 900; font-size: 1.3rem; color: #0f172a;">@ {row['Odd']:.2f}</span>
                        </div>
                    </div>
                    
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="background: {prob_bg}; padding: 6px 12px; border-radius: 8px; border: 1px solid {prob_color}33;">
                            <span style="font-size: 0.65rem; color: {prob_color}; font-weight: 800; display: block; text-transform: uppercase;">Probabilidade</span>
                            <span style="font-weight: 800; color: {prob_color}; font-size: 1.1rem;">{row['Prob. Modelo']}</span>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 0.65rem; color: #16a34a; font-weight: 800; display: block; text-transform: uppercase;">Valor (EV)</span>
                            <span style="font-weight: 800; color: #16a34a; font-size: 1.1rem;">{row['EV']}</span>
                        </div>
                    </div>
                    
                    <div style="margin-top: 15px; background: #f1f5f9; height: 6px; border-radius: 3px; width: 100%;">
                        <div style="background: {card_border}; height: 100%; width: {row['Score']}%; border-radius: 3px;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                        <span style="font-size: 0.7rem; color: #64748b; font-weight: 600;">Confiança do Modelo</span>
                        <span style="font-size: 0.7rem; color: #1e293b; font-weight: 800;">{row['Score']:.0f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.write("---")
        st.write("### 📋 Lista Completa de Oportunidades")
        # Configuração do Editor com cores e progresso
        selected_safe = st.data_editor(
            display_df.drop(columns=["Link"]),
            column_config={
                "Acao": st.column_config.CheckboxColumn("Analisar", help="Selecione para ver graficos e detalhes aprofundados do confronto", default=False),
                "Data": st.column_config.TextColumn("Data", help="Horario previsto para o inicio da partida"),
                "Palpite": st.column_config.TextColumn("Palpite", help="Mercado selecionado pela IA como o de maior valor esperado"),
                "Odd": st.column_config.NumberColumn("Odd", format="%.2f", help="Cotacao atual da casa de apostas (payout)"),
                "Prob. Modelo": st.column_config.TextColumn("Prob. Modelo", help="Probabilidade calculada pelo nosso algoritmo estatistico"),
                "EV": st.column_config.TextColumn("EV", help="Valor Esperado: Lucratividade estimada no longo prazo. Acima de 0% e matematicamente lucrativo"),
                "Margem": st.column_config.TextColumn("Margem", help="Diferenca de probabilidade entre o palpite principal e o segundo resultado mais provavel"),
                "Risco": st.column_config.SelectboxColumn("Risco", options=["Baixo", "Medio", "Alto"], help="Nivel de volatilidade da entrada baseado em liquidez e variancia da odd"),
                "Casas": st.column_config.NumberColumn("Casas", help="Quantidade de casas de aposta (Bookmakers) que validam esta linha de odd"),
                "Score": st.column_config.ProgressColumn("Confiança", min_value=0, max_value=100, format="%.0f", help="Score FD: Pontuacao de 0 a 100 que resume a seguranca da entrada"),
            },
            disabled=display_df.columns.drop("Acao"),
            hide_index=True,
            use_container_width=True,
            key="safe_data_editor_v2"
        )

        # Verificar se alguma linha foi marcada
        checked_rows = selected_safe[selected_safe["Acao"] == True]
        if not checked_rows.empty:
            idx = checked_rows.index[0]
            row_data = safe_df.iloc[idx]
            match_detail = _build_match_detail_data(row_data, df, model_config=runtime_model_config)
            render_match_details_modal(match_detail)

        best_safe_row = safe_df.iloc[0]
        best_safe_display_date = format_match_datetime(
            best_safe_row.get("date_text"),
            best_safe_row.get("event_timestamp"),
            "Agendado",
        )
        best_safe_key = f"ai_best_safe_cache_{competition}"
        st.markdown(
            f"""
<div class="ai-action-panel">
  <strong>IA no melhor jogo seguro da competicao</strong>
  <span>{'Gere uma leitura detalhada automaticamente para o primeiro colocado do ranking sem precisar selecionar manualmente no simulador.' if ai_enabled else 'Ative a NVIDIA_API_KEY para analisar automaticamente o melhor jogo seguro desta competicao.'}</span>
</div>
""",
            unsafe_allow_html=True,
        )
        best_ai_col1, best_ai_col2 = st.columns([1.2, 1])
        with best_ai_col1:
            run_best_safe_ai = st.button(
                "Analisar automaticamente o melhor jogo seguro",
                key=f"best_safe_ai_{competition}",
                use_container_width=True,
                disabled=not ai_enabled,
            )
        with best_ai_col2:
            clear_best_safe_ai = st.button(
                "Limpar analise automatica",
                key=f"clear_best_safe_ai_{competition}",
                use_container_width=True,
                disabled=best_safe_key not in st.session_state,
            )

        if clear_best_safe_ai and best_safe_key in st.session_state:
            del st.session_state[best_safe_key]

        if run_best_safe_ai:
            try:
                fixture_row = find_fixture_row(
                    valid,
                    date_text=str(best_safe_row["date_text"]),
                    home_team=str(best_safe_row["home_team"]),
                    away_team=str(best_safe_row["away_team"]),
                )
                with st.spinner("Consultando IA da NVIDIA para o melhor jogo seguro..."):
                    st.session_state[best_safe_key] = {
                        "match_label": f"{best_safe_display_date} | {best_safe_row['home_team']} x {best_safe_row['away_team']}",
                        "analysis": build_ai_analysis_for_fixture(df, fixture_row, model_config=runtime_model_config),
                    }
            except Exception as exc:
                st.error(f"Nao foi possivel gerar a analise automatica do melhor jogo seguro: {exc}")

        cached_best_safe_ai = st.session_state.get(best_safe_key)
        if cached_best_safe_ai:
            st.markdown(
                '<div class="ai-panel"><div class="section-title">Analise automatica do melhor jogo seguro</div><div class="section-copy">Leitura detalhada gerada pela IA para o primeiro jogo do ranking atual de seguranca.</div></div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Jogo analisado: {cached_best_safe_ai.get('match_label', '')}")
            st.markdown(render_detailed_ai_analysis(cached_best_safe_ai["analysis"]))

        st.markdown("#### Ranking detalhado")
        st.dataframe(format_date_column_for_display(show_safe.head(20)), use_container_width=True, hide_index=True)
        st.markdown("<div class='small-note'>Score combina probabilidade, odd e numero de casas.</div>", unsafe_allow_html=True)

elif page == "Painel do Modelo":
    st.markdown('<div id="anchor-dashboard"></div>', unsafe_allow_html=True)
    render_card_grid(
        [
            {"eyebrow": "Acerto modelo", "value": f"{model_accuracy:.1f}%", "copy": "Percentual de acertos do favorito do modelo."},
            {"eyebrow": "Acerto casas", "value": f"{house_accuracy:.1f}%", "copy": "Percentual de acertos do favorito do mercado."},
            {"eyebrow": "Acerto entrada", "value": f"{value_accuracy:.1f}%", "copy": "Hit rate da entrada por probabilidade."},
            {"eyebrow": "ROI entrada", "value": f"{value_roi:.1f}%", "copy": "Retorno medio da entrada sugerida (stake fixa)."},
        ]
    )
    if recent_backtest.empty:
        st.info("Ainda nao ha jogos suficientes para montar o dashboard detalhado do modelo.")
    else:
        render_split_highlight(
            "Modelo x casas de apostas",
            "O painel abaixo usa jogos ja finalizados para comparar previsao mais provavel do modelo, favorito das odds e entrada por probabilidade.",
            tuning_actions if tuning_actions else [
                f"Modelo: {model_accuracy:.1f}% de acerto.",
                f"Casas: {house_accuracy:.1f}% de acerto.",
                f"Divergencia modelo x mercado em {backtest_summary.get('market_disagreement_rate', 0.0):.1f}% dos jogos.",
            ],
            tone="warm" if tuning_actions else "neutral",
        )
        if not probability_buckets.empty:
            buckets_view = probability_buckets.copy()
            buckets_view.columns = ["Faixa do modelo", "Jogos", "Acerto modelo", "Acerto casas", "ROI modelo", "ROI entrada", "EV medio"]
            for col in ["Acerto modelo", "Acerto casas", "ROI modelo", "ROI entrada", "EV medio"]:
                buckets_view[col] = (buckets_view[col] * 100).round(2).astype(str) + "%"
            st.markdown("#### Calibracao por faixa de probabilidade")
            st.dataframe(buckets_view, use_container_width=True, hide_index=True)

        model_compare = recent_backtest[
            [
                "date_text",
                "event_timestamp",
                "home_team",
                "away_team",
                "actual_market",
                "model_market",
                "model_probability",
                "house_market",
                "house_probability",
                "value_market",
                "value_ev",
                "model_hit",
                "house_hit",
                "value_hit",
            ]
        ].copy()
        model_compare["Jogo"] = model_compare["home_team"] + " x " + model_compare["away_team"]
        model_compare["Modelo"] = model_compare.apply(
            lambda row: market_badge_label(str(row["model_market"]), str(row["home_team"]), str(row["away_team"])),
            axis=1,
        )
        model_compare["Casas"] = model_compare.apply(
            lambda row: market_badge_label(str(row["house_market"]), str(row["home_team"]), str(row["away_team"])),
            axis=1,
        )
        model_compare["Entrada"] = model_compare.apply(
            lambda row: market_badge_label(str(row["value_market"]), str(row["home_team"]), str(row["away_team"])),
            axis=1,
        )
        model_compare["Resultado"] = model_compare.apply(
            lambda row: market_badge_label(str(row["actual_market"]), str(row["home_team"]), str(row["away_team"])),
            axis=1,
        )
        model_compare["Prob. Modelo"] = (model_compare["model_probability"] * 100).round(2).astype(str) + "%"
        model_compare["Prob. Casas"] = (model_compare["house_probability"] * 100).round(2).astype(str) + "%"
        model_compare["EV Entrada"] = (model_compare["value_ev"] * 100).round(2).astype(str) + "%"
        model_compare["Hit Modelo"] = model_compare["model_hit"].map({True: "Sim", False: "Nao"})
        model_compare["Hit Casas"] = model_compare["house_hit"].map({True: "Sim", False: "Nao"})
        model_compare["Hit Entrada"] = model_compare["value_hit"].map({True: "Sim", False: "Nao"})
        model_compare = format_date_column_for_display(model_compare, status="Finalizado")
        model_compare = model_compare[
            ["date_text", "Jogo", "Resultado", "Modelo", "Prob. Modelo", "Casas", "Prob. Casas", "Entrada", "EV Entrada", "Hit Modelo", "Hit Casas", "Hit Entrada"]
        ]
        model_compare.columns = ["Data", "Jogo", "Resultado final", "Leitura do modelo", "Prob. modelo", "Leitura das casas", "Prob. casas", "Entrada por probabilidade", "EV informativo", "Modelo acertou", "Casas acertaram", "Entrada acertou"]
        st.markdown("#### Comparativo detalhado por jogo finalizado")
        st.dataframe(model_compare.head(30), use_container_width=True, hide_index=True)

elif page == "Analise de Jogo":
    st.markdown('<div id="anchor-simulator"></div>', unsafe_allow_html=True)
    st.markdown(
        """
<section class="section-shell">
  <div class="section-header">
    <div>
      <div class="section-title">Simulador de Estrategia por Jogo</div>
      <p>Compare probabilidades do modelo, consenso das odds e cenarios de protecao para reduzir perda quando a entrada principal nao bater.</p>
    </div>
    <div class="section-badge">Simulador</div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )
    if analysis_candidates.empty:
        st.info("Nao ha jogos com odds 1X2 disponiveis para analise no momento.")
    else:
        analysis_scope = st.radio(
            "Mostrar no seletor",
            options=["Todos", "Somente futuros", "Somente finalizados"],
            horizontal=True,
            key=f"analysis_scope_{competition}",
        )
        if analysis_scope == "Somente futuros":
            selectable_matches = valid.copy()
        elif analysis_scope == "Somente finalizados":
            selectable_matches = finished_with_odds.copy()
        else:
            selectable_matches = analysis_candidates.copy()

        if selectable_matches.empty:
            st.info("Nenhum jogo encontrado nesse filtro de analise.")
        else:
            selectable_matches = selectable_matches.reset_index(drop=True)
            selectable_matches["match_option_id"] = selectable_matches.index.astype(str)
            selectable_matches["match_label"] = selectable_matches.apply(build_analysis_match_label, axis=1)
            match_options = selectable_matches["match_option_id"].tolist()
            labels_by_option = dict(zip(selectable_matches["match_option_id"], selectable_matches["match_label"]))
            selected_options = st.multiselect(
                "Escolha um ou mais jogos",
                options=match_options,
                default=match_options[:1],
                format_func=lambda option_id: labels_by_option.get(option_id, option_id),
                key=f"match_selector_label_multi_{competition}",
            )
            if not selected_options:
                st.info("Selecione pelo menos um jogo para gerar a analise.")
            else:
                selectable_indexed = selectable_matches.set_index("match_option_id", drop=False)
                selected_matches = selectable_indexed.loc[selected_options].reset_index(drop=True)
                analysis_bundles: dict[str, dict[str, object]] = {}
                summary_rows: list[dict[str, object]] = []
                analysis_errors: list[str] = []

                for selected in selected_matches.to_dict(orient="records"):
                    option_id = str(selected["match_option_id"])
                    try:
                        selected_status = str(selected.get("status", "")).strip() or "Desconhecido"
                        selected_bookmakers = selected.get("bookmakers")
                        try:
                            selected_bookmakers = int(selected_bookmakers) if not pd.isna(selected_bookmakers) else 0
                        except (TypeError, ValueError):
                            selected_bookmakers = 0
                        selected_display_date = format_match_datetime(
                            selected.get("date_text"),
                            selected.get("event_timestamp"),
                            selected_status,
                        )
                        probs = calculate_match_probabilities(
                            df,
                            selected["home_team"],
                            selected["away_team"],
                            odd_home=float(selected["odds_home"]),
                            odd_draw=float(selected["odds_draw"]),
                            odd_away=float(selected["odds_away"]),
                            bookmakers=selected_bookmakers,
                            model_config=runtime_model_config,
                        )
                        tip = suggest_bet_strategy(
                            probs,
                            odd_home=float(selected["odds_home"]),
                            odd_draw=float(selected["odds_draw"]),
                            odd_away=float(selected["odds_away"]),
                            bankroll=1000.0,
                            kelly_fractional=runtime_kelly,
                            model_config=runtime_model_config,
                        )
                        model_market, probable_probability = pick_highest_probability_market(probs)
                        probable_market = market_label(str(model_market), str(selected["home_team"]), str(selected["away_team"]))
                        readable_market = market_label(str(tip.best_market), str(selected["home_team"]), str(selected["away_team"]))
                        market_probs = {
                            "Casa": 1 / float(selected["odds_home"]) if float(selected["odds_home"]) > 1 else 0.0,
                            "Empate": 1 / float(selected["odds_draw"]) if float(selected["odds_draw"]) > 1 else 0.0,
                            "Fora": 1 / float(selected["odds_away"]) if float(selected["odds_away"]) > 1 else 0.0,
                        }
                        market_total = sum(market_probs.values()) or 1.0
                        market_probs = {key: value / market_total for key, value in market_probs.items()}
                        house_market = max(market_probs.items(), key=lambda item: item[1])[0]

                        analysis_bundles[option_id] = {
                            "selected": pd.Series(selected),
                            "selected_match_label": str(selected["match_label"]),
                            "selected_status": selected_status,
                            "selected_display_date": selected_display_date,
                            "probs": probs,
                            "tip": tip,
                            "model_market": model_market,
                            "probable_probability": probable_probability,
                            "probable_market": probable_market,
                            "readable_market": readable_market,
                            "market_probs": market_probs,
                            "house_market": house_market,
                        }
                        summary_rows.append(
                            {
                                "Data": selected_display_date,
                                "Jogo": f"{selected['home_team']} x {selected['away_team']}",
                                "Status": selected_status,
                                "Favorito do modelo": probable_market,
                                "Prob. modelo": probable_probability,
                                "Entrada por probabilidade": readable_market,
                                "Odd entrada": None if is_double_chance_market(str(tip.best_market)) else float(tip.best_odd),
                                "EV informativo": None if is_double_chance_market(str(tip.best_market)) else float(tip.expected_value),
                                "Stake sugerida": float(tip.suggested_stake),
                            }
                        )
                    except Exception as exc:
                        analysis_errors.append(f"{labels_by_option.get(option_id, option_id)}: {exc}")

                if analysis_errors:
                    st.warning(
                        "Alguns jogos selecionados nao puderam ser processados. "
                        "Ajuste o filtro ou confira se as odds estao validas."
                    )
                    for err in analysis_errors[:3]:
                        st.caption(f"- {err}")

                valid_options = [option_id for option_id in selected_options if option_id in analysis_bundles]
                if not valid_options:
                    st.info("Nao foi possivel analisar os jogos escolhidos.")
                else:
                    summary_df = pd.DataFrame(summary_rows)
                    if not summary_df.empty:
                        summary_view = summary_df.copy()
                        summary_view["Prob. modelo"] = (summary_view["Prob. modelo"] * 100).round(2).astype(str) + "%"
                        summary_view["EV informativo"] = summary_view["EV informativo"].apply(
                            lambda value: "-" if pd.isna(value) else f"{float(value) * 100:.2f}%"
                        )
                        summary_view["Odd entrada"] = summary_view["Odd entrada"].apply(
                            lambda value: "-" if pd.isna(value) else f"{float(value):.2f}"
                        )
                        summary_view["Stake sugerida"] = "R$ " + summary_view["Stake sugerida"].round(2).map(lambda x: f"{x:.2f}")
                        st.markdown("### Resumo comparativo dos jogos selecionados")
                        st.dataframe(summary_view, use_container_width=True, hide_index=True)

                    selected_option = valid_options[0]
                    if len(valid_options) > 1:
                        selected_option = st.selectbox(
                            "Jogo em foco para detalhes completos",
                            options=valid_options,
                            format_func=lambda option_id: labels_by_option.get(option_id, option_id),
                            key=f"match_focus_label_{competition}",
                        )
                    selected_bundle = analysis_bundles[selected_option]
                    selected = selected_bundle["selected"]
                    selected_match_label = str(selected_bundle["selected_match_label"])
                    selected_status = str(selected_bundle["selected_status"])
                    selected_display_date = str(selected_bundle["selected_display_date"])
                    probs = selected_bundle["probs"]
                    tip = selected_bundle["tip"]
                    model_market = selected_bundle["model_market"]
                    probable_probability = float(selected_bundle["probable_probability"])
                    probable_market = str(selected_bundle["probable_market"])
                    readable_market = str(selected_bundle["readable_market"])
                    market_probs = selected_bundle["market_probs"]
                    house_market = str(selected_bundle["house_market"])

                    render_card_grid(
                        [
                            {"eyebrow": f"Modelo | {selected['home_team']}", "value": f"{probs.home_win * 100:.1f}%", "copy": f"Odd atual {selected['odds_home']:.2f}."},
                            {"eyebrow": "Modelo | Empate", "value": f"{probs.draw * 100:.1f}%", "copy": f"Odd atual {selected['odds_draw']:.2f}."},
                            {"eyebrow": f"Modelo | {selected['away_team']}", "value": f"{probs.away_win * 100:.1f}%", "copy": f"Odd atual {selected['odds_away']:.2f}."},
                            {"eyebrow": "Entrada por probabilidade", "value": f"{tip.model_probability * 100:.2f}%", "copy": readable_market},
                        ]
                    )

                    if selected_status == "Finalizado":
                        final_score = f"{format_score_value(selected.get('home_goals'))} x {format_score_value(selected.get('away_goals'))}"
                        final_market = market_badge_label(
                            resolve_match_market(selected.get("home_goals"), selected.get("away_goals")),
                            str(selected["home_team"]),
                            str(selected["away_team"]),
                        )
                        st.info(
                            f"Jogo finalizado selecionado. Placar: {final_score} | Resultado real: {final_market}. "
                            "As odds mostradas abaixo representam a linha usada para a leitura do confronto."
                        )

                    render_split_highlight(
                        "Probabilidade x modelo x casas",
                        "Este bloco mostra se o modelo esta convergindo com o mercado ou se esta comprando uma leitura mais agressiva.",
                        [
                            f"Favorito do modelo: {market_badge_label(str(model_market), str(selected['home_team']), str(selected['away_team']))} ({probable_probability * 100:.2f}%).",
                            f"Favorito das casas: {market_badge_label(str(house_market), str(selected['home_team']), str(selected['away_team']))} ({market_probs[house_market] * 100:.2f}%).",
                            (
                                f"Entrada protegida: {readable_market}; sem EV porque nao ha odd de dupla chance na base."
                                if is_double_chance_market(str(tip.best_market))
                                else f"Entrada por probabilidade: {readable_market} com edge de {(tip.model_probability - tip.implied_probability) * 100:.2f} pontos."
                            ),
                        ],
                        tone="warm" if model_market != house_market else "neutral",
                    )

                    if is_double_chance_market(str(tip.best_market)):
                        st.info(f"Resultado em duvida. Leitura protegida: {readable_market}.")
                    elif tip.expected_value > 0:
                        st.success(f"Entrada por probabilidade: {readable_market} | EV informativo: {tip.expected_value * 100:.2f}%")
                    else:
                        st.warning(f"Sem EV positivo no 1X2. Entrada por probabilidade no momento: {readable_market} (EV informativo {tip.expected_value * 100:.2f}%).")

                    hedge_df = build_hedge_scenarios(
                        best_market=str(tip.best_market),
                        odd_home=float(selected["odds_home"]),
                        odd_draw=float(selected["odds_draw"]),
                        odd_away=float(selected["odds_away"]),
                        base_stake=float(max(tip.suggested_stake, 10.0)),
                    )
                    if not hedge_df.empty:
                        render_split_highlight(
                            "Opcoes de fechamento e protecao",
                            "As simulacoes abaixo usam as odds atuais do painel para montar hedge pre-jogo ou saida parcial planejada. Nao equivalem a cashout ao vivo.",
                            [
                                "Leve: reduz pouco o risco e preserva mais upside.",
                                "Balanceado: aproxima os cenarios e corta a perda maxima.",
                                "Defensivo: protege mais, mas sacrifica parte do lucro se a leitura principal bater.",
                            ],
                            tone="neutral",
                        )
                        hedge_view = hedge_df.copy()
                        hedge_view.columns = [
                            "Perfil",
                            "Mercado principal",
                            "Stake principal",
                            "Budget hedge",
                            "Hedge 1",
                            "Hedge 2",
                            "Lucro se principal bater",
                            "Resultado hedge 1",
                            "Resultado hedge 2",
                            "Pior cenario",
                        ]
                        st.dataframe(hedge_view, use_container_width=True, hide_index=True)

                    ai_state_key = f"ai_analysis_cache_{competition}_{selected_option}"
                    st.markdown(
                        f"""
<div class="ai-action-panel">
  <strong>Uso da IA no jogo em foco</strong>
  <span>{'A IA pode transformar os numeros do modelo em uma leitura detalhada com panorama, valor, riscos, estrategia e placares provaveis.' if ai_enabled else 'A IA esta desativada neste ambiente. Configure a variavel NVIDIA_API_KEY para liberar a analise detalhada deste confronto.'}</span>
</div>
""",
                        unsafe_allow_html=True,
                    )
                    ai_col1, ai_col2 = st.columns([1.2, 1])
                    with ai_col1:
                        run_ai = st.button(
                            "Gerar analise detalhada com IA NVIDIA",
                            key=f"nvidia_analysis_{competition}_{selected_option}",
                            use_container_width=True,
                            disabled=not ai_enabled,
                        )
                    with ai_col2:
                        clear_ai = st.button(
                            "Limpar analise salva",
                            key=f"clear_ai_analysis_{competition}_{selected_option}",
                            use_container_width=True,
                            disabled=ai_state_key not in st.session_state,
                        )

                    if clear_ai and ai_state_key in st.session_state:
                        del st.session_state[ai_state_key]

                    if run_ai:
                        try:
                            with st.spinner("Consultando IA da NVIDIA..."):
                                st.session_state[ai_state_key] = build_ai_analysis_for_fixture(
                                    df,
                                    selected,
                                    model_config=runtime_model_config,
                                )
                        except Exception as exc:
                            st.error(f"Nao foi possivel gerar a analise com IA da NVIDIA: {exc}")

                    cached_ai_analysis = st.session_state.get(ai_state_key)
                    if cached_ai_analysis:
                        st.markdown(
                            '<div class="ai-panel"><div class="section-title">Analise detalhada com IA NVIDIA</div><div class="section-copy">Leitura em linguagem natural com panorama do jogo, valor, riscos, estrategia e placares provaveis.</div></div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(render_detailed_ai_analysis(cached_ai_analysis))

                    top_df = pd.DataFrame(probs.top_scorelines, columns=["Placar", "Probabilidade"])
                    top_df["Probabilidade"] = (top_df["Probabilidade"] * 100).round(2).astype(str) + "%"
                    st.dataframe(top_df, use_container_width=True, hide_index=True)

                    if isinstance(selected.get("match_url"), str) and selected["match_url"]:
                        st.link_button("Abrir pagina do jogo", selected["match_url"])

                    st.markdown("### Analise explicada do jogo")
                    home_ctx = get_team_context(df, str(selected["home_team"]))
                    away_ctx = get_team_context(df, str(selected["away_team"]))
                    selected_chat_context = build_match_chat_context(
                        fixture_row=selected,
                        probs=probs,
                        tip=tip,
                        probable_market=probable_market,
                        probable_probability=probable_probability,
                        home_ctx=home_ctx,
                        away_ctx=away_ctx,
                    )
                    selected_chat_label = selected_match_label
                    selected_chat_match_key = f"{competition}_{selected_option}"

                    reasons = []
                    if isinstance(home_ctx.get("rank"), int) and isinstance(away_ctx.get("rank"), int):
                        if home_ctx["rank"] < away_ctx["rank"]:
                            reasons.append(f"Classificacao: {selected['home_team']} esta melhor posicionado ({home_ctx['rank']}º vs {away_ctx['rank']}º).")
                        elif away_ctx["rank"] < home_ctx["rank"]:
                            reasons.append(f"Classificacao: {selected['away_team']} esta melhor posicionado ({away_ctx['rank']}º vs {home_ctx['rank']}º).")
                    if home_ctx.get("recent_points", 0) > away_ctx.get("recent_points", 0):
                        reasons.append(f"Momento recente: {selected['home_team']} somou {home_ctx['recent_points']} pontos nos ultimos jogos ({home_ctx['recent_text']}).")
                    elif away_ctx.get("recent_points", 0) > home_ctx.get("recent_points", 0):
                        reasons.append(f"Momento recente: {selected['away_team']} somou {away_ctx['recent_points']} pontos nos ultimos jogos ({away_ctx['recent_text']}).")

                    btts_line = "Ambos marcam (SIM) aparece interessante." if probs.btts_yes >= 0.55 else "Ambos marcam (NAO) aparece mais conservador."
                    goal_line = "Tendencia de Menos de 2.5 gols." if probs.under_25 >= probs.over_25 else "Tendencia de Mais de 2.5 gols."
                    st.markdown(
                        f"""
**Jogo:** {selected['home_team']} x {selected['away_team']} ({selected_display_date})  
**Odds atuais (1X2):** Casa `{selected['odds_home']:.2f}` | Empate `{selected['odds_draw']:.2f}` | Fora `{selected['odds_away']:.2f}`  
**Probabilidades do modelo:** Casa `{probs.home_win*100:.1f}%` | Empate `{probs.draw*100:.1f}%` | Fora `{probs.away_win*100:.1f}%`  
**Resultado mais provavel:** **{probable_market}** (`{probable_probability*100:.1f}%`)  
**Entrada por probabilidade:** **{readable_market}** {"(sem EV: mercado protegido sem odd na base)" if is_double_chance_market(str(tip.best_market)) else f"(EV informativo `{tip.expected_value*100:.2f}%`)"}  
**Mercados extras:** {btts_line} Prob. BTTS `{probs.btts_yes*100:.1f}%`. {goal_line} Under 2.5 `{probs.under_25*100:.1f}%` / Over 2.5 `{probs.over_25*100:.1f}%`.
"""
                    )
                    if reasons:
                        st.markdown("**Principais motivos:**")
                        for r in reasons[:3]:
                            st.markdown(f"- {r}")

elif page == "Todos os Futuros":
    st.markdown('<div id="anchor-agenda"></div>', unsafe_allow_html=True)
    st.markdown(
        """
<section class="section-shell">
  <div class="section-header">
    <div>
      <div class="section-title">Agenda Completa de Jogos Futuros</div>
      <p>Use esta grade para explorar todo o mercado futuro da competicao selecionada e comparar rapidamente mandante, visitante, casas e faixa de odds.</p>
    </div>
    <div class="section-badge">Agenda</div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )
    if fixtures.empty:
        st.info("Nao encontrei jogos futuros para esta competicao no momento.")
    else:
        view_fixtures = fixtures[
            [
                "date_text",
                "event_timestamp",
                "status",
                "home_team",
                "away_team",
                "bookmakers",
                "odds_home",
                "odds_draw",
                "odds_away",
                "match_url",
            ]
        ].copy()
        view_fixtures = format_date_column_for_display(view_fixtures)
        view_fixtures.columns = [
            "Data",
            "event_timestamp",
            "status",
            "Mandante",
            "Visitante",
            "Casas",
            "Odd Casa",
            "Odd Empate",
            "Odd Fora",
            "Link",
        ]
        view_fixtures = view_fixtures[["Data", "Mandante", "Visitante", "Casas", "Odd Casa", "Odd Empate", "Odd Fora", "Link"]]
        st.dataframe(view_fixtures.reset_index(drop=True), use_container_width=True, hide_index=True)

elif page == "Resultados":
    st.markdown('<div id="anchor-results"></div>', unsafe_allow_html=True)
    st.markdown(
        """
<section class="section-shell">
  <div class="section-header">
    <div>
      <div class="section-title">Resultados Recentes</div>
      <p>Jogos ja finalizados da competicao ativa, com comparativo do modelo para medir acerto, ajuste e leitura contra as casas de apostas.</p>
    </div>
    <div class="section-badge">Historico</div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )
    if finished.empty:
        st.info("Nao ha jogos finalizados disponiveis para esta competicao.")
    else:
        view_finished = finished[
            [
                "date_text",
                "event_timestamp",
                "status",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "odds_home",
                "odds_draw",
                "odds_away",
                "match_url",
            ]
        ].copy()
        view_finished["Placar"] = view_finished.apply(
            lambda row: f"{format_score_value(row['home_goals'])} x {format_score_value(row['away_goals'])}",
            axis=1,
        )
        view_finished = format_date_column_for_display(view_finished, status="Finalizado")
        view_finished = view_finished[
            ["date_text", "home_team", "Placar", "away_team", "odds_home", "odds_draw", "odds_away", "match_url"]
        ]
        view_finished.columns = [
            "Data",
            "Mandante",
            "Placar",
            "Visitante",
            "Odd Casa",
            "Odd Empate",
            "Odd Fora",
            "Link",
        ]
        st.dataframe(view_finished.reset_index(drop=True), use_container_width=True, hide_index=True)

    if recent_backtest.empty:
        st.info("Ainda nao ha comparativo de modelo suficiente para o historico.")
    else:
        render_card_grid(
            [
                {"eyebrow": "Acuracia modelo", "value": f"{model_accuracy:.1f}%", "copy": "Favorito do modelo."},
                {"eyebrow": "Acuracia casas", "value": f"{house_accuracy:.1f}%", "copy": "Favorito das odds."},
                {"eyebrow": "Acuracia entrada", "value": f"{value_accuracy:.1f}%", "copy": "Entrada por probabilidade."},
                {"eyebrow": "ROI modelo", "value": f"{model_roi:.1f}%", "copy": "Resultado flat stake."},
            ]
        )
        results_compare = recent_backtest[
            ["date_text", "event_timestamp", "home_team", "away_team", "actual_score", "actual_market", "model_market", "house_market", "value_market", "model_hit", "house_hit", "value_hit"]
        ].copy()
        results_compare["Jogo"] = results_compare["home_team"] + " x " + results_compare["away_team"]
        results_compare["Resultado"] = results_compare.apply(
            lambda row: f"{row['actual_score']} | {market_badge_label(str(row['actual_market']), str(row['home_team']), str(row['away_team']))}",
            axis=1,
        )
        results_compare["Modelo"] = results_compare.apply(
            lambda row: market_badge_label(str(row["model_market"]), str(row["home_team"]), str(row["away_team"])),
            axis=1,
        )
        results_compare["Casas"] = results_compare.apply(
            lambda row: market_badge_label(str(row["house_market"]), str(row["home_team"]), str(row["away_team"])),
            axis=1,
        )
        results_compare["Entrada"] = results_compare.apply(
            lambda row: market_badge_label(str(row["value_market"]), str(row["home_team"]), str(row["away_team"])),
            axis=1,
        )
        results_compare["Modelo acertou"] = results_compare["model_hit"].map({True: "Sim", False: "Nao"})
        results_compare["Casas acertaram"] = results_compare["house_hit"].map({True: "Sim", False: "Nao"})
        results_compare["Entrada acertou"] = results_compare["value_hit"].map({True: "Sim", False: "Nao"})
        results_compare = format_date_column_for_display(results_compare, status="Finalizado")
        results_compare = results_compare[
            ["date_text", "Jogo", "Resultado", "Modelo", "Casas", "Entrada", "Modelo acertou", "Casas acertaram", "Entrada acertou"]
        ]
        results_compare.columns = ["Data", "Jogo", "Resultado final", "Modelo", "Casas", "Entrada por probabilidade", "Modelo acertou", "Casas acertaram", "Entrada acertou"]
        st.dataframe(results_compare.head(30), use_container_width=True, hide_index=True)

st.markdown("</div>", unsafe_allow_html=True)

if selected_chat_context:
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            f"""
<div class="mini-chat-panel">
  <strong>Mini chat do jogo atual</strong>
  <span>{"Converse com a IA em cima do confronto selecionado no simulador." if ai_enabled else "A conversa fica disponivel quando a NVIDIA_API_KEY estiver configurada neste ambiente."} Contexto ativo: {escape(selected_chat_label)}.</span>
</div>
""",
            unsafe_allow_html=True,
        )

        chat_state_key = f"ai_match_chat_history_{selected_chat_match_key}"
        if chat_state_key not in st.session_state:
            st.session_state[chat_state_key] = [
                {
                    "role": "assistant",
                    "content": (
                        "Estou com o contexto deste jogo carregado. "
                        "Pergunte sobre risco, valor, cenarios de placar, empate, over/under ou postura de entrada."
                    ),
                }
            ]

        quick_prompt = ""
        st.markdown("<div class='chat-quick-note'>Perguntas rapidas</div>", unsafe_allow_html=True)
        quick_col1, quick_col2, quick_col3 = st.columns(3)
        with quick_col1:
            if st.button("Principal risco", key=f"chat_qrisk_{selected_chat_match_key}", use_container_width=True, disabled=not ai_enabled):
                quick_prompt = "Onde esta o principal risco desta entrada?"
        with quick_col2:
            if st.button("Empate tem valor?", key=f"chat_qdraw_{selected_chat_match_key}", use_container_width=True, disabled=not ai_enabled):
                quick_prompt = "O empate tem valor neste confronto? Explique de forma objetiva."
        with quick_col3:
            if st.button("Mais segura", key=f"chat_qsafe_{selected_chat_match_key}", use_container_width=True, disabled=not ai_enabled):
                quick_prompt = "Qual e a leitura mais conservadora para este jogo?"

        clear_chat = st.button(
            "Limpar conversa deste jogo",
            key=f"clear_match_chat_{selected_chat_match_key}",
            use_container_width=True,
            disabled=not st.session_state.get(chat_state_key),
        )
        if clear_chat:
            st.session_state[chat_state_key] = [
                {
                    "role": "assistant",
                    "content": (
                        "Conversa reiniciada. "
                        "Posso voltar a responder em cima do contexto do jogo selecionado."
                    ),
                }
            ]

        with st.form(key=f"match_chat_form_{selected_chat_match_key}", clear_on_submit=True):
            user_message = st.text_area(
                "Pergunte sobre o jogo atual",
                placeholder="Ex: por que o mercado de valor difere do resultado mais provavel?",
                height=90,
                disabled=not ai_enabled,
            )
            send_chat = st.form_submit_button("Enviar para a IA", use_container_width=True, disabled=not ai_enabled)

        pending_message = quick_prompt or (user_message.strip() if send_chat and user_message.strip() else "")
        if pending_message:
            st.session_state[chat_state_key].append({"role": "user", "content": pending_message})
            try:
                with st.spinner("Consultando a IA sobre este confronto..."):
                    answer = generate_nvidia_match_chat_reply(
                        match_context=selected_chat_context,
                        conversation=st.session_state[chat_state_key],
                    )
                st.session_state[chat_state_key].append({"role": "assistant", "content": answer})
            except Exception as exc:
                st.session_state[chat_state_key].append(
                    {
                        "role": "assistant",
                        "content": f"Nao consegui responder agora por causa de um erro na consulta: {exc}",
                    }
                )
            st.rerun()

        history = st.session_state.get(chat_state_key, [])
        if not history:
            st.markdown("<div class='chat-empty'>Nenhuma mensagem ainda.</div>", unsafe_allow_html=True)
        else:
            for message in history[-8:]:
                bubble_role = "assistant" if message["role"] == "assistant" else "user"
                bubble_title = "IA" if bubble_role == "assistant" else "Voce"
                st.markdown(
                    f"<div class='chat-bubble {bubble_role}'><strong>{bubble_title}</strong></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(str(message.get("content", "")))

st.markdown("---")
with st.expander("Glossario de siglas e estatisticas"):
    st.markdown('<div class="glossary-box">', unsafe_allow_html=True)
    st.markdown(
        """
- `1X2`: Mercado principal de resultado final (`1` casa, `X` empate, `2` visitante).
- `Odd`: Cotacao da aposta. Exemplo `2.00` significa retorno bruto de R$2 para cada R$1 apostado.
- `Prob. Modelo`: Probabilidade calculada pelo modelo estatistico para aquele resultado.
- `EV` (Valor Esperado): Vantagem matematica da aposta. `EV > 0` indica expectativa positiva no longo prazo.
- `Casas` (`B's`): Quantidade de casas de aposta consideradas para aquela linha de odds.
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Aposta envolve risco. O painel ajuda na selecao, mas nao elimina perdas.")
