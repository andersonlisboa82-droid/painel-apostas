from __future__ import annotations

import json
import os
import time
from datetime import datetime
from html import escape
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

from analytics import (
    build_safe_bets_table,
    calculate_match_probabilities,
    get_team_context,
    suggest_bet_strategy,
)
from scraper import COMPETITIONS, load_competition_matches

st.set_page_config(page_title="Sistema Apostas Futebol", layout="wide")


APP_TIMEZONE = ZoneInfo("America/Sao_Paulo")
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")


def current_app_timestamp() -> str:
    return datetime.now(APP_TIMEZONE).strftime("%d/%m/%Y %H:%M:%S")


def render_floating_dock(competition_name: str, risk_profile: str) -> None:
    dock_html = """
<script>
const competitionName = __COMPETITION__;
const riskLabel = __RISK__;
const parentWindow = window.parent;
const doc = parentWindow.document;

if (typeof parentWindow.__fdDockCleanup === "function") {
  parentWindow.__fdDockCleanup();
}

const previousDock = doc.getElementById("fd-floating-dock");
if (previousDock) previousDock.remove();
const previousStyle = doc.getElementById("fd-floating-dock-style");
if (previousStyle) previousStyle.remove();

const style = doc.createElement("style");
style.id = "fd-floating-dock-style";
style.textContent = `
#fd-floating-dock {
  position: fixed;
  right: 22px;
  bottom: 22px;
  z-index: 999999;
  display: grid;
  gap: 12px;
  justify-items: end;
  pointer-events: none;
}
#fd-floating-dock .fd-meta {
  pointer-events: auto;
  display: grid;
  gap: 2px;
  min-width: 170px;
  padding: 12px 14px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(8,22,37,0.94), rgba(20,48,79,0.96));
  color: #f8fafc;
  border: 1px solid rgba(148,163,184,0.22);
  box-shadow: 0 18px 40px rgba(15,23,42,0.24);
  backdrop-filter: blur(16px);
}
#fd-floating-dock .fd-meta strong {
  font-size: 0.9rem;
  line-height: 1.1;
}
#fd-floating-dock .fd-meta span {
  font-size: 0.76rem;
  color: #cfe4ff;
}
#fd-floating-dock .fd-stack {
  display: grid;
  gap: 10px;
}
#fd-floating-dock .fd-btn {
  pointer-events: auto;
  display: inline-flex;
  align-items: center;
  justify-content: flex-start;
  gap: 10px;
  min-width: 54px;
  height: 54px;
  padding: 0 18px;
  border: 0;
  border-radius: 999px;
  background: rgba(255,255,255,0.96);
  color: #112031;
  cursor: pointer;
  box-shadow: 0 18px 34px rgba(15,23,42,0.16);
  border: 1px solid rgba(148,163,184,0.22);
  transition: transform .18s ease, box-shadow .18s ease, width .18s ease, background .18s ease;
  overflow: hidden;
}
#fd-floating-dock .fd-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 22px 42px rgba(15,23,42,0.22);
}
#fd-floating-dock .fd-btn:active {
  transform: translateY(0);
}
#fd-floating-dock .fd-btn svg {
  width: 18px;
  height: 18px;
  flex: 0 0 18px;
}
#fd-floating-dock .fd-btn .fd-label {
  font: 700 0.82rem/1 "Plus Jakarta Sans", "Segoe UI", sans-serif;
  white-space: nowrap;
  letter-spacing: -0.01em;
}
#fd-floating-dock .fd-btn[data-action="safe"] {
  background: linear-gradient(135deg, #eff6ff, #ecfeff);
  color: #1d4ed8;
}
#fd-floating-dock .fd-btn[data-action="simulator"] {
  background: linear-gradient(135deg, #f8fafc, #eff6ff);
  color: #0f766e;
}
  #fd-floating-dock .fd-btn[data-action="agenda"] {
  background: linear-gradient(135deg, #fff7ed, #fffbeb);
  color: #d97706;
}
#fd-floating-dock .fd-btn[data-action="results"] {
  background: linear-gradient(135deg, #ecfdf3, #f0fdf4);
  color: #059669;
}
#fd-floating-dock .fd-btn[data-action="refresh"] {
  background: linear-gradient(135deg, #081625, #14304f);
  color: #f8fafc;
}
#fd-floating-dock .fd-btn[data-action="top"] {
  opacity: 0;
  transform: translateY(12px);
  pointer-events: none;
}
#fd-floating-dock.is-scrolled .fd-btn[data-action="top"] {
  opacity: 1;
  transform: translateY(0);
  pointer-events: auto;
}
@media (max-width: 760px) {
  #fd-floating-dock {
    right: 12px;
    left: 12px;
    bottom: 12px;
    justify-items: stretch;
  }
  #fd-floating-dock .fd-meta {
    min-width: 0;
  }
  #fd-floating-dock .fd-stack {
    grid-template-columns: repeat(6, minmax(0, 1fr));
  }
  #fd-floating-dock .fd-btn {
    width: 100%;
    min-width: 0;
    justify-content: center;
    padding: 0;
  }
  #fd-floating-dock .fd-btn .fd-label {
    display: none;
  }
}
`;
doc.head.appendChild(style);

const dock = doc.createElement("div");
dock.id = "fd-floating-dock";
dock.innerHTML = `
  <div class="fd-meta">
    <strong>${competitionName}</strong>
    <span>Atalho rapido - ${riskLabel}</span>
  </div>
  <div class="fd-stack">
    <button type="button" class="fd-btn" data-action="top" aria-label="Voltar ao topo" title="Voltar ao topo">
      <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 5l-7 7m7-7l7 7m-7-7v14" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path></svg>
      <span class="fd-label">Topo</span>
    </button>
    <button type="button" class="fd-btn" data-action="safe" aria-label="Ir para jogos seguros" title="Ir para jogos seguros">
      <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 3l7 3v5c0 4.5-2.9 8.7-7 10-4.1-1.3-7-5.5-7-10V6l7-3z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"></path><path d="M9 12.2l2 2 4-4.2" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path></svg>
      <span class="fd-label">Seguros</span>
    </button>
    <button type="button" class="fd-btn" data-action="simulator" aria-label="Ir para simulador" title="Ir para simulador">
      <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 17l5-5 4 3 7-8" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path><path d="M15 7h5v5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path></svg>
      <span class="fd-label">Simular</span>
    </button>
    <button type="button" class="fd-btn" data-action="agenda" aria-label="Ir para agenda" title="Ir para agenda">
      <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><rect x="4" y="5" width="16" height="15" rx="3" stroke="currentColor" stroke-width="1.8"></rect><path d="M8 3v4M16 3v4M4 10h16" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"></path></svg>
      <span class="fd-label">Agenda</span>
    </button>
    <button type="button" class="fd-btn" data-action="results" aria-label="Ir para resultados" title="Ir para resultados">
      <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M5 12h14M12 5v14" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"></path><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.8"></circle></svg>
      <span class="fd-label">Resultados</span>
    </button>
    <button type="button" class="fd-btn" data-action="refresh" aria-label="Recarregar painel" title="Recarregar painel">
      <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M20 11a8 8 0 1 0 2 5.3" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"></path><path d="M20 4v7h-7" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path></svg>
      <span class="fd-label">Atualizar</span>
    </button>
  </div>
`;
doc.body.appendChild(dock);

const scrollToId = (id) => {
  const target = doc.getElementById(id);
  if (target) {
    target.scrollIntoView({ behavior: "smooth", block: "start" });
  }
};

const clickTab = (label) => {
  const tabs = Array.from(doc.querySelectorAll('[data-baseweb="tab"]'));
  const target = tabs.find((tab) => (tab.innerText || "").trim().toLowerCase() === label.toLowerCase());
  if (target) {
    target.click();
    return true;
  }
  return false;
};

const openTabAndScroll = (label, anchorId) => {
  clickTab(label);
  parentWindow.setTimeout(() => scrollToId(anchorId), 120);
  parentWindow.setTimeout(() => scrollToId(anchorId), 360);
};

const updateScrolledState = () => {
  const top = parentWindow.scrollY || doc.documentElement.scrollTop || doc.body.scrollTop || 0;
  dock.classList.toggle("is-scrolled", top > 260);
};

dock.querySelectorAll("[data-action]").forEach((button) => {
  button.addEventListener("click", () => {
    const action = button.getAttribute("data-action");
    if (action === "top") scrollToId("panel-top-anchor");
    if (action === "safe") openTabAndScroll("Jogos Seguros", "anchor-safe");
    if (action === "simulator") openTabAndScroll("Analise de Jogo", "anchor-simulator");
    if (action === "agenda") openTabAndScroll("Todos os Futuros", "anchor-agenda");
    if (action === "results") openTabAndScroll("Resultados", "anchor-results");
    if (action === "refresh") parentWindow.location.reload();
  });
});

parentWindow.addEventListener("scroll", updateScrolledState, { passive: true });
updateScrolledState();

parentWindow.__fdDockCleanup = () => {
  parentWindow.removeEventListener("scroll", updateScrolledState);
  const mountedDock = doc.getElementById("fd-floating-dock");
  if (mountedDock) mountedDock.remove();
  const mountedStyle = doc.getElementById("fd-floating-dock-style");
  if (mountedStyle) mountedStyle.remove();
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


@st.cache_data(ttl=900, show_spinner=False)
def get_data(competition: str) -> pd.DataFrame:
    return load_competition_matches(competition)


def market_label(market: str, home_team: str, away_team: str) -> str:
    if market == "Casa":
        return f"Vitoria {home_team}"
    if market == "Fora":
        return f"Vitoria {away_team}"
    return "Empate"


def format_score_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:.1f}"


def filter_matches_by_team(frame: pd.DataFrame, team_query: str) -> pd.DataFrame:
    query = (team_query or "").strip().casefold()
    if not query or frame.empty:
        return frame.copy()

    home = frame["home_team"].fillna("").astype(str).str.casefold()
    away = frame["away_team"].fillna("").astype(str).str.casefold()
    mask = home.str.contains(query, regex=False) | away.str.contains(query, regex=False)
    return frame.loc[mask].copy()


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
        "melhor aposta por valor:": "Melhor aposta por valor",
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
        for title in ["Resultado mais provavel", "Melhor aposta por valor", "Cuidado antes de apostar"]:
            body = " ".join(collected[title]).strip(" .:-")
            if body:
                blocks.append(f"**{title}**\n{body}.")
        if blocks:
            return "\n\n".join(blocks)

    plain_text = " ".join(joined.split())
    markers = [
        ("Resultado mais provavel", ["O resultado mais provavel", "Resultado mais provavel"]),
        ("Melhor aposta por valor", ["A melhor aposta por valor", "Melhor aposta por valor"]),
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
    labels = ["Resultado mais provavel", "Melhor aposta por valor", "Cuidado antes de apostar"]
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
        ("Melhor aposta por valor", valor),
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


def build_ai_analysis_for_fixture(matches_df: pd.DataFrame, fixture_row: pd.Series) -> dict[str, object]:
    probs = calculate_match_probabilities(matches_df, fixture_row["home_team"], fixture_row["away_team"])
    tip = suggest_bet_strategy(
        probs,
        odd_home=float(fixture_row["odds_home"]),
        odd_draw=float(fixture_row["odds_draw"]),
        odd_away=float(fixture_row["odds_away"]),
        bankroll=1000.0,
        kelly_fractional=0.25,
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
        date_text=str(fixture_row["date_text"]),
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
    api_key = (os.getenv("NVIDIA_API_KEY") or "").strip()
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
- Mercado de Valor sugerido: {value_market} (EV {expected_value * 100:.2f}%)

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
    top_scorelines = ", ".join(
        f"{score} ({prob * 100:.1f}%)" for score, prob in probs.top_scorelines[:4]
    )
    return f"""
Contexto do jogo atual:
- Jogo: {fixture_row['home_team']} x {fixture_row['away_team']}
- Data/Horario exibido no painel: {fixture_row['date_text']}
- Odds 1X2: Casa {float(fixture_row['odds_home']):.2f} | Empate {float(fixture_row['odds_draw']):.2f} | Fora {float(fixture_row['odds_away']):.2f}
- Probabilidades do modelo: Casa {probs.home_win * 100:.1f}% | Empate {probs.draw * 100:.1f}% | Fora {probs.away_win * 100:.1f}%
- Resultado mais provavel: {probable_market} ({probable_probability * 100:.1f}%)
- Melhor aposta por valor: {readable_market}
- EV da aposta por valor: {tip.expected_value * 100:.2f}%
- Prob. modelo da aposta por valor: {tip.model_probability * 100:.2f}%
- Prob. implicita da odd: {tip.implied_probability * 100:.2f}%
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
:root {{
  --bg: #eef3f8;
  --card: rgba(255,255,255,0.9);
  --line: rgba(148,163,184,0.28);
  --line-strong: rgba(148,163,184,0.42);
  --text: #112031;
  --muted: #5b6b7d;
  --blue: #1d4ed8;
  --teal: #0f766e;
  --amber: #d97706;
  --shadow: 0 22px 50px rgba(15,23,42,0.09);
}}
.stApp {{
  background:
    radial-gradient(900px 420px at 0% 0%, rgba(59,130,246,0.16), transparent 55%),
    radial-gradient(860px 420px at 100% 0%, rgba(16,185,129,0.14), transparent 50%),
    var(--bg);
}}
.block-container {{
  padding-top: 1.1rem;
  max-width: 1320px;
}}
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(8,22,37,0.98), rgba(20,48,79,0.98));
  border-right: 1px solid rgba(148,163,184,0.16);
}}
[data-testid="stSidebar"] * {{
  color: #e5eef8;
}}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMarkdown h3 {{
  color: #ffffff !important;
}}
[data-testid="stSidebar"] .stCaption {{
  color: #bfd6ee !important;
}}
.topbar {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 14px;
  margin-bottom: 16px;
  padding: 14px 16px;
  border-radius: 20px;
  background: rgba(255,255,255,.72);
  border: 1px solid rgba(148,163,184,.22);
  box-shadow: 0 16px 34px rgba(15,23,42,.05);
  backdrop-filter: blur(16px);
}}
.brand-block {{
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
}}
.brand-mark {{
  width: 42px;
  height: 42px;
  border-radius: 14px;
  background: linear-gradient(135deg, #1d4ed8, #0f766e);
  color: #fff;
  display: grid;
  place-items: center;
  font-size: 15px;
  font-weight: 800;
  letter-spacing: .08em;
}}
.brand-copy strong {{
  display: block;
  font-size: .96rem;
  letter-spacing: -.02em;
}}
.brand-copy span {{
  display: block;
  margin-top: 3px;
  color: var(--muted);
  font-size: .83rem;
}}
.topbar-meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}}
.meta-pill {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 9px 12px;
  border-radius: 999px;
  background: #f8fafc;
  border: 1px solid rgba(148,163,184,.25);
  color: #334155;
  font-size: .82rem;
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
}}
.stButton > button[kind="primary"] {{
  color: #ffffff !important;
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
hr {{
  border: none;
  border-top: 1px solid var(--line);
  margin: 1rem 0;
}}
@media (max-width: 1100px) {{
  .hero-grid, .hero-metrics, .info-strip {{
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }}
}}
@media (max-width: 760px) {{
  .topbar, .hero-grid, .hero-metrics, .info-strip {{
    display: grid;
    grid-template-columns: 1fr;
  }}
}}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Configuracoes")
    competition = st.selectbox("Competicao", options=list(COMPETITIONS.keys()))
    team_filter = st.text_input("Filtrar por time", placeholder="Ex: Flamengo")
    risk_profile = st.selectbox(
        "Perfil de risco",
        options=["Baixo risco", "Medio risco", "Alto risco", "Personalizado"],
        index=0,
    )

    profile_presets = {
        "Baixo risco": {"min_prob": 0.58, "min_ev": 0.02, "max_odd": 2.30},
        "Medio risco": {"min_prob": 0.50, "min_ev": 0.01, "max_odd": 2.80},
        "Alto risco": {"min_prob": 0.40, "min_ev": 0.00, "max_odd": 4.00},
    }

    competition_min_books = {
        "Brasileirao": {"Baixo risco": 2, "Medio risco": 2, "Alto risco": 1},
        "Premier League": {"Baixo risco": 5, "Medio risco": 4, "Alto risco": 3},
        "La Liga": {"Baixo risco": 8, "Medio risco": 6, "Alto risco": 4},
        "Copa do Mundo": {"Baixo risco": 10, "Medio risco": 8, "Alto risco": 5},
    }

    if risk_profile == "Personalizado":
        min_prob = st.slider("Prob. minima do modelo", min_value=0.40, max_value=0.80, value=0.55, step=0.01)
        min_ev = st.slider("EV minimo", min_value=0.00, max_value=0.15, value=0.02, step=0.005)
        max_odd = st.slider("Odd maxima", min_value=1.20, max_value=4.00, value=2.20, step=0.05)
        min_books = st.slider("Minimo de casas (B's)", min_value=1, max_value=20, value=8, step=1)
    else:
        preset = profile_presets[risk_profile]
        min_prob = preset["min_prob"]
        min_ev = preset["min_ev"]
        max_odd = preset["max_odd"]
        min_books = competition_min_books.get(competition, {}).get(risk_profile, 3)
        st.caption(
            f"Filtro aplicado: Prob >= {min_prob:.2f} | "
            f"EV >= {min_ev:.2f} | Odd <= {max_odd:.2f} | Casas >= {min_books}"
        )

    if st.button("Atualizar agora", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    ai_enabled = bool((os.getenv("NVIDIA_API_KEY") or "").strip())
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

try:
    df = get_data(competition)
except Exception as exc:
    st.error(f"Falha ao buscar dados da internet: {exc}")
    st.stop()

finished = df[df["status"] == "Finalizado"].copy()
fixtures = df[df["status"] == "Agendado"].copy()
valid = fixtures.dropna(subset=["odds_home", "odds_draw", "odds_away"]).copy()
ai_enabled = bool((os.getenv("NVIDIA_API_KEY") or "").strip())

if fixtures.empty:
    st.warning("Nao encontrei jogos futuros para esta competicao no momento.")
    st.stop()

safe_df = build_safe_bets_table(
    matches_df=df,
    bankroll=1000.0,
    kelly_fractional=0.25,
    min_model_prob=float(min_prob),
    min_expected_value=float(min_ev),
    max_odd=float(max_odd),
    min_bookmakers=int(min_books),
)

if team_filter.strip():
    finished = filter_matches_by_team(finished, team_filter)
    fixtures = filter_matches_by_team(fixtures, team_filter)
    valid = filter_matches_by_team(valid, team_filter)
    safe_df = filter_matches_by_team(safe_df, team_filter)

relaxed_note = ""
if safe_df.empty and risk_profile != "Personalizado":
    relax_steps = [
        (max(min_prob - 0.05, 0.35), max(min_ev - 0.01, -0.02), max_odd + 0.30, max(min_books - 1, 1)),
        (max(min_prob - 0.10, 0.30), max(min_ev - 0.02, -0.05), max_odd + 0.70, 1),
    ]
    for rp, rev, rodd, rbooks in relax_steps:
        safe_df = build_safe_bets_table(
            matches_df=df,
            bankroll=1000.0,
            kelly_fractional=0.25,
            min_model_prob=float(rp),
            min_expected_value=float(rev),
            max_odd=float(rodd),
            min_bookmakers=int(rbooks),
        )
        if team_filter.strip():
            safe_df = filter_matches_by_team(safe_df, team_filter)
        if not safe_df.empty:
            relaxed_note = (
                f"Filtro do perfil foi relaxado automaticamente para exibir opcoes: "
                f"Prob >= {rp:.2f} | EV >= {rev:.2f} | Odd <= {rodd:.2f} | Casas >= {rbooks}."
            )
            break

competition_name = str(competition)
team_filter_display = escape(team_filter.strip())
finished_count = len(finished)
fixtures_count = len(fixtures)
odds_count = len(valid)
safe_count = len(safe_df)
recommendation_count = len(valid)
odds_coverage = _safe_percent(odds_count, fixtures_count)
safe_rate = _safe_percent(safe_count, odds_count)
recommendation_rate = _safe_percent(recommendation_count, odds_count)

st.markdown(
    f"""
<div id="panel-top-anchor"></div>
<section class="topbar">
  <div class="brand-block">
    <div class="brand-mark">FD</div>
    <div class="brand-copy">
      <strong>Football Data Desk</strong>
      <span>Painel executivo no Streamlit com a mesma linguagem visual do dashboard HTML.</span>
    </div>
  </div>
  <div class="topbar-meta">
    <div class="meta-pill"><span class="status-dot"></span><strong>Atualizado</strong> {current_app_timestamp()}</div>
    <div class="meta-pill"><strong>{competition_name}</strong> em foco</div>
    <div class="meta-pill"><strong>{odds_coverage}%</strong> cobertura de odds</div>
    {"<div class='meta-pill'><strong>Filtro</strong> " + team_filter_display + "</div>" if team_filter.strip() else ""}
  </div>
</section>

<section class="dashboard-hero">
  <div class="hero-grid">
    <div>
      <div class="hero-tag">Painel analitico no Streamlit</div>
      <h1>Mercados mais claros para decidir melhor</h1>
    </div>
  </div>
</section>
""",
    unsafe_allow_html=True,
)

selected_chat_context = ""
selected_chat_label = ""
selected_chat_match_key = ""

tab1, tab2, tab3, tab4 = st.tabs(["Jogos Seguros", "Analise de Jogo", "Todos os Futuros", "Resultados"])

with tab1:
    st.markdown('<div id="anchor-safe"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-card"><div class="section-title">Ranking de Jogos Mais Seguros</div><div class="section-copy">Comece por aqui para ver as selecoes mais conservadoras da competicao ativa, com equilibrio entre probabilidade do modelo, EV, odd e numero de casas.</div></div>',
        unsafe_allow_html=True,
    )
    if relaxed_note:
        st.info(relaxed_note)
    if safe_df.empty:
        st.warning(
            "Nenhum jogo passou no filtro atual."
            + (f" O filtro por time `{team_filter}` tambem esta sendo aplicado." if team_filter.strip() else " Reduza os filtros no menu lateral.")
        )
        fallback_df = build_safe_bets_table(
            matches_df=df,
            bankroll=1000.0,
            kelly_fractional=0.25,
            min_model_prob=0.35,
            min_expected_value=-0.05,
            max_odd=4.00,
            min_bookmakers=1,
        )
        if team_filter.strip():
            fallback_df = filter_matches_by_team(fallback_df, team_filter)
        if not fallback_df.empty:
            fallback_df = fallback_df.copy()
            fallback_df["market_label"] = fallback_df.apply(
                lambda r: market_label(str(r["market"]), str(r["home_team"]), str(r["away_team"])),
                axis=1,
            )
            alt = fallback_df[
                ["date_text", "home_team", "away_team", "market_label", "odd", "model_probability", "expected_value", "bookmakers"]
            ].copy()
            alt.columns = ["Data", "Mandante", "Visitante", "Resultado sugerido", "Odd", "Prob. Modelo", "EV", "Casas"]
            alt["Prob. Modelo"] = (alt["Prob. Modelo"] * 100).round(2).astype(str) + "%"
            alt["EV"] = (alt["EV"] * 100).round(2).astype(str) + "%"
            st.markdown("#### Sugestoes alternativas (filtro amplo)")
            st.dataframe(alt.head(12), use_container_width=True, hide_index=True)
    else:
        safe_df = safe_df.copy()
        safe_df["market_label"] = safe_df.apply(
            lambda r: market_label(str(r["market"]), str(r["home_team"]), str(r["away_team"])),
            axis=1,
        )
        show_safe = safe_df[
            [
                "date_text",
                "home_team",
                "away_team",
                "market_label",
                "odd",
                "model_probability",
                "expected_value",
                "bookmakers",
                "safety_score",
                "match_url",
            ]
        ].copy()
        show_safe.columns = [
            "Data",
            "Mandante",
            "Visitante",
            "Resultado sugerido",
            "Odd",
            "Prob. Modelo",
            "EV",
            "Casas",
            "Score",
            "Link",
        ]
        show_safe["Prob. Modelo"] = (show_safe["Prob. Modelo"] * 100).round(2).astype(str) + "%"
        show_safe["EV"] = (show_safe["EV"] * 100).round(2).astype(str) + "%"
        show_safe["Score"] = (show_safe["Score"] * 100).round(1)
        st.markdown("#### Resumo rapido: jogos para apostar agora")
        resume = show_safe[["Data", "Mandante", "Visitante", "Resultado sugerido", "Odd", "EV"]].head(8).copy()
        resume["Jogo"] = resume["Mandante"] + " x " + resume["Visitante"]
        resume = resume[["Data", "Jogo", "Resultado sugerido", "Odd", "EV"]]
        st.dataframe(resume, use_container_width=True, hide_index=True)

        best_safe_row = safe_df.iloc[0]
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
                        "match_label": f"{best_safe_row['date_text']} | {best_safe_row['home_team']} x {best_safe_row['away_team']}",
                        "analysis": build_ai_analysis_for_fixture(df, fixture_row),
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
        st.dataframe(show_safe.head(20), use_container_width=True, hide_index=True)
        st.markdown("<div class='small-note'>Score combina probabilidade, EV, odd e numero de casas.</div>", unsafe_allow_html=True)

with tab2:
    st.markdown('<div id="anchor-simulator"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-card"><div class="section-title">Simulador de Estrategia por Jogo</div><div class="section-copy">Escolha um confronto para comparar probabilidades, odds, mercado de valor e a nova analise detalhada com IA no mesmo fluxo.</div></div>',
        unsafe_allow_html=True,
    )
    if valid.empty:
        st.info("Nao ha jogos futuros com odds 1X2 no momento.")
    else:
        valid = valid.reset_index(drop=True)
        valid["match_option_id"] = valid.index.astype(str)
        valid["match_label"] = valid.apply(
            lambda r: f"{r['date_text']} | {r['home_team']} x {r['away_team']}",
            axis=1,
        )
        match_options = valid["match_option_id"].tolist()
        labels_by_option = dict(zip(valid["match_option_id"], valid["match_label"]))

        selected_option = st.selectbox(
            "Escolha o jogo",
            options=match_options,
            format_func=lambda option_id: labels_by_option.get(option_id, option_id),
            key=f"match_selector_label_{competition}",
        )
        selected = valid.loc[valid["match_option_id"] == selected_option].iloc[0]
        selected_match_label = str(selected["match_label"])

        st.caption(f"Jogo selecionado: {selected_match_label}")
        st.caption(f"Dados usados na analise: {selected['home_team']} x {selected['away_team']}")

        probs = calculate_match_probabilities(df, selected["home_team"], selected["away_team"])
        tip = suggest_bet_strategy(
            probs,
            odd_home=float(selected["odds_home"]),
            odd_draw=float(selected["odds_draw"]),
            odd_away=float(selected["odds_away"]),
            bankroll=1000.0,
            kelly_fractional=0.25,
        )
        readable_market = market_label(str(tip.best_market), str(selected["home_team"]), str(selected["away_team"]))
        probable_market, probable_probability = max(
            [
                (f"Vitoria {selected['home_team']}", probs.home_win),
                ("Empate", probs.draw),
                (f"Vitoria {selected['away_team']}", probs.away_win),
            ],
            key=lambda item: item[1],
        )

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Vitoria {selected['home_team']}", f"{probs.home_win * 100:.1f}%")
        c2.metric("Empate", f"{probs.draw * 100:.1f}%")
        c3.metric(f"Vitoria {selected['away_team']}", f"{probs.away_win * 100:.1f}%")

        o1, o2, o3 = st.columns(3)
        o1.metric("Odd Casa", f"{selected['odds_home']:.2f}")
        o2.metric("Odd Empate", f"{selected['odds_draw']:.2f}")
        o3.metric("Odd Fora", f"{selected['odds_away']:.2f}")

        st.info(
            f"Resultado mais provavel: {probable_market} "
            f"({probable_probability * 100:.2f}% de chance pelo modelo)."
        )

        if tip.expected_value > 0:
            st.success(
                f"Melhor aposta por valor: {readable_market} | EV: {tip.expected_value * 100:.2f}%"
            )
        else:
            st.warning(
                f"Sem EV positivo no 1X2. Melhor aposta por valor no momento: "
                f"{readable_market} (EV {tip.expected_value * 100:.2f}%)."
            )

        st.write(
            f"Prob. modelo da aposta por valor ({readable_market}): **{tip.model_probability * 100:.2f}%** | "
            f"Prob. implicita da odd: **{tip.implied_probability * 100:.2f}%**"
        )

        ai_state_key = f"ai_analysis_cache_{competition}_{selected_option}"
        st.markdown(
            f"""
<div class="ai-action-panel">
  <strong>Uso da IA no jogo selecionado</strong>
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
                    st.session_state[ai_state_key] = build_ai_analysis_for_fixture(df, selected)
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

        value_bet_line = (
            f"Vitoria {selected['home_team']}"
            if tip.best_market == "Casa"
            else f"Vitoria {selected['away_team']}"
            if tip.best_market == "Fora"
            else "Empate"
        )

        reasons = []
        if isinstance(home_ctx.get("rank"), int) and isinstance(away_ctx.get("rank"), int):
            if home_ctx["rank"] < away_ctx["rank"]:
                reasons.append(
                    f"Classificacao: {selected['home_team']} esta melhor posicionado "
                    f"({home_ctx['rank']}º vs {away_ctx['rank']}º)."
                )
            elif away_ctx["rank"] < home_ctx["rank"]:
                reasons.append(
                    f"Classificacao: {selected['away_team']} esta melhor posicionado "
                    f"({away_ctx['rank']}º vs {home_ctx['rank']}º)."
                )

        if home_ctx.get("recent_points", 0) > away_ctx.get("recent_points", 0):
            reasons.append(
                f"Momento recente: {selected['home_team']} somou {home_ctx['recent_points']} pontos "
                f"nos ultimos jogos ({home_ctx['recent_text']})."
            )
        elif away_ctx.get("recent_points", 0) > home_ctx.get("recent_points", 0):
            reasons.append(
                f"Momento recente: {selected['away_team']} somou {away_ctx['recent_points']} pontos "
                f"nos ultimos jogos ({away_ctx['recent_text']})."
            )

        btts_line = (
            "Ambos marcam (SIM) aparece interessante."
            if probs.btts_yes >= 0.55
            else "Ambos marcam (NAO) aparece mais conservador."
        )
        goal_line = (
            "Tendencia de Menos de 2.5 gols."
            if probs.under_25 >= probs.over_25
            else "Tendencia de Mais de 2.5 gols."
        )

        st.markdown(
            f"""
**Jogo:** {selected['home_team']} x {selected['away_team']} ({selected['date_text']})  
**Odds atuais (1X2):** Casa `{selected['odds_home']:.2f}` | Empate `{selected['odds_draw']:.2f}` | Fora `{selected['odds_away']:.2f}`  
**Probabilidades do modelo:** Casa `{probs.home_win*100:.1f}%` | Empate `{probs.draw*100:.1f}%` | Fora `{probs.away_win*100:.1f}%`  
**Resultado mais provavel:** **{probable_market}** (`{probable_probability*100:.1f}%`)  
**Melhor aposta por valor:** **{value_bet_line}** (EV `{tip.expected_value*100:.2f}%`)  
**Mercados extras:** {btts_line} Prob. BTTS `{probs.btts_yes*100:.1f}%`. {goal_line} Under 2.5 `{probs.under_25*100:.1f}%` / Over 2.5 `{probs.over_25*100:.1f}%`.
"""
        )

        if reasons:
            st.markdown("**Principais motivos:**")
            for r in reasons[:3]:
                st.markdown(f"- {r}")

with tab3:
    st.markdown('<div id="anchor-agenda"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-card"><div class="section-title">Agenda Completa de Jogos Futuros</div><div class="section-copy">Use esta grade para explorar todo o mercado futuro da competicao selecionada e comparar rapidamente mandante, visitante, casas e faixa de odds.</div></div>',
        unsafe_allow_html=True,
    )
    view_fixtures = fixtures[
        [
            "date_text",
            "home_team",
            "away_team",
            "bookmakers",
            "odds_home",
            "odds_draw",
            "odds_away",
            "match_url",
        ]
    ].copy()
    view_fixtures.columns = [
        "Data",
        "Mandante",
        "Visitante",
        "Casas",
        "Odd Casa",
        "Odd Empate",
        "Odd Fora",
        "Link",
    ]
    st.dataframe(view_fixtures.reset_index(drop=True), use_container_width=True, hide_index=True)

with tab4:
    st.markdown('<div id="anchor-results"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-card"><div class="section-title">Resultados Recentes</div><div class="section-copy">Aqui ficam os jogos ja finalizados da competicao ativa, com placar fechado e horario/data ajustados para a leitura no fuso de Sao Paulo.</div></div>',
        unsafe_allow_html=True,
    )
    if finished.empty:
        st.info("Nao ha jogos finalizados disponiveis para esta competicao.")
    else:
        view_finished = finished[
            [
                "date_text",
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
render_floating_dock(competition_name=competition_name, risk_profile=risk_profile)
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
