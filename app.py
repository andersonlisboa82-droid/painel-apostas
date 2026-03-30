from __future__ import annotations

import json
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

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


@st.cache_data(ttl=900, show_spinner=False)
def get_data(competition: str) -> pd.DataFrame:
    return load_competition_matches(competition)


def market_label(market: str, home_team: str, away_team: str) -> str:
    if market == "Casa":
        return f"Vitoria {home_team}"
    if market == "Fora":
        return f"Vitoria {away_team}"
    return "Empate"


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
    api_key = (os.getenv("NVIDIA_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("Defina a variavel de ambiente NVIDIA_API_KEY para usar a analise com IA.")

    prompt = f"""
Analise o jogo em portugues do Brasil, de forma curta, clara e objetiva.
Retorne apenas JSON valido.
Nao mostre raciocinio interno, planejamento, notas, markdown ou texto fora do JSON.

Jogo: {home_team} x {away_team}
Data/Horario: {date_text} (horario de Sao Paulo)

Odds 1X2:
- Casa: {odds_home:.2f}
- Empate: {odds_draw:.2f}
- Fora: {odds_away:.2f}

Probabilidades do modelo:
- Vitoria {home_team}: {home_win_prob * 100:.2f}%
- Empate: {draw_prob * 100:.2f}%
- Vitoria {away_team}: {away_win_prob * 100:.2f}%

Resultado mais provavel:
- {probable_market} ({probable_probability * 100:.2f}%)

Melhor aposta por valor:
- {value_market}
- Probabilidade do modelo: {value_probability * 100:.2f}%
- Probabilidade implicita da odd: {value_implied_probability * 100:.2f}%
- EV: {expected_value * 100:.2f}%

Explique:
1. Qual e o resultado mais provavel.
2. Por que a melhor aposta por valor pode ser diferente.
3. Um cuidado importante antes de apostar.

Responda exatamente neste formato JSON:
{{
  "resultado_mais_provavel": "texto curto",
  "melhor_aposta_por_valor": "texto curto",
  "cuidado_antes_de_apostar": "texto curto"
}}

Cada valor deve ter 1 ou 2 frases curtas.
Use tom simples, natural e direto.
Evite repetir os nomes dos campos, os mesmos numeros e a mesma explicacao em mais de um bloco.
Evite frases vagas como "as probabilidades podem variar" ou "a analise depende dos dados disponiveis".
No bloco de cuidado, diga um risco concreto da aposta, como zebra, odd esticada, variancia ou necessidade de gestao de banca.
No bloco de valor, explique a diferenca entre probabilidade do modelo e odd de forma humana, sem jargao excessivo.
Nao invente dados.
"""

    payload = {
        "model": NVIDIA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Voce e um analista de apostas esportivas. "
                    "Responda em portugues claro, curto, natural e honesto. "
                    "Entregue somente JSON valido. "
                    "Nunca exponha raciocinio interno. "
                    "Evite repeticao e frases genericas."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "top_p": 0.8,
        "max_tokens": 220,
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


st.markdown(
    """
<style>
:root {
  --bg: #f4f7fb;
  --card: #ffffff;
  --line: #dbe4ef;
  --text: #1a2433;
  --muted: #5a6b82;
  --accent: #0f766e;
  --accent-2: #0b5ed7;
}
.stApp {
  background:
    radial-gradient(1200px 500px at -10% -10%, #dbeafe 0%, transparent 60%),
    radial-gradient(900px 400px at 100% 0%, #dcfce7 0%, transparent 60%),
    var(--bg);
}
.block-container {
  padding-top: 1.2rem;
  max-width: 1240px;
}
.dashboard-hero {
  background: linear-gradient(130deg, #0f172a 0%, #0b2239 55%, #0f766e 100%);
  border: 1px solid #0f2a46;
  border-radius: 18px;
  padding: 1.1rem 1.2rem;
  margin-bottom: 1rem;
  color: #f8fafc;
  box-shadow: 0 14px 28px rgba(15, 23, 42, 0.2);
}
.dashboard-hero h1 {
  font-size: 1.45rem;
  margin: 0;
  letter-spacing: 0.2px;
}
.dashboard-hero p {
  margin: .35rem 0 0;
  color: #dbe8f6;
  font-size: .93rem;
}
.panel-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: .8rem .95rem;
  box-shadow: 0 8px 24px rgba(15, 23, 42, .06);
}
.section-title {
  margin: 0.2rem 0 .7rem;
  font-size: 1rem;
  color: var(--text);
  font-weight: 700;
}
.stDataFrame, div[data-testid="stDataFrame"] {
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid var(--line);
}
.small-note {
  color: var(--muted);
  font-size: .86rem;
  margin-top: .2rem;
}
hr {
  border: none;
  border-top: 1px solid var(--line);
  margin: 1rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="dashboard-hero">
  <h1>Painel Inteligente de Apostas</h1>
  <p>Atualizado em {current_app_timestamp()} (horario de Sao Paulo)</p>
</div>
""",
    unsafe_allow_html=True,
)

top_a, top_b = st.columns([1, 1])
with top_a:
    st.info("Filtros: use o menu lateral em 'Configuracoes'.")
with top_b:
    if st.button("Atualizar dados agora", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with st.sidebar:
    st.markdown("### Configuracoes")
    competition = st.selectbox("Competicao", options=list(COMPETITIONS.keys()))
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
    st.caption("IA NVIDIA: defina a variavel de ambiente `NVIDIA_API_KEY` para habilitar a analise em linguagem natural.")

try:
    df = get_data(competition)
except Exception as exc:
    st.error(f"Falha ao buscar dados da internet: {exc}")
    st.stop()

finished = df[df["status"] == "Finalizado"].copy()
fixtures = df[df["status"] == "Agendado"].copy()
valid = fixtures.dropna(subset=["odds_home", "odds_draw", "odds_away"]).copy()

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
        if not safe_df.empty:
            relaxed_note = (
                f"Filtro do perfil foi relaxado automaticamente para exibir opcoes: "
                f"Prob >= {rp:.2f} | EV >= {rev:.2f} | Odd <= {rodd:.2f} | Casas >= {rbooks}."
            )
            break

tab1, tab2, tab3 = st.tabs(["Jogos Seguros", "Analise de Jogo", "Todos os Futuros"])

with tab1:
    st.markdown('<div class="section-title">Ranking de Jogos Mais Seguros</div>', unsafe_allow_html=True)
    if relaxed_note:
        st.info(relaxed_note)
    if safe_df.empty:
        st.warning("Nenhum jogo passou no filtro atual. Reduza os filtros no menu lateral.")
        fallback_df = build_safe_bets_table(
            matches_df=df,
            bankroll=1000.0,
            kelly_fractional=0.25,
            min_model_prob=0.35,
            min_expected_value=-0.05,
            max_odd=4.00,
            min_bookmakers=1,
        )
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
        st.markdown("#### Ranking detalhado")
        st.dataframe(show_safe.head(20), use_container_width=True, hide_index=True)
        st.markdown("<div class='small-note'>Score combina probabilidade, EV, odd e numero de casas.</div>", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-title">Simulador de Estrategia por Jogo</div>', unsafe_allow_html=True)
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

        if st.button("Analisar com IA NVIDIA", key=f"nvidia_analysis_{competition}_{selected_option}"):
            try:
                with st.spinner("Consultando IA da NVIDIA..."):
                    ai_analysis = generate_nvidia_match_analysis(
                        home_team=str(selected["home_team"]),
                        away_team=str(selected["away_team"]),
                        date_text=str(selected["date_text"]),
                        odds_home=float(selected["odds_home"]),
                        odds_draw=float(selected["odds_draw"]),
                        odds_away=float(selected["odds_away"]),
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
                parsed = json.loads(ai_analysis)
                resultado_text = str(parsed.get("resultado_mais_provavel", "")).strip()
                valor_text = str(parsed.get("melhor_aposta_por_valor", "")).strip()
                cuidado_text = str(parsed.get("cuidado_antes_de_apostar", "")).strip()
                if not (resultado_text and valor_text and cuidado_text):
                    resultado_text, valor_text, cuidado_text = extract_ai_analysis_blocks(ai_analysis)
                st.markdown("### Analise com IA NVIDIA")
                st.markdown(render_ai_analysis_blocks(resultado_text, valor_text, cuidado_text))
            except Exception as exc:
                st.error(f"Nao foi possivel gerar a analise com IA da NVIDIA: {exc}")

        top_df = pd.DataFrame(probs.top_scorelines, columns=["Placar", "Probabilidade"])
        top_df["Probabilidade"] = (top_df["Probabilidade"] * 100).round(2).astype(str) + "%"
        st.dataframe(top_df, use_container_width=True, hide_index=True)

        if isinstance(selected.get("match_url"), str) and selected["match_url"]:
            st.link_button("Abrir pagina do jogo", selected["match_url"])

        st.markdown("### Analise explicada do jogo")
        home_ctx = get_team_context(df, str(selected["home_team"]))
        away_ctx = get_team_context(df, str(selected["away_team"]))

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
    st.markdown('<div class="section-title">Agenda Completa de Jogos Futuros</div>', unsafe_allow_html=True)
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

st.markdown("---")
with st.expander("Glossario de siglas e estatisticas"):
    st.markdown(
        """
- `1X2`: Mercado principal de resultado final (`1` casa, `X` empate, `2` visitante).
- `Odd`: Cotacao da aposta. Exemplo `2.00` significa retorno bruto de R$2 para cada R$1 apostado.
- `Prob. Modelo`: Probabilidade calculada pelo modelo estatistico para aquele resultado.
- `EV` (Valor Esperado): Vantagem matematica da aposta. `EV > 0` indica expectativa positiva no longo prazo.
- `Casas` (`B's`): Quantidade de casas de aposta consideradas para aquela linha de odds.
"""
    )

st.caption("Aposta envolve risco. O painel ajuda na selecao, mas nao elimina perdas.")
