from __future__ import annotations

from datetime import datetime
from html import escape
import json
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from analytics import (
    build_safe_bets_table,
    calculate_match_probabilities,
    suggest_bet_strategy,
    get_team_context,
)
from real_match_stats import build_match_stats_cache_key, load_real_match_stats_cache
from scraper import COMPETITIONS, load_competition_matches


APP_TIMEZONE = ZoneInfo("America/Sao_Paulo")
PORTAL_RELEASE_LABEL = "11/04/2026 | v2-modern-canvas"

AI_PROMPT_TEMPLATE = """Atue como um analista quantitativo profissional de futebol especializado em modelagem estatistica, leitura de mercado e identificacao de value bets (+EV), com abordagem semelhante a analistas de apostas institucionais.

Seu objetivo e analisar os jogos da data selecionada (__DATA_SELECIONADA__) e identificar as melhores oportunidades de aposta com base em dados historicos recentes, metricas avancadas e movimentacao das odds.

A analise deve priorizar confiabilidade estatistica, consistencia dos padroes e risco controlado.

--------------------------------------------------

ETAPA 1 - FILTRO DOS JOGOS

Selecione apenas os 5 jogos com maior confiabilidade estatistica na data analisada.

Se nao houver 5 jogos com evidencia estatistica suficiente, selecione menos jogos.

Nunca force previsoes.

--------------------------------------------------

ETAPA 2 - MODELAGEM ESTATISTICA

Para cada jogo, calcule probabilidades reais estimadas:

Vitoria mandante
Empate
Vitoria visitante
Over 2.5 gols
Under 2.5 gols
BTTS (ambos marcam)
Possivel placar provavel

Compare com odds implicitas do mercado quando disponiveis.

Identifique possiveis value bets (+EV).

Se nao houver value bet, informe claramente.

--------------------------------------------------

ETAPA 3 - FORMA RECENTE (ultimos 5 a 10 jogos)

Analise:

sequencia de vitorias/derrotas
media de gols marcados
media de gols sofridos
frequencia de Over 2.5
frequencia de BTTS
consistencia defensiva
variabilidade de desempenho

--------------------------------------------------

ETAPA 4 - CONFRONTOS DIRETOS (H2H)

Identifique:

tendencia de gols
dominio recorrente
frequencia historica de BTTS
frequencia Over/Under
padroes repetitivos relevantes

Priorize confrontos recentes.

--------------------------------------------------

ETAPA 5 - PERFORMANCE CASA vs FORA

Compare:

taxa de vitoria
media de gols
xG ofensivo
xGA defensivo
estabilidade tatica
diferenca de desempenho mandante vs visitante

--------------------------------------------------

ETAPA 6 - METRICAS AVANCADAS

Utilize:

xG
xGA
finalizacoes por jogo
conversao ofensiva
escanteios medios
cartoes medios
gols no segundo tempo
times que marcam primeiro
times que sofrem viradas

Identifique padroes ocultos relevantes.

--------------------------------------------------

ETAPA 7 - ESCALACOES E ELENCO

Considere:

lesoes
suspensoes
rotacao
retorno de titulares
dependencia de artilheiros
impacto tatico das ausencias

--------------------------------------------------

ETAPA 8 - MOVIMENTACAO DAS ODDS

Compare:

odd de abertura
odd atual
direcao do mercado
possivel entrada de dinheiro sharp
possivel influencia do publico

Identifique distorcoes relevantes.

--------------------------------------------------

ETAPA 9 - CONTEXTO COMPETITIVO

Avalie:

motivacao na tabela
briga por titulo
zona de rebaixamento
mata-mata vs rodada regular
necessidade de resultado
gestao de elenco

--------------------------------------------------

ETAPA 10 - FATORES EXTERNOS

Considere:

clima
condicoes do campo
viagem recente
fadiga
sequencia de jogos
arbitro (cartoes e penaltis)

--------------------------------------------------

ETAPA 11 - SCORE DE RISCO

Calcule score de risco de 0 a 100:

Forma recente -> 25%
Casa/Fora -> 15%
H2H -> 10%
xG e metricas avancadas -> 20%
Movimento das odds -> 20%
Volatilidade recente -> 10%

Classificacao:

0-39 = RISCO BAIXO
40-69 = RISCO MEDIO
70-100 = RISCO ALTO

--------------------------------------------------

ETAPA 12 - CONFIDENCE SCORE

Calcule indice de confianca de 0 a 10 baseado em:

consistencia estatistica
estabilidade das odds
convergencia dos indicadores
historico recente confiavel

--------------------------------------------------

ETAPA 13 - SAIDA POR JOGO

Para cada jogo apresentar:

JOGO: Time A vs Time B

Probabilidade vitoria mandante:
Probabilidade empate:
Probabilidade vitoria visitante:

Probabilidade Over 2.5:
Probabilidade BTTS:

Escanteios esperados:
Cartoes esperados:

Score de risco:
Classificacao final:

Confidence Score (0-10):

Value bet identificada:
(sim ou nao)

Se sim, explicar qual.

Fornecer justificativa objetiva baseada nos dados.

Apresentar:

1 aposta conservadora
1 aposta equilibrada
1 aposta agressiva

--------------------------------------------------

ETAPA 14 - GERENCIAMENTO DE BANCA

Sugira divisao:

Aposta principal
Aposta de cobertura
Aposta outsider

Utilize logica semelhante ao metodo Kelly simplificado.

Simule expectativa de retorno (%ROI estimado).

--------------------------------------------------

ETAPA 15 - CONCLUSAO FINAL

Ao final da analise:

Liste as 3 melhores apostas do dia com maior valor esperado (+EV)

Informe:

se e melhor apostar agora (early line)
ou esperar closing line

Se nao houver apostas com valor estatistico claro, informe explicitamente."""


def _current_app_timestamp() -> str:
    return datetime.now(APP_TIMEZONE).strftime("%d/%m/%Y %H:%M:%S")


def _fmt_odd(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2f}"


def _fmt_score(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:.1f}"


def _format_match_datetime(date_text: object, event_timestamp: object = None, status: str = "") -> str:
    parsed_ts = pd.to_datetime(event_timestamp, errors="coerce", utc=True)
    if not pd.isna(parsed_ts):
        local_dt = parsed_ts.tz_convert(APP_TIMEZONE)
        if str(status).strip() == "Finalizado":
            return local_dt.strftime("%d/%m/%Y")
        return local_dt.strftime("%d/%m/%Y %H:%M")
    raw = str(date_text or "").strip()
    return raw if raw else "-"


def _match_filter_date(date_text: object, event_timestamp: object = None, status: str = "") -> str:
    parsed_ts = pd.to_datetime(event_timestamp, errors="coerce", utc=True)
    if not pd.isna(parsed_ts):
        return parsed_ts.tz_convert(APP_TIMEZONE).strftime("%Y-%m-%d")

    display_date = _format_match_datetime(date_text, event_timestamp, status)
    match = re.search(r"(\d{2})/(\d{2})/(\d{4})", display_date)
    if match:
        return f"{match.group(3)}-{match.group(2)}-{match.group(1)}"
    return ""


def _market_label(market: str, home_team: str, away_team: str) -> str:
    if market == "Casa":
        return f"Vitoria {home_team}"
    if market == "Fora":
        return f"Vitoria {away_team}"
    return "Empate"


def _actual_market_label(home_goals: float | None, away_goals: float | None, home_team: str, away_team: str) -> str:
    if home_goals is None or away_goals is None or pd.isna(home_goals) or pd.isna(away_goals):
        return "-"
    if float(home_goals) > float(away_goals):
        return f"Vitoria {home_team}"
    if float(home_goals) < float(away_goals):
        return f"Vitoria {away_team}"
    return "Empate"


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _sort_matches_for_display(frame: pd.DataFrame, *, ascending: bool) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    if "event_timestamp" in out.columns:
        out["_event_dt"] = pd.to_datetime(out["event_timestamp"], errors="coerce", utc=True)
        if out["_event_dt"].notna().any():
            out = out.sort_values(by=["_event_dt", "home_team", "away_team"], ascending=ascending)
            return out.drop(columns="_event_dt")
    return out.reset_index(drop=True)


def _best_model_result(probs, home_team: str, away_team: str) -> tuple[str, float]:
    options = [
        (f"Vitoria {home_team}", float(probs.home_win)),
        ("Empate", float(probs.draw)),
        (f"Vitoria {away_team}", float(probs.away_win)),
    ]
    return max(options, key=lambda item: item[1])


def _risk_stage_from_metrics(probability: float, expected_value: float, odd: float, bookmakers: int) -> str:
    if probability >= 0.62 and expected_value >= 0.03 and 1.0 < odd <= 1.95 and bookmakers >= 10:
        return "Baixo risco"
    if probability >= 0.55 and expected_value >= 0.02 and 1.0 < odd <= 2.20 and bookmakers >= 8:
        return "Medio risco"
    if probability >= 0.48 and expected_value >= 0.01 and 1.0 < odd <= 2.90 and bookmakers >= 5:
        return "Alto risco"
    return "Fora dos criterios"


def _risk_stage_context(stage: str, suggested_label: str, has_valid_odd: bool = True) -> str:
    if stage == "Baixo risco":
        return f"{suggested_label} entra no perfil mais conservador, com odd mais controlada e boa sustentacao estatistica."
    if stage == "Medio risco":
        return f"{suggested_label} fica no perfil equilibrado, combinando boa confianca do modelo com retorno potencial moderado."
    if stage == "Alto risco":
        return f"{suggested_label} entra no perfil agressivo, aceitando maior variacao para buscar edge mais alto."
    if not has_valid_odd:
        return "Sem odd valida para enquadrar o jogo na regua principal de risco do portal."
    return f"{suggested_label} existe na leitura, mas ainda fica fora da regua principal de risco do portal."


def _classify_model_risk(row, probs, home_team: str, away_team: str) -> tuple[str, str]:
    market_rows = [
        ("Casa", f"Vitoria {home_team}", float(probs.home_win), getattr(row, "odds_home", None)),
        ("Empate", "Empate", float(probs.draw), getattr(row, "odds_draw", None)),
        ("Fora", f"Vitoria {away_team}", float(probs.away_win), getattr(row, "odds_away", None)),
    ]
    best_market, best_label, best_prob, best_odd = max(market_rows, key=lambda item: item[2])

    odd_value = float(best_odd) if best_odd is not None and not pd.isna(best_odd) and float(best_odd) > 1.0 else 0.0
    bookmakers = int(getattr(row, "bookmakers", 0)) if getattr(row, "bookmakers", None) is not None and not pd.isna(getattr(row, "bookmakers", None)) else 0
    expected_value = (best_prob * odd_value) - 1.0 if odd_value > 1.0 else -1.0

    stage = _risk_stage_from_metrics(best_prob, expected_value, odd_value, bookmakers)
    return stage, _risk_stage_context(stage, best_label, has_valid_odd=odd_value > 1.0)


def _get_detail_json(
    df: pd.DataFrame,
    row,
    probs,
    tip=None,
    cached_real_stats_payload: dict[str, object] | None = None,
) -> str:
    home_ctx = get_team_context(df, str(row.home_team))
    away_ctx = get_team_context(df, str(row.away_team))
    status = str(getattr(row, "status", ""))
    display_date = _format_match_datetime(
        getattr(row, "date_text", ""),
        getattr(row, "event_timestamp", None),
        status,
    )
    final_score = ""
    actual_result = ""
    model_result, model_probability = _best_model_result(probs, str(row.home_team), str(row.away_team))
    model_risk_stage, model_risk_context = _classify_model_risk(row, probs, str(row.home_team), str(row.away_team))
    model_hit = ""
    if status == "Finalizado":
        final_score = f"{_fmt_score(getattr(row, 'home_goals', None))} x {_fmt_score(getattr(row, 'away_goals', None))}"
        actual_result = _actual_market_label(
            getattr(row, "home_goals", None),
            getattr(row, "away_goals", None),
            str(row.home_team),
            str(row.away_team),
        )
        if actual_result != "-":
            model_hit = "Acertou" if actual_result == model_result else "Errou"

    data = {
        "home": str(row.home_team),
        "away": str(row.away_team),
        "date": display_date,
        "status": status,
        "competition": str(getattr(row, "competition", "")),
        "date_text_raw": str(getattr(row, "date_text", "")),
        "event_timestamp": str(getattr(row, "event_timestamp", "") or ""),
        "match_url": str(getattr(row, "match_url", "") or ""),
        "final_score": final_score,
        "actual_result": actual_result,
        "model_result": model_result,
        "model_probability": round(model_probability * 100, 1),
        "model_risk_stage": model_risk_stage,
        "model_risk_context": model_risk_context,
        "model_hit": model_hit,
        "probs": {
            "home": round(probs.home_win * 100, 1),
            "draw": round(probs.draw * 100, 1),
            "away": round(probs.away_win * 100, 1),
            "btts": round(probs.btts_yes * 100, 1),
            "over25": round(probs.over_25 * 100, 1),
            "under25": round(probs.under_25 * 100, 1),
            "scorelines": [[s, round(p * 100, 1)] for s, p in probs.top_scorelines],
        },
        "odds": {
            "home": float(row.odds_home) if hasattr(row, "odds_home") else float(row.odd),
            "draw": float(row.odds_draw) if hasattr(row, "odds_draw") else 0.0,
            "away": float(row.odds_away) if hasattr(row, "odds_away") else 0.0,
        },
        "context": {"home": home_ctx, "away": away_ctx},
    }
    if cached_real_stats_payload and cached_real_stats_payload.get("available"):
        data["prefetched_real_stats"] = cached_real_stats_payload
    if tip:
        data["tip"] = {
            "market": _market_label(tip.best_market, str(row.home_team), str(row.away_team)),
            "odd": round(tip.best_odd, 2),
            "prob": round(tip.model_probability * 100, 1),
            "ev": round(tip.expected_value * 100, 2),
            "stake": round(tip.suggested_stake, 2),
        }
    return json.dumps(data, ensure_ascii=False)


def _build_competition_section(
    name: str,
    df: pd.DataFrame,
    real_stats_cache: dict[str, dict[str, object]] | None = None,
) -> tuple[str, dict[str, int | str], dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    finished = df[df["status"] == "Finalizado"].copy()
    fixtures = df[df["status"] == "Agendado"].copy()
    finished = _sort_matches_for_display(finished, ascending=False)
    fixtures = _sort_matches_for_display(fixtures, ascending=True)
    fixtures_valid = fixtures.dropna(subset=["odds_home", "odds_draw", "odds_away"])
    comp_id = f"comp-{_slugify(name)}"

    rows_html = []
    rec_rows = []
    finished_rows = []
    analysis_options = []
    detail_store: dict[str, object] = {}
    risk_entries: list[dict[str, object]] = []
    match_catalog: list[dict[str, object]] = []
    detail_index = 0
    finished_hits = 0
    finished_evaluated = 0

    def register_detail(detail_json: str) -> str:
        nonlocal detail_index
        try:
            detail_data = json.loads(detail_json)
        except json.JSONDecodeError:
            return ""
        if not detail_data:
            return ""
        detail_index += 1
        detail_key = f"{comp_id}-detail-{detail_index}"
        detail_store[detail_key] = detail_data
        return detail_key

    for row in fixtures.itertuples(index=False):
        display_date = _format_match_datetime(row.date_text, getattr(row, "event_timestamp", None), str(row.status))
        filter_date = _match_filter_date(row.date_text, getattr(row, "event_timestamp", None), str(row.status))
        detail_json = "{}"
        try:
            probs = calculate_match_probabilities(df, row.home_team, row.away_team)
            detail_json = _get_detail_json(df, row, probs)
        except Exception:
            pass
        detail_key = register_detail(detail_json)
        if detail_key:
            detail_data = detail_store.get(detail_key, {})
            match_catalog.append(
                {
                    "competition": name,
                    "status": str(getattr(row, "status", "")),
                    "filter_date": filter_date,
                    "date_label": display_date,
                    "matchup": f"{row.home_team} x {row.away_team}",
                    "home": str(row.home_team),
                    "away": str(row.away_team),
                    "detail_key": detail_key,
                    "model_result": str(detail_data.get("model_result", "-")),
                    "model_probability": float(detail_data.get("model_probability", 0.0) or 0.0),
                    "model_risk_stage": str(detail_data.get("model_risk_stage", "Fora dos criterios")),
                }
            )

        rows_html.append(
            "<tr "
            "data-filter-scope=\"general\" "
            f"data-date=\"{escape(filter_date)}\" "
            f"data-odd-home=\"{_fmt_odd(row.odds_home)}\" "
            f"data-odd-draw=\"{_fmt_odd(row.odds_draw)}\" "
            f"data-odd-away=\"{_fmt_odd(row.odds_away)}\" "
            ">"
            f"<td>{escape(display_date)}</td>"
            f"<td>{escape(str(row.home_team))}</td>"
            f"<td>{escape(str(row.away_team))}</td>"
            f"<td>{escape(_fmt_odd(row.odds_home))}</td>"
            f"<td>{escape(_fmt_odd(row.odds_draw))}</td>"
            f"<td>{escape(_fmt_odd(row.odds_away))}</td>"
            f"<td>{escape(str(row.bookmakers) if row.bookmakers is not None else '-')}</td>"
            f"<td><button class='btn-mini' data-detail-key='{escape(detail_key)}' onclick='showMatchDetails(this)'>Visualizar</button></td>"
            "</tr>"
        )

    for row in fixtures_valid.head(20).itertuples(index=False):
        display_date = _format_match_datetime(row.date_text, getattr(row, "event_timestamp", None), str(row.status))
        try:
            probs = calculate_match_probabilities(df, row.home_team, row.away_team)
            detail_json = _get_detail_json(df, row, probs)
        except Exception:
            detail_json = "{}"
        detail_key = register_detail(detail_json)
        analysis_options.append(
            f"<option data-detail-key=\"{escape(detail_key)}\">[Agendado] {escape(display_date)} | {escape(str(row.home_team))} x {escape(str(row.away_team))}</option>"
        )

    for row in fixtures_valid.head(20).itertuples(index=False):
        display_date = _format_match_datetime(row.date_text, getattr(row, "event_timestamp", None), str(row.status))
        filter_date = _match_filter_date(row.date_text, getattr(row, "event_timestamp", None), str(row.status))
        try:
            probs = calculate_match_probabilities(df, row.home_team, row.away_team)
            tip = suggest_bet_strategy(
                probs,
                odd_home=float(row.odds_home),
                odd_draw=float(row.odds_draw),
                odd_away=float(row.odds_away),
                bankroll=1000.0,
                kelly_fractional=0.25,
            )
            detail_json = _get_detail_json(df, row, probs, tip)
            detail_key = register_detail(detail_json)
            bookmakers = int(row.bookmakers) if row.bookmakers is not None else 0
            risk_stage = _risk_stage_from_metrics(float(tip.model_probability), float(tip.expected_value), float(tip.best_odd), bookmakers)
            if risk_stage != "Fora dos criterios":
                risk_entries.append(
                    {
                        "stage": risk_stage,
                        "competition": name,
                        "filter_date": filter_date,
                        "date_text": display_date,
                        "matchup": f"{row.home_team} x {row.away_team}",
                        "suggestion": _market_label(tip.best_market, str(row.home_team), str(row.away_team)),
                        "odd": float(tip.best_odd),
                        "probability": float(tip.model_probability),
                        "ev": float(tip.expected_value),
                        "bookmakers": bookmakers,
                        "detail_key": detail_key,
                    }
                )
            rec_rows.append(
                f"<tr data-filter-scope=\"model\" data-date=\"{escape(filter_date)}\" data-odd=\"{tip.best_odd:.2f}\" data-prob=\"{tip.model_probability:.4f}\" data-ev=\"{tip.expected_value:.4f}\" data-books=\"{bookmakers}\">"
                f"<td>{escape(display_date)}</td>"
                f"<td>{escape(str(row.home_team))} x {escape(str(row.away_team))}</td>"
                f"<td>{escape(_market_label(tip.best_market, str(row.home_team), str(row.away_team)))}</td>"
                f"<td>{tip.best_odd:.2f}</td>"
                f"<td>{tip.model_probability * 100:.1f}%</td>"
                f"<td>{tip.expected_value * 100:.2f}%</td>"
                f"<td>R$ {tip.suggested_stake:.2f}</td>"
                f"<td><button class='btn-mini' data-detail-key='{escape(detail_key)}' onclick='showMatchDetails(this)'>Visualizar</button></td>"
                "</tr>"
            )
        except Exception:
            continue

    for row in finished.head(20).itertuples(index=False):
        display_date = _format_match_datetime(row.date_text, getattr(row, "event_timestamp", None), str(row.status))
        filter_date = _match_filter_date(row.date_text, getattr(row, "event_timestamp", None), str(row.status))
        detail_json = "{}"
        model_result = "-"
        model_probability = "-"
        model_hit_label = "-"
        cached_real_stats_payload = None
        if real_stats_cache is not None:
            cache_key = build_match_stats_cache_key(
                str(row.home_team),
                str(row.away_team),
                str(getattr(row, "date_text", "")),
                getattr(row, "event_timestamp", None),
            )
            cached_real_stats_payload = real_stats_cache.get(cache_key)
        try:
            probs = calculate_match_probabilities(df, row.home_team, row.away_team)
            model_result, best_prob = _best_model_result(probs, str(row.home_team), str(row.away_team))
            model_probability = f"{best_prob * 100:.1f}%"
            actual_result = _actual_market_label(
                getattr(row, "home_goals", None),
                getattr(row, "away_goals", None),
                str(row.home_team),
                str(row.away_team),
            )
            if actual_result != "-":
                finished_evaluated += 1
                if actual_result == model_result:
                    finished_hits += 1
                    model_hit_label = "Acertou"
                else:
                    model_hit_label = "Errou"
            detail_json = _get_detail_json(
                df,
                row,
                probs,
                cached_real_stats_payload=cached_real_stats_payload,
            )
        except Exception:
            pass
        detail_key = register_detail(detail_json)
        if detail_key:
            detail_data = detail_store.get(detail_key, {})
            match_catalog.append(
                {
                    "competition": name,
                    "status": str(getattr(row, "status", "")),
                    "filter_date": filter_date,
                    "date_label": display_date,
                    "matchup": f"{row.home_team} x {row.away_team}",
                    "home": str(row.home_team),
                    "away": str(row.away_team),
                    "detail_key": detail_key,
                    "model_result": str(detail_data.get("model_result", "-")),
                    "model_probability": float(detail_data.get("model_probability", 0.0) or 0.0),
                    "model_risk_stage": str(detail_data.get("model_risk_stage", "Fora dos criterios")),
                }
            )

        score_text = f"{_fmt_score(row.home_goals)} x {_fmt_score(row.away_goals)}"
        finished_rows.append(
            "<tr "
            "data-filter-scope=\"general\" "
            f"data-date=\"{escape(filter_date)}\" "
            f"data-odd-home=\"{_fmt_odd(row.odds_home)}\" "
            f"data-odd-draw=\"{_fmt_odd(row.odds_draw)}\" "
            f"data-odd-away=\"{_fmt_odd(row.odds_away)}\" "
            ">"
            f"<td>{escape(display_date)}</td>"
            f"<td>{escape(str(row.home_team))}</td>"
            f"<td>{escape(score_text)}</td>"
            f"<td>{escape(str(row.away_team))}</td>"
            f"<td>{escape(model_result)}</td>"
            f"<td>{escape(model_probability)}</td>"
            f"<td>{escape(model_hit_label)}</td>"
            f"<td>{escape(_fmt_odd(row.odds_home))}</td>"
            f"<td>{escape(_fmt_odd(row.odds_draw))}</td>"
            f"<td>{escape(_fmt_odd(row.odds_away))}</td>"
            f"<td><button class='btn-mini' data-detail-key='{escape(detail_key)}' onclick='showMatchDetails(this)'>Visualizar</button></td>"
            "</tr>"
        )
        analysis_options.append(
            f"<option data-detail-key=\"{escape(detail_key)}\">[Finalizado] {escape(display_date)} | {escape(str(row.home_team))} {escape(score_text)} {escape(str(row.away_team))}</option>"
        )

    safe_rows = []
    safe_df = build_safe_bets_table(
        matches_df=df,
        bankroll=1000.0,
        kelly_fractional=0.25,
        min_model_prob=0.55,
        min_expected_value=0.02,
        max_odd=2.20,
        min_bookmakers=8,
    )
    if not safe_df.empty:
        for row in safe_df.head(10).itertuples(index=False):
            display_date = _format_match_datetime(row.date_text, getattr(row, "event_timestamp", None), "Agendado")
            filter_date = _match_filter_date(row.date_text, getattr(row, "event_timestamp", None), "Agendado")
            try:
                probs = calculate_match_probabilities(df, row.home_team, row.away_team)
                # Find original odds for detail view
                match_orig = fixtures_valid[(fixtures_valid["home_team"] == row.home_team) & (fixtures_valid["away_team"] == row.away_team)].iloc[0]
                tip = suggest_bet_strategy(
                    probs,
                    odd_home=float(match_orig.odds_home),
                    odd_draw=float(match_orig.odds_draw),
                    odd_away=float(match_orig.odds_away),
                    bankroll=1000.0,
                    kelly_fractional=0.25,
                )
                detail_json = _get_detail_json(df, match_orig, probs, tip)
            except Exception:
                detail_json = "{}"
            detail_key = register_detail(detail_json)

            safe_rows.append(
                f"<tr data-filter-scope=\"model\" data-date=\"{escape(filter_date)}\" data-odd=\"{row.odd:.2f}\" data-prob=\"{row.model_probability:.4f}\" data-ev=\"{row.expected_value:.4f}\" data-books=\"{row.bookmakers}\">"
                f"<td>{escape(display_date)}</td>"
                f"<td>{escape(str(row.home_team))} x {escape(str(row.away_team))}</td>"
                f"<td>{escape(_market_label(str(row.market), str(row.home_team), str(row.away_team)))}</td>"
                f"<td>{row.odd:.2f}</td>"
                f"<td>{row.model_probability * 100:.1f}%</td>"
                f"<td>{row.expected_value * 100:.2f}%</td>"
                f"<td>{row.bookmakers}</td>"
                f"<td>R$ {row.stake:.2f}</td>"
                f"<td><button class='btn-mini' data-detail-key='{escape(detail_key)}' onclick='showMatchDetails(this)'>Visualizar</button></td>"
                "</tr>"
            )

    stats = {
        "id": comp_id,
        "name": name,
        "finished": len(finished),
        "finished_hits": finished_hits,
        "finished_evaluated": finished_evaluated,
        "finished_hit_rate": round((finished_hits / finished_evaluated) * 100) if finished_evaluated else 0,
        "fixtures": len(fixtures),
        "fixtures_valid": len(fixtures_valid),
        "safe": len(safe_rows),
        "recommendations": len(rec_rows),
    }

    section_html = f"""
<section class="card competition-card" id="{comp_id}" data-comp="{escape(name)}">
  <div class="card-head">
    <div class="title-block">
      <div class="eyebrow">Competicao monitorada</div>
      <div class="title-row">
        <h2>{escape(name)}</h2>
        <div class="badge">Leitura organizada por seguranca, valor e agenda</div>
      </div>
    </div>
    <div class="stats-rail">
      <div class="stat-chip"><span>Finalizados</span><strong>{len(finished)}</strong></div>
      <div class="stat-chip"><span>Acerto modelo</span><strong>{stats["finished_hit_rate"]}%</strong></div>
      <div class="stat-chip"><span>Futuros</span><strong>{len(fixtures)}</strong></div>
      <div class="stat-chip"><span>Com odds</span><strong>{len(fixtures_valid)}</strong></div>
      <div class="stat-chip"><span>Top seguros</span><strong>{len(safe_rows)}</strong></div>
    </div>
  </div>

  <div class="analysis-selector">
    <label for="selector-{comp_id}">Escolher jogo para analisar</label>
    <div class="analysis-selector-row">
      <select id="selector-{comp_id}">
        {''.join(analysis_options) if analysis_options else '<option>Nenhum jogo disponivel para analise</option>'}
      </select>
      <button class="btn secondary" type="button" onclick="showSelectedMatch('selector-{comp_id}')" {'disabled' if not analysis_options else ''}>Abrir analise</button>
    </div>
  </div>

  <div class="panel panel-safe">
    <div class="panel-head">
      <div class="panel-title">
        <span class="panel-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none">
            <path d="M12 3l7 3v5c0 4.5-2.9 8.7-7 10-4.1-1.3-7-5.5-7-10V6l7-3z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>
            <path d="M9 12.2l2 2 4-4.2" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </span>
        <div>
        <div class="panel-kicker">Prioridade 1</div>
        <h3>Jogos mais seguros</h3>
        </div>
      </div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Data</th><th>Jogo</th><th>Resultado sugerido</th><th>Odd</th><th>Prob. Modelo</th><th>EV</th><th>Casas</th><th>Stake</th><th>Acao</th></tr></thead>
        <tbody>{''.join(safe_rows) if safe_rows else '<tr><td colspan="9">Nenhum jogo passou no filtro conservador.</td></tr>'}</tbody>
      </table>
    </div>
  </div>

  <div class="panel panel-value">
    <div class="panel-head">
      <div class="panel-title">
        <span class="panel-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none">
            <path d="M4 17l5-5 4 3 7-8" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M15 7h5v5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </span>
        <div>
        <div class="panel-kicker">Prioridade 2</div>
        <h3>Melhores entradas</h3>
        </div>
      </div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Data</th><th>Jogo</th><th>Resultado sugerido</th><th>Odd</th><th>Prob. Modelo</th><th>EV</th><th>Stake</th><th>Acao</th></tr></thead>
        <tbody>{''.join(rec_rows) if rec_rows else '<tr><td colspan="8">Sem recomendacoes disponiveis.</td></tr>'}</tbody>
      </table>
    </div>
  </div>

  <div class="panel panel-agenda">
    <div class="panel-head">
      <div class="panel-title">
        <span class="panel-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none">
            <rect x="4" y="5" width="16" height="15" rx="3" stroke="currentColor" stroke-width="1.8"/>
            <path d="M8 3v4M16 3v4M4 10h16" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
          </svg>
        </span>
        <div>
        <div class="panel-kicker">Prioridade 3</div>
        <h3>Agenda de jogos futuros</h3>
        </div>
      </div>
    </div>
    <div class="table-wrap table-wrap-match-list">
      <table>
        <thead><tr><th>Data</th><th>Mandante</th><th>Visitante</th><th>Odd Casa</th><th>Odd Empate</th><th>Odd Fora</th><th>Casas</th><th>Acao</th></tr></thead>
        <tbody>{''.join(rows_html) if rows_html else '<tr><td colspan="8">Sem jogos futuros.</td></tr>'}</tbody>
      </table>
    </div>
  </div>

  <div class="panel panel-results">
    <div class="panel-head">
      <div class="panel-title">
        <span class="panel-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none">
            <path d="M6 12h12M12 6v12" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
            <circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.8"/>
          </svg>
        </span>
        <div>
        <div class="panel-kicker">Historico</div>
        <h3>Resultados recentes</h3>
        </div>
      </div>
    </div>
    <div class="table-wrap table-wrap-match-list">
      <table>
        <thead><tr><th>Data</th><th>Mandante</th><th>Placar</th><th>Visitante</th><th>Leitura modelo</th><th>Prob. Modelo</th><th>Acerto</th><th>Odd Casa</th><th>Odd Empate</th><th>Odd Fora</th><th>Acao</th></tr></thead>
        <tbody>{''.join(finished_rows) if finished_rows else '<tr><td colspan="11">Sem jogos finalizados.</td></tr>'}</tbody>
      </table>
    </div>
  </div>

  <p class="empty-state" hidden>Nenhuma linha desta competicao atende aos filtros atuais.</p>
</section>
"""
    return section_html, stats, detail_store, risk_entries, match_catalog


def _build_risk_block(title: str, stage: str, description: str, meta: list[str], entries: list[dict[str, object]], tone_class: str) -> str:
    rows = []
    ordered_entries = sorted(entries, key=lambda item: (-float(item["probability"]), -float(item["ev"]), str(item["competition"]), str(item["matchup"])))[:12]
    for entry in ordered_entries:
        rows.append(
            f"<tr data-filter-scope='model' data-date='{escape(str(entry.get('filter_date', '')))}' data-odd='{float(entry['odd']):.2f}' data-prob='{float(entry['probability']):.4f}' data-ev='{float(entry['ev']):.4f}' data-books='{int(entry['bookmakers'])}'>"
            f"<td>{escape(str(entry['competition']))}</td>"
            f"<td>{escape(str(entry['date_text']))}</td>"
            f"<td>{escape(str(entry['matchup']))}</td>"
            f"<td>{escape(str(entry['suggestion']))}</td>"
            f"<td>{float(entry['odd']):.2f}</td>"
            f"<td>{float(entry['probability']) * 100:.1f}%</td>"
            f"<td>{float(entry['ev']) * 100:.2f}%</td>"
            f"<td>{int(entry['bookmakers'])}</td>"
            f"<td><button class='btn-mini' data-detail-key='{escape(str(entry['detail_key']))}' onclick='showMatchDetails(this)'>Visualizar</button></td>"
            "</tr>"
        )

    return f"""
    <article class="risk-card {tone_class}">
      <span class="risk-badge">{escape(title)}</span>
      <strong>{escape(description)}</strong>
      <div class="risk-meta">{''.join(f'<span>{escape(item)}</span>' for item in meta)}</div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Competicao</th><th>Data</th><th>Jogo</th><th>Leitura</th><th>Odd</th><th>Prob. Modelo</th><th>EV</th><th>Casas</th><th>Acao</th></tr></thead>
          <tbody>{''.join(rows) if rows else '<tr><td colspan="9">Nenhum jogo se encaixa nesta faixa no momento.</td></tr>'}</tbody>
        </table>
      </div>
    </article>
    """


def build_index_html(competition_frames: dict[str, pd.DataFrame] | None = None) -> str:
    sections: list[str] = []
    competition_stats: list[dict[str, int | str]] = []
    detail_registry: dict[str, object] = {}
    global_risk_entries: list[dict[str, object]] = []
    global_match_catalog: list[dict[str, object]] = []
    real_stats_cache = load_real_match_stats_cache()

    for comp in COMPETITIONS:
        if competition_frames and comp in competition_frames:
            df = competition_frames[comp].copy()
        else:
            df = load_competition_matches(comp)
        section_html, stats, section_details, section_risk_entries, section_match_catalog = _build_competition_section(
            comp,
            df,
            real_stats_cache=real_stats_cache,
        )
        sections.append(section_html)
        competition_stats.append(stats)
        detail_registry.update(section_details)
        global_risk_entries.extend(section_risk_entries)
        global_match_catalog.extend(section_match_catalog)

    total_finished = sum(int(item["finished"]) for item in competition_stats)
    total_fixtures = sum(int(item["fixtures"]) for item in competition_stats)
    total_odds = sum(int(item["fixtures_valid"]) for item in competition_stats)
    total_safe = sum(int(item["safe"]) for item in competition_stats)
    total_recommendations = sum(int(item["recommendations"]) for item in competition_stats)
    competition_count = len(competition_stats)
    odds_coverage = round((total_odds / total_fixtures) * 100) if total_fixtures else 0
    safe_rate = round((total_safe / total_odds) * 100) if total_odds else 0
    recommendations_rate = round((total_recommendations / total_odds) * 100) if total_odds else 0

    competition_options = "".join(f"<option>{escape(str(item['name']))}</option>" for item in competition_stats)
    competition_filter_cards = "".join(
        (
            f"<button class=\"competition-filter-card\" type=\"button\" data-comp-filter=\"{escape(str(item['name']))}\">"
            f"<strong>{escape(str(item['name']))}</strong>"
            f"<span>{int(item['fixtures'])} jogos futuros</span>"
            f"<small>{int(item['safe'])} seguros • {int(item['fixtures_valid'])} com odds</small>"
            "</button>"
        )
        for item in competition_stats
    )
    competition_jump_links = "".join(
        f"<a href=\"#{escape(str(item['id']))}\" class=\"jump-link\" data-comp-name=\"{escape(str(item['name']))}\">{escape(str(item['name']))}<span>{int(item['safe'])} seguros</span></a>"
        for item in competition_stats
    )
    side_league_cards = "".join(
        (
            f"<a href=\"#{escape(str(item['id']))}\" class=\"rail-link\" data-comp-name=\"{escape(str(item['name']))}\">"
            f"<div class=\"rail-link-head\"><strong>{escape(str(item['name']))}</strong><span>{int(item['fixtures'])} jogos</span></div>"
            f"<div class=\"rail-link-meta\"><span>{int(item['safe'])} seguros</span><span>{int(item['fixtures_valid'])} odds</span></div>"
            f"<div class=\"rail-track\"><i style=\"width:{round((int(item['fixtures_valid']) / int(item['fixtures'])) * 100) if int(item['fixtures']) else 0}%\"></i></div>"
            "</a>"
        )
        for item in competition_stats
    )
    ai_prompt_html = escape(AI_PROMPT_TEMPLATE)
    ai_prompt_js = AI_PROMPT_TEMPLATE
    match_catalog_json = json.dumps(global_match_catalog, ensure_ascii=False)
    low_risk_entries = [item for item in global_risk_entries if item["stage"] == "Baixo risco"]
    medium_risk_entries = [item for item in global_risk_entries if item["stage"] == "Medio risco"]
    high_risk_entries = [item for item in global_risk_entries if item["stage"] == "Alto risco"]
    risk_blocks_html = "".join(
        [
            _build_risk_block(
                "Baixo risco",
                "Baixo risco",
                "Jogos com leitura mais conservadora, maior estabilidade estatistica e cobertura de mercado mais confiavel.",
                [f"{len(low_risk_entries)} jogos no radar", "Odds ate 1.95", "Prob. minima 62%", "Casas min. 10"],
                low_risk_entries,
                "low",
            ),
            _build_risk_block(
                "Medio risco",
                "Medio risco",
                "Faixa equilibrada para buscar bom acerto sem abrir mao de algum retorno potencial nas linhas sugeridas.",
                [f"{len(medium_risk_entries)} jogos no radar", "Odds ate 2.20", "Prob. minima 55%", "Casas min. 8"],
                medium_risk_entries,
                "medium",
            ),
            _build_risk_block(
                "Alto risco",
                "Alto risco",
                "Jogos mais agressivos, com edge maior e oscilacao mais alta. Faz sentido quando a leitura aceita mais variancia.",
                [f"{len(high_risk_entries)} jogos no radar", "Odds ate 2.90", "Prob. minima 48%", "Casas min. 5"],
                high_risk_entries,
                "high",
            ),
        ]
    )

    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Departamento de Dados de Futebol | Index Inicial</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');
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
      --shadow: 0 24px 60px rgba(15,23,42,0.10);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      font-family: 'Manrope', 'Segoe UI', sans-serif;
      color: var(--text);
      background:
        radial-gradient(900px 420px at 0% 0%, rgba(59,130,246,0.16), transparent 55%),
        radial-gradient(860px 420px at 100% 0%, rgba(16,185,129,0.14), transparent 50%),
        linear-gradient(180deg, #f8fafc 0%, var(--bg) 65%, #e9eef5 100%);
    }}
    .container {{ width: 100%; max-width: none; margin: 0 auto; padding: 24px clamp(18px, 2.2vw, 30px) 44px; }}
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
    .brand-block {{ display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
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
    .brand-copy strong {{ display: block; font-size: .96rem; letter-spacing: -.02em; font-family: "Space Grotesk", sans-serif; }}
    .brand-copy span {{ display: block; margin-top: 3px; color: var(--muted); font-size: .83rem; }}
    .topbar-meta {{ display: flex; flex-wrap: wrap; gap: 10px; }}
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
    .meta-pill strong {{ color: #0f172a; }}
    .status-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #22c55e;
      box-shadow: 0 0 0 4px rgba(34,197,94,.14);
    }}
    .hero {{
      position: relative;
      overflow: hidden;
      padding: 28px;
      border-radius: 28px;
      color: #f8fafc;
      background: linear-gradient(135deg, #081625 0%, #14304f 54%, #0f766e 100%);
      box-shadow: 0 28px 70px rgba(8,22,37,0.26);
    }}
    .hero::before {{
      content: "";
      position: absolute;
      inset: 0;
      background:
        radial-gradient(circle at top right, rgba(96,165,250,.22), transparent 28%),
        linear-gradient(120deg, transparent 0%, rgba(255,255,255,.06) 100%);
      pointer-events: none;
    }}
    .hero-grid {{ display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(260px, 0.9fr); gap: 18px; align-items: end; }}
    .hero-grid, .metrics, .quick-nav {{ position: relative; z-index: 1; }}
    .hero-tag {{ display: inline-flex; padding: 8px 12px; border-radius: 999px; font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.16); color: #dbeafe; }}
    .hero h1 {{ margin: 14px 0 0; max-width: 12ch; font-size: clamp(2.2rem, 4vw, 3.6rem); line-height: .96; letter-spacing: -.04em; font-family: "Space Grotesk", sans-serif; }}
    .hero p {{ margin: 14px 0 0; max-width: 64ch; color: rgba(226,232,240,.88); line-height: 1.68; }}
    .hero-stack {{ display: grid; gap: 12px; }}
    .hero-note {{ padding: 16px 18px; border-radius: 18px; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); }}
    .hero-note span {{ display: block; font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #bfdbfe; }}
    .hero-note strong {{ display: block; margin-top: 8px; font-size: 1.7rem; line-height: 1.05; font-family: "Space Grotesk", sans-serif; }}
    .hero-note p {{ display: none; }}
    .metrics {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-top: 20px; }}
    .metric {{ padding: 16px; border-radius: 18px; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); }}
    .metric span {{ display: block; font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: #cfe4ff; }}
    .metric strong {{ display: block; margin-top: 10px; font-size: 1.65rem; letter-spacing: -.04em; font-family: "Space Grotesk", sans-serif; }}
    .metric p {{ margin: 8px 0 0; font-size: .86rem; color: rgba(226,232,240,.8); line-height: 1.45; }}
    .metric-track {{ margin-top: 12px; height: 7px; border-radius: 999px; background: rgba(255,255,255,.12); overflow: hidden; }}
    .metric-track i {{ display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #93c5fd, #6ee7b7); }}
    .quick-nav {{ margin-top: 16px; display: flex; flex-wrap: wrap; gap: 10px; }}
    .quick-nav::before {{ content: "Areas monitoradas"; width: 100%; margin-bottom: 2px; font-size: .76rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; color: #bfdbfe; }}
    .jump-link {{ display: inline-flex; align-items: center; gap: 10px; padding: 10px 14px; border-radius: 999px; background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.12); color: #fff; text-decoration: none; font-weight: 600; box-shadow: inset 0 1px 0 rgba(255,255,255,.08); }}
    .jump-link span {{ font-size: .76rem; letter-spacing: .05em; text-transform: uppercase; color: #bfdbfe; }}
    .jump-link.active {{ background: rgba(255,255,255,.22); border-color: rgba(255,255,255,.32); box-shadow: inset 0 1px 0 rgba(255,255,255,.16), 0 8px 20px rgba(8,22,37,.18); }}
    .jump-link.active span {{ color: #ffffff; }}
    .dashboard-shell {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
      align-items: start;
      margin-top: 18px;
    }}
    .side-rail {{
      display: none;
    }}
    .rail-card {{
      padding: 18px;
      border-radius: 22px;
      background: linear-gradient(180deg, rgba(255,255,255,.96), rgba(246,249,252,.86));
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}
    .rail-card h3 {{
      margin: 0;
      font-size: 1rem;
      font-family: "Space Grotesk", sans-serif;
    }}
    .rail-card p {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: .88rem;
      line-height: 1.55;
    }}
    .rail-list {{
      display: grid;
      gap: 10px;
      margin-top: 14px;
    }}
    .rail-link {{
      display: grid;
      gap: 8px;
      padding: 12px;
      border-radius: 16px;
      background: #fff;
      border: 1px solid rgba(148,163,184,.22);
      text-decoration: none;
      color: inherit;
    }}
    .rail-link.active {{
      border-color: rgba(29,78,216,.34);
      background: linear-gradient(135deg, rgba(239,246,255,.96), rgba(236,253,245,.92));
      box-shadow: 0 14px 28px rgba(29,78,216,.10);
    }}
    .rail-link-head, .rail-link-meta {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .rail-link-head strong {{
      font-size: .92rem;
      letter-spacing: -.02em;
      font-family: "Space Grotesk", sans-serif;
    }}
    .rail-link-head span, .rail-link-meta span {{
      color: var(--muted);
      font-size: .78rem;
    }}
    .rail-track {{
      height: 7px;
      border-radius: 999px;
      background: #e5edf7;
      overflow: hidden;
    }}
    .rail-track i {{
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--blue), var(--teal));
    }}
    .spark-card {{
      display: grid;
      gap: 10px;
    }}
    .spark-head {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
    }}
    .spark-head strong {{
      font-size: .92rem;
      font-family: "Space Grotesk", sans-serif;
    }}
    .spark-head span {{
      color: var(--muted);
      font-size: .8rem;
    }}
    .sparkline {{
      width: 100%;
      height: 72px;
      border-radius: 16px;
      background: linear-gradient(180deg, #f8fbff, #eef5fb);
      border: 1px solid rgba(148,163,184,.18);
      padding: 10px;
    }}
    .sparkline svg {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    .sparkline polyline {{
      fill: none;
      stroke: var(--blue);
      stroke-width: 2.2;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    .sparkline path {{
      fill: rgba(29,78,216,.08);
      stroke: none;
    }}
    .dashboard-main {{
      min-width: 0;
    }}
    .card {{ margin-top: 18px; padding: 20px; border-radius: 22px; background: var(--card); border: 1px solid var(--line); box-shadow: var(--shadow); backdrop-filter: blur(18px); }}
    .controls {{ display: grid; gap: 18px; }}
    .controls::before {{
      content: "";
      display: block;
      height: 4px;
      width: 120px;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--blue), var(--teal));
    }}
    .controls-head, .panel-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; }}
    .card-head {{
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(360px, 1fr);
      gap: 18px;
      align-items: start;
    }}
    .eyebrow {{ display: inline-flex; margin-bottom: 8px; font-size: 12px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; color: var(--blue); }}
    h2 {{ margin: 0; font-size: clamp(1.35rem, 2vw, 1.85rem); letter-spacing: -.03em; font-family: "Space Grotesk", sans-serif; }}
    h3 {{ margin: 4px 0 0; font-size: 1.06rem; letter-spacing: -.02em; font-family: "Space Grotesk", sans-serif; }}
    .copy, .section-copy, .panel-copy {{ margin: 8px 0 0; color: var(--muted); line-height: 1.6; }}
    .summary-box {{ min-width: min(100%, 280px); padding: 14px 16px; border-radius: 18px; background: linear-gradient(135deg, #eff6ff, #ecfeff); border: 1px solid rgba(59,130,246,.14); color: #15314a; font-size: .92rem; line-height: 1.55; box-shadow: inset 0 1px 0 rgba(255,255,255,.55); }}
    .filters {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .competition-filter-shell {{
      display: grid;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .competition-filter-shell label {{
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .04em;
      text-transform: uppercase;
      color: #334155;
    }}
    .competition-filter-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .competition-filter-card {{
      display: grid;
      gap: 6px;
      padding: 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,.98), rgba(241,245,249,.84));
      color: var(--text);
      text-align: left;
      cursor: pointer;
      transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease, background .18s ease;
    }}
    .competition-filter-card:hover {{
      transform: translateY(-2px);
      box-shadow: 0 18px 32px rgba(15,23,42,.08);
      border-color: rgba(29,78,216,.24);
    }}
    .competition-filter-card strong {{
      font-size: .98rem;
      font-family: "Space Grotesk", sans-serif;
      letter-spacing: -.02em;
    }}
    .competition-filter-card span,
    .competition-filter-card small {{
      color: var(--muted);
      line-height: 1.45;
    }}
    .competition-filter-card.active {{
      border-color: rgba(29,78,216,.34);
      background: linear-gradient(135deg, rgba(239,246,255,.98), rgba(236,253,245,.94));
      box-shadow: 0 18px 34px rgba(29,78,216,.12);
    }}
    .competition-filter-card.all-card {{
      border-style: dashed;
    }}
    .field {{ display: grid; gap: 8px; padding: 14px; border-radius: 18px; background: linear-gradient(180deg, rgba(255,255,255,.96), rgba(241,245,249,.76)); border: 1px solid var(--line); }}
    .field label {{ font-size: 12px; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; color: #334155; }}
    .field input, .field select {{ height: 44px; border-radius: 12px; border: 1px solid var(--line-strong); padding: 0 14px; font: inherit; color: var(--text); background: rgba(255,255,255,.98); outline: none; }}
    .field input:focus, .field select:focus {{ border-color: rgba(29,78,216,.48); box-shadow: 0 0 0 4px rgba(29,78,216,.08); }}
    .hint {{ font-size: 12px; color: var(--muted); line-height: 1.45; }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .btn, .btn-link {{ height: 46px; padding: 0 16px; border-radius: 14px; border: 1px solid transparent; font: inherit; font-weight: 700; text-decoration: none; cursor: pointer; display: inline-flex; align-items: center; justify-content: center; transition: transform .18s ease, box-shadow .18s ease, background .18s ease; }}
    .btn.primary {{ background: linear-gradient(135deg, var(--blue), #2563eb); color: #fff; box-shadow: 0 16px 28px rgba(37,99,235,.22); }}
    .btn.secondary, .btn-link {{ background: #f8fafc; border-color: var(--line-strong); color: var(--text); }}
    .btn:hover, .btn-link:hover {{ transform: translateY(-1px); }}
    .btn[disabled], .btn-link[disabled], .btn-float[disabled] {{ opacity: .68; cursor: wait; transform: none; box-shadow: none; }}
    .btn-mini {{ padding: 4px 10px; font-size: 11px; font-weight: 700; border-radius: 8px; border: 1px solid var(--line-strong); background: #fff; cursor: pointer; }}
    .btn-mini:hover {{ background: var(--blue); color: #fff; border-color: var(--blue); }}
    .launcher-card {{ display: grid; gap: 18px; }}
    .launcher-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .launcher-item {{
      display: grid;
      gap: 10px;
      padding: 18px;
      border-radius: 20px;
      background: linear-gradient(180deg, rgba(255,255,255,.98), rgba(241,245,249,.82));
      border: 1px solid var(--line);
      box-shadow: inset 0 1px 0 rgba(255,255,255,.55);
    }}
    .launcher-item strong {{
      font-size: 1rem;
      letter-spacing: -.02em;
      font-family: "Space Grotesk", sans-serif;
    }}
    .launcher-item span {{
      color: var(--muted);
      font-size: .9rem;
      line-height: 1.55;
    }}
    .risk-strip {{
      display: grid;
      gap: 16px;
      margin-top: 14px;
    }}
    .risk-strip-head {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .risk-strip-head p {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: .92rem;
      line-height: 1.58;
    }}
    .risk-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }}
    .risk-card {{
      display: grid;
      gap: 12px;
      padding: 18px;
      border-radius: 22px;
      background: linear-gradient(180deg, rgba(255,255,255,.98), rgba(241,245,249,.82));
      border: 1px solid var(--line);
      box-shadow: inset 0 1px 0 rgba(255,255,255,.55);
    }}
    .risk-card.low {{ border-color: rgba(34,197,94,.24); }}
    .risk-card.medium {{ border-color: rgba(245,158,11,.26); }}
    .risk-card.high {{ border-color: rgba(244,63,94,.24); }}
    .risk-badge {{
      display: inline-flex;
      align-items: center;
      width: fit-content;
      padding: 7px 11px;
      border-radius: 999px;
      font-size: .76rem;
      font-weight: 800;
      letter-spacing: .08em;
      text-transform: uppercase;
      border: 1px solid currentColor;
    }}
    .risk-card.low .risk-badge {{ color: #15803d; background: rgba(34,197,94,.10); }}
    .risk-card.medium .risk-badge {{ color: #b45309; background: rgba(245,158,11,.12); }}
    .risk-card.high .risk-badge {{ color: #be123c; background: rgba(244,63,94,.10); }}
    .risk-card strong {{
      font-size: 1.02rem;
      letter-spacing: -.02em;
      font-family: "Space Grotesk", sans-serif;
    }}
    .risk-card p {{
      margin: 0;
      color: var(--muted);
      font-size: .9rem;
      line-height: 1.6;
    }}
    .risk-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .risk-meta span {{
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      background: #f8fafc;
      border: 1px solid rgba(148,163,184,.22);
      color: #334155;
      font-size: .78rem;
      font-weight: 600;
    }}
    .risk-card .table-wrap {{ margin-top: 2px; }}
    .ai-module {{ margin-top: 18px; padding: 20px; border-radius: 22px; background: linear-gradient(135deg, rgba(8,22,37,.98), rgba(20,48,79,.98)); color: #f8fafc; border: 1px solid rgba(148,163,184,.16); box-shadow: 0 22px 50px rgba(8,22,37,.18); }}
    .ai-module .eyebrow {{ color: #bfdbfe; }}
    .ai-module h2 {{ color: #ffffff; }}
    .ai-module .copy {{ color: rgba(226,232,240,.84); }}
    .ai-module-grid {{ display: grid; grid-template-columns: minmax(250px, 320px) minmax(0, 1fr); gap: 16px; margin-top: 18px; }}
    .ai-module-side {{ display: grid; gap: 12px; }}
    .ai-mini-card {{ padding: 16px; border-radius: 18px; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.12); }}
    .ai-mini-card strong {{ display: block; font-size: .96rem; font-family: "Space Grotesk", sans-serif; }}
    .ai-mini-card p {{ display: none; }}
    .ai-mini-card label {{ display: block; margin-bottom: 8px; font-size: 12px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; color: #bfdbfe; }}
    .ai-mini-card input {{ width: 100%; height: 44px; border-radius: 12px; border: 1px solid rgba(191,219,254,.22); padding: 0 14px; font: inherit; color: #ffffff; background: rgba(255,255,255,.08); outline: none; }}
    .ai-mini-card input:focus {{ border-color: rgba(191,219,254,.48); box-shadow: 0 0 0 4px rgba(29,78,216,.12); }}
    .ai-prompt-shell {{ padding: 16px; border-radius: 18px; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.12); }}
    .ai-prompt-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }}
    .ai-prompt-head strong {{ font-size: 1rem; font-family: "Space Grotesk", sans-serif; }}
    .ai-prompt-head span {{ color: #bfdbfe; font-size: .82rem; }}
    .ai-prompt-area {{ width: 100%; min-height: 620px; resize: vertical; border-radius: 16px; border: 1px solid rgba(191,219,254,.18); padding: 16px; font: 500 .92rem/1.6 "Manrope", "Segoe UI", sans-serif; color: #e2e8f0; background: rgba(8,22,37,.72); outline: none; }}
    .ai-prompt-actions {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }}
    .ai-prompt-preview {{ min-height: 180px; padding: 16px; border-radius: 16px; border: 1px solid rgba(191,219,254,.16); background: rgba(8,22,37,.62); color: #dbeafe; white-space: pre-wrap; line-height: 1.7; }}
    .ai-status {{ margin-top: 10px; color: #bbf7d0; font-size: .85rem; min-height: 1.2em; }}
    .ai-response {{ margin-top: 14px; padding: 16px; border-radius: 16px; background: rgba(8,22,37,.72); border: 1px solid rgba(191,219,254,.16); min-height: 140px; color: #e2e8f0; white-space: pre-wrap; line-height: 1.7; }}
    .legend {{ margin-top: 16px; display: flex; flex-wrap: wrap; gap: 10px; }}
    .pill {{ display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 999px; border: 1px solid var(--line); background: rgba(255,255,255,.82); font-size: .84rem; }}
    .pill::before {{ content: ""; width: 10px; height: 10px; border-radius: 50%; background: currentColor; opacity: .78; }}
    .metric p {{ display: none; }}
    .competition-card {{
      position: relative;
      padding: 22px;
      overflow: hidden;
      background:
        linear-gradient(180deg, rgba(255,255,255,.95), rgba(255,255,255,.86)),
        radial-gradient(circle at top right, rgba(37,99,235,.06), transparent 26%);
    }}
    .competition-card::before {{
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      width: 5px;
      height: 100%;
      background: linear-gradient(180deg, var(--blue), var(--teal));
    }}
    .title-block {{ max-width: none; min-width: 0; }}
    .title-row {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
    .badge {{ padding: 8px 12px; border-radius: 999px; background: linear-gradient(135deg, #ecfeff, #eff6ff); border: 1px solid rgba(15,118,110,.16); color: var(--teal); font-size: .8rem; font-weight: 700; }}
    .stats-rail {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; min-width: 0; width: 100%; }}
    .stat-chip {{ padding: 14px; border-radius: 16px; background: linear-gradient(180deg, #fff, #f8fafc); border: 1px solid var(--line); }}
    .stat-chip span {{ display: block; font-size: 12px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }}
    .stat-chip strong {{ display: block; margin-top: 8px; font-size: 1.34rem; letter-spacing: -.04em; font-family: "Space Grotesk", sans-serif; }}
    .panel {{ margin-top: 18px; padding: 16px; border-radius: 18px; background: linear-gradient(180deg, rgba(255,255,255,.98), rgba(248,250,252,.82)); border: 1px solid var(--line); box-shadow: inset 0 1px 0 rgba(255,255,255,.6); }}
    .panel-safe {{ border-color: rgba(16,185,129,.22); }}
    .panel-value {{ border-color: rgba(37,99,235,.22); }}
    .panel-agenda {{ border-color: rgba(245,158,11,.24); }}
    .panel-results {{ border-color: rgba(100,116,139,.24); }}
    .panel-kicker {{ font-size: 12px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; color: var(--amber); }}
    .panel-safe .panel-kicker {{ color: #059669; }}
    .panel-value .panel-kicker {{ color: var(--blue); }}
    .panel-agenda .panel-kicker {{ color: var(--amber); }}
    .panel-results .panel-kicker {{ color: #475569; }}
    .panel-title {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .panel-icon {{
      width: 42px;
      height: 42px;
      border-radius: 14px;
      display: grid;
      place-items: center;
      background: #f8fafc;
      border: 1px solid rgba(148,163,184,.22);
    }}
    .panel-icon svg {{
      width: 22px;
      height: 22px;
      display: block;
    }}
    .panel-safe .panel-icon {{ color: #059669; background: rgba(16,185,129,.08); border-color: rgba(16,185,129,.18); }}
    .panel-value .panel-icon {{ color: var(--blue); background: rgba(29,78,216,.08); border-color: rgba(29,78,216,.18); }}
    .panel-agenda .panel-icon {{ color: var(--amber); background: rgba(245,158,11,.10); border-color: rgba(245,158,11,.2); }}
    .panel-results .panel-icon {{ color: #475569; background: rgba(148,163,184,.10); border-color: rgba(148,163,184,.2); }}
    .analysis-selector {{
      margin: 16px 0 0;
      padding: 14px 16px;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(255,255,255,.98), rgba(248,250,252,.84));
      border: 1px solid var(--line);
      display: grid;
      gap: 10px;
    }}
    .analysis-selector label {{
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .08em;
      text-transform: uppercase;
      color: #334155;
    }}
    .analysis-selector-row {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
    }}
    .analysis-selector select {{
      width: 100%;
      height: 46px;
      border-radius: 14px;
      border: 1px solid var(--line-strong);
      padding: 0 14px;
      font: inherit;
      color: var(--text);
      background: #fff;
      outline: none;
    }}
    .analysis-selector select:focus {{
      border-color: rgba(29,78,216,.48);
      box-shadow: 0 0 0 4px rgba(29,78,216,.08);
    }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 16px; background: #fff; }}
    .table-wrap-match-list {{
      max-height: 520px;
      overflow-y: auto;
      scrollbar-width: thin;
      scrollbar-color: rgba(148,163,184,.9) rgba(241,245,249,.9);
    }}
    .table-wrap-match-list::-webkit-scrollbar {{
      width: 10px;
      height: 10px;
    }}
    .table-wrap-match-list::-webkit-scrollbar-track {{
      background: rgba(241,245,249,.9);
    }}
    .table-wrap-match-list::-webkit-scrollbar-thumb {{
      background: rgba(148,163,184,.9);
      border-radius: 999px;
      border: 2px solid rgba(241,245,249,.9);
    }}
    table {{ width: 100%; min-width: 780px; border-collapse: collapse; background: #fff; }}
    th, td {{ border-bottom: 1px solid #e8edf4; padding: 11px 12px; text-align: left; font-size: 13px; line-height: 1.5; }}
    th {{ position: sticky; top: 0; background: #eff6ff; color: #14324f; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; }}
    tbody tr:hover td {{ background: #f8fbff; }}
    tr.risk-low td {{ background: #ecfdf3; }}
    tr.risk-med td {{ background: #fff7e6; }}
    tr.risk-high td {{ background: #fff1f2; }}
    tr.risk-none td {{ background: #f8fafc; }}
    .empty-state {{ margin: 16px 0 2px; padding: 14px 16px; border-radius: 14px; background: #fff7ed; border: 1px solid rgba(245,158,11,.22); color: #9a3412; font-size: .92rem; }}
    .date-focus[hidden] {{ display: none; }}
    .date-focus-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      flex-wrap: wrap;
    }}
    .date-focus-head p {{
      margin: 8px 0 0;
      color: var(--muted);
      line-height: 1.6;
    }}
    .date-focus-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .date-match-card {{
      display: grid;
      gap: 12px;
      padding: 18px;
      border-radius: 22px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,.98), rgba(246,249,252,.88));
      box-shadow: 0 16px 28px rgba(15,23,42,.06);
    }}
    .date-match-top {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: flex-start;
      flex-wrap: wrap;
    }}
    .date-match-top strong {{
      font-size: 1rem;
      font-family: "Space Grotesk", sans-serif;
      letter-spacing: -.02em;
    }}
    .date-match-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .date-match-meta span,
    .date-match-note {{
      display: inline-flex;
      align-items: center;
      padding: 7px 10px;
      border-radius: 999px;
      background: #f8fafc;
      border: 1px solid rgba(148,163,184,.20);
      color: #334155;
      font-size: .78rem;
      font-weight: 700;
    }}
    .date-match-copy {{
      color: var(--muted);
      font-size: .9rem;
      line-height: 1.55;
    }}
    .date-match-actions {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .date-focus-empty {{
      margin-top: 16px;
      padding: 16px 18px;
      border-radius: 18px;
      background: #fff7ed;
      border: 1px solid rgba(245,158,11,.22);
      color: #9a3412;
    }}
    
    /* MODAL STYLES */
    .modal-overlay {{
      position: fixed;
      inset: 0;
      background: rgba(8, 22, 37, 0.6);
      backdrop-filter: blur(8px);
      z-index: 2000;
      display: none;
      overflow-y: auto;
      padding: 20px;
    }}
    .modal-card {{
      background: #fff;
      width: 100%;
      max-width: 960px;
      max-height: 90vh;
      border-radius: 28px;
      overflow-y: auto;
      box-shadow: 0 40px 100px rgba(0,0,0,0.3);
      position: relative;
      margin: 24px auto;
      animation: modalShow 0.3s ease-out;
    }}
    @keyframes modalShow {{
      from {{ opacity: 0; transform: translateY(20px) scale(0.98); }}
      to {{ opacity: 1; transform: translateY(0) scale(1); }}
    }}
    .modal-head {{
      padding: 24px 32px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      background: rgba(255,255,255,0.9);
      backdrop-filter: blur(10px);
      z-index: 10;
    }}
    .modal-close {{
      width: 40px;
      height: 40px;
      border-radius: 50%;
      border: 1px solid var(--line-strong);
      background: #fff;
      display: grid;
      place-items: center;
      cursor: pointer;
      font-size: 20px;
      transition: all 0.2s;
    }}
    .modal-close:hover {{ background: #f1f5f9; transform: rotate(90deg); }}
    .modal-body {{ padding: 32px; }}
    .modal-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
    .modal-section {{ margin-bottom: 32px; }}
    .ai-modal-card {{
      background: linear-gradient(135deg, rgba(8,22,37,.98), rgba(20,48,79,.98));
      color: #f8fafc;
      max-width: 1100px;
    }}
    .ai-modal-card .modal-head {{
      background: rgba(8,22,37,.9);
      border-bottom-color: rgba(191,219,254,.16);
    }}
    .ai-modal-card .modal-close {{
      background: rgba(255,255,255,.1);
      border-color: rgba(191,219,254,.16);
      color: #fff;
    }}
    .ai-modal-card .modal-close:hover {{ background: rgba(255,255,255,.18); }}
    .ai-modal-card .copy {{ color: rgba(226,232,240,.82); margin-top: 8px; }}
    .filter-modal-card {{ max-width: 1180px; }}
    .filter-modal-card .modal-body {{ display: grid; gap: 18px; }}
    .chart-container {{
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 20px;
      position: relative;
      height: 300px;
    }}
    .context-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }}
    .context-card {{ padding: 16px; border-radius: 16px; background: #f1f5f9; border: 1px solid var(--line); }}
    .context-card h4 {{ margin: 0 0 10px; font-size: 0.9rem; color: var(--blue); font-family: "Space Grotesk", sans-serif; }}
    .context-stat {{ display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.85rem; }}
    .context-stat strong {{ color: var(--text); }}
    .real-stats-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }}
    .real-stat-card {{ padding: 16px; border-radius: 16px; background: linear-gradient(180deg, #f8fafc, #eef5fb); border: 1px solid var(--line); }}
    .real-stat-card span {{ display: block; font-size: .76rem; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }}
    .real-stat-card strong {{ display: block; margin-top: 8px; font-size: 1.15rem; letter-spacing: -.03em; font-family: "Space Grotesk", sans-serif; }}
    .real-stat-card small {{ display: block; margin-top: 8px; color: var(--muted); font-size: .82rem; line-height: 1.5; }}
    .real-stats-status {{ margin-top: 14px; color: var(--muted); line-height: 1.55; }}
    .real-stats-link {{ margin-top: 12px; display: inline-flex; }}
    .projection-status {{ margin-top: 14px; color: var(--muted); line-height: 1.55; }}
    .projection-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }}
    .accuracy-good {{ color: #15803d !important; }}
    .accuracy-mid {{ color: #b45309 !important; }}
    .accuracy-low {{ color: #be123c !important; }}
    
    .glossary {{ display: grid; gap: 14px; }}
    .glossary-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .glossary-item {{ padding: 16px; border-radius: 18px; background: linear-gradient(180deg, rgba(255,255,255,.96), rgba(241,245,249,.76)); border: 1px solid var(--line); }}
    .glossary-item strong {{ display: block; margin-bottom: 6px; font-size: .95rem; font-family: "Space Grotesk", sans-serif; }}
    .glossary-item p {{ margin: 0; color: var(--muted); line-height: 1.6; }}
    .foot {{ margin-top: 18px; color: var(--muted); font-size: .9rem; text-align: center; }}
    
    .floating-actions {{
      position: fixed;
      top: 14px;
      right: 18px;
      display: flex;
      flex-direction: row;
      gap: 10px;
      z-index: 1000;
      padding: 8px;
      border-radius: 999px;
      background: rgba(8,22,37,.72);
      border: 1px solid rgba(191,219,254,.22);
      box-shadow: 0 12px 28px rgba(8,22,37,.30);
      backdrop-filter: blur(8px);
    }}
    .btn-float {{
      width: 52px;
      height: 52px;
      border-radius: 50%;
      background: var(--blue);
      color: #fff;
      display: grid;
      place-items: center;
      box-shadow: 0 12px 24px rgba(29,78,216,.3);
      cursor: pointer;
      border: none;
      transition: transform .2s ease;
    }}
    .btn-float:hover {{ transform: scale(1.1); }}
    .btn-float svg {{ width: 24px; height: 24px; fill: currentColor; }}
    .btn-float.loading svg {{ animation: spin-refresh 1s linear infinite; }}
    .refresh-overlay {{
      position: fixed;
      inset: 0;
      display: grid;
      place-items: center;
      padding: 20px;
      background: rgba(8,22,37,.58);
      backdrop-filter: blur(8px);
      z-index: 3500;
      opacity: 0;
      pointer-events: none;
      transition: opacity .22s ease;
    }}
    .refresh-overlay.show {{
      opacity: 1;
      pointer-events: auto;
    }}
    .refresh-overlay-card {{
      width: min(560px, 100%);
      padding: 22px 24px;
      border-radius: 22px;
      background: linear-gradient(135deg, rgba(8,22,37,.98), rgba(20,48,79,.98));
      border: 1px solid rgba(191,219,254,.2);
      box-shadow: 0 30px 70px rgba(8,22,37,.45);
      color: #e2e8f0;
      display: grid;
      justify-items: center;
      text-align: center;
      gap: 12px;
    }}
    .refresh-spinner {{
      width: 62px;
      height: 62px;
      border-radius: 50%;
      border: 4px solid rgba(191,219,254,.28);
      border-top-color: #93c5fd;
      border-right-color: #6ee7b7;
      animation: spin-refresh .95s linear infinite;
    }}
    .refresh-overlay-card strong {{
      font-size: 1.05rem;
      letter-spacing: -.02em;
      font-family: "Space Grotesk", sans-serif;
      color: #ffffff;
    }}
    .refresh-overlay-card p {{
      margin: 0;
      color: #bfdbfe;
      font-size: .92rem;
      line-height: 1.58;
    }}
    .refresh-progress-track {{
      width: 100%;
      height: 10px;
      border-radius: 999px;
      background: rgba(191,219,254,.22);
      overflow: hidden;
      border: 1px solid rgba(191,219,254,.22);
    }}
    .refresh-progress-track i {{
      display: block;
      height: 100%;
      width: 0%;
      border-radius: inherit;
      background: linear-gradient(90deg, #60a5fa, #6ee7b7);
      transition: width .28s ease;
    }}
    .refresh-progress-label {{
      margin-top: -2px !important;
      font-size: .84rem !important;
      color: #dbeafe !important;
      font-weight: 700;
      letter-spacing: .02em;
    }}
    .refresh-stage-wrap {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid rgba(191,219,254,.2);
      background: rgba(15,32,52,.58);
      text-align: left;
    }}
    .refresh-stage-head {{
      margin: 0 0 8px;
      font-size: .72rem;
      letter-spacing: .08em;
      text-transform: uppercase;
      color: #93c5fd;
      font-weight: 700;
    }}
    .refresh-stage-list {{
      margin: 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 8px;
    }}
    .refresh-stage-list li {{
      display: grid;
      grid-template-columns: 12px 1fr;
      align-items: center;
      gap: 10px;
      color: rgba(191,219,254,.86);
      font-size: .82rem;
      line-height: 1.3;
      transition: color .2s ease;
    }}
    .refresh-stage-list li i {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      border: 2px solid rgba(147,197,253,.45);
      background: transparent;
      transition: border-color .2s ease, background-color .2s ease, box-shadow .2s ease;
    }}
    .refresh-stage-list li.is-pending {{
      color: rgba(191,219,254,.72);
    }}
    .refresh-stage-list li.is-active {{
      color: #e2e8f0;
      font-weight: 700;
    }}
    .refresh-stage-list li.is-active i {{
      border-color: #60a5fa;
      background: #60a5fa;
      box-shadow: 0 0 0 6px rgba(96,165,250,.18);
      animation: stage-pulse 1s ease infinite;
    }}
    .refresh-stage-list li.is-done {{
      color: #bbf7d0;
    }}
    .refresh-stage-list li.is-done i {{
      border-color: #22c55e;
      background: #22c55e;
      box-shadow: 0 0 0 4px rgba(34,197,94,.14);
      animation: none;
    }}
    .refresh-stage-list li.is-error {{
      color: #fecaca;
      font-weight: 700;
    }}
    .refresh-stage-list li.is-error i {{
      border-color: #ef4444;
      background: #ef4444;
      box-shadow: 0 0 0 5px rgba(239,68,68,.18);
      animation: none;
    }}
    body.refresh-locked {{
      overflow: hidden;
    }}
    @keyframes stage-pulse {{
      0% {{ box-shadow: 0 0 0 0 rgba(96,165,250,.22); }}
      70% {{ box-shadow: 0 0 0 8px rgba(96,165,250,0); }}
      100% {{ box-shadow: 0 0 0 0 rgba(96,165,250,0); }}
    }}
    @keyframes spin-refresh {{
      from {{ transform: rotate(0deg); }}
      to {{ transform: rotate(360deg); }}
    }}
    
    @media (max-width: 1180px) {{
      .side-rail {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .card-head {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 1100px) {{ .hero-grid, .metrics, .filters, .glossary-grid, .ai-module-grid, .modal-grid, .launcher-grid, .risk-grid, .real-stats-grid, .projection-grid, .competition-filter-grid, .date-focus-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 760px) {{
      .container {{ padding: 18px 14px 32px; }}
      .topbar {{ align-items: flex-start; }}
      .hero, .card, .competition-card {{ padding: 18px; }}
      .hero-grid, .metrics, .filters, .stats-rail, .glossary-grid, .side-rail, .ai-module-grid, .modal-grid, .context-grid, .launcher-grid, .risk-grid, .real-stats-grid, .projection-grid, .competition-filter-grid, .date-focus-grid {{ grid-template-columns: 1fr; }}
      .topbar, .brand-block, .topbar-meta {{ flex-direction: column; align-items: flex-start; }}
      .btn, .btn-link {{ width: 100%; }}
      .floating-actions {{ top: 10px; right: 10px; gap: 8px; padding: 6px; }}
      .btn-float {{ width: 44px; height: 44px; }}
      .btn-float svg {{ width: 20px; height: 20px; }}
      .analysis-selector-row {{ grid-template-columns: 1fr; }}
      table {{ min-width: 640px; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <section class="topbar">
      <div class="brand-block">
        <div class="brand-mark">FD</div>
        <div class="brand-copy">
          <strong>Departamento de Dados de Futebol</strong>
          <span>Painel executivo para leitura rapida de oportunidades por competicao.</span>
        </div>
      </div>
      <div class="topbar-meta">
        <div class="meta-pill"><span class="status-dot"></span><strong>Atualizado</strong> <span id="portalUpdatedAt">{_current_app_timestamp()}</span></div>
        <div class="meta-pill"><strong>Release</strong> {PORTAL_RELEASE_LABEL}</div>
        <div class="meta-pill"><strong>{competition_count}</strong> competicoes no radar</div>
        <div class="meta-pill"><strong>{odds_coverage}%</strong> cobertura de odds</div>
      </div>
    </section>

    <section class="hero">
      <div class="hero-grid">
        <div></div>
        <div class="hero-stack">
          <div class="hero-note">
            <span>Painel inicial</span>
            <strong>{len(competition_stats)} competicoes</strong>
            <p>{total_fixtures} jogos futuros observados, {total_odds} linhas com odds completas, {total_safe} selecoes conservadoras e {total_finished} jogos finalizados para comparar a performance do modelo.</p>
          </div>
        </div>
      </div>

      <div class="metrics">
        <div class="metric"><span>Competicoes</span><strong>{competition_count}</strong><p>Torneios monitorados no painel.</p><div class="metric-track"><i style="width:100%"></i></div></div>
        <div class="metric"><span>Jogos futuros</span><strong>{total_fixtures}</strong><p>Partidas disponiveis para leitura.</p><div class="metric-track"><i style="width:100%"></i></div></div>
        <div class="metric"><span>Odds completas</span><strong>{total_odds}</strong><p>{odds_coverage}% dos jogos futuros com linha principal completa.</p><div class="metric-track"><i style="width:{odds_coverage}%"></i></div></div>
        <div class="metric"><span>Top seguros</span><strong>{total_safe}</strong><p>{safe_rate}% das linhas com odds entram no filtro conservador.</p><div class="metric-track"><i style="width:{safe_rate}%"></i></div></div>
        <div class="metric"><span>Entradas EV</span><strong>{total_recommendations}</strong><p>{recommendations_rate}% das linhas com odds geram sugestao de stake.</p><div class="metric-track"><i style="width:{recommendations_rate}%"></i></div></div>
      </div>

      <div class="quick-nav">{competition_jump_links}</div>
    </section>

    <div class="dashboard-shell">
      <aside class="side-rail">
        <section class="rail-card" style="margin-bottom: 14px;">
          <div class="eyebrow">Filtro de Data</div>
          <h3>Busca por data</h3>
          <p>Reune todos os jogos da data no painel especial.</p>
          <div class="field" style="margin-top: 14px; padding: 0; background: transparent; border: none;">
            <input id="fdate" type="date" style="width: 100%; border: 1px solid var(--line-strong);" />
          </div>
        </section>

        <section class="rail-card">
          <div class="eyebrow">Radar lateral</div>
          <h3>Focos por competicao</h3>
          <p>Toque nos cards para filtrar o dashboard inteiro.</p>
          <div class="competition-filter-grid" style="grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); margin-top: 14px;">
            <button class="competition-filter-card all-card active" type="button" data-comp-filter="">
              <strong>Todas as competicoes</strong>
              <small>Limpar o foco de competicao.</small>
            </button>
            {competition_filter_cards}
          </div>
        </section>
      </aside>

      <main class="dashboard-main">
        <section id="dateFocusPanel" class="card date-focus" hidden>
          <div class="date-focus-head">
            <div>
              <div class="eyebrow">Recorte por data</div>
              <h2 id="dateFocusTitle">Jogos da data filtrada</h2>
              <p id="dateFocusCopy">Selecione uma data no filtro para reunir todos os confrontos do dia em um unico lugar, com acesso direto a analise.</p>
            </div>
            <div id="dateFocusMeta" class="summary-box">Sem data aplicada.</div>
          </div>
          <div id="dateFocusGrid" class="date-focus-grid"></div>
          <div id="dateFocusEmpty" class="date-focus-empty" hidden>Nenhum jogo desta data atende aos filtros atuais de competicao ou busca por time.</div>
        </section>

        <section class="card risk-strip">
          <div class="risk-strip-head">
            <div>
              <div class="eyebrow">Jogos por risco</div>
              <h2>Blocos prontos com os jogos que ja se encaixam</h2>
              <p>Cada bloco abaixo organiza automaticamente os confrontos pelas faixas de risco do portal. Assim voce le o contexto do risco e ja abre o jogo certo sem precisar configurar nada antes.</p>
            </div>
            <div id="resultsSummary" class="summary-box">Mostrando todas as competicoes e tabelas disponiveis.</div>
          </div>
          <div class="risk-grid">{risk_blocks_html}</div>
        </section>

        <div class="legend">
          <span class="pill" style="color:#15803d;">Baixo risco</span>
          <span class="pill" style="color:#b45309;">Medio risco</span>
          <span class="pill" style="color:#be123c;">Alto risco</span>
          <span class="pill" style="color:#64748b;">Fora dos criterios</span>
        </div>

        {''.join(sections)}

        <section class="glossary card">
          <div>
            <div class="eyebrow">Guia rapido</div>
            <h2>Glossario de siglas e metricas</h2>
          </div>
          <div class="glossary-grid">
            <div class="glossary-item"><strong>1X2</strong><p>Mercado principal do jogo. 1 = vitoria mandante, X = empate, 2 = vitoria visitante.</p></div>
            <div class="glossary-item"><strong>Odd</strong><p>Cotacao da aposta. Exemplo: odd 2.00 retorna R$ 2,00 para cada R$ 1,00 apostado em retorno bruto.</p></div>
            <div class="glossary-item"><strong>Prob. Modelo</strong><p>Probabilidade estimada pelo modelo para o resultado sugerido na linha.</p></div>
            <div class="glossary-item"><strong>EV</strong><p>Valor esperado da aposta. EV positivo indica vantagem matematica no longo prazo.</p></div>
            <div class="glossary-item"><strong>Casas</strong><p>Quantidade de casas de apostas consideradas na linha de odd usada na comparacao.</p></div>
            <div class="glossary-item"><strong>Stake</strong><p>Valor sugerido para a aposta com base em banca e Kelly fracionado.</p></div>
          </div>
        </section>

        <section class="card launcher-card">
          <div class="controls-head">
            <div>
              <div class="eyebrow">Acoes do portal</div>
              <h2>Central de comandos do painel</h2>
              <p class="copy">Deixamos todos os atalhos operacionais reunidos no final da pagina para manter o topo focado nas leituras e nos jogos.</p>
            </div>
          </div>
          <div class="launcher-grid">
            <article class="launcher-item">
              <strong>Filtros inteligentes</strong>
              <span>Ajuste competicao, risco, time, odds, EV e casas em uma unica janela de configuracao.</span>
              <button id="openFilterModal" class="btn secondary" type="button">Configurar</button>
            </article>
            <article class="launcher-item">
              <strong>Limpeza rapida</strong>
              <span>Remove os filtros ativos e devolve o painel para a leitura completa de todas as competicoes.</span>
              <button id="clearFilterLauncher" class="btn secondary" type="button">Limpar filtros</button>
            </article>
            <article class="launcher-item">
              <strong>Modulo de IA</strong>
              <span>Abra o prompt institucional em modal para atualizar a data, copiar o briefing e executar a leitura quantitativa.</span>
              <button id="openAiPromptModal" class="btn secondary" type="button">Abrir modulo</button>
            </article>
            <article class="launcher-item">
              <strong>Atualizar placares</strong>
              <span>Faz um novo scraping, regenera o portal e recarrega a tela com os resultados e status mais recentes.</span>
              <button id="refreshPortalLauncher" class="btn secondary" type="button">Atualizar agora</button>
            </article>
            <article class="launcher-item">
              <strong>Recarregar pagina</strong>
              <span>Atualiza a visualizacao atual do portal caso queira forcar uma nova leitura do HTML carregado.</span>
              <button id="reloadPageLauncher" class="btn secondary" type="button">Recarregar</button>
            </article>
          </div>
        </section>

        <div class="foot">Aposta envolve risco. O painel ajuda na leitura e comparacao, mas nao elimina perdas nem substitui gestao de banca.</div>
      </main>
    </div>
  </div>

  <div id="matchModal" class="modal-overlay">
    <div class="modal-card">
      <div class="modal-head">
        <div>
          <div id="modalDate" class="eyebrow">00/00/0000</div>
          <h2 id="modalTitle">Time A vs Time B</h2>
        </div>
        <button class="modal-close" onclick="closeMatchDetails()">Ã—</button>
      </div>
      <div class="modal-body">
        <div class="modal-grid">
          <div class="modal-section">
            <h3>Probabilidades 1X2</h3>
            <div class="chart-container"><canvas id="chart1x2"></canvas></div>
          </div>
          <div class="modal-section">
            <h3>Mercados Alternativos</h3>
            <div class="chart-container"><canvas id="chartAlt"></canvas></div>
          </div>
        </div>
        
        <div class="modal-grid">
          <div class="modal-section">
            <h3>Top 5 Placares Provaveis</h3>
            <div class="chart-container"><canvas id="chartScores"></canvas></div>
          </div>
          <div class="modal-section">
            <h3>Estrategia e Contexto</h3>
            <div id="strategyBox" class="summary-box" style="margin-bottom:16px;"></div>
            <div class="context-grid">
              <div class="context-card">
                <h4 id="homeTeamName">Mandante</h4>
                <div class="context-stat">Posicao: <strong id="homeRank">-</strong></div>
                <div class="context-stat">Pontos: <strong id="homePoints">-</strong></div>
                <div class="context-stat">Forma: <strong id="homeForm">-</strong></div>
              </div>
              <div class="context-card">
                <h4 id="awayTeamName">Visitante</h4>
                <div class="context-stat">Posicao: <strong id="awayRank">-</strong></div>
                <div class="context-stat">Pontos: <strong id="awayPoints">-</strong></div>
                <div class="context-stat">Forma: <strong id="awayForm">-</strong></div>
              </div>
            </div>
          </div>
        </div>

        <div class="modal-section">
          <h3>Estatisticas Reais do Jogo</h3>
          <div class="real-stats-grid">
            <div class="real-stat-card">
              <span>Escanteios</span>
              <strong id="realCorners">-</strong>
            </div>
            <div class="real-stat-card">
              <span>Cartoes Amarelos</span>
              <strong id="realYellowCards">-</strong>
            </div>
            <div class="real-stat-card">
              <span>Faltas</span>
              <strong id="realFouls">-</strong>
            </div>
            <div class="real-stat-card">
              <span>Posse de Bola</span>
              <strong id="realPossession">-</strong>
            </div>
          </div>
          <div id="realStatsStatus" class="real-stats-status">Abra um jogo finalizado para consultar cartoes e escanteios reais.</div>
          <a id="realStatsSourceLink" class="btn-link real-stats-link" href="#" target="_blank" rel="noopener noreferrer" hidden>Ver fonte externa</a>
        </div>

        <div class="modal-grid">
          <div class="modal-section">
            <h3>Medias Recentes por Time</h3>
            <div class="context-grid">
              <div class="context-card">
                <h4 id="projectionHomeTeamName">Mandante</h4>
                <div class="context-stat">Esc. a favor: <strong id="homeAvgCornersFor">-</strong></div>
                <div class="context-stat">Esc. cedidos: <strong id="homeAvgCornersAgainst">-</strong></div>
                <div class="context-stat">Cartoes proprios: <strong id="homeAvgYellowFor">-</strong></div>
                <div class="context-stat">Cartoes do rival: <strong id="homeAvgYellowAgainst">-</strong></div>
                <div class="context-stat">Amostra: <strong id="homeProjectionSample">-</strong></div>
              </div>
              <div class="context-card">
                <h4 id="projectionAwayTeamName">Visitante</h4>
                <div class="context-stat">Esc. a favor: <strong id="awayAvgCornersFor">-</strong></div>
                <div class="context-stat">Esc. cedidos: <strong id="awayAvgCornersAgainst">-</strong></div>
                <div class="context-stat">Cartoes proprios: <strong id="awayAvgYellowFor">-</strong></div>
                <div class="context-stat">Cartoes do rival: <strong id="awayAvgYellowAgainst">-</strong></div>
                <div class="context-stat">Amostra: <strong id="awayProjectionSample">-</strong></div>
              </div>
            </div>
          </div>
          <div class="modal-section">
            <h3>Projecao do Confronto</h3>
            <div class="projection-grid">
              <div class="real-stat-card">
                <span>Proj. Escanteios</span>
                <strong id="projectedCorners">-</strong>
              </div>
              <div class="real-stat-card">
                <span>Proj. Cartoes</span>
                <strong id="projectedYellowCards">-</strong>
              </div>
            </div>
            <div id="projectionStatus" class="projection-status">As medias recentes por time serao carregadas junto com os detalhes do jogo.</div>
          </div>
        </div>

        <div class="modal-section">
          <h3>Aderencia ao Real</h3>
          <div class="projection-grid">
            <div class="real-stat-card">
              <span>Escanteios</span>
              <strong id="cornersAccuracyLabel">-</strong>
              <small id="cornersAccuracyDetail">-</small>
            </div>
            <div class="real-stat-card">
              <span>Cartoes</span>
              <strong id="yellowCardsAccuracyLabel">-</strong>
              <small id="yellowCardsAccuracyDetail">-</small>
            </div>
          </div>
          <div id="projectionAccuracyStatus" class="projection-status">A analise de aderencia sera calculada quando houver projecao e estatistica real do jogo.</div>
        </div>
      </div>
    </div>
  </div>

  <div id="filterModal" class="modal-overlay">
    <div class="modal-card filter-modal-card">
      <div class="modal-head">
        <div>
          <div class="eyebrow">Filtros inteligentes</div>
          <h2>Configurar painel</h2>
          <p class="copy">Ajuste a leitura do portal sem deixar esses controles ocupando a tela principal.</p>
        </div>
        <button class="modal-close" type="button" onclick="closeFilterModal()">Ã—</button>
      </div>
      <div class="modal-body">
        <div class="competition-filter-shell">
          <label>Competicao</label>
          <input id="fcomp" type="hidden" value="" />
          <div class="competition-filter-grid">
            <button class="competition-filter-card all-card active" type="button" data-comp-filter="">
              <strong>Todas as competicoes</strong>
              <span>Visao consolidada do portal</span>
              <small>Mostra todos os campeonatos monitorados</small>
            </button>
            {competition_filter_cards}
          </div>
          <div class="hint">Toque em um card para focar em uma competicao ou volte para a visao completa.</div>
        </div>
        <div class="filters">
          <div class="field"><label for="frisk">Perfil de risco</label><select id="frisk"><option>Baixo risco</option><option>Medio risco</option><option>Alto risco</option><option>Personalizado</option></select><div class="hint">Aplica faixas padrao para as tabelas do modelo, sem esconder os resultados finalizados.</div></div>
          <!-- O filtro de data foi movido para a barra lateral a pedido do usuario -->
          <div class="field"><label for="fteam">Time</label><input id="fteam" type="text" placeholder="Ex: Flamengo" /><div class="hint">Busca o nome do time em qualquer tabela visivel.</div></div>
          <div class="field"><label for="fbooks">Casas minimas</label><input id="fbooks" type="number" step="1" min="0" placeholder="8" /><div class="hint">Evita linhas com baixa cobertura de bookmakers nas tabelas do modelo.</div></div>
          <div class="field"><label for="foddmin">Odd minima</label><input id="foddmin" type="number" step="0.01" min="1.01" placeholder="1.30" /><div class="hint">Define a base minima da faixa de odd nas leituras recomendadas.</div></div>
          <div class="field"><label for="foddmax">Odd maxima</label><input id="foddmax" type="number" step="0.01" min="1.01" placeholder="2.20" /><div class="hint">Limita selecoes acima de uma odd alvo nas tabelas do modelo.</div></div>
          <div class="field"><label for="fprobmin">Probabilidade minima</label><input id="fprobmin" type="number" step="0.01" min="0" max="1" placeholder="0.55" /><div class="hint">Usada apenas nas tabelas com leitura do modelo.</div></div>
          <div class="field"><label for="fevmin">EV minimo</label><input id="fevmin" type="number" step="0.005" min="0" max="1" placeholder="0.02" /><div class="hint">Mostra entradas do modelo com vantagem esperada minima.</div></div>
        </div>
        <div class="actions">
          <button id="applyFilter" class="btn primary" type="button">Aplicar filtro</button>
          <button id="clearFilter" class="btn secondary" type="button">Limpar filtros</button>
          <button id="refreshPortalData" class="btn-link" type="button">Atualizar placares</button>
          <button id="reloadPage" class="btn secondary" type="button">Recarregar pagina</button>
          <button class="btn secondary" type="button" onclick="closeFilterModal()">Fechar</button>
        </div>
      </div>
    </div>
  </div>

  <div id="aiPromptModal" class="modal-overlay">
    <div class="modal-card ai-modal-card">
      <div class="modal-head">
        <div>
          <div class="eyebrow">Leitura quantitativa</div>
          <h2>Prompt institucional</h2>
          <p class="copy">Edite o briefing completo, atualize a data quando precisar e execute a leitura sem poluir a tela principal.</p>
        </div>
        <button class="modal-close" type="button" onclick="closeAiPromptModal()">Ã—</button>
      </div>
      <div class="modal-body">
        <div class="ai-module-side">
          <div class="ai-mini-card">
            <label for="aiSelectedDate">Data da analise</label>
            <input id="aiSelectedDate" type="date" />
          </div>
        </div>
        <div class="ai-prompt-head">
          <strong>Prompt pronto para a IA</strong>
          <span>Data atualizada automaticamente</span>
        </div>
        <div id="aiPromptPreview" class="ai-prompt-preview"></div>
        <textarea id="aiPromptArea" class="ai-prompt-area">{ai_prompt_html}</textarea>
        <div class="ai-prompt-actions">
          <button id="modalUpdateAiPrompt" class="btn primary" type="button">Atualizar prompt</button>
          <button id="modalCopyAiPrompt" class="btn secondary" type="button">Copiar prompt</button>
          <button id="modalRunAiPrompt" class="btn primary" type="button">Executar leitura com IA</button>
          <button class="btn secondary" type="button" onclick="closeAiPromptModal()">Fechar</button>
        </div>
        <div id="aiPromptStatus" class="ai-status"></div>
        <div id="aiResponse" class="ai-response">A resposta da IA vai aparecer aqui depois da execucao.</div>
      </div>
    </div>
  </div>

  <div class="floating-actions">
    <button id="scrollToTop" class="btn-float" title="Voltar ao topo">
      <svg viewBox="0 0 24 24"><path d="M12 4l-8 8h16l-8-8z"/></svg>
    </button>
    <button id="scrollToBottom" class="btn-float" title="Ir para o final da pagina">
      <svg viewBox="0 0 24 24"><path d="M12 20l8-8H4l8 8z"/></svg>
    </button>
    <button id="quickRefresh" class="btn-float" title="Atualizar placares em tempo real" type="button">
      <svg viewBox="0 0 24 24"><path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg>
    </button>
  </div>

  <div id="refreshOverlay" class="refresh-overlay" aria-live="polite" aria-busy="true">
    <div class="refresh-overlay-card">
      <div class="refresh-spinner" aria-hidden="true"></div>
      <strong id="refreshOverlayTitle">Atualizando painel</strong>
      <p id="refreshOverlayDetail">Buscando resultados...</p>
      <div class="refresh-progress-track"><i id="refreshOverlayProgressBar"></i></div>
      <p id="refreshOverlayProgressText" class="refresh-progress-label">0%</p>
      <div class="refresh-stage-wrap">
        <p class="refresh-stage-head">Detalhes da automacao</p>
        <ul id="refreshStageList" class="refresh-stage-list">
          <li class="is-pending" data-stage="cache"><i></i><span>Limpando cache local</span></li>
          <li class="is-pending" data-stage="fetch_matches"><i></i><span>Buscando resultados e odds</span></li>
          <li class="is-pending" data-stage="prefetch_real_stats"><i></i><span>Atualizando cartoes e escanteios</span></li>
          <li class="is-pending" data-stage="prepare_frames"><i></i><span>Consolidando dados por competicao</span></li>
          <li class="is-pending" data-stage="build_html"><i></i><span>Regenerando HTML do painel</span></li>
          <li class="is-pending" data-stage="write_file"><i></i><span>Salvando arquivo atualizado</span></li>
          <li class="is-pending" data-stage="done"><i></i><span>Atualizacao concluida</span></li>
        </ul>
      </div>
    </div>
  </div>

  <script>
    const scrollTopBtn = document.getElementById('scrollToTop');
    const scrollBottomBtn = document.getElementById('scrollToBottom');
    const refreshOverlay = document.getElementById('refreshOverlay');
    const refreshOverlayTitle = document.getElementById('refreshOverlayTitle');
    const refreshOverlayDetail = document.getElementById('refreshOverlayDetail');
    const refreshOverlayProgressBar = document.getElementById('refreshOverlayProgressBar');
    const refreshOverlayProgressText = document.getElementById('refreshOverlayProgressText');
    const refreshStageList = document.getElementById('refreshStageList');
    const refreshStageOrder = ['cache', 'fetch_matches', 'prefetch_real_stats', 'prepare_frames', 'build_html', 'write_file', 'done'];
    const refreshAutomationSteps = [
      'Buscando resultados e odds mais recentes...',
      'Consolidando partidas e estatisticas por competicao...',
      'Atualizando cache de cartoes e escanteios...',
      'Regenerando o HTML do painel...',
      'Preparando recarga da tela com os novos dados...'
    ];
    let refreshAutomationTicker = null;
    let refreshAutomationIndex = 0;

    function extractUpdatedAtFromUrl(rawUrl) {{
      try {{
        if (!rawUrl || !/^https?:/i.test(rawUrl)) return '';
        const parsed = new URL(rawUrl);
        return (parsed.searchParams.get('updated_at') || '').trim();
      }} catch (error) {{
        return '';
      }}
    }}

    function syncUpdatedBadgeWithParentQuery() {{
      const target = document.getElementById('portalUpdatedAt');
      if (!target) return;

      const candidates = [];
      try {{
        candidates.push(document.referrer || '');
      }} catch (error) {{}}
      try {{
        candidates.push(window.top && window.top.location && window.top.location.href ? window.top.location.href : '');
      }} catch (error) {{}}
      try {{
        candidates.push(window.parent && window.parent.location && window.parent.location.href ? window.parent.location.href : '');
      }} catch (error) {{}}

      for (const candidate of candidates) {{
        const extracted = extractUpdatedAtFromUrl(candidate);
        if (extracted) {{
          target.textContent = extracted;
          return;
        }}
      }}
    }}

    function getDocumentScrollHeight() {{
      return Math.max(
        document.body.scrollHeight,
        document.documentElement.scrollHeight,
        document.body.offsetHeight,
        document.documentElement.offsetHeight,
        document.body.clientHeight,
        document.documentElement.clientHeight
      );
    }}

    function updateScrollIndicators() {{
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0;
      const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;
      const docHeight = getDocumentScrollHeight();
      const nearTop = scrollTop <= 240;
      const nearBottom = scrollTop + viewportHeight >= docHeight - 240;

      if (scrollTopBtn) {{
        scrollTopBtn.disabled = nearTop;
        scrollTopBtn.setAttribute('aria-disabled', nearTop ? 'true' : 'false');
      }}
      if (scrollBottomBtn) {{
        scrollBottomBtn.disabled = nearBottom;
        scrollBottomBtn.setAttribute('aria-disabled', nearBottom ? 'true' : 'false');
      }}
    }}

    window.addEventListener('scroll', updateScrollIndicators, {{ passive: true }});
    window.addEventListener('resize', updateScrollIndicators);

    if (scrollTopBtn) {{
      scrollTopBtn.onclick = function() {{
        window.scrollTo({{ top: 0, behavior: 'smooth' }});
      }};
    }}

    if (scrollBottomBtn) {{
      scrollBottomBtn.onclick = function() {{
        window.scrollTo({{ top: getDocumentScrollHeight(), behavior: 'smooth' }});
      }};
    }}

    installParentRefreshBridge();
    updateScrollIndicators();
    syncUpdatedBadgeWithParentQuery();

    function stopRefreshAutomationTicker() {{
      if (!refreshAutomationTicker) return;
      window.clearInterval(refreshAutomationTicker);
      refreshAutomationTicker = null;
    }}

    function setRefreshOverlayMessage(title, detail) {{
      if (refreshOverlayTitle && title) refreshOverlayTitle.textContent = title;
      if (refreshOverlayDetail && detail) refreshOverlayDetail.textContent = detail;
    }}

    function setRefreshOverlayProgress(percent) {{
      const bounded = Math.max(0, Math.min(100, Number(percent) || 0));
      if (refreshOverlayProgressBar) {{
        refreshOverlayProgressBar.style.width = bounded.toFixed(0) + '%';
      }}
      if (refreshOverlayProgressText) {{
        refreshOverlayProgressText.textContent = bounded.toFixed(0) + '%';
      }}
    }}

    function setRefreshStageClass(stageElement, className) {{
      if (!stageElement) return;
      stageElement.classList.remove('is-pending', 'is-active', 'is-done', 'is-error');
      stageElement.classList.add(className);
    }}

    function resetRefreshStageChecklist() {{
      if (!refreshStageList) return;
      refreshStageOrder.forEach((stageKey) => {{
        const stageElement = refreshStageList.querySelector('[data-stage=\"' + stageKey + '\"]');
        setRefreshStageClass(stageElement, 'is-pending');
      }});
    }}

    function updateRefreshStageChecklist(stage, hasError) {{
      if (!refreshStageList) return;
      const stageKey = String(stage || '').trim();
      const activeIndex = refreshStageOrder.indexOf(stageKey);
      refreshStageOrder.forEach((key, index) => {{
        const stageElement = refreshStageList.querySelector('[data-stage=\"' + key + '\"]');
        if (!stageElement) return;
        if (hasError) {{
          if (activeIndex >= 0 && index < activeIndex) {{
            setRefreshStageClass(stageElement, 'is-done');
          }} else if (activeIndex >= 0 && index === activeIndex) {{
            setRefreshStageClass(stageElement, 'is-error');
          }} else {{
            setRefreshStageClass(stageElement, 'is-pending');
          }}
          return;
        }}
        if (activeIndex < 0) {{
          setRefreshStageClass(stageElement, 'is-pending');
        }} else if (index < activeIndex) {{
          setRefreshStageClass(stageElement, 'is-done');
        }} else if (index === activeIndex) {{
          setRefreshStageClass(stageElement, stageKey === 'done' ? 'is-done' : 'is-active');
        }} else {{
          setRefreshStageClass(stageElement, 'is-pending');
        }}
      }});
    }}

    function startRefreshAutomationTicker() {{
      stopRefreshAutomationTicker();
      refreshAutomationIndex = 0;
      setRefreshOverlayMessage('Atualizacao em andamento', refreshAutomationSteps[refreshAutomationIndex]);
      setRefreshOverlayProgress(8);
      refreshAutomationTicker = window.setInterval(() => {{
        refreshAutomationIndex = (refreshAutomationIndex + 1) % refreshAutomationSteps.length;
        setRefreshOverlayMessage('Atualizacao em andamento', refreshAutomationSteps[refreshAutomationIndex]);
        setRefreshOverlayProgress(8 + (refreshAutomationIndex * 12));
      }}, 1400);
    }}

    function setRefreshOverlayVisible(isVisible, title, detail, autoCycle) {{
      if (!refreshOverlay) return;
      if (isVisible) {{
        refreshOverlay.classList.add('show');
        document.body.classList.add('refresh-locked');
        if (title || detail) {{
          setRefreshOverlayMessage(title || 'Atualizacao em andamento', detail || refreshAutomationSteps[0]);
        }}
        setRefreshOverlayProgress(0);
        resetRefreshStageChecklist();
        updateRefreshStageChecklist('cache', false);
        if (autoCycle) startRefreshAutomationTicker();
        return;
      }}
      stopRefreshAutomationTicker();
      refreshOverlay.classList.remove('show');
      document.body.classList.remove('refresh-locked');
    }}

    const riskPresets = {{
      "Baixo risco": {{ oddMin: 1.20, oddMax: 1.95, probMin: 0.62, evMin: 0.03, booksMin: 10 }},
      "Medio risco": {{ oddMin: 1.20, oddMax: 2.20, probMin: 0.55, evMin: 0.02, booksMin: 8 }},
      "Alto risco": {{ oddMin: 1.20, oddMax: 2.90, probMin: 0.48, evMin: 0.01, booksMin: 5 }},
    }};

    let charts = {{}};
    let activeMatchStatsRequest = 0;
    const matchDetailsStore = {json.dumps(detail_registry, ensure_ascii=False)};
    const dateMatchCatalog = {match_catalog_json};

    function setRealStatsCardValue(elementId, value) {{
      const element = document.getElementById(elementId);
      if (!element) return;
      element.textContent = value || '-';
    }}

    function formatRealStatPair(metric) {{
      if (!metric || metric.home === undefined || metric.away === undefined) return '-';
      const unit = metric.unit || '';
      return `${{metric.home}}${{unit}} x ${{metric.away}}${{unit}}`;
    }}

    function formatAverageValue(value) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
      return Number(value).toFixed(1);
    }}

    function formatProjectionPair(metric) {{
      if (!metric || metric.home === null || metric.home === undefined || metric.away === null || metric.away === undefined) return '-';
      return `${{Number(metric.home).toFixed(1)}} x ${{Number(metric.away).toFixed(1)}}`;
    }}

    function parseMetricNumber(value) {{
      if (value === null || value === undefined || value === '') return null;
      const parsed = Number(value);
      return Number.isNaN(parsed) ? null : parsed;
    }}

    function clearAccuracyTone(elementId) {{
      const element = document.getElementById(elementId);
      if (!element) return;
      element.classList.remove('accuracy-good', 'accuracy-mid', 'accuracy-low');
    }}

    function setAccuracyCardValue(labelId, detailId, accuracyPayload) {{
      clearAccuracyTone(labelId);
      setRealStatsCardValue(labelId, accuracyPayload ? accuracyPayload.label : '-');
      setRealStatsCardValue(detailId, accuracyPayload ? accuracyPayload.detail : '-');
      if (!accuracyPayload || !accuracyPayload.toneClass) return;
      const labelElement = document.getElementById(labelId);
      if (labelElement) {{
        labelElement.classList.add(accuracyPayload.toneClass);
      }}
    }}

    function classifyProjectionAccuracy(sideError, totalError) {{
      const referenceError = Math.max(sideError, totalError / 2);
      if (referenceError <= 0.8) return {{ label: 'Muito proxima', toneClass: 'accuracy-good' }};
      if (referenceError <= 1.5) return {{ label: 'Proxima', toneClass: 'accuracy-good' }};
      if (referenceError <= 2.5) return {{ label: 'Moderada', toneClass: 'accuracy-mid' }};
      return {{ label: 'Distante', toneClass: 'accuracy-low' }};
    }}

    function buildProjectionAccuracy(metricProjection, metricReal) {{
      const projectedHome = parseMetricNumber(metricProjection && metricProjection.home);
      const projectedAway = parseMetricNumber(metricProjection && metricProjection.away);
      const realHome = parseMetricNumber(metricReal && metricReal.home);
      const realAway = parseMetricNumber(metricReal && metricReal.away);

      if (projectedHome === null || projectedAway === null || realHome === null || realAway === null) return null;

      const projectedTotal = parseMetricNumber(metricProjection && metricProjection.total);
      const finalProjectedTotal = projectedTotal === null ? projectedHome + projectedAway : projectedTotal;
      const realTotal = realHome + realAway;
      const sideError = (Math.abs(projectedHome - realHome) + Math.abs(projectedAway - realAway)) / 2;
      const totalError = Math.abs(finalProjectedTotal - realTotal);
      const accuracy = classifyProjectionAccuracy(sideError, totalError);

      return {{
        label: accuracy.label,
        toneClass: accuracy.toneClass,
        detail: `Proj. ${{formatProjectionPair({{ home: projectedHome, away: projectedAway }})}} | Real ${{formatRealStatPair({{ home: realHome, away: realAway }})}} | Erro ${{sideError.toFixed(1)}} por time`,
        sideError,
        totalError
      }};
    }}

    function getEventSortValue(item) {{
      const timestamp = item && item.event_timestamp ? Date.parse(item.event_timestamp) : NaN;
      return Number.isNaN(timestamp) ? 0 : timestamp;
    }}

    function buildRecentTeamHistory(teamName, currentData) {{
      const currentSortValue = getEventSortValue(currentData) || Number.MAX_SAFE_INTEGER;
      return Object.values(matchDetailsStore)
        .filter((item) => item && item.status === 'Finalizado' && (item.home === teamName || item.away === teamName))
        .filter((item) => {{
          const itemSortValue = getEventSortValue(item);
          if (!itemSortValue || itemSortValue >= currentSortValue) return false;
          return !(item.home === currentData.home && item.away === currentData.away && item.event_timestamp === currentData.event_timestamp);
        }})
        .sort((a, b) => getEventSortValue(b) - getEventSortValue(a))
        .slice(0, 12)
        .map((item) => ({{
          home_team: item.home,
          away_team: item.away,
          status: item.status,
          date_text: item.date_text_raw || item.date || '',
          event_timestamp: item.event_timestamp || ''
        }}));
    }}

    function resetRealStatsBlock(message) {{
      setRealStatsCardValue('realCorners', '-');
      setRealStatsCardValue('realYellowCards', '-');
      setRealStatsCardValue('realFouls', '-');
      setRealStatsCardValue('realPossession', '-');
      const status = document.getElementById('realStatsStatus');
      if (status) {{
        status.textContent = message || 'Abra um jogo finalizado para consultar cartoes e escanteios reais.';
      }}
      const sourceLink = document.getElementById('realStatsSourceLink');
      if (sourceLink) {{
        sourceLink.hidden = true;
        sourceLink.removeAttribute('href');
      }}
    }}

    function resetProjectionBlock(message, accuracyMessage) {{
      document.getElementById('projectionHomeTeamName').textContent = 'Mandante';
      document.getElementById('projectionAwayTeamName').textContent = 'Visitante';
      ['homeAvgCornersFor', 'homeAvgCornersAgainst', 'homeAvgYellowFor', 'homeAvgYellowAgainst', 'homeProjectionSample',
       'awayAvgCornersFor', 'awayAvgCornersAgainst', 'awayAvgYellowFor', 'awayAvgYellowAgainst', 'awayProjectionSample',
       'projectedCorners', 'projectedYellowCards'].forEach((id) => setRealStatsCardValue(id, '-'));
      setAccuracyCardValue('cornersAccuracyLabel', 'cornersAccuracyDetail', null);
      setAccuracyCardValue('yellowCardsAccuracyLabel', 'yellowCardsAccuracyDetail', null);
      const status = document.getElementById('projectionStatus');
      if (status) {{
        status.textContent = message || 'As medias recentes por time serao carregadas junto com os detalhes do jogo.';
      }}
      const accuracyStatus = document.getElementById('projectionAccuracyStatus');
      if (accuracyStatus) {{
        accuracyStatus.textContent = accuracyMessage || 'A analise de aderencia sera calculada quando houver projecao e estatistica real do jogo.';
      }}
    }}

    function applyTeamProjection(data, projectionPayload) {{
      document.getElementById('projectionHomeTeamName').textContent = data.home;
      document.getElementById('projectionAwayTeamName').textContent = data.away;

      const homeProfile = projectionPayload && projectionPayload.home ? projectionPayload.home : null;
      const awayProfile = projectionPayload && projectionPayload.away ? projectionPayload.away : null;
      const projection = projectionPayload && projectionPayload.projection ? projectionPayload.projection : {{}};

      if (homeProfile && homeProfile.available && homeProfile.averages) {{
        setRealStatsCardValue('homeAvgCornersFor', formatAverageValue(homeProfile.averages.corners_for));
        setRealStatsCardValue('homeAvgCornersAgainst', formatAverageValue(homeProfile.averages.corners_against));
        setRealStatsCardValue('homeAvgYellowFor', formatAverageValue(homeProfile.averages.yellow_for));
        setRealStatsCardValue('homeAvgYellowAgainst', formatAverageValue(homeProfile.averages.yellow_against));
        setRealStatsCardValue('homeProjectionSample', String(homeProfile.sample_size || 0));
      }}

      if (awayProfile && awayProfile.available && awayProfile.averages) {{
        setRealStatsCardValue('awayAvgCornersFor', formatAverageValue(awayProfile.averages.corners_for));
        setRealStatsCardValue('awayAvgCornersAgainst', formatAverageValue(awayProfile.averages.corners_against));
        setRealStatsCardValue('awayAvgYellowFor', formatAverageValue(awayProfile.averages.yellow_for));
        setRealStatsCardValue('awayAvgYellowAgainst', formatAverageValue(awayProfile.averages.yellow_against));
        setRealStatsCardValue('awayProjectionSample', String(awayProfile.sample_size || 0));
      }}

      setRealStatsCardValue('projectedCorners', formatProjectionPair(projection.corners));
      setRealStatsCardValue('projectedYellowCards', formatProjectionPair(projection.yellow_cards));

      const status = document.getElementById('projectionStatus');
      if (status) {{
        status.textContent = projectionPayload && projectionPayload.message
          ? projectionPayload.message
          : 'Nao foi possivel calcular medias recentes e projecoes para este confronto.';
      }}
    }}

    function applyProjectionAccuracy(data, statsPayload, projectionPayload) {{
      setAccuracyCardValue('cornersAccuracyLabel', 'cornersAccuracyDetail', null);
      setAccuracyCardValue('yellowCardsAccuracyLabel', 'yellowCardsAccuracyDetail', null);

      const accuracyStatus = document.getElementById('projectionAccuracyStatus');
      if (data.status !== 'Finalizado') {{
        if (accuracyStatus) {{
          accuracyStatus.textContent = 'A aderencia ao real aparece apenas em jogos finalizados.';
        }}
        return;
      }}

      const projection = projectionPayload && projectionPayload.projection ? projectionPayload.projection : {{}};
      const cornersAccuracy = buildProjectionAccuracy(projection.corners, statsPayload && statsPayload.corners ? statsPayload.corners : null);
      const yellowAccuracy = buildProjectionAccuracy(
        projection.yellow_cards,
        statsPayload && statsPayload.yellow_cards ? statsPayload.yellow_cards : null
      );

      setAccuracyCardValue('cornersAccuracyLabel', 'cornersAccuracyDetail', cornersAccuracy);
      setAccuracyCardValue('yellowCardsAccuracyLabel', 'yellowCardsAccuracyDetail', yellowAccuracy);

      if (!accuracyStatus) return;

      const summaries = [];
      if (cornersAccuracy) {{
        summaries.push(`escanteios ${{cornersAccuracy.label.toLowerCase()}} (erro total ${{cornersAccuracy.totalError.toFixed(1)}})`);
      }}
      if (yellowAccuracy) {{
        summaries.push(`cartoes ${{yellowAccuracy.label.toLowerCase()}} (erro total ${{yellowAccuracy.totalError.toFixed(1)}})`);
      }}

      if (summaries.length) {{
        accuracyStatus.textContent = 'Comparacao com o real: ' + summaries.join(' | ') + '.';
        return;
      }}

      if (!statsPayload || (!statsPayload.corners && !statsPayload.yellow_cards)) {{
        accuracyStatus.textContent = 'Nao encontrei estatisticas reais suficientes para comparar a projecao com o jogo.';
        return;
      }}

      accuracyStatus.textContent = projectionPayload && projectionPayload.message
        ? projectionPayload.message
        : 'Nao foi possivel calcular a aderencia entre projecao e real para este jogo.';
    }}

    function applyRealStatsPayload(payload, customMessage) {{
      if (!payload || !payload.stats) return false;
      setRealStatsCardValue('realCorners', formatRealStatPair(payload.stats.corners));
      setRealStatsCardValue('realYellowCards', formatRealStatPair(payload.stats.yellow_cards));
      setRealStatsCardValue('realFouls', formatRealStatPair(payload.stats.fouls));
      setRealStatsCardValue('realPossession', formatRealStatPair(payload.stats.possession));

      const status = document.getElementById('realStatsStatus');
      if (status) {{
        status.textContent = customMessage || payload.message || 'Estatisticas reais carregadas.';
      }}

      const sourceLink = document.getElementById('realStatsSourceLink');
      if (sourceLink) {{
        if (payload.source_url) {{
          sourceLink.href = payload.source_url;
          sourceLink.hidden = false;
        }} else {{
          sourceLink.hidden = true;
          sourceLink.removeAttribute('href');
        }}
      }}
      return true;
    }}

    async function loadRealMatchStats(data, requestId) {{
      const status = document.getElementById('realStatsStatus');
      const apiHost = resolvePortalHost();
      const isStreamlitCloud = isStreamlitCloudRuntime(apiHost);
      const prefetchedRealStats = data && data.prefetched_real_stats && data.prefetched_real_stats.available
        ? data.prefetched_real_stats
        : null;

      if (isStreamlitCloud) {{
        resetRealStatsBlock('Estatisticas reais sob demanda ficam disponiveis apenas no ambiente local do portal.');
        resetProjectionBlock(
          'Medias e projecoes por time ficam disponiveis apenas no ambiente local do portal.',
          'A comparacao com o real fica disponivel apenas no ambiente local do portal.'
        );
        return;
      }}

      if (prefetchedRealStats) {{
        applyRealStatsPayload(prefetchedRealStats, 'Estatisticas reais carregadas do historico salvo no portal.');
      }} else if (data.status === 'Finalizado' && status) {{
        status.textContent = 'Buscando cartoes e escanteios reais do jogo...';
      }} else {{
        resetRealStatsBlock('Estatisticas reais do jogo ficam disponiveis apenas apos o encerramento da partida.');
      }}
      resetProjectionBlock(
        'Calculando medias recentes e projecoes por time...',
        data.status === 'Finalizado'
          ? 'A aderencia da projecao sera calculada quando a consulta terminar.'
          : 'A aderencia ao real aparece apenas em jogos finalizados.'
      );

      try {{
        const response = await fetch('http://' + apiHost + ':8765/api/match-stats', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            home_team: data.home,
            away_team: data.away,
            status: data.status,
            date_text: data.date_text_raw || data.date || '',
            event_timestamp: data.event_timestamp || '',
            home_history: buildRecentTeamHistory(data.home, data),
            away_history: buildRecentTeamHistory(data.away, data)
          }})
        }});
        const payload = await response.json();
        if (requestId !== activeMatchStatsRequest) return;
        if (!response.ok || !payload.ok) {{
          throw new Error(payload.error || 'Falha ao consultar estatisticas reais.');
        }}

        let statsForAccuracy = prefetchedRealStats && prefetchedRealStats.stats ? prefetchedRealStats.stats : null;
        if (payload.available && payload.stats) {{
          applyRealStatsPayload(payload);
          statsForAccuracy = payload.stats;
        }} else if (!prefetchedRealStats) {{
          resetRealStatsBlock(payload.message || 'Nao encontrei estatisticas reais para este jogo.');
        }}

        applyTeamProjection(data, payload.team_projection);
        applyProjectionAccuracy(data, statsForAccuracy, payload.team_projection);
      }} catch (error) {{
        if (requestId !== activeMatchStatsRequest) return;
        if (!prefetchedRealStats) {{
          resetRealStatsBlock('Nao foi possivel carregar cartoes e escanteios reais agora.');
        }} else if (status) {{
          status.textContent = 'Estatisticas reais carregadas do historico salvo no portal. Nao foi possivel atualizar a consulta externa agora.';
        }}
        resetProjectionBlock(
          'Nao foi possivel calcular medias recentes e projecoes por time agora.',
          'Nao foi possivel comparar a projecao com o real agora.'
        );
      }}
    }}

    function showMatchDetails(btn) {{
      const detailKey = btn.getAttribute('data-detail-key');
      if (!detailKey) return;
      const data = matchDetailsStore[detailKey];
      if (!data || Object.keys(data).length === 0) return;
      activeMatchStatsRequest += 1;
      
      document.getElementById('modalTitle').textContent = data.home + ' x ' + data.away;
      const modalMeta = [];
      if (data.status) modalMeta.push(data.status);
      if (data.date) modalMeta.push(data.date);
      if (data.final_score) modalMeta.push('Placar ' + data.final_score);
      document.getElementById('modalDate').textContent = modalMeta.join(' • ');
      
      const stratBox = document.getElementById('strategyBox');
      if (data.tip) {{
        const riskLine = data.model_risk_stage ? `<br><strong>Estagio de risco:</strong> ${{data.model_risk_stage}}` : '';
        const riskContext = data.model_risk_context ? `<br><strong>Contexto do risco:</strong> ${{data.model_risk_context}}` : '';
        stratBox.innerHTML = `<strong>Sugestao:</strong> ${{data.tip.market}}<br>
          <strong>Odd:</strong> ${{data.tip.odd.toFixed(2)}} | <strong>Prob:</strong> ${{data.tip.prob}}% | <strong>EV:</strong> ${{data.tip.ev}}%<br>
          <strong>Stake Sugerida:</strong> R$ ${{data.tip.stake.toFixed(2)}}${{riskLine}}${{riskContext}}`;
      }} else if (data.status === 'Finalizado' && data.model_result) {{
        const actualLine = data.actual_result && data.actual_result !== '-' ? `<br><strong>Resultado real:</strong> ${{data.actual_result}}` : '';
        const hitLine = data.model_hit ? ` | <strong>Modelo:</strong> ${{data.model_hit}}` : '';
        const riskLine = data.model_risk_stage ? `<br><strong>Estagio de risco:</strong> ${{data.model_risk_stage}}` : '';
        const riskContext = data.model_risk_context ? `<br><strong>Contexto do risco:</strong> ${{data.model_risk_context}}` : '';
        stratBox.innerHTML = `<strong>Leitura do modelo:</strong> ${{data.model_result}}<br>
          <strong>Prob. Modelo:</strong> ${{data.model_probability}}%${{hitLine}}${{riskLine}}${{actualLine}}${{riskContext}}`;
      }} else if (data.model_result) {{
        const riskLine = data.model_risk_stage ? `<br><strong>Estagio de risco:</strong> ${{data.model_risk_stage}}` : '';
        const riskContext = data.model_risk_context ? `<br><strong>Contexto do risco:</strong> ${{data.model_risk_context}}` : '';
        stratBox.innerHTML = `<strong>Leitura do modelo:</strong> ${{data.model_result}}<br>
          <strong>Prob. Modelo:</strong> ${{data.model_probability}}%${{riskLine}}${{riskContext}}`;
      }} else {{
        stratBox.innerHTML = 'Sem recomendacao de entrada para este jogo com base nos criterios do modelo.';
      }}
      
      document.getElementById('homeTeamName').textContent = data.home;
      document.getElementById('homeRank').textContent = data.context.home.rank || '-';
      document.getElementById('homePoints').textContent = data.context.home.points;
      document.getElementById('homeForm').textContent = data.context.home.recent_text;
      
      document.getElementById('awayTeamName').textContent = data.away;
      document.getElementById('awayRank').textContent = data.context.away.rank || '-';
      document.getElementById('awayPoints').textContent = data.context.away.points;
      document.getElementById('awayForm').textContent = data.context.away.recent_text;
      
      const matchModal = document.getElementById('matchModal');
      matchModal.style.display = 'block';
      positionVisibleModal('matchModal');
      resetRealStatsBlock(
        data.status === 'Finalizado'
          ? 'Buscando cartoes e escanteios reais do jogo...'
          : 'Estatisticas reais do jogo ficam disponiveis apenas apos o encerramento da partida.'
      );
      resetProjectionBlock(
        'Calculando medias recentes e projecoes por time...',
        data.status === 'Finalizado'
          ? 'A aderencia da projecao sera calculada quando a consulta terminar.'
          : 'A aderencia ao real aparece apenas em jogos finalizados.'
      );
      loadRealMatchStats(data, activeMatchStatsRequest);
      requestAnimationFrame(() => renderCharts(data));
    }}

    function closeMatchDetails() {{
      activeMatchStatsRequest += 1;
      destroyCharts();
      resetRealStatsBlock();
      resetProjectionBlock();
      document.getElementById('matchModal').style.display = 'none';
    }}

    function openFilterModal() {{
      const modal = document.getElementById('filterModal');
      modal.style.display = 'block';
      positionVisibleModal('filterModal');
    }}

    function closeFilterModal() {{
      document.getElementById('filterModal').style.display = 'none';
    }}

    function showSelectedMatch(selectId) {{
      const select = document.getElementById(selectId);
      if (!select) return;
      const option = select.options[select.selectedIndex];
      if (!option) return;
      const detailKey = option.getAttribute('data-detail-key');
      if (!detailKey) return;

      const tempButton = document.createElement('button');
      tempButton.setAttribute('data-detail-key', detailKey);
      showMatchDetails(tempButton);
    }}

    function openAiPromptModal() {{
      const modal = document.getElementById('aiPromptModal');
      modal.style.display = 'block';
      positionVisibleModal('aiPromptModal');
    }}

    function closeAiPromptModal() {{
      document.getElementById('aiPromptModal').style.display = 'none';
    }}

    function getVisibleModalOffset() {{
      try {{
        if (window.frameElement && window.frameElement.getBoundingClientRect) {{
          const frameRect = window.frameElement.getBoundingClientRect();
          return Math.max(24, Math.round(-frameRect.top + 24));
        }}
      }} catch (error) {{
        // Ignora restricoes de contexto e usa o fallback padrao.
      }}
      return 24;
    }}

    function positionVisibleModal(modalId) {{
      const modal = document.getElementById(modalId);
      if (!modal) return;
      const card = modal.querySelector('.modal-card');
      if (!card) return;
      const topOffset = getVisibleModalOffset();
      card.style.marginTop = topOffset + 'px';
      card.style.marginBottom = '24px';
    }}

    function destroyCharts() {{
      if (charts.chart1x2) {{
        charts.chart1x2.destroy();
        charts.chart1x2 = null;
      }}
      if (charts.chartAlt) {{
        charts.chartAlt.destroy();
        charts.chartAlt = null;
      }}
      if (charts.chartScores) {{
        charts.chartScores.destroy();
        charts.chartScores = null;
      }}
    }}

    function resetChartCanvas(canvasId) {{
      const currentCanvas = document.getElementById(canvasId);
      if (!currentCanvas || !currentCanvas.parentNode) return currentCanvas;
      const nextCanvas = document.createElement('canvas');
      nextCanvas.id = canvasId;
      currentCanvas.parentNode.replaceChild(nextCanvas, currentCanvas);
      return nextCanvas;
    }}

    function renderCharts(data) {{
      destroyCharts();
      
      const ctx1 = resetChartCanvas('chart1x2').getContext('2d');
      charts.chart1x2 = new Chart(ctx1, {{
        type: 'bar',
        data: {{
          labels: ['Mandante', 'Empate', 'Visitante'],
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
      
      const ctx2 = resetChartCanvas('chartAlt').getContext('2d');
      charts.chartAlt = new Chart(ctx2, {{
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
      
      const ctx3 = resetChartCanvas('chartScores').getContext('2d');
      charts.chartScores = new Chart(ctx3, {{
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
    }}

    function applyRiskPreset() {{
      const risk = document.getElementById('frisk').value;
      if (risk === 'Personalizado') return;
      const preset = riskPresets[risk];
      if (!preset) return;
      document.getElementById('foddmin').value = preset.oddMin.toFixed(2);
      document.getElementById('foddmax').value = preset.oddMax.toFixed(2);
      document.getElementById('fprobmin').value = preset.probMin.toFixed(2);
      document.getElementById('fevmin').value = preset.evMin.toFixed(3);
      document.getElementById('fbooks').value = String(preset.booksMin);
    }}

    function applyRiskProfile(risk) {{
      const riskField = document.getElementById('frisk');
      if (!riskField) return;
      riskField.value = risk;
      applyRiskPreset();
      applyFilters();
      const summary = document.getElementById('resultsSummary');
      if (summary) {{
        summary.textContent = 'Perfil ' + risk.toLowerCase() + ' aplicado com os filtros padrao do portal.';
      }}
    }}

    function resetPortalFilters() {{
      document.getElementById('fcomp').value = '';
      document.getElementById('frisk').value = 'Baixo risco';
      document.getElementById('fdate').value = '';
      document.getElementById('fteam').value = '';
      document.getElementById('foddmin').value = '';
      document.getElementById('foddmax').value = '';
      document.getElementById('fprobmin').value = '';
      document.getElementById('fevmin').value = '';
      document.getElementById('fbooks').value = '';
      applyRiskPreset();
      applyFilters();
    }}

    function renderDateFocusMatches() {{
      const selectedDate = document.getElementById('fdate').value;
      const panel = document.getElementById('dateFocusPanel');
      const grid = document.getElementById('dateFocusGrid');
      const empty = document.getElementById('dateFocusEmpty');
      const title = document.getElementById('dateFocusTitle');
      const copy = document.getElementById('dateFocusCopy');
      const meta = document.getElementById('dateFocusMeta');
      const competition = (document.getElementById('fcomp').value || '').trim().toLowerCase();
      const team = (document.getElementById('fteam').value || '').trim().toLowerCase();

      if (!selectedDate) {{
        panel.hidden = true;
        grid.innerHTML = '';
        empty.hidden = true;
        return;
      }}

      const matches = dateMatchCatalog
        .filter((item) => item && item.filter_date === selectedDate)
        .filter((item) => !competition || (item.competition || '').trim().toLowerCase() === competition)
        .filter((item) => {{
          if (!team) return true;
          const haystack = `${{item.competition || ''}} ${{item.matchup || ''}} ${{item.home || ''}} ${{item.away || ''}}`.toLowerCase();
          return haystack.includes(team);
        }})
        .sort((a, b) => getEventSortValue(matchDetailsStore[a.detail_key] || a) - getEventSortValue(matchDetailsStore[b.detail_key] || b));

      panel.hidden = false;
      title.textContent = 'Jogos de ' + formatSelectedDate(selectedDate);
      copy.textContent = 'Todos os confrontos desta data aparecem aqui mesmo quando nao entram nas faixas principais de risco. Abra a analise para ver leitura, placares provaveis e contexto.';
      meta.textContent = matches.length + (matches.length === 1 ? ' jogo reunido' : ' jogos reunidos') + ' com analise direta.';

      if (!matches.length) {{
        grid.innerHTML = '';
        empty.hidden = false;
        return;
      }}

      empty.hidden = true;
      grid.innerHTML = matches.map((item) => {{
        const detail = matchDetailsStore[item.detail_key] || {{}};
        const probability = Number(item.model_probability || 0);
        const tipLine = detail.tip
          ? `Sugestao do modelo: ${{detail.tip.market}} | Odd ${{detail.tip.odd.toFixed(2)}} | EV ${{detail.tip.ev}}%`
          : `Leitura principal: ${{item.model_result || 'Sem leitura'}}${{probability ? ' | ' + probability.toFixed(1) + '%' : ''}}`;
        const statusLabel = item.status || 'Agendado';
        const matchupLabel = item.matchup || ((item.home || '') + ' x ' + (item.away || ''));
        return `
          <article class="date-match-card">
            <div class="date-match-top">
              <div>
                <span class="eyebrow">${{item.competition || 'Competicao'}}</span>
                <strong>${{matchupLabel}}</strong>
              </div>
              <div class="date-match-meta">
                <span>${{statusLabel}}</span>
                <span>${{item.date_label || formatSelectedDate(selectedDate)}}</span>
              </div>
            </div>
            <div class="date-match-note">${{item.model_risk_stage || 'Fora dos criterios'}}</div>
            <div class="date-match-copy">${{tipLine}}</div>
            <div class="date-match-actions">
              <button class="btn secondary" type="button" data-detail-key="${{item.detail_key}}" onclick="showMatchDetails(this)">Abrir analise</button>
            </div>
          </article>
        `;
      }}).join('');
    }}

    function classifyRowsByRisk() {{
      const rows = Array.from(document.querySelectorAll('tr[data-odd]'));
      rows.forEach(row => {{
        row.classList.remove('risk-low', 'risk-med', 'risk-high', 'risk-none');
        const odd = parseFloat(row.getAttribute('data-odd') || '');
        const prob = parseFloat(row.getAttribute('data-prob') || '');
        const ev = parseFloat(row.getAttribute('data-ev') || '');
        const books = parseInt(row.getAttribute('data-books') || '', 10);
        if (Number.isNaN(odd) || Number.isNaN(prob) || Number.isNaN(ev) || Number.isNaN(books)) {{
          row.classList.add('risk-none');
          return;
        }}
        if (prob >= 0.62 && ev >= 0.03 && odd <= 1.95 && books >= 10) row.classList.add('risk-low');
        else if (prob >= 0.55 && ev >= 0.02 && odd <= 2.20 && books >= 8) row.classList.add('risk-med');
        else if (prob >= 0.48 && ev >= 0.01 && odd <= 2.90 && books >= 5) row.classList.add('risk-high');
        else row.classList.add('risk-none');
      }});
    }}

    function updateResultsSummary(shownCards, visibleRows) {{
      const competition = document.getElementById('fcomp').value || 'todas as competicoes';
      const risk = document.getElementById('frisk').value;
      const selectedDate = document.getElementById('fdate').value;
      const team = (document.getElementById('fteam').value || '').trim();
      const summary = document.getElementById('resultsSummary');
      const parts = [
        shownCards + (shownCards === 1 ? ' competicao visivel' : ' competicoes visiveis'),
        visibleRows + (visibleRows === 1 ? ' linha encontrada' : ' linhas encontradas'),
        'perfil ' + risk.toLowerCase(),
      ];
      if (team) parts.push('busca por "' + team + '"');
      if (selectedDate) parts.push('data ' + formatSelectedDate(selectedDate));
      summary.textContent = 'Mostrando ' + competition + ': ' + parts.join(' • ') + '.';
    }}

    function updateCompetitionNavState() {{
      const activeCompetition = (document.getElementById('fcomp').value || '').trim().toLowerCase();
      const navLinks = Array.from(document.querySelectorAll('.jump-link[data-comp-name], .rail-link[data-comp-name]'));
      navLinks.forEach((link) => {{
        const linkCompetition = (link.getAttribute('data-comp-name') || '').trim().toLowerCase();
        link.classList.toggle('active', !!activeCompetition && linkCompetition === activeCompetition);
      }});
      const filterCards = Array.from(document.querySelectorAll('.competition-filter-card[data-comp-filter]'));
      filterCards.forEach((card) => {{
        const cardCompetition = (card.getAttribute('data-comp-filter') || '').trim().toLowerCase();
        if (!activeCompetition) {{
          card.classList.toggle('active', cardCompetition === '');
        }} else {{
          card.classList.toggle('active', cardCompetition === activeCompetition);
        }}
      }});
    }}

    function toggleCompetitionFilter(compName) {{
      const competitionField = document.getElementById('fcomp');
      if (!competitionField) return;
      const normalized = (compName || '').trim();
      competitionField.value = competitionField.value === normalized ? '' : normalized;
      applyFilters();
    }}

    const aiPromptTemplate = {ai_prompt_js!r};

    function formatSelectedDate(value) {{
      if (!value) return 'data nao definida';
      const parts = value.split('-');
      if (parts.length !== 3) return value;
      return parts[2] + '/' + parts[1] + '/' + parts[0];
    }}

    function updateAiPrompt() {{
      const dateValue = document.getElementById('aiSelectedDate').value;
      const promptArea = document.getElementById('aiPromptArea');
      const prompt = aiPromptTemplate.replace('__DATA_SELECIONADA__', formatSelectedDate(dateValue));
      promptArea.value = prompt;
      refreshAiPromptPreview();
      document.getElementById('aiPromptStatus').textContent = 'Prompt atualizado para a data selecionada.';
    }}

    function refreshAiPromptPreview() {{
      const promptArea = document.getElementById('aiPromptArea');
      const preview = document.getElementById('aiPromptPreview');
      if (!promptArea || !preview) return;
      const lines = promptArea.value.split(/\\r?\\n/).map(line => line.trim()).filter(Boolean);
      preview.textContent = lines.length ? lines.slice(0, 8).join('\\n') : 'Nenhum prompt carregado.';
    }}

    function setRefreshButtonsLoading(isLoading) {{
      ['refreshPortalData', 'refreshPortalLauncher', 'quickRefresh'].forEach((id) => {{
        const element = document.getElementById(id);
        if (!element) return;
        element.disabled = isLoading;
        if (id === 'quickRefresh') {{
          element.classList.toggle('loading', isLoading);
        }}
      }});
    }}

    function resolvePortalHost() {{
      const candidates = [];
      try {{
        if (window.location && window.location.hostname) candidates.push(window.location.hostname);
      }} catch (error) {{}}
      try {{
        if (window.top && window.top.location && window.top.location.hostname) candidates.push(window.top.location.hostname);
      }} catch (error) {{}}
      try {{
        if (window.parent && window.parent.location && window.parent.location.hostname) candidates.push(window.parent.location.hostname);
      }} catch (error) {{}}
      try {{
        if (document.referrer && /^https?:/i.test(document.referrer)) {{
          const refUrl = new URL(document.referrer);
          if (refUrl.hostname) candidates.push(refUrl.hostname);
        }}
      }} catch (error) {{}}
      for (const host of candidates) {{
        const cleanHost = String(host || '').trim();
        if (cleanHost && cleanHost !== 'about:srcdoc') return cleanHost;
      }}
      return '127.0.0.1';
    }}

    function installParentRefreshBridge() {{
      try {{
        const parentWindow = window.parent;
        if (!parentWindow || parentWindow === window) return;
        if (parentWindow.__fdPortalRefreshBridgeInstalled) return;
        parentWindow.__fdPortalRefreshBridgeInstalled = true;
        parentWindow.addEventListener('message', (event) => {{
          const payload = event && event.data ? event.data : null;
          if (!payload || payload.type !== 'fd-portal-refresh') return;
          const targetUrl = String(payload.url || '').trim();
          if (!/^https?:\\/\\//i.test(targetUrl)) return;
          try {{
            parentWindow.location.assign(targetUrl);
          }} catch (error) {{
            parentWindow.location.href = targetUrl;
          }}
        }});
      }} catch (error) {{}}
    }}

    function isStreamlitCloudRuntime(apiHost) {{
      const host = String(apiHost || '').toLowerCase();
      if (host.includes('streamlit.app')) return true;

      const probes = [];
      try {{
        probes.push(document.referrer || '');
      }} catch (error) {{}}
      try {{
        probes.push(window.location && window.location.href ? window.location.href : '');
      }} catch (error) {{}}
      try {{
        probes.push(window.top && window.top.location && window.top.location.href ? window.top.location.href : '');
      }} catch (error) {{}}

      for (const probe of probes) {{
        if (String(probe || '').toLowerCase().includes('streamlit.app')) {{
          return true;
        }}
      }}
      return false;
    }}

    function reloadPortalShell() {{
      try {{
        if (window.top && window.top !== window) {{
          window.top.location.reload();
          return;
        }}
      }} catch (error) {{}}
      window.location.reload();
    }}

    function requestStreamlitPortalRefresh() {{
      const candidates = [];
      try {{
        if (document.referrer && /^https?:/i.test(document.referrer)) {{
          candidates.push(document.referrer);
        }}
      }} catch (error) {{}}
      try {{
        if (window.top && window.top.location && /^https?:/i.test(window.top.location.href)) {{
          candidates.push(window.top.location.href);
        }}
      }} catch (error) {{}}
      try {{
        if (window.parent && window.parent.location && /^https?:/i.test(window.parent.location.href)) {{
          candidates.push(window.parent.location.href);
        }}
      }} catch (error) {{}}
      try {{
        if (window.location && /^https?:/i.test(window.location.href)) {{
          candidates.push(window.location.href);
        }}
      }} catch (error) {{}}

      let targetUrl = null;
      for (const candidate of candidates) {{
        try {{
          targetUrl = new URL(candidate);
          break;
        }} catch (error) {{}}
      }}
      if (!targetUrl) {{
        targetUrl = new URL('https://' + resolvePortalHost() + '/');
      }}

      targetUrl.searchParams.set('view', 'portal');
      targetUrl.searchParams.set('refresh_portal', '1');
      targetUrl.searchParams.set('refresh_nonce', String(Date.now()));
      const finalUrl = targetUrl.toString();
      try {{
        const directLink = document.createElement('a');
        directLink.href = finalUrl;
        directLink.target = '_top';
        directLink.rel = 'noopener';
        directLink.style.display = 'none';
        document.body.appendChild(directLink);
        directLink.click();
        window.setTimeout(() => directLink.remove(), 250);
      }} catch (error) {{}}
      try {{
        if (window.parent) {{
          window.parent.postMessage({{ type: 'fd-portal-refresh', url: finalUrl }}, '*');
        }}
      }} catch (error) {{}}
      try {{
        if (window.parent && window.parent !== window) {{
          window.parent.location.assign(finalUrl);
          return;
        }}
      }} catch (error) {{}}
      try {{
        window.open(finalUrl, '_top');
        return;
      }} catch (error) {{}}
      try {{
        window.location.assign(finalUrl);
        return;
      }} catch (error) {{}}
      window.location.href = finalUrl;
    }}

    function waitMs(ms) {{
      return new Promise((resolve) => window.setTimeout(resolve, ms));
    }}

    async function startPortalRefreshJob(apiHost) {{
      const response = await fetch('http://' + apiHost + ':8765/api/refresh-portal/start', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ source: 'portal-ui' }})
      }});
      const data = await response.json();
      if (!response.ok || !data.ok || !data.job_id) {{
        throw new Error(data.error || 'Nao foi possivel iniciar o job de atualizacao.');
      }}
      return String(data.job_id);
    }}

    async function fetchPortalRefreshStatus(apiHost, jobId) {{
      const response = await fetch('http://' + apiHost + ':8765/api/refresh-portal/status?job_id=' + encodeURIComponent(jobId));
      const data = await response.json();
      if (!response.ok || !data.ok || !data.job) {{
        throw new Error(data.error || 'Nao foi possivel consultar o status da atualizacao.');
      }}
      return data.job;
    }}

    async function refreshPortalData() {{
      const summary = document.getElementById('resultsSummary');
      const apiHost = resolvePortalHost();
      const isStreamlitCloud = isStreamlitCloudRuntime(apiHost);
      const isStaticPortal = window.location.port === '8000' || window.location.pathname.toLowerCase().endsWith('/index.html');
      let keepOverlayVisible = false;
      let lastKnownStage = 'cache';

      if (summary) {{
        summary.textContent = 'Atualizando placares e reprocessando o portal...';
      }}
      setRefreshOverlayVisible(true, 'Atualizacao em andamento', 'Buscando resultados...', true);

      if (isStreamlitCloud) {{
        keepOverlayVisible = true;
        setRefreshOverlayMessage('Atualizacao em andamento', 'Solicitando atualizacao no backend do Streamlit...');
        if (summary) {{
          summary.textContent = 'Solicitando atualizacao no backend do Streamlit e recarregando o portal...';
        }}
        setRefreshButtonsLoading(true);
        requestStreamlitPortalRefresh();
        window.setTimeout(() => {{
          if (document.visibilityState !== 'hidden') {{
            stopRefreshAutomationTicker();
            setRefreshOverlayVisible(false);
            setRefreshButtonsLoading(false);
            if (summary) {{
              summary.textContent = 'Atualizacao solicitada. Se a tela nao recarregar em alguns segundos, atualize o navegador (F5).';
            }}
          }}
        }}, 6000);
        return;
      }}

      setRefreshButtonsLoading(true);
      try {{
        const jobId = await startPortalRefreshJob(apiHost);
        stopRefreshAutomationTicker();
        setRefreshOverlayMessage('Atualizacao em andamento', 'Buscando resultados e odds mais recentes...');
        setRefreshOverlayProgress(6);
        updateRefreshStageChecklist('cache', false);

        const monitorStart = Date.now();
        let finalJob = null;
        while (Date.now() - monitorStart <= 300000) {{
          const job = await fetchPortalRefreshStatus(apiHost, jobId);
          finalJob = job;
          if (job.stage) {{
            lastKnownStage = String(job.stage);
            updateRefreshStageChecklist(lastKnownStage, false);
          }}
          if (typeof job.progress === 'number') {{
            setRefreshOverlayProgress(job.progress);
          }}
          if (job.message) {{
            setRefreshOverlayMessage('Atualizacao em andamento', String(job.message));
          }}
          if (job.done) {{
            break;
          }}
          await waitMs(900);
        }}

        if (!finalJob) {{
          throw new Error('Nao foi possivel acompanhar o status da atualizacao.');
        }}
        if (!finalJob.done) {{
          throw new Error('Tempo limite ao atualizar o painel.');
        }}
        if (!finalJob.ok) {{
          throw new Error(finalJob.error || finalJob.message || 'Falha ao atualizar os placares.');
        }}

        const result = finalJob.result || {{}};
        updateRefreshStageChecklist('done', false);
        if (summary) {{
          const report = result.real_stats_report || null;
          const statsLine = report
            ? ' Cartoes/escanteios historicos: ' + (report.available_total || 0) + ' jogos salvos, ' + (report.saved_now || 0) + ' novos nesta atualizacao.'
            : '';
          summary.textContent = 'Placares atualizados em ' + (result.updated_at || 'agora') + '.' + statsLine + ' Recarregando o painel...';
          setRefreshOverlayProgress(100);
          setRefreshOverlayMessage('Atualizacao concluida', 'Placares atualizados em ' + (result.updated_at || 'agora') + '.' + statsLine + ' Recarregando o painel...');
        }}
        keepOverlayVisible = true;
        window.setTimeout(() => reloadPortalShell(), 700);
      }} catch (error) {{
        stopRefreshAutomationTicker();
        const message = error && error.message ? error.message : 'falha desconhecida.';
        if (summary) {{
          summary.textContent = 'Nao foi possivel atualizar os placares: ' + message;
        }}
        updateRefreshStageChecklist(lastKnownStage || 'fetch_matches', true);
        setRefreshOverlayProgress(100);
        setRefreshOverlayMessage('Falha na atualizacao', 'A automacao encontrou um erro: ' + message);
        if (!isStaticPortal) {{
          keepOverlayVisible = true;
          window.setTimeout(() => reloadPortalShell(), 700);
        }}
      }} finally {{
        window.setTimeout(() => setRefreshButtonsLoading(false), 150);
        if (!keepOverlayVisible) {{
          window.setTimeout(() => setRefreshOverlayVisible(false), 1400);
        }}
      }}
    }}

    async function copyAiPrompt() {{
      const promptArea = document.getElementById('aiPromptArea');
      const status = document.getElementById('aiPromptStatus');
      promptArea.select();
      promptArea.setSelectionRange(0, promptArea.value.length);
      try {{
        await navigator.clipboard.writeText(promptArea.value);
        status.textContent = 'Prompt copiado com sucesso.';
      }} catch (error) {{
        document.execCommand('copy');
        status.textContent = 'Prompt copiado.';
      }}
    }}

    async function runAiPrompt() {{
      const selectedDate = document.getElementById('aiSelectedDate').value;
      const promptArea = document.getElementById('aiPromptArea');
      const status = document.getElementById('aiPromptStatus');
      const responseBox = document.getElementById('aiResponse');
      const apiHost = resolvePortalHost();
      const isStreamlitCloud = isStreamlitCloudRuntime(apiHost);
      if (!selectedDate) {{
        status.textContent = 'Selecione uma data antes de executar a leitura.';
        return;
      }}
      if (isStreamlitCloud) {{
        responseBox.textContent = 'No deploy web, use o modulo IA Institucional do proprio portal Streamlit.';
        status.textContent = 'A API local da IA nao fica exposta na Streamlit Cloud.';
        return;
      }}

      status.textContent = 'Consultando a IA local...';
      responseBox.textContent = 'Processando leitura quantitativa, aguarde...';
      try {{
        const response = await fetch('http://' + apiHost + ':8765/api/ai-analysis', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            selected_date: selectedDate,
            prompt: promptArea.value
          }})
        }});
        const data = await response.json();
        if (!response.ok || !data.ok) {{
          throw new Error(data.error || 'Falha ao consultar a IA.');
        }}
        responseBox.textContent = data.analysis || 'A IA nao retornou texto.';
        status.textContent = 'Leitura concluida com sucesso.';
      }} catch (error) {{
        responseBox.textContent = 'Nao foi possivel gerar a leitura agora.';
        status.textContent = 'Erro: ' + (error && error.message ? error.message : 'falha desconhecida.');
      }}
    }}

    function applyFilters() {{
      const comp = (document.getElementById('fcomp').value || '').toLowerCase();
      const selectedDate = document.getElementById('fdate').value || '';
      const team = (document.getElementById('fteam').value || '').toLowerCase().trim();
      const oddMinRaw = document.getElementById('foddmin').value;
      const oddMaxRaw = document.getElementById('foddmax').value;
      const probMinRaw = document.getElementById('fprobmin').value;
      const evMinRaw = document.getElementById('fevmin').value;
      const booksMinRaw = document.getElementById('fbooks').value;
      const oddMin = oddMinRaw ? parseFloat(oddMinRaw) : null;
      const oddMax = oddMaxRaw ? parseFloat(oddMaxRaw) : null;
      const probMin = probMinRaw ? parseFloat(probMinRaw) : null;
      const evMin = evMinRaw ? parseFloat(evMinRaw) : null;
      const booksMin = booksMinRaw ? parseInt(booksMinRaw, 10) : null;
      const cards = Array.from(document.querySelectorAll('.card[data-comp]'));
      let shownCards = 0;
      let visibleRows = 0;

      cards.forEach(card => {{
        const cardComp = (card.getAttribute('data-comp') || '').toLowerCase();
        const showCard = !comp || cardComp === comp;
        const emptyState = card.querySelector('.empty-state');
        card.style.display = showCard ? '' : 'none';
        if (!showCard) {{
          if (emptyState) emptyState.hidden = true;
          return;
        }}

        shownCards += 1;
        const rows = Array.from(card.querySelectorAll('tbody tr'));
        rows.forEach(row => {{
          const txt = (row.textContent || '').toLowerCase();
          const rowDate = row.getAttribute('data-date') || '';
          const teamOk = !team || txt.includes(team);
          const dateOk = !selectedDate || rowDate === selectedDate;
          let oddOk = true;
          const filterScope = row.getAttribute('data-filter-scope') || 'general';
          const oneOdd = row.getAttribute('data-odd');
          if (filterScope === 'model' && oneOdd) {{
            const value = parseFloat(oneOdd);
            if (!Number.isNaN(value)) {{
              if (oddMin !== null && value < oddMin) oddOk = false;
              if (oddMax !== null && value > oddMax) oddOk = false;
            }}
            const rowProb = parseFloat(row.getAttribute('data-prob') || '');
            const rowEv = parseFloat(row.getAttribute('data-ev') || '');
            const rowBooks = parseInt(row.getAttribute('data-books') || '', 10);
            if (probMin !== null && !Number.isNaN(rowProb) && rowProb < probMin) oddOk = false;
            if (evMin !== null && !Number.isNaN(rowEv) && rowEv < evMin) oddOk = false;
            if (booksMin !== null && !Number.isNaN(rowBooks) && rowBooks < booksMin) oddOk = false;
          }}
          row.style.display = teamOk && oddOk && dateOk ? '' : 'none';
        }});

        const cardVisibleRows = rows.filter(row => row.style.display !== 'none').length;
        if (emptyState) emptyState.hidden = cardVisibleRows !== 0;
        visibleRows += cardVisibleRows;
      }});

      classifyRowsByRisk();
      renderDateFocusMatches();
      updateResultsSummary(shownCards, visibleRows);
      updateCompetitionNavState();
    }}

    document.getElementById('openFilterModal').addEventListener('click', openFilterModal);
    document.getElementById('applyFilter').addEventListener('click', applyFilters);
    document.querySelectorAll('.competition-filter-card[data-comp-filter]').forEach((card) => {{
      card.addEventListener('click', () => toggleCompetitionFilter(card.getAttribute('data-comp-filter') || ''));
    }});
    document.getElementById('fdate').addEventListener('change', applyFilters);
    document.getElementById('fteam').addEventListener('input', applyFilters);
    document.getElementById('frisk').addEventListener('change', () => {{ applyRiskPreset(); applyFilters(); }});
    document.getElementById('clearFilter').addEventListener('click', resetPortalFilters);
    document.getElementById('clearFilterLauncher').addEventListener('click', resetPortalFilters);
    document.getElementById('refreshPortalData').addEventListener('click', refreshPortalData);
    document.getElementById('refreshPortalLauncher').addEventListener('click', refreshPortalData);
    document.getElementById('quickRefresh').addEventListener('click', refreshPortalData);
    document.getElementById('reloadPage').addEventListener('click', reloadPortalShell);
    document.getElementById('reloadPageLauncher').addEventListener('click', reloadPortalShell);
    document.getElementById('openAiPromptModal').addEventListener('click', openAiPromptModal);
    document.getElementById('modalUpdateAiPrompt').addEventListener('click', updateAiPrompt);
    document.getElementById('modalCopyAiPrompt').addEventListener('click', copyAiPrompt);
    document.getElementById('modalRunAiPrompt').addEventListener('click', runAiPrompt);
    document.getElementById('aiSelectedDate').addEventListener('change', updateAiPrompt);

    document.getElementById('filterModal').addEventListener('click', (event) => {{
      if (event.target.id === 'filterModal') closeFilterModal();
    }});

    document.getElementById('aiPromptModal').addEventListener('click', (event) => {{
      if (event.target.id === 'aiPromptModal') closeAiPromptModal();
    }});

    window.addEventListener('click', (event) => {{
      if (event.target.id === 'matchModal') closeMatchDetails();
    }});

    window.addEventListener('keydown', (event) => {{
      if (event.key === 'Escape') {{
        closeMatchDetails();
        closeFilterModal();
        closeAiPromptModal();
      }}
    }});

    window.addEventListener('resize', () => {{
      ['matchModal', 'filterModal', 'aiPromptModal'].forEach((modalId) => {{
        const modal = document.getElementById(modalId);
        if (modal && modal.style.display !== 'none') {{
          positionVisibleModal(modalId);
        }}
      }});
    }});

    Array.from(document.querySelectorAll('.jump-link[data-comp-name], .rail-link[data-comp-name]')).forEach((link) => {{
      link.addEventListener('click', (event) => {{
        event.preventDefault();
        toggleCompetitionFilter(link.getAttribute('data-comp-name') || '');
        const anchor = link.getAttribute('href');
        if (anchor && anchor.startsWith('#')) {{
          const target = document.querySelector(anchor);
          if (target && document.getElementById('fcomp').value) {{
            target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
          }} else {{
            window.scrollTo({{ top: 0, behavior: 'smooth' }});
          }}
        }}
      }});
    }});

    Array.from(document.querySelectorAll('.competition-card[data-comp]')).forEach((card) => {{
      card.addEventListener('click', (event) => {{
        const target = event.target;
        if (!target || typeof target.closest !== 'function') return;
        if (target.closest('button, a, input, select, textarea, .btn, .btn-mini, .btn-link, .modal-close')) return;

        const competitionField = document.getElementById('fcomp');
        if (!competitionField) return;
        const cardCompetition = (card.getAttribute('data-comp') || '').trim();
        if (!cardCompetition) return;
        if ((competitionField.value || '').trim() === cardCompetition) return;

        competitionField.value = cardCompetition;
        applyFilters();
        window.scrollTo({{ top: 0, behavior: 'smooth' }});
      }});
    }});

    const today = new Date();
    document.getElementById('aiSelectedDate').value = today.toISOString().slice(0, 10);
    applyRiskPreset();
    applyFilters();
    classifyRowsByRisk();
    updateAiPrompt();
  </script>
</body>
</html>
"""

    return html


def main() -> None:
    html = build_index_html()
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()

