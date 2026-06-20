from __future__ import annotations

from datetime import datetime
from html import escape
import json
import math
from pathlib import Path
import re
import unicodedata
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup

from analytics import (
    calculate_match_probabilities,
    probability_map,
    result_to_market,
    suggest_bet_strategy,
)
from nvidia_client import request_nvidia_completion
from scraper import load_competition_matches


APP_TIMEZONE = ZoneInfo("America/Sao_Paulo")
OUTPUT_FILE = Path(__file__).resolve().parent / "copa_do_mundo.html"
TELEMUNDO_SCHEDULE_URL = (
    "https://www.telemundo.com/deportes/copa-mundial-de-la-fifa-2026/"
    "calendario-de-transmisiones-de-la-copa-mundial-de-la-fifa-2026-por-tel-rcna261413"
)
FIFA_SCHEDULE_RELEASE_URL = (
    "https://inside.fifa.com/organisation/president/news/world-cup-2026-match-schedule-fixtures-ronaldo-infantino"
)
ESPN_WORLD_CUP_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
FIFA_MENS_RANKING_URL = "https://api.fifa.com/api/v3/rankings?gender=1&count=211&dateId=FRS_Male_Football_20260401"
TEAM_PRIORS_CACHE = Path(__file__).resolve().parent / "world_cup_2026_team_priors.json"
MODEL_ADJUSTMENTS_FILE = Path(__file__).resolve().parent / "world_cup_2026_model_adjustments.json"
RESULTS_OVERRIDES_FILE = Path(__file__).resolve().parent / "world_cup_2026_results_overrides.json"
PLACEHOLDER_TEAMS = {"Por definir", "TBD", "-"}
TEAM_NAME_ALIASES = {
    "mexico": "Mexico",
    "sudafrica": "South Africa",
    "republica de corea": "South Korea",
    "corea del sur": "South Korea",
    "korea republic": "South Korea",
    "chequia": "Czechia",
    "canada": "Canada",
    "bosnia y herzegovina": "Bosnia and Herzegovina",
    "bosnia herzegovina": "Bosnia and Herzegovina",
    "estados unidos": "USA",
    "united states": "USA",
    "united states of america": "USA",
    "catar": "Qatar",
    "suiza": "Switzerland",
    "brasil": "Brazil",
    "marruecos": "Morocco",
    "haiti": "Haiti",
    "escocia": "Scotland",
    "turquia": "Turkey",
    "turkiye": "Turkey",
    "alemania": "Germany",
    "curazao": "Curacao",
    "curacao": "Curacao",
    "paises bajos": "Netherlands",
    "japon": "Japan",
    "costa de marfil": "Ivory Coast",
    "cote d ivoire": "Ivory Coast",
    "ecuador": "Ecuador",
    "suecia": "Sweden",
    "tunez": "Tunisia",
    "espana": "Spain",
    "cabo verde": "Cape Verde",
    "belgica": "Belgium",
    "egipto": "Egypt",
    "arabia saudita": "Saudi Arabia",
    "uruguay": "Uruguay",
    "iran": "Iran",
    "nueva zelanda": "New Zealand",
    "francia": "France",
    "senegal": "Senegal",
    "irak": "Iraq",
    "noruega": "Norway",
    "argentina": "Argentina",
    "argelia": "Algeria",
    "austria": "Austria",
    "jordania": "Jordan",
    "portugal": "Portugal",
    "rd del congo": "D.R. Congo",
    "congo dr": "D.R. Congo",
    "dr congo": "D.R. Congo",
    "inglaterra": "England",
    "croacia": "Croatia",
    "ghana": "Ghana",
    "panama": "Panama",
    "uzbekistan": "Uzbekistan",
    "colombia": "Colombia",
    "paraguay": "Paraguay",
    "australia": "Australia",
    "suriname": "Suriname",
    "bolivia": "Bolivia",
    "jamaica": "Jamaica",
    "new caledonia": "New Caledonia",
    "d.r. congo": "D.R. Congo",
}
TEAM_FLAG_CODES = {
    "Mexico": "mx",
    "South Africa": "za",
    "South Korea": "kr",
    "Czechia": "cz",
    "Canada": "ca",
    "Bosnia and Herzegovina": "ba",
    "USA": "us",
    "Paraguay": "py",
    "Qatar": "qa",
    "Switzerland": "ch",
    "Brazil": "br",
    "Morocco": "ma",
    "Haiti": "ht",
    "Scotland": "gb-sct",
    "Australia": "au",
    "Turkey": "tr",
    "Germany": "de",
    "Curacao": "cw",
    "Netherlands": "nl",
    "Japan": "jp",
    "Ivory Coast": "ci",
    "Ecuador": "ec",
    "Sweden": "se",
    "Tunisia": "tn",
    "Spain": "es",
    "Cape Verde": "cv",
    "Belgium": "be",
    "Egypt": "eg",
    "Saudi Arabia": "sa",
    "Uruguay": "uy",
    "Iran": "ir",
    "New Zealand": "nz",
    "France": "fr",
    "Senegal": "sn",
    "Iraq": "iq",
    "Norway": "no",
    "Argentina": "ar",
    "Algeria": "dz",
    "Austria": "at",
    "Jordan": "jo",
    "Portugal": "pt",
    "D.R. Congo": "cd",
    "England": "gb-eng",
    "Croatia": "hr",
    "Ghana": "gh",
    "Panama": "pa",
    "Uzbekistan": "uz",
    "Colombia": "co",
    "Suriname": "sr",
    "Bolivia": "bo",
    "Jamaica": "jm",
    "New Caledonia": "nc",
}


def _sort_matches(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.copy()
    ordered["_event_dt"] = pd.to_datetime(ordered.get("event_timestamp"), errors="coerce", utc=True)
    if ordered["_event_dt"].notna().any():
        ordered = ordered.sort_values(
            by=["_event_dt", "status", "home_team", "away_team"],
            ascending=[True, True, True, True],
        )
    else:
        ordered = ordered.sort_values(by=["status", "date_text", "home_team", "away_team"])
    return ordered.reset_index(drop=True)


def _ascii_slug(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower().strip()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _canonical_team_name(name: str) -> str:
    cleaned = _ascii_slug(name)
    if not cleaned:
        return name.strip()
    return TEAM_NAME_ALIASES.get(cleaned, name.strip())


def _extract_match_teams(matchup: str) -> tuple[str | None, str | None]:
    if " vs. " in matchup:
        home_raw, away_raw = matchup.split(" vs. ", 1)
    elif " vs " in matchup:
        home_raw, away_raw = matchup.split(" vs ", 1)
    else:
        return None, None
    home_team = home_raw.strip()
    away_team = away_raw.strip()
    if home_team in PLACEHOLDER_TEAMS or away_team in PLACEHOLDER_TEAMS:
        return None, None
    return _canonical_team_name(home_team), _canonical_team_name(away_team)


def _split_matchup_display(matchup: str) -> tuple[str, str]:
    if " vs. " in matchup:
        return tuple(part.strip() for part in matchup.split(" vs. ", 1))
    if " vs " in matchup:
        return tuple(part.strip() for part in matchup.split(" vs ", 1))
    return matchup.strip(), ""


def _team_flag_url(team_name: str | None) -> str | None:
    if not team_name:
        return None
    code = TEAM_FLAG_CODES.get(_canonical_team_name(team_name))
    if not code:
        return None
    return f"https://flagcdn.com/w40/{code}.png"


def _team_html(display_name: str, canonical_name: str | None) -> str:
    flag_url = _team_flag_url(canonical_name or display_name)
    if flag_url:
        return (
            f'<span class="team-with-flag">'
            f'<img class="team-flag" src="{escape(flag_url)}" alt="Bandeira de {escape(display_name)}" loading="lazy" />'
            f"<span>{escape(display_name)}</span>"
            f"</span>"
        )
    return f"<span>{escape(display_name)}</span>"


def _schedule_datetime_brt(date_label: str, time_et: str) -> datetime:
    date_match = re.search(r"(\d{1,2}) de (junio|julio)", _ascii_slug(date_label))
    if not date_match:
        raise ValueError(f"Data invalida na grade: {date_label}")
    month_map = {"junio": 6, "julio": 7}
    day = int(date_match.group(1))
    month = month_map[date_match.group(2)]
    hour, minute = _parse_time_to_24h(time_et)
    et_dt = datetime(2026, month, day, hour, minute, tzinfo=ZoneInfo("America/New_York"))
    return et_dt.astimezone(APP_TIMEZONE)


def _fmt_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.1f}%"


def _fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_fifa_rank(value: object) -> str:
    rank = _safe_int_or_none(value)
    return f"#{rank}" if rank is not None else "-"


def _fmt_fifa_points(value: object) -> str:
    points = _safe_float_or_none(value)
    return f"{points:.2f} pts" if points is not None else "sem pontos"


def _team_label_from_market(market: str, home_team: str, away_team: str) -> str:
    if market == "Casa":
        return f"Vitoria {home_team}"
    if market == "Fora":
        return f"Vitoria {away_team}"
    return "Empate"


def _confidence_score(history_size: int, top_market_prob: float, top_score_prob: float, bookmakers: int) -> int:
    history_component = min(history_size, 12) / 12 * 28
    market_component = max(0.0, min(1.0, top_market_prob)) * 42
    score_component = max(0.0, min(0.22, top_score_prob)) / 0.22 * 18
    books_component = min(max(bookmakers, 0), 16) / 16 * 12
    return int(round(max(18.0, min(96.0, history_component + market_component + score_component + books_component))))


def _confidence_label(score: int) -> tuple[str, str]:
    if score >= 76:
        return "Alta leitura", "high"
    if score >= 56:
        return "Leitura moderada", "mid"
    return "Leitura sensivel", "low"


def _prediction_blurb(
    *,
    home_team: str,
    away_team: str,
    top_market: str,
    top_market_prob: float,
    suggested_score: str,
    btts_yes: float,
    over_25: float,
    finished: bool,
) -> str:
    if top_market == "Casa":
        opening = f"O modelo puxa vantagem para {home_team}"
    elif top_market == "Fora":
        opening = f"O modelo encontra espaco para {away_team}"
    else:
        opening = "A distribuicao aponta um jogo bem apertado"

    style = "com tendencia de gols" if over_25 >= 0.52 else "com placar mais contido"
    exchange = "e boa chance de ambos marcarem" if btts_yes >= 0.5 else "e risco menor de BTTS"
    context = "retroanalise pre-jogo" if finished else "leitura pre-jogo"
    return (
        f"{opening} ({top_market_prob * 100:.1f}%), {style}, {exchange}. "
        f"No recorte de {context}, o placar sugerido mais forte foi {suggested_score}."
    )


def _generate_ai_summary(records: list[dict[str, object]], stats: dict[str, int | float]) -> str:
    if not records:
        return "A IA nao recebeu jogos suficientes para resumir a competicao."

    compact_lines: list[str] = []
    for item in records:
        compact_lines.append(
            " | ".join(
                [
                    f"Jogo: {item['home_team']} x {item['away_team']}",
                    f"Status: {item['status']}",
                    f"Data: {item['date_text']}",
                    f"Placar sugerido: {item['suggested_score']}",
                    f"{item['home_team']} {item['home_win_pct']}",
                    f"Empate {item['draw_pct']}",
                    f"{item['away_team']} {item['away_win_pct']}",
                    f"BTTS {item['btts_pct']}",
                    f"Over 2.5 {item['over_25_pct']}",
                    f"Confianca {item['confidence_score']}/100",
                    f"Insight: {item['prediction_blurb']}",
                ]
            )
        )

    prompt = (
        "Escreva um resumo executivo curto para uma pagina HTML de previsoes da Copa do Mundo 2026. "
        "Use apenas os dados abaixo. Nao invente lesoes, desfalques, xG externo, noticias ou mercado adicional. "
        "Entregue 2 paragrafos curtos em portugues do Brasil. O primeiro resume o panorama geral da rodada. "
        "O segundo destaca cautelas sobre tamanho da base e confianca do modelo.\n\n"
        f"Estatisticas gerais: jogos={stats['total_matches']}, finalizados={stats['finished_matches']}, "
        f"agendados={stats['scheduled_matches']}, previsoes={stats['predicted_matches']}.\n\n"
        "Dados por jogo:\n"
        + "\n".join(compact_lines)
    )

    try:
        return request_nvidia_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "Voce resume paineis quantitativos de futebol com clareza executiva. "
                        "Use apenas a base fornecida e seja honesto sobre limites do modelo."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            top_p=0.85,
            max_tokens=420,
        )
    except Exception as exc:
        return (
            "Resumo IA indisponivel nesta execucao. "
            f"Motivo: {exc}. O painel continua exibindo as probabilidades e os placares sugeridos pelo modelo local."
        )


def _build_records(df: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, int | float]]:
    ordered = _sort_matches(df)
    finished_sorted = ordered[ordered["status"] == "Finalizado"].copy().reset_index(drop=True)
    records: list[dict[str, object]] = []

    for row in ordered.itertuples(index=False):
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        finished = str(row.status) == "Finalizado"

        history_size = 0
        prediction_source = ordered.copy()
        if finished:
            current_mask = (
                (finished_sorted["home_team"] == home_team)
                & (finished_sorted["away_team"] == away_team)
                & (finished_sorted["date_text"] == str(row.date_text))
            )
            current_indices = finished_sorted.index[current_mask].tolist()
            if current_indices:
                history_size = int(current_indices[0])
                prediction_source = finished_sorted.iloc[: current_indices[0]].copy()
            else:
                prediction_source = finished_sorted.copy()
                history_size = len(prediction_source)
        else:
            prediction_source = ordered.copy()
            history_size = int((ordered["status"] == "Finalizado").sum())

        has_prediction = not prediction_source.empty and int((prediction_source["status"] == "Finalizado").sum()) > 0
        probs = None
        suggested_score = "Base insuficiente"
        top_score_prob = 0.0
        top_market = "-"
        top_market_prob = 0.0
        prediction_blurb = (
            "Ainda nao ha historico finalizado suficiente na rota local para calcular uma leitura confiavel."
        )
        confidence_score = 18
        confidence_text, confidence_tone = _confidence_label(confidence_score)
        actual_score = "-"
        outcome_text = "Sem retroanalise"
        best_market_label = "-"
        best_odd = None
        best_ev = None

        if row.home_goals is not None and row.away_goals is not None and not pd.isna(row.home_goals) and not pd.isna(row.away_goals):
            actual_score = f"{int(row.home_goals)} x {int(row.away_goals)}"

        if has_prediction:
            try:
                probs = calculate_match_probabilities(prediction_source, home_team, away_team)
                model_probs = probability_map(probs)
                top_market, top_market_prob = max(model_probs.items(), key=lambda item: item[1])
                if probs.top_scorelines:
                    suggested_score, top_score_prob = probs.top_scorelines[0]

                bookmakers = 0 if row.bookmakers is None or pd.isna(row.bookmakers) else int(row.bookmakers)
                confidence_score = _confidence_score(history_size, top_market_prob, top_score_prob, bookmakers)
                confidence_text, confidence_tone = _confidence_label(confidence_score)
                prediction_blurb = _prediction_blurb(
                    home_team=home_team,
                    away_team=away_team,
                    top_market=top_market,
                    top_market_prob=top_market_prob,
                    suggested_score=suggested_score,
                    btts_yes=probs.btts_yes,
                    over_25=probs.over_25,
                    finished=finished,
                )

                if row.odds_home is not None and row.odds_draw is not None and row.odds_away is not None:
                    tip = suggest_bet_strategy(
                        probs=probs,
                        odd_home=float(row.odds_home),
                        odd_draw=float(row.odds_draw),
                        odd_away=float(row.odds_away),
                        bankroll=1000.0,
                        kelly_fractional=0.25,
                    )
                    best_market_label = _team_label_from_market(tip.best_market, home_team, away_team)
                    best_odd = tip.best_odd
                    best_ev = tip.expected_value

                if finished and actual_score != "-":
                    actual_market = result_to_market(float(row.home_goals), float(row.away_goals))
                    if actual_score == suggested_score:
                        outcome_text = "Placar sugerido acertou em cheio"
                    elif actual_market == top_market:
                        outcome_text = "Direcao do mercado prevista corretamente"
                    else:
                        outcome_text = "Modelo divergiu do resultado final"
            except Exception as exc:
                prediction_blurb = f"Leitura indisponivel nesta partida: {exc}"

        top_scores = probs.top_scorelines[:3] if probs is not None else []
        records.append(
            {
                "status": str(row.status),
                "date_text": str(row.date_text),
                "home_team": home_team,
                "away_team": away_team,
                "match_text": f"{home_team} x {away_team}",
                "search_blob": f"{home_team} {away_team} {row.date_text} {row.status}".lower(),
                "has_prediction": has_prediction and probs is not None,
                "home_win_pct": _fmt_pct(probs.home_win if probs is not None else None),
                "draw_pct": _fmt_pct(probs.draw if probs is not None else None),
                "away_win_pct": _fmt_pct(probs.away_win if probs is not None else None),
                "btts_pct": _fmt_pct(probs.btts_yes if probs is not None else None),
                "over_25_pct": _fmt_pct(probs.over_25 if probs is not None else None),
                "under_25_pct": _fmt_pct(probs.under_25 if probs is not None else None),
                "expected_home_goals": _fmt_num(probs.expected_home_goals if probs is not None else None),
                "expected_away_goals": _fmt_num(probs.expected_away_goals if probs is not None else None),
                "suggested_score": suggested_score,
                "top_scores": top_scores,
                "prediction_blurb": prediction_blurb,
                "confidence_score": confidence_score,
                "confidence_text": confidence_text,
                "confidence_tone": confidence_tone,
                "actual_score": actual_score,
                "outcome_text": outcome_text,
                "history_size": history_size,
                "best_market_label": best_market_label,
                "best_odd": _fmt_num(best_odd, 2) if best_odd is not None else "-",
                "best_ev": f"{best_ev * 100:.2f}%" if best_ev is not None else "-",
                "odds_home": _fmt_num(row.odds_home, 2),
                "odds_draw": _fmt_num(row.odds_draw, 2),
                "odds_away": _fmt_num(row.odds_away, 2),
            }
        )

    total_matches = len(records)
    finished_matches = sum(1 for item in records if item["status"] == "Finalizado")
    scheduled_matches = sum(1 for item in records if item["status"] == "Agendado")
    predicted_matches = sum(1 for item in records if item["has_prediction"])
    avg_confidence = round(
        sum(int(item["confidence_score"]) for item in records if item["has_prediction"]) / max(predicted_matches, 1)
    )

    stats = {
        "total_matches": total_matches,
        "finished_matches": finished_matches,
        "scheduled_matches": scheduled_matches,
        "predicted_matches": predicted_matches,
        "avg_confidence": avg_confidence if predicted_matches else 0,
    }
    return records, stats


def _match_card_html(item: dict[str, object]) -> str:
    status_class = "status-finished" if item["status"] == "Finalizado" else "status-upcoming"
    confidence_class = f"confidence-{item['confidence_tone']}"

    top_score_html = "".join(
        f"<span>{escape(score)} <strong>{prob * 100:.1f}%</strong></span>"
        for score, prob in item["top_scores"]
    )
    if not top_score_html:
        top_score_html = "<span>Sem scorelines suficientes</span>"

    prediction_block = (
        f"""
        <div class="prob-grid">
          <div class="prob-item">
            <label>{escape(str(item['home_team']))}</label>
            <strong>{escape(str(item['home_win_pct']))}</strong>
          </div>
          <div class="prob-item">
            <label>Empate</label>
            <strong>{escape(str(item['draw_pct']))}</strong>
          </div>
          <div class="prob-item">
            <label>{escape(str(item['away_team']))}</label>
            <strong>{escape(str(item['away_win_pct']))}</strong>
          </div>
        </div>
        <div class="scoreline-shell">
          <div>
            <span class="mini-label">Placar sugerido</span>
            <strong>{escape(str(item['suggested_score']))}</strong>
          </div>
          <div>
            <span class="mini-label">Gols esperados</span>
            <strong>{escape(str(item['expected_home_goals']))} x {escape(str(item['expected_away_goals']))}</strong>
          </div>
        </div>
        <div class="micro-list">{top_score_html}</div>
        <div class="fact-grid">
          <span>BTTS {escape(str(item['btts_pct']))}</span>
          <span>Over 2.5 {escape(str(item['over_25_pct']))}</span>
          <span>Under 2.5 {escape(str(item['under_25_pct']))}</span>
          <span>Historico base {escape(str(item['history_size']))}</span>
        </div>
        <p class="insight">{escape(str(item['prediction_blurb']))}</p>
        """
        if item["has_prediction"]
        else """
        <div class="empty-prediction">
          O modelo local ainda nao encontrou historico suficiente nesta rota para entregar uma previsao solida.
        </div>
        """
    )

    result_block = (
        f"""
        <div class="actual-box">
          <span class="mini-label">Placar real</span>
          <strong>{escape(str(item['actual_score']))}</strong>
          <p>{escape(str(item['outcome_text']))}</p>
        </div>
        """
        if item["status"] == "Finalizado"
        else ""
    )

    return f"""
    <article class="match-card" data-status="{escape(str(item['status']).lower())}" data-search="{escape(str(item['search_blob']))}">
      <div class="match-top">
        <span class="status-pill {status_class}">{escape(str(item['status']))}</span>
        <span class="match-date">{escape(str(item['date_text']))}</span>
      </div>
      <h3>{escape(str(item['home_team']))} <span>x</span> {escape(str(item['away_team']))}</h3>
      <div class="confidence-box {confidence_class}">
        <span>{escape(str(item['confidence_text']))}</span>
        <strong>{escape(str(item['confidence_score']))}/100</strong>
      </div>
      {prediction_block}
      <div class="odds-strip">
        <span>1 {escape(str(item['odds_home']))}</span>
        <span>X {escape(str(item['odds_draw']))}</span>
        <span>2 {escape(str(item['odds_away']))}</span>
      </div>
      <div class="bet-box">
        <span class="mini-label">Leitura de valor</span>
        <strong>{escape(str(item['best_market_label']))}</strong>
        <p>Odd {escape(str(item['best_odd']))} | EV {escape(str(item['best_ev']))}</p>
      </div>
      {result_block}
    </article>
    """


def build_world_cup_html() -> str:
    df = load_competition_matches("Copa do Mundo")
    records, stats = _build_records(df)
    generated_at = datetime.now(APP_TIMEZONE).strftime("%d/%m/%Y %H:%M:%S")
    ai_summary = _generate_ai_summary(records, stats)
    cards_html = "".join(_match_card_html(item) for item in records)

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Copa do Mundo 2026 | Probabilidades e Placares Sugeridos</title>
  <style>
    /* ============================================================
       DESIGN TOKENS — Copa do Mundo 2026 × Brasil
       ============================================================ */
    :root {{
      /* Cores Brasil */
      --verde:        #009c3b;
      --verde-deep:   #006828;
      --verde-light:  #00c94a;
      --amarelo:      #FFDF00;
      --amarelo-deep: #e6c800;
      --amarelo-soft: #fff3a0;
      --azul:         #002776;
      --azul-mid:     #1a4db8;
      --branco:       #ffffff;

      /* UI */
      --ink:          #0d1f14;
      --ink-mid:      #1e3a26;
      --muted:        #5a7060;
      --paper:        rgba(255,255,255,0.92);
      --card-bg:      rgba(255,255,255,0.97);
      --line:         rgba(0,40,10,0.10);
      --shadow-card:  0 8px 32px rgba(0,40,10,0.10), 0 1px 4px rgba(0,40,10,0.06);
      --shadow-hover: 0 20px 56px rgba(0,40,10,0.16), 0 2px 8px rgba(0,40,10,0.10);

      /* Gradientes */
      --grad-hero:    linear-gradient(135deg, #004d1c 0%, #006828 35%, #002776 100%);
      --grad-gold:    linear-gradient(135deg, #FFDF00, #ffa500);
      --grad-brasil:  linear-gradient(90deg, #009c3b, #FFDF00, #009c3b);
    }}

    /* ============================================================
       RESET & BASE
       ============================================================ */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      font-family: 'Outfit', 'Inter', sans-serif;
      color: var(--ink);
      background: #f0f4f1;
      overflow-x: hidden;
      min-height: 100vh;
    }}

    /* ============================================================
       ANIMATED BACKGROUND
       ============================================================ */
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      z-index: 0;
      pointer-events: none;
      background:
        linear-gradient(160deg, var(--verde-deep) 0%, #004d20 28%, var(--azul) 62%, #001550 100%);
    }}
    body::after {{
      content: "";
      position: fixed;
      inset: 0;
      z-index: 0;
      pointer-events: none;
      background:
        radial-gradient(ellipse 80% 40% at 20% 15%, rgba(255,223,0,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 10%, rgba(0,156,59,0.25) 0%, transparent 55%),
        radial-gradient(ellipse 50% 60% at 50% 80%, rgba(0,39,118,0.30) 0%, transparent 70%),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='60' height='60'%3E%3Ccircle cx='30' cy='30' r='1.5' fill='rgba(255,255,255,0.04)'/%3E%3C/svg%3E");
      animation: bgPulse 8s ease-in-out infinite alternate;
    }}
    @keyframes bgPulse {{
      from {{ opacity: 0.7; }}
      to   {{ opacity: 1; }}
    }}

    /* ============================================================
       SHELL
       ============================================================ */
    .shell {{
      position: relative;
      z-index: 1;
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px 20px 60px;
    }}

    /* ============================================================
       HERO
       ============================================================ */
    .hero {{
      position: relative;
      overflow: hidden;
      padding: 40px 36px 44px;
      border-radius: 28px;
      color: #fff;
      background: var(--grad-hero);
      box-shadow: 0 32px 100px rgba(0,20,8,0.42), 0 0 0 1px rgba(255,255,255,0.06);
    }}

    /* Faixa dourada animada no topo */
    .hero::before {{
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 5px;
      background: var(--grad-brasil);
      background-size: 200% 100%;
      animation: shineBar 3s linear infinite;
    }}
    @keyframes shineBar {{
      0%   {{ background-position: 0% 50%; }}
      100% {{ background-position: 200% 50%; }}
    }}

    /* Hexágono / bola de futebol decorativa */
    .hero::after {{
      content: "⚽";
      position: absolute;
      right: 36px;
      bottom: -18px;
      font-size: clamp(8rem, 20vw, 16rem);
      line-height: 1;
      opacity: 0.07;
      pointer-events: none;
      user-select: none;
      animation: floatBall 6s ease-in-out infinite;
    }}
    @keyframes floatBall {{
      0%, 100% {{ transform: translateY(0) rotate(-8deg); }}
      50%       {{ transform: translateY(-14px) rotate(8deg); }}
    }}

    .hero-poster-mark {{
      position: absolute;
      right: 180px;
      top: -8px;
      z-index: 0;
      font-family: 'Bebas Neue', sans-serif;
      font-size: clamp(5rem, 16vw, 11rem);
      line-height: 0.85;
      letter-spacing: -0.04em;
      color: rgba(255,223,0,0.08);
      pointer-events: none;
      user-select: none;
      text-shadow: 0 0 80px rgba(255,223,0,0.12);
    }}

    .hero-grid {{
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: minmax(0, 1.7fr) minmax(260px, 0.8fr);
      gap: 28px;
      align-items: start;
    }}

    /* Tag badge */
    .hero-tag {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 18px;
      border-radius: 999px;
      border: 1.5px solid rgba(255,223,0,0.38);
      background: rgba(255,223,0,0.12);
      font: 700 0.75rem/1 'Outfit', sans-serif;
      letter-spacing: 0.10em;
      text-transform: uppercase;
      color: var(--amarelo);
      box-shadow: 0 0 16px rgba(255,223,0,0.15);
    }}
    .hero-tag::before {{
      content: "";
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--amarelo);
      animation: blink 1.4s ease-in-out infinite;
      flex-shrink: 0;
    }}
    @keyframes blink {{
      0%, 100% {{ opacity: 1; box-shadow: 0 0 6px var(--amarelo); }}
      50%       {{ opacity: 0.4; box-shadow: none; }}
    }}

    h1 {{
      margin: 16px 0 14px;
      font-family: 'Bebas Neue', sans-serif;
      font-size: clamp(2.8rem, 6vw, 5.5rem);
      line-height: 0.92;
      letter-spacing: 0.02em;
      color: #fff;
      text-shadow: 0 4px 24px rgba(0,0,0,0.28);
    }}
    h1 span {{ color: var(--amarelo); }}

    .hero p {{
      max-width: 68ch;
      color: rgba(255,255,255,0.76);
      font-size: 1.02rem;
      line-height: 1.72;
    }}

    .hero-band {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 22px;
    }}

    .hero-chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 9px 16px;
      border-radius: 999px;
      background: rgba(255,255,255,0.10);
      border: 1px solid rgba(255,255,255,0.18);
      font: 600 0.82rem/1 'Outfit', sans-serif;
      color: rgba(255,255,255,0.90);
      backdrop-filter: blur(8px);
      transition: background 0.2s, border-color 0.2s;
    }}
    .hero-chip:hover {{ background: rgba(255,223,0,0.18); border-color: rgba(255,223,0,0.45); }}
    .hero-chip i {{ width: 8px; height: 8px; border-radius: 50%; background: var(--amarelo); flex-shrink: 0; }}

    /* Side panels */
    .hero-side {{ display: grid; gap: 14px; }}
    .hero-panel {{
      padding: 20px;
      border-radius: 20px;
      background: rgba(255,255,255,0.10);
      border: 1px solid rgba(255,255,255,0.18);
      backdrop-filter: blur(14px);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.10), 0 8px 32px rgba(0,0,0,0.16);
      transition: background 0.25s, transform 0.25s;
    }}
    .hero-panel:hover {{
      background: rgba(255,255,255,0.16);
      transform: translateY(-2px);
    }}
    .hero-panel span {{
      display: block;
      font: 700 0.70rem/1 'Outfit', sans-serif;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: rgba(255,223,0,0.85);
      margin-bottom: 8px;
    }}
    .hero-panel strong {{
      display: block;
      font-family: 'Bebas Neue', sans-serif;
      font-size: 1.9rem;
      letter-spacing: 0.04em;
      color: #fff;
      line-height: 1;
    }}

    /* Stats strip */
    .stats {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 14px;
      margin-top: 22px;
    }}
    .stat {{
      position: relative;
      padding: 18px 16px 20px;
      border-radius: 20px;
      background: rgba(255,255,255,0.10);
      border: 1px solid rgba(255,255,255,0.16);
      backdrop-filter: blur(16px);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.12), 0 8px 28px rgba(0,0,0,0.18);
      overflow: hidden;
      transition: transform 0.25s, background 0.25s;
    }}
    .stat:hover {{ transform: translateY(-4px); background: rgba(255,255,255,0.16); }}
    .stat span {{
      display: block;
      font: 700 0.68rem/1 'Outfit', sans-serif;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: rgba(255,223,0,0.90);
    }}
    .stat strong {{
      display: block;
      margin-top: 10px;
      font-family: 'Bebas Neue', sans-serif;
      font-size: 2.6rem;
      letter-spacing: 0.02em;
      line-height: 1;
      color: #fff;
    }}
    /* Barra inferior colorida */
    .stat::after {{
      content: "";
      position: absolute;
      left: 0; right: 0; bottom: 0;
      height: 3px;
      background: linear-gradient(90deg, var(--verde), var(--amarelo), var(--verde));
      background-size: 200% 100%;
      animation: slideGrad 3s linear infinite;
    }}
    @keyframes slideGrad {{
      0%   {{ background-position: 0% 50%; }}
      100% {{ background-position: 200% 50%; }}
    }}

    /* ============================================================
       MAIN LAYOUT
       ============================================================ */
    .layout {{ display: block; margin-top: 22px; }}

    .main-card {{
      position: relative;
      background: rgba(255,255,255,0.97);
      border: 1px solid rgba(255,255,255,0.95);
      border-radius: 24px;
      box-shadow: 0 24px 64px rgba(0,40,14,0.12);
      backdrop-filter: blur(20px);
      padding: 28px;
      overflow: hidden;
    }}
    /* Faixa tricolor no topo do card */
    .main-card::before {{
      content: "";
      position: absolute;
      left: 0; top: 0; right: 0;
      height: 4px;
      background: linear-gradient(90deg, var(--verde-deep) 33%, var(--amarelo) 33% 66%, var(--azul) 66%);
    }}

    h2 {{
      font-family: 'Bebas Neue', sans-serif;
      font-size: clamp(1.6rem, 3vw, 2.4rem);
      letter-spacing: 0.05em;
      color: var(--ink);
    }}

    /* ============================================================
       TOOLBAR / FILTROS
       ============================================================ */
    .toolbar {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      flex-wrap: wrap;
      align-items: flex-end;
      margin-bottom: 22px;
      padding: 20px;
      background: linear-gradient(135deg, rgba(0,104,40,0.06), rgba(255,223,0,0.08));
      border-radius: 18px;
      border: 1px solid rgba(0,104,40,0.12);
    }}
    .field {{ min-width: min(100%, 220px); flex: 1; }}
    .field label {{
      display: block;
      margin-bottom: 8px;
      font: 700 0.70rem/1 'Outfit', sans-serif;
      letter-spacing: 0.10em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .field input, .field select {{
      width: 100%;
      height: 50px;
      border-radius: 14px;
      border: 1.5px solid rgba(0,104,40,0.18);
      background: rgba(255,255,255,0.95);
      font: 500 0.94rem/1 'Outfit', sans-serif;
      padding: 0 16px;
      color: var(--ink);
      outline: none;
      transition: border-color 0.2s, box-shadow 0.2s;
    }}
    .field input:focus, .field select:focus {{
      border-color: var(--verde);
      box-shadow: 0 0 0 3px rgba(0,156,59,0.15);
    }}

    .results-count {{
      color: var(--muted);
      font: 500 0.92rem/1 'Outfit', sans-serif;
      margin-bottom: 14px;
      padding: 0 4px;
    }}

    /* ============================================================
       ANALYSIS GRID
       ============================================================ */
    .analysis-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 18px;
    }}

    .analysis-card {{
      position: relative;
      display: grid;
      gap: 14px;
      padding: 20px;
      border-radius: 22px;
      background: var(--card-bg);
      border: 1.5px solid rgba(0,40,10,0.07);
      box-shadow: var(--shadow-card);
      overflow: hidden;
      transition: transform 0.28s cubic-bezier(.22,.68,0,1.2), box-shadow 0.28s, border-color 0.28s;
    }}
    .analysis-card:hover {{
      transform: translateY(-6px) scale(1.01);
      box-shadow: var(--shadow-hover);
      border-color: rgba(0,156,59,0.22);
    }}

    /* Barra lateral colorida */
    .analysis-card::before {{
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 4px;
      border-radius: 22px 0 0 22px;
      background: linear-gradient(180deg, var(--verde), var(--amarelo), var(--azul-mid));
    }}

    /* Glow no hover */
    .analysis-card::after {{
      content: "";
      position: absolute;
      inset: 0;
      border-radius: inherit;
      background: radial-gradient(ellipse at top right, rgba(255,223,0,0.08), transparent 60%);
      pointer-events: none;
      opacity: 0;
      transition: opacity 0.28s;
    }}
    .analysis-card:hover::after {{ opacity: 1; }}

    /* ============================================================
       STAGE PILLS
       ============================================================ */
    .analysis-top {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .stage-pill, .date-pill {{
      display: inline-flex;
      align-items: center;
      padding: 6px 12px;
      border-radius: 999px;
      font: 700 0.70rem/1 'Outfit', sans-serif;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .stage-group    {{ background: rgba(0,39,118,0.10); color: var(--azul-mid); border: 1px solid rgba(0,39,118,0.14); }}
    .stage-knockout {{ background: rgba(0,104,40,0.12); color: var(--verde-deep); border: 1px solid rgba(0,104,40,0.16); }}
    .stage-final    {{
      background: linear-gradient(135deg, rgba(255,223,0,0.24), rgba(255,165,0,0.18));
      color: #7a5a00;
      border: 1px solid rgba(255,200,0,0.30);
    }}
    .date-pill {{ background: rgba(0,40,14,0.05); color: var(--muted); border: 1px solid rgba(0,40,14,0.08); }}

    /* ============================================================
       MATCHUP
       ============================================================ */
    .analysis-card h3 {{
      margin: 0;
      font: 700 clamp(0.90rem, 1.4vw, 1.10rem)/1.15 'Outfit', sans-serif;
      letter-spacing: -0.01em;
      color: var(--ink);
    }}
    .matchup-line {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: nowrap;
    }}
    .team-with-flag {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      flex: 1 1 0;
    }}
    .team-with-flag span {{
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 700;
    }}
    .team-flag {{
      width: 26px;
      height: 18px;
      border-radius: 4px;
      object-fit: cover;
      border: 1px solid rgba(0,40,10,0.10);
      box-shadow: 0 2px 8px rgba(0,40,10,0.10);
      flex: 0 0 auto;
      transition: transform 0.2s;
    }}
    .analysis-card:hover .team-flag {{ transform: scale(1.08); }}
    .matchup-x {{
      font-family: 'Bebas Neue', sans-serif;
      font-size: 1.2rem;
      color: var(--verde);
      letter-spacing: 0.05em;
      flex-shrink: 0;
    }}

    /* ============================================================
       META / MICRO ROW
       ============================================================ */
    .meta-row, .micro-row {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .meta-row span, .micro-row span, .scoreline-list span {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 6px 11px;
      border-radius: 999px;
      background: rgba(0,40,14,0.05);
      border: 1px solid rgba(0,40,14,0.08);
      font: 500 0.80rem/1 'Outfit', sans-serif;
      color: var(--ink-mid);
    }}

    /* ============================================================
       PROBABILIDADES
       ============================================================ */
    .prob-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }}
    .prob-item {{
      padding: 12px 10px;
      border-radius: 14px;
      background: rgba(0,40,14,0.03);
      border: 1px solid rgba(0,40,14,0.06);
      text-align: center;
      transition: background 0.2s;
    }}
    .prob-item:hover {{ background: rgba(0,156,59,0.08); border-color: rgba(0,156,59,0.20); }}
    .prob-item label, .block-label {{
      display: block;
      font: 700 0.66rem/1 'Outfit', sans-serif;
      text-transform: uppercase;
      letter-spacing: 0.10em;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .prob-item strong {{
      display: block;
      font-family: 'Bebas Neue', sans-serif;
      font-size: 1.35rem;
      letter-spacing: 0.04em;
      color: var(--verde-deep);
      margin-top: 0;
    }}

    /* ============================================================
       SCORELINE LIST
       ============================================================ */
    .scoreline-list {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .scoreline-list span {{
      font: 600 0.80rem/1 'Outfit', sans-serif;
      background: rgba(255,223,0,0.12);
      border-color: rgba(255,200,0,0.25);
    }}
    .scoreline-list span strong {{ font-weight: 700; color: #7a5a00; }}

    /* ============================================================
       DUAL CARD (Sugestão IA + Placar Real)
       ============================================================ */
    .dual-card {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .prediction-card, .result-card {{
      padding: 14px;
      border-radius: 16px;
      border: 1px solid var(--line);
      min-height: 90px;
    }}
    .prediction-card {{
      background: linear-gradient(135deg, rgba(0,104,40,0.07), rgba(255,223,0,0.10));
      border-color: rgba(0,104,40,0.14);
    }}
    .prediction-card strong {{
      font-family: 'Bebas Neue', sans-serif;
      font-size: 1.5rem;
      letter-spacing: 0.06em;
      color: var(--verde-deep);
      margin-top: 6px;
    }}
    .result-card {{
      background: linear-gradient(135deg, rgba(0,39,118,0.06), rgba(255,255,255,0.90));
      border-color: rgba(0,39,118,0.12);
    }}
    .result-card strong {{
      font-family: 'Bebas Neue', sans-serif;
      font-size: 1.5rem;
      letter-spacing: 0.06em;
      color: var(--azul);
      margin-top: 6px;
    }}
    .waiting-card, .pending-result-card {{
      background: rgba(255,255,255,0.72);
      border-color: rgba(0,40,14,0.08);
    }}
    .pending-result-card strong {{ color: var(--muted) !important; font-size: 1rem !important; font-family: 'Outfit', sans-serif !important; font-weight: 600 !important; letter-spacing: 0 !important; }}
    .prediction-card p, .result-card p {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 0.78rem;
      line-height: 1.58;
    }}

    /* ============================================================
       EMPTY STATE
       ============================================================ */
    .empty-state {{
      display: none;
      padding: 40px 20px;
      text-align: center;
      color: var(--muted);
      border-radius: 20px;
      background: rgba(255,255,255,0.80);
      border: 2px dashed rgba(0,104,40,0.18);
      font: 600 1.05rem/1.5 'Outfit', sans-serif;
    }}
    .empty-state::before {{ content: "🔍"; display: block; font-size: 2.4rem; margin-bottom: 12px; }}

    /* ============================================================
       BOTÃO ATUALIZAR PAINEL
       ============================================================ */
    .btn-update {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 0 22px;
      height: 50px;
      border-radius: 14px;
      background: linear-gradient(135deg, var(--verde), var(--verde-deep));
      border: none;
      color: #fff;
      font: 700 0.88rem/1 'Outfit', sans-serif;
      letter-spacing: 0.05em;
      cursor: pointer;
      box-shadow: 0 4px 16px rgba(0,104,40,0.30);
      transition: transform 0.2s, box-shadow 0.2s, opacity 0.2s;
      white-space: nowrap;
      flex-shrink: 0;
    }}
    .btn-update:hover:not(:disabled) {{
      transform: translateY(-2px);
      box-shadow: 0 8px 28px rgba(0,104,40,0.40);
    }}
    .btn-update:active:not(:disabled) {{ transform: translateY(0); }}
    .btn-update:disabled {{ opacity: 0.70; cursor: not-allowed; }}
    .btn-update svg {{ width: 16px; height: 16px; flex-shrink: 0; }}
    .btn-update .spin {{ animation: rotateSpin 0.9s linear infinite; }}
    @keyframes rotateSpin {{ to {{ transform: rotate(360deg); }} }}
    .update-toast {{
      display: none;
      position: fixed;
      bottom: 28px;
      right: 28px;
      z-index: 999;
      padding: 14px 20px;
      border-radius: 16px;
      font: 600 0.90rem/1.4 'Outfit', sans-serif;
      box-shadow: 0 12px 40px rgba(0,20,8,0.22);
      max-width: 320px;
      animation: toastIn 0.3s cubic-bezier(.22,.68,0,1.2);
    }}
    @keyframes toastIn {{
      from {{ opacity: 0; transform: translateY(16px); }}
      to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .update-toast.ok  {{ background: var(--verde-deep); color: #fff; border: 1px solid rgba(255,255,255,0.14); }}
    .update-toast.err {{ background: #7a1010; color: #fff; border: 1px solid rgba(255,255,255,0.14); }}

    /* ============================================================
       FOOTER NOTE
       ============================================================ */
    .footer-note {{
      margin-top: 24px;
      color: var(--muted);
      font: 400 0.88rem/1.72 'Outfit', sans-serif;
      padding: 18px 20px;
      background: rgba(255,255,255,0.60);
      border-radius: 16px;
      border: 1px solid rgba(0,40,14,0.08);
    }}
    .footer-note a {{ color: var(--verde); font-weight: 600; text-decoration: none; }}
    .footer-note a:hover {{ text-decoration: underline; }}

    /* ============================================================
       MISC
       ============================================================ */
    .copy {{ margin: 10px 0 0; color: var(--muted); line-height: 1.72; font-size: 0.94rem; }}
    .info-box {{
      padding: 16px 18px;
      border-radius: 16px;
      background: linear-gradient(135deg, rgba(255,223,0,0.14), rgba(0,104,40,0.06));
      border: 1px solid rgba(255,200,0,0.22);
      line-height: 1.75;
      color: #273445;
      font-size: 0.93rem;
    }}
    .mini-box {{ padding: 14px; border-radius: 16px; background: rgba(255,255,255,0.72); border: 1px solid var(--line); }}
    .mini-box strong {{ display: block; font-size: 0.95rem; font-weight: 700; }}
    .mini-box p {{ margin: 6px 0 0; color: var(--muted); line-height: 1.62; font-size: 0.88rem; }}

    /* ============================================================
       RESPONSIVE
       ============================================================ */
    @media (max-width: 1200px) {{
      .hero-grid {{ grid-template-columns: 1fr; }}
      .analysis-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 720px) {{
      .hero {{ padding: 26px 20px 32px; border-radius: 20px; }}
      .main-card {{ padding: 18px; border-radius: 20px; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .analysis-grid {{ grid-template-columns: 1fr; }}
      .prob-grid, .dual-card {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: clamp(2.2rem, 10vw, 3.5rem); }}
    }}
    @media (max-width: 440px) {{
      .stats {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="hero-grid">
        <div>
          <span class="hero-tag">Copa do Mundo 2026 | HTML dedicado</span>
          <div class="hero-stage">Tema visual inspirado em poster oficial, estadio e noite de decisao.</div>
          <h1>Todos os jogos da rota com placares sugeridos por IA.</h1>
          <p>
            Esta pagina cruza a raspagem local da competicao com o modelo probabilistico do projeto
            para sugerir placares, probabilidades 1X2, tendencia de gols e leitura de valor. O resumo em
            linguagem natural usa IA apenas sobre a base interna, sem inventar dados externos.
          </p>
          <div class="tournament-band">
            <span class="band-chip"><i></i>Rota Copa do Mundo 2026</span>
            <span class="band-chip"><i></i>Modelo local + IA resumidora</span>
            <span class="band-chip"><i></i>Placares sugeridos e retroanalise</span>
          </div>
        </div>
        <div class="hero-side">
          <div class="hero-panel">
            <span>Arquivo gerado</span>
            <strong>{escape(generated_at)}</strong>
          </div>
          <div class="hero-panel">
            <span>Cobertura atual da rota</span>
            <strong>{stats['total_matches']} jogos</strong>
          </div>
        </div>
      </div>

      <div class="stats">
        <div class="stat">
          <span>Finalizados</span>
          <strong>{stats['finished_matches']}</strong>
        </div>
        <div class="stat">
          <span>Agendados</span>
          <strong>{stats['scheduled_matches']}</strong>
        </div>
        <div class="stat">
          <span>Com previsao</span>
          <strong>{stats['predicted_matches']}</strong>
        </div>
        <div class="stat">
          <span>Confianca media</span>
          <strong>{stats['avg_confidence']}/100</strong>
        </div>
      </div>
    </section>

    <section class="grid">
      <aside class="side-card">
        <div>
          <h2>Leitura IA</h2>
          <p class="copy">Resumo executivo produzido a partir das probabilidades e scorelines calculados localmente.</p>
        </div>
        <div class="ai-summary">{escape(ai_summary)}</div>

        <div>
          <h2>Metodologia</h2>
          <div class="method-list">
            <div class="method-item">
              <strong>Probabilidades 1X2</strong>
              <p>O motor usa o modelo Poisson do proprio projeto para estimar a chance de cada selecao e empate em campo neutro.</p>
            </div>
            <div class="method-item">
              <strong>Placar sugerido</strong>
              <p>O placar principal vem do scoreline com maior probabilidade entre as combinacoes calculadas pelo modelo.</p>
            </div>
            <div class="method-item">
              <strong>Camada de IA</strong>
              <p>A IA resume os dados em linguagem natural, mas nao cria fatos novos fora da base interna raspada.</p>
            </div>
          </div>
        </div>

        <div class="mini-meta">
          <div class="mini-box">
            <strong>Fonte atual</strong>
            <p>BetExplorer na rota configurada como Copa do Mundo 2026 dentro do projeto.</p>
          </div>
          <div class="mini-box">
            <strong>Leitura honesta</strong>
            <p>Se a rota tiver poucos jogos, a pagina continua funcionando, mas a confianca do modelo cai e isso aparece no card.</p>
          </div>
        </div>
      </aside>

      <section class="main-card">
        <div class="toolbar">
          <button class="btn-update" id="btnUpdateCopa" onclick="refreshCopa(this)">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:16px;height:16px;margin-right:8px;vertical-align:middle;position:relative;top:-1px;"><polyline points="1 4 1 10 7 10"></polyline><path d="M3.51 15a9 9 0 1 0 .49-3.51"></path></svg>
            Atualizar Painel
          </button>
          <div class="search-box">
            <label class="mini-label" for="searchInput">Buscar time ou data</label>
            <input id="searchInput" type="text" placeholder="Ex.: Jamaica, Bolivia, 31.03." />
          </div>
          <div>
            <label class="mini-label">Filtrar status</label>
            <div class="filter-row">
              <button class="filter-btn active" data-filter="todos" type="button">Todos</button>
              <button class="filter-btn" data-filter="finalizado" type="button">Finalizados</button>
              <button class="filter-btn" data-filter="agendado" type="button">Agendados</button>
            </div>
          </div>
        </div>

        <div id="resultsCount" class="results-count"></div>
        <div id="emptyState" class="empty-state">Nenhum jogo combina com os filtros aplicados.</div>
        <div id="matchGrid" class="match-grid">{cards_html}</div>

        <p class="footer-note">
          Arquivo gerado automaticamente por <code>gerar_copa_mundo_html.py</code>. Os placares sugeridos sao
          estimativas estatisticas e nao garantias de resultado. A cobertura atual depende do que a rota publica
          no momento da raspagem.
        </p>
      </section>
    </section>
  </main>

  <script>
    const cards = Array.from(document.querySelectorAll('.match-card'));
    const searchInput = document.getElementById('searchInput');
    const filterButtons = Array.from(document.querySelectorAll('.filter-btn'));
    const resultsCount = document.getElementById('resultsCount');
    const emptyState = document.getElementById('emptyState');

    let currentFilter = 'todos';

    function applyFilters() {{
      const query = (searchInput.value || '').trim().toLowerCase();
      let visible = 0;

      cards.forEach((card) => {{
        const status = card.dataset.status || '';
        const searchBlob = card.dataset.search || '';
        const statusOk = currentFilter === 'todos' || status === currentFilter;
        const searchOk = !query || searchBlob.includes(query);
        const show = statusOk && searchOk;
        card.style.display = show ? '' : 'none';
        if (show) visible += 1;
      }});

      resultsCount.textContent = visible === 1 ? '1 jogo visivel' : `${{visible}} jogos visiveis`;
      emptyState.style.display = visible === 0 ? 'block' : 'none';
    }}

    filterButtons.forEach((button) => {{
      button.addEventListener('click', () => {{
        filterButtons.forEach((item) => item.classList.remove('active'));
        button.classList.add('active');
        currentFilter = button.dataset.filter || 'todos';
        applyFilters();
      }});
    }});

    searchInput.addEventListener('input', applyFilters);
    applyFilters();

    function reloadPortalShell() {{
      const url = new URL(window.location.href);
      url.searchParams.set('view', 'copa');
      url.searchParams.set('copa_reload_nonce', Date.now().toString());
      try {{
        if (window.top && window.top !== window) {{
          window.top.location.href = url.toString();
          return;
        }}
      }} catch (error) {{}}
      location.href = url.toString();
    }}

    async function refreshCopa(btn) {{
      btn.disabled = true;
      const originalHtml = btn.innerHTML;
      btn.innerHTML = `<svg class="spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:16px;height:16px;margin-right:8px;vertical-align:middle;position:relative;top:-1px;"><line x1="12" y1="2" x2="12" y2="6"></line><line x1="12" y1="18" x2="12" y2="22"></line><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line><line x1="2" y1="12" x2="6" y2="12"></line><line x1="18" y1="12" x2="22" y2="12"></line><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line><line x1="16.24" y1="4.93" x2="19.07" y2="7.76"></line></svg> Atualizando...`;
      
      try {{
        const res = await fetch("http://127.0.0.1:8765/api/refresh-copa", {{ method: "POST" }});
        const data = await res.json();
        if (data.ok) {{
          showToast("Painel atualizado com sucesso! Recarregando...", "ok");
          setTimeout(() => reloadPortalShell(), 1500);
        }} else {{
          showToast("Erro: " + (data.error || "Falha ao atualizar."), "err");
          btn.disabled = false;
          btn.innerHTML = originalHtml;
        }}
      }} catch (err) {{
        showToast("Erro de conexao. O servidor Portal AI esta rodando?", "err");
        btn.disabled = false;
        btn.innerHTML = originalHtml;
      }}
    }}

    function showToast(msg, type) {{
      const el = document.getElementById("updateToast");
      if (!el) return;
      el.textContent = msg;
      el.className = "update-toast " + type;
      el.style.display = "block";
      setTimeout(() => el.style.display = "none", 4000);
    }}
  </script>
</body>
</html>
"""


def _strip_flag_emoji(text: str) -> str:
    cleaned_chars: list[str] = []
    for char in text:
        codepoint = ord(char)
        if 0x1F1E6 <= codepoint <= 0x1F1FF:
            continue
        if 0xE0061 <= codepoint <= 0xE007F:
            continue
        cleaned_chars.append(char)
    return "".join(cleaned_chars)


def _clean_schedule_text(text: str) -> str:
    no_flags = _strip_flag_emoji(text or "")
    normalized = unicodedata.normalize("NFKC", no_flags)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.replace("I nglaterra", "Inglaterra")


def _parse_time_to_24h(time_text: str) -> tuple[int, int]:
    match = re.match(r"(\d{1,2})(?::(\d{2}))?\s*([ap])\.m\.", time_text.strip(), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Horario invalido: {time_text}")

    hour = int(match.group(1))
    minute = int(match.group(2) or "00")
    period = match.group(3).lower()

    if period == "a":
        if hour == 12:
            hour = 0
    else:
        if hour != 12:
            hour += 12

    return hour, minute


def _convert_et_to_brt(date_label: str, time_et: str) -> str:
    try:
        brt_dt = _schedule_datetime_brt(date_label, time_et)
    except Exception:
        return "-"
    return brt_dt.strftime("%d/%m %H:%M")


def _stage_label_pt(raw_stage: str) -> str:
    mapping = {
        "Ronda de grupos": "Fase de grupos",
        "Dieciseisavos de Final:": "16 avos de final",
        "Octavos de Final:": "Oitavas de final",
        "Cuartos de Final": "Quartas de final",
        "Semifinales": "Semifinais",
        "Partido por el Tercer Lugar": "Disputa do 3º lugar",
        "Final": "Final",
    }
    return mapping.get(raw_stage, raw_stage)


def _fetch_telemundo_schedule_lines() -> list[str]:
    response = requests.get(
        TELEMUNDO_SCHEDULE_URL,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return [_clean_schedule_text(line) for line in soup.get_text("\n", strip=True).splitlines()]


def _parse_published_schedule() -> list[dict[str, str]]:
    lines = _fetch_telemundo_schedule_lines()
    try:
        start_idx = lines.index("Hora (ET) | Partido | Cadena") + 1
    except ValueError as exc:
        raise ValueError("Nao consegui localizar a tabela da Copa do Mundo 2026 na fonte publicada.") from exc

    date_pattern = re.compile(
        r"^(Lunes|Martes|Miércoles|Miercoles|Jueves|Viernes|Sábado|Sabado|Domingo)\s+\d{1,2}\s+de\s+(junio|julio)$",
        re.IGNORECASE,
    )
    time_pattern = re.compile(r"^(\d{1,2}(?::\d{2})?\s*[ap]\.m\.\s+\(ET\))\s*\|\s*(.*?)\s*(?:\|\s*(.*))?$", re.IGNORECASE)
    stage_lines = {
        "Ronda de grupos",
        "Fase de eliminación directa",
        "Dieciseisavos de Final:",
        "Octavos de Final:",
        "Cuartos de Final",
        "Semifinales",
        "Partido por el Tercer Lugar",
        "Final",
    }

    records: list[dict[str, str]] = []
    current_stage = "Ronda de grupos"
    current_date = ""
    i = start_idx

    while i < len(lines):
        line = lines[i]
        if not line or line.startswith("Opciones de anuncios") or line.startswith("©"):
            break

        if line in stage_lines:
            if line != "Fase de eliminación directa":
                current_stage = line
            i += 1
            continue

        if date_pattern.match(line):
            current_date = line
            i += 1
            continue

        time_match = time_pattern.match(line)
        if not time_match:
            i += 1
            continue

        time_et = time_match.group(1).replace("  ", " ").strip()
        match_text = (time_match.group(2) or "").strip()
        channel_text = (time_match.group(3) or "").strip()

        if not match_text:
            i += 1
            if i < len(lines):
                match_text = lines[i]
        elif "vs" not in _ascii_slug(match_text) and i + 1 < len(lines):
            next_line = lines[i + 1]
            if next_line and not next_line.startswith("|"):
                i += 1
                match_text = f"{match_text} {next_line}".strip()

        if not channel_text:
            i += 1
            if i < len(lines):
                next_line = lines[i]
                if next_line.startswith("|"):
                    channel_text = next_line.lstrip("|").strip()
                elif " | " not in next_line:
                    channel_text = next_line.strip()

        matchup = _clean_schedule_text(match_text)
        channel = _clean_schedule_text(channel_text)
        if not matchup:
            i += 1
            continue

        record = {
            "stage": _stage_label_pt(current_stage),
            "date_label": current_date,
            "time_et": time_et.replace(" (ET)", ""),
            "time_brt": _convert_et_to_brt(current_date, time_et),
            "matchup": matchup,
            "channel": channel or "-",
            "search": f"{current_stage} {current_date} {matchup}".lower(),
        }
        records.append(record)
        i += 1

    if len(records) < 100:
        raise ValueError(f"A tabela publicada retornou apenas {len(records)} jogos; o esperado era perto de 104.")

    return records


def _extract_json_block(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _fallback_prior(team: str) -> dict[str, object]:
    return {
        "strength": 62,
        "fifa_rank": None,
        "fifa_points": None,
        "attack": 1.0,
        "defense": 1.0,
        "draw_bias": 1.0,
        "note": "fallback neutro",
        "team": team,
    }


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _safe_int_or_none(value: object) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _safe_float_or_none(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object, default: float) -> float:
    resolved = _safe_float_or_none(value)
    return float(default) if resolved is None else resolved


def _default_model_adjustments() -> dict[str, object]:
    return {
        "draw_multiplier": 1.0,
        "home_goal_multiplier": 1.0,
        "away_goal_multiplier": 1.0,
        "home_advantage_points": 0.0,
        "confidence_bias": 0.0,
        "team_rating_adjustments": {},
    }


def _normalize_model_adjustments_payload(adjustments: dict[str, object] | None) -> dict[str, object]:
    raw = _default_model_adjustments()
    if isinstance(adjustments, dict):
        raw.update(adjustments)
    normalized = {
        "draw_multiplier": round(_clamp(float(raw.get("draw_multiplier", 1.0)), 0.78, 1.48), 3),
        "home_goal_multiplier": round(_clamp(float(raw.get("home_goal_multiplier", 1.0)), 0.72, 1.34), 3),
        "away_goal_multiplier": round(_clamp(float(raw.get("away_goal_multiplier", 1.0)), 0.72, 1.34), 3),
        "home_advantage_points": round(_clamp(float(raw.get("home_advantage_points", 0.0)), -2.0, 3.0), 2),
        "confidence_bias": round(_clamp(float(raw.get("confidence_bias", 0.0)), -12.0, 12.0), 2),
        "team_rating_adjustments": {},
    }

    raw_team_adjustments = raw.get("team_rating_adjustments", {})
    if isinstance(raw_team_adjustments, dict):
        for team_name, raw_shift in raw_team_adjustments.items():
            try:
                normalized["team_rating_adjustments"][_canonical_team_name(str(team_name))] = round(
                    _clamp(float(raw_shift), -18.0, 18.0),
                    2,
                )
            except Exception:
                continue

    normalized["team_rating_adjustments"] = dict(sorted(normalized["team_rating_adjustments"].items()))
    return normalized


def _load_model_adjustments() -> dict[str, object]:
    adjustments = _default_model_adjustments()

    if MODEL_ADJUSTMENTS_FILE.exists():
        try:
            loaded = json.loads(MODEL_ADJUSTMENTS_FILE.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                adjustments.update(loaded)
        except Exception:
            adjustments = _default_model_adjustments()

    normalized = _normalize_model_adjustments_payload(adjustments)
    MODEL_ADJUSTMENTS_FILE.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def get_world_cup_model_adjustments() -> dict[str, object]:
    return _load_model_adjustments()


def update_world_cup_model_adjustments(
    adjustments: dict[str, object] | None = None,
    *,
    reset: bool = False,
) -> dict[str, object]:
    base = _default_model_adjustments() if reset else _load_model_adjustments()
    payload = dict(base)
    if isinstance(adjustments, dict):
        payload.update(adjustments)
    normalized = _normalize_model_adjustments_payload(payload)
    MODEL_ADJUSTMENTS_FILE.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def _request_team_priors_from_ai(teams: list[str]) -> dict[str, dict[str, object]]:
    prompt = (
        "Voce vai montar uma base pre-torneio para a Copa do Mundo 2026. "
        "Para cada selecao abaixo, devolva um JSON com os campos: "
        "team, strength, fifa_rank, fifa_points, attack, defense, draw_bias, note. "
        "Use estes limites: strength inteiro de 45 a 92; attack numero de 0.82 a 1.22; "
        "defense numero de 0.82 a 1.22 onde maior significa defesa melhor; "
        "draw_bias numero de 0.92 a 1.10; note com no maximo 6 palavras. "
        "Seja conservador e coerente com o contexto pre-torneio de 2026. "
        "Nao escreva markdown, nao explique nada, devolva apenas JSON em lista.\n\n"
        "Times:\n- " + "\n- ".join(teams)
    )
    raw = request_nvidia_completion(
        [
            {
                "role": "system",
                "content": (
                    "Voce classifica selecoes nacionais para um motor preditivo de futebol. "
                    "Seja conciso, numerico e consistente."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        top_p=0.85,
        max_tokens=3200,
    )
    parsed = json.loads(_extract_json_block(raw))
    priors: dict[str, dict[str, object]] = {}
    if isinstance(parsed, dict):
        parsed = parsed.get("teams", [])
    for item in parsed:
        if not isinstance(item, dict):
            continue
        team = str(item.get("team") or "").strip()
        if not team:
            continue
        priors[_canonical_team_name(team)] = {
            "team": _canonical_team_name(team),
            "strength": int(round(float(item.get("strength", 62)))),
            "fifa_rank": _safe_int_or_none(item.get("fifa_rank")),
            "fifa_points": _safe_float_or_none(item.get("fifa_points")),
            "attack": round(float(item.get("attack", 1.0)), 3),
            "defense": round(float(item.get("defense", 1.0)), 3),
            "draw_bias": round(float(item.get("draw_bias", 1.0)), 3),
            "note": str(item.get("note") or "base IA").strip()[:80],
        }
    return priors


def _normalize_team_prior(team: str, prior: dict[str, object] | None) -> dict[str, object]:
    fallback = _fallback_prior(team)
    raw = prior if isinstance(prior, dict) else {}
    normalized = {
        "team": _canonical_team_name(str(raw.get("team") or team)),
        "strength": int(round(_clamp(_safe_float(raw.get("strength"), float(fallback["strength"])), 45.0, 92.0))),
        "fifa_rank": _safe_int_or_none(raw.get("fifa_rank")),
        "fifa_points": _safe_float_or_none(raw.get("fifa_points")),
        "attack": round(_clamp(_safe_float(raw.get("attack"), float(fallback["attack"])), 0.82, 1.22), 3),
        "defense": round(_clamp(_safe_float(raw.get("defense"), float(fallback["defense"])), 0.82, 1.22), 3),
        "draw_bias": round(_clamp(_safe_float(raw.get("draw_bias"), float(fallback["draw_bias"])), 0.92, 1.10), 3),
        "note": str(raw.get("note") or fallback["note"]).strip()[:80],
    }
    return normalized


def _team_name_from_fifa_entry(entry: dict[str, object]) -> str | None:
    raw_names = entry.get("TeamName")
    if isinstance(raw_names, list):
        for raw_name in raw_names:
            if isinstance(raw_name, dict) and str(raw_name.get("Description") or "").strip():
                return _canonical_team_name(str(raw_name["Description"]))
    return None


def _fetch_fifa_ranking_map() -> dict[str, dict[str, float | int]]:
    response = requests.get(
        FIFA_MENS_RANKING_URL,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://inside.fifa.com/fifa-world-ranking/men",
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("Results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return {}

    ranking_map: dict[str, dict[str, float | int]] = {}
    for entry in results:
        if not isinstance(entry, dict):
            continue
        team = _team_name_from_fifa_entry(entry)
        rank = _safe_int_or_none(entry.get("Rank"))
        points = _safe_float_or_none(entry.get("DecimalTotalPoints"))
        if team and rank is not None:
            ranking_map[team] = {
                "fifa_rank": rank,
                "fifa_points": round(points, 2) if points is not None else 0.0,
            }
    return ranking_map


def _load_team_priors(teams: list[str]) -> dict[str, dict[str, object]]:
    existing: dict[str, dict[str, object]] = {}
    if TEAM_PRIORS_CACHE.exists():
        try:
            loaded = json.loads(TEAM_PRIORS_CACHE.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = {
                    _canonical_team_name(str(k)): _normalize_team_prior(_canonical_team_name(str(k)), v)
                    for k, v in loaded.items()
                    if isinstance(v, dict)
                }
        except Exception:
            existing = {}

    missing = [team for team in teams if team not in existing]
    if missing:
        try:
            ai_priors = _request_team_priors_from_ai(missing)
            existing.update(ai_priors)
            TEAM_PRIORS_CACHE.write_text(
                json.dumps(existing, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            for team in missing:
                existing[team] = _fallback_prior(team)
            TEAM_PRIORS_CACHE.write_text(
                json.dumps(existing, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    try:
        fifa_ranking = _fetch_fifa_ranking_map()
        for team in teams:
            ranking = fifa_ranking.get(team)
            if ranking:
                existing.setdefault(team, _fallback_prior(team))
                existing[team]["fifa_rank"] = ranking["fifa_rank"]
                existing[team]["fifa_points"] = ranking["fifa_points"]
    except Exception:
        pass

    normalized = {team: _normalize_team_prior(team, existing.get(team)) for team in teams}
    TEAM_PRIORS_CACHE.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def _poisson_probability(goals: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if goals == 0 else 0.0
    return math.exp(-lam) * (lam**goals) / math.factorial(goals)


def _coerce_int_goal(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _add_result_lookup_entry(
    lookup: dict[tuple[str, str, str], dict[str, object]],
    home_team: object,
    away_team: object,
    date_key: str,
    home_goals: object,
    away_goals: object,
    *,
    event_dt: object = None,
    source: str = "",
    source_url: str = "",
) -> None:
    home = _canonical_team_name(str(home_team))
    away = _canonical_team_name(str(away_team))
    home_score = _coerce_int_goal(home_goals)
    away_score = _coerce_int_goal(away_goals)
    if not home or not away or not date_key or home_score is None or away_score is None:
        return

    lookup[(home, away, date_key)] = {
        "home_goals": home_score,
        "away_goals": away_score,
        "actual_score": f"{home_score} x {away_score}",
        "event_dt": event_dt,
        "source": source,
        "source_url": source_url,
    }
    lookup[(away, home, date_key)] = {
        "home_goals": away_score,
        "away_goals": home_score,
        "actual_score": f"{away_score} x {home_score}",
        "event_dt": event_dt,
        "source": source,
        "source_url": source_url,
    }


def _parse_result_date_key(value: object) -> str | None:
    if value is None:
        return None
    value_text = str(value).strip()
    parsed = pd.NaT
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value_text):
        parsed = pd.to_datetime(value_text, errors="coerce", format="%Y-%m-%d")
    elif re.fullmatch(r"\d{2}/\d{2}/\d{4}", value_text):
        parsed = pd.to_datetime(value_text, errors="coerce", format="%d/%m/%Y")
    if pd.isna(parsed):
        parsed = pd.to_datetime(value_text, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return None
    if int(parsed.year) != 2026 or int(parsed.month) not in {6, 7}:
        return None
    return parsed.strftime("%Y-%m-%d")


def _load_world_cup_result_overrides() -> dict[tuple[str, str, str], dict[str, object]]:
    lookup: dict[tuple[str, str, str], dict[str, object]] = {}
    if not RESULTS_OVERRIDES_FILE.exists():
        return lookup
    try:
        rows = json.loads(RESULTS_OVERRIDES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return lookup
    if not isinstance(rows, list):
        return lookup

    for row in rows:
        if not isinstance(row, dict):
            continue
        date_key = _parse_result_date_key(row.get("date"))
        if not date_key:
            continue
        _add_result_lookup_entry(
            lookup,
            row.get("home_team"),
            row.get("away_team"),
            date_key,
            row.get("home_goals"),
            row.get("away_goals"),
            source=str(row.get("source", "")),
            source_url=str(row.get("source_url", "")),
        )
    return lookup


def _load_espn_world_cup_results_lookup() -> dict[tuple[str, str, str], dict[str, object]]:
    lookup: dict[tuple[str, str, str], dict[str, object]] = {}
    headers = {"User-Agent": "Mozilla/5.0"}

    for date_range in ("202606", "202607"):
        try:
            response = requests.get(
                ESPN_WORLD_CUP_SCOREBOARD_URL,
                params={"dates": date_range},
                headers=headers,
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            continue

        events = payload.get("events", [])
        if not isinstance(events, list):
            continue

        for event in events:
            if not isinstance(event, dict):
                continue
            status_type = event.get("status", {}).get("type", {})
            if not isinstance(status_type, dict) or not bool(status_type.get("completed")):
                continue

            event_dt = pd.to_datetime(event.get("date"), errors="coerce", utc=True)
            if pd.isna(event_dt):
                continue
            local_dt = event_dt.tz_convert(APP_TIMEZONE)
            date_key = local_dt.strftime("%Y-%m-%d")

            competitions = event.get("competitions", [])
            competition = competitions[0] if isinstance(competitions, list) and competitions else {}
            competitors = competition.get("competitors", []) if isinstance(competition, dict) else []
            if not isinstance(competitors, list):
                continue

            home_entry = None
            away_entry = None
            for competitor in competitors:
                if not isinstance(competitor, dict):
                    continue
                home_away = str(competitor.get("homeAway", "")).strip().lower()
                if home_away == "home":
                    home_entry = competitor
                elif home_away == "away":
                    away_entry = competitor

            if not home_entry or not away_entry:
                continue

            home_team = home_entry.get("team", {}).get("displayName") if isinstance(home_entry.get("team"), dict) else ""
            away_team = away_entry.get("team", {}).get("displayName") if isinstance(away_entry.get("team"), dict) else ""
            source_url = ""
            links = event.get("links", [])
            if isinstance(links, list):
                for link in links:
                    if isinstance(link, dict) and link.get("href"):
                        source_url = str(link.get("href"))
                        break

            _add_result_lookup_entry(
                lookup,
                home_team,
                away_team,
                date_key,
                home_entry.get("score"),
                away_entry.get("score"),
                event_dt=local_dt,
                source="ESPN",
                source_url=source_url,
            )

    return lookup


def _load_official_results_lookup() -> dict[tuple[str, str, str], dict[str, object]]:
    lookup = _load_espn_world_cup_results_lookup()
    try:
        df = load_competition_matches("Copa do Mundo")
    except Exception:
        lookup.update(_load_world_cup_result_overrides())
        return lookup

    if "status" not in df.columns:
        lookup.update(_load_world_cup_result_overrides())
        return lookup

    finished = df[df["status"] == "Finalizado"].copy()
    if finished.empty:
        lookup.update(_load_world_cup_result_overrides())
        return lookup

    if "event_timestamp" in finished.columns:
        finished["event_dt_local"] = pd.to_datetime(finished["event_timestamp"], errors="coerce", utc=True)
    else:
        finished["event_dt_local"] = pd.NaT

    for row in finished.itertuples(index=False):
        local_dt = None
        date_key = None
        if not pd.isna(row.event_dt_local):
            local_dt = row.event_dt_local.tz_convert(APP_TIMEZONE) if row.event_dt_local.tzinfo else row.event_dt_local
            if int(local_dt.year) == 2026 and int(local_dt.month) in {6, 7}:
                date_key = local_dt.strftime("%Y-%m-%d")
        if not date_key and hasattr(row, "date_text"):
            date_key = _parse_result_date_key(row.date_text)
        if not date_key:
            continue
        _add_result_lookup_entry(
            lookup,
            row.home_team,
            row.away_team,
            date_key,
            row.home_goals,
            row.away_goals,
            event_dt=local_dt,
            source="BetExplorer",
        )
    lookup.update(_load_world_cup_result_overrides())
    return lookup


def _find_official_result(
    lookup: dict[tuple[str, str, str], dict[str, object]],
    home_team: str,
    away_team: str,
    date_key: str,
) -> dict[str, object] | None:
    exact_result = lookup.get((home_team, away_team, date_key))
    if exact_result:
        return exact_result

    parsed_date = pd.to_datetime(date_key, errors="coerce")
    if pd.isna(parsed_date):
        return None

    candidates: list[tuple[int, dict[str, object]]] = []
    for offset in (-1, 1):
        nearby_key = (parsed_date + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        nearby_result = lookup.get((home_team, away_team, nearby_key))
        if nearby_result:
            candidates.append((abs(offset), nearby_result))

    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    return None


def _outcome_from_score(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "Casa"
    if home_goals < away_goals:
        return "Fora"
    return "Empate"


def _outcome_from_scoreline(scoreline: str) -> str | None:
    try:
        home_raw, away_raw = scoreline.split(" x ", 1)
        return _outcome_from_score(int(home_raw), int(away_raw))
    except (AttributeError, TypeError, ValueError):
        return None


def _select_suggested_score(
    scorelines: list[tuple[str, float]],
    predicted_outcome: str,
    *,
    min_relative_probability: float = 0.82,
) -> str:
    if not scorelines:
        return "-"

    top_score, top_probability = scorelines[0]
    if _outcome_from_scoreline(top_score) == predicted_outcome:
        return top_score

    best_for_outcome = next(
        ((score, probability) for score, probability in scorelines if _outcome_from_scoreline(score) == predicted_outcome),
        None,
    )
    if not best_for_outcome:
        return top_score

    score, probability = best_for_outcome
    if top_probability <= 0 or probability >= top_probability * min_relative_probability:
        return score
    return top_score


def _team_form_component(
    team: str,
    team_state: dict[str, dict[str, float]],
    model_adjustments: dict[str, object],
) -> float:
    state = team_state.get(team, {})
    manual_shift = float(model_adjustments.get("team_rating_adjustments", {}).get(team, 0.0))
    return float(state.get("rating_shift", 0.0)) + manual_shift


def _strength_from_fifa_rank(rank: object) -> float | None:
    fifa_rank = _safe_int_or_none(rank)
    if fifa_rank is None or fifa_rank <= 0:
        return None
    clamped_rank = _clamp(float(fifa_rank), 1.0, 211.0)
    return 92.0 - ((clamped_rank - 1.0) / 210.0) * 47.0


def _effective_team_strength(prior: dict[str, object]) -> float:
    internal_strength = _safe_float(prior.get("strength"), 62.0)
    rank_strength = _strength_from_fifa_rank(prior.get("fifa_rank"))
    if rank_strength is None:
        return internal_strength
    return (rank_strength * 0.65) + (internal_strength * 0.35)


def _predict_match(
    home_team: str,
    away_team: str,
    priors: dict[str, dict[str, object]],
    team_state: dict[str, dict[str, float]],
    global_state: dict[str, float],
    model_adjustments: dict[str, object],
) -> dict[str, object]:
    home_prior = priors.get(home_team, _fallback_prior(home_team))
    away_prior = priors.get(away_team, _fallback_prior(away_team))

    rating_diff = (
        _effective_team_strength(home_prior) - _effective_team_strength(away_prior)
        + _team_form_component(home_team, team_state, model_adjustments)
        - _team_form_component(away_team, team_state, model_adjustments)
        + float(model_adjustments.get("home_advantage_points", 0.0))
    )
    rating_factor = math.exp(rating_diff / 80.0)

    expected_home = (
        1.10
        * float(home_prior["attack"])
        / max(float(away_prior["defense"]), 0.78)
        * rating_factor
        * float(global_state.get("home_goal_multiplier", 1.0))
    )
    expected_away = (
        1.10
        * float(away_prior["attack"])
        / max(float(home_prior["defense"]), 0.78)
        / rating_factor
        * float(global_state.get("away_goal_multiplier", 1.0))
    )

    expected_home = max(0.22, min(3.4, expected_home))
    expected_away = max(0.18, min(3.1, expected_away))

    draw_multiplier = (
        float(global_state.get("draw_multiplier", 1.0))
        * ((float(home_prior["draw_bias"]) + float(away_prior["draw_bias"])) / 2.0)
    )

    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    scorelines: list[tuple[str, float]] = []
    for hg in range(7):
        ph = _poisson_probability(hg, expected_home)
        for ag in range(7):
            pa = _poisson_probability(ag, expected_away)
            p = ph * pa
            if hg == ag:
                p *= draw_multiplier
            scorelines.append((f"{hg} x {ag}", p))
            if hg > ag:
                home_win += p
            elif hg == ag:
                draw += p
            else:
                away_win += p

    total = home_win + draw + away_win
    if total <= 0:
        total = 1.0
    home_win /= total
    draw /= total
    away_win /= total
    scorelines = [(score, prob / total) for score, prob in scorelines]
    scorelines.sort(key=lambda item: item[1], reverse=True)

    predicted_outcome = max(
        [("Casa", home_win), ("Empate", draw), ("Fora", away_win)],
        key=lambda item: item[1],
    )[0]
    suggested_score = _select_suggested_score(scorelines, predicted_outcome)

    confidence_raw = (
        44
        + abs(home_win - away_win) * 42
        + scorelines[0][1] * 55
        + abs(rating_diff) * 0.18
        + float(model_adjustments.get("confidence_bias", 0.0))
    )
    confidence = int(round(max(28.0, min(92.0, confidence_raw))))

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "expected_home": expected_home,
        "expected_away": expected_away,
        "suggested_score": suggested_score,
        "predicted_outcome": predicted_outcome,
        "confidence": confidence,
        "scorelines": scorelines[:4],
        "analysis": (
            f"Base IA: {home_team} ({home_prior['note']}) vs {away_team} ({away_prior['note']}). "
            f"O estudo ve {home_team} {home_win * 100:.1f}%, empate {draw * 100:.1f}% "
            f"e {away_team} {away_win * 100:.1f}%."
        ),
    }


def _update_live_state(
    home_team: str,
    away_team: str,
    prediction: dict[str, object],
    result: dict[str, object],
    team_state: dict[str, dict[str, float]],
    history: list[dict[str, float]],
    global_state: dict[str, float],
    model_adjustments: dict[str, object],
) -> None:
    actual_home = int(result["home_goals"])
    actual_away = int(result["away_goals"])
    expected_home = float(prediction["expected_home"])
    expected_away = float(prediction["expected_away"])
    actual_outcome = _outcome_from_score(actual_home, actual_away)

    home_points = 3 if actual_outcome == "Casa" else 1 if actual_outcome == "Empate" else 0
    away_points = 3 if actual_outcome == "Fora" else 1 if actual_outcome == "Empate" else 0
    expected_home_points = 3 * float(prediction["home_win"]) + float(prediction["draw"])
    expected_away_points = 3 * float(prediction["away_win"]) + float(prediction["draw"])

    team_state.setdefault(home_team, {"rating_shift": 0.0})
    team_state.setdefault(away_team, {"rating_shift": 0.0})
    sample_weight = min(1.0, max(0.25, (len(history) + 1) / 10.0))
    rating_learning_rate = 1.05 * sample_weight
    team_state[home_team]["rating_shift"] = _clamp(
        float(team_state[home_team].get("rating_shift", 0.0)) + (home_points - expected_home_points) * rating_learning_rate,
        -12.0,
        12.0,
    )
    team_state[away_team]["rating_shift"] = _clamp(
        float(team_state[away_team].get("rating_shift", 0.0)) + (away_points - expected_away_points) * rating_learning_rate,
        -12.0,
        12.0,
    )

    history.append(
        {
            "pred_draw": float(prediction["draw"]),
            "actual_draw": 1.0 if actual_outcome == "Empate" else 0.0,
            "pred_home_goals": expected_home,
            "pred_away_goals": expected_away,
            "actual_home_goals": float(actual_home),
            "actual_away_goals": float(actual_away),
        }
    )

    if len(history) >= 4:
        avg_pred_draw = sum(item["pred_draw"] for item in history) / len(history)
        avg_actual_draw = sum(item["actual_draw"] for item in history) / len(history)
        avg_pred_home = sum(item["pred_home_goals"] for item in history) / len(history)
        avg_pred_away = sum(item["pred_away_goals"] for item in history) / len(history)
        avg_actual_home = sum(item["actual_home_goals"] for item in history) / len(history)
        avg_actual_away = sum(item["actual_away_goals"] for item in history) / len(history)
        multiplier_weight = min(0.55, len(history) / 18.0)

        if avg_pred_draw > 0:
            draw_ratio = avg_actual_draw / avg_pred_draw
            global_state["draw_multiplier"] = _clamp(
                float(model_adjustments.get("draw_multiplier", 1.0)) * (1.0 + (draw_ratio - 1.0) * multiplier_weight),
                0.88,
                1.22,
            )
        if avg_pred_home > 0:
            home_ratio = avg_actual_home / avg_pred_home
            global_state["home_goal_multiplier"] = _clamp(
                float(model_adjustments.get("home_goal_multiplier", 1.0)) * (1.0 + (home_ratio - 1.0) * multiplier_weight),
                0.88,
                1.18,
            )
        if avg_pred_away > 0:
            away_ratio = avg_actual_away / avg_pred_away
            global_state["away_goal_multiplier"] = _clamp(
                float(model_adjustments.get("away_goal_multiplier", 1.0)) * (1.0 + (away_ratio - 1.0) * multiplier_weight),
                0.88,
                1.18,
            )


def _build_enriched_schedule(records: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    parsed_records: list[dict[str, object]] = []
    for item in records:
        dt_brt = _schedule_datetime_brt(item["date_label"], f"{item['time_et']} (ET)")
        home_team, away_team = _extract_match_teams(item["matchup"])
        display_home_team, display_away_team = _split_matchup_display(item["matchup"])
        parsed_records.append(
            {
                **item,
                "datetime_brt": dt_brt,
                "datetime_brt_iso": dt_brt.isoformat(),
                "date_sp": dt_brt.strftime("%d/%m/%Y"),
                "time_sp": dt_brt.strftime("%H:%M"),
                "home_team": home_team,
                "away_team": away_team,
                "display_home_team": display_home_team,
                "display_away_team": display_away_team,
                "search": f"{item['stage']} {item['date_label']} {dt_brt.strftime('%d/%m/%Y')} {item['matchup']}".lower(),
            }
        )

    parsed_records.sort(key=lambda item: item["datetime_brt"])
    teams = sorted({team for item in parsed_records for team in [item["home_team"], item["away_team"]] if team})
    priors = _load_team_priors(teams) if teams else {}
    model_adjustments = _load_model_adjustments()
    results_lookup = _load_official_results_lookup()
    team_state: dict[str, dict[str, float]] = {}
    history: list[dict[str, float]] = []
    global_state = {
        "draw_multiplier": float(model_adjustments["draw_multiplier"]),
        "home_goal_multiplier": float(model_adjustments["home_goal_multiplier"]),
        "away_goal_multiplier": float(model_adjustments["away_goal_multiplier"]),
    }

    enriched: list[dict[str, object]] = []
    exact_hits = 0
    outcome_hits = 0
    finished_matches = 0

    for item in parsed_records:
        home_team = item["home_team"]
        away_team = item["away_team"]
        record = dict(item)
        record["has_prediction"] = False
        record["has_result"] = False

        if home_team and away_team:
            home_prior = priors.get(str(home_team), _fallback_prior(str(home_team)))
            away_prior = priors.get(str(away_team), _fallback_prior(str(away_team)))
            record.update(
                {
                    "home_fifa_rank": home_prior.get("fifa_rank"),
                    "home_fifa_points": home_prior.get("fifa_points"),
                    "away_fifa_rank": away_prior.get("fifa_rank"),
                    "away_fifa_points": away_prior.get("fifa_points"),
                    "home_fifa_rank_display": _fmt_fifa_rank(home_prior.get("fifa_rank")),
                    "home_fifa_points_display": _fmt_fifa_points(home_prior.get("fifa_points")),
                    "away_fifa_rank_display": _fmt_fifa_rank(away_prior.get("fifa_rank")),
                    "away_fifa_points_display": _fmt_fifa_points(away_prior.get("fifa_points")),
                }
            )
            prediction = _predict_match(home_team, away_team, priors, team_state, global_state, model_adjustments)
            record.update(
                {
                    "has_prediction": True,
                    "suggested_score": prediction["suggested_score"],
                    "predicted_outcome": prediction["predicted_outcome"],
                    "predicted_outcome_label": _team_label_from_market(
                        str(prediction["predicted_outcome"]),
                        str(home_team),
                        str(away_team),
                    ),
                    "home_win_pct": _fmt_pct(prediction["home_win"]),
                    "draw_pct": _fmt_pct(prediction["draw"]),
                    "away_win_pct": _fmt_pct(prediction["away_win"]),
                    "expected_goals": f"{prediction['expected_home']:.2f} x {prediction['expected_away']:.2f}",
                    "confidence": prediction["confidence"],
                    "analysis": prediction["analysis"],
                    "top_scores": prediction["scorelines"],
                    "pred_home_goals_raw": prediction["expected_home"],
                    "pred_away_goals_raw": prediction["expected_away"],
                }
            )

            result = _find_official_result(
                results_lookup,
                home_team,
                away_team,
                record["datetime_brt"].strftime("%Y-%m-%d"),
            )
            if result:
                actual_outcome = _outcome_from_score(int(result["home_goals"]), int(result["away_goals"]))
                exact_hit = record["suggested_score"] == result["actual_score"]
                outcome_hit = record["predicted_outcome"] == actual_outcome
                finished_matches += 1
                exact_hits += 1 if exact_hit else 0
                outcome_hits += 1 if outcome_hit else 0
                record.update(
                    {
                        "has_result": True,
                        "actual_score": result["actual_score"],
                        "actual_outcome": actual_outcome,
                        "exact_hit": exact_hit,
                        "outcome_hit": outcome_hit,
                        "result_label": (
                            "Acerto exato" if exact_hit else "Acerto de direcao" if outcome_hit else "Erro da leitura"
                        ),
                    }
                )
                _update_live_state(
                    home_team,
                    away_team,
                    prediction,
                    result,
                    team_state,
                    history,
                    global_state,
                    model_adjustments,
                )

        enriched.append(record)

    stats = {
        "total_matches": len(enriched),
        "predicted_matches": sum(1 for item in enriched if item.get("has_prediction")),
        "finished_matches": finished_matches,
        "exact_hit_pct": round((exact_hits / finished_matches) * 100, 1) if finished_matches else 0.0,
        "outcome_hit_pct": round((outcome_hits / finished_matches) * 100, 1) if finished_matches else 0.0,
        "draw_multiplier": round(global_state["draw_multiplier"], 3),
        "home_goal_multiplier": round(global_state["home_goal_multiplier"], 3),
        "away_goal_multiplier": round(global_state["away_goal_multiplier"], 3),
        "base_draw_multiplier": round(float(model_adjustments["draw_multiplier"]), 3),
        "base_home_goal_multiplier": round(float(model_adjustments["home_goal_multiplier"]), 3),
        "base_away_goal_multiplier": round(float(model_adjustments["away_goal_multiplier"]), 3),
        "home_advantage_points": round(float(model_adjustments["home_advantage_points"]), 2),
        "confidence_bias": round(float(model_adjustments["confidence_bias"]), 2),
        "manual_team_adjustment_count": len(model_adjustments.get("team_rating_adjustments", {})),
        "manual_team_adjustments_preview": ", ".join(list(model_adjustments.get("team_rating_adjustments", {}).keys())[:6]) or "-",
        "model_adjustments": model_adjustments,
        "date_count": len({item["date_sp"] for item in enriched}),
        "group_matches": sum(1 for item in enriched if item["stage"] == "Fase de grupos"),
    }
    return enriched, stats


def _schedule_card_html(item: dict[str, str]) -> str:
    stage_class = (
        "stage-group" if item["stage"] == "Fase de grupos"
        else "stage-final"
        if item["stage"] == "Final"
        else "stage-knockout"
    )
    return f"""
    <article class="schedule-card" data-stage="{escape(item['stage'])}" data-search="{escape(item['search'])}">
      <div class="schedule-top">
        <span class="stage-pill {stage_class}">{escape(item['stage'])}</span>
        <span class="date-pill">{escape(item['date_label'])}</span>
      </div>
      <h3>{escape(item['matchup'])}</h3>
      <div class="time-grid">
        <div class="time-box">
          <span>Horario ET</span>
          <strong>{escape(item['time_et'])}</strong>
        </div>
        <div class="time-box">
          <span>Horario de Brasilia</span>
          <strong>{escape(item['time_brt'])}</strong>
        </div>
      </div>
      <div class="channel-box">
        <span>Referencia publicada</span>
        <strong>{escape(item['channel'])}</strong>
      </div>
    </article>
    """


def build_world_cup_schedule_html() -> str:
    records = _parse_published_schedule()
    generated_at = datetime.now(APP_TIMEZONE).strftime("%d/%m/%Y %H:%M:%S")
    total_matches = len(records)
    group_matches = sum(1 for item in records if item["stage"] == "Fase de grupos")
    knockout_matches = total_matches - group_matches
    date_count = len({item["date_label"] for item in records})
    cards_html = "".join(_schedule_card_html(item) for item in records)

    stage_options = "".join(
        f"<option>{escape(stage)}</option>"
        for stage in [
            "Fase de grupos",
            "16 avos de final",
            "Oitavas de final",
            "Quartas de final",
            "Semifinais",
            "Disputa do 3o lugar",
            "Final",
        ]
    )

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Copa do Mundo 2026 | Tabela Oficial com Datas e Horarios</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
    :root {{
      --ink: #10233a;
      --muted: #5f6d7f;
      --paper: rgba(255,255,253,0.99);
      --line: rgba(16,35,58,0.10);
      --royal: #1746a2;
      --berry: #7a1738;
      --gold: #d7a93f;
      --mint: #2e8b7c;
      --night: #0a1a2f;
      --shadow: 0 24px 60px rgba(16,35,58,0.14);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Space Grotesk", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 8%, rgba(215,169,63,0.24), transparent 20%),
        radial-gradient(circle at 90% 6%, rgba(122,23,56,0.18), transparent 22%),
        repeating-linear-gradient(90deg, rgba(255,255,255,0.08) 0, rgba(255,255,255,0.08) 1px, transparent 1px, transparent 120px),
        linear-gradient(180deg, #f7f0e1 0%, #ede3cf 52%, #f2ebde 100%);
    }}
    .shell {{ max-width: 1380px; margin: 0 auto; padding: 24px 18px 40px; }}
    .hero {{
      position: relative;
      overflow: hidden;
      padding: 30px;
      border-radius: 32px;
      color: #fff8eb;
      background:
        radial-gradient(circle at 84% 18%, rgba(215,169,63,0.20), transparent 20%),
        radial-gradient(circle at 18% 24%, rgba(217,72,65,0.16), transparent 24%),
        linear-gradient(135deg, rgba(10,26,47,0.99), rgba(23,70,162,0.92));
      box-shadow: 0 28px 80px rgba(16,35,58,0.28);
    }}
    .hero::before {{
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(120deg, transparent 0%, rgba(255,248,235,0.06) 40%, transparent 72%),
        repeating-linear-gradient(135deg, rgba(255,248,235,0.03) 0, rgba(255,248,235,0.03) 10px, transparent 10px, transparent 28px);
      pointer-events: none;
    }}
    .hero-grid {{ position: relative; z-index: 1; display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(280px, 0.9fr); gap: 22px; }}
    .hero-tag {{
      display: inline-flex; padding: 8px 14px; border-radius: 999px; border: 1px solid rgba(255,248,235,0.16);
      background: rgba(255,248,235,0.10); font-size: 12px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
    }}
    h1 {{ margin: 14px 0 12px; max-width: 12ch; font: 700 clamp(2.4rem, 5vw, 4.7rem)/0.94 "Fraunces", serif; letter-spacing: -0.04em; }}
    .hero p {{ max-width: 72ch; margin: 0; color: rgba(255,248,235,0.82); line-height: 1.75; }}
    .hero-band {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
    .hero-chip {{
      display: inline-flex; align-items: center; gap: 8px; padding: 10px 14px; border-radius: 999px;
      background: rgba(255,248,235,0.10); border: 1px solid rgba(255,248,235,0.12); font-size: 0.84rem; font-weight: 700;
    }}
    .hero-chip i {{ width: 9px; height: 9px; border-radius: 50%; background: linear-gradient(135deg, var(--gold), #f5d991); }}
    .hero-side {{ display: grid; gap: 12px; }}
    .hero-panel {{
      padding: 18px; border-radius: 22px; background: rgba(255,248,235,0.10); border: 1px solid rgba(255,248,235,0.14);
      backdrop-filter: blur(10px);
    }}
    .hero-panel span {{ display: block; font-size: 11px; text-transform: uppercase; letter-spacing: 0.12em; color: rgba(255,248,235,0.74); }}
    .hero-panel strong {{ display: block; margin-top: 8px; font-size: 1.7rem; letter-spacing: -0.04em; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-top: 18px; }}
    .stat {{ padding: 16px; border-radius: 20px; background: rgba(255,248,235,0.10); border: 1px solid rgba(255,248,235,0.14); }}
    .stat span {{ display: block; font-size: 11px; text-transform: uppercase; letter-spacing: 0.12em; color: rgba(255,248,235,0.72); }}
    .stat strong {{ display: block; margin-top: 10px; font-size: 1.9rem; letter-spacing: -0.05em; }}
    .layout {{ display: grid; grid-template-columns: minmax(280px, 320px) minmax(0, 1fr); gap: 18px; margin-top: 18px; align-items: start; }}
    .side-card, .main-card {{
      position: relative; background: var(--paper); border: 1px solid rgba(255,255,255,0.55); border-radius: 28px;
      box-shadow: var(--shadow); backdrop-filter: blur(14px);
    }}
    .side-card::before, .main-card::before {{
      content: ""; position: absolute; left: 22px; top: 0; width: 180px; height: 4px; border-radius: 999px;
      background: linear-gradient(90deg, var(--berry), var(--gold), var(--mint));
    }}
    .side-card {{ position: sticky; top: 18px; padding: 22px; display: grid; gap: 18px; }}
    .main-card {{ padding: 22px; }}
    h2 {{ margin: 0; font: 700 clamp(1.35rem, 3vw, 2.1rem)/1.05 "Fraunces", serif; letter-spacing: -0.03em; }}
    .copy {{ margin: 10px 0 0; color: var(--muted); line-height: 1.72; }}
    .info-box {{
      padding: 16px; border-radius: 20px; background: linear-gradient(135deg, rgba(215,169,63,0.14), rgba(122,23,56,0.08));
      border: 1px solid rgba(215,169,63,0.16); line-height: 1.75; color: #273445;
    }}
    .mini-box {{ padding: 14px; border-radius: 18px; background: rgba(255,255,255,0.68); border: 1px solid var(--line); }}
    .mini-box strong {{ display: block; font-size: 0.98rem; }}
    .mini-box p {{ margin: 6px 0 0; color: var(--muted); line-height: 1.62; }}
    .toolbar {{ display: flex; justify-content: space-between; gap: 14px; flex-wrap: wrap; align-items: end; margin-bottom: 18px; }}
    .field {{ min-width: min(100%, 260px); flex: 1; }}
    .field label {{ display: block; margin-bottom: 8px; font-size: 0.74rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
    .field input, .field select {{
      width: 100%; height: 50px; border-radius: 16px; border: 1px solid var(--line); background: rgba(255,255,255,0.92);
      font: inherit; padding: 0 16px; color: var(--ink); outline: none;
    }}
    .results-count {{ color: var(--muted); font-size: 0.95rem; margin-bottom: 10px; }}
    .schedule-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .schedule-card {{
      position: relative; display: grid; gap: 14px; padding: 18px; border-radius: 24px;
      background: radial-gradient(circle at top right, rgba(215,169,63,0.10), transparent 22%), linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,244,236,0.92));
      border: 1px solid rgba(16,35,58,0.08); box-shadow: 0 12px 30px rgba(16,35,58,0.08);
    }}
    .schedule-card::before {{
      content: ""; position: absolute; inset: 0 auto 0 0; width: 5px; border-radius: 24px 0 0 24px;
      background: linear-gradient(180deg, var(--berry), var(--gold), var(--mint));
    }}
    .schedule-top {{ display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap; align-items: center; }}
    .stage-pill, .date-pill {{
      display: inline-flex; align-items: center; padding: 8px 12px; border-radius: 999px; font-size: 0.76rem; font-weight: 700;
      text-transform: uppercase; letter-spacing: 0.08em;
    }}
    .stage-group {{ background: rgba(23,70,162,0.12); color: var(--royal); }}
    .stage-knockout {{ background: rgba(122,23,56,0.12); color: var(--berry); }}
    .stage-final {{ background: rgba(215,169,63,0.18); color: #8b6718; }}
    .date-pill {{ background: rgba(16,35,58,0.05); color: var(--muted); }}
    .schedule-card h3 {{ margin: 0; font: 700 clamp(1.1rem, 2.2vw, 1.6rem)/1.18 "Fraunces", serif; letter-spacing: -0.03em; }}
    .time-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .time-box {{
      padding: 14px; border-radius: 18px; background: linear-gradient(135deg, rgba(10,26,47,0.98), rgba(23,70,162,0.94));
      color: #fff8eb; border: 1px solid rgba(215,169,63,0.20);
    }}
    .time-box span, .channel-box span {{ display: block; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.08em; color: rgba(255,248,235,0.72); }}
    .time-box strong, .channel-box strong {{ display: block; margin-top: 8px; font-size: 1.18rem; letter-spacing: -0.03em; }}
    .channel-box {{
      padding: 14px; border-radius: 18px; background: linear-gradient(135deg, rgba(122,23,56,0.05), rgba(215,169,63,0.10)), rgba(255,255,255,0.82);
      border: 1px solid var(--line);
    }}
    .channel-box span {{ color: var(--muted); }}
    .empty-state {{
      display: none; padding: 30px 18px; text-align: center; color: var(--muted); border-radius: 22px;
      background: rgba(255,255,255,0.72); border: 1px dashed rgba(16,35,58,0.16);
    }}
    .footer-note {{ margin-top: 18px; color: var(--muted); font-size: 0.92rem; line-height: 1.72; }}
    .footer-note a {{ color: var(--royal); }}
    @media (max-width: 1080px) {{
      .hero-grid, .layout, .schedule-grid {{ grid-template-columns: 1fr; }}
      .side-card {{ position: static; }}
    }}
    @media (max-width: 720px) {{
      .stats, .time-grid {{ grid-template-columns: 1fr; }}
      .hero, .side-card, .main-card {{ padding: 18px; border-radius: 22px; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="hero-grid">
        <div>
          <span class="hero-tag">Copa do Mundo 2026 | Datas e horarios</span>
          <h1>Tabela completa publicada da Copa do Mundo de 2026.</h1>
          <p>
            Esta pagina foi organizada para seguir a grade publicada da Copa do Mundo de 2026, com os confrontos,
            as datas e os horarios exibidos em ET e convertidos para Brasilia. O foco aqui e a tabela oficial
            publicada, nao as previsoes do modelo.
          </p>
          <div class="hero-band">
            <span class="hero-chip"><i></i>104 jogos previstos</span>
            <span class="hero-chip"><i></i>ET + horario de Brasilia</span>
            <span class="hero-chip"><i></i>Fonte FIFA + grade publicada</span>
          </div>
        </div>
        <div class="hero-side">
          <div class="hero-panel">
            <span>Pagina gerada em</span>
            <strong>{generated_at}</strong>
          </div>
          <div class="hero-panel">
            <span>Recorte da tabela</span>
            <strong>11/06/2026 a 19/07/2026</strong>
          </div>
        </div>
      </div>
      <div class="stats">
        <div class="stat"><span>Total de jogos</span><strong>{total_matches}</strong></div>
        <div class="stat"><span>Fase de grupos</span><strong>{group_matches}</strong></div>
        <div class="stat"><span>Mata-mata</span><strong>{knockout_matches}</strong></div>
        <div class="stat"><span>Dias com jogos</span><strong>{date_count}</strong></div>
      </div>
    </section>
    <section class="layout">
      <aside class="side-card">
        <div>
          <h2>Como ler</h2>
          <p class="copy">Cada card mostra a fase, a data publicada, o horario original em ET e a conversao para Brasilia.</p>
        </div>
        <div class="info-box">
          A grade usada aqui segue a divulgacao posterior ao sorteio final da Copa do Mundo de 2026.
          Quando a fonte publica atualizar horario ou confronto, basta regenerar a pagina para refletir a mudanca.
        </div>
        <div class="mini-box">
          <strong>Fonte-base de horarios</strong>
          <p>Grade publicada em 2 de marco de 2026 pela Telemundo em horario ET.</p>
        </div>
        <div class="mini-box">
          <strong>Validacao institucional</strong>
          <p>FIFA informou em 6 de dezembro de 2025 que a tabela de 104 jogos foi fechada apos o sorteio final.</p>
        </div>
        <div class="mini-box">
          <strong>Horario do Brasil</strong>
          <p>A conversao usa America/New_York para ET e America/Sao_Paulo para Brasilia.</p>
        </div>
      </aside>
      <section class="main-card">
        <div class="toolbar">
          <div class="field">
            <label for="searchInput">Buscar time, data ou fase</label>
            <input id="searchInput" type="text" placeholder="Ex.: Brasil, 26 de junio, final" />
          </div>
          <div class="field" style="max-width:280px;">
            <label for="stageFilter">Filtrar fase</label>
            <select id="stageFilter">
              <option value="">Todas</option>
              {stage_options}
            </select>
          </div>
        </div>
        <div id="resultsCount" class="results-count"></div>
        <div id="emptyState" class="empty-state">Nenhum jogo combina com os filtros aplicados.</div>
        <div id="scheduleGrid" class="schedule-grid">{cards_html}</div>
        <p class="footer-note">
          Fontes consultadas: <a href="{FIFA_SCHEDULE_RELEASE_URL}">Inside FIFA</a> e
          <a href="{TELEMUNDO_SCHEDULE_URL}">Telemundo Deportes</a>.
          Os horarios em ET seguem a publicacao consultada; a coluna de Brasilia foi convertida automaticamente no gerador.
        </p>
      </section>
    </section>
  </main>
  <div id="updateToast" class="update-toast"></div>
  <script>
    const cards = Array.from(document.querySelectorAll('.schedule-card'));
    const searchInput = document.getElementById('searchInput');
    const stageFilter = document.getElementById('stageFilter');
    const resultsCount = document.getElementById('resultsCount');
    const emptyState = document.getElementById('emptyState');

    function applyFilters() {{
      const query = (searchInput.value || '').trim().toLowerCase();
      const stage = (stageFilter.value || '').trim();
      let visible = 0;

      cards.forEach((card) => {{
        const searchBlob = card.dataset.search || '';
        const cardStage = card.dataset.stage || '';
        const matchesQuery = !query || searchBlob.includes(query);
        const matchesStage = !stage || cardStage === stage;
        const show = matchesQuery && matchesStage;
        card.style.display = show ? '' : 'none';
        if (show) visible += 1;
      }});

      resultsCount.textContent = visible === 1 ? '1 jogo visivel' : `${{visible}} jogos visiveis`;
      emptyState.style.display = visible === 0 ? 'block' : 'none';
    }}

    searchInput.addEventListener('input', applyFilters);
    stageFilter.addEventListener('change', applyFilters);
    applyFilters();
  </script>
</body>
</html>
"""


def _analysis_card_html(item: dict[str, object]) -> str:
    stage_class = (
        "stage-group" if item["stage"] == "Fase de grupos"
        else "stage-final"
        if item["stage"] == "Final"
        else "stage-knockout"
    )
    top_scores = "".join(
        f"<span>{escape(score)} <strong>{prob * 100:.1f}%</strong></span>"
        for score, prob in item.get("top_scores", [])
    ) or "<span>Sem combinacoes calculadas</span>"

    prediction_block = (
        f"""
        <div class="prediction-card">
          <span class="block-label">Sugestao IA</span>
          <strong>{escape(str(item.get('suggested_score', '-')))}</strong>
          <p>{escape(str(item.get('analysis', '-')))}</p>
        </div>
        """
        if item.get("has_prediction")
        else """
        <div class="prediction-card waiting-card">
          <span class="block-label">Sugestao IA</span>
          <strong>Aguardando definicao</strong>
          <p>O confronto ainda nao tem dois times definidos, entao a leitura entra assim que a chave fechar.</p>
        </div>
        """
    )

    result_block = (
        f"""
        <div class="result-card">
          <span class="block-label">Placar real</span>
          <strong>{escape(str(item.get('actual_score', '-')))}</strong>
          <p>{escape(str(item.get('result_label', '-')))}</p>
        </div>
        """
        if item.get("has_result")
        else """
        <div class="result-card pending-result-card">
          <span class="block-label">Placar real</span>
          <strong>Ainda nao finalizado</strong>
          <p>Assim que a atualizacao trouxer o resultado oficial, este card mostra o score real e entra na calibracao.</p>
        </div>
        """
    )
    home_html = _team_html(str(item.get("display_home_team") or item.get("home_team") or ""), item.get("home_team"))
    away_html = _team_html(str(item.get("display_away_team") or item.get("away_team") or ""), item.get("away_team"))
    home_rank_html = (
        f"{escape(str(item.get('display_home_team') or item.get('home_team') or 'Selecao A'))} "
        f"{escape(str(item.get('home_fifa_rank_display', '-')))}"
        f" · {escape(str(item.get('home_fifa_points_display', 'sem pontos')))}"
    )
    away_rank_html = (
        f"{escape(str(item.get('display_away_team') or item.get('away_team') or 'Selecao B'))} "
        f"{escape(str(item.get('away_fifa_rank_display', '-')))}"
        f" · {escape(str(item.get('away_fifa_points_display', 'sem pontos')))}"
    )

    return f"""
    <article class="analysis-card"
      data-stage="{escape(str(item['stage']))}"
      data-date="{escape(str(item['date_sp']))}"
      data-search="{escape(str(item['search']))}">
      <div class="analysis-top">
        <span class="stage-pill {stage_class}">{escape(str(item['stage']))}</span>
        <span class="date-pill">{escape(str(item['date_sp']))} &bull; {escape(str(item['time_sp']))}</span>
      </div>
      <h3 class="matchup-line">{home_html}<span class="matchup-x">x</span>{away_html}</h3>
      <div class="ranking-row">
        <span>Ranking FIFA {home_rank_html}</span>
        <span>Ranking FIFA {away_rank_html}</span>
      </div>
      <div class="meta-row">
        <span>SP {escape(str(item['time_sp']))}</span>
        <span>ET {escape(str(item['time_et']))}</span>
        <span>Confianca {escape(str(item.get('confidence', '-')))}</span>
      </div>
      <div class="prob-grid">
        <div class="prob-item"><label>{escape(str(item.get('home_team', 'Selecao A')))}</label><strong>{escape(str(item.get('home_win_pct', '-')))}</strong></div>
        <div class="prob-item"><label>Empate</label><strong>{escape(str(item.get('draw_pct', '-')))}</strong></div>
        <div class="prob-item"><label>{escape(str(item.get('away_team', 'Selecao B')))}</label><strong>{escape(str(item.get('away_win_pct', '-')))}</strong></div>
      </div>
      <div class="micro-row">
        <span>Gols esperados {escape(str(item.get('expected_goals', '-')))}</span>
        <span>Leitura {escape(str(item.get('predicted_outcome_label', '-')))}</span>
      </div>
      <div class="scoreline-list">{top_scores}</div>
      <div class="dual-card">
        {prediction_block}
        {result_block}
      </div>
    </article>
    """


def build_world_cup_schedule_html() -> str:
    raw_records = _parse_published_schedule()
    records, stats = _build_enriched_schedule(raw_records)
    generated_at = datetime.now(APP_TIMEZONE).strftime("%d/%m/%Y %H:%M:%S")
    cards_html = "".join(_analysis_card_html(item) for item in records)

    stage_options = "".join(
        f"<option>{escape(stage)}</option>"
        for stage in [
            "Fase de grupos",
            "16 avos de final",
            "Oitavas de final",
            "Quartas de final",
            "Semifinais",
            "Disputa do 3º lugar",
            "Final",
        ]
    )
    date_options = "".join(
        f"<option>{escape(date_value)}</option>"
        for date_value in sorted({str(item['date_sp']) for item in records}, key=lambda value: datetime.strptime(value, "%d/%m/%Y"))
    )
    model_adjustments_json = json.dumps(stats.get("model_adjustments", _default_model_adjustments()), ensure_ascii=False)
    model_summary_html = f"""
        <div class="model-grid">
          <div class="model-card model-card-wide">
            <span class="model-kicker">Modelo ajustavel</span>
            <strong>Todos os {stats['total_matches']} jogos oficiais entram no painel; {stats['predicted_matches']} ja tem leitura de placar.</strong>
            <p>
              O motor usa a tabela oficial da Copa, calcula probabilidades por confronto e recalibra o modelo quando resultados reais entram.
              Agora voce pode ajustar os criterios direto na tela pela Central de Calibragem e aplicar em 1 clique.
            </p>
          </div>
          <div class="model-card">
            <span class="model-kicker">Empate</span>
            <strong>{stats['base_draw_multiplier']:.3f} -> {stats['draw_multiplier']:.3f}</strong>
            <p>Base manual e multiplicador efetivo apos recalibracao.</p>
          </div>
          <div class="model-card">
            <span class="model-kicker">Gols selecao A</span>
            <strong>{stats['base_home_goal_multiplier']:.3f} -> {stats['home_goal_multiplier']:.3f}</strong>
            <p>Ajuste de volume de gols para o primeiro time listado no confronto.</p>
          </div>
          <div class="model-card">
            <span class="model-kicker">Gols selecao B</span>
            <strong>{stats['base_away_goal_multiplier']:.3f} -> {stats['away_goal_multiplier']:.3f}</strong>
            <p>Ajuste de volume de gols para o segundo time listado no confronto.</p>
          </div>
          <div class="model-card">
            <span class="model-kicker">Parametros extras</span>
            <strong>Vantagem neutra {stats['home_advantage_points']:.2f} | Confianca {stats['confidence_bias']:+.1f}</strong>
            <p>{stats['manual_team_adjustment_count']} selecoes com ajuste manual. Preview: {escape(str(stats['manual_team_adjustments_preview']))}</p>
          </div>
        </div>
    """

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Copa do Mundo 2026 | IA por Confronto</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
    :root {{
      --ink: #10233a;
      --muted: #5f6d7f;
      --paper: rgba(255,255,255,0.88);
      --line: rgba(16,35,58,0.10);
      --royal: #1c56b8;
      --berry: #0f6a3c;
      --gold: #f5c400;
      --mint: #35b869;
      --shadow: 0 24px 60px rgba(16,35,58,0.14);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      position: relative;
      margin: 0;
      min-height: 100vh;
      font-family: "Space Grotesk", sans-serif;
      color: var(--ink);
      background:
        linear-gradient(180deg, #0a2c1d 0%, #0e5a35 20%, #1c56b8 42%, #efe6d8 42.2%, #efe6d8 100%);
      overflow-x: hidden;
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background:
        radial-gradient(ellipse at 50% 16%, rgba(255,255,255,0.18) 0 10%, transparent 11%),
        radial-gradient(ellipse at 50% 16%, rgba(255,255,255,0.08) 0 19%, transparent 20%),
        radial-gradient(ellipse at 50% 16%, rgba(8,20,37,0.92) 0 29%, transparent 30%),
        radial-gradient(ellipse at 50% 16%, rgba(8,20,37,0.52) 0 39%, transparent 40%),
        linear-gradient(180deg, rgba(255,255,255,0.08), transparent 18%);
      opacity: 0.72;
      z-index: 0;
    }}
    body::after {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background:
        radial-gradient(circle at 16% 26%, rgba(53,184,105,0.30) 0 8%, transparent 9%),
        radial-gradient(circle at 24% 20%, rgba(245,196,0,0.28) 0 7%, transparent 8%),
        radial-gradient(circle at 80% 22%, rgba(28,86,184,0.34) 0 8%, transparent 9%),
        radial-gradient(circle at 72% 30%, rgba(53,184,105,0.24) 0 7%, transparent 8%),
        conic-gradient(from 210deg at 18% 26%, transparent 0 14deg, rgba(53,184,105,0.34) 14deg 70deg, transparent 70deg 360deg),
        conic-gradient(from 160deg at 82% 24%, transparent 0 18deg, rgba(245,196,0,0.32) 18deg 84deg, transparent 84deg 360deg),
        linear-gradient(180deg, transparent 0 42%, rgba(16,106,60,0.30) 43%, rgba(16,106,60,0.12) 100%);
      filter: blur(36px) saturate(130%);
      opacity: 0.8;
      z-index: 0;
    }}
    .shell {{ position: relative; z-index: 1; max-width: 1440px; margin: 0 auto; padding: 24px 18px 40px; }}
    .hero {{
      position: relative;
      overflow: hidden;
      padding: 30px;
      border-radius: 32px;
      color: #fff8eb;
      background:
        radial-gradient(circle at 82% 18%, rgba(245,196,0,0.26), transparent 18%),
        radial-gradient(circle at 18% 18%, rgba(53,184,105,0.20), transparent 22%),
        linear-gradient(135deg, rgba(10,61,35,0.98), rgba(16,106,60,0.96) 46%, rgba(28,86,184,0.94));
      box-shadow: 0 28px 80px rgba(16,35,58,0.28);
    }}
    .hero::before {{
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(120deg, transparent 0%, rgba(255,248,235,0.08) 34%, transparent 66%),
        linear-gradient(90deg, rgba(255,255,255,0.04) 0 1px, transparent 1px 28px),
        linear-gradient(135deg, transparent 0 36%, rgba(245,196,0,0.18) 36% 42%, transparent 42% 100%),
        linear-gradient(225deg, transparent 0 64%, rgba(53,184,105,0.18) 64% 70%, transparent 70% 100%);
      pointer-events: none;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      left: -6%;
      right: -6%;
      bottom: -38px;
      height: 168px;
      border-radius: 52% 52% 0 0;
      background:
        radial-gradient(ellipse at center, rgba(31,108,67,0.92) 0 32%, rgba(31,108,67,0.0) 62%),
        linear-gradient(180deg, rgba(255,255,255,0.18), rgba(255,255,255,0.00) 62%),
        repeating-linear-gradient(90deg, rgba(255,255,255,0.08) 0 2px, transparent 2px 92px);
      opacity: 0.88;
      pointer-events: none;
    }}
    .hero-poster-mark {{
      position: absolute;
      right: 24px;
      top: 10px;
      z-index: 0;
      font: 700 clamp(5rem, 18vw, 12rem)/0.8 "Fraunces", serif;
      letter-spacing: -0.09em;
      color: rgba(255,255,255,0.10);
      text-shadow:
        0 0 30px rgba(255,255,255,0.08),
        0 0 60px rgba(245,196,0,0.14);
      pointer-events: none;
      user-select: none;
    }}
    .hero-grid {{ position: relative; z-index: 1; display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(280px, 0.9fr); gap: 22px; }}
    .hero-tag {{
      display: inline-flex; padding: 9px 16px; border-radius: 999px; border: 1px solid rgba(255,248,235,0.20);
      background: linear-gradient(135deg, rgba(255,255,255,0.16), rgba(255,255,255,0.05));
      font-size: 16px; font-weight: 800; letter-spacing: 0.09em; text-transform: uppercase;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }}
    h1 {{ margin: 14px 0 12px; max-width: 12ch; font: 700 clamp(2.4rem, 5vw, 4.7rem)/0.94 "Fraunces", serif; letter-spacing: -0.04em; }}
    .hero p {{ max-width: 72ch; margin: 0; color: rgba(255,248,235,0.82); line-height: 1.75; }}
    .hero-band {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
    .hero-chip {{
      display: inline-flex; align-items: center; gap: 8px; padding: 10px 14px; border-radius: 999px;
      background: linear-gradient(135deg, rgba(255,255,255,0.16), rgba(255,255,255,0.06));
      border: 1px solid rgba(255,248,235,0.16); font-size: 0.84rem; font-weight: 700;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }}
    .hero-chip i {{ width: 9px; height: 9px; border-radius: 50%; background: linear-gradient(135deg, var(--gold), #f5d991); }}
    .hero-side {{ display: grid; gap: 12px; }}
    .hero-panel {{
      padding: 18px;
      border-radius: 22px;
      background: linear-gradient(135deg, rgba(255,255,255,0.28), rgba(255,255,255,0.14));
      border: 1px solid rgba(255,248,235,0.30);
      backdrop-filter: blur(10px);
      box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.12),
        0 12px 28px rgba(7,19,35,0.18);
    }}
    .hero-panel span {{ display: block; font-size: 11px; text-transform: uppercase; letter-spacing: 0.12em; color: rgba(255,248,235,0.74); }}
    .hero-panel strong {{
      display: block;
      margin-top: 8px;
      font-size: 2rem;
      letter-spacing: -0.05em;
      color: #ffffff;
      text-shadow:
        0 2px 0 rgba(7,19,35,0.22),
        0 6px 16px rgba(255,255,255,0.10);
    }}
    .stats {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 14px; margin-top: 18px; }}
    .stat {{
      position: relative;
      padding: 16px;
      border-radius: 20px;
      background:
        radial-gradient(circle at top right, rgba(255,255,255,0.20), transparent 28%),
        linear-gradient(160deg, rgba(255,255,255,0.26), rgba(33,66,120,0.30));
      border: 1px solid rgba(255,248,235,0.34);
      box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.18),
        0 14px 28px rgba(7,19,35,0.28);
      backdrop-filter: blur(14px);
    }}
    .stat span {{
      display: block;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: rgba(255,248,235,0.96);
      font-weight: 700;
      text-shadow: 0 1px 0 rgba(7,19,35,0.20);
    }}
    .stat strong {{
      display: block;
      margin-top: 10px;
      font-size: 2.35rem;
      letter-spacing: -0.06em;
      color: #ffffff;
      text-shadow:
        0 2px 0 rgba(7,19,35,0.26),
        0 10px 24px rgba(255,255,255,0.16);
    }}
    .stat::after {{
      content: "";
      position: absolute;
      left: 0;
      right: 0;
      bottom: 0;
      height: 4px;
      border-radius: 0 0 20px 20px;
      background: linear-gradient(90deg, rgba(255,82,82,0.85), rgba(255,193,7,0.92), rgba(46,204,113,0.82), rgba(54,162,235,0.88));
      background: linear-gradient(90deg, rgba(16,106,60,0.92), rgba(245,196,0,0.96), rgba(28,86,184,0.92));
    }}
    .stat::before {{
      content: "";
      position: absolute;
      inset: 0;
      border-radius: inherit;
      background: linear-gradient(180deg, rgba(255,255,255,0.08), transparent 45%);
      pointer-events: none;
    }}
    .layout {{ display: block; margin-top: 18px; }}
    .main-card {{
      position: relative; background: linear-gradient(180deg, rgba(255,255,255,1), var(--paper)); border: 1px solid rgba(255,255,255,0.92); border-radius: 28px;
      box-shadow: 0 26px 60px rgba(16,35,58,0.10); backdrop-filter: blur(14px);
    }}
    .main-card::before {{
      content: ""; position: absolute; left: 22px; top: 0; width: 180px; height: 4px; border-radius: 999px;
      background: linear-gradient(90deg, var(--berry), var(--gold), var(--mint));
    }}
    .main-card::after {{
      content: "";
      position: absolute;
      inset: 0;
      border-radius: inherit;
      background:
        radial-gradient(circle at top right, rgba(215,169,63,0.10), transparent 18%),
        radial-gradient(circle at bottom left, rgba(28,86,184,0.08), transparent 22%),
        linear-gradient(135deg, transparent 0 36%, rgba(245,196,0,0.05) 36% 39%, transparent 39% 100%),
        linear-gradient(225deg, transparent 0 62%, rgba(53,184,105,0.05) 62% 65%, transparent 65% 100%);
      pointer-events: none;
    }}
    .main-card {{ padding: 22px; }}
    h2 {{ margin: 0; font: 700 clamp(1.35rem, 3vw, 2.1rem)/1.05 "Fraunces", serif; letter-spacing: -0.03em; }}
    .copy {{ margin: 10px 0 0; color: var(--muted); line-height: 1.72; }}
    .info-box {{
      padding: 16px; border-radius: 20px; background: linear-gradient(135deg, rgba(215,169,63,0.14), rgba(122,23,56,0.08));
      border: 1px solid rgba(215,169,63,0.16); line-height: 1.75; color: #273445;
    }}
    .mini-box {{ padding: 14px; border-radius: 18px; background: rgba(255,255,255,0.68); border: 1px solid var(--line); }}
    .mini-box strong {{ display: block; font-size: 0.98rem; }}
    .mini-box p {{ margin: 6px 0 0; color: var(--muted); line-height: 1.62; }}
    .btn-update {{
      position: relative;
      isolation: isolate;
      display: inline-flex;
      align-items: center;
      gap: 12px;
      min-width: 214px;
      min-height: 58px;
      padding: 10px 16px 10px 12px;
      border: 1px solid rgba(16,35,58,0.10);
      border-radius: 20px;
      cursor: pointer;
      font: inherit;
      text-align: left;
      color: var(--ink);
      background:
        linear-gradient(135deg, rgba(255,255,255,0.96), rgba(247,250,245,0.94));
      box-shadow:
        0 14px 30px rgba(16,35,58,0.10),
        inset 0 1px 0 rgba(255,255,255,0.90);
      transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
      appearance: none;
      overflow: hidden;
    }}
    .btn-update::before {{
      content: "";
      position: absolute;
      inset: 0;
      z-index: -1;
      background:
        linear-gradient(120deg, transparent 0 36%, rgba(245,196,0,0.14) 36% 52%, transparent 52% 100%),
        radial-gradient(circle at top right, rgba(28,86,184,0.12), transparent 28%);
      pointer-events: none;
    }}
    .btn-update:hover:not(:disabled) {{
      transform: translateY(-2px);
      border-color: rgba(16,106,60,0.20);
      box-shadow:
        0 18px 36px rgba(16,35,58,0.14),
        inset 0 1px 0 rgba(255,255,255,0.96);
    }}
    .btn-update:disabled {{ opacity: 0.78; cursor: not-allowed; }}
    .btn-update-icon {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 38px;
      height: 38px;
      border-radius: 14px;
      color: #fffaf0;
      background: linear-gradient(135deg, rgba(16,106,60,0.98), rgba(28,86,184,0.96));
      box-shadow: 0 10px 18px rgba(16,35,58,0.18);
      flex: 0 0 auto;
    }}
    .btn-update svg {{ width: 18px; height: 18px; flex: 0 0 auto; }}
    .btn-update-copy {{
      display: grid;
      gap: 2px;
      min-width: 0;
    }}
    .btn-update-title {{
      display: block;
      font-size: 0.96rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      line-height: 1.05;
    }}
    .btn-update-subtitle {{
      display: block;
      color: var(--muted);
      font-size: 0.73rem;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      line-height: 1.1;
    }}
    .btn-update .spin {{ animation: rotateSpin 0.9s linear infinite; }}
    @keyframes rotateSpin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
    .toolbar {{ display: flex; justify-content: space-between; gap: 14px; flex-wrap: wrap; align-items: end; margin-bottom: 18px; }}
    .toolbar-actions {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    .field {{ min-width: min(100%, 220px); flex: 1; }}
    .field label {{ display: block; margin-bottom: 8px; font-size: 0.74rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
    .field input, .field select, .field textarea {{
      width: 100%; height: 50px; border-radius: 16px; border: 1px solid var(--line); background: rgba(255,255,255,0.92);
      font: inherit; padding: 0 16px; color: var(--ink); outline: none;
    }}
    .field textarea {{
      height: 108px;
      padding: 12px 14px;
      resize: vertical;
      line-height: 1.45;
      border-radius: 14px;
    }}
    .btn-secondary {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 48px;
      padding: 0 16px;
      border-radius: 14px;
      border: 1px solid rgba(16,35,58,0.12);
      background: rgba(255,255,255,0.88);
      color: var(--ink);
      font: inherit;
      font-size: 0.84rem;
      font-weight: 800;
      letter-spacing: 0.02em;
      cursor: pointer;
      transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
      box-shadow: 0 8px 18px rgba(16,35,58,0.08);
    }}
    .btn-secondary:hover:not(:disabled) {{
      transform: translateY(-1px);
      border-color: rgba(16,106,60,0.30);
      box-shadow: 0 12px 22px rgba(16,35,58,0.10);
    }}
    .btn-secondary:disabled {{
      opacity: 0.7;
      cursor: not-allowed;
    }}
    .calibration-panel {{
      display: none;
      margin: 8px 0 16px;
      padding: 16px;
      border-radius: 20px;
      border: 1px solid rgba(16,35,58,0.10);
      background: linear-gradient(160deg, rgba(255,255,255,0.95), rgba(245,250,244,0.94));
      box-shadow: 0 12px 26px rgba(16,35,58,0.08);
    }}
    .calibration-panel.show {{ display: block; }}
    .calibration-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 14px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }}
    .calibration-header strong {{
      font-size: 1.02rem;
      letter-spacing: -0.02em;
    }}
    .calibration-header p {{
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.55;
      max-width: 72ch;
    }}
    .calibration-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 10px;
    }}
    .calibration-actions {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
      margin-top: 10px;
    }}
    .calibration-note {{
      margin: 2px 0 0;
      font-size: 0.78rem;
      color: var(--muted);
      line-height: 1.45;
    }}
    .model-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 0 0 18px;
    }}
    .model-card {{
      position: relative;
      padding: 14px 16px;
      border-radius: 18px;
      background:
        radial-gradient(circle at top right, rgba(245,196,0,0.12), transparent 26%),
        linear-gradient(180deg, rgba(255,255,255,0.94), rgba(247,250,245,0.94));
      border: 1px solid rgba(16,35,58,0.08);
      box-shadow: 0 10px 24px rgba(16,35,58,0.07);
    }}
    .model-card-wide {{ grid-column: span 2; }}
    .model-kicker {{
      display: block;
      font-size: 0.72rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: var(--berry);
    }}
    .model-card strong {{
      display: block;
      margin-top: 8px;
      font-size: 1.04rem;
      letter-spacing: -0.02em;
      color: var(--ink);
    }}
    .model-card p {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.55;
    }}
    .results-count {{ color: var(--muted); font-size: 0.95rem; margin-bottom: 10px; }}
    .analysis-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }}
    .analysis-card {{
      position: relative; display: grid; gap: 14px; padding: 18px; border-radius: 24px;
      background:
        radial-gradient(circle at top right, rgba(215,169,63,0.12), transparent 22%),
        radial-gradient(circle at bottom left, rgba(53,184,105,0.10), transparent 22%),
        linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,244,236,0.92));
      border: 1px solid rgba(16,35,58,0.08); box-shadow: 0 12px 30px rgba(16,35,58,0.08);
      content-visibility: auto;
      contain: layout paint style;
      contain-intrinsic-size: 420px;
    }}
    .analysis-card::before {{
      content: ""; position: absolute; inset: 0 auto 0 0; width: 5px; border-radius: 24px 0 0 24px;
      background: linear-gradient(180deg, var(--berry), var(--gold), var(--mint));
    }}
    .analysis-top {{ display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap; align-items: center; }}
    .stage-pill, .date-pill {{
      display: inline-flex; align-items: center; padding: 8px 12px; border-radius: 999px; font-size: 0.76rem; font-weight: 700;
      text-transform: uppercase; letter-spacing: 0.08em;
    }}
    .stage-group {{ background: rgba(28,86,184,0.12); color: var(--royal); }}
    .stage-knockout {{ background: rgba(16,106,60,0.12); color: var(--berry); }}
    .stage-final {{ background: rgba(245,196,0,0.20); color: #8b6718; }}
    .date-pill {{ background: rgba(16,35,58,0.05); color: var(--muted); }}
    .analysis-card h3 {{ margin: 0; font: 700 clamp(0.95rem, 1.55vw, 1.2rem)/1.1 "Fraunces", serif; letter-spacing: -0.03em; }}
    .matchup-line {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: nowrap;
    }}
    .team-with-flag {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      flex: 1 1 0;
    }}
    .team-with-flag span {{
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .team-flag {{
      width: 22px;
      height: 16px;
      border-radius: 4px;
      object-fit: cover;
      border: 1px solid rgba(16,35,58,0.10);
      box-shadow: 0 4px 10px rgba(16,35,58,0.08);
      flex: 0 0 auto;
    }}
    .matchup-x {{
      color: var(--berry);
      font-size: 0.95em;
      opacity: 0.78;
    }}
    .ranking-row {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }}
    .ranking-row span {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 0;
      padding: 7px 9px;
      border-radius: 12px;
      background: rgba(28,86,184,0.07);
      border: 1px solid rgba(28,86,184,0.10);
      color: var(--ink);
      font-size: 0.78rem;
      font-weight: 750;
      text-align: center;
    }}
    .meta-row, .micro-row {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .meta-row span, .micro-row span, .scoreline-list span {{
      display: inline-flex; align-items: center; gap: 6px; padding: 8px 10px; border-radius: 999px;
      background: rgba(16,35,58,0.05); border: 1px solid rgba(16,35,58,0.08); font-size: 0.84rem;
    }}
    .prob-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    .prob-item {{ padding: 12px; border-radius: 16px; background: rgba(16,35,58,0.04); border: 1px solid rgba(16,35,58,0.06); }}
    .prob-item label, .block-label {{
      display: block; font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted);
    }}
    .prob-item strong, .prediction-card strong, .result-card strong {{
      display: block; margin-top: 8px; font-size: 1.18rem; letter-spacing: -0.03em;
    }}
    .scoreline-list {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .dual-card {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .prediction-card, .result-card {{
      padding: 14px; border-radius: 18px; border: 1px solid var(--line); min-height: 100%;
    }}
    .prediction-card {{ background: linear-gradient(135deg, rgba(122,23,56,0.05), rgba(215,169,63,0.10)), rgba(255,255,255,0.82); }}
    .result-card {{ background: linear-gradient(135deg, rgba(46,139,124,0.10), rgba(255,255,255,0.85)); }}
    .waiting-card, .pending-result-card {{ background: rgba(255,255,255,0.72); }}
    .prediction-card p, .result-card p {{ margin: 6px 0 0; color: var(--muted); line-height: 1.6; }}
    .empty-state {{
      display: none; padding: 30px 18px; text-align: center; color: var(--muted); border-radius: 22px;
      background: rgba(255,255,255,0.72); border: 1px dashed rgba(16,35,58,0.16);
    }}
    .load-more-wrap {{
      display: flex;
      justify-content: center;
      margin-top: 18px;
    }}
    .btn-load-more {{
      display: none;
      align-items: center;
      justify-content: center;
      min-width: 220px;
      height: 48px;
      padding: 0 18px;
      border: 1px solid rgba(16,35,58,0.10);
      border-radius: 999px;
      background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(247,250,245,0.96));
      color: var(--ink);
      font: inherit;
      font-weight: 800;
      cursor: pointer;
      box-shadow: 0 12px 24px rgba(16,35,58,0.08);
    }}
    .btn-load-more:hover {{ transform: translateY(-1px); }}
    .update-toast {{
      display: none;
      position: fixed;
      right: 20px;
      bottom: 20px;
      z-index: 20;
      max-width: min(420px, calc(100vw - 32px));
      padding: 14px 16px;
      border-radius: 16px;
      font-weight: 700;
      box-shadow: 0 18px 36px rgba(16,35,58,0.22);
    }}
    .update-toast.ok {{ background: #0f6a3c; color: #fff; }}
    .update-toast.err {{ background: #8f1f2d; color: #fff; }}
    .footer-note {{ margin-top: 18px; color: var(--muted); font-size: 0.92rem; line-height: 1.72; }}
    .footer-note a {{ color: var(--royal); }}
    @media (max-width: 1160px) {{
      .hero-grid, .analysis-grid, .model-grid {{ grid-template-columns: 1fr; }}
      .model-card-wide {{ grid-column: auto; }}
      .calibration-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 720px) {{
      .stats, .prob-grid, .ranking-row, .dual-card {{ grid-template-columns: 1fr; }}
      .hero, .main-card {{ padding: 18px; border-radius: 22px; }}
      .calibration-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="hero-poster-mark">2026</div>
      <div class="hero-grid">
        <div>
          <span class="hero-tag">Copa do Mundo 2026 | IA por confronto</span>
          <div class="hero-band">
            <span class="hero-chip"><i></i>Horario de Sao Paulo no painel</span>
            <span class="hero-chip"><i></i>Placar sugerido jogo a jogo</span>
            <span class="hero-chip"><i></i>Acerto e recalibracao continua</span>
          </div>
        </div>
        <div class="hero-side">
          <div class="hero-panel"><span>Painel gerado em</span><strong>{generated_at}</strong></div>
          <div class="hero-panel"><span>Janela da competicao</span><strong>11/06/2026 a 19/07/2026</strong></div>
        </div>
      </div>
      <div class="stats">
        <div class="stat"><span>Jogos no painel</span><strong>{stats['total_matches']}</strong></div>
        <div class="stat"><span>Jogos com leitura</span><strong>{stats['predicted_matches']}</strong></div>
        <div class="stat"><span>Jogos finalizados</span><strong>{stats['finished_matches']}</strong></div>
        <div class="stat"><span>Acerto exato</span><strong>{stats['exact_hit_pct']:.1f}%</strong></div>
        <div class="stat"><span>Acerto de direcao</span><strong>{stats['outcome_hit_pct']:.1f}%</strong></div>
      </div>
    </section>
    <section class="layout">
      <section class="main-card">
        <div class="toolbar">
          <div class="toolbar-actions">
            <button class="btn-update" id="btnUpdateCopa" onclick="refreshCopa(this)">
              <span class="btn-update-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-3.51"/></svg>
              </span>
              <span class="btn-update-copy">
                <span class="btn-update-title">Atualizar Painel</span>
                <span class="btn-update-subtitle">Sincronizar dados</span>
              </span>
            </button>
            <button class="btn-secondary" id="btnToggleCopaCalibration" type="button">Central de calibragem</button>
          </div>
          <div class="field">
            <label for="searchInput">Buscar time, data ou fase</label>
            <input id="searchInput" type="text" placeholder="Ex.: Brasil, final, 19/07/2026" />
          </div>
          <div class="field" style="max-width:240px;">
            <label for="dateFilter">Filtrar data</label>
            <select id="dateFilter">
              <option value="">Todas</option>
              {date_options}
            </select>
          </div>
          <div class="field" style="max-width:240px;">
            <label for="stageFilter">Filtrar fase</label>
            <select id="stageFilter">
              <option value="">Todas</option>
              {stage_options}
            </select>
          </div>
        </div>
        <section id="copaCalibrationPanel" class="calibration-panel">
          <div class="calibration-header">
            <div>
              <strong>Critérios do modelo da Copa (editáveis em tela)</strong>
              <p>Ajuste os parâmetros base, clique em recalibrar e o painel regenera automaticamente com os novos critérios.</p>
            </div>
          </div>
          <div class="calibration-grid">
            <div class="field">
              <label for="cfgDrawMultiplier">Multiplicador de empate</label>
              <input id="cfgDrawMultiplier" type="number" min="0.78" max="1.48" step="0.001" />
            </div>
            <div class="field">
              <label for="cfgHomeGoalMultiplier">Multiplicador gols selecao A</label>
              <input id="cfgHomeGoalMultiplier" type="number" min="0.72" max="1.34" step="0.001" />
            </div>
            <div class="field">
              <label for="cfgAwayGoalMultiplier">Multiplicador gols selecao B</label>
              <input id="cfgAwayGoalMultiplier" type="number" min="0.72" max="1.34" step="0.001" />
            </div>
            <div class="field">
              <label for="cfgHomeAdvantagePoints">Vies posicional A/B (pontos)</label>
              <input id="cfgHomeAdvantagePoints" type="number" min="-2.0" max="3.0" step="0.01" />
            </div>
            <div class="field">
              <label for="cfgConfidenceBias">Bias de confiança</label>
              <input id="cfgConfidenceBias" type="number" min="-12.0" max="12.0" step="0.1" />
            </div>
            <div class="field">
              <label for="cfgTeamAdjustments">Ajustes por seleção (JSON)</label>
              <textarea id="cfgTeamAdjustments" placeholder='{{ "Brazil": 1.2, "Argentina": 0.8 }}'></textarea>
              <p class="calibration-note">Formato: objeto JSON com "Seleção": shift entre -18 e +18.</p>
            </div>
          </div>
          <div class="calibration-actions">
            <button class="btn-secondary" id="btnResetCopaCalibration" type="button">Restaurar padrão</button>
            <button class="btn-secondary" id="btnApplyCopaCalibration" type="button">Recalibrar modelo</button>
          </div>
        </section>
        <div id="resultsCount" class="results-count"></div>
        <div id="emptyState" class="empty-state">Nenhum jogo combina com os filtros aplicados.</div>
        <div id="analysisGrid" class="analysis-grid">{cards_html}</div>
        <div class="load-more-wrap">
          <button id="loadMoreBtn" class="btn-load-more" type="button">Carregar mais jogos</button>
        </div>
        {model_summary_html}
        <p class="footer-note">
          Fontes consultadas: <a href="{FIFA_SCHEDULE_RELEASE_URL}">Inside FIFA</a> e
          <a href="{TELEMUNDO_SCHEDULE_URL}">Telemundo Deportes</a>.
          O cache de forca-base da IA fica localmente no projeto, a leitura se recalibra conforme os resultados oficiais entram na base
          e os ajustes manuais do modelo podem ser feitos no arquivo <code>{escape(MODEL_ADJUSTMENTS_FILE.name)}</code>.
        </p>
      </section>
    </section>
  </main>
  <div id="updateToast" class="update-toast"></div>
  <script>
    function reloadPortalShell() {{
      const url = new URL(window.location.href);
      url.searchParams.set('view', 'copa');
      url.searchParams.set('copa_reload_nonce', Date.now().toString());
      try {{
        if (window.top && window.top !== window) {{
          window.top.location.href = url.toString();
          return;
        }}
      }} catch (error) {{}}
      location.href = url.toString();
    }}

    const COPA_API_URL = "http://127.0.0.1:8765/api/refresh-copa";
    let currentModelAdjustments = {model_adjustments_json};

    function lockActionButton(btn, title, subtitle) {{
      if (!btn) return () => {{}};
      const originalHtml = btn.innerHTML;
      const originalText = btn.textContent;
      btn.disabled = true;
      if (btn.classList.contains('btn-update')) {{
        btn.innerHTML = `<span class="btn-update-icon"><svg class="spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"/><line x1="16.24" y1="4.93" x2="19.07" y2="7.76"/></svg></span><span class="btn-update-copy"><span class="btn-update-title">${{title}}</span><span class="btn-update-subtitle">${{subtitle}}</span></span>`;
      }} else {{
        btn.textContent = title;
      }}
      return () => {{
        btn.disabled = false;
        if (btn.classList.contains('btn-update')) {{
          btn.innerHTML = originalHtml;
        }} else {{
          btn.textContent = originalText || '';
        }}
      }};
    }}

    async function refreshCopa(
      btn,
      requestPayload = null,
      loadingTitle = "Atualizando",
      loadingSubtitle = "Buscando resultados",
      successMessage = "Painel atualizado com sucesso! Recarregando..."
    ) {{
      const unlock = lockActionButton(btn, loadingTitle, loadingSubtitle);
      try {{
        const fetchOptions = {{ method: "POST" }};
        if (requestPayload) {{
          fetchOptions.headers = {{ "Content-Type": "application/json" }};
          fetchOptions.body = JSON.stringify(requestPayload);
        }}
        const res = await fetch(COPA_API_URL, fetchOptions);
        const data = await res.json();
        if (data.ok) {{
          if (data.model_adjustments && typeof data.model_adjustments === 'object') {{
            currentModelAdjustments = data.model_adjustments;
          }}
          showToast(successMessage, "ok");
          setTimeout(() => reloadPortalShell(), 1200);
        }} else {{
          showToast("Erro: " + (data.error || "Falha ao atualizar."), "err");
          unlock();
        }}
      }} catch (err) {{
        showToast("Erro de conexao. O servidor Portal AI esta rodando?", "err");
        unlock();
      }}
    }}

    function showToast(msg, type) {{
      const el = document.getElementById("updateToast");
      el.textContent = msg;
      el.className = "update-toast " + type;
      el.style.display = "block";
      setTimeout(() => el.style.display = "none", 4000);
    }}

    function parseTeamAdjustments(rawText) {{
      const cleaned = (rawText || '').trim();
      if (!cleaned) return {{}};
      const parsed = JSON.parse(cleaned);
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {{
        throw new Error("Use um JSON em formato de objeto.");
      }}
      return parsed;
    }}

    function formatTeamAdjustments(adjustments) {{
      if (!adjustments || typeof adjustments !== 'object') return '';
      const keys = Object.keys(adjustments);
      if (!keys.length) return '';
      return JSON.stringify(adjustments, null, 2);
    }}

    function setInputValue(id, value) {{
      const element = document.getElementById(id);
      if (!element) return;
      element.value = String(value ?? '');
    }}

    function populateCalibrationForm(config) {{
      const source = config && typeof config === 'object' ? config : {{}};
      setInputValue('cfgDrawMultiplier', source.draw_multiplier ?? 1.0);
      setInputValue('cfgHomeGoalMultiplier', source.home_goal_multiplier ?? 1.0);
      setInputValue('cfgAwayGoalMultiplier', source.away_goal_multiplier ?? 1.0);
      setInputValue('cfgHomeAdvantagePoints', source.home_advantage_points ?? 2.3);
      setInputValue('cfgConfidenceBias', source.confidence_bias ?? 0.0);
      setInputValue('cfgTeamAdjustments', formatTeamAdjustments(source.team_rating_adjustments || {{}}));
    }}

    function readNumericField(id, fallback) {{
      const element = document.getElementById(id);
      if (!element) return fallback;
      const numeric = Number(element.value);
      return Number.isFinite(numeric) ? numeric : fallback;
    }}

    function buildCalibrationPayload() {{
      const teamAdjustmentsRaw = document.getElementById('cfgTeamAdjustments');
      const teamAdjustments = parseTeamAdjustments(teamAdjustmentsRaw ? teamAdjustmentsRaw.value : '');
      return {{
        reset_model_adjustments: false,
        model_adjustments: {{
          draw_multiplier: readNumericField('cfgDrawMultiplier', 1.0),
          home_goal_multiplier: readNumericField('cfgHomeGoalMultiplier', 1.0),
          away_goal_multiplier: readNumericField('cfgAwayGoalMultiplier', 1.0),
          home_advantage_points: readNumericField('cfgHomeAdvantagePoints', 2.3),
          confidence_bias: readNumericField('cfgConfidenceBias', 0.0),
          team_rating_adjustments: teamAdjustments,
        }},
      }};
    }}

    const cards = Array.from(document.querySelectorAll('.analysis-card'));
    const searchInput = document.getElementById('searchInput');
    const stageFilter = document.getElementById('stageFilter');
    const dateFilter = document.getElementById('dateFilter');
    const resultsCount = document.getElementById('resultsCount');
    const emptyState = document.getElementById('emptyState');
    const loadMoreBtn = document.getElementById('loadMoreBtn');
    const btnToggleCopaCalibration = document.getElementById('btnToggleCopaCalibration');
    const btnApplyCopaCalibration = document.getElementById('btnApplyCopaCalibration');
    const btnResetCopaCalibration = document.getElementById('btnResetCopaCalibration');
    const copaCalibrationPanel = document.getElementById('copaCalibrationPanel');
    const PAGE_SIZE = 18;
    let visibleLimit = PAGE_SIZE;

    populateCalibrationForm(currentModelAdjustments);
    if (btnToggleCopaCalibration && copaCalibrationPanel) {{
      btnToggleCopaCalibration.addEventListener('click', () => {{
        const isOpen = copaCalibrationPanel.classList.toggle('show');
        btnToggleCopaCalibration.textContent = isOpen ? 'Fechar calibragem' : 'Central de calibragem';
      }});
    }}
    if (btnApplyCopaCalibration) {{
      btnApplyCopaCalibration.addEventListener('click', async () => {{
        try {{
          const payload = buildCalibrationPayload();
          await refreshCopa(
            btnApplyCopaCalibration,
            payload,
            "Recalibrando",
            "Aplicando criterios",
            "Modelo recalibrado! Regenerando painel..."
          );
        }} catch (error) {{
          showToast("JSON invalido nos ajustes por selecao. Revise o formato.", "err");
        }}
      }});
    }}
    if (btnResetCopaCalibration) {{
      btnResetCopaCalibration.addEventListener('click', async () => {{
        await refreshCopa(
          btnResetCopaCalibration,
          {{ reset_model_adjustments: true }},
          "Restaurando",
          "Padrao do modelo",
          "Criterios padrao restaurados. Regenerando painel..."
        );
      }});
    }}

    function applyFilters(resetLimit = false) {{
      if (resetLimit) visibleLimit = PAGE_SIZE;
      const query = (searchInput.value || '').trim().toLowerCase();
      const stage = (stageFilter.value || '').trim();
      const dateValue = (dateFilter.value || '').trim();
      const matchedCards = [];

      cards.forEach((card) => {{
        const searchBlob = card.dataset.search || '';
        const cardStage = card.dataset.stage || '';
        const cardDate = card.dataset.date || '';
        const matchesQuery = !query || searchBlob.includes(query);
        const matchesStage = !stage || cardStage === stage;
        const matchesDate = !dateValue || cardDate === dateValue;
        const show = matchesQuery && matchesStage && matchesDate;
        card.style.display = 'none';
        if (show) matchedCards.push(card);
      }});

      matchedCards.forEach((card, index) => {{
        card.style.display = index < visibleLimit ? '' : 'none';
      }});

      const visible = Math.min(visibleLimit, matchedCards.length);
      const hidden = Math.max(0, matchedCards.length - visibleLimit);
      resultsCount.textContent = matchedCards.length === 1 ? '1 confronto visivel' : `${{matchedCards.length}} confrontos visiveis`;
      emptyState.style.display = matchedCards.length === 0 ? 'block' : 'none';
      loadMoreBtn.style.display = hidden > 0 ? 'inline-flex' : 'none';
      if (hidden > 0) {{
        loadMoreBtn.textContent = `Carregar mais ${{Math.min(PAGE_SIZE, hidden)}} jogos`;
      }}
    }}

    searchInput.addEventListener('input', () => applyFilters(true));
    stageFilter.addEventListener('change', () => applyFilters(true));
    dateFilter.addEventListener('change', () => applyFilters(true));
    loadMoreBtn.addEventListener('click', () => {{
      visibleLimit += PAGE_SIZE;
      applyFilters(false);
    }});
    applyFilters(true);
  </script>
</body>
</html>
"""

def build_world_cup_html() -> str:
    return build_world_cup_schedule_html()


TEAM_PT_BR_NAMES_REPLACE = {
    "Mexico": "México", "México": "México", "Sudáfrica": "África do Sul", "South Africa": "África do Sul",
    "South Korea": "Coreia do Sul", "República de Corea": "Coreia do Sul", "Corea del Sur": "Coreia do Sul",
    "Czechia": "República Tcheca", "Chequia": "República Tcheca",
    "Canada": "Canadá", "Canadá": "Canadá",
    "Bosnia and Herzegovina": "Bósnia", "Bosnia y Herzegovina": "Bósnia",
    "USA": "Estados Unidos", "Estados Unidos": "Estados Unidos",
    "Paraguay": "Paraguai", "Paraguay": "Paraguai",
    "Qatar": "Catar", "Catar": "Catar",
    "Switzerland": "Suíça", "Suiza": "Suíça",
    "Brazil": "Brasil", "Brasil": "Brasil",
    "Morocco": "Marrocos", "Marruecos": "Marrocos",
    "Haiti": "Haiti",
    "Scotland": "Escócia", "Escocia": "Escócia",
    "Australia": "Austrália", "Australia": "Austrália",
    "Turkey": "Turquia", "Turquía": "Turquia", "Turquia": "Turquia",
    "Germany": "Alemanha", "Alemania": "Alemanha",
    "Curacao": "Curaçao", "Curazao": "Curaçao",
    "Netherlands": "Holanda", "Países Bajos": "Holanda", "Paises Bajos": "Holanda",
    "Japan": "Japão", "Japón": "Japão", "Japon": "Japão",
    "Ivory Coast": "Costa do Marfim", "Costa de Marfil": "Costa do Marfim",
    "Ecuador": "Equador",
    "Sweden": "Suécia", "Suecia": "Suécia",
    "Tunisia": "Tunísia", "Túnez": "Tunísia", "Tunez": "Tunísia",
    "Spain": "Espanha", "España": "Espanha", "Espana": "Espanha",
    "Cape Verde": "Cabo Verde",
    "Belgium": "Bélgica", "Bélgica": "Bélgica", "Belgica": "Bélgica",
    "Egypt": "Egito", "Egipto": "Egito",
    "Saudi Arabia": "Arábia Saudita", "Arabia Saudita": "Arábia Saudita",
    "Uruguay": "Uruguai",
    "Iran": "Irã", "Irán": "Irã", "Iran": "Irã",
    "New Zealand": "Nova Zelândia", "Nueva Zelanda": "Nova Zelândia",
    "France": "França", "Francia": "França",
    "Senegal": "Senegal",
    "Iraq": "Iraque", "Irak": "Iraque",
    "Norway": "Noruega",
    "Argentina": "Argentina",
    "Algeria": "Argélia", "Argelia": "Argélia",
    "Austria": "Áustria", "Austria": "Áustria",
    "Jordan": "Jordânia", "Jordania": "Jordânia",
    "Portugal": "Portugal",
    "D.R. Congo": "RD Congo", "RD del Congo": "RD Congo",
    "England": "Inglaterra",
    "Croatia": "Croácia", "Croacia": "Croácia",
    "Ghana": "Gana",
    "Panama": "Panamá", "Panamá": "Panamá", "Panama": "Panamá",
    "Uzbekistan": "Uzbequistão", "Uzbekistán": "Uzbequistão", "Uzbekistan": "Uzbequistão",
    "Colombia": "Colômbia", "Colombia": "Colômbia",
    "Suriname": "Suriname",
    "Bolivia": "Bolívia", "Bolivia": "Bolívia",
    "Jamaica": "Jamaica",
    "New Caledonia": "Nova Caledônia"
}

def apply_team_translations(html: str) -> str:
    # Sort keys by length descending to replace longer phrases first
    import re
    sorted_keys = sorted(TEAM_PT_BR_NAMES_REPLACE.keys(), key=len, reverse=True)
    for k in sorted_keys:
        val = TEAM_PT_BR_NAMES_REPLACE[k]
        if k != val:
            # We use regex to only match whole words so "Panama" doesnt match inside another word if available
            html = re.sub(r'(?<![A-Za-z0-9_])' + re.escape(k) + r'(?![A-Za-z0-9_])', val, html)
    return html

def main() -> None:
    html = build_world_cup_schedule_html()
    html = apply_team_translations(html)
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    print(str(OUTPUT_FILE))


if __name__ == "__main__":
    main()
