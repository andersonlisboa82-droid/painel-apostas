from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from difflib import SequenceMatcher
import re
import threading
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent
SOURCE_NAME = "KickingData"
FALLBACK_SOURCE_NAME = "ESPN"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
MATCH_URL_TEMPLATE = "https://kickingdata.com/match/{slug}/"
REAL_MATCH_STATS_CACHE_FILE = BASE_DIR / "real_match_stats_cache.json"
REAL_MATCH_STATS_CACHE_VERSION = 1
TEAM_SLUG_ALIASES: dict[str, list[str]] = {
    "A. Italiano": ["audax-italiano", "a-italiano"],
    "Ath Bilbao": ["athletic-bilbao", "athletic-club", "ath-bilbao"],
    "Athletico-PR": ["athletico-paranaense", "athletico-pr", "athletico"],
    "Atletico-MG": ["atletico-mineiro", "atletico-mg", "atletico"],
    "Botafogo RJ": ["botafogo", "botafogo-rj"],
    "Chapecoense-SC": ["chapecoense", "chapecoense-sc"],
    "D.R. Congo": ["dr-congo", "congo-dr", "d-r-congo"],
    "Dep. Cuenca": ["deportivo-cuenca", "dep-cuenca"],
    "Dep. Riestra": ["deportivo-riestra", "dep-riestra"],
    "Flamengo RJ": ["flamengo", "flamengo-rj"],
    "Ind. del Valle": ["independiente-del-valle", "ind-del-valle"],
    "Ind. Medellin": ["independiente-medellin", "ind-medellin"],
    "Ind. Rivadavia": ["independiente-rivadavia", "ind-rivadavia"],
    "Junior": ["junior", "atletico-junior"],
    "Libertad Asuncion": ["libertad", "libertad-asuncion"],
    "Olimpia Asuncion": ["olimpia", "olimpia-asuncion"],
    "U. Catolica": ["universidad-catolica", "u-catolica"],
    "U. de Deportes": ["universitario-de-deportes", "universidad-de-deportes", "u-de-deportes"],
    "U. de Chile": ["universidad-de-chile", "u-de-chile"],
}
TEAM_MATCH_ALIASES: dict[str, list[str]] = {
    "Athletico-PR": ["Athletico Paranaense", "Athletico"],
    "Atletico-MG": ["Atletico Mineiro", "Atletico MG"],
    "Botafogo RJ": ["Botafogo"],
    "Chapecoense-SC": ["Chapecoense"],
    "Flamengo RJ": ["Flamengo"],
    "A. Italiano": ["Audax Italiano"],
    "Ath Bilbao": ["Athletic Club", "Athletic Bilbao"],
    "Dep. Cuenca": ["Deportivo Cuenca"],
    "Dep. Riestra": ["Deportivo Riestra"],
    "Ind. del Valle": ["Independiente del Valle"],
    "Ind. Medellin": ["Independiente Medellin"],
    "Ind. Rivadavia": ["Independiente Rivadavia"],
    "Libertad Asuncion": ["Libertad"],
    "Olimpia Asuncion": ["Olimpia"],
    "U. Catolica": ["Universidad Catolica"],
    "U. de Deportes": ["Universitario"],
    "U. de Chile": ["Universidad de Chile"],
}
STATE_SUFFIXES = ("-rj", "-sp", "-mg", "-sc", "-pr", "-rs", "-lp")
_persistent_cache_lock = threading.Lock()


def _slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _team_slug_candidates(team: str) -> list[str]:
    base_slug = _slugify(team)
    candidates = list(TEAM_SLUG_ALIASES.get(team, []))
    if base_slug:
        candidates.append(base_slug)
        if base_slug.endswith(STATE_SUFFIXES):
            candidates.append(base_slug.rsplit("-", 1)[0])
        if base_slug.startswith("u-"):
            candidates.append(base_slug.replace("u-", "universidad-", 1))
        if base_slug.startswith("ind-"):
            candidates.append(base_slug.replace("ind-", "independiente-", 1))
        if base_slug.startswith("dep-"):
            candidates.append(base_slug.replace("dep-", "deportivo-", 1))
    return _unique(candidates)


def _parse_date_candidate(raw_date: str) -> date | None:
    text = (raw_date or "").strip()
    if not text:
        return None

    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        pass

    for pattern in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue

    compact_match = re.fullmatch(r"(\d{2})\.(\d{2})\.", text)
    if compact_match:
        day = int(compact_match.group(1))
        month = int(compact_match.group(2))
        current_year = datetime.now().year
        return date(current_year, month, day)

    return None


def _date_candidates(date_text: str, event_timestamp: str | None) -> list[str]:
    candidates: list[date] = []
    for raw_date in [event_timestamp or "", date_text]:
        parsed_date = _parse_date_candidate(raw_date)
        if not parsed_date:
            continue
        candidates.extend([parsed_date - timedelta(days=1), parsed_date, parsed_date + timedelta(days=1)])

    unique_dates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        date_label = candidate.isoformat()
        if date_label in seen:
            continue
        seen.add(date_label)
        unique_dates.append(date_label)
    return unique_dates


@lru_cache(maxsize=512)
def _fetch_match_page(url: str) -> str:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    if response.status_code != 200:
        return ""
    return response.text


def _parse_numeric_value(raw_value: str) -> int | float | None:
    text = (raw_value or "").strip().replace("%", "").replace(",", ".")
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if value.is_integer():
        return int(value)
    return round(value, 2)


def _parse_metric_blocks(html: str) -> dict[str, dict[str, int | float]]:
    soup = BeautifulSoup(html, "html.parser")
    metrics: dict[str, dict[str, int | float]] = {}
    for block in soup.select("div.team-stats"):
        title = block.select_one(".team-stats__title")
        if not title:
            continue
        spans = title.find_all("span", recursive=False)
        if len(spans) < 3:
            continue
        home_value = _parse_numeric_value(spans[0].get_text(" ", strip=True))
        label = spans[1].get_text(" ", strip=True).lower()
        away_value = _parse_numeric_value(spans[2].get_text(" ", strip=True))
        if home_value is None or away_value is None or not label:
            continue
        metrics[label] = {"home": home_value, "away": away_value}
    return metrics


def _find_metric(metrics: dict[str, dict[str, int | float]], keyword: str) -> dict[str, int | float] | None:
    for label, values in metrics.items():
        if keyword in label:
            return values
    return None


def _format_metric(label: str, values: dict[str, int | float] | None, unit: str = "") -> dict[str, object] | None:
    if not values:
        return None
    return {
        "label": label,
        "home": values["home"],
        "away": values["away"],
        "unit": unit,
    }


def _normalize_team_name_for_matching(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    ascii_text = ascii_text.replace("&", " and ")
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    tokens = [token for token in ascii_text.split() if token not in {"fc", "ac", "club"}]
    return " ".join(tokens).strip()


def _team_match_candidates(team: str) -> list[str]:
    raw_candidates = [team]
    raw_candidates.extend(TEAM_MATCH_ALIASES.get(team, []))

    normalized_candidates: list[str] = []
    for value in raw_candidates:
        normalized_value = _normalize_team_name_for_matching(value)
        if normalized_value:
            normalized_candidates.append(normalized_value)
            parts = normalized_value.split()
            if parts and parts[-1] in {"rj", "sp", "mg", "sc", "pr", "rs", "lp"}:
                short_value = " ".join(parts[:-1]).strip()
                if short_value:
                    normalized_candidates.append(short_value)
    return _unique(normalized_candidates)


def _similarity_score(expected_candidates: list[str], observed_name: str) -> float:
    observed_normalized = _normalize_team_name_for_matching(observed_name)
    if not observed_normalized:
        return 0.0
    if not expected_candidates:
        return 0.0
    return max(SequenceMatcher(None, candidate, observed_normalized).ratio() for candidate in expected_candidates)


@lru_cache(maxsize=64)
def _fetch_espn_scoreboard(date_token: str) -> dict[str, object]:
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard?dates={date_token}"
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    if response.status_code != 200:
        return {}
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=512)
def _fetch_espn_summary(event_id: str) -> dict[str, object]:
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/all/summary?event={event_id}"
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    if response.status_code != 200:
        return {}
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_home_away_competitors(competition_payload: dict[str, object]) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    competitors_raw = competition_payload.get("competitors")
    competitors = competitors_raw if isinstance(competitors_raw, list) else []
    home_competitor: dict[str, object] | None = None
    away_competitor: dict[str, object] | None = None

    for competitor in competitors:
        if not isinstance(competitor, dict):
            continue
        side = str(competitor.get("homeAway") or "").lower()
        if side == "home" and home_competitor is None:
            home_competitor = competitor
        elif side == "away" and away_competitor is None:
            away_competitor = competitor

    if (home_competitor is None or away_competitor is None) and len(competitors) >= 2:
        fallback_a = competitors[0] if isinstance(competitors[0], dict) else None
        fallback_b = competitors[1] if isinstance(competitors[1], dict) else None
        if home_competitor is None:
            home_competitor = fallback_a
        if away_competitor is None:
            away_competitor = fallback_b
    return home_competitor, away_competitor


def _extract_espn_stat_value(raw_value: object) -> int | float | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, (int, float)):
        numeric_value = float(raw_value)
    else:
        parsed = _parse_numeric_value(str(raw_value))
        if parsed is None:
            return None
        numeric_value = float(parsed)
    if numeric_value.is_integer():
        return int(numeric_value)
    return round(numeric_value, 2)


def _lookup_real_match_stats_espn(
    home_team: str,
    away_team: str,
    date_text: str,
    event_timestamp: str | None = None,
) -> dict[str, object]:
    date_candidates = _date_candidates(date_text, event_timestamp)
    if not date_candidates:
        return {
            "available": False,
            "source": FALLBACK_SOURCE_NAME,
            "message": "Nao foi possivel montar a busca na fonte de fallback.",
        }

    home_expected = _team_match_candidates(home_team)
    away_expected = _team_match_candidates(away_team)
    if not home_expected or not away_expected:
        return {
            "available": False,
            "source": FALLBACK_SOURCE_NAME,
            "message": "Nao foi possivel normalizar os times para busca no fallback.",
        }

    best_event: dict[str, object] | None = None
    for iso_date in date_candidates:
        date_token = iso_date.replace("-", "")
        scoreboard = _fetch_espn_scoreboard(date_token)
        events = scoreboard.get("events")
        if not isinstance(events, list):
            continue

        for event in events:
            if not isinstance(event, dict):
                continue
            competitions = event.get("competitions")
            if not isinstance(competitions, list) or not competitions:
                continue
            competition = competitions[0] if isinstance(competitions[0], dict) else {}
            home_competitor, away_competitor = _extract_home_away_competitors(competition)
            if not home_competitor or not away_competitor:
                continue

            home_name = str(((home_competitor.get("team") or {}).get("displayName") or "")).strip()
            away_name = str(((away_competitor.get("team") or {}).get("displayName") or "")).strip()
            if not home_name or not away_name:
                continue

            home_score = _similarity_score(home_expected, home_name)
            away_score = _similarity_score(away_expected, away_name)
            if home_score < 0.52 or away_score < 0.52:
                continue

            event_id = str(event.get("id") or competition.get("id") or "").strip()
            if not event_id:
                continue

            total_score = home_score + away_score
            status_type = ((competition.get("status") or {}).get("type") or {})
            if bool(status_type.get("completed")):
                total_score += 0.05

            source_url = ""
            links = event.get("links")
            if isinstance(links, list):
                for link in links:
                    href = str((link or {}).get("href") or "").strip() if isinstance(link, dict) else ""
                    if href and "espn.com" in href:
                        source_url = href
                        break
            if not source_url:
                source_url = f"https://www.espn.com/soccer/match/_/gameId/{event_id}"

            candidate_event = {
                "score": total_score,
                "event_id": event_id,
                "home_id": str(((home_competitor.get("team") or {}).get("id") or "")).strip(),
                "away_id": str(((away_competitor.get("team") or {}).get("id") or "")).strip(),
                "home_name": home_name,
                "away_name": away_name,
                "source_url": source_url,
            }
            if best_event is None or float(candidate_event["score"]) > float(best_event["score"]):
                best_event = candidate_event

    if not best_event:
        return {
            "available": False,
            "source": FALLBACK_SOURCE_NAME,
            "message": "Fallback ESPN nao encontrou o jogo para consulta de estatisticas reais.",
        }

    summary_payload = _fetch_espn_summary(str(best_event["event_id"]))
    boxscore = summary_payload.get("boxscore") if isinstance(summary_payload, dict) else {}
    teams_payload = boxscore.get("teams") if isinstance(boxscore, dict) else None
    teams = teams_payload if isinstance(teams_payload, list) else []
    if not teams:
        return {
            "available": False,
            "source": FALLBACK_SOURCE_NAME,
            "source_url": str(best_event["source_url"]),
            "message": "Fallback ESPN encontrou o jogo, mas nao retornou boxscore de estatisticas.",
        }

    parsed_teams: list[dict[str, object]] = []
    for team_payload in teams:
        if not isinstance(team_payload, dict):
            continue
        team_data = team_payload.get("team") if isinstance(team_payload.get("team"), dict) else {}
        team_name = str(team_data.get("displayName") or "").strip()
        team_id = str(team_data.get("id") or "").strip()
        statistics_raw = team_payload.get("statistics")
        statistics_list = statistics_raw if isinstance(statistics_raw, list) else []
        statistics_map: dict[str, int | float] = {}
        for stat in statistics_list:
            if not isinstance(stat, dict):
                continue
            stat_name = str(stat.get("name") or "").strip()
            if not stat_name:
                continue
            raw_value = stat.get("displayValue")
            if raw_value is None:
                raw_value = stat.get("value")
            parsed_value = _extract_espn_stat_value(raw_value)
            if parsed_value is None:
                continue
            statistics_map[stat_name] = parsed_value
        parsed_teams.append({"id": team_id, "name": team_name, "stats": statistics_map})

    if not parsed_teams:
        return {
            "available": False,
            "source": FALLBACK_SOURCE_NAME,
            "source_url": str(best_event["source_url"]),
            "message": "Fallback ESPN nao retornou estatisticas estruturadas para o jogo.",
        }

    home_entry = next((entry for entry in parsed_teams if str(entry.get("id")) == str(best_event["home_id"])), None)
    away_entry = next((entry for entry in parsed_teams if str(entry.get("id")) == str(best_event["away_id"])), None)

    if home_entry is None:
        home_entry = max(
            parsed_teams,
            key=lambda entry: _similarity_score(home_expected, str(entry.get("name") or "")),
            default=None,
        )
    if away_entry is None:
        remaining_entries = [entry for entry in parsed_teams if entry is not home_entry] or parsed_teams
        away_entry = max(
            remaining_entries,
            key=lambda entry: _similarity_score(away_expected, str(entry.get("name") or "")),
            default=None,
        )

    home_stats = home_entry.get("stats") if isinstance(home_entry, dict) else {}
    away_stats = away_entry.get("stats") if isinstance(away_entry, dict) else {}
    if not isinstance(home_stats, dict):
        home_stats = {}
    if not isinstance(away_stats, dict):
        away_stats = {}

    def metric_from_stat(stat_name: str, label: str, unit: str = "") -> dict[str, object] | None:
        home_value = _extract_espn_stat_value(home_stats.get(stat_name))
        away_value = _extract_espn_stat_value(away_stats.get(stat_name))
        if home_value is None or away_value is None:
            return None
        return {"label": label, "home": home_value, "away": away_value, "unit": unit}

    stats = {
        key: value
        for key, value in {
            "corners": metric_from_stat("wonCorners", "Escanteios"),
            "yellow_cards": metric_from_stat("yellowCards", "Cartoes amarelos"),
            "fouls": metric_from_stat("foulsCommitted", "Faltas"),
            "possession": metric_from_stat("possessionPct", "Posse de bola", unit="%"),
        }.items()
        if value is not None
    }
    if not stats:
        return {
            "available": False,
            "source": FALLBACK_SOURCE_NAME,
            "source_url": str(best_event["source_url"]),
            "message": "Fallback ESPN encontrou o jogo, mas sem os indicadores esperados.",
        }

    return {
        "available": True,
        "source": FALLBACK_SOURCE_NAME,
        "source_url": str(best_event["source_url"]),
        "message": "Estatisticas reais carregadas com sucesso (fallback ESPN).",
        "stats": stats,
    }


def _mean(values: list[int | float]) -> float | None:
    clean_values = [float(value) for value in values if value is not None]
    if not clean_values:
        return None
    return round(sum(clean_values) / len(clean_values), 2)


def _project_metric(primary_value: float | None, secondary_value: float | None) -> float | None:
    if primary_value is None or secondary_value is None:
        return None
    return round((primary_value + secondary_value) / 2, 2)


def _target_event_datetime(event_timestamp: str | None) -> pd.Timestamp | None:
    if not event_timestamp:
        return None
    parsed = pd.to_datetime(event_timestamp, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed


def build_match_stats_cache_key(
    home_team: str,
    away_team: str,
    date_text: str,
    event_timestamp: str | None = None,
) -> str:
    event_date = None
    event_dt = _target_event_datetime(event_timestamp)
    if event_dt is not None and not pd.isna(event_dt):
        event_date = event_dt.date().isoformat()
    if not event_date:
        parsed_date = _parse_date_candidate(date_text)
        event_date = parsed_date.isoformat() if parsed_date else _slugify(date_text or "sem-data")
    return f"{_slugify(home_team)}__{_slugify(away_team)}__{event_date}"


def _load_persistent_cache_data() -> dict[str, object]:
    if not REAL_MATCH_STATS_CACHE_FILE.exists():
        return {"version": REAL_MATCH_STATS_CACHE_VERSION, "matches": {}}
    try:
        data = json.loads(REAL_MATCH_STATS_CACHE_FILE.read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError):
        return {"version": REAL_MATCH_STATS_CACHE_VERSION, "matches": {}}
    matches = data.get("matches")
    if not isinstance(matches, dict):
        matches = {}
    return {"version": REAL_MATCH_STATS_CACHE_VERSION, "matches": matches}


def load_real_match_stats_cache() -> dict[str, dict[str, object]]:
    with _persistent_cache_lock:
        payload = _load_persistent_cache_data()
    matches = payload.get("matches", {})
    return {
        str(cache_key): value
        for cache_key, value in matches.items()
        if isinstance(value, dict)
    }


def _store_persisted_match_stats(cache_key: str, payload: dict[str, object]) -> None:
    if not cache_key or not payload.get("available"):
        return
    stored_payload = {
        "available": True,
        "source": payload.get("source", SOURCE_NAME),
        "source_url": payload.get("source_url"),
        "message": payload.get("message") or "Estatisticas reais carregadas com sucesso.",
        "stats": payload.get("stats", {}),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with _persistent_cache_lock:
        data = _load_persistent_cache_data()
        matches = data.get("matches", {})
        matches[cache_key] = stored_payload
        output = {
            "version": REAL_MATCH_STATS_CACHE_VERSION,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "matches": matches,
        }
        REAL_MATCH_STATS_CACHE_FILE.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def get_persisted_match_stats(
    home_team: str,
    away_team: str,
    date_text: str,
    event_timestamp: str | None = None,
) -> dict[str, object] | None:
    cache_key = build_match_stats_cache_key(home_team, away_team, date_text, event_timestamp=event_timestamp)
    cache = load_real_match_stats_cache()
    payload = cache.get(cache_key)
    if not isinstance(payload, dict) or not payload.get("available"):
        return None
    return payload


@lru_cache(maxsize=256)
def _lookup_real_match_stats_live(
    home_team: str,
    away_team: str,
    status: str,
    date_text: str,
    event_timestamp: str | None = None,
) -> dict[str, object]:
    if (status or "").strip().lower() != "finalizado":
        return {
            "available": False,
            "source": SOURCE_NAME,
            "message": "Estatisticas reais ficam disponiveis apenas para jogos finalizados.",
        }

    home_candidates = _team_slug_candidates(home_team)
    away_candidates = _team_slug_candidates(away_team)
    date_candidates = _date_candidates(date_text, event_timestamp)

    if not home_candidates or not away_candidates or not date_candidates:
        return {
            "available": False,
            "source": SOURCE_NAME,
            "message": "Nao foi possivel montar a busca externa para este jogo.",
        }

    attempted_urls: list[str] = []
    for match_date in date_candidates:
        for home_slug in home_candidates:
            for away_slug in away_candidates:
                url = MATCH_URL_TEMPLATE.format(slug=f"{home_slug}-{away_slug}-{match_date}")
                if url in attempted_urls:
                    continue
                attempted_urls.append(url)
                html = _fetch_match_page(url)
                if not html or "team-stats__modern-wrapper" not in html:
                    continue

                parsed_metrics = _parse_metric_blocks(html)
                corners = _format_metric("Escanteios", _find_metric(parsed_metrics, "corners"))
                yellow_cards = _format_metric("Cartoes amarelos", _find_metric(parsed_metrics, "yellow cards"))
                fouls = _format_metric("Faltas", _find_metric(parsed_metrics, "fouls"))
                possession = _format_metric("Posse de bola", _find_metric(parsed_metrics, "possession"), unit="%")

                stats = {
                    key: value
                    for key, value in {
                        "corners": corners,
                        "yellow_cards": yellow_cards,
                        "fouls": fouls,
                        "possession": possession,
                    }.items()
                    if value is not None
                }

                if not stats:
                    continue

                return {
                    "available": True,
                    "source": SOURCE_NAME,
                    "source_url": url,
                    "message": "Estatisticas reais carregadas com sucesso.",
                    "stats": stats,
                }

    fallback_payload = _lookup_real_match_stats_espn(
        home_team=home_team,
        away_team=away_team,
        date_text=date_text,
        event_timestamp=event_timestamp,
    )
    if fallback_payload.get("available"):
        return fallback_payload

    return {
        "available": False,
        "source": FALLBACK_SOURCE_NAME,
        "message": (
            "Nao encontrei cartoes, escanteios, faltas e posse reais para este jogo "
            "nas fontes complementares (KickingData e ESPN)."
        ),
    }


def get_real_match_stats(
    home_team: str,
    away_team: str,
    status: str,
    date_text: str,
    event_timestamp: str | None = None,
    prefer_persistent_cache: bool = True,
    persist_result: bool = True,
    force_refresh: bool = False,
) -> dict[str, object]:
    return {
        "available": False,
        "source": SOURCE_NAME,
        "message": "Atualizacao de estatisticas (escanteios, cartoes, posse) desativada.",
    }

    cache_key = build_match_stats_cache_key(home_team, away_team, date_text, event_timestamp=event_timestamp)
    cached_payload: dict[str, object] | None = None
    cached_missing_metrics: list[str] = []
    if prefer_persistent_cache and not force_refresh:
        cached_payload = get_persisted_match_stats(
            home_team=home_team,
            away_team=away_team,
            date_text=date_text,
            event_timestamp=event_timestamp,
        )
        if cached_payload:
            cached_stats = cached_payload.get("stats", {}) if isinstance(cached_payload, dict) else {}
            if not isinstance(cached_stats, dict):
                cached_stats = {}
            required_metrics = ("corners", "yellow_cards", "fouls", "possession")
            cached_missing_metrics = [metric for metric in required_metrics if metric not in cached_stats]
            if not cached_missing_metrics:
                return {**cached_payload, "cache_hit": True, "cache_key": cache_key}

    payload = _lookup_real_match_stats_live(
        home_team=home_team,
        away_team=away_team,
        status=status,
        date_text=date_text,
        event_timestamp=event_timestamp,
    )
    if persist_result and payload.get("available"):
        _store_persisted_match_stats(cache_key, payload)
    if cached_payload and cached_missing_metrics and not payload.get("available"):
        cached_message = str(cached_payload.get("message") or "Estatisticas reais carregadas do cache.")
        missing_text = ", ".join(cached_missing_metrics)
        return {
            **cached_payload,
            "cache_hit": True,
            "cache_key": cache_key,
            "message": (
                f"{cached_message} Nao foi possivel atualizar agora os campos faltantes "
                f"({missing_text}) na fonte complementar."
            ),
        }
    return {**payload, "cache_hit": False, "cache_key": cache_key}


def build_team_stat_profile(
    matches_df: pd.DataFrame,
    team_name: str,
    event_timestamp: str | None = None,
    sample_target: int = 3,
    scan_limit: int = 10,
) -> dict[str, object]:
    if matches_df.empty:
        return {
            "available": False,
            "team": team_name,
            "message": "Nao ha base local suficiente para montar a media recente do time.",
        }

    finished = matches_df[matches_df["status"] == "Finalizado"].copy()
    finished = finished[(finished["home_team"] == team_name) | (finished["away_team"] == team_name)].copy()
    if finished.empty:
        return {
            "available": False,
            "team": team_name,
            "message": "Nao encontrei jogos finalizados recentes para este time na base local.",
        }

    finished["_event_dt"] = pd.to_datetime(finished["event_timestamp"], errors="coerce")
    current_event_dt = _target_event_datetime(event_timestamp)
    if current_event_dt is not None:
        finished = finished[finished["_event_dt"] < current_event_dt].copy()
    finished = finished.sort_values(by="_event_dt", ascending=False, na_position="last").reset_index(drop=True)

    corners_for: list[int | float] = []
    corners_against: list[int | float] = []
    yellow_for: list[int | float] = []
    yellow_against: list[int | float] = []
    candidate_rows = list(finished.head(scan_limit).itertuples(index=False))
    scanned = len(candidate_rows)

    def load_stats(row) -> dict[str, object]:
        return get_real_match_stats(
            home_team=str(row.home_team),
            away_team=str(row.away_team),
            status=str(row.status),
            date_text=str(row.date_text),
            event_timestamp=getattr(row, "event_timestamp", None),
        )

    with ThreadPoolExecutor(max_workers=min(6, max(1, len(candidate_rows)))) as executor:
        stats_payloads = list(executor.map(load_stats, candidate_rows))

    for row, stats_payload in zip(candidate_rows, stats_payloads):
        stats = stats_payload.get("stats") if stats_payload.get("available") else None
        if not stats:
            continue

        team_side = "home" if str(row.home_team) == team_name else "away"
        opp_side = "away" if team_side == "home" else "home"

        corners = stats.get("corners")
        if corners:
            corners_for.append(float(corners[team_side]))
            corners_against.append(float(corners[opp_side]))

        yellow_cards = stats.get("yellow_cards")
        if yellow_cards:
            yellow_for.append(float(yellow_cards[team_side]))
            yellow_against.append(float(yellow_cards[opp_side]))

        if len(corners_for) >= sample_target and len(yellow_for) >= sample_target:
            break

    averages = {
        "corners_for": _mean(corners_for),
        "corners_against": _mean(corners_against),
        "yellow_for": _mean(yellow_for),
        "yellow_against": _mean(yellow_against),
    }
    sample_size = max(len(corners_for), len(yellow_for))
    available = any(value is not None for value in averages.values())

    if not available:
        return {
            "available": False,
            "team": team_name,
            "sample_size": 0,
            "message": "Nao encontrei estatisticas reais recentes suficientes para este time.",
        }

    return {
        "available": True,
        "team": team_name,
        "sample_size": sample_size,
        "scanned_matches": scanned,
        "averages": averages,
        "message": f"Media recente montada com {sample_size} jogo(s) com estatistica real disponivel.",
    }


def build_projection_payload(home_profile: dict[str, object], away_profile: dict[str, object]) -> dict[str, object]:
    home_averages = home_profile.get("averages", {}) if home_profile.get("available") else {}
    away_averages = away_profile.get("averages", {}) if away_profile.get("available") else {}

    projected_corners_home = _project_metric(
        home_averages.get("corners_for"),
        away_averages.get("corners_against"),
    )
    projected_corners_away = _project_metric(
        away_averages.get("corners_for"),
        home_averages.get("corners_against"),
    )
    projected_yellow_home = _project_metric(
        home_averages.get("yellow_for"),
        away_averages.get("yellow_against"),
    )
    projected_yellow_away = _project_metric(
        away_averages.get("yellow_for"),
        home_averages.get("yellow_against"),
    )

    projections = {
        "corners": {
            "home": projected_corners_home,
            "away": projected_corners_away,
            "total": round(projected_corners_home + projected_corners_away, 2)
            if projected_corners_home is not None and projected_corners_away is not None
            else None,
        },
        "yellow_cards": {
            "home": projected_yellow_home,
            "away": projected_yellow_away,
            "total": round(projected_yellow_home + projected_yellow_away, 2)
            if projected_yellow_home is not None and projected_yellow_away is not None
            else None,
        },
    }

    projection_available = any(
        metric.get("home") is not None or metric.get("away") is not None
        for metric in projections.values()
    )

    return {
        "available": projection_available,
        "home": home_profile,
        "away": away_profile,
        "projection": projections,
        "message": "Projecao simples baseada nas medias recentes com estatistica real disponivel."
        if projection_available
        else "Nao encontrei base recente suficiente para projetar cartoes e escanteios deste confronto.",
    }


def build_team_stat_projection(
    matches_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    event_timestamp: str | None = None,
) -> dict[str, object]:
    home_profile = build_team_stat_profile(matches_df, home_team, event_timestamp=event_timestamp)
    away_profile = build_team_stat_profile(matches_df, away_team, event_timestamp=event_timestamp)
    return build_projection_payload(home_profile, away_profile)


def prefetch_finished_match_stats(
    matches_df: pd.DataFrame,
    *,
    competitions: list[str] | None = None,
    per_competition_limit: int | None = 20,
    max_workers: int = 6,
    force_refresh: bool = False,
) -> dict[str, object]:
    if matches_df.empty:
        return {
            "candidates": 0,
            "already_cached": 0,
            "fetched_now": 0,
            "saved_now": 0,
            "available_total": 0,
            "missing_total": 0,
        }

    finished = matches_df[matches_df["status"] == "Finalizado"].copy()
    if competitions:
        finished = finished[finished["competition"].isin(competitions)].copy()
    if finished.empty:
        return {
            "candidates": 0,
            "already_cached": 0,
            "fetched_now": 0,
            "saved_now": 0,
            "available_total": 0,
            "missing_total": 0,
        }

    finished["_event_dt"] = pd.to_datetime(finished["event_timestamp"], errors="coerce", utc=True)
    finished = finished.sort_values(
        by=["competition", "_event_dt", "home_team", "away_team"],
        ascending=[True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)

    if per_competition_limit is not None:
        finished = (
            finished.groupby("competition", group_keys=False)
            .head(per_competition_limit)
            .reset_index(drop=True)
        )

    persistent_cache = load_real_match_stats_cache()
    candidate_rows: list[tuple[object, str, bool]] = []
    seen_cache_keys: set[str] = set()

    for row in finished.itertuples(index=False):
        cache_key = build_match_stats_cache_key(
            str(row.home_team),
            str(row.away_team),
            str(row.date_text),
            getattr(row, "event_timestamp", None),
        )
        if cache_key in seen_cache_keys:
            continue
        seen_cache_keys.add(cache_key)
        candidate_rows.append((row, cache_key, cache_key in persistent_cache))

    rows_to_fetch = candidate_rows if force_refresh else [item for item in candidate_rows if not item[2]]

    def fetch_row(item: tuple[object, str, bool]) -> dict[str, object]:
        row, cache_key, was_cached = item
        payload = get_real_match_stats(
            home_team=str(row.home_team),
            away_team=str(row.away_team),
            status=str(row.status),
            date_text=str(row.date_text),
            event_timestamp=getattr(row, "event_timestamp", None),
            prefer_persistent_cache=not force_refresh,
            persist_result=True,
            force_refresh=force_refresh,
        )
        return {
            "cache_key": cache_key,
            "was_cached": was_cached,
            "available": bool(payload.get("available")),
            "cache_hit": bool(payload.get("cache_hit")),
        }

    fetched_results: list[dict[str, object]] = []
    if rows_to_fetch:
        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(rows_to_fetch)))) as executor:
            fetched_results = list(executor.map(fetch_row, rows_to_fetch))

    saved_now = sum(1 for item in fetched_results if item.get("available"))
    already_cached = sum(1 for _, _, was_cached in candidate_rows if was_cached)
    available_total = already_cached + saved_now

    return {
        "candidates": len(candidate_rows),
        "already_cached": already_cached,
        "fetched_now": len(rows_to_fetch),
        "saved_now": saved_now,
        "available_total": available_total,
        "missing_total": max(len(candidate_rows) - available_total, 0),
    }
