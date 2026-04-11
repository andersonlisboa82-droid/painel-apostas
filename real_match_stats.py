from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import threading
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent
SOURCE_NAME = "KickingData"
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

    return {
        "available": False,
        "source": SOURCE_NAME,
        "message": "Nao encontrei cartoes e escanteios reais para este jogo na fonte complementar.",
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
    if (status or "").strip().lower() != "finalizado":
        return {
            "available": False,
            "source": SOURCE_NAME,
            "message": "Estatisticas reais ficam disponiveis apenas para jogos finalizados.",
        }

    cache_key = build_match_stats_cache_key(home_team, away_team, date_text, event_timestamp=event_timestamp)
    if prefer_persistent_cache and not force_refresh:
        cached_payload = get_persisted_match_stats(
            home_team=home_team,
            away_team=away_team,
            date_text=date_text,
            event_timestamp=event_timestamp,
        )
        if cached_payload:
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
