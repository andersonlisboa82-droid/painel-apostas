from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dt_time
import re
import unicodedata
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

SOURCE_TIMEZONE = ZoneInfo("Europe/Prague")
TARGET_TIMEZONE = ZoneInfo("America/Sao_Paulo")

COMPETITIONS = {
    "Brasileirao": "https://www.betexplorer.com/br/football/brazil/serie-a-betano/",
    "La Liga": "https://www.betexplorer.com/br/football/spain/laliga/",
    "Premier League": "https://www.betexplorer.com/br/football/england/premier-league/",
    "Copa Sul-Americana": "https://www.betexplorer.com/br/football/south-america/copa-sudamericana/",
    "Libertadores da America": "https://www.betexplorer.com/br/football/south-america/copa-libertadores/",
    "Copa do Mundo": "https://www.betexplorer.com/br/football/world/world-cup-2026/",
}


@dataclass
class MatchRow:
    competition: str
    status: str
    date_text: str
    event_timestamp: str | None
    home_team: str
    away_team: str
    home_goals: float | None
    away_goals: float | None
    odds_home: float | None
    odds_draw: float | None
    odds_away: float | None
    bookmakers: int | None
    match_url: str | None


def _fetch_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def _extract_odd(td) -> float | None:
    if td is None:
        return None

    direct = td.get("data-odd")
    if direct:
        try:
            return float(direct)
        except ValueError:
            pass

    btn = td.find("button")
    if btn and btn.get("data-odd"):
        try:
            return float(btn["data-odd"])
        except ValueError:
            return None

    nested = td.find(attrs={"data-odd": True})
    if nested:
        try:
            return float(nested["data-odd"])
        except ValueError:
            return None

    return None


def _parse_match_text(text: str) -> tuple[str, str] | None:
    clean = re.sub(r"\s+", " ", text).strip()
    if " - " not in clean:
        return None
    home, away = clean.split(" - ", 1)
    return home.strip(), away.strip()


def _parse_score(text: str) -> tuple[float, float] | None:
    match = re.search(r"(\d+)\s*[:x]\s*(\d+)", text)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def _get_bookmakers_count(td) -> int | None:
    if td is None:
        return None
    txt = td.get_text(" ", strip=True)
    if not txt:
        return None
    m = re.search(r"(\d+)", txt)
    return int(m.group(1)) if m else None


def _infer_year(month: int, now_target: datetime) -> int:
    year = now_target.year
    if month <= 2 and now_target.month >= 11:
        return year + 1
    if month >= 11 and now_target.month <= 2:
        return year - 1
    return year


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _format_target_datetime(target_dt: datetime, include_time: bool) -> str:
    now_target = datetime.now(TARGET_TIMEZONE)
    day_diff = (target_dt.date() - now_target.date()).days

    if include_time:
        if day_diff == 0:
            return target_dt.strftime("Hoje %H:%M")
        if day_diff == -1:
            return target_dt.strftime("Ontem %H:%M")
        if day_diff == 1:
            return target_dt.strftime("Amanha %H:%M")
        return target_dt.strftime("%d.%m. %H:%M")

    if day_diff == 0:
        return "Hoje"
    if day_diff == -1:
        return "Ontem"
    if day_diff == 1:
        return "Amanha"
    return target_dt.strftime("%d.%m.")


def _convert_source_datetime_to_target_parts(text: str) -> tuple[str, str | None]:
    clean = re.sub(r"\s+", " ", text).strip()
    ascii_clean = _strip_accents(clean).casefold()

    relative_with_time = re.fullmatch(r"(hoje|ontem|amanha)\s+(\d{2}):(\d{2})", ascii_clean)
    if relative_with_time:
        label = relative_with_time.group(1)
        hour = int(relative_with_time.group(2))
        minute = int(relative_with_time.group(3))
        source_now = datetime.now(SOURCE_TIMEZONE)
        offset_days = {"ontem": -1, "hoje": 0, "amanha": 1}[label]
        source_date = source_now.date().fromordinal(source_now.date().toordinal() + offset_days)
        source_dt = datetime.combine(source_date, dt_time(hour=hour, minute=minute), tzinfo=SOURCE_TIMEZONE)
        target_dt = source_dt.astimezone(TARGET_TIMEZONE)
        return _format_target_datetime(target_dt, include_time=True), target_dt.isoformat()

    relative_no_time = re.fullmatch(r"(hoje|ontem|amanha)", ascii_clean)
    if relative_no_time:
        label = relative_no_time.group(1)
        source_now = datetime.now(SOURCE_TIMEZONE)
        offset_days = {"ontem": -1, "hoje": 0, "amanha": 1}[label]
        source_date = source_now.date().fromordinal(source_now.date().toordinal() + offset_days)
        source_dt = datetime.combine(source_date, dt_time(hour=12, minute=0), tzinfo=SOURCE_TIMEZONE)
        target_dt = source_dt.astimezone(TARGET_TIMEZONE)
        return _format_target_datetime(target_dt, include_time=False), target_dt.isoformat()

    full_match = re.fullmatch(r"(\d{2})\.(\d{2})\.\s+(\d{2}):(\d{2})", clean)
    if full_match:
        day = int(full_match.group(1))
        month = int(full_match.group(2))
        hour = int(full_match.group(3))
        minute = int(full_match.group(4))

        now_target = datetime.now(TARGET_TIMEZONE)
        year = _infer_year(month, now_target)
        source_dt = datetime(year, month, day, hour, minute, tzinfo=SOURCE_TIMEZONE)
        target_dt = source_dt.astimezone(TARGET_TIMEZONE)
        return _format_target_datetime(target_dt, include_time=True), target_dt.isoformat()

    date_only_match = re.fullmatch(r"(\d{2})\.(\d{2})\.", clean)
    if date_only_match:
        day = int(date_only_match.group(1))
        month = int(date_only_match.group(2))
        now_target = datetime.now(TARGET_TIMEZONE)
        year = _infer_year(month, now_target)
        source_dt = datetime(year, month, day, 12, 0, tzinfo=SOURCE_TIMEZONE)
        target_dt = source_dt.astimezone(TARGET_TIMEZONE)
        return _format_target_datetime(target_dt, include_time=False), target_dt.isoformat()

    return clean, None


def _convert_source_datetime_to_target(text: str) -> str:
    display_text, _ = _convert_source_datetime_to_target_parts(text)
    return display_text


def _parse_results(competition: str, base_url: str) -> list[MatchRow]:
    url = urljoin(base_url, "results/")
    soup = _fetch_soup(url)
    table = soup.find("table", class_="table-main")
    if table is None:
        return []

    rows: list[MatchRow] = []

    for tr in table.find_all("tr"):
        match_anchor = tr.find("a", class_="in-match")
        if not match_anchor:
            continue

        cells = tr.find_all("td")
        if len(cells) < 6:
            continue

        teams = _parse_match_text(match_anchor.get_text(" ", strip=True))
        if teams is None:
            continue

        score_text = cells[1].get_text(" ", strip=True)
        score = _parse_score(score_text)
        if score is None:
            continue

        odds_cells = tr.find_all("td", class_="table-main__odds")
        odd_h = _extract_odd(odds_cells[0]) if len(odds_cells) > 0 else None
        odd_d = _extract_odd(odds_cells[1]) if len(odds_cells) > 1 else None
        odd_a = _extract_odd(odds_cells[2]) if len(odds_cells) > 2 else None

        date_text, event_timestamp = _convert_source_datetime_to_target_parts(cells[-1].get_text(" ", strip=True))
        href = match_anchor.get("href")

        rows.append(
            MatchRow(
                competition=competition,
                status="Finalizado",
                date_text=date_text,
                event_timestamp=event_timestamp,
                home_team=teams[0],
                away_team=teams[1],
                home_goals=score[0],
                away_goals=score[1],
                odds_home=odd_h,
                odds_draw=odd_d,
                odds_away=odd_a,
                bookmakers=None,
                match_url=urljoin("https://www.betexplorer.com", href) if href else None,
            )
        )

    return rows


def _parse_fixtures(competition: str, base_url: str) -> list[MatchRow]:
    url = urljoin(base_url, "fixtures/")
    soup = _fetch_soup(url)
    table = soup.find("table", class_="table-main")
    if table is None:
        return []

    rows: list[MatchRow] = []
    last_datetime_text = ""

    for tr in table.find_all("tr"):
        match_anchor = tr.find("a", class_="in-match")
        if not match_anchor:
            continue

        cells = tr.find_all("td")
        if len(cells) < 7:
            continue

        datetime_text = cells[0].get_text(" ", strip=True)
        if datetime_text:
            last_datetime_text = datetime_text
        else:
            datetime_text = last_datetime_text
        datetime_text, event_timestamp = _convert_source_datetime_to_target_parts(datetime_text)

        teams = _parse_match_text(match_anchor.get_text(" ", strip=True))
        if teams is None:
            continue

        bs_cell = tr.find("td", class_="table-main__bs")
        bookmakers = _get_bookmakers_count(bs_cell)

        odds_cells = tr.find_all("td", class_="table-main__odds")
        odd_h = _extract_odd(odds_cells[0]) if len(odds_cells) > 0 else None
        odd_d = _extract_odd(odds_cells[1]) if len(odds_cells) > 1 else None
        odd_a = _extract_odd(odds_cells[2]) if len(odds_cells) > 2 else None

        href = match_anchor.get("href")

        rows.append(
            MatchRow(
                competition=competition,
                status="Agendado",
                date_text=datetime_text,
                event_timestamp=event_timestamp,
                home_team=teams[0],
                away_team=teams[1],
                home_goals=None,
                away_goals=None,
                odds_home=odd_h,
                odds_draw=odd_d,
                odds_away=odd_a,
                bookmakers=bookmakers,
                match_url=urljoin("https://www.betexplorer.com", href) if href else None,
            )
        )

    return rows


def load_competition_matches(competition: str) -> pd.DataFrame:
    if competition not in COMPETITIONS:
        raise ValueError(f"Competicao desconhecida: {competition}")

    base_url = COMPETITIONS[competition]
    result_rows = _parse_results(competition, base_url)
    fixture_rows = _parse_fixtures(competition, base_url)

    data = [r.__dict__ for r in (result_rows + fixture_rows)]
    if not data:
        raise ValueError(f"Sem dados disponiveis para {competition}.")

    df = pd.DataFrame(data)

    df = df.drop_duplicates(
        subset=["competition", "status", "home_team", "away_team", "date_text"],
        keep="first",
    ).reset_index(drop=True)

    return df


def load_all_matches() -> pd.DataFrame:
    frames = [load_competition_matches(comp) for comp in COMPETITIONS]
    return pd.concat(frames, ignore_index=True)
