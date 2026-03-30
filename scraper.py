from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
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

SOURCE_TIMEZONE = ZoneInfo("UTC")
TARGET_TIMEZONE = ZoneInfo("America/Sao_Paulo")

COMPETITIONS = {
    "Brasileirao": "https://www.betexplorer.com/br/football/brazil/serie-a-betano/",
    "La Liga": "https://www.betexplorer.com/br/football/spain/laliga/",
    "Premier League": "https://www.betexplorer.com/br/football/england/premier-league/",
    "Copa do Mundo": "https://www.betexplorer.com/br/football/world/world-cup-2026/",
}


@dataclass
class MatchRow:
    competition: str
    status: str
    date_text: str
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


def _convert_fixture_datetime_to_target(text: str) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    match = re.fullmatch(r"(\d{2})\.(\d{2})\.\s+(\d{2}):(\d{2})", clean)
    if not match:
        return clean

    day = int(match.group(1))
    month = int(match.group(2))
    hour = int(match.group(3))
    minute = int(match.group(4))

    now_target = datetime.now(TARGET_TIMEZONE)
    year = _infer_year(month, now_target)
    source_dt = datetime(year, month, day, hour, minute, tzinfo=SOURCE_TIMEZONE)
    target_dt = source_dt.astimezone(TARGET_TIMEZONE)
    return target_dt.strftime("%d.%m. %H:%M")


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

        date_text = cells[-1].get_text(" ", strip=True)
        href = match_anchor.get("href")

        rows.append(
            MatchRow(
                competition=competition,
                status="Finalizado",
                date_text=date_text,
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
        datetime_text = _convert_fixture_datetime_to_target(datetime_text)

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
