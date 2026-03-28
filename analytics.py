from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass
class MatchProbabilities:
    home_win: float
    draw: float
    away_win: float
    expected_home_goals: float
    expected_away_goals: float
    btts_yes: float
    under_25: float
    over_25: float
    top_scorelines: list[tuple[str, float]]


@dataclass
class BettingSuggestion:
    best_market: str
    best_odd: float
    model_probability: float
    implied_probability: float
    expected_value: float
    kelly_fraction: float
    suggested_stake: float


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def _safe_mean(series: pd.Series, default: float) -> float:
    value = series.mean()
    if pd.isna(value):
        return default
    return float(value)


def _expected_goals(played: pd.DataFrame, home_team: str, away_team: str) -> tuple[float, float]:
    league_home_avg = _safe_mean(played["home_goals"], default=1.35)
    league_away_avg = _safe_mean(played["away_goals"], default=1.10)

    home_home = played[played["home_team"] == home_team]
    away_away = played[played["away_team"] == away_team]
    home_def = played[played["away_team"] == home_team]
    away_def = played[played["home_team"] == away_team]

    home_scored_home = _safe_mean(home_home["home_goals"], league_home_avg)
    home_conceded_home = _safe_mean(home_def["home_goals"], league_away_avg)

    away_scored_away = _safe_mean(away_away["away_goals"], league_away_avg)
    away_conceded_away = _safe_mean(away_def["away_goals"], league_home_avg)

    attack_home = home_scored_home / max(league_home_avg, 0.01)
    defense_away = away_conceded_away / max(league_home_avg, 0.01)

    attack_away = away_scored_away / max(league_away_avg, 0.01)
    defense_home = home_conceded_home / max(league_away_avg, 0.01)

    expected_home = max(0.15, attack_home * defense_away * league_home_avg)
    expected_away = max(0.10, attack_away * defense_home * league_away_avg)

    return expected_home, expected_away


def calculate_match_probabilities(
    matches_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    max_goals: int = 5,
) -> MatchProbabilities:
    played = matches_df[matches_df["status"] == "Finalizado"].copy()
    if played.empty:
        raise ValueError("Nao ha jogos finalizados suficientes para calcular probabilidades.")

    expected_home, expected_away = _expected_goals(played, home_team, away_team)

    probs_home = [_poisson_pmf(i, expected_home) for i in range(max_goals + 1)]
    probs_away = [_poisson_pmf(i, expected_away) for i in range(max_goals + 1)]

    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    scoreline_probs: list[tuple[str, float]] = []

    for hg, ph in enumerate(probs_home):
        for ag, pa in enumerate(probs_away):
            p = ph * pa
            scoreline_probs.append((f"{hg} x {ag}", p))

            if hg > ag:
                home_win += p
            elif hg == ag:
                draw += p
            else:
                away_win += p

    normalizer = home_win + draw + away_win
    if normalizer > 0:
        home_win /= normalizer
        draw /= normalizer
        away_win /= normalizer

    p_home_0 = probs_home[0] if probs_home else 0.0
    p_away_0 = probs_away[0] if probs_away else 0.0
    p_00 = p_home_0 * p_away_0
    btts_yes = max(0.0, min(1.0, 1.0 - p_home_0 - p_away_0 + p_00))

    under_25 = 0.0
    for hg, ph in enumerate(probs_home):
        for ag, pa in enumerate(probs_away):
            if hg + ag <= 2:
                under_25 += ph * pa
    over_25 = max(0.0, 1.0 - under_25)

    top_scorelines = sorted(scoreline_probs, key=lambda x: x[1], reverse=True)[:5]

    return MatchProbabilities(
        home_win=home_win,
        draw=draw,
        away_win=away_win,
        expected_home_goals=expected_home,
        expected_away_goals=expected_away,
        btts_yes=btts_yes,
        under_25=under_25,
        over_25=over_25,
        top_scorelines=top_scorelines,
    )


def get_team_context(matches_df: pd.DataFrame, team: str, recent_n: int = 5) -> dict:
    played = matches_df[matches_df["status"] == "Finalizado"].copy()
    if played.empty:
        return {"rank": None, "points": 0, "recent_points": 0, "recent_text": "-"}

    table: dict[str, dict[str, int]] = {}
    for row in played.itertuples(index=False):
        home = str(row.home_team)
        away = str(row.away_team)
        hg = int(row.home_goals)
        ag = int(row.away_goals)

        if home not in table:
            table[home] = {"pts": 0, "gf": 0, "ga": 0}
        if away not in table:
            table[away] = {"pts": 0, "gf": 0, "ga": 0}

        table[home]["gf"] += hg
        table[home]["ga"] += ag
        table[away]["gf"] += ag
        table[away]["ga"] += hg

        if hg > ag:
            table[home]["pts"] += 3
        elif hg < ag:
            table[away]["pts"] += 3
        else:
            table[home]["pts"] += 1
            table[away]["pts"] += 1

    ranking = sorted(
        table.items(),
        key=lambda item: (
            item[1]["pts"],
            item[1]["gf"] - item[1]["ga"],
            item[1]["gf"],
        ),
        reverse=True,
    )
    rank_map = {name: i + 1 for i, (name, _) in enumerate(ranking)}

    team_games = played[(played["home_team"] == team) | (played["away_team"] == team)].head(recent_n)
    recent = []
    recent_points = 0
    for row in team_games.itertuples(index=False):
        is_home = row.home_team == team
        gf = int(row.home_goals if is_home else row.away_goals)
        ga = int(row.away_goals if is_home else row.home_goals)
        if gf > ga:
            recent.append("V")
            recent_points += 3
        elif gf == ga:
            recent.append("E")
            recent_points += 1
        else:
            recent.append("D")

    return {
        "rank": rank_map.get(team),
        "points": table.get(team, {}).get("pts", 0),
        "recent_points": recent_points,
        "recent_text": "-".join(recent) if recent else "-",
    }


def suggest_bet_strategy(
    probs: MatchProbabilities,
    odd_home: float,
    odd_draw: float,
    odd_away: float,
    bankroll: float,
    kelly_fractional: float = 0.25,
) -> BettingSuggestion:
    markets = [
        ("Casa", probs.home_win, odd_home),
        ("Empate", probs.draw, odd_draw),
        ("Fora", probs.away_win, odd_away),
    ]

    candidates = []
    for label, p_model, odd in markets:
        if odd is None or odd <= 1.0:
            continue

        implied = 1.0 / odd
        ev = p_model * odd - 1.0
        kelly_full = max(0.0, (p_model * odd - 1.0) / (odd - 1.0))
        kelly = kelly_full * kelly_fractional
        stake = bankroll * kelly

        candidates.append((label, odd, p_model, implied, ev, kelly, stake))

    if not candidates:
        raise ValueError("Odds invalidas para montar estrategia.")

    best = max(candidates, key=lambda x: x[4])

    return BettingSuggestion(
        best_market=best[0],
        best_odd=best[1],
        model_probability=best[2],
        implied_probability=best[3],
        expected_value=best[4],
        kelly_fraction=best[5],
        suggested_stake=best[6],
    )


def build_safe_bets_table(
    matches_df: pd.DataFrame,
    bankroll: float,
    kelly_fractional: float = 0.25,
    min_model_prob: float = 0.55,
    min_expected_value: float = 0.02,
    max_odd: float = 2.20,
    min_bookmakers: int = 8,
) -> pd.DataFrame:
    fixtures = matches_df[matches_df["status"] == "Agendado"].copy()
    fixtures = fixtures.dropna(subset=["odds_home", "odds_draw", "odds_away"])

    rows: list[dict] = []
    for row in fixtures.itertuples(index=False):
        try:
            probs = calculate_match_probabilities(matches_df, row.home_team, row.away_team)
            tip = suggest_bet_strategy(
                probs=probs,
                odd_home=float(row.odds_home),
                odd_draw=float(row.odds_draw),
                odd_away=float(row.odds_away),
                bankroll=float(bankroll),
                kelly_fractional=float(kelly_fractional),
            )
        except Exception:
            continue

        bookmakers = int(row.bookmakers) if row.bookmakers is not None else 0
        safety_score = (
            tip.model_probability * 0.55
            + max(0.0, min(0.10, tip.expected_value)) * 2.5
            + max(0, min(bookmakers, 20)) / 20 * 0.20
            + max(0.0, 1.0 - (tip.best_odd - 1.20) / 1.20) * 0.05
        )

        rows.append(
            {
                "date_text": row.date_text,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "market": tip.best_market,
                "odd": tip.best_odd,
                "model_probability": tip.model_probability,
                "implied_probability": tip.implied_probability,
                "expected_value": tip.expected_value,
                "stake": tip.suggested_stake,
                "kelly_fraction": tip.kelly_fraction,
                "bookmakers": bookmakers,
                "safety_score": safety_score,
                "match_url": row.match_url,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    filtered = out[
        (out["model_probability"] >= min_model_prob)
        & (out["expected_value"] >= min_expected_value)
        & (out["odd"] <= max_odd)
        & (out["bookmakers"] >= min_bookmakers)
    ].copy()

    if filtered.empty:
        return filtered

    filtered = filtered.sort_values(
        by=["safety_score", "expected_value", "model_probability"],
        ascending=False,
    ).reset_index(drop=True)
    return filtered
