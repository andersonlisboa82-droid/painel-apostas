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


def result_to_market(home_goals: float, away_goals: float) -> str:
    if float(home_goals) > float(away_goals):
        return "Casa"
    if float(home_goals) < float(away_goals):
        return "Fora"
    return "Empate"


def probability_map(probs: MatchProbabilities) -> dict[str, float]:
    return {
        "Casa": float(probs.home_win),
        "Empate": float(probs.draw),
        "Fora": float(probs.away_win),
    }


def normalized_implied_probabilities(odd_home: float, odd_draw: float, odd_away: float) -> dict[str, float]:
    raw = {
        "Casa": 1.0 / float(odd_home) if odd_home and float(odd_home) > 1.0 else 0.0,
        "Empate": 1.0 / float(odd_draw) if odd_draw and float(odd_draw) > 1.0 else 0.0,
        "Fora": 1.0 / float(odd_away) if odd_away and float(odd_away) > 1.0 else 0.0,
    }
    total = sum(raw.values())
    if total <= 0:
        return raw
    return {market: value / total for market, value in raw.items()}


def pick_highest_probability_market(probs: MatchProbabilities) -> tuple[str, float]:
    return max(probability_map(probs).items(), key=lambda item: item[1])


def get_market_odd(row: pd.Series | dict, market: str) -> float | None:
    mapping = {
        "Casa": row.get("odds_home"),
        "Empate": row.get("odds_draw"),
        "Fora": row.get("odds_away"),
    }
    value = mapping.get(market)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _sort_finished_matches_for_backtest(matches_df: pd.DataFrame) -> pd.DataFrame:
    finished = matches_df[matches_df["status"] == "Finalizado"].copy()
    if finished.empty:
        return finished

    if "event_timestamp" in finished.columns:
        finished["_event_dt"] = pd.to_datetime(finished["event_timestamp"], errors="coerce", utc=True)
        if finished["_event_dt"].notna().any():
            finished = finished.sort_values(by=["_event_dt", "home_team", "away_team"], ascending=True)
            return finished.drop(columns="_event_dt").reset_index(drop=True)

    return finished.iloc[::-1].reset_index(drop=True)


def build_backtest_table(
    matches_df: pd.DataFrame,
    *,
    bankroll: float = 1000.0,
    kelly_fractional: float = 0.25,
    min_history_matches: int = 30,
    max_evaluated_matches: int = 120,
) -> pd.DataFrame:
    finished = _sort_finished_matches_for_backtest(matches_df)
    finished = finished.dropna(subset=["home_goals", "away_goals", "odds_home", "odds_draw", "odds_away"])
    if finished.empty or len(finished) <= min_history_matches:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    start_idx = max(min_history_matches, len(finished) - max_evaluated_matches)

    for idx in range(start_idx, len(finished)):
        row = finished.iloc[idx]
        history = finished.iloc[:idx].copy()
        if len(history) < min_history_matches:
            continue

        try:
            probs = calculate_match_probabilities(history, str(row["home_team"]), str(row["away_team"]))
            tip = suggest_bet_strategy(
                probs=probs,
                odd_home=float(row["odds_home"]),
                odd_draw=float(row["odds_draw"]),
                odd_away=float(row["odds_away"]),
                bankroll=float(bankroll),
                kelly_fractional=float(kelly_fractional),
            )
        except Exception:
            continue

        model_probs = probability_map(probs)
        model_market, model_probability = max(model_probs.items(), key=lambda item: item[1])
        house_probs = normalized_implied_probabilities(
            float(row["odds_home"]),
            float(row["odds_draw"]),
            float(row["odds_away"]),
        )
        house_market, house_probability = max(house_probs.items(), key=lambda item: item[1])
        actual_market = result_to_market(float(row["home_goals"]), float(row["away_goals"]))

        model_odd = get_market_odd(row, model_market)
        house_odd = get_market_odd(row, house_market)
        value_odd = get_market_odd(row, str(tip.best_market))

        rows.append(
            {
                "date_text": row["date_text"],
                "event_timestamp": row.get("event_timestamp"),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "actual_market": actual_market,
                "actual_score": f"{int(row['home_goals'])} x {int(row['away_goals'])}",
                "model_market": model_market,
                "model_probability": model_probability,
                "model_odd": model_odd,
                "model_hit": model_market == actual_market,
                "model_edge": model_probability - house_probs.get(model_market, 0.0),
                "house_market": house_market,
                "house_probability": house_probability,
                "house_odd": house_odd,
                "house_hit": house_market == actual_market,
                "value_market": str(tip.best_market),
                "value_probability": float(tip.model_probability),
                "value_odd": value_odd,
                "value_hit": str(tip.best_market) == actual_market,
                "value_ev": float(tip.expected_value),
                "value_edge": float(tip.model_probability) - float(tip.implied_probability),
                "bookmakers": int(row["bookmakers"]) if "bookmakers" in row and not pd.isna(row["bookmakers"]) else 0,
                "history_size": int(len(history)),
                "market_disagreement": model_market != house_market,
                "model_profit": (float(model_odd) - 1.0) if model_odd and model_market == actual_market else -1.0,
                "house_profit": (float(house_odd) - 1.0) if house_odd and house_market == actual_market else -1.0,
                "value_profit": (float(value_odd) - 1.0) if value_odd and str(tip.best_market) == actual_market else -1.0,
            }
        )

    return pd.DataFrame(rows)


def summarize_backtest(backtest_df: pd.DataFrame) -> dict[str, object]:
    if backtest_df.empty:
        return {
            "total_matches": 0,
            "model_accuracy": 0.0,
            "house_accuracy": 0.0,
            "value_accuracy": 0.0,
            "model_roi": 0.0,
            "house_roi": 0.0,
            "value_roi": 0.0,
            "market_disagreement_rate": 0.0,
            "avg_model_edge": 0.0,
            "avg_value_ev": 0.0,
            "tuning_actions": [],
        }

    def _pct(series: pd.Series) -> float:
        return round(float(series.mean()) * 100, 2)

    def _roi(series: pd.Series) -> float:
        return round(float(series.mean()) * 100, 2)

    model_accuracy = _pct(backtest_df["model_hit"])
    house_accuracy = _pct(backtest_df["house_hit"])
    value_accuracy = _pct(backtest_df["value_hit"])
    model_roi = _roi(backtest_df["model_profit"])
    house_roi = _roi(backtest_df["house_profit"])
    value_roi = _roi(backtest_df["value_profit"])
    disagreement_rate = _pct(backtest_df["market_disagreement"])
    avg_model_edge = round(float(backtest_df["model_edge"].mean()) * 100, 2)
    avg_value_ev = round(float(backtest_df["value_ev"].mean()) * 100, 2)

    tuning_actions: list[str] = []
    if model_accuracy < house_accuracy:
        tuning_actions.append("Suba a probabilidade minima exigida e priorize jogos em que modelo e mercado convergem.")
    if value_roi < 0:
        tuning_actions.append("Reduza a fracao de Kelly e corte odds mais altas ate o ROI do valor voltar para terreno positivo.")
    if disagreement_rate > 40 and model_accuracy < 55:
        tuning_actions.append("Quando houver muita divergencia com as casas, trate a entrada como agressiva e diminua exposicao.")
    if model_accuracy < 50:
        tuning_actions.append("Se o acerto ficar abaixo de 50%, use filtro extra por numero de casas e descarte jogos muito equilibrados.")

    return {
        "total_matches": int(len(backtest_df)),
        "model_accuracy": model_accuracy,
        "house_accuracy": house_accuracy,
        "value_accuracy": value_accuracy,
        "model_roi": model_roi,
        "house_roi": house_roi,
        "value_roi": value_roi,
        "market_disagreement_rate": disagreement_rate,
        "avg_model_edge": avg_model_edge,
        "avg_value_ev": avg_value_ev,
        "tuning_actions": tuning_actions,
    }


def build_probability_buckets(backtest_df: pd.DataFrame) -> pd.DataFrame:
    if backtest_df.empty:
        return pd.DataFrame()

    out = backtest_df.copy()
    out["faixa_modelo"] = pd.cut(
        out["model_probability"],
        bins=[0.0, 0.45, 0.55, 0.65, 1.01],
        labels=["Ate 45%", "45% a 55%", "55% a 65%", "Acima de 65%"],
        include_lowest=True,
    )

    grouped = (
        out.groupby("faixa_modelo", observed=False)
        .agg(
            jogos=("model_hit", "size"),
            acerto_modelo=("model_hit", "mean"),
            acerto_casas=("house_hit", "mean"),
            roi_modelo=("model_profit", "mean"),
            roi_valor=("value_profit", "mean"),
            ev_medio=("value_ev", "mean"),
        )
        .reset_index()
    )
    return grouped


def build_hedge_scenarios(
    *,
    best_market: str,
    odd_home: float,
    odd_draw: float,
    odd_away: float,
    base_stake: float,
) -> pd.DataFrame:
    market_odds = {
        "Casa": float(odd_home),
        "Empate": float(odd_draw),
        "Fora": float(odd_away),
    }
    if best_market not in market_odds or base_stake <= 0:
        return pd.DataFrame()

    hedge_markets = [market for market in market_odds if market != best_market and market_odds[market] > 1.0]
    if len(hedge_markets) != 2:
        return pd.DataFrame()

    profiles = [
        ("Leve", 0.20),
        ("Balanceado", 0.35),
        ("Defensivo", 0.50),
    ]
    rows: list[dict[str, object]] = []

    for profile_name, hedge_ratio in profiles:
        hedge_budget = float(base_stake) * hedge_ratio
        weight = sum(1.0 / market_odds[market] for market in hedge_markets)
        stakes = {
            market: hedge_budget / (market_odds[market] * weight)
            for market in hedge_markets
        }

        outcome_net: dict[str, float] = {}
        for outcome, odd in market_odds.items():
            net = (float(base_stake) * (odd - 1.0)) if outcome == best_market else -float(base_stake)
            for hedge_market in hedge_markets:
                hedge_stake = stakes[hedge_market]
                hedge_odd = market_odds[hedge_market]
                net += hedge_stake * (hedge_odd - 1.0) if outcome == hedge_market else -hedge_stake
            outcome_net[outcome] = net

        rows.append(
            {
                "perfil": profile_name,
                "mercado_principal": best_market,
                "stake_principal": round(float(base_stake), 2),
                "orcamento_hedge": round(hedge_budget, 2),
                f"hedge_{hedge_markets[0].lower()}": round(stakes[hedge_markets[0]], 2),
                f"hedge_{hedge_markets[1].lower()}": round(stakes[hedge_markets[1]], 2),
                "lucro_se_principal_bater": round(outcome_net[best_market], 2),
                f"resultado_{hedge_markets[0].lower()}": round(outcome_net[hedge_markets[0]], 2),
                f"resultado_{hedge_markets[1].lower()}": round(outcome_net[hedge_markets[1]], 2),
                "pior_cenario": round(min(outcome_net.values()), 2),
            }
        )

    return pd.DataFrame(rows)
