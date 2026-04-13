from __future__ import annotations

import copy
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


PROBABILITY_CALIBRATION_BINS = [0.0, 0.20, 0.35, 0.50, 0.65, 1.01]
DEFAULT_MODEL_CONFIG: dict[str, object] = {
    "poisson": {
        "max_goals": 5,
        "league_home_default": 1.35,
        "league_away_default": 1.10,
        "min_expected_home": 0.15,
        "min_expected_away": 0.10,
    },
    "calibration": {
        "enabled": True,
        "bins": PROBABILITY_CALIBRATION_BINS,
        "min_history_matches": 40,
        "min_bucket_matches": 8,
        "baseline_weight": 0.20,
        "max_adjustment_weight": 0.70,
        "weight_sample_size": 30.0,
    },
    "betting": {
        "kelly_fractional": 0.25,
    },
    "safe_score": {
        "prob_weight": 0.55,
        "ev_weight": 2.5,
        "ev_cap": 0.10,
        "bookmakers_weight": 0.20,
        "bookmakers_cap": 20,
        "odd_weight": 0.05,
        "odd_reference": 1.20,
        "odd_span": 1.20,
    },
}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def default_model_config() -> dict[str, object]:
    return copy.deepcopy(DEFAULT_MODEL_CONFIG)


def normalize_model_config(model_config: dict[str, object] | None) -> dict[str, object]:
    normalized = default_model_config()
    if not isinstance(model_config, dict):
        return normalized

    poisson_default = DEFAULT_MODEL_CONFIG["poisson"] if isinstance(DEFAULT_MODEL_CONFIG["poisson"], dict) else {}
    poisson_raw = model_config.get("poisson", {})
    if isinstance(poisson_raw, dict):
        poisson_cfg = normalized["poisson"]
        if isinstance(poisson_cfg, dict):
            poisson_cfg["max_goals"] = _safe_int(poisson_raw.get("max_goals"), _safe_int(poisson_default.get("max_goals"), 5))
            poisson_cfg["max_goals"] = max(3, min(10, int(poisson_cfg["max_goals"])))
            poisson_cfg["league_home_default"] = _clamp(
                _safe_float(poisson_raw.get("league_home_default"), _safe_float(poisson_default.get("league_home_default"), 1.35)),
                0.4,
                4.0,
            )
            poisson_cfg["league_away_default"] = _clamp(
                _safe_float(poisson_raw.get("league_away_default"), _safe_float(poisson_default.get("league_away_default"), 1.10)),
                0.3,
                4.0,
            )
            poisson_cfg["min_expected_home"] = _clamp(
                _safe_float(poisson_raw.get("min_expected_home"), _safe_float(poisson_default.get("min_expected_home"), 0.15)),
                0.01,
                2.5,
            )
            poisson_cfg["min_expected_away"] = _clamp(
                _safe_float(poisson_raw.get("min_expected_away"), _safe_float(poisson_default.get("min_expected_away"), 0.10)),
                0.01,
                2.5,
            )

    calibration_default = DEFAULT_MODEL_CONFIG["calibration"] if isinstance(DEFAULT_MODEL_CONFIG["calibration"], dict) else {}
    calibration_raw = model_config.get("calibration", {})
    if isinstance(calibration_raw, dict):
        calibration_cfg = normalized["calibration"]
        if isinstance(calibration_cfg, dict):
            enabled_raw = calibration_raw.get("enabled")
            calibration_cfg["enabled"] = bool(enabled_raw) if enabled_raw is not None else bool(calibration_default.get("enabled", True))
            bins_raw = calibration_raw.get("bins")
            bins_candidate = []
            if isinstance(bins_raw, list):
                for item in bins_raw:
                    try:
                        bins_candidate.append(float(item))
                    except (TypeError, ValueError):
                        continue
            bins_candidate = sorted(set(bins_candidate))
            if len(bins_candidate) < 2:
                bins_candidate = [float(item) for item in calibration_default.get("bins", PROBABILITY_CALIBRATION_BINS)]
            if not bins_candidate:
                bins_candidate = PROBABILITY_CALIBRATION_BINS.copy()
            if bins_candidate[0] > 0.0:
                bins_candidate = [0.0] + bins_candidate
            if bins_candidate[-1] <= 1.0:
                bins_candidate.append(1.01)
            calibration_cfg["bins"] = bins_candidate
            calibration_cfg["min_history_matches"] = max(
                10,
                _safe_int(calibration_raw.get("min_history_matches"), _safe_int(calibration_default.get("min_history_matches"), 40)),
            )
            calibration_cfg["min_bucket_matches"] = max(
                1,
                _safe_int(calibration_raw.get("min_bucket_matches"), _safe_int(calibration_default.get("min_bucket_matches"), 8)),
            )
            calibration_cfg["baseline_weight"] = _clamp(
                _safe_float(calibration_raw.get("baseline_weight"), _safe_float(calibration_default.get("baseline_weight"), 0.20)),
                0.0,
                1.0,
            )
            calibration_cfg["max_adjustment_weight"] = _clamp(
                _safe_float(
                    calibration_raw.get("max_adjustment_weight"),
                    _safe_float(calibration_default.get("max_adjustment_weight"), 0.70),
                ),
                0.0,
                1.0,
            )
            calibration_cfg["weight_sample_size"] = _clamp(
                _safe_float(calibration_raw.get("weight_sample_size"), _safe_float(calibration_default.get("weight_sample_size"), 30.0)),
                1.0,
                500.0,
            )

    betting_default = DEFAULT_MODEL_CONFIG["betting"] if isinstance(DEFAULT_MODEL_CONFIG["betting"], dict) else {}
    betting_raw = model_config.get("betting", {})
    if isinstance(betting_raw, dict):
        betting_cfg = normalized["betting"]
        if isinstance(betting_cfg, dict):
            betting_cfg["kelly_fractional"] = _clamp(
                _safe_float(betting_raw.get("kelly_fractional"), _safe_float(betting_default.get("kelly_fractional"), 0.25)),
                0.0,
                1.0,
            )

    safe_default = DEFAULT_MODEL_CONFIG["safe_score"] if isinstance(DEFAULT_MODEL_CONFIG["safe_score"], dict) else {}
    safe_raw = model_config.get("safe_score", {})
    if isinstance(safe_raw, dict):
        safe_cfg = normalized["safe_score"]
        if isinstance(safe_cfg, dict):
            safe_cfg["prob_weight"] = _clamp(
                _safe_float(safe_raw.get("prob_weight"), _safe_float(safe_default.get("prob_weight"), 0.55)),
                0.0,
                5.0,
            )
            safe_cfg["ev_weight"] = _clamp(
                _safe_float(safe_raw.get("ev_weight"), _safe_float(safe_default.get("ev_weight"), 2.5)),
                0.0,
                10.0,
            )
            safe_cfg["ev_cap"] = _clamp(
                _safe_float(safe_raw.get("ev_cap"), _safe_float(safe_default.get("ev_cap"), 0.10)),
                0.0,
                1.0,
            )
            safe_cfg["bookmakers_weight"] = _clamp(
                _safe_float(safe_raw.get("bookmakers_weight"), _safe_float(safe_default.get("bookmakers_weight"), 0.20)),
                0.0,
                5.0,
            )
            safe_cfg["bookmakers_cap"] = max(
                1,
                _safe_int(safe_raw.get("bookmakers_cap"), _safe_int(safe_default.get("bookmakers_cap"), 20)),
            )
            safe_cfg["odd_weight"] = _clamp(
                _safe_float(safe_raw.get("odd_weight"), _safe_float(safe_default.get("odd_weight"), 0.05)),
                0.0,
                5.0,
            )
            safe_cfg["odd_reference"] = _clamp(
                _safe_float(safe_raw.get("odd_reference"), _safe_float(safe_default.get("odd_reference"), 1.20)),
                1.01,
                20.0,
            )
            safe_cfg["odd_span"] = _clamp(
                _safe_float(safe_raw.get("odd_span"), _safe_float(safe_default.get("odd_span"), 1.20)),
                0.01,
                20.0,
            )

    return normalized


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def _safe_mean(series: pd.Series, default: float) -> float:
    value = series.mean()
    if pd.isna(value):
        return default
    return float(value)


def _expected_goals(
    played: pd.DataFrame,
    home_team: str,
    away_team: str,
    *,
    model_config: dict[str, object] | None = None,
) -> tuple[float, float]:
    cfg = normalize_model_config(model_config)
    poisson_cfg = cfg["poisson"] if isinstance(cfg.get("poisson"), dict) else {}
    league_home_default = _safe_float(poisson_cfg.get("league_home_default"), 1.35)
    league_away_default = _safe_float(poisson_cfg.get("league_away_default"), 1.10)
    min_expected_home = _safe_float(poisson_cfg.get("min_expected_home"), 0.15)
    min_expected_away = _safe_float(poisson_cfg.get("min_expected_away"), 0.10)

    league_home_avg = _safe_mean(played["home_goals"], default=league_home_default)
    league_away_avg = _safe_mean(played["away_goals"], default=league_away_default)

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

    expected_home = max(min_expected_home, attack_home * defense_away * league_home_avg)
    expected_away = max(min_expected_away, attack_away * defense_home * league_away_avg)

    return expected_home, expected_away


def _calculate_raw_match_probabilities(
    played: pd.DataFrame,
    home_team: str,
    away_team: str,
    max_goals: int | None = None,
    *,
    model_config: dict[str, object] | None = None,
) -> MatchProbabilities:
    cfg = normalize_model_config(model_config)
    poisson_cfg = cfg["poisson"] if isinstance(cfg.get("poisson"), dict) else {}
    resolved_max_goals = _safe_int(max_goals, _safe_int(poisson_cfg.get("max_goals"), 5)) if max_goals is not None else _safe_int(poisson_cfg.get("max_goals"), 5)
    resolved_max_goals = max(3, min(10, resolved_max_goals))
    expected_home, expected_away = _expected_goals(played, home_team, away_team, model_config=cfg)

    probs_home = [_poisson_pmf(i, expected_home) for i in range(resolved_max_goals + 1)]
    probs_away = [_poisson_pmf(i, expected_away) for i in range(resolved_max_goals + 1)]

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


def _bucket_for_probability(probability: float, bins: list[float]) -> int:
    clamped = max(0.0, min(1.0, float(probability)))
    for idx in range(len(bins) - 1):
        lower = bins[idx]
        upper = bins[idx + 1]
        if lower <= clamped < upper:
            return idx
    return max(0, len(bins) - 2)


def build_probability_calibration(
    matches_df: pd.DataFrame,
    *,
    min_history_matches: int | None = None,
    min_bucket_matches: int | None = None,
    max_goals: int | None = None,
    model_config: dict[str, object] | None = None,
) -> dict[str, object]:
    cfg = normalize_model_config(model_config)
    calibration_cfg = cfg["calibration"] if isinstance(cfg.get("calibration"), dict) else {}
    poisson_cfg = cfg["poisson"] if isinstance(cfg.get("poisson"), dict) else {}
    resolved_min_history = (
        max(10, _safe_int(min_history_matches, 40))
        if min_history_matches is not None
        else max(10, _safe_int(calibration_cfg.get("min_history_matches"), 40))
    )
    resolved_min_bucket = (
        max(1, _safe_int(min_bucket_matches, 8))
        if min_bucket_matches is not None
        else max(1, _safe_int(calibration_cfg.get("min_bucket_matches"), 8))
    )
    resolved_max_goals = (
        max(3, min(10, _safe_int(max_goals, 5)))
        if max_goals is not None
        else max(3, min(10, _safe_int(poisson_cfg.get("max_goals"), 5)))
    )
    bins = calibration_cfg.get("bins", PROBABILITY_CALIBRATION_BINS)
    if not isinstance(bins, list):
        bins = PROBABILITY_CALIBRATION_BINS
    bins = [float(item) for item in bins]

    if "status" in matches_df.columns:
        finished = matches_df[matches_df["status"] == "Finalizado"].copy()
    else:
        finished = matches_df.copy()
    finished = _sort_finished_matches_for_backtest(finished)
    finished = finished.dropna(subset=["home_goals", "away_goals"])

    if finished.empty or len(finished) <= resolved_min_history:
        return {
            "bins": bins,
            "min_bucket_matches": resolved_min_bucket,
            "markets": {},
            "market_base": {},
        }

    rows: list[dict[str, object]] = []
    for idx in range(resolved_min_history, len(finished)):
        row = finished.iloc[idx]
        history = finished.iloc[:idx].copy()
        try:
            probs = _calculate_raw_match_probabilities(
                history,
                str(row["home_team"]),
                str(row["away_team"]),
                max_goals=resolved_max_goals,
                model_config=cfg,
            )
        except Exception:
            continue

        actual_market = result_to_market(float(row["home_goals"]), float(row["away_goals"]))
        for market, probability in probability_map(probs).items():
            rows.append(
                {
                    "market": market,
                    "bucket": _bucket_for_probability(float(probability), bins),
                    "hit": market == actual_market,
                }
            )

    if not rows:
        return {
            "bins": bins,
            "min_bucket_matches": resolved_min_bucket,
            "markets": {},
            "market_base": {},
        }

    calibration_df = pd.DataFrame(rows)
    market_base = (
        calibration_df.groupby("market", observed=False)["hit"].mean().to_dict()
        if not calibration_df.empty
        else {}
    )

    markets: dict[str, dict[int, dict[str, float | int]]] = {}
    grouped = calibration_df.groupby(["market", "bucket"], observed=False)["hit"].agg(["mean", "size"]).reset_index()
    for item in grouped.itertuples(index=False):
        market = str(item.market)
        bucket = int(item.bucket)
        markets.setdefault(market, {})[bucket] = {
            "hit_rate": float(item.mean),
            "samples": int(item.size),
        }

    return {
        "bins": bins,
        "min_bucket_matches": resolved_min_bucket,
        "markets": markets,
        "market_base": {str(k): float(v) for k, v in market_base.items()},
    }


def _get_probability_calibration(
    matches_df: pd.DataFrame,
    *,
    min_history_matches: int,
    min_bucket_matches: int,
    max_goals: int,
    bins: list[float],
    model_config: dict[str, object] | None = None,
) -> dict[str, object]:
    cache = matches_df.attrs.setdefault("_probability_calibration_cache", {})
    cache_key = (min_history_matches, min_bucket_matches, max_goals, tuple(round(float(item), 6) for item in bins), len(matches_df))
    if cache_key not in cache:
        cache[cache_key] = build_probability_calibration(
            matches_df,
            min_history_matches=min_history_matches,
            min_bucket_matches=min_bucket_matches,
            max_goals=max_goals,
            model_config=model_config,
        )
    return cache[cache_key]


def _apply_probability_calibration(
    probs: MatchProbabilities,
    calibration: dict[str, object],
    *,
    model_config: dict[str, object] | None = None,
) -> MatchProbabilities:
    cfg = normalize_model_config(model_config)
    calibration_cfg = cfg["calibration"] if isinstance(cfg.get("calibration"), dict) else {}
    market_map = probability_map(probs)
    bins = calibration.get("bins", PROBABILITY_CALIBRATION_BINS)
    min_bucket_matches = int(calibration.get("min_bucket_matches", 0))
    calibration_markets = calibration.get("markets", {})
    market_base = calibration.get("market_base", {})
    baseline_weight = _clamp(_safe_float(calibration_cfg.get("baseline_weight"), 0.20), 0.0, 1.0)
    max_adjustment_weight = _clamp(_safe_float(calibration_cfg.get("max_adjustment_weight"), 0.70), 0.0, 1.0)
    weight_sample_size = max(1.0, _safe_float(calibration_cfg.get("weight_sample_size"), 30.0))

    adjusted: dict[str, float] = {}
    for market, raw_probability in market_map.items():
        bucket = _bucket_for_probability(float(raw_probability), list(bins))
        bucket_info = calibration_markets.get(market, {}).get(bucket)
        if not bucket_info:
            adjusted[market] = float(raw_probability)
            continue

        samples = int(bucket_info.get("samples", 0))
        if samples < min_bucket_matches:
            adjusted[market] = float(raw_probability)
            continue

        empirical_rate = float(bucket_info.get("hit_rate", raw_probability))
        baseline_rate = float(market_base.get(market, empirical_rate))
        stabilized_rate = empirical_rate * (1.0 - baseline_weight) + baseline_rate * baseline_weight
        weight = min(max_adjustment_weight, (samples / weight_sample_size) * max_adjustment_weight)
        adjusted_probability = (float(raw_probability) * (1.0 - weight)) + (stabilized_rate * weight)
        adjusted[market] = max(0.01, adjusted_probability)

    normalizer = sum(adjusted.values())
    if normalizer <= 0:
        return probs

    return MatchProbabilities(
        home_win=adjusted["Casa"] / normalizer,
        draw=adjusted["Empate"] / normalizer,
        away_win=adjusted["Fora"] / normalizer,
        expected_home_goals=probs.expected_home_goals,
        expected_away_goals=probs.expected_away_goals,
        btts_yes=probs.btts_yes,
        under_25=probs.under_25,
        over_25=probs.over_25,
        top_scorelines=probs.top_scorelines,
    )


def calculate_match_probabilities(
    matches_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    max_goals: int | None = None,
    *,
    min_calibration_history: int | None = None,
    min_calibration_bucket: int | None = None,
    model_config: dict[str, object] | None = None,
) -> MatchProbabilities:
    cfg = normalize_model_config(model_config)
    poisson_cfg = cfg["poisson"] if isinstance(cfg.get("poisson"), dict) else {}
    calibration_cfg = cfg["calibration"] if isinstance(cfg.get("calibration"), dict) else {}
    resolved_max_goals = (
        max(3, min(10, _safe_int(max_goals, 5)))
        if max_goals is not None
        else max(3, min(10, _safe_int(poisson_cfg.get("max_goals"), 5)))
    )
    resolved_min_history = (
        max(10, _safe_int(min_calibration_history, 40))
        if min_calibration_history is not None
        else max(10, _safe_int(calibration_cfg.get("min_history_matches"), 40))
    )
    resolved_min_bucket = (
        max(1, _safe_int(min_calibration_bucket, 8))
        if min_calibration_bucket is not None
        else max(1, _safe_int(calibration_cfg.get("min_bucket_matches"), 8))
    )
    bins = calibration_cfg.get("bins", PROBABILITY_CALIBRATION_BINS)
    if not isinstance(bins, list):
        bins = PROBABILITY_CALIBRATION_BINS
    bins = [float(item) for item in bins]

    played = matches_df[matches_df["status"] == "Finalizado"].copy()
    if played.empty:
        raise ValueError("Nao ha jogos finalizados suficientes para calcular probabilidades.")

    raw_probs = _calculate_raw_match_probabilities(
        played,
        home_team,
        away_team,
        max_goals=resolved_max_goals,
        model_config=cfg,
    )
    if not bool(calibration_cfg.get("enabled", True)):
        return raw_probs

    calibration = _get_probability_calibration(
        matches_df,
        min_history_matches=resolved_min_history,
        min_bucket_matches=resolved_min_bucket,
        max_goals=resolved_max_goals,
        bins=bins,
        model_config=cfg,
    )
    if not calibration.get("markets"):
        return raw_probs
    return _apply_probability_calibration(raw_probs, calibration, model_config=cfg)


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
    kelly_fractional: float | None = None,
    *,
    model_config: dict[str, object] | None = None,
) -> BettingSuggestion:
    cfg = normalize_model_config(model_config)
    betting_cfg = cfg["betting"] if isinstance(cfg.get("betting"), dict) else {}
    resolved_kelly_fractional = (
        _clamp(_safe_float(kelly_fractional, 0.25), 0.0, 1.0)
        if kelly_fractional is not None
        else _clamp(_safe_float(betting_cfg.get("kelly_fractional"), 0.25), 0.0, 1.0)
    )

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
        kelly = kelly_full * resolved_kelly_fractional
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
    kelly_fractional: float | None = None,
    min_model_prob: float = 0.55,
    min_expected_value: float = 0.02,
    max_odd: float = 2.20,
    min_bookmakers: int = 8,
    *,
    model_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    cfg = normalize_model_config(model_config)
    betting_cfg = cfg["betting"] if isinstance(cfg.get("betting"), dict) else {}
    safe_cfg = cfg["safe_score"] if isinstance(cfg.get("safe_score"), dict) else {}
    resolved_kelly_fractional = (
        _clamp(_safe_float(kelly_fractional, 0.25), 0.0, 1.0)
        if kelly_fractional is not None
        else _clamp(_safe_float(betting_cfg.get("kelly_fractional"), 0.25), 0.0, 1.0)
    )
    prob_weight = _safe_float(safe_cfg.get("prob_weight"), 0.55)
    ev_weight = _safe_float(safe_cfg.get("ev_weight"), 2.5)
    ev_cap = max(0.0, _safe_float(safe_cfg.get("ev_cap"), 0.10))
    bookmakers_weight = _safe_float(safe_cfg.get("bookmakers_weight"), 0.20)
    bookmakers_cap = max(1, _safe_int(safe_cfg.get("bookmakers_cap"), 20))
    odd_weight = _safe_float(safe_cfg.get("odd_weight"), 0.05)
    odd_reference = max(1.01, _safe_float(safe_cfg.get("odd_reference"), 1.20))
    odd_span = max(0.01, _safe_float(safe_cfg.get("odd_span"), 1.20))

    fixtures = matches_df[matches_df["status"] == "Agendado"].copy()
    fixtures = fixtures.dropna(subset=["odds_home", "odds_draw", "odds_away"])

    rows: list[dict] = []
    for row in fixtures.itertuples(index=False):
        try:
            probs = calculate_match_probabilities(matches_df, row.home_team, row.away_team, model_config=cfg)
            tip = suggest_bet_strategy(
                probs=probs,
                odd_home=float(row.odds_home),
                odd_draw=float(row.odds_draw),
                odd_away=float(row.odds_away),
                bankroll=float(bankroll),
                kelly_fractional=resolved_kelly_fractional,
                model_config=cfg,
            )
        except Exception:
            continue

        bookmakers = int(row.bookmakers) if row.bookmakers is not None else 0
        safety_score = (
            tip.model_probability * prob_weight
            + max(0.0, min(ev_cap, tip.expected_value)) * ev_weight
            + max(0, min(bookmakers, bookmakers_cap)) / bookmakers_cap * bookmakers_weight
            + max(0.0, 1.0 - (tip.best_odd - odd_reference) / odd_span) * odd_weight
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
    kelly_fractional: float | None = None,
    min_history_matches: int = 30,
    max_evaluated_matches: int = 120,
    model_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    cfg = normalize_model_config(model_config)
    betting_cfg = cfg["betting"] if isinstance(cfg.get("betting"), dict) else {}
    resolved_kelly_fractional = (
        _clamp(_safe_float(kelly_fractional, 0.25), 0.0, 1.0)
        if kelly_fractional is not None
        else _clamp(_safe_float(betting_cfg.get("kelly_fractional"), 0.25), 0.0, 1.0)
    )

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
            probs = calculate_match_probabilities(history, str(row["home_team"]), str(row["away_team"]), model_config=cfg)
            tip = suggest_bet_strategy(
                probs=probs,
                odd_home=float(row["odds_home"]),
                odd_draw=float(row["odds_draw"]),
                odd_away=float(row["odds_away"]),
                bankroll=float(bankroll),
                kelly_fractional=resolved_kelly_fractional,
                model_config=cfg,
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
