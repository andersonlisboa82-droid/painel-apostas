from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from analytics import calculate_match_probabilities, get_team_context, suggest_bet_strategy
from gerar_copa_mundo_html import build_world_cup_schedule_html, update_world_cup_model_adjustments
from gerar_html import build_index_html
from nvidia_client import request_nvidia_completion
from real_match_stats import (
    build_projection_payload,
    build_team_stat_profile,
    build_team_stat_projection,
    get_real_match_stats,
    prefetch_finished_match_stats,
)
from scraper import COMPETITIONS, load_all_matches, load_competition_matches


HOST = str(os.getenv("PORTAL_AI_HOST", "0.0.0.0")).strip() or "0.0.0.0"
try:
    PORT = int(str(os.getenv("PORT", os.getenv("PORTAL_AI_PORT", "8765"))).strip() or "8765")
except ValueError:
    PORT = 8765
CACHE_TTL_SECONDS = 900
APP_TIMEZONE = ZoneInfo("America/Sao_Paulo")


@dataclass
class CachedMatches:
    loaded_at: float
    frame: pd.DataFrame


_cache_lock = threading.Lock()
_matches_cache: CachedMatches | None = None
_refresh_jobs_lock = threading.Lock()
_refresh_jobs: dict[str, dict[str, object]] = {}
REFRESH_JOB_TTL_SECONDS = 3600


def market_label(market: str, home_team: str, away_team: str) -> str:
    if market == "Casa":
        return f"Vitoria {home_team}"
    if market == "Fora":
        return f"Vitoria {away_team}"
    return "Empate"


def get_cached_matches() -> pd.DataFrame:
    global _matches_cache
    with _cache_lock:
        if _matches_cache and (time.time() - _matches_cache.loaded_at) < CACHE_TTL_SECONDS:
            return _matches_cache.frame.copy()

        frame = _load_all_matches_parallel()
        _matches_cache = CachedMatches(loaded_at=time.time(), frame=frame.copy())
        return frame


def _emit_refresh_progress(
    callback: Callable[[int, str, str], None] | None,
    progress: int,
    message: str,
    stage: str,
) -> None:
    if callback is None:
        return
    try:
        callback(int(progress), str(message), str(stage))
    except Exception:
        return


def _load_all_matches_parallel(
    progress_callback: Callable[[int, str, str], None] | None = None,
) -> pd.DataFrame:
    competitions = list(COMPETITIONS.keys())
    total = len(competitions)
    if total == 0:
        return load_all_matches()

    results: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    done = 0

    _emit_refresh_progress(
        progress_callback,
        18,
        "Iniciando coleta paralela de resultados e odds...",
        "fetch_matches",
    )

    max_workers = min(6, total)
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="scrape-comp") as executor:
        future_to_comp = {
            executor.submit(load_competition_matches, competition): competition
            for competition in competitions
        }
        for future in as_completed(future_to_comp):
            competition = future_to_comp[future]
            try:
                frame = future.result()
                results[competition] = frame
            except Exception as exc:
                failures[competition] = str(exc)
            done += 1
            bounded_progress = 18 + int((done / total) * 18)
            _emit_refresh_progress(
                progress_callback,
                bounded_progress,
                f"Coleta concluida: {done}/{total} competicoes.",
                "fetch_matches",
            )

    if not results:
        details = "; ".join(f"{k}: {v}" for k, v in failures.items()) or "sem detalhes"
        raise RuntimeError(f"Falha ao coletar dados de todas as competicoes ({details}).")

    ordered_frames = [results[competition] for competition in competitions if competition in results]
    frame = pd.concat(ordered_frames, ignore_index=True)

    if failures:
        failed_list = ", ".join(sorted(failures))
        _emit_refresh_progress(
            progress_callback,
            36,
            f"Coleta parcial concluida. Competicoes com falha: {failed_list}.",
            "fetch_matches",
        )
    else:
        _emit_refresh_progress(
            progress_callback,
            36,
            "Coleta de resultados e odds concluida para todas as competicoes.",
            "fetch_matches",
        )

    return frame


def _run_with_stage_heartbeat(
    progress_callback: Callable[[int, str, str], None] | None,
    *,
    progress: int,
    stage: str,
    start_message: str,
    pulse_message: str,
    work: Callable[[], object],
    pulse_seconds: float = 8.0,
) -> object:
    _emit_refresh_progress(progress_callback, progress, start_message, stage)
    if progress_callback is None:
        return work()

    stop_event = threading.Event()
    started_at = time.time()

    def heartbeat() -> None:
        while not stop_event.wait(pulse_seconds):
            elapsed = int(time.time() - started_at)
            _emit_refresh_progress(
                progress_callback,
                progress,
                f"{pulse_message} ({elapsed}s decorridos)...",
                stage,
            )

    ticker = threading.Thread(target=heartbeat, daemon=True, name=f"refresh-heartbeat-{stage}")
    ticker.start()
    try:
        return work()
    finally:
        stop_event.set()
        ticker.join(timeout=0.2)


def _now_iso() -> str:
    return datetime.now(APP_TIMEZONE).isoformat()


def _cleanup_refresh_jobs_locked(now_ts: float) -> None:
    stale_ids = [
        job_id
        for job_id, payload in _refresh_jobs.items()
        if now_ts - float(payload.get("_updated_at_ts", now_ts)) > REFRESH_JOB_TTL_SECONDS
    ]
    for job_id in stale_ids:
        _refresh_jobs.pop(job_id, None)


def _create_refresh_job(kind: str) -> str:
    job_id = uuid4().hex
    now_ts = time.time()
    now_iso = _now_iso()
    with _refresh_jobs_lock:
        _cleanup_refresh_jobs_locked(now_ts)
        _refresh_jobs[job_id] = {
            "job_id": job_id,
            "kind": kind,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "message": "Aguardando inicio da automacao...",
            "created_at": now_iso,
            "updated_at": now_iso,
            "_updated_at_ts": now_ts,
            "done": False,
            "ok": False,
        }
    return job_id


def _update_refresh_job(job_id: str, **fields: object) -> None:
    now_ts = time.time()
    now_iso = _now_iso()
    with _refresh_jobs_lock:
        payload = _refresh_jobs.get(job_id)
        if payload is None:
            return
        payload.update(fields)
        payload["updated_at"] = now_iso
        payload["_updated_at_ts"] = now_ts


def _get_refresh_job(job_id: str) -> dict[str, object] | None:
    with _refresh_jobs_lock:
        payload = _refresh_jobs.get(job_id)
        if payload is None:
            return None
        public_payload = {k: v for k, v in payload.items() if not str(k).startswith("_")}
        return public_payload


def start_refresh_portal_job() -> str:
    job_id = _create_refresh_job("portal")

    def worker() -> None:
        _update_refresh_job(
            job_id,
            status="running",
            stage="init",
            progress=2,
            message="Inicializando automacao de atualizacao...",
        )

        def progress_callback(progress: int, message: str, stage: str) -> None:
            bounded_progress = max(0, min(100, int(progress)))
            _update_refresh_job(
                job_id,
                status="running",
                stage=stage,
                progress=bounded_progress,
                message=message,
            )

        try:
            payload = refresh_portal_snapshot_with_progress(progress_callback=progress_callback)
            _update_refresh_job(
                job_id,
                status="completed",
                stage="done",
                progress=100,
                message="Atualizacao concluida.",
                done=True,
                ok=True,
                result=payload,
                finished_at=_now_iso(),
            )
        except Exception as exc:
            _update_refresh_job(
                job_id,
                status="failed",
                stage="error",
                progress=100,
                message=f"Falha na automacao: {exc}",
                done=True,
                ok=False,
                error=str(exc),
                finished_at=_now_iso(),
            )

    threading.Thread(target=worker, daemon=True, name=f"refresh-portal-{job_id[:8]}").start()
    return job_id


def refresh_portal_snapshot() -> dict[str, object]:
    return refresh_portal_snapshot_with_progress(progress_callback=None)


def refresh_portal_snapshot_with_progress(
    progress_callback: Callable[[int, str, str], None] | None = None,
    *,
    prefetch_real_stats: bool = True,
) -> dict[str, object]:
    global _matches_cache
    _emit_refresh_progress(progress_callback, 5, "Limpando cache local de partidas...", "cache")
    with _cache_lock:
        _matches_cache = None

    matches_frame = _load_all_matches_parallel(progress_callback=progress_callback)
    with _cache_lock:
        _matches_cache = CachedMatches(loaded_at=time.time(), frame=matches_frame.copy())
    if prefetch_real_stats:
        real_stats_report = _run_with_stage_heartbeat(
            progress_callback,
            progress=38,
            stage="prefetch_real_stats",
            start_message="Atualizando historico de cartoes, escanteios, faltas e posse...",
            pulse_message="Atualizando historico de estatisticas reais",
            work=lambda: prefetch_finished_match_stats(
                matches_frame,
                competitions=list(COMPETITIONS.keys()),
                per_competition_limit=20,
                max_workers=6,
                force_refresh=False,
            ),
        )
    else:
        _emit_refresh_progress(
            progress_callback,
            38,
            "Pulando pre-carga de estatisticas reais para acelerar atualizacao...",
            "prefetch_real_stats",
        )
        real_stats_report = {
            "ok": True,
            "skipped": True,
            "message": "Pre-carga de estatisticas reais pulada para modo rapido.",
            "available_total": 0,
            "saved_now": 0,
            "missing_total": 0,
            "errors_total": 0,
        }
    _emit_refresh_progress(progress_callback, 62, "Consolidando dados por competicao...", "prepare_frames")
    competition_frames = {
        competition: matches_frame[matches_frame["competition"] == competition].copy()
        for competition in COMPETITIONS
    }
    html = _run_with_stage_heartbeat(
        progress_callback,
        progress=82,
        stage="build_html",
        start_message="Regenerando o HTML do painel...",
        pulse_message="Regenerando o HTML do painel",
        work=lambda: build_index_html(competition_frames=competition_frames),
    )
    dashboard_path = BASE_DIR / "index.html"
    _emit_refresh_progress(progress_callback, 94, "Salvando arquivo atualizado do painel...", "write_file")
    dashboard_path.write_text(html, encoding="utf-8")
    updated_at = datetime.now(APP_TIMEZONE).strftime("%d/%m/%Y %H:%M:%S")
    _emit_refresh_progress(progress_callback, 100, "Atualizacao concluida.", "done")
    return {"updated_at": updated_at, "dashboard_path": str(dashboard_path), "real_stats_report": real_stats_report}


def refresh_copa_snapshot() -> dict[str, str]:
    """Regenera o copa_do_mundo.html buscando dados frescos do betexplorer."""
    html = build_world_cup_schedule_html()
    copa_path = BASE_DIR / "copa_do_mundo.html"
    copa_path.write_text(html, encoding="utf-8")
    updated_at = datetime.now(APP_TIMEZONE).strftime("%d/%m/%Y %H:%M:%S")
    return {"updated_at": updated_at, "copa_path": str(copa_path)}


def build_match_context_for_date(selected_date: date) -> tuple[pd.DataFrame, str]:
    all_matches = get_cached_matches()
    fixtures = all_matches[all_matches["status"] == "Agendado"].copy()
    fixtures = fixtures.dropna(subset=["odds_home", "odds_draw", "odds_away"])
    if fixtures.empty:
        return pd.DataFrame(), "Nao ha jogos futuros com odds completas no momento."

    fixtures["_event_dt"] = pd.to_datetime(fixtures["event_timestamp"], errors="coerce")
    fixtures = fixtures[fixtures["_event_dt"].dt.date == selected_date].copy()
    fixtures = fixtures.sort_values(by=["_event_dt", "competition", "home_team", "away_team"]).reset_index(drop=True)
    if fixtures.empty:
        return pd.DataFrame(), f"Nao encontrei jogos com odds completas para a data {selected_date.strftime('%d/%m/%Y')}."

    context_blocks: list[str] = []
    enriched_rows: list[dict[str, object]] = []
    for row in fixtures.itertuples(index=False):
        league_df = all_matches[all_matches["competition"] == row.competition].copy()
        try:
            probs = calculate_match_probabilities(league_df, str(row.home_team), str(row.away_team))
            tip = suggest_bet_strategy(
                probs=probs,
                odd_home=float(row.odds_home),
                odd_draw=float(row.odds_draw),
                odd_away=float(row.odds_away),
                bankroll=1000.0,
                kelly_fractional=0.25,
            )
            home_ctx = get_team_context(league_df, str(row.home_team))
            away_ctx = get_team_context(league_df, str(row.away_team))
            top_scorelines = ", ".join(f"{score} ({prob * 100:.1f}%)" for score, prob in probs.top_scorelines[:3])
            context_blocks.append(
                "\n".join(
                    [
                        f"Jogo: {row.home_team} x {row.away_team}",
                        f"Competicao: {row.competition}",
                        f"Data exibida: {row.date_text}",
                        f"Odds 1X2: casa {float(row.odds_home):.2f} | empate {float(row.odds_draw):.2f} | fora {float(row.odds_away):.2f}",
                        f"Casas consideradas: {int(row.bookmakers) if row.bookmakers is not None and not pd.isna(row.bookmakers) else 0}",
                        f"Probabilidades do modelo: casa {probs.home_win * 100:.1f}% | empate {probs.draw * 100:.1f}% | fora {probs.away_win * 100:.1f}%",
                        f"BTTS: {probs.btts_yes * 100:.1f}% | Over 2.5: {probs.over_25 * 100:.1f}% | Under 2.5: {probs.under_25 * 100:.1f}%",
                        f"Gols esperados: mandante {probs.expected_home_goals:.2f} | visitante {probs.expected_away_goals:.2f}",
                        f"Resultado mais provavel no modelo: {market_label(str(tip.best_market), str(row.home_team), str(row.away_team))} | Prob {tip.model_probability * 100:.2f}% | EV informativo {tip.expected_value * 100:.2f}% | stake {tip.suggested_stake:.2f}",
                        f"Contexto mandante: posicao {home_ctx.get('rank')} | pontos {home_ctx.get('points')} | forma {home_ctx.get('recent_text')}",
                        f"Contexto visitante: posicao {away_ctx.get('rank')} | pontos {away_ctx.get('points')} | forma {away_ctx.get('recent_text')}",
                        f"Placares provaveis: {top_scorelines or '-'}",
                    ]
                )
            )
            enriched_rows.append(
                {
                    "competition": row.competition,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "date_text": row.date_text,
                }
            )
        except Exception:
            context_blocks.append(
                "\n".join(
                    [
                        f"Jogo: {row.home_team} x {row.away_team}",
                        f"Competicao: {row.competition}",
                        f"Data exibida: {row.date_text}",
                        f"Odds 1X2: casa {float(row.odds_home):.2f} | empate {float(row.odds_draw):.2f} | fora {float(row.odds_away):.2f}",
                        "Leitura estatistica adicional indisponivel nesta base para este confronto.",
                    ]
                )
            )

    return pd.DataFrame(enriched_rows), "\n\n---\n\n".join(context_blocks)


def run_ai_analysis(prompt: str, selected_date: date) -> str:
    filtered_matches, match_context = build_match_context_for_date(selected_date)
    if filtered_matches.empty:
        raise ValueError(match_context)

    system_prompt = (
        "Voce e um analista quantitativo profissional de futebol. "
        "Responda em portugues do Brasil, com postura institucional, objetiva e honesta. "
        "Use o prompt do usuario como estrutura principal. "
        "Trabalhe apenas com os dados fornecidos nesta base local. "
        "Quando uma etapa do prompt exigir informacao indisponivel nesta base, marque explicitamente como "
        "'nao disponivel nesta base' e siga sem inventar nada. "
        "Priorize consistencia estatistica, controle de risco e clareza executiva."
    )
    user_prompt = (
        f"{prompt}\n\n"
        f"DADOS DISPONIVEIS NO PORTAL PARA {selected_date.strftime('%d/%m/%Y')}:\n"
        f"{match_context}\n\n"
        "Entregue a analise final organizada, com as melhores oportunidades do dia, "
        "sem inventar escalações, lesoes, xG externo ou movimentos de odds que nao estejam presentes na base."
    )
    return request_nvidia_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        top_p=0.9,
        max_tokens=1800,
    )


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _normalize_percent(value: object) -> float | None:
    number = _safe_float(value)
    if number is None:
        return None
    if 0.0 <= number <= 1.0:
        return number * 100.0
    return number


def _resolve_outcome_key(value: object) -> str | None:
    raw = str(value or "").strip().casefold()
    if not raw:
        return None
    if raw in {"casa", "home", "mandante", "1"}:
        return "Casa"
    if raw in {"empate", "draw", "x"}:
        return "Empate"
    if raw in {"fora", "away", "visitante", "2"}:
        return "Fora"
    return None


def _market_implied_prob_percent(odd: float | None) -> float | None:
    if odd is None or odd <= 1.0:
        return None
    return 100.0 / odd


def run_multi_match_analysis(
    selected_matches: list[dict[str, object]],
    *,
    user_focus: str = "",
) -> tuple[str, int]:
    normalized_blocks: list[str] = []
    selected_probs: list[float] = []
    selected_odds: list[float] = []
    valid_count = 0

    for idx, raw_item in enumerate(selected_matches, start=1):
        if not isinstance(raw_item, dict):
            continue

        competition = str(raw_item.get("competition") or "Competicao nao informada").strip()
        home_team = str(raw_item.get("home_team") or raw_item.get("home") or "").strip()
        away_team = str(raw_item.get("away_team") or raw_item.get("away") or "").strip()
        date_label = str(raw_item.get("date_label") or raw_item.get("date") or "").strip()
        status_label = str(raw_item.get("status") or "").strip()
        model_risk_stage = str(raw_item.get("model_risk_stage") or "").strip()
        outcome_key = _resolve_outcome_key(raw_item.get("outcome_key") or raw_item.get("selected_outcome"))
        if not home_team or not away_team or outcome_key is None:
            continue

        probabilities = raw_item.get("probabilities")
        probs = probabilities if isinstance(probabilities, dict) else {}
        odds_payload = raw_item.get("odds")
        odds = odds_payload if isinstance(odds_payload, dict) else {}

        home_prob = _normalize_percent(probs.get("home"))
        draw_prob = _normalize_percent(probs.get("draw"))
        away_prob = _normalize_percent(probs.get("away"))
        btts_prob = _normalize_percent(probs.get("btts"))
        over25_prob = _normalize_percent(probs.get("over25"))
        under25_prob = _normalize_percent(probs.get("under25"))

        odd_home = _safe_float(odds.get("home"))
        odd_draw = _safe_float(odds.get("draw"))
        odd_away = _safe_float(odds.get("away"))

        selected_prob = None
        selected_odd = None
        if outcome_key == "Casa":
            selected_prob = home_prob
            selected_odd = odd_home
        elif outcome_key == "Empate":
            selected_prob = draw_prob
            selected_odd = odd_draw
        elif outcome_key == "Fora":
            selected_prob = away_prob
            selected_odd = odd_away

        outcome_label = str(raw_item.get("outcome_label") or "").strip()
        if not outcome_label:
            outcome_label = market_label(outcome_key, home_team, away_team)

        implied_prob = _market_implied_prob_percent(selected_odd)
        edge_pp = None
        if selected_prob is not None and implied_prob is not None:
            edge_pp = selected_prob - implied_prob

        block_lines = [
            f"Selecao {idx}: {competition} | {status_label or 'Status nao informado'}",
            f"Jogo: {home_team} x {away_team}",
            f"Data exibida: {date_label or '-'}",
            f"Resultado escolhido: {outcome_label}",
            (
                "Probabilidade do resultado escolhido: "
                + (f"{selected_prob:.1f}%" if selected_prob is not None else "nao disponivel")
            ),
            (
                "Odd do resultado escolhido: "
                + (f"{selected_odd:.2f}" if selected_odd is not None else "nao disponivel")
            ),
            (
                "Prob. implicita da odd: "
                + (f"{implied_prob:.1f}%" if implied_prob is not None else "nao disponivel")
            ),
            (
                "Edge modelo vs mercado: "
                + (f"{edge_pp:+.1f} p.p." if edge_pp is not None else "nao disponivel")
            ),
            (
                f"Probabilidades 1X2 do modelo: casa {home_prob:.1f}% | empate {draw_prob:.1f}% | fora {away_prob:.1f}%"
                if home_prob is not None and draw_prob is not None and away_prob is not None
                else "Probabilidades 1X2 do modelo: nao disponiveis"
            ),
            (
                f"BTTS {btts_prob:.1f}% | Over 2.5 {over25_prob:.1f}% | Under 2.5 {under25_prob:.1f}%"
                if btts_prob is not None and over25_prob is not None and under25_prob is not None
                else "BTTS/Over/Under: nao disponivel nesta base"
            ),
        ]
        if model_risk_stage:
            block_lines.append(f"Faixa de risco no portal: {model_risk_stage}")

        normalized_blocks.append("\n".join(block_lines))
        if selected_prob is not None:
            selected_probs.append(max(0.0, min(100.0, selected_prob)))
        if selected_odd is not None and selected_odd > 1.0:
            selected_odds.append(selected_odd)
        valid_count += 1

    if valid_count == 0:
        raise ValueError("Nenhuma selecao valida foi enviada para analise.")

    average_prob = sum(selected_probs) / len(selected_probs) if selected_probs else None
    combined_hit_prob = 1.0
    if selected_probs:
        for prob in selected_probs:
            combined_hit_prob *= max(0.0, min(1.0, prob / 100.0))
        combined_hit_prob *= 100.0
    else:
        combined_hit_prob = None

    combined_odd = 1.0
    if selected_odds:
        for odd in selected_odds:
            combined_odd *= odd
    else:
        combined_odd = None

    summary_lines = [
        f"Total de selecoes analisadas: {valid_count}",
        (
            f"Probabilidade media das selecoes escolhidas: {average_prob:.1f}%"
            if average_prob is not None
            else "Probabilidade media: nao disponivel"
        ),
        (
            f"Probabilidade conjunta aproximada (independencia): {combined_hit_prob:.2f}%"
            if combined_hit_prob is not None
            else "Probabilidade conjunta aproximada: nao disponivel"
        ),
        (
            f"Odd acumulada aproximada: {combined_odd:.2f}"
            if combined_odd is not None
            else "Odd acumulada aproximada: nao disponivel"
        ),
    ]

    focus_text = user_focus.strip()
    if not focus_text:
        focus_text = "Sem foco extra informado; priorize controle de risco e transparencia estatistica."

    system_prompt = (
        "Voce e um analista quantitativo de apostas em futebol, objetivo e institucional. "
        "Use apenas os dados enviados. Nao invente informacoes externas. "
        "Explique riscos, consistencia estatistica e limite de confianca com clareza."
    )
    selections_context = "\n\n---\n\n".join(normalized_blocks)
    user_prompt = (
        "Analise as selecoes escolhidas pelo usuario e avalie a probabilidade dos resultados selecionados.\n\n"
        f"FOCO DO USUARIO:\n{focus_text}\n\n"
        f"RESUMO GERAL:\n{chr(10).join(summary_lines)}\n\n"
        f"SELECOES:\n{selections_context}\n\n"
        "Entregue:\n"
        "1) Tabela textual curta com cada selecao, probabilidade escolhida, odd, edge e classificacao de risco (baixo/medio/alto).\n"
        "2) Leitura consolidada da combinacao (coerencia das escolhas, pontos fortes, fragilidades e cenarios de quebra).\n"
        "3) Conclusao objetiva: manter, ajustar ou descartar parte das selecoes.\n"
        "4) Sugestao de gestao de banca em 3 faixas (conservadora, equilibrada, agressiva).\n"
        "Se faltar dado em alguma selecao, sinalize explicitamente como 'nao disponivel nesta base'."
    )
    analysis = request_nvidia_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.25,
        top_p=0.9,
        max_tokens=1600,
    )
    return analysis, valid_count


class PortalAIHandler(BaseHTTPRequestHandler):
    server_version = "PortalAIServer/1.0"



class PortalAIHandler(BaseHTTPRequestHandler):
    server_version = "PortalAIServer/1.0"

    def _send_json(self, payload: dict[str, object], status_code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Requested-With")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send_json({"ok": True})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/health":
            self._send_json({"ok": True, "service": "portal-ai", "port": PORT})
            return
        if path == "/api/refresh-portal/status":
            job_id = str((query.get("job_id") or [""])[0]).strip()
            if not job_id:
                self._send_json({"ok": False, "error": "Informe o job_id."}, status_code=400)
                return
            payload = _get_refresh_job(job_id)
            if payload is None:
                self._send_json({"ok": False, "error": "Job nao encontrado."}, status_code=404)
                return
            self._send_json({"ok": True, "job": payload})
            return
        self._send_json({"ok": False, "error": "Rota nao encontrada."}, status_code=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/refresh-portal/start":
            job_id = start_refresh_portal_job()
            self._send_json({"ok": True, "job_id": job_id})
            return

        if path == "/api/refresh-portal":
            try:
                payload = refresh_portal_snapshot()
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status_code=500)
                return
            self._send_json({"ok": True, **payload})
            return

        if path == "/api/refresh-copa":
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length) if content_length > 0 else b""
                payload_data: dict[str, object] = {}
                if raw_body:
                    try:
                        parsed_body = json.loads(raw_body.decode("utf-8"))
                        if isinstance(parsed_body, dict):
                            payload_data = parsed_body
                    except Exception:
                        payload_data = {}

                reset_adjustments = bool(payload_data.get("reset_model_adjustments", False))
                model_adjustments_payload = payload_data.get("model_adjustments")
                applied_adjustments = None
                if reset_adjustments or isinstance(model_adjustments_payload, dict):
                    applied_adjustments = update_world_cup_model_adjustments(
                        model_adjustments_payload if isinstance(model_adjustments_payload, dict) else None,
                        reset=reset_adjustments,
                    )

                payload = refresh_copa_snapshot()
                if applied_adjustments is not None:
                    payload["model_adjustments"] = applied_adjustments
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status_code=500)
                return
            self._send_json({"ok": True, **payload})
            return

        if path == "/api/refresh-match-stats":
            try:
                matches_frame = get_cached_matches()
                payload = prefetch_finished_match_stats(
                    matches_frame,
                    competitions=list(COMPETITIONS.keys()),
                    per_competition_limit=20,
                    max_workers=6,
                    force_refresh=False,
                )
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status_code=500)
                return
            self._send_json({"ok": True, **payload})
            return

        if path == "/api/match-stats":
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            try:
                payload = json.loads(raw_body.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._send_json({"ok": False, "error": "JSON invalido."}, status_code=400)
                return

            home_team = str(payload.get("home_team") or "").strip()
            away_team = str(payload.get("away_team") or "").strip()
            status = str(payload.get("status") or "").strip()
            date_text = str(payload.get("date_text") or "").strip()
            event_timestamp = str(payload.get("event_timestamp") or "").strip() or None
            home_history = payload.get("home_history") if isinstance(payload.get("home_history"), list) else None
            away_history = payload.get("away_history") if isinstance(payload.get("away_history"), list) else None

            if not home_team or not away_team:
                self._send_json({"ok": False, "error": "Times do jogo nao informados."}, status_code=400)
                return

            try:
                stats_payload = get_real_match_stats(
                    home_team=home_team,
                    away_team=away_team,
                    status=status,
                    date_text=date_text,
                    event_timestamp=event_timestamp,
                )
                use_local_history = bool(home_history) and bool(away_history)
                if use_local_history:
                    home_profile = build_team_stat_profile(pd.DataFrame(home_history), home_team, event_timestamp=None)
                    away_profile = build_team_stat_profile(pd.DataFrame(away_history), away_team, event_timestamp=None)
                    projection_payload = build_projection_payload(home_profile, away_profile)
                    if projection_payload.get("available") is not True:
                        matches_frame = get_cached_matches()
                        projection_payload = build_team_stat_projection(
                            matches_df=matches_frame,
                            home_team=home_team,
                            away_team=away_team,
                            event_timestamp=event_timestamp,
                        )
                else:
                    matches_frame = get_cached_matches()
                    projection_payload = build_team_stat_projection(
                        matches_df=matches_frame,
                        home_team=home_team,
                        away_team=away_team,
                        event_timestamp=event_timestamp,
                    )
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status_code=500)
                return

            self._send_json({"ok": True, **stats_payload, "team_projection": projection_payload})
            return

        if path == "/api/multi-match-analysis":
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            try:
                payload = json.loads(raw_body.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._send_json({"ok": False, "error": "JSON invalido."}, status_code=400)
                return

            selected_matches_raw = payload.get("selected_matches")
            selected_matches = selected_matches_raw if isinstance(selected_matches_raw, list) else []
            if not selected_matches:
                self._send_json({"ok": False, "error": "Selecione pelo menos um jogo para a analise."}, status_code=400)
                return
            if len(selected_matches) > 20:
                self._send_json(
                    {"ok": False, "error": "Limite maximo de 20 jogos por analise. Reduza a selecao e tente novamente."},
                    status_code=400,
                )
                return

            user_focus = str(payload.get("user_focus") or "").strip()
            try:
                analysis, processed_count = run_multi_match_analysis(selected_matches, user_focus=user_focus)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status_code=500)
                return

            self._send_json({"ok": True, "analysis": analysis, "processed_matches": processed_count})
            return

        if path != "/api/ai-analysis":
            self._send_json({"ok": False, "error": "Rota nao encontrada."}, status_code=404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "JSON invalido."}, status_code=400)
            return

        selected_date_raw = str(payload.get("selected_date") or "").strip()
        prompt = str(payload.get("prompt") or "").strip()
        if not selected_date_raw:
            self._send_json({"ok": False, "error": "Informe a data da analise."}, status_code=400)
            return
        if not prompt:
            self._send_json({"ok": False, "error": "O prompt da analise veio vazio."}, status_code=400)
            return

        try:
            selected_date = date.fromisoformat(selected_date_raw)
        except ValueError:
            self._send_json({"ok": False, "error": "Data invalida. Use o formato YYYY-MM-DD."}, status_code=400)
            return

        try:
            analysis = run_ai_analysis(prompt=prompt, selected_date=selected_date)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status_code=500)
            return

        self._send_json({"ok": True, "analysis": analysis})

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), PortalAIHandler)
    print(f"Portal AI server ativo em http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()

