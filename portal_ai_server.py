from __future__ import annotations

import json
import sys
import threading
import time
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
from gerar_copa_mundo_html import build_world_cup_schedule_html
from gerar_html import build_index_html
from nvidia_client import request_nvidia_completion
from real_match_stats import (
    build_projection_payload,
    build_team_stat_profile,
    build_team_stat_projection,
    get_real_match_stats,
    prefetch_finished_match_stats,
)
from scraper import COMPETITIONS, load_all_matches


HOST = "0.0.0.0"
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

        frame = load_all_matches()
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
) -> dict[str, object]:
    global _matches_cache
    _emit_refresh_progress(progress_callback, 5, "Limpando cache local de partidas...", "cache")
    with _cache_lock:
        _matches_cache = None

    _emit_refresh_progress(progress_callback, 18, "Buscando resultados e odds mais recentes...", "fetch_matches")
    matches_frame = get_cached_matches()
    _emit_refresh_progress(progress_callback, 38, "Atualizando historico de cartoes e escanteios...", "prefetch_real_stats")
    real_stats_report = prefetch_finished_match_stats(
        matches_frame,
        competitions=list(COMPETITIONS.keys()),
        per_competition_limit=20,
        max_workers=6,
        force_refresh=False,
    )
    _emit_refresh_progress(progress_callback, 62, "Consolidando dados por competicao...", "prepare_frames")
    competition_frames = {
        competition: matches_frame[matches_frame["competition"] == competition].copy()
        for competition in COMPETITIONS
    }
    _emit_refresh_progress(progress_callback, 82, "Regenerando o HTML do painel...", "build_html")
    html = build_index_html(competition_frames=competition_frames)
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
                        f"Melhor leitura por valor: {market_label(str(tip.best_market), str(row.home_team), str(row.away_team))} | EV {tip.expected_value * 100:.2f}% | stake {tip.suggested_stake:.2f}",
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


class PortalAIHandler(BaseHTTPRequestHandler):
    server_version = "PortalAIServer/1.0"

    def _send_json(self, payload: dict[str, object], status_code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
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
                payload = refresh_copa_snapshot()
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
                if home_history is not None and away_history is not None:
                    home_profile = build_team_stat_profile(pd.DataFrame(home_history), home_team, event_timestamp=None)
                    away_profile = build_team_stat_profile(pd.DataFrame(away_history), away_team, event_timestamp=None)
                    projection_payload = build_projection_payload(home_profile, away_profile)
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

