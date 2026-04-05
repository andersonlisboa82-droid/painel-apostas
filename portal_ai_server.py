from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from analytics import calculate_match_probabilities, get_team_context, suggest_bet_strategy
from nvidia_client import request_nvidia_completion
from scraper import load_all_matches


HOST = "0.0.0.0"
PORT = 8765
CACHE_TTL_SECONDS = 900


@dataclass
class CachedMatches:
    loaded_at: float
    frame: pd.DataFrame


_cache_lock = threading.Lock()
_matches_cache: CachedMatches | None = None


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
        if self.path == "/health":
            self._send_json({"ok": True, "service": "portal-ai", "port": PORT})
            return
        self._send_json({"ok": False, "error": "Rota nao encontrada."}, status_code=404)

    def do_POST(self) -> None:
        if self.path != "/api/ai-analysis":
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
