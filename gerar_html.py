from __future__ import annotations

from datetime import datetime
from html import escape
import json
import re
from zoneinfo import ZoneInfo

import pandas as pd

from analytics import (
    build_safe_bets_table,
    calculate_match_probabilities,
    suggest_bet_strategy,
    get_team_context,
)
from scraper import COMPETITIONS, load_competition_matches


APP_TIMEZONE = ZoneInfo("America/Sao_Paulo")

AI_PROMPT_TEMPLATE = """Atue como um analista quantitativo profissional de futebol especializado em modelagem estatistica, leitura de mercado e identificacao de value bets (+EV), com abordagem semelhante a analistas de apostas institucionais.

Seu objetivo e analisar os jogos da data selecionada (__DATA_SELECIONADA__) e identificar as melhores oportunidades de aposta com base em dados historicos recentes, metricas avancadas e movimentacao das odds.

A analise deve priorizar confiabilidade estatistica, consistencia dos padroes e risco controlado.

--------------------------------------------------

ETAPA 1 - FILTRO DOS JOGOS

Selecione apenas os 5 jogos com maior confiabilidade estatistica na data analisada.

Se nao houver 5 jogos com evidencia estatistica suficiente, selecione menos jogos.

Nunca force previsoes.

--------------------------------------------------

ETAPA 2 - MODELAGEM ESTATISTICA

Para cada jogo, calcule probabilidades reais estimadas:

Vitoria mandante
Empate
Vitoria visitante
Over 2.5 gols
Under 2.5 gols
BTTS (ambos marcam)
Possivel placar provavel

Compare com odds implicitas do mercado quando disponiveis.

Identifique possiveis value bets (+EV).

Se nao houver value bet, informe claramente.

--------------------------------------------------

ETAPA 3 - FORMA RECENTE (ultimos 5 a 10 jogos)

Analise:

sequencia de vitorias/derrotas
media de gols marcados
media de gols sofridos
frequencia de Over 2.5
frequencia de BTTS
consistencia defensiva
variabilidade de desempenho

--------------------------------------------------

ETAPA 4 - CONFRONTOS DIRETOS (H2H)

Identifique:

tendencia de gols
dominio recorrente
frequencia historica de BTTS
frequencia Over/Under
padroes repetitivos relevantes

Priorize confrontos recentes.

--------------------------------------------------

ETAPA 5 - PERFORMANCE CASA vs FORA

Compare:

taxa de vitoria
media de gols
xG ofensivo
xGA defensivo
estabilidade tatica
diferenca de desempenho mandante vs visitante

--------------------------------------------------

ETAPA 6 - METRICAS AVANCADAS

Utilize:

xG
xGA
finalizacoes por jogo
conversao ofensiva
escanteios medios
cartoes medios
gols no segundo tempo
times que marcam primeiro
times que sofrem viradas

Identifique padroes ocultos relevantes.

--------------------------------------------------

ETAPA 7 - ESCALACOES E ELENCO

Considere:

lesoes
suspensoes
rotacao
retorno de titulares
dependencia de artilheiros
impacto tatico das ausencias

--------------------------------------------------

ETAPA 8 - MOVIMENTACAO DAS ODDS

Compare:

odd de abertura
odd atual
direcao do mercado
possivel entrada de dinheiro sharp
possivel influencia do publico

Identifique distorcoes relevantes.

--------------------------------------------------

ETAPA 9 - CONTEXTO COMPETITIVO

Avalie:

motivacao na tabela
briga por titulo
zona de rebaixamento
mata-mata vs rodada regular
necessidade de resultado
gestao de elenco

--------------------------------------------------

ETAPA 10 - FATORES EXTERNOS

Considere:

clima
condicoes do campo
viagem recente
fadiga
sequencia de jogos
arbitro (cartoes e penaltis)

--------------------------------------------------

ETAPA 11 - SCORE DE RISCO

Calcule score de risco de 0 a 100:

Forma recente -> 25%
Casa/Fora -> 15%
H2H -> 10%
xG e metricas avancadas -> 20%
Movimento das odds -> 20%
Volatilidade recente -> 10%

Classificacao:

0-39 = RISCO BAIXO
40-69 = RISCO MEDIO
70-100 = RISCO ALTO

--------------------------------------------------

ETAPA 12 - CONFIDENCE SCORE

Calcule indice de confianca de 0 a 10 baseado em:

consistencia estatistica
estabilidade das odds
convergencia dos indicadores
historico recente confiavel

--------------------------------------------------

ETAPA 13 - SAIDA POR JOGO

Para cada jogo apresentar:

JOGO: Time A vs Time B

Probabilidade vitoria mandante:
Probabilidade empate:
Probabilidade vitoria visitante:

Probabilidade Over 2.5:
Probabilidade BTTS:

Escanteios esperados:
Cartoes esperados:

Score de risco:
Classificacao final:

Confidence Score (0-10):

Value bet identificada:
(sim ou nao)

Se sim, explicar qual.

Fornecer justificativa objetiva baseada nos dados.

Apresentar:

1 aposta conservadora
1 aposta equilibrada
1 aposta agressiva

--------------------------------------------------

ETAPA 14 - GERENCIAMENTO DE BANCA

Sugira divisao:

Aposta principal
Aposta de cobertura
Aposta outsider

Utilize logica semelhante ao metodo Kelly simplificado.

Simule expectativa de retorno (%ROI estimado).

--------------------------------------------------

ETAPA 15 - CONCLUSAO FINAL

Ao final da analise:

Liste as 3 melhores apostas do dia com maior valor esperado (+EV)

Informe:

se e melhor apostar agora (early line)
ou esperar closing line

Se nao houver apostas com valor estatistico claro, informe explicitamente."""


def _current_app_timestamp() -> str:
    return datetime.now(APP_TIMEZONE).strftime("%d/%m/%Y %H:%M:%S")


def _fmt_odd(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2f}"


def _market_label(market: str, home_team: str, away_team: str) -> str:
    if market == "Casa":
        return f"Vitoria {home_team}"
    if market == "Fora":
        return f"Vitoria {away_team}"
    return "Empate"


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _get_detail_json(df: pd.DataFrame, row, probs, tip=None) -> str:
    home_ctx = get_team_context(df, str(row.home_team))
    away_ctx = get_team_context(df, str(row.away_team))

    data = {
        "home": str(row.home_team),
        "away": str(row.away_team),
        "date": str(row.date_text),
        "probs": {
            "home": round(probs.home_win * 100, 1),
            "draw": round(probs.draw * 100, 1),
            "away": round(probs.away_win * 100, 1),
            "btts": round(probs.btts_yes * 100, 1),
            "over25": round(probs.over_25 * 100, 1),
            "under25": round(probs.under_25 * 100, 1),
            "scorelines": [[s, round(p * 100, 1)] for s, p in probs.top_scorelines],
        },
        "odds": {
            "home": float(row.odds_home) if hasattr(row, "odds_home") else float(row.odd),
            "draw": float(row.odds_draw) if hasattr(row, "odds_draw") else 0.0,
            "away": float(row.odds_away) if hasattr(row, "odds_away") else 0.0,
        },
        "context": {"home": home_ctx, "away": away_ctx},
    }
    if tip:
        data["tip"] = {
            "market": _market_label(tip.best_market, str(row.home_team), str(row.away_team)),
            "odd": round(tip.best_odd, 2),
            "prob": round(tip.model_probability * 100, 1),
            "ev": round(tip.expected_value * 100, 2),
            "stake": round(tip.suggested_stake, 2),
        }
    return escape(json.dumps(data))


def _build_competition_section(name: str, df: pd.DataFrame) -> tuple[str, dict[str, int | str]]:
    finished = df[df["status"] == "Finalizado"].copy()
    fixtures = df[df["status"] == "Agendado"].copy()
    fixtures_valid = fixtures.dropna(subset=["odds_home", "odds_draw", "odds_away"])

    rows_html = []
    rec_rows = []

    for row in fixtures.head(40).itertuples(index=False):
        detail_json = "{}"
        try:
            probs = calculate_match_probabilities(df, row.home_team, row.away_team)
            detail_json = _get_detail_json(df, row, probs)
        except Exception:
            pass

        rows_html.append(
            "<tr "
            f"data-odd-home=\"{_fmt_odd(row.odds_home)}\" "
            f"data-odd-draw=\"{_fmt_odd(row.odds_draw)}\" "
            f"data-odd-away=\"{_fmt_odd(row.odds_away)}\" "
            f"data-details=\"{detail_json}\""
            ">"
            f"<td>{escape(str(row.date_text))}</td>"
            f"<td>{escape(str(row.home_team))}</td>"
            f"<td>{escape(str(row.away_team))}</td>"
            f"<td>{escape(_fmt_odd(row.odds_home))}</td>"
            f"<td>{escape(_fmt_odd(row.odds_draw))}</td>"
            f"<td>{escape(_fmt_odd(row.odds_away))}</td>"
            f"<td>{escape(str(row.bookmakers) if row.bookmakers is not None else '-')}</td>"
            "<td><button class='btn-mini' onclick='showMatchDetails(this)'>Visualizar</button></td>"
            "</tr>"
        )

    for row in fixtures_valid.head(20).itertuples(index=False):
        try:
            probs = calculate_match_probabilities(df, row.home_team, row.away_team)
            tip = suggest_bet_strategy(
                probs,
                odd_home=float(row.odds_home),
                odd_draw=float(row.odds_draw),
                odd_away=float(row.odds_away),
                bankroll=1000.0,
                kelly_fractional=0.25,
            )
            detail_json = _get_detail_json(df, row, probs, tip)
            rec_rows.append(
                f"<tr data-odd=\"{tip.best_odd:.2f}\" data-prob=\"{tip.model_probability:.4f}\" data-ev=\"{tip.expected_value:.4f}\" data-books=\"{int(row.bookmakers) if row.bookmakers is not None else 0}\" data-details=\"{detail_json}\">"
                f"<td>{escape(str(row.date_text))}</td>"
                f"<td>{escape(str(row.home_team))} x {escape(str(row.away_team))}</td>"
                f"<td>{escape(_market_label(tip.best_market, str(row.home_team), str(row.away_team)))}</td>"
                f"<td>{tip.best_odd:.2f}</td>"
                f"<td>{tip.model_probability * 100:.1f}%</td>"
                f"<td>{tip.expected_value * 100:.2f}%</td>"
                f"<td>R$ {tip.suggested_stake:.2f}</td>"
                "<td><button class='btn-mini' onclick='showMatchDetails(this)'>Visualizar</button></td>"
                "</tr>"
            )
        except Exception:
            continue

    safe_rows = []
    safe_df = build_safe_bets_table(
        matches_df=df,
        bankroll=1000.0,
        kelly_fractional=0.25,
        min_model_prob=0.55,
        min_expected_value=0.02,
        max_odd=2.20,
        min_bookmakers=8,
    )
    if not safe_df.empty:
        for row in safe_df.head(10).itertuples(index=False):
            try:
                probs = calculate_match_probabilities(df, row.home_team, row.away_team)
                # Find original odds for detail view
                match_orig = fixtures_valid[(fixtures_valid["home_team"] == row.home_team) & (fixtures_valid["away_team"] == row.away_team)].iloc[0]
                tip = suggest_bet_strategy(
                    probs,
                    odd_home=float(match_orig.odds_home),
                    odd_draw=float(match_orig.odds_draw),
                    odd_away=float(match_orig.odds_away),
                    bankroll=1000.0,
                    kelly_fractional=0.25,
                )
                detail_json = _get_detail_json(df, match_orig, probs, tip)
            except Exception:
                detail_json = "{}"

            safe_rows.append(
                f"<tr data-odd=\"{row.odd:.2f}\" data-prob=\"{row.model_probability:.4f}\" data-ev=\"{row.expected_value:.4f}\" data-books=\"{row.bookmakers}\" data-details=\"{detail_json}\">"
                f"<td>{escape(str(row.date_text))}</td>"
                f"<td>{escape(str(row.home_team))} x {escape(str(row.away_team))}</td>"
                f"<td>{escape(_market_label(str(row.market), str(row.home_team), str(row.away_team)))}</td>"
                f"<td>{row.odd:.2f}</td>"
                f"<td>{row.model_probability * 100:.1f}%</td>"
                f"<td>{row.expected_value * 100:.2f}%</td>"
                f"<td>{row.bookmakers}</td>"
                f"<td>R$ {row.stake:.2f}</td>"
                "<td><button class='btn-mini' onclick='showMatchDetails(this)'>Visualizar</button></td>"
                "</tr>"
            )

    comp_id = f"comp-{_slugify(name)}"
    stats = {
        "id": comp_id,
        "name": name,
        "finished": len(finished),
        "fixtures": len(fixtures),
        "fixtures_valid": len(fixtures_valid),
        "safe": len(safe_rows),
        "recommendations": len(rec_rows),
    }

    section_html = f"""
<section class="card competition-card" id="{comp_id}" data-comp="{escape(name)}">
  <div class="card-head">
    <div class="title-block">
      <div class="eyebrow">Competicao monitorada</div>
      <div class="title-row">
        <h2>{escape(name)}</h2>
        <div class="badge">Leitura organizada por seguranca, valor e agenda</div>
      </div>
    </div>
    <div class="stats-rail">
      <div class="stat-chip"><span>Finalizados</span><strong>{len(finished)}</strong></div>
      <div class="stat-chip"><span>Futuros</span><strong>{len(fixtures)}</strong></div>
      <div class="stat-chip"><span>Com odds</span><strong>{len(fixtures_valid)}</strong></div>
      <div class="stat-chip"><span>Top seguros</span><strong>{len(safe_rows)}</strong></div>
    </div>
  </div>

  <div class="panel panel-safe">
    <div class="panel-head">
      <div class="panel-title">
        <span class="panel-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none">
            <path d="M12 3l7 3v5c0 4.5-2.9 8.7-7 10-4.1-1.3-7-5.5-7-10V6l7-3z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>
            <path d="M9 12.2l2 2 4-4.2" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </span>
        <div>
        <div class="panel-kicker">Prioridade 1</div>
        <h3>Jogos mais seguros</h3>
        </div>
      </div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Data</th><th>Jogo</th><th>Resultado sugerido</th><th>Odd</th><th>Prob. Modelo</th><th>EV</th><th>Casas</th><th>Stake</th><th>Acao</th></tr></thead>
        <tbody>{''.join(safe_rows) if safe_rows else '<tr><td colspan="9">Nenhum jogo passou no filtro conservador.</td></tr>'}</tbody>
      </table>
    </div>
  </div>

  <div class="panel panel-value">
    <div class="panel-head">
      <div class="panel-title">
        <span class="panel-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none">
            <path d="M4 17l5-5 4 3 7-8" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M15 7h5v5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </span>
        <div>
        <div class="panel-kicker">Prioridade 2</div>
        <h3>Melhores entradas</h3>
        </div>
      </div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Data</th><th>Jogo</th><th>Resultado sugerido</th><th>Odd</th><th>Prob. Modelo</th><th>EV</th><th>Stake</th><th>Acao</th></tr></thead>
        <tbody>{''.join(rec_rows) if rec_rows else '<tr><td colspan="8">Sem recomendacoes disponiveis.</td></tr>'}</tbody>
      </table>
    </div>
  </div>

  <div class="panel panel-agenda">
    <div class="panel-head">
      <div class="panel-title">
        <span class="panel-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none">
            <rect x="4" y="5" width="16" height="15" rx="3" stroke="currentColor" stroke-width="1.8"/>
            <path d="M8 3v4M16 3v4M4 10h16" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
          </svg>
        </span>
        <div>
        <div class="panel-kicker">Prioridade 3</div>
        <h3>Agenda de jogos futuros</h3>
        </div>
      </div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Data</th><th>Mandante</th><th>Visitante</th><th>Odd Casa</th><th>Odd Empate</th><th>Odd Fora</th><th>Casas</th><th>Acao</th></tr></thead>
        <tbody>{''.join(rows_html) if rows_html else '<tr><td colspan="8">Sem jogos futuros.</td></tr>'}</tbody>
      </table>
    </div>
  </div>

  <p class="empty-state" hidden>Nenhuma linha desta competicao atende aos filtros atuais.</p>
</section>
"""
    return section_html, stats


def main() -> None:
    sections: list[str] = []
    competition_stats: list[dict[str, int | str]] = []

    for comp in COMPETITIONS:
        df = load_competition_matches(comp)
        section_html, stats = _build_competition_section(comp, df)
        sections.append(section_html)
        competition_stats.append(stats)

    total_finished = sum(int(item["finished"]) for item in competition_stats)
    total_fixtures = sum(int(item["fixtures"]) for item in competition_stats)
    total_odds = sum(int(item["fixtures_valid"]) for item in competition_stats)
    total_safe = sum(int(item["safe"]) for item in competition_stats)
    total_recommendations = sum(int(item["recommendations"]) for item in competition_stats)
    competition_count = len(competition_stats)
    odds_coverage = round((total_odds / total_fixtures) * 100) if total_fixtures else 0
    safe_rate = round((total_safe / total_odds) * 100) if total_odds else 0
    recommendations_rate = round((total_recommendations / total_odds) * 100) if total_odds else 0

    competition_options = "".join(f"<option>{escape(str(item['name']))}</option>" for item in competition_stats)
    competition_jump_links = "".join(
        f"<a href=\"#{escape(str(item['id']))}\" class=\"jump-link\">{escape(str(item['name']))}<span>{int(item['safe'])} seguros</span></a>"
        for item in competition_stats
    )
    side_league_cards = "".join(
        (
            f"<a href=\"#{escape(str(item['id']))}\" class=\"rail-link\">"
            f"<div class=\"rail-link-head\"><strong>{escape(str(item['name']))}</strong><span>{int(item['fixtures'])} jogos</span></div>"
            f"<div class=\"rail-link-meta\"><span>{int(item['safe'])} seguros</span><span>{int(item['fixtures_valid'])} odds</span></div>"
            f"<div class=\"rail-track\"><i style=\"width:{round((int(item['fixtures_valid']) / int(item['fixtures'])) * 100) if int(item['fixtures']) else 0}%\"></i></div>"
            "</a>"
        )
        for item in competition_stats
    )
    ai_prompt_html = escape(AI_PROMPT_TEMPLATE)
    ai_prompt_js = AI_PROMPT_TEMPLATE

    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Football Data Desk | Index Inicial</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    :root {{
      --bg: #eef3f8;
      --card: rgba(255,255,255,0.9);
      --line: rgba(148,163,184,0.28);
      --line-strong: rgba(148,163,184,0.42);
      --text: #112031;
      --muted: #5b6b7d;
      --blue: #1d4ed8;
      --teal: #0f766e;
      --amber: #d97706;
      --shadow: 0 22px 50px rgba(15,23,42,0.09);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif;
      color: var(--text);
      background:
        radial-gradient(900px 420px at 0% 0%, rgba(59,130,246,0.16), transparent 55%),
        radial-gradient(860px 420px at 100% 0%, rgba(16,185,129,0.14), transparent 50%),
        linear-gradient(180deg, #f8fafc 0%, var(--bg) 65%, #e9eef5 100%);
    }}
    .container {{ max-width: 1320px; margin: 0 auto; padding: 24px 20px 44px; }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 14px;
      margin-bottom: 16px;
      padding: 14px 16px;
      border-radius: 20px;
      background: rgba(255,255,255,.72);
      border: 1px solid rgba(148,163,184,.22);
      box-shadow: 0 16px 34px rgba(15,23,42,.05);
      backdrop-filter: blur(16px);
    }}
    .brand-block {{ display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
    .brand-mark {{
      width: 42px;
      height: 42px;
      border-radius: 14px;
      background: linear-gradient(135deg, #1d4ed8, #0f766e);
      color: #fff;
      display: grid;
      place-items: center;
      font-size: 15px;
      font-weight: 800;
      letter-spacing: .08em;
    }}
    .brand-copy strong {{ display: block; font-size: .96rem; letter-spacing: -.02em; }}
    .brand-copy span {{ display: block; margin-top: 3px; color: var(--muted); font-size: .83rem; }}
    .topbar-meta {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .meta-pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 9px 12px;
      border-radius: 999px;
      background: #f8fafc;
      border: 1px solid rgba(148,163,184,.25);
      color: #334155;
      font-size: .82rem;
      font-weight: 600;
    }}
    .meta-pill strong {{ color: #0f172a; }}
    .status-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #22c55e;
      box-shadow: 0 0 0 4px rgba(34,197,94,.14);
    }}
    .hero {{
      position: relative;
      overflow: hidden;
      padding: 28px;
      border-radius: 28px;
      color: #f8fafc;
      background: linear-gradient(135deg, #081625 0%, #14304f 54%, #0f766e 100%);
      box-shadow: 0 28px 70px rgba(8,22,37,0.26);
    }}
    .hero::before {{
      content: "";
      position: absolute;
      inset: 0;
      background:
        radial-gradient(circle at top right, rgba(96,165,250,.22), transparent 28%),
        linear-gradient(120deg, transparent 0%, rgba(255,255,255,.06) 100%);
      pointer-events: none;
    }}
    .hero-grid {{ display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(260px, 0.9fr); gap: 18px; align-items: end; }}
    .hero-grid, .metrics, .quick-nav {{ position: relative; z-index: 1; }}
    .hero-tag {{ display: inline-flex; padding: 8px 12px; border-radius: 999px; font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.16); color: #dbeafe; }}
    .hero h1 {{ margin: 14px 0 0; max-width: 12ch; font-size: clamp(2.2rem, 4vw, 3.6rem); line-height: .96; letter-spacing: -.04em; }}
    .hero p {{ margin: 14px 0 0; max-width: 64ch; color: rgba(226,232,240,.88); line-height: 1.68; }}
    .hero-stack {{ display: grid; gap: 12px; }}
    .hero-note {{ padding: 16px 18px; border-radius: 18px; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); }}
    .hero-note span {{ display: block; font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #bfdbfe; }}
    .hero-note strong {{ display: block; margin-top: 8px; font-size: 1.7rem; line-height: 1.05; }}
    .hero-note p {{ margin-top: 8px; font-size: .9rem; color: rgba(226,232,240,.82); }}
    .metrics {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-top: 20px; }}
    .metric {{ padding: 16px; border-radius: 18px; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); }}
    .metric span {{ display: block; font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: #cfe4ff; }}
    .metric strong {{ display: block; margin-top: 10px; font-size: 1.65rem; letter-spacing: -.04em; }}
    .metric p {{ margin: 8px 0 0; font-size: .86rem; color: rgba(226,232,240,.8); line-height: 1.45; }}
    .metric-track {{ margin-top: 12px; height: 7px; border-radius: 999px; background: rgba(255,255,255,.12); overflow: hidden; }}
    .metric-track i {{ display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #93c5fd, #6ee7b7); }}
    .quick-nav {{ margin-top: 16px; display: flex; flex-wrap: wrap; gap: 10px; }}
    .quick-nav::before {{ content: "Areas monitoradas"; width: 100%; margin-bottom: 2px; font-size: .76rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; color: #bfdbfe; }}
    .jump-link {{ display: inline-flex; align-items: center; gap: 10px; padding: 10px 14px; border-radius: 999px; background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.12); color: #fff; text-decoration: none; font-weight: 600; box-shadow: inset 0 1px 0 rgba(255,255,255,.08); }}
    .jump-link span {{ font-size: .76rem; letter-spacing: .05em; text-transform: uppercase; color: #bfdbfe; }}
    .dashboard-shell {{
      display: grid;
      grid-template-columns: minmax(250px, 280px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
      margin-top: 18px;
    }}
    .side-rail {{
      position: sticky;
      top: 18px;
      display: grid;
      gap: 14px;
    }}
    .rail-card {{
      padding: 18px;
      border-radius: 22px;
      background: linear-gradient(180deg, rgba(255,255,255,.96), rgba(246,249,252,.86));
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}
    .rail-card h3 {{
      margin: 0;
      font-size: 1rem;
    }}
    .rail-card p {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: .88rem;
      line-height: 1.55;
    }}
    .rail-list {{
      display: grid;
      gap: 10px;
      margin-top: 14px;
    }}
    .rail-link {{
      display: grid;
      gap: 8px;
      padding: 12px;
      border-radius: 16px;
      background: #fff;
      border: 1px solid rgba(148,163,184,.22);
      text-decoration: none;
      color: inherit;
    }}
    .rail-link-head, .rail-link-meta {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .rail-link-head strong {{
      font-size: .92rem;
      letter-spacing: -.02em;
    }}
    .rail-link-head span, .rail-link-meta span {{
      color: var(--muted);
      font-size: .78rem;
    }}
    .rail-track {{
      height: 7px;
      border-radius: 999px;
      background: #e5edf7;
      overflow: hidden;
    }}
    .rail-track i {{
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--blue), var(--teal));
    }}
    .spark-card {{
      display: grid;
      gap: 10px;
    }}
    .spark-head {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
    }}
    .spark-head strong {{
      font-size: .92rem;
    }}
    .spark-head span {{
      color: var(--muted);
      font-size: .8rem;
    }}
    .sparkline {{
      width: 100%;
      height: 72px;
      border-radius: 16px;
      background: linear-gradient(180deg, #f8fbff, #eef5fb);
      border: 1px solid rgba(148,163,184,.18);
      padding: 10px;
    }}
    .sparkline svg {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    .sparkline polyline {{
      fill: none;
      stroke: var(--blue);
      stroke-width: 2.2;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    .sparkline path {{
      fill: rgba(29,78,216,.08);
      stroke: none;
    }}
    .dashboard-main {{
      min-width: 0;
    }}
    .card {{ margin-top: 18px; padding: 20px; border-radius: 22px; background: var(--card); border: 1px solid var(--line); box-shadow: var(--shadow); backdrop-filter: blur(18px); }}
    .controls {{ display: grid; gap: 18px; }}
    .controls::before {{
      content: "";
      display: block;
      height: 4px;
      width: 120px;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--blue), var(--teal));
    }}
    .controls-head, .card-head, .panel-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; }}
    .eyebrow {{ display: inline-flex; margin-bottom: 8px; font-size: 12px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; color: var(--blue); }}
    h2 {{ margin: 0; font-size: clamp(1.35rem, 2vw, 1.85rem); letter-spacing: -.03em; }}
    h3 {{ margin: 4px 0 0; font-size: 1.06rem; letter-spacing: -.02em; }}
    .copy, .section-copy, .panel-copy {{ margin: 8px 0 0; color: var(--muted); line-height: 1.6; }}
    .summary-box {{ min-width: min(100%, 280px); padding: 14px 16px; border-radius: 18px; background: linear-gradient(135deg, #eff6ff, #ecfeff); border: 1px solid rgba(59,130,246,.14); color: #15314a; font-size: .92rem; line-height: 1.55; box-shadow: inset 0 1px 0 rgba(255,255,255,.55); }}
    .filters {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .field {{ display: grid; gap: 8px; padding: 14px; border-radius: 18px; background: linear-gradient(180deg, rgba(255,255,255,.96), rgba(241,245,249,.76)); border: 1px solid var(--line); }}
    .field label {{ font-size: 12px; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; color: #334155; }}
    .field input, .field select {{ height: 44px; border-radius: 12px; border: 1px solid var(--line-strong); padding: 0 14px; font: inherit; color: var(--text); background: rgba(255,255,255,.98); outline: none; }}
    .field input:focus, .field select:focus {{ border-color: rgba(29,78,216,.48); box-shadow: 0 0 0 4px rgba(29,78,216,.08); }}
    .hint {{ font-size: 12px; color: var(--muted); line-height: 1.45; }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .btn, .btn-link {{ height: 46px; padding: 0 16px; border-radius: 14px; border: 1px solid transparent; font: inherit; font-weight: 700; text-decoration: none; cursor: pointer; display: inline-flex; align-items: center; justify-content: center; transition: transform .18s ease, box-shadow .18s ease, background .18s ease; }}
    .btn.primary {{ background: linear-gradient(135deg, var(--blue), #2563eb); color: #fff; box-shadow: 0 16px 28px rgba(37,99,235,.22); }}
    .btn.secondary, .btn-link {{ background: #f8fafc; border-color: var(--line-strong); color: var(--text); }}
    .btn:hover, .btn-link:hover {{ transform: translateY(-1px); }}
    .btn-mini {{ padding: 4px 10px; font-size: 11px; font-weight: 700; border-radius: 8px; border: 1px solid var(--line-strong); background: #fff; cursor: pointer; }}
    .btn-mini:hover {{ background: var(--blue); color: #fff; border-color: var(--blue); }}
    .ai-module {{ margin-top: 18px; padding: 20px; border-radius: 22px; background: linear-gradient(135deg, rgba(8,22,37,.98), rgba(20,48,79,.98)); color: #f8fafc; border: 1px solid rgba(148,163,184,.16); box-shadow: 0 22px 50px rgba(8,22,37,.18); }}
    .ai-module .eyebrow {{ color: #bfdbfe; }}
    .ai-module h2 {{ color: #ffffff; }}
    .ai-module .copy {{ color: rgba(226,232,240,.84); }}
    .ai-module-grid {{ display: grid; grid-template-columns: minmax(250px, 320px) minmax(0, 1fr); gap: 16px; margin-top: 18px; }}
    .ai-module-side {{ display: grid; gap: 12px; }}
    .ai-mini-card {{ padding: 16px; border-radius: 18px; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.12); }}
    .ai-mini-card strong {{ display: block; font-size: .96rem; }}
    .ai-mini-card p {{ margin: 8px 0 0; color: rgba(226,232,240,.78); line-height: 1.55; }}
    .ai-mini-card label {{ display: block; margin-bottom: 8px; font-size: 12px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; color: #bfdbfe; }}
    .ai-mini-card input {{ width: 100%; height: 44px; border-radius: 12px; border: 1px solid rgba(191,219,254,.22); padding: 0 14px; font: inherit; color: #ffffff; background: rgba(255,255,255,.08); outline: none; }}
    .ai-mini-card input:focus {{ border-color: rgba(191,219,254,.48); box-shadow: 0 0 0 4px rgba(29,78,216,.12); }}
    .ai-prompt-shell {{ padding: 16px; border-radius: 18px; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.12); }}
    .ai-prompt-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }}
    .ai-prompt-head strong {{ font-size: 1rem; }}
    .ai-prompt-head span {{ color: #bfdbfe; font-size: .82rem; }}
    .ai-prompt-area {{ width: 100%; min-height: 620px; resize: vertical; border-radius: 16px; border: 1px solid rgba(191,219,254,.18); padding: 16px; font: 500 .92rem/1.6 "Plus Jakarta Sans", "Segoe UI", sans-serif; color: #e2e8f0; background: rgba(8,22,37,.72); outline: none; }}
    .ai-prompt-actions {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }}
    .ai-status {{ margin-top: 10px; color: #bbf7d0; font-size: .85rem; min-height: 1.2em; }}
    .ai-response {{ margin-top: 14px; padding: 16px; border-radius: 16px; background: rgba(8,22,37,.72); border: 1px solid rgba(191,219,254,.16); min-height: 140px; color: #e2e8f0; white-space: pre-wrap; line-height: 1.7; }}
    .legend {{ margin-top: 16px; display: flex; flex-wrap: wrap; gap: 10px; }}
    .pill {{ display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 999px; border: 1px solid var(--line); background: rgba(255,255,255,.82); font-size: .84rem; }}
    .pill::before {{ content: ""; width: 10px; height: 10px; border-radius: 50%; background: currentColor; opacity: .78; }}
    .competition-card {{
      position: relative;
      padding: 22px;
      overflow: hidden;
      background:
        linear-gradient(180deg, rgba(255,255,255,.95), rgba(255,255,255,.86)),
        radial-gradient(circle at top right, rgba(37,99,235,.06), transparent 26%);
    }}
    .competition-card::before {{
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      width: 5px;
      height: 100%;
      background: linear-gradient(180deg, var(--blue), var(--teal));
    }}
    .title-block {{ max-width: 760px; }}
    .title-row {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
    .badge {{ padding: 8px 12px; border-radius: 999px; background: linear-gradient(135deg, #ecfeff, #eff6ff); border: 1px solid rgba(15,118,110,.16); color: var(--teal); font-size: .8rem; font-weight: 700; }}
    .stats-rail {{ display: grid; grid-template-columns: repeat(2, minmax(120px, 1fr)); gap: 10px; min-width: min(100%, 320px); }}
    .stat-chip {{ padding: 14px; border-radius: 16px; background: linear-gradient(180deg, #fff, #f8fafc); border: 1px solid var(--line); }}
    .stat-chip span {{ display: block; font-size: 12px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }}
    .stat-chip strong {{ display: block; margin-top: 8px; font-size: 1.34rem; letter-spacing: -.04em; }}
    .panel {{ margin-top: 18px; padding: 16px; border-radius: 18px; background: linear-gradient(180deg, rgba(255,255,255,.98), rgba(248,250,252,.82)); border: 1px solid var(--line); box-shadow: inset 0 1px 0 rgba(255,255,255,.6); }}
    .panel-safe {{ border-color: rgba(16,185,129,.22); }}
    .panel-value {{ border-color: rgba(37,99,235,.22); }}
    .panel-agenda {{ border-color: rgba(245,158,11,.24); }}
    .panel-kicker {{ font-size: 12px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; color: var(--amber); }}
    .panel-safe .panel-kicker {{ color: #059669; }}
    .panel-value .panel-kicker {{ color: var(--blue); }}
    .panel-agenda .panel-kicker {{ color: var(--amber); }}
    .panel-title {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .panel-icon {{
      width: 42px;
      height: 42px;
      border-radius: 14px;
      display: grid;
      place-items: center;
      background: #f8fafc;
      border: 1px solid rgba(148,163,184,.22);
    }}
    .panel-icon svg {{
      width: 22px;
      height: 22px;
      display: block;
    }}
    .panel-safe .panel-icon {{ color: #059669; background: rgba(16,185,129,.08); border-color: rgba(16,185,129,.18); }}
    .panel-value .panel-icon {{ color: var(--blue); background: rgba(29,78,216,.08); border-color: rgba(29,78,216,.18); }}
    .panel-agenda .panel-icon {{ color: var(--amber); background: rgba(245,158,11,.10); border-color: rgba(245,158,11,.2); }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 16px; background: #fff; }}
    table {{ width: 100%; min-width: 780px; border-collapse: collapse; background: #fff; }}
    th, td {{ border-bottom: 1px solid #e8edf4; padding: 11px 12px; text-align: left; font-size: 13px; line-height: 1.5; }}
    th {{ position: sticky; top: 0; background: #eff6ff; color: #14324f; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; }}
    tbody tr:hover td {{ background: #f8fbff; }}
    tr.risk-low td {{ background: #ecfdf3; }}
    tr.risk-med td {{ background: #fff7e6; }}
    tr.risk-high td {{ background: #fff1f2; }}
    tr.risk-none td {{ background: #f8fafc; }}
    .empty-state {{ margin: 16px 0 2px; padding: 14px 16px; border-radius: 14px; background: #fff7ed; border: 1px solid rgba(245,158,11,.22); color: #9a3412; font-size: .92rem; }}
    
    /* MODAL STYLES */
    .modal-overlay {{
      position: fixed;
      inset: 0;
      background: rgba(8, 22, 37, 0.6);
      backdrop-filter: blur(8px);
      z-index: 2000;
      display: none;
      place-items: center;
      padding: 20px;
    }}
    .modal-card {{
      background: #fff;
      width: 100%;
      max-width: 960px;
      max-height: 90vh;
      border-radius: 28px;
      overflow-y: auto;
      box-shadow: 0 40px 100px rgba(0,0,0,0.3);
      position: relative;
      animation: modalShow 0.3s ease-out;
    }}
    @keyframes modalShow {{
      from {{ opacity: 0; transform: translateY(20px) scale(0.98); }}
      to {{ opacity: 1; transform: translateY(0) scale(1); }}
    }}
    .modal-head {{
      padding: 24px 32px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      background: rgba(255,255,255,0.9);
      backdrop-filter: blur(10px);
      z-index: 10;
    }}
    .modal-close {{
      width: 40px;
      height: 40px;
      border-radius: 50%;
      border: 1px solid var(--line-strong);
      background: #fff;
      display: grid;
      place-items: center;
      cursor: pointer;
      font-size: 20px;
      transition: all 0.2s;
    }}
    .modal-close:hover {{ background: #f1f5f9; transform: rotate(90deg); }}
    .modal-body {{ padding: 32px; }}
    .modal-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
    .modal-section {{ margin-bottom: 32px; }}
    .chart-container {{
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 20px;
      position: relative;
      height: 300px;
    }}
    .context-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }}
    .context-card {{ padding: 16px; border-radius: 16px; background: #f1f5f9; border: 1px solid var(--line); }}
    .context-card h4 {{ margin: 0 0 10px; font-size: 0.9rem; color: var(--blue); }}
    .context-stat {{ display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.85rem; }}
    .context-stat strong {{ color: var(--text); }}
    
    .glossary {{ display: grid; gap: 14px; }}
    .glossary-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .glossary-item {{ padding: 16px; border-radius: 18px; background: linear-gradient(180deg, rgba(255,255,255,.96), rgba(241,245,249,.76)); border: 1px solid var(--line); }}
    .glossary-item strong {{ display: block; margin-bottom: 6px; font-size: .95rem; }}
    .glossary-item p {{ margin: 0; color: var(--muted); line-height: 1.6; }}
    .foot {{ margin-top: 18px; color: var(--muted); font-size: .9rem; text-align: center; }}
    
    .floating-actions {{
      position: fixed;
      bottom: 24px;
      right: 24px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      z-index: 1000;
    }}
    .btn-float {{
      width: 52px;
      height: 52px;
      border-radius: 50%;
      background: var(--blue);
      color: #fff;
      display: grid;
      place-items: center;
      box-shadow: 0 12px 24px rgba(29,78,216,.3);
      cursor: pointer;
      border: none;
      transition: transform .2s ease;
    }}
    .btn-float:hover {{ transform: scale(1.1); }}
    .btn-float svg {{ width: 24px; height: 24px; fill: currentColor; }}
    
    @media (max-width: 1180px) {{
      .dashboard-shell {{ grid-template-columns: 1fr; }}
      .side-rail {{ position: static; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 1100px) {{ .hero-grid, .metrics, .filters, .glossary-grid, .ai-module-grid, .modal-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 760px) {{
      .container {{ padding: 18px 14px 32px; }}
      .topbar {{ align-items: flex-start; }}
      .hero, .card, .competition-card {{ padding: 18px; }}
      .hero-grid, .metrics, .filters, .stats-rail, .glossary-grid, .side-rail, .ai-module-grid, .modal-grid, .context-grid {{ grid-template-columns: 1fr; }}
      .topbar, .brand-block, .topbar-meta {{ flex-direction: column; align-items: flex-start; }}
      .btn, .btn-link {{ width: 100%; }}
      table {{ min-width: 640px; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <section class="topbar">
      <div class="brand-block">
        <div class="brand-mark">FD</div>
        <div class="brand-copy">
          <strong>Football Data Desk</strong>
          <span>Painel executivo para leitura rapida de oportunidades por competicao.</span>
        </div>
      </div>
      <div class="topbar-meta">
        <div class="meta-pill"><span class="status-dot"></span><strong>Atualizado</strong> {_current_app_timestamp()}</div>
        <div class="meta-pill"><strong>{competition_count}</strong> competicoes no radar</div>
        <div class="meta-pill"><strong>{odds_coverage}%</strong> cobertura de odds</div>
      </div>
    </section>

    <section class="hero">
      <div class="hero-grid">
        <div>
          <div class="hero-tag">Central quantitativa institucional</div>
          <h1>Radar premium de probabilidades, valor e protecao operacional</h1>
          <p>Gerado em {_current_app_timestamp()} no horario de Sao Paulo a partir de scraping de paginas publicas do BetExplorer. Esta capa foi redesenhada para funcionar como portal executivo de entrada, concentrando jogos seguros, comparativo modelo x mercado, agenda e historico de calibracao.</p>
        </div>
        <div class="hero-stack">
          <div class="hero-note">
            <span>Fluxo sugerido</span>
            <strong>Radar  Modelo  Hedge</strong>
            <p>Comece pelos blocos conservadores, valide o desempenho recente do modelo contra o mercado e depois avance para cenarios de fechamento e protecao.</p>
          </div>
          <div class="hero-note">
            <span>Painel inicial</span>
            <strong>{len(competition_stats)} competicoes</strong>
            <p>{total_fixtures} jogos futuros observados, {total_odds} linhas com odds completas, {total_safe} selecoes conservadoras e {total_finished} jogos finalizados para comparar a performance do modelo.</p>
          </div>
        </div>
      </div>

      <div class="metrics">
        <div class="metric"><span>Competicoes</span><strong>{competition_count}</strong><p>Torneios monitorados no painel.</p><div class="metric-track"><i style="width:100%"></i></div></div>
        <div class="metric"><span>Jogos futuros</span><strong>{total_fixtures}</strong><p>Partidas disponiveis para leitura.</p><div class="metric-track"><i style="width:100%"></i></div></div>
        <div class="metric"><span>Odds completas</span><strong>{total_odds}</strong><p>{odds_coverage}% dos jogos futuros com linha principal completa.</p><div class="metric-track"><i style="width:{odds_coverage}%"></i></div></div>
        <div class="metric"><span>Top seguros</span><strong>{total_safe}</strong><p>{safe_rate}% das linhas com odds entram no filtro conservador.</p><div class="metric-track"><i style="width:{safe_rate}%"></i></div></div>
        <div class="metric"><span>Entradas EV</span><strong>{total_recommendations}</strong><p>{recommendations_rate}% das linhas com odds geram sugestao de stake.</p><div class="metric-track"><i style="width:{recommendations_rate}%"></i></div></div>
      </div>

      <div class="quick-nav">{competition_jump_links}</div>
    </section>

    <div class="dashboard-shell">
      <aside class="side-rail">
        <section class="rail-card">
          <div class="eyebrow">Radar lateral</div>
          <h3>Navegacao rapida</h3>
          <p>Use esta lateral para saltar entre competicoes e acompanhar a cobertura de odds sem perder o contexto.</p>
          <div class="rail-list">{side_league_cards}</div>
        </section>
        <section class="rail-card spark-card">
          <div class="eyebrow">Leitura expressa</div>
          <div class="spark-head"><strong>Cobertura de mercado</strong><span>{odds_coverage}%</span></div>
          <div class="sparkline" aria-hidden="true">
            <svg viewBox="0 0 160 70" preserveAspectRatio="none">
              <path d="M0 70 L0 50 L32 48 L64 44 L96 40 L128 28 L160 20 L160 70 Z"></path>
              <polyline points="0,50 32,48 64,44 96,40 128,28 160,20"></polyline>
            </svg>
          </div>
          <div class="spark-head"><strong>Filtro conservador</strong><span>{safe_rate}%</span></div>
          <div class="sparkline" aria-hidden="true">
            <svg viewBox="0 0 160 70" preserveAspectRatio="none">
              <path d="M0 70 L0 58 L32 54 L64 48 L96 46 L128 42 L160 36 L160 70 Z"></path>
              <polyline points="0,58 32,54 64,48 96,46 128,42 160,36"></polyline>
            </svg>
          </div>
          <div class="spark-head"><strong>Entradas com EV</strong><span>{recommendations_rate}%</span></div>
          <div class="sparkline" aria-hidden="true">
            <svg viewBox="0 0 160 70" preserveAspectRatio="none">
              <path d="M0 70 L0 56 L32 46 L64 40 L96 30 L128 18 L160 10 L160 70 Z"></path>
              <polyline points="0,56 32,46 64,40 96,30 128,18 160,10"></polyline>
            </svg>
          </div>
        </section>
      </aside>

      <main class="dashboard-main">
        <section class="controls card">
          <div class="controls-head">
            <div>
              <div class="eyebrow">Filtros inteligentes</div>
              <h2>Refine o painel sem perder contexto</h2>
              <p class="copy">Os filtros abaixo ajudam a reduzir ruido. O preset de risco preenche uma faixa inicial, mas voce pode personalizar os campos para buscar um perfil mais agressivo ou mais conservador.</p>
            </div>
            <div id="resultsSummary" class="summary-box">Mostrando todas as competicoes e tabelas disponiveis.</div>
          </div>

          <div class="filters">
            <div class="field"><label for="fcomp">Competicao</label><select id="fcomp"><option value="">Todas</option>{competition_options}</select><div class="hint">Filtra o painel para um campeonato especifico.</div></div>
            <div class="field"><label for="frisk">Perfil de risco</label><select id="frisk"><option>Baixo risco</option><option>Medio risco</option><option>Alto risco</option><option>Personalizado</option></select><div class="hint">Aplica faixas padrao para odd, probabilidade, EV e casas.</div></div>
            <div class="field"><label for="fteam">Time</label><input id="fteam" type="text" placeholder="Ex: Flamengo" /><div class="hint">Busca o nome do time em qualquer tabela visivel.</div></div>
            <div class="field"><label for="fbooks">Casas minimas</label><input id="fbooks" type="number" step="1" min="0" placeholder="8" /><div class="hint">Evita linhas com baixa cobertura de bookmakers.</div></div>
            <div class="field"><label for="foddmin">Odd minima</label><input id="foddmin" type="number" step="0.01" min="1.01" placeholder="1.30" /><div class="hint">Define a base minima da faixa de odd.</div></div>
            <div class="field"><label for="foddmax">Odd maxima</label><input id="foddmax" type="number" step="0.01" min="1.01" placeholder="2.20" /><div class="hint">Limita selecoes acima de uma odd alvo.</div></div>
            <div class="field"><label for="fprobmin">Probabilidade minima</label><input id="fprobmin" type="number" step="0.01" min="0" max="1" placeholder="0.55" /><div class="hint">Usada apenas nas tabelas com leitura do modelo.</div></div>
            <div class="field"><label for="fevmin">EV minimo</label><input id="fevmin" type="number" step="0.005" min="0" max="1" placeholder="0.02" /><div class="hint">Mostra entradas com vantagem esperada minima.</div></div>
          </div>

          <div class="actions">
            <button id="applyFilter" class="btn primary" type="button">Aplicar filtro</button>
            <button id="clearFilter" class="btn secondary" type="button">Limpar filtros</button>
            <a href="./atualizar_painel.bat" class="btn-link">Atualizar dados</a>
            <button id="reloadPage" class="btn secondary" type="button">Recarregar pagina</button>
          </div>
        </section>

        <section class="ai-module">
          <div>
            <div class="eyebrow">Modulo de IA</div>
            <h2>Prompt institucional para leitura quantitativa</h2>
            <p class="copy">Use este modulo para montar um briefing completo e enviar para a IA de sua preferencia. A data selecionada entra no prompt automaticamente para facilitar a leitura do dia.</p>
          </div>
          <div class="ai-module-grid">
            <div class="ai-module-side">
              <div class="ai-mini-card">
                <label for="aiSelectedDate">Data da analise</label>
                <input id="aiSelectedDate" type="date" />
                <p>Escolha a data que deseja analisar antes de copiar o prompt para a IA.</p>
              </div>
              <div class="ai-mini-card">
                <strong>Como usar</strong>
                <p>1. Selecione a data. 2. Copie o prompt completo. 3. Cole na IA para gerar a leitura institucional dos jogos daquele dia.</p>
              </div>
              <div class="ai-mini-card">
                <strong>Objetivo do modulo</strong>
                <p>Padronizar a leitura estatistica com foco em value bet, score de risco, confidence score e distribuicao de banca.</p>
              </div>
            </div>
            <div class="ai-prompt-shell">
              <div class="ai-prompt-head">
                <strong>Prompt pronto para a IA</strong>
                <span>Data atualizada automaticamente</span>
              </div>
              <textarea id="aiPromptArea" class="ai-prompt-area">{ai_prompt_html}</textarea>
              <div class="ai-prompt-actions">
                <button id="updateAiPrompt" class="btn primary" type="button">Atualizar prompt</button>
                <button id="copyAiPrompt" class="btn secondary" type="button">Copiar prompt</button>
                <button id="runAiPrompt" class="btn primary" type="button">Executar leitura com IA</button>
              </div>
              <div id="aiPromptStatus" class="ai-status"></div>
              <div id="aiResponse" class="ai-response">A resposta da IA vai aparecer aqui depois da execucao.</div>
            </div>
          </div>
        </section>

        <div class="legend">
          <span class="pill" style="color:#15803d;">Baixo risco</span>
          <span class="pill" style="color:#b45309;">Medio risco</span>
          <span class="pill" style="color:#be123c;">Alto risco</span>
          <span class="pill" style="color:#64748b;">Fora dos criterios</span>
        </div>

        {''.join(sections)}

        <section class="glossary card">
          <div>
            <div class="eyebrow">Guia rapido</div>
            <h2>Glossario de siglas e metricas</h2>
          </div>
          <div class="glossary-grid">
            <div class="glossary-item"><strong>1X2</strong><p>Mercado principal do jogo. 1 = vitoria mandante, X = empate, 2 = vitoria visitante.</p></div>
            <div class="glossary-item"><strong>Odd</strong><p>Cotacao da aposta. Exemplo: odd 2.00 retorna R$ 2,00 para cada R$ 1,00 apostado em retorno bruto.</p></div>
            <div class="glossary-item"><strong>Prob. Modelo</strong><p>Probabilidade estimada pelo modelo para o resultado sugerido na linha.</p></div>
            <div class="glossary-item"><strong>EV</strong><p>Valor esperado da aposta. EV positivo indica vantagem matematica no longo prazo.</p></div>
            <div class="glossary-item"><strong>Casas</strong><p>Quantidade de casas de apostas consideradas na linha de odd usada na comparacao.</p></div>
            <div class="glossary-item"><strong>Stake</strong><p>Valor sugerido para a aposta com base em banca e Kelly fracionado.</p></div>
          </div>
        </section>

        <div class="foot">Aposta envolve risco. O painel ajuda na leitura e comparacao, mas nao elimina perdas nem substitui gestao de banca.</div>
      </main>
    </div>
  </div>

  <div id="matchModal" class="modal-overlay">
    <div class="modal-card">
      <div class="modal-head">
        <div>
          <div id="modalDate" class="eyebrow">00/00/0000</div>
          <h2 id="modalTitle">Time A vs Time B</h2>
        </div>
        <button class="modal-close" onclick="closeMatchDetails()">×</button>
      </div>
      <div class="modal-body">
        <div class="modal-grid">
          <div class="modal-section">
            <h3>Probabilidades 1X2</h3>
            <div class="chart-container"><canvas id="chart1x2"></canvas></div>
          </div>
          <div class="modal-section">
            <h3>Mercados Alternativos</h3>
            <div class="chart-container"><canvas id="chartAlt"></canvas></div>
          </div>
        </div>
        
        <div class="modal-grid">
          <div class="modal-section">
            <h3>Top 5 Placares Provaveis</h3>
            <div class="chart-container"><canvas id="chartScores"></canvas></div>
          </div>
          <div class="modal-section">
            <h3>Estrategia e Contexto</h3>
            <div id="strategyBox" class="summary-box" style="margin-bottom:16px;"></div>
            <div class="context-grid">
              <div class="context-card">
                <h4 id="homeTeamName">Mandante</h4>
                <div class="context-stat">Posicao: <strong id="homeRank">-</strong></div>
                <div class="context-stat">Pontos: <strong id="homePoints">-</strong></div>
                <div class="context-stat">Forma: <strong id="homeForm">-</strong></div>
              </div>
              <div class="context-card">
                <h4 id="awayTeamName">Visitante</h4>
                <div class="context-stat">Posicao: <strong id="awayRank">-</strong></div>
                <div class="context-stat">Pontos: <strong id="awayPoints">-</strong></div>
                <div class="context-stat">Forma: <strong id="awayForm">-</strong></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="floating-actions">
    <button id="scrollToTop" class="btn-float" title="Voltar ao topo" style="display:none;">
      <svg viewBox="0 0 24 24"><path d="M12 4l-8 8h16l-8-8z"/></svg>
    </button>
    <button id="quickRefresh" class="btn-float" title="Recarregar dados" onclick="window.location.reload();">
      <svg viewBox="0 0 24 24"><path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg>
    </button>
  </div>

  <script>
    const scrollBtn = document.getElementById('scrollToTop');
    window.onscroll = function() {{
      if (document.body.scrollTop > 300 || document.documentElement.scrollTop > 300) {{
        scrollBtn.style.display = "grid";
      }} else {{
        scrollBtn.style.display = "none";
      }}
    }};
    scrollBtn.onclick = function() {{
      window.scrollTo({{ top: 0, behavior: 'smooth' }});
    }};

    const riskPresets = {{
      "Baixo risco": {{ oddMin: 1.20, oddMax: 1.95, probMin: 0.62, evMin: 0.03, booksMin: 10 }},
      "Medio risco": {{ oddMin: 1.20, oddMax: 2.20, probMin: 0.55, evMin: 0.02, booksMin: 8 }},
      "Alto risco": {{ oddMin: 1.20, oddMax: 2.90, probMin: 0.48, evMin: 0.01, booksMin: 5 }},
    }};

    let charts = {{}};

    function showMatchDetails(btn) {{
      const row = btn.closest('tr');
      const rawData = row.getAttribute('data-details');
      if (!rawData || rawData === '{{}}') return;
      const data = JSON.parse(rawData);
      
      document.getElementById('modalTitle').textContent = data.home + ' x ' + data.away;
      document.getElementById('modalDate').textContent = data.date;
      
      const stratBox = document.getElementById('strategyBox');
      if (data.tip) {{
        stratBox.innerHTML = `<strong>Sugestao:</strong> ${{data.tip.market}}<br>
          <strong>Odd:</strong> ${{data.tip.odd.toFixed(2)}} | <strong>Prob:</strong> ${{data.tip.prob}}% | <strong>EV:</strong> ${{data.tip.ev}}%<br>
          <strong>Stake Sugerida:</strong> R$ ${{data.tip.stake.toFixed(2)}}`;
      }} else {{
        stratBox.innerHTML = 'Sem recomendacao de entrada para este jogo com base nos criterios do modelo.';
      }}
      
      document.getElementById('homeTeamName').textContent = data.home;
      document.getElementById('homeRank').textContent = data.context.home.rank || '-';
      document.getElementById('homePoints').textContent = data.context.home.points;
      document.getElementById('homeForm').textContent = data.context.home.recent_text;
      
      document.getElementById('awayTeamName').textContent = data.away;
      document.getElementById('awayRank').textContent = data.context.away.rank || '-';
      document.getElementById('awayPoints').textContent = data.context.away.points;
      document.getElementById('awayForm').textContent = data.context.away.recent_text;
      
      renderCharts(data);
      document.getElementById('matchModal').style.display = 'grid';
    }}

    function closeMatchDetails() {{
      document.getElementById('matchModal').style.display = 'none';
    }}

    function renderCharts(data) {{
      if (charts.c1x2) charts.chart1x2.destroy();
      if (charts.cAlt) charts.chartAlt.destroy();
      if (charts.cScores) charts.chartScores.destroy();
      
      const ctx1 = document.getElementById('chart1x2').getContext('2d');
      charts.chart1x2 = new Chart(ctx1, {{
        type: 'bar',
        data: {{
          labels: ['Mandante', 'Empate', 'Visitante'],
          datasets: [{{
            label: 'Modelo %',
            data: [data.probs.home, data.probs.draw, data.probs.away],
            backgroundColor: ['rgba(29, 78, 216, 0.7)', 'rgba(148, 163, 184, 0.7)', 'rgba(15, 118, 110, 0.7)'],
            borderRadius: 8
          }}, {{
            label: 'Mercado %',
            data: [
              data.odds.home > 0 ? (100/data.odds.home).toFixed(1) : 0,
              data.odds.draw > 0 ? (100/data.odds.draw).toFixed(1) : 0,
              data.odds.away > 0 ? (100/data.odds.away).toFixed(1) : 0
            ],
            backgroundColor: 'rgba(0,0,0,0.1)',
            borderRadius: 8
          }}]
        }},
        options: {{ responsive: true, maintainAspectRatio: false }}
      }});
      
      const ctx2 = document.getElementById('chartAlt').getContext('2d');
      charts.chartAlt = new Chart(ctx2, {{
        type: 'doughnut',
        data: {{
          labels: ['BTTS Sim', 'Over 2.5', 'Under 2.5'],
          datasets: [{{
            data: [data.probs.btts, data.probs.over25, data.probs.under25],
            backgroundColor: ['#fbbf24', '#ef4444', '#10b981']
          }}]
        }},
        options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ position: 'bottom' }} }} }}
      }});
      
      const ctx3 = document.getElementById('chartScores').getContext('2d');
      charts.chartScores = new Chart(ctx3, {{
        type: 'bar',
        data: {{
          labels: data.probs.scorelines.map(s => s[0]),
          datasets: [{{
            label: 'Probabilidade %',
            data: data.probs.scorelines.map(s => s[1]),
            backgroundColor: 'rgba(29, 78, 216, 0.6)',
            borderRadius: 6
          }}]
        }},
        options: {{ indexAxis: 'y', responsive: true, maintainAspectRatio: false }}
      }});
    }}

    function applyRiskPreset() {{
      const risk = document.getElementById('frisk').value;
      if (risk === 'Personalizado') return;
      const preset = riskPresets[risk];
      if (!preset) return;
      document.getElementById('foddmin').value = preset.oddMin.toFixed(2);
      document.getElementById('foddmax').value = preset.oddMax.toFixed(2);
      document.getElementById('fprobmin').value = preset.probMin.toFixed(2);
      document.getElementById('fevmin').value = preset.evMin.toFixed(3);
      document.getElementById('fbooks').value = String(preset.booksMin);
    }}

    function classifyRowsByRisk() {{
      const rows = Array.from(document.querySelectorAll('tr[data-odd]'));
      rows.forEach(row => {{
        row.classList.remove('risk-low', 'risk-med', 'risk-high', 'risk-none');
        const odd = parseFloat(row.getAttribute('data-odd') || '');
        const prob = parseFloat(row.getAttribute('data-prob') || '');
        const ev = parseFloat(row.getAttribute('data-ev') || '');
        const books = parseInt(row.getAttribute('data-books') || '', 10);
        if (Number.isNaN(odd) || Number.isNaN(prob) || Number.isNaN(ev) || Number.isNaN(books)) {{
          row.classList.add('risk-none');
          return;
        }}
        if (prob >= 0.62 && ev >= 0.03 && odd <= 1.95 && books >= 10) row.classList.add('risk-low');
        else if (prob >= 0.55 && ev >= 0.02 && odd <= 2.20 && books >= 8) row.classList.add('risk-med');
        else if (prob >= 0.48 && ev >= 0.01 && odd <= 2.90 && books >= 5) row.classList.add('risk-high');
        else row.classList.add('risk-none');
      }});
    }}

    function updateResultsSummary(shownCards, visibleRows) {{
      const competition = document.getElementById('fcomp').value || 'todas as competicoes';
      const risk = document.getElementById('frisk').value;
      const team = (document.getElementById('fteam').value || '').trim();
      const summary = document.getElementById('resultsSummary');
      const parts = [
        shownCards + (shownCards === 1 ? ' competicao visivel' : ' competicoes visiveis'),
        visibleRows + (visibleRows === 1 ? ' linha encontrada' : ' linhas encontradas'),
        'perfil ' + risk.toLowerCase(),
      ];
      if (team) parts.push('busca por "' + team + '"');
      summary.textContent = 'Mostrando ' + competition + ': ' + parts.join(' • ') + '.';
    }}

    const aiPromptTemplate = {ai_prompt_js!r};

    function formatSelectedDate(value) {{
      if (!value) return 'data nao definida';
      const parts = value.split('-');
      if (parts.length !== 3) return value;
      return parts[2] + '/' + parts[1] + '/' + parts[0];
    }}

    function updateAiPrompt() {{
      const dateValue = document.getElementById('aiSelectedDate').value;
      const promptArea = document.getElementById('aiPromptArea');
      const prompt = aiPromptTemplate.replace('__DATA_SELECIONADA__', formatSelectedDate(dateValue));
      promptArea.value = prompt;
      document.getElementById('aiPromptStatus').textContent = 'Prompt atualizado para a data selecionada.';
    }}

    async function copyAiPrompt() {{
      const promptArea = document.getElementById('aiPromptArea');
      const status = document.getElementById('aiPromptStatus');
      promptArea.select();
      promptArea.setSelectionRange(0, promptArea.value.length);
      try {{
        await navigator.clipboard.writeText(promptArea.value);
        status.textContent = 'Prompt copiado com sucesso.';
      }} catch (error) {{
        document.execCommand('copy');
        status.textContent = 'Prompt copiado.';
      }}
    }}

    async function runAiPrompt() {{
      const selectedDate = document.getElementById('aiSelectedDate').value;
      const promptArea = document.getElementById('aiPromptArea');
      const status = document.getElementById('aiPromptStatus');
      const responseBox = document.getElementById('aiResponse');
      const apiHost = window.location.hostname ? window.location.hostname : '127.0.0.1';
      if (!selectedDate) {{
        status.textContent = 'Selecione uma data antes de executar a leitura.';
        return;
      }}

      status.textContent = 'Consultando a IA local...';
      responseBox.textContent = 'Processando leitura quantitativa, aguarde...';
      try {{
        const response = await fetch('http://' + apiHost + ':8765/api/ai-analysis', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            selected_date: selectedDate,
            prompt: promptArea.value
          }})
        }});
        const data = await response.json();
        if (!response.ok || !data.ok) {{
          throw new Error(data.error || 'Falha ao consultar a IA.');
        }}
        responseBox.textContent = data.analysis || 'A IA nao retornou texto.';
        status.textContent = 'Leitura concluida com sucesso.';
      }} catch (error) {{
        responseBox.textContent = 'Nao foi possivel gerar a leitura agora.';
        status.textContent = 'Erro: ' + (error && error.message ? error.message : 'falha desconhecida.');
      }}
    }}

    function applyFilters() {{
      const comp = (document.getElementById('fcomp').value || '').toLowerCase();
      const team = (document.getElementById('fteam').value || '').toLowerCase().trim();
      const oddMinRaw = document.getElementById('foddmin').value;
      const oddMaxRaw = document.getElementById('foddmax').value;
      const probMinRaw = document.getElementById('fprobmin').value;
      const evMinRaw = document.getElementById('fevmin').value;
      const booksMinRaw = document.getElementById('fbooks').value;
      const oddMin = oddMinRaw ? parseFloat(oddMinRaw) : null;
      const oddMax = oddMaxRaw ? parseFloat(oddMaxRaw) : null;
      const probMin = probMinRaw ? parseFloat(probMinRaw) : null;
      const evMin = evMinRaw ? parseFloat(evMinRaw) : null;
      const booksMin = booksMinRaw ? parseInt(booksMinRaw, 10) : null;
      const cards = Array.from(document.querySelectorAll('.card[data-comp]'));
      let shownCards = 0;
      let visibleRows = 0;

      cards.forEach(card => {{
        const cardComp = (card.getAttribute('data-comp') || '').toLowerCase();
        const showCard = !comp || cardComp === comp;
        const emptyState = card.querySelector('.empty-state');
        card.style.display = showCard ? '' : 'none';
        if (!showCard) {{
          if (emptyState) emptyState.hidden = true;
          return;
        }}

        shownCards += 1;
        const rows = Array.from(card.querySelectorAll('tbody tr'));
        rows.forEach(row => {{
          const txt = (row.textContent || '').toLowerCase();
          const teamOk = !team || txt.includes(team);
          let oddOk = true;
          const oneOdd = row.getAttribute('data-odd');
          if (oneOdd) {{
            const value = parseFloat(oneOdd);
            if (!Number.isNaN(value)) {{
              if (oddMin !== null && value < oddMin) oddOk = false;
              if (oddMax !== null && value > oddMax) oddOk = false;
            }}
            const rowProb = parseFloat(row.getAttribute('data-prob') || '');
            const rowEv = parseFloat(row.getAttribute('data-ev') || '');
            const rowBooks = parseInt(row.getAttribute('data-books') || '', 10);
            if (probMin !== null && !Number.isNaN(rowProb) && rowProb < probMin) oddOk = false;
            if (evMin !== null && !Number.isNaN(rowEv) && rowEv < evMin) oddOk = false;
            if (booksMin !== null && !Number.isNaN(rowBooks) && rowBooks < booksMin) oddOk = false;
          }} else {{
            const odds = ['data-odd-home', 'data-odd-draw', 'data-odd-away']
              .map(attr => parseFloat(row.getAttribute(attr) || ''))
              .filter(value => !Number.isNaN(value));
            if (odds.length > 0) {{
              const minValue = Math.min(...odds);
              const maxValue = Math.max(...odds);
              if (oddMin !== null && maxValue < oddMin) oddOk = false;
              if (oddMax !== null && minValue > oddMax) oddOk = false;
            }}
          }}
          row.style.display = teamOk && oddOk ? '' : 'none';
        }});

        const cardVisibleRows = rows.filter(row => row.style.display !== 'none').length;
        if (emptyState) emptyState.hidden = cardVisibleRows !== 0;
        visibleRows += cardVisibleRows;
      }});

      classifyRowsByRisk();
      updateResultsSummary(shownCards, visibleRows);
    }}

    document.getElementById('applyFilter').addEventListener('click', applyFilters);
    document.getElementById('fcomp').addEventListener('change', applyFilters);
    document.getElementById('fteam').addEventListener('input', applyFilters);
    document.getElementById('frisk').addEventListener('change', () => {{ applyRiskPreset(); applyFilters(); }});
    document.getElementById('clearFilter').addEventListener('click', () => {{
      document.getElementById('fcomp').value = '';
      document.getElementById('frisk').value = 'Baixo risco';
      document.getElementById('fteam').value = '';
      document.getElementById('foddmin').value = '';
      document.getElementById('foddmax').value = '';
      document.getElementById('fprobmin').value = '';
      document.getElementById('fevmin').value = '';
      document.getElementById('fbooks').value = '';
      applyRiskPreset();
      applyFilters();
    }});
    document.getElementById('reloadPage').addEventListener('click', () => {{ window.location.reload(); }});
    document.getElementById('updateAiPrompt').addEventListener('click', updateAiPrompt);
    document.getElementById('copyAiPrompt').addEventListener('click', copyAiPrompt);
    document.getElementById('runAiPrompt').addEventListener('click', runAiPrompt);
    document.getElementById('aiSelectedDate').addEventListener('change', updateAiPrompt);

    const today = new Date();
    document.getElementById('aiSelectedDate').value = today.toISOString().slice(0, 10);
    applyRiskPreset();
    applyFilters();
    classifyRowsByRisk();
    updateAiPrompt();
  </script>
</body>
</html>
"""

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
