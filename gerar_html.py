from __future__ import annotations

from datetime import datetime
from html import escape

import pandas as pd

from analytics import build_safe_bets_table, calculate_match_probabilities, suggest_bet_strategy
from scraper import COMPETITIONS, load_competition_matches


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


def _build_competition_section(name: str, df: pd.DataFrame) -> str:
    finished = df[df["status"] == "Finalizado"].copy()
    fixtures = df[df["status"] == "Agendado"].copy()
    fixtures_valid = fixtures.dropna(subset=["odds_home", "odds_draw", "odds_away"])

    rows_html = []
    rec_rows = []

    for row in fixtures.head(40).itertuples(index=False):
        rows_html.append(
            "<tr "
            f"data-odd-home=\"{_fmt_odd(row.odds_home)}\" "
            f"data-odd-draw=\"{_fmt_odd(row.odds_draw)}\" "
            f"data-odd-away=\"{_fmt_odd(row.odds_away)}\""
            ">"
            f"<td>{escape(str(row.date_text))}</td>"
            f"<td>{escape(str(row.home_team))}</td>"
            f"<td>{escape(str(row.away_team))}</td>"
            f"<td>{escape(_fmt_odd(row.odds_home))}</td>"
            f"<td>{escape(_fmt_odd(row.odds_draw))}</td>"
            f"<td>{escape(_fmt_odd(row.odds_away))}</td>"
            f"<td>{escape(str(row.bookmakers) if row.bookmakers is not None else '-')}</td>"
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
            rec_rows.append(
                f"<tr data-odd=\"{tip.best_odd:.2f}\" data-prob=\"{tip.model_probability:.4f}\" data-ev=\"{tip.expected_value:.4f}\" data-books=\"{int(row.bookmakers) if row.bookmakers is not None else 0}\">"
                f"<td>{escape(str(row.date_text))}</td>"
                f"<td>{escape(str(row.home_team))} x {escape(str(row.away_team))}</td>"
                f"<td>{escape(_market_label(tip.best_market, str(row.home_team), str(row.away_team)))}</td>"
                f"<td>{tip.best_odd:.2f}</td>"
                f"<td>{tip.model_probability * 100:.1f}%</td>"
                f"<td>{tip.expected_value * 100:.2f}%</td>"
                f"<td>R$ {tip.suggested_stake:.2f}</td>"
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
            safe_rows.append(
                f"<tr data-odd=\"{row.odd:.2f}\" data-prob=\"{row.model_probability:.4f}\" data-ev=\"{row.expected_value:.4f}\" data-books=\"{row.bookmakers}\">"
                f"<td>{escape(str(row.date_text))}</td>"
                f"<td>{escape(str(row.home_team))} x {escape(str(row.away_team))}</td>"
                f"<td>{escape(_market_label(str(row.market), str(row.home_team), str(row.away_team)))}</td>"
                f"<td>{row.odd:.2f}</td>"
                f"<td>{row.model_probability * 100:.1f}%</td>"
                f"<td>{row.expected_value * 100:.2f}%</td>"
                f"<td>{row.bookmakers}</td>"
                f"<td>R$ {row.stake:.2f}</td>"
                "</tr>"
            )

    return f"""
<section class="card" data-comp="{escape(name)}">
  <div class="card-head">
    <h2>{escape(name)}</h2>
    <div class="badge">Finalizados {len(finished)} • Futuros {len(fixtures)} • Odds {len(fixtures_valid)}</div>
  </div>

  <h3>Jogos Mais Seguros (filtro conservador)</h3>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Data</th><th>Jogo</th><th>Resultado sugerido</th><th>Odd</th><th>Prob. Modelo</th><th>EV</th><th>Casas</th><th>Stake</th></tr></thead>
      <tbody>{''.join(safe_rows) if safe_rows else '<tr><td colspan="8">Nenhum jogo passou no filtro conservador.</td></tr>'}</tbody>
    </table>
  </div>

  <h3>Melhores Entradas (EV/Kelly, banca R$1000)</h3>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Data</th><th>Jogo</th><th>Resultado sugerido</th><th>Odd</th><th>Prob. Modelo</th><th>EV</th><th>Stake</th></tr></thead>
      <tbody>{''.join(rec_rows) if rec_rows else '<tr><td colspan="7">Sem recomendacoes disponiveis.</td></tr>'}</tbody>
    </table>
  </div>

  <h3>Agenda de Jogos Futuros (top 40)</h3>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Data</th><th>Mandante</th><th>Visitante</th><th>Odd Casa</th><th>Odd Empate</th><th>Odd Fora</th><th>Casas</th></tr></thead>
      <tbody>{''.join(rows_html) if rows_html else '<tr><td colspan="7">Sem jogos futuros.</td></tr>'}</tbody>
    </table>
  </div>
</section>
"""


def main() -> None:
    sections = []
    for comp in COMPETITIONS:
        df = load_competition_matches(comp)
        sections.append(_build_competition_section(comp, df))

    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Painel Moderno de Apostas</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --card: #ffffff;
      --line: #dbe4ef;
      --text: #1a2433;
      --muted: #5a6b82;
      --accent: #0f766e;
      --accent2: #0b5ed7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(1200px 500px at -10% -10%, #dbeafe 0%, transparent 60%),
        radial-gradient(900px 400px at 100% 0%, #dcfce7 0%, transparent 60%),
        var(--bg);
    }}
    .container {{ max-width: 1240px; margin: 0 auto; padding: 20px; }}
    .hero {{
      background: linear-gradient(130deg, #0f172a 0%, #0b2239 55%, #0f766e 100%);
      border: 1px solid #0f2a46;
      border-radius: 18px;
      padding: 18px;
      color: #f8fafc;
      box-shadow: 0 14px 28px rgba(15, 23, 42, 0.2);
    }}
    .hero h1 {{ margin: 0; font-size: 1.55rem; }}
    .hero p {{ margin: 8px 0 0; color: #dbe8f6; font-size: 0.94rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; margin-top: 12px; }}
    .mini {{ background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18); border-radius: 12px; padding: 10px; }}
    .mini .n {{ font-weight: 700; font-size: 1.1rem; }}
    .mini .l {{ font-size: 0.82rem; color: #dbe8f6; }}
    .card {{
      margin-top: 14px;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 8px 24px rgba(15,23,42,.06);
    }}
    .card-head {{ display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; }}
    h2 {{ margin: 0; font-size: 1.08rem; }}
    h3 {{ margin: 14px 0 8px; font-size: .98rem; color: #1f2d40; }}
    .badge {{ background: #ecfeff; color: #0f766e; border: 1px solid #bbf7d0; padding: 6px 10px; border-radius: 999px; font-size: 0.82rem; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 780px; background: #fff; }}
    th, td {{ border-bottom: 1px solid #e8edf4; padding: 8px; text-align: left; font-size: 13px; }}
    th {{ background: #eef4fb; position: sticky; top: 0; }}
    .foot {{ margin-top: 14px; color: var(--muted); font-size: 0.86rem; }}
    .legend {{
      margin-top: 10px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      font-size: 12px;
      color: #334155;
    }}
    .legend .pill {{
      border-radius: 999px;
      padding: 4px 10px;
      border: 1px solid #cbd5e1;
      background: #f8fafc;
    }}
    tr.risk-low td {{ background: #ecfdf3; }}
    tr.risk-med td {{ background: #fff7e6; }}
    tr.risk-high td {{ background: #fff1f2; }}
    tr.risk-none td {{ background: #f8fafc; }}
    .glossary {{
      margin-top: 14px;
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      box-shadow: 0 8px 24px rgba(15,23,42,.06);
    }}
    .glossary h3 {{ margin: 0 0 8px; }}
    .glossary ul {{ margin: 0; padding-left: 18px; }}
    .glossary li {{ margin: 6px 0; font-size: 13px; color: #334155; }}
    .controls {{
      margin-top: 12px;
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      box-shadow: 0 8px 24px rgba(15,23,42,.06);
    }}
    .controls label {{ font-size: 12px; color: #334155; font-weight: 600; }}
    .controls input, .controls select, .controls button, .controls a {{
      height: 34px;
      border-radius: 8px;
      border: 1px solid #cbd5e1;
      padding: 0 10px;
      font-size: 13px;
      background: #fff;
      color: #0f172a;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      cursor: pointer;
    }}
    .controls a, .controls button.primary {{
      background: #0b5ed7;
      border-color: #0b5ed7;
      color: #fff;
    }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }} }}
  </style>
</head>
<body>
  <div class="container">
    <section class="hero">
      <h1>Painel Moderno de Apostas (Sem API)</h1>
      <p>Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} • Fonte: scraping de paginas publicas do BetExplorer</p>
      <div class="grid">
        <div class="mini"><div class="n">4</div><div class="l">Competicoes monitoradas</div></div>
        <div class="mini"><div class="n">1X2</div><div class="l">Odds principais</div></div>
        <div class="mini"><div class="n">EV + Kelly</div><div class="l">Filtro de estrategia</div></div>
        <div class="mini"><div class="n">Top Seguros</div><div class="l">Ranking conservador</div></div>
      </div>
    </section>

    <section class="controls">
      <label for="fcomp">Competicao</label>
      <select id="fcomp">
        <option value="">Todas</option>
        <option>Brasileirao</option>
        <option>La Liga</option>
        <option>Premier League</option>
        <option>Copa do Mundo</option>
      </select>
      <label for="frisk">Perfil de risco</label>
      <select id="frisk">
        <option>Baixo risco</option>
        <option>Medio risco</option>
        <option>Alto risco</option>
        <option>Personalizado</option>
      </select>
      <label for="fteam">Time</label>
      <input id="fteam" type="text" placeholder="Ex: Flamengo" />
      <label for="foddmin">Odd min</label>
      <input id="foddmin" type="number" step="0.01" min="1.01" placeholder="1.30" />
      <label for="foddmax">Odd max</label>
      <input id="foddmax" type="number" step="0.01" min="1.01" placeholder="2.20" />
      <label for="fprobmin">Prob min</label>
      <input id="fprobmin" type="number" step="0.01" min="0" max="1" placeholder="0.55" />
      <label for="fevmin">EV min</label>
      <input id="fevmin" type="number" step="0.005" min="0" max="1" placeholder="0.02" />
      <label for="fbooks">Casas min</label>
      <input id="fbooks" type="number" step="1" min="0" placeholder="8" />
      <button id="applyFilter" class="primary" type="button">Aplicar filtro</button>
      <button id="clearFilter" type="button">Limpar</button>
      <a href="./atualizar_painel.bat">Atualizar dados (script)</a>
      <button id="reloadPage" type="button">Recarregar pagina</button>
    </section>
    <div class="legend">
      <span class="pill">Verde = baixo risco</span>
      <span class="pill">Amarelo = medio risco</span>
      <span class="pill">Vermelho = alto risco</span>
      <span class="pill">Cinza = fora dos criterios</span>
    </div>

    {''.join(sections)}

    <section class="glossary">
      <h3>Glossario de siglas e estatisticas</h3>
      <ul>
        <li><b>1X2</b>: Mercado principal do jogo. <b>1</b> = vitoria mandante, <b>X</b> = empate, <b>2</b> = vitoria visitante.</li>
        <li><b>Odd</b>: Cotacao da aposta. Exemplo: odd 2.00 retorna R$ 2,00 para cada R$ 1,00 apostado (retorno bruto).</li>
        <li><b>Prob. Modelo</b>: Probabilidade estimada pelo modelo para o resultado sugerido.</li>
        <li><b>EV</b> (Valor Esperado): Indicador matematico da vantagem da aposta. EV positivo tende a ser melhor no longo prazo.</li>
        <li><b>Casas</b> (B's): Quantidade de casas de apostas consideradas naquela linha de odd.</li>
        <li><b>Stake</b>: Valor sugerido para apostar naquele jogo, com base em banca e Kelly fracionado.</li>
      </ul>
    </section>

    <div class="foot">Aposta envolve risco. O painel ajuda na selecao, mas nao elimina perdas.</div>
  </div>
  <script>
    const riskPresets = {{
      "Baixo risco": {{ oddMin: 1.20, oddMax: 1.95, probMin: 0.62, evMin: 0.03, booksMin: 10 }},
      "Medio risco": {{ oddMin: 1.20, oddMax: 2.20, probMin: 0.55, evMin: 0.02, booksMin: 8 }},
      "Alto risco": {{ oddMin: 1.20, oddMax: 2.90, probMin: 0.48, evMin: 0.01, booksMin: 5 }},
    }};

    function applyRiskPreset() {{
      const risk = document.getElementById('frisk').value;
      if (risk === 'Personalizado') return;
      const p = riskPresets[risk];
      if (!p) return;
      document.getElementById('foddmin').value = p.oddMin.toFixed(2);
      document.getElementById('foddmax').value = p.oddMax.toFixed(2);
      document.getElementById('fprobmin').value = p.probMin.toFixed(2);
      document.getElementById('fevmin').value = p.evMin.toFixed(3);
      document.getElementById('fbooks').value = String(p.booksMin);
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
        if (prob >= 0.62 && ev >= 0.03 && odd <= 1.95 && books >= 10) {{
          row.classList.add('risk-low');
        }} else if (prob >= 0.55 && ev >= 0.02 && odd <= 2.20 && books >= 8) {{
          row.classList.add('risk-med');
        }} else if (prob >= 0.48 && ev >= 0.01 && odd <= 2.90 && books >= 5) {{
          row.classList.add('risk-high');
        }} else {{
          row.classList.add('risk-none');
        }}
      }});
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

      cards.forEach(card => {{
        const cardComp = (card.getAttribute('data-comp') || '').toLowerCase();
        const showCard = !comp || cardComp === comp;
        card.style.display = showCard ? '' : 'none';
        if (!showCard) return;

        const rows = card.querySelectorAll('tbody tr');
        rows.forEach(row => {{
          const txt = (row.textContent || '').toLowerCase();
          const teamOk = !team || txt.includes(team);

          let oddOk = true;
          const oneOdd = row.getAttribute('data-odd');
          if (oneOdd) {{
            const v = parseFloat(oneOdd);
            if (!Number.isNaN(v)) {{
              if (oddMin !== null && v < oddMin) oddOk = false;
              if (oddMax !== null && v > oddMax) oddOk = false;
            }}
            const rp = parseFloat(row.getAttribute('data-prob') || '');
            const re = parseFloat(row.getAttribute('data-ev') || '');
            const rb = parseInt(row.getAttribute('data-books') || '', 10);
            if (probMin !== null && !Number.isNaN(rp) && rp < probMin) oddOk = false;
            if (evMin !== null && !Number.isNaN(re) && re < evMin) oddOk = false;
            if (booksMin !== null && !Number.isNaN(rb) && rb < booksMin) oddOk = false;
          }} else {{
            const oh = parseFloat(row.getAttribute('data-odd-home') || '');
            const od = parseFloat(row.getAttribute('data-odd-draw') || '');
            const oa = parseFloat(row.getAttribute('data-odd-away') || '');
            const vals = [oh, od, oa].filter(v => !Number.isNaN(v));
            if (vals.length > 0) {{
              const minV = Math.min(...vals);
              const maxV = Math.max(...vals);
              if (oddMin !== null && maxV < oddMin) oddOk = false;
              if (oddMax !== null && minV > oddMax) oddOk = false;
            }}
          }}

          row.style.display = (teamOk && oddOk) ? '' : 'none';
        }});
      }});
      classifyRowsByRisk();
    }}

    document.getElementById('applyFilter').addEventListener('click', applyFilters);
    document.getElementById('fcomp').addEventListener('change', applyFilters);
    document.getElementById('fteam').addEventListener('input', applyFilters);
    document.getElementById('frisk').addEventListener('change', () => {{
      applyRiskPreset();
      applyFilters();
    }});

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

    document.getElementById('reloadPage').addEventListener('click', () => {{
      window.location.reload();
    }});

    applyRiskPreset();
    applyFilters();
    classifyRowsByRisk();
  </script>
</body>
</html>
"""

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
