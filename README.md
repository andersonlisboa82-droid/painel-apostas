# Sistema de Apostas Sem API

Projeto em Python para acompanhar jogos e odds de:
- Brasileirao
- La Liga
- Premier League
- Copa do Mundo (rota 2026)

Sem uso de API com chave. Os dados sao raspados de paginas publicas do BetExplorer.

## O que o sistema faz
- Lista jogos futuros (fixtures) com odds 1X2.
- Mostra quantidade de casas consideradas (campo `B's` da pagina).
- Usa historico de resultados da mesma competicao para modelagem Poisson.
- Calcula probabilidade de Casa/Empate/Fora.
- Indica melhor mercado por Valor Esperado (EV).
- Sugere stake por Kelly fracionado.

## Requisitos
- Python 3.11 recomendado para deploy

## Instalacao
```bash
pip install -r requirements.txt
```

Para Streamlit Cloud, o projeto fixa o runtime em `python-3.11`.

## Execucao
```bash
streamlit run app.py
```

## Estrutura
- `app.py`: painel Streamlit
- `scraper.py`: scraping de resultados, fixtures e odds
- `analytics.py`: probabilidade (Poisson) + estrategia (EV/Kelly)

## Observacoes importantes
- Odds podem mudar rapidamente (mercado dinamico).
- O sistema trabalha com odds 1X2 agregadas mostradas no portal (melhor odd listada), nao com todas as linhas detalhadas por cada casa.
- Como e scraping, mudancas no HTML da fonte podem exigir manutencao.
- A recomendacao e estatistica, nao garantia de lucro.
