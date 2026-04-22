# Sistema de Apostas Sem API

Projeto em Python para acompanhar jogos e odds de:
- Brasileirao
- Copa do Brasil
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
streamlit run app.py --server.port 8503
```
Depois, acesse: `http://127.0.0.1:8503/?view=app`

## IA no Streamlit Cloud (sem erro de API local)
No deploy da Streamlit Cloud, a UI do `index.html` nao consegue acessar `localhost:8765`.
Para funcionar em producao, publique `portal_ai_server.py` em um backend HTTPS e informe a URL no app.

### 1) Publicar backend da IA (Render)
Este repositorio ja inclui `render.yaml` com o servico pronto.

No Render:
1. Crie um **New Web Service** apontando para este repo.
2. Render vai ler o `render.yaml` automaticamente.
3. Configure a env var `NVIDIA_API_KEY`.
4. Deploy.

URL esperada apos deploy (exemplo):
`https://painel-apostas-portal-ai.onrender.com`

### 2) Configurar Streamlit Cloud
No app da Streamlit Cloud, em **Settings > Secrets**, adicione:

```toml
PORTAL_REMOTE_API_BASE_URL = "https://painel-apostas-portal-ai.onrender.com"
NVIDIA_API_KEY = "SUA_CHAVE_NVIDIA_AQUI"
```

Depois clique em **Reboot app**.

Com isso:
- o portal continua local quando voce estiver local;
- na Cloud, o frontend usa a API remota automaticamente;
- a mensagem `A API local da IA nao fica exposta na Streamlit Cloud` some quando a URL remota estiver configurada.

## Estrutura
- `app.py`: painel Streamlit
- `scraper.py`: scraping de resultados, fixtures e odds
- `analytics.py`: probabilidade (Poisson) + estrategia (EV/Kelly)

## Observacoes importantes
- Odds podem mudar rapidamente (mercado dinamico).
- O sistema trabalha com odds 1X2 agregadas mostradas no portal (melhor odd listada), nao com todas as linhas detalhadas por cada casa.
- Como e scraping, mudancas no HTML da fonte podem exigir manutencao.
- A recomendacao e estatistica, nao garantia de lucro.
