# SALMOS - Stochastic ALM Optimization System
## Sistema de Otimização de Asset & Liability Management Bancário

Projeto acadêmico desenvolvido no contexto da proposta de início para um Doutorado na Poli-USP.
Correa Neto, W.

---

## Estrutura do Projeto

```
alm_bank/
│
├── data/
│   ├── raw/              # Dados brutos (curvas, balanços, mercado)
│   ├── processed/        # Dados tratados e normalizados
│   └── scenarios/        # Cenários de juros e stress gerados
│
├── models/
│   ├── balance_sheet/    # M1: Balanço patrimonial do banco maduro
│   ├── cash_flow/        # M3: Motor de projeção de fluxos de caixa
│   ├── optimization/     # M4: Otimizador multi-objetivo (NII + risco)
│   ├── hedge/            # M5: Hedge de duration/convexidade (DV01, swaps)
│   ├── liquidity/        # M6: LCR, NSFR, stress de liquidez
│   └── capital/          # M7: RWA, RAROC, Basileia III
│
├── outputs/
│   ├── charts/           # Gráficos gerados (.png, .svg)
│   ├── reports/          # Relatórios em PDF/Excel
│   └── tables/           # Tabelas exportadas (.csv, .xlsx)
│
├── notebooks/            # Jupyter Notebooks de análise e validação
├── utils/                # Funções auxiliares (datas, interpolação, etc.)
├── tests/                # Testes unitários por módulo
└── docs/                 # Documentação metodológica
```

---

## Módulos do Sistema

| # | Módulo | Status | Descrição |
|---|--------|--------|-----------|
| M1 | Balance Sheet | ✅ Fase 1 | Balanço de banco maduro brasileiro |
| M2 | Market Scenarios | ✅ Fase 1 | Curvas DI/IPCA, 3 cenários |
| M3 | Cash Flow Engine | 🔜 Fase 2 | Projeção de fluxos e gap de reprecificação |
| M4 | Optimizer | 🔜 Fase 3 | Otimizador multi-objetivo (cvxpy) |
| M5 | Hedge Module | 🔜 Fase 4 | DV01, duration gap, swaps |
| M6 | Liquidity (LCR/NSFR) | 🔜 Fase 4 | Basileia III liquidity ratios |
| M7 | Capital (RWA/RAROC) | 🔜 Fase 5 | RWA por FPR, alocação de capital |
| M8 | Dashboard | 🔜 Fase 6 | Reporting integrado |

---

## Como Executar

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Rodar balanço inicial
python models/balance_sheet/balance_sheet.py

# 3. Gerar cenários de mercado
python models/balance_sheet/market_scenarios.py

# 4. Executar análise completa (quando todos módulos prontos)
python main.py
```

---

## Dependências Principais
- `pandas`, `numpy` — estruturas de dados
- `scipy`, `cvxpy` — otimização
- `matplotlib`, `seaborn`, `plotly` — visualização
- `openpyxl` — exportação Excel

---

## Referências Metodológicas
- Basileia III — BIS (2010, 2017)
- Resolução CMN 4.557/2017 — BCB (Gestão de Riscos)
- Circular BCB 3.644/2013 — LCR
- Circular BCB 3.648/2013 — NSFR
- Fabozzi & Choudhry — *The Handbook of European Fixed Income Securities*
- Resti & Sironi — *Risk Management and Shareholders' Value in Banking*
