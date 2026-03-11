"""
M2 - Market Scenarios Module
=============================
Geração de cenários de curva de juros para o ALM:
  - Curva DI (pré-fixada) via interpolação Nelson-Siegel
  - Curva IPCA implícita via NTN-B sintéticas
  - 3 cenários: BASE | STRESS ALTA | STRESS BAIXA
  - Exporta curvas para uso nos módulos de cash flow e otimização

Referência: Metodologia B3 (curvas de referência) + BCB Focus
"""

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = BASE_DIR / "data" / "scenarios"
CHARTS_DIR = BASE_DIR / "outputs" / "charts"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# BUCKETS TEMPORAIS PADRÃO (anos)
# ─────────────────────────────────────────────
TENORS = np.array([
    1/12, 2/12, 3/12, 6/12,          # curtíssimo prazo
    1.0, 1.5, 2.0, 3.0,              # curto/médio
    4.0, 5.0, 6.0, 7.0,              # médio prazo
    8.0, 10.0, 15.0, 20.0, 30.0      # longo prazo
])

TENOR_LABELS = [
    "1M", "2M", "3M", "6M",
    "1A", "1.5A", "2A", "3A",
    "4A", "5A", "6A", "7A",
    "8A", "10A", "15A", "20A", "30A"
]

# ─────────────────────────────────────────────
# PARÂMETROS NELSON-SIEGEL
# ─────────────────────────────────────────────
@dataclass
class NelsonSiegelParams:
    """
    Modelo Nelson-Siegel:
      r(t) = β0 + β1*(1 - e^(-t/λ))/(t/λ) + β2*((1 - e^(-t/λ))/(t/λ) - e^(-t/λ))
    β0: nível longo
    β1: inclinação (spread curto x longo)
    β2: curvatura
    λ:  fator de decaimento
    """
    beta0: float   # nível longo (%)
    beta1: float   # inclinação
    beta2: float   # curvatura
    lam: float     # lambda (anos)
    nome: str = ""

    def taxa(self, t: np.ndarray) -> np.ndarray:
        """Retorna taxa a.a. para cada tenor t (em anos)."""
        t = np.where(t < 1e-6, 1e-6, t)
        exp_t = np.exp(-t / self.lam)
        fator1 = (1 - exp_t) / (t / self.lam)
        fator2 = fator1 - exp_t
        return self.beta0 + self.beta1 * fator1 + self.beta2 * fator2


# ─────────────────────────────────────────────
# CENÁRIOS CALIBRADOS (Base = Dez/2024)
# ─────────────────────────────────────────────

# Calibração da curva DI base aproximando mercado B3 Dez/2024
# SELIC = 11.75% | curva invertida/flat com leve inclinação
CURVA_DI_BASE = NelsonSiegelParams(
    beta0=12.80,   # nível longo
    beta1=-1.30,   # curva levemente invertida no curto
    beta2=-2.50,   # curvatura negativa (hump invertido)
    lam=3.5,
    nome="DI_Base"
)

# Stress Alta: choque de +200 bps na curva (crise fiscal/inflacionária)
CURVA_DI_STRESS_ALTA = NelsonSiegelParams(
    beta0=14.80,
    beta1=0.80,    # curva passa a ter inclinação positiva
    beta2=-1.80,
    lam=3.5,
    nome="DI_Stress_Alta"
)

# Stress Baixa: corte de -250 bps (ciclo de queda)
CURVA_DI_STRESS_BAIXA = NelsonSiegelParams(
    beta0=10.30,
    beta1=-2.80,
    beta2=-3.20,
    lam=3.5,
    nome="DI_Stress_Baixa"
)

# IPCA esperado: inflação implícita por tenor (BE — breakeven inflation)
# Derivado de: taxa pré - taxa real NTN-B (Fischer)
IPCA_IMPLICITO_BASE   = np.array([4.8, 4.7, 4.6, 4.5, 4.4, 4.4, 4.3, 4.2,
                                   4.1, 4.0, 3.9, 3.9, 3.8, 3.7, 3.6, 3.6, 3.5])
IPCA_IMPLICITO_STRESS = np.array([6.5, 6.4, 6.3, 6.2, 6.0, 5.9, 5.8, 5.6,
                                   5.4, 5.2, 5.0, 4.9, 4.8, 4.7, 4.5, 4.4, 4.3])
IPCA_IMPLICITO_BAIXO  = np.array([3.2, 3.2, 3.1, 3.0, 3.0, 2.9, 2.9, 2.8,
                                   2.8, 2.7, 2.7, 2.7, 2.6, 2.6, 2.6, 2.5, 2.5])


# ─────────────────────────────────────────────
# CLASSE PRINCIPAL DE CENÁRIOS
# ─────────────────────────────────────────────
class MarketScenarios:
    """
    Gera e armazena os cenários de mercado para uso no ALM.
    Inclui: curvas DI, IPCA, spread de crédito, projeção de crescimento.
    """

    def __init__(self, tenors: np.ndarray = TENORS):
        self.tenors = tenors
        self.tenor_labels = TENOR_LABELS
        self.curvas_di: Dict[str, np.ndarray] = {}
        self.curvas_ipca: Dict[str, np.ndarray] = {}
        self.curvas_real: Dict[str, np.ndarray] = {}
        self.spreads_credito: Dict[str, Dict] = {}
        self.projecoes_crescimento: Dict = {}
        self._gerar_curvas_di()
        self._gerar_curvas_ipca()
        self._gerar_spreads_credito()
        self._gerar_projecoes_crescimento()

    # ─── CURVAS DI ────────────────────────────
    def _gerar_curvas_di(self):
        """Gera as 3 curvas DI via Nelson-Siegel."""
        cenarios_ns = {
            "base":         CURVA_DI_BASE,
            "stress_alta":  CURVA_DI_STRESS_ALTA,
            "stress_baixa": CURVA_DI_STRESS_BAIXA,
        }
        for nome, params in cenarios_ns.items():
            self.curvas_di[nome] = params.taxa(self.tenors)

    # ─── CURVAS IPCA E REAL ───────────────────
    def _gerar_curvas_ipca(self):
        """Gera curvas IPCA implícita e taxa real (NTN-B sintética)."""
        self.curvas_ipca["base"]         = IPCA_IMPLICITO_BASE
        self.curvas_ipca["stress_alta"]  = IPCA_IMPLICITO_STRESS
        self.curvas_ipca["stress_baixa"] = IPCA_IMPLICITO_BAIXO

        # Taxa real = [(1 + DI) / (1 + IPCA)] - 1 (aproximação Fischer)
        for cen in ["base", "stress_alta", "stress_baixa"]:
            di   = self.curvas_di[cen] / 100
            ipca = self.curvas_ipca[cen] / 100
            self.curvas_real[cen] = ((1 + di) / (1 + ipca) - 1) * 100

    # ─── SPREADS DE CRÉDITO ───────────────────
    def _gerar_spreads_credito(self):
        """
        Spreads de crédito por segmento (sobre CDI ou pré).
        Cenário base vs stress (widening de spreads).
        Unidade: % a.a.
        """
        self.spreads_credito = {
            "base": {
                "imobiliario_pf":    8.20,
                "consignado_pf":     10.75,  # spread sobre pré
                "pessoal_pf":        26.65,
                "veiculos_pf":       15.05,
                "cartao_pf":        177.25,
                "capital_giro_pj":    4.50,
                "corporativo_pj":     2.80,
                "trade_finance":      2.20,
                "rural":              0.50,   # subsidiado
            },
            "stress_alta": {  # recessão: spreads sobem 30-50%
                "imobiliario_pf":   10.00,
                "consignado_pf":    13.00,
                "pessoal_pf":       34.00,
                "veiculos_pf":      19.50,
                "cartao_pf":       220.00,
                "capital_giro_pj":   6.80,
                "corporativo_pj":    4.50,
                "trade_finance":     3.20,
                "rural":             1.00,
            },
            "stress_baixa": {  # expansão: spreads comprimem ~20%
                "imobiliario_pf":    6.80,
                "consignado_pf":     8.50,
                "pessoal_pf":       21.50,
                "veiculos_pf":      12.00,
                "cartao_pf":       145.00,
                "capital_giro_pj":   3.20,
                "corporativo_pj":    2.00,
                "trade_finance":     1.60,
                "rural":             0.30,
            },
        }

    # ─── CRESCIMENTO DE CARTEIRA ──────────────
    def _gerar_projecoes_crescimento(self):
        """
        Projeção de crescimento anual das carteiras (% a.a.).
        Horizonte: 5 anos.
        Inclui: novos contratos líquidos de amortizações.
        """
        self.projecoes_crescimento = {
            "horizonte_anos": 5,
            "base": {
                "imobiliario_pf":   8.0,
                "consignado_pf":    6.0,
                "pessoal_pf":       4.0,
                "veiculos_pf":      5.0,
                "cartao_pf":        7.0,
                "capital_giro_pj":  5.0,
                "corporativo_pj":   6.0,
                "trade_finance":    4.0,
                "rural":            5.0,
            },
            "stress_alta": {  # recessão: retração do crédito
                "imobiliario_pf":   2.0,
                "consignado_pf":    3.0,
                "pessoal_pf":      -2.0,
                "veiculos_pf":     -1.0,
                "cartao_pf":        1.0,
                "capital_giro_pj": -3.0,
                "corporativo_pj":  -1.0,
                "trade_finance":   -2.0,
                "rural":            3.0,
            },
            "stress_baixa": {  # boom de crédito
                "imobiliario_pf":  14.0,
                "consignado_pf":    9.0,
                "pessoal_pf":       8.0,
                "veiculos_pf":     10.0,
                "cartao_pf":       12.0,
                "capital_giro_pj": 10.0,
                "corporativo_pj":  11.0,
                "trade_finance":    8.0,
                "rural":            6.0,
            },
        }

    # ─── DATAFRAMES ───────────────────────────
    def to_dataframe_di(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "tenor_anos": self.tenors,
            "tenor_label": self.tenor_labels,
            "di_base":        self.curvas_di["base"],
            "di_stress_alta": self.curvas_di["stress_alta"],
            "di_stress_baixa":self.curvas_di["stress_baixa"],
        })
        return df

    def to_dataframe_ipca(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "tenor_anos": self.tenors,
            "tenor_label": self.tenor_labels,
            "ipca_base":         self.curvas_ipca["base"],
            "ipca_stress_alta":  self.curvas_ipca["stress_alta"],
            "ipca_stress_baixa": self.curvas_ipca["stress_baixa"],
            "real_base":         self.curvas_real["base"],
            "real_stress_alta":  self.curvas_real["stress_alta"],
            "real_stress_baixa": self.curvas_real["stress_baixa"],
        })
        return df

    def taxa_para_tenor(self, tenor_anos: float, cenario: str = "base",
                        indexador: str = "CDI") -> float:
        """
        Retorna taxa interpolada para um tenor específico.
        Útil para precificação de novos contratos.
        """
        if indexador in ("CDI", "PRE"):
            curva = self.curvas_di[cenario]
        elif indexador == "IPCA":
            curva = self.curvas_ipca[cenario]
        elif indexador == "REAL":
            curva = self.curvas_real[cenario]
        else:
            raise ValueError(f"Indexador {indexador} não suportado")
        return float(np.interp(tenor_anos, self.tenors, curva))

    def salvar(self):
        """Exporta todos os cenários."""
        self.to_dataframe_di().to_csv(DATA_DIR / "curvas_di.csv", index=False)
        self.to_dataframe_ipca().to_csv(DATA_DIR / "curvas_ipca.csv", index=False)
        with open(DATA_DIR / "spreads_credito.json", "w") as f:
            json.dump(self.spreads_credito, f, indent=2)
        with open(DATA_DIR / "projecoes_crescimento.json", "w") as f:
            json.dump(self.projecoes_crescimento, f, indent=2)
        print(f"✅ Cenários salvos em: {DATA_DIR}")

    def imprimir_resumo(self):
        df = self.to_dataframe_di()
        print("\n" + "="*70)
        print("  CURVAS DE JUROS (% a.a.) — Tenores Selecionados")
        print("="*70)
        print(f"  {'Tenor':<8} {'DI Base':>10} {'DI Alta':>10} {'DI Baixa':>10} {'IPCA Base':>10} {'Real Base':>10}")
        print("-"*70)
        idx_sel = [0, 3, 4, 7, 9, 13, 15]
        df_ipca = self.to_dataframe_ipca()
        for i in idx_sel:
            r = df.iloc[i]
            ri = df_ipca.iloc[i]
            print(f"  {r['tenor_label']:<8} {r['di_base']:>9.2f}% "
                  f"{r['di_stress_alta']:>9.2f}% {r['di_stress_baixa']:>9.2f}%"
                  f" {ri['ipca_base']:>9.2f}%  {ri['real_base']:>9.2f}%")
        print("="*70)


# ─────────────────────────────────────────────
# EXECUÇÃO DIRETA
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    cenarios = MarketScenarios()
    cenarios.salvar()
    cenarios.imprimir_resumo()

    df_di   = cenarios.to_dataframe_di()
    df_ipca = cenarios.to_dataframe_ipca()

    # ── GRÁFICO 4: Curvas DI — 3 cenários ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Curvas de Juros — Cenários ALM (Dez/2024)", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(df_di["tenor_anos"], df_di["di_base"],         "b-o",  ms=4, lw=2,   label="DI Base")
    ax.plot(df_di["tenor_anos"], df_di["di_stress_alta"],  "r--s", ms=4, lw=1.5, label="DI Stress Alta (+200bps)")
    ax.plot(df_di["tenor_anos"], df_di["di_stress_baixa"], "g--^", ms=4, lw=1.5, label="DI Stress Baixa (-250bps)")
    ax.fill_between(df_di["tenor_anos"],
                    df_di["di_stress_baixa"],
                    df_di["di_stress_alta"],
                    alpha=0.08, color="gray", label="Envelope de stress")
    ax.set_title("Curva DI (Pré-fixada)", fontweight="bold")
    ax.set_xlabel("Prazo (anos)")
    ax.set_ylabel("Taxa % a.a.")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0, 10)

    ax2 = axes[1]
    ax2.plot(df_ipca["tenor_anos"], df_ipca["ipca_base"],         "b-o",  ms=4, lw=2,   label="IPCA Base")
    ax2.plot(df_ipca["tenor_anos"], df_ipca["ipca_stress_alta"],  "r--s", ms=4, lw=1.5, label="IPCA Stress Alta")
    ax2.plot(df_ipca["tenor_anos"], df_ipca["ipca_stress_baixa"], "g--^", ms=4, lw=1.5, label="IPCA Stress Baixa")
    ax2.plot(df_ipca["tenor_anos"], df_ipca["real_base"],         "k-.",  ms=3, lw=1.5, label="Taxa Real Base")
    ax2.set_title("Inflação Implícita (IPCA) e Taxa Real", fontweight="bold")
    ax2.set_xlabel("Prazo (anos)")
    ax2.set_ylabel("Taxa % a.a.")
    ax2.legend(fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax2.grid(True, alpha=0.4)
    ax2.set_xlim(0, 10)

    plt.tight_layout()
    path_g4 = CHARTS_DIR / "04_curvas_juros_cenarios.png"
    plt.savefig(path_g4, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Gráfico salvo: {path_g4}")

    # ── GRÁFICO 5: Spread de crédito por segmento ────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    spreads = cenarios.spreads_credito
    produtos = list(spreads["base"].keys())
    x = np.arange(len(produtos))
    w = 0.28

    bars1 = ax.bar(x - w, [spreads["base"][p]        for p in produtos], w, label="Base",        color="#2196F3")
    bars2 = ax.bar(x,     [spreads["stress_alta"][p]  for p in produtos], w, label="Stress Alta",  color="#F44336", alpha=0.8)
    bars3 = ax.bar(x + w, [spreads["stress_baixa"][p] for p in produtos], w, label="Stress Baixa", color="#4CAF50", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in produtos], fontsize=8)
    ax.set_title("Spreads de Crédito por Segmento e Cenário (% a.a.)", fontweight="bold")
    ax.set_ylabel("Spread % a.a.")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path_g5 = CHARTS_DIR / "05_spreads_credito.png"
    plt.savefig(path_g5, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Gráfico salvo: {path_g5}")

    print("\n🏁 M2 — Market Scenarios concluído com sucesso!")
