"""
M7 - Capital Module
====================
Implementa os quatro pilares de gestão de capital do ALM:

  1. RWA por Abordagem Padronizada (FPR refinado por subclasse)
     Resolução BCB 4.958/2021 + Circular 3.360/2007

  2. RAROC por Linha de Negócio
     RAROC = (Receita Líquida - Perda Esperada - Custo Operacional)
             / Capital Econômico Alocado

  3. Alocação de Capital Econômico por Segmento
     Baseado em VaR de crédito (abordagem paramétrica)
     CE = UL × α  onde UL = √(PD×LGD²×EAD² + correlação_sistêmica)

  4. Pricing via RAROC Mínimo
     Spread_mínimo = (RAROC_alvo × CE + PE + Custo_funding + Custo_op)
                     / EAD

Referências:
  - Saunders & Cornett (2010): Financial Institutions Management
  - Resti & Sironi (2007): Risk Management and Shareholders' Value
  - BIS BCBS (2006): Basel II — IRB Approach for Credit Risk
  - RAROC: Bankers Trust / Merton & Perold (1993)
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

DATA_DIR   = BASE_DIR / "data" / "processed"
SCEN_DIR   = BASE_DIR / "data" / "scenarios"
OUTPUT_DIR = BASE_DIR / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# PARÂMETROS GLOBAIS DE CAPITAL
# ─────────────────────────────────────────────────────────────
RAROC_ALVO          = 0.15    # 15% a.a. — hurdle rate do banco
NIVEL_CONFIANCA_CE  = 0.999   # 99.9% — padrão Basileia II IRB
CUSTO_CAPITAL_EQUITY = 0.15   # Ke = 15% (CAPM implícito)
CUSTO_FUNDING_BASE  = 11.75   # CDI base (% a.a.)
IB_MINIMO           = 0.105   # 10.5% com buffer

# Correlação ativo sistêmico (ρ) por segmento — Basileia II IRB
# ρ = 0.12×(1-e^(-50×PD))/(1-e^(-50)) + 0.24×(1-(1-e^(-50×PD))/(1-e^(-50)))
# Simplificado: valores tabelados
RHO_SISTEMICO = {
    "imobiliario_pf":   0.15,
    "consignado_pf":    0.10,
    "pessoal_pf":       0.08,
    "veiculos_pf":      0.10,
    "cartao_pf":        0.06,
    "capital_giro_pj":  0.15,
    "corporativo_pj":   0.20,
    "trade_finance":    0.12,
    "rural":            0.12,
    "tvm_soberano":     0.00,
    "tvm_privado":      0.10,
    "outros":           0.12,
}

# Parâmetros de crédito por segmento
PARAMS_CREDITO = {
    "Crédito imobiliário PF":  {"pd": 0.0080, "lgd": 0.20, "m": 18.0, "custo_op": 0.50},
    "Crédito consignado PF":   {"pd": 0.0250, "lgd": 0.30, "m":  3.8, "custo_op": 1.20},
    "Crédito pessoal PF":      {"pd": 0.0850, "lgd": 0.55, "m":  2.2, "custo_op": 2.50},
    "Crédito veículos PF":     {"pd": 0.0380, "lgd": 0.40, "m":  4.5, "custo_op": 1.50},
    "Cartão crédito PF":       {"pd": 0.0650, "lgd": 0.75, "m":  0.5, "custo_op": 4.00},
    "Capital de giro PJ PME":  {"pd": 0.0420, "lgd": 0.45, "m":  1.5, "custo_op": 1.80},
    "Crédito corporativo PJ":  {"pd": 0.0180, "lgd": 0.35, "m":  3.2, "custo_op": 0.80},
    "Trade finance":           {"pd": 0.0050, "lgd": 0.25, "m":  0.7, "custo_op": 0.60},
    "Crédito rural":           {"pd": 0.0200, "lgd": 0.30, "m":  2.8, "custo_op": 1.00},
}


# ─────────────────────────────────────────────────────────────
# CLASSE PRINCIPAL
# ─────────────────────────────────────────────────────────────
class CapitalModule:
    """Módulo completo de gestão e alocação de capital."""

    def __init__(self):
        self.ativos   = pd.read_csv(DATA_DIR / "ativos.csv")
        self.passivos = pd.read_csv(DATA_DIR / "passivos.csv")
        with open(DATA_DIR / "patrimonio.json") as f:
            self.patrimonio = json.load(f)

    # ─── 1. RWA DETALHADO ────────────────────────────────────
    def calcular_rwa_detalhado(self) -> pd.DataFrame:
        """
        RWA por produto com FPR refinado e decomposição:
        RWA_crédito + RWA_mercado (simplificado) + RWA_operacional
        """
        rows = []
        for _, row in self.ativos.iterrows():
            fpr   = row["fpr"]
            saldo = row["saldo"]
            rwa   = saldo * fpr

            # Fator de mitigação (garantias simplificado)
            mitigacao = 0.0
            if row["produto"] == "Crédito imobiliário PF":
                mitigacao = saldo * 0.35 * 0.30   # 30% dos imóveis como colateral
            elif row["produto"] == "Trade finance":
                mitigacao = saldo * 0.20 * 0.50

            rwa_mitigado = max(0, rwa - mitigacao)

            rows.append({
                "produto":          row["produto"],
                "categoria":        row["categoria"],
                "saldo_R$M":        saldo,
                "fpr":              fpr,
                "rwa_bruto_R$M":    rwa,
                "mitigacao_R$M":    mitigacao,
                "rwa_liquido_R$M":  rwa_mitigado,
                "capital_req_R$M":  rwa_mitigado * IB_MINIMO,
            })

        df = pd.DataFrame(rows)

        # RWA Operacional (Abordagem do Indicador Básico — 15% da RB média 3 anos)
        nii_estimado = 61_026   # R$ milhões — NII base da Fase 2
        rwa_operacional = nii_estimado * 0.15 / IB_MINIMO
        df_op = pd.DataFrame([{
            "produto": "RWA Operacional (ABI)",
            "categoria": "Risco Operacional",
            "saldo_R$M": 0,
            "fpr": None,
            "rwa_bruto_R$M": rwa_operacional,
            "mitigacao_R$M": 0,
            "rwa_liquido_R$M": rwa_operacional,
            "capital_req_R$M": rwa_operacional * IB_MINIMO,
        }])
        df = pd.concat([df, df_op], ignore_index=True)
        df.to_csv(OUTPUT_DIR / "rwa_detalhado.csv", index=False)
        return df

    # ─── 2. CAPITAL ECONÔMICO ────────────────────────────────
    def capital_economico_segmento(self) -> pd.DataFrame:
        """
        Capital Econômico via VaR de crédito paramétrico (Basileia II IRB).

        Fórmula Basileia II para Unexpected Loss (UL):
          UL = EAD × LGD × [N(√(1/(1-ρ))×N⁻¹(PD) + √(ρ/(1-ρ))×N⁻¹(α)) - PD]
          onde α = nível de confiança (99.9%)

        CE = UL (capital para cobrir perdas inesperadas)
        PE = EAD × PD × LGD (perda esperada — coberta por provisão/spread)
        """
        from scipy.stats import norm
        alpha = NIVEL_CONFIANCA_CE

        rows = []
        for produto, params in PARAMS_CREDITO.items():
            # Busca saldo no balanço
            saldo_row = self.ativos[self.ativos["produto"] == produto]
            if saldo_row.empty:
                continue
            ead = saldo_row["saldo"].iloc[0]   # EAD ≈ saldo contábil

            pd_  = params["pd"]
            lgd  = params["lgd"]
            m    = params["m"]    # maturity (anos)

            # Correlação sistêmica
            seg_key = produto.lower().replace("crédito ", "").replace(" pf", "_pf").replace(" pj", "_pj")
            seg_map = {
                "imobiliário pf": "imobiliario_pf",
                "consignado pf":  "consignado_pf",
                "pessoal pf":     "pessoal_pf",
                "veículos pf":    "veiculos_pf",
                "cartão crédito pf": "cartao_pf",
                "capital de giro pj pme": "capital_giro_pj",
                "corporativo pj": "corporativo_pj",
                "trade finance":  "trade_finance",
                "rural":          "rural",
            }
            rho_key = seg_map.get(produto.lower().replace("crédito ", ""), "outros")
            rho = RHO_SISTEMICO.get(rho_key, 0.12)

            # Fórmula IRB Basileia II
            z_pd    = norm.ppf(pd_)
            z_alpha = norm.ppf(alpha)
            condicional_pd = norm.cdf(
                (z_pd + np.sqrt(rho) * z_alpha) / np.sqrt(1 - rho)
            )

            # Ajuste de maturidade (simplificado — Basileia II)
            b_pd = (0.11852 - 0.05478 * np.log(pd_)) ** 2
            ma = (1 + (m - 2.5) * b_pd) / (1 - 1.5 * b_pd)
            ma = np.clip(ma, 1.0, 5.0)

            ul     = ead * lgd * (condicional_pd - pd_) * ma
            pe     = ead * pd_ * lgd
            ce     = ul   # Capital Econômico = UL

            rows.append({
                "produto":              produto,
                "ead_R$M":              ead,
                "pd_pct":               pd_ * 100,
                "lgd_pct":              lgd * 100,
                "rho_sistemico":        rho,
                "maturity_anos":        m,
                "pe_R$M":               pe,
                "ul_R$M":               ul,
                "ce_R$M":               ce,
                "ce_pct_ead":           ce / ead * 100,
                "custo_ce_R$M":         ce * CUSTO_CAPITAL_EQUITY,
            })

        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_DIR / "capital_economico.csv", index=False)
        return df

    # ─── 3. RAROC POR LINHA DE NEGÓCIO ───────────────────────
    def calcular_raroc(self, df_ce: pd.DataFrame) -> pd.DataFrame:
        """
        RAROC = (NII_segmento - PE - Custo_Op - Custo_Funding) / CE

        NII_segmento = EAD × taxa_ativa  (receita bruta de juros)
        Custo_funding = EAD × custo_médio_passivo
        PE = perda esperada anual
        Custo_Op = % do EAD (processamento, cobrança etc.)
        """
        custo_funding_medio = (
            (self.passivos["saldo"] * self.passivos.get("custo", pd.Series(
                [CUSTO_FUNDING_BASE * 0.98] * len(self.passivos)
            ))).sum() / self.passivos["saldo"].sum() / 100
            if "custo" in self.passivos.columns else CUSTO_FUNDING_BASE * 0.98 / 100
        )

        rows = []
        for _, ce_row in df_ce.iterrows():
            produto = ce_row["produto"]
            ead     = ce_row["ead_R$M"]
            ce      = ce_row["ce_R$M"]
            pe      = ce_row["pe_R$M"]

            # Taxa ativa do produto
            ativo_row = self.ativos[self.ativos["produto"] == produto]
            if ativo_row.empty:
                continue
            taxa_ativa = ativo_row["taxa_total"].iloc[0] / 100

            # Custo operacional (% a.a. do EAD)
            custo_op_pct = PARAMS_CREDITO.get(produto, {}).get("custo_op", 1.5) / 100

            receita_juros  = ead * taxa_ativa
            custo_fund     = ead * (CUSTO_FUNDING_BASE / 100) * 0.98
            custo_op       = ead * custo_op_pct
            custo_ce_op    = ce  * CUSTO_CAPITAL_EQUITY   # custo do capital alocado

            nii_liquido    = receita_juros - custo_fund - pe - custo_op
            raroc          = nii_liquido / ce if ce > 0 else 0.0
            eva            = nii_liquido - custo_ce_op   # EVA = lucro econômico

            rows.append({
                "produto":             produto,
                "ead_R$M":             ead,
                "ce_R$M":              ce,
                "receita_juros_R$M":   receita_juros,
                "custo_funding_R$M":   custo_fund,
                "pe_R$M":              pe,
                "custo_op_R$M":        custo_op,
                "nii_liquido_R$M":     nii_liquido,
                "raroc_pct":           raroc * 100,
                "raroc_acima_alvo":    raroc >= RAROC_ALVO,
                "eva_R$M":             eva,
                "spread_efetivo_pct":  (taxa_ativa - CUSTO_FUNDING_BASE/100) * 100,
            })

        df = pd.DataFrame(rows).sort_values("raroc_pct", ascending=False)
        df.to_csv(OUTPUT_DIR / "raroc_linhas_negocio.csv", index=False)
        return df

    # ─── 4. PRICING VIA RAROC MÍNIMO ─────────────────────────
    def pricing_raroc_minimo(self, df_ce: pd.DataFrame,
                              raroc_alvo: float = RAROC_ALVO) -> pd.DataFrame:
        """
        Calcula o spread mínimo por segmento para atingir RAROC ≥ alvo.

        Spread_mín = (raroc_alvo × CE/EAD + PE/EAD + custo_op + custo_funding)
                    expressa em % a.a.

        Compara com spread atual de mercado → sinaliza onde banco está
        precificando acima/abaixo do mínimo econômico.
        """
        rows = []
        for _, ce_row in df_ce.iterrows():
            produto = ce_row["produto"]
            ead     = ce_row["ead_R$M"]
            ce      = ce_row["ce_R$M"]
            pe      = ce_row["pe_R$M"]

            ativo_row = self.ativos[self.ativos["produto"] == produto]
            if ativo_row.empty:
                continue

            taxa_atual  = ativo_row["taxa_total"].iloc[0]
            indexador   = ativo_row["indexador"].iloc[0]
            custo_op    = PARAMS_CREDITO.get(produto, {}).get("custo_op", 1.5)

            # Taxa mínima para RAROC_alvo
            # receita_mín = custo_funding + PE/EAD + custo_op + raroc_alvo × CE/EAD
            custo_fund_pct = CUSTO_FUNDING_BASE * 0.98
            pe_pct         = (pe / ead) * 100 if ead > 0 else 0
            ce_pct         = (ce / ead) * 100 if ead > 0 else 0
            taxa_minima    = custo_fund_pct + pe_pct + custo_op + raroc_alvo * ce_pct

            # Para PRE: taxa total = taxa_minima (já inclui funding)
            # Para CDI: spread_minimo = taxa_minima - CDI
            if indexador == "CDI":
                spread_atual   = taxa_atual - CUSTO_FUNDING_BASE
                spread_minimo  = taxa_minima - CUSTO_FUNDING_BASE
            else:
                spread_atual   = taxa_atual
                spread_minimo  = taxa_minima

            folga = spread_atual - spread_minimo

            rows.append({
                "produto":            produto,
                "indexador":          indexador,
                "taxa_atual_pct":     taxa_atual,
                "taxa_minima_pct":    taxa_minima,
                "spread_atual_pct":   spread_atual,
                "spread_minimo_pct":  spread_minimo,
                "folga_spread_pct":   folga,
                "ce_pct_ead":         ce_pct,
                "pe_pct_ead":         pe_pct,
                "custo_op_pct":       custo_op,
                "adequado_raroc":     folga >= 0,
                "raroc_implicito":    (
                    (taxa_atual - custo_fund_pct - pe_pct - custo_op) / ce_pct
                    if ce_pct > 0 else 0
                ),
            })

        df = pd.DataFrame(rows).sort_values("folga_spread_pct", ascending=False)
        df.to_csv(OUTPUT_DIR / "pricing_raroc_minimo.csv", index=False)
        return df

    def imprimir_raroc(self, df_raroc: pd.DataFrame):
        print(f"\n{'='*75}")
        print(f"  RAROC POR LINHA DE NEGÓCIO (Hurdle Rate: {RAROC_ALVO:.0%})")
        print(f"{'='*75}")
        print(f"  {'Produto':<26} {'EAD':>8} {'CE':>7} {'NII Líq':>9} "
              f"{'RAROC':>7} {'EVA':>8} {'OK?':>5}")
        print(f"  {'─'*73}")
        for _, r in df_raroc.iterrows():
            ok = "✅" if r["raroc_acima_alvo"] else "❌"
            print(f"  {r['produto']:<26} {r['ead_R$M']:>7,.0f} "
                  f"{r['ce_R$M']:>6,.0f} {r['nii_liquido_R$M']:>8,.0f} "
                  f"{r['raroc_pct']:>6.1f}% {r['eva_R$M']:>7,.0f} {ok:>5}")
        print(f"  {'─'*73}")
        total_nii = df_raroc["nii_liquido_R$M"].sum()
        total_ce  = df_raroc["ce_R$M"].sum()
        total_eva = df_raroc["eva_R$M"].sum()
        raroc_med = total_nii / total_ce * 100 if total_ce > 0 else 0
        print(f"  {'TOTAL':<26} {'':>8} {total_ce:>6,.0f} {total_nii:>8,.0f} "
              f"{raroc_med:>6.1f}% {total_eva:>7,.0f}")
        print(f"{'='*75}")

    def imprimir_pricing(self, df_pricing: pd.DataFrame):
        print(f"\n{'='*75}")
        print(f"  PRICING MÍNIMO VIA RAROC ≥ {RAROC_ALVO:.0%}")
        print(f"{'='*75}")
        print(f"  {'Produto':<26} {'Tx Atual':>9} {'Tx Mín':>8} "
              f"{'Folga':>7} {'RAROC Impl':>11} {'OK?':>4}")
        print(f"  {'─'*73}")
        for _, r in df_pricing.iterrows():
            ok = "✅" if r["adequado_raroc"] else "❌"
            print(f"  {r['produto']:<26} {r['taxa_atual_pct']:>8.2f}% "
                  f"{r['taxa_minima_pct']:>7.2f}% "
                  f"{r['folga_spread_pct']:>+6.2f}% "
                  f"{r['raroc_implicito']:>10.1f}% {ok:>4}")
        print(f"{'='*75}")


if __name__ == "__main__":
    cap = CapitalModule()
    df_rwa = cap.calcular_rwa_detalhado()
    df_ce  = cap.capital_economico_segmento()
    df_raroc = cap.calcular_raroc(df_ce)
    df_prec  = cap.pricing_raroc_minimo(df_ce)
    cap.imprimir_raroc(df_raroc)
    cap.imprimir_pricing(df_prec)
