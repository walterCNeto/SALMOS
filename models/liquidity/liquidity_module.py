"""
M6 - Liquidity Module
======================
Implementa os três pilares de gestão de liquidez do ALM:

  1. Stress Test 30 dias (Basileia III / Circular BCB 3.644/2013)
     - Outflows por tipo de contraparte e instrumento
     - Inflows de créditos a receber
     - LCR = HQLA / Net Cash Outflows (NCO)

  2. Custo de Carregamento do Buffer HQLA
     - Carry negativo: custo de oportunidade de manter HQLA vs. crédito
     - Custo total do buffer por nível (1, 2A, 2B)

  3. Projeção Dinâmica LCR/NSFR — 12 meses
     - Evolução mensal com crescimento de carteira (cenários)
     - Sinaliza antecipadamente quando restrições podem ser violadas

Referências:
  Circular BCB 3.644/2013 (LCR) | Circular BCB 3.648/2013 (NSFR)
  BIS BCBS 238 (2013) | BIS BCBS 295 (2014)
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

DATA_DIR   = BASE_DIR / "data" / "processed"
SCEN_DIR   = BASE_DIR / "data" / "scenarios"
OUTPUT_DIR = BASE_DIR / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# FATORES REGULATÓRIOS — BASILEIA III / BCB
# ─────────────────────────────────────────────────────────────

# Haircuts HQLA (Circular BCB 3.644, Art. 5-7)
HAIRCUT_HQLA = {
    "nivel1":  0.00,   # 0% haircut — Tesouro Nacional, BCB, soberanos AAA
    "nivel2A": 0.15,   # 15% haircut — soberanos/PSE AA-, covered bonds
    "nivel2B": 0.25,   # 25% haircut — ações, RMBS AA, corp bonds A-BBB
    "nao_hqla": 1.00,  # 100% — não conta como HQLA
}

# Taxas de outflow (Cenário stress Basileia III — 30 dias)
OUTFLOW_STRESS = {
    # Depósitos varejistas estáveis (cobertos por FGC)
    "deposito_pf_estavel":      0.05,
    "deposito_pf_instavel":     0.10,
    # Depósitos varejistas — pequenas empresas
    "deposito_pme_estavel":     0.05,
    "deposito_pme_instavel":    0.10,
    # Depósitos corporativos / wholesale
    "deposito_pj_operacional":  0.25,
    "deposito_pj_nao_operac":   0.40,
    # Funding de mercado
    "cdb_varejo_estavel":       0.05,
    "cdb_varejo_instavel":      0.10,
    "cdb_wholesale":            1.00,   # 100% outflow em stress
    "lf_curto_prazo":           1.00,
    "lf_longo_prazo":           0.00,   # > 30 dias = 0 outflow
    "lci_lca_curto":            0.05,
    "lci_lca_longo":            0.00,
    "interbancario_passivo":    1.00,
    "funding_externo_curto":    1.00,
    "funding_externo_longo":    0.00,
    "outras_obrigacoes":        0.20,
}

# Taxas de inflow (cap em 75% do total outflow)
INFLOW_STRESS = {
    "credito_imobiliario":      0.00,   # hipotecas: 0% inflow (renovação esperada)
    "credito_consignado":       0.50,
    "credito_pessoal":          0.50,
    "credito_veiculos":         0.50,
    "cartao_credito":           0.00,   # assumido como renovado
    "capital_giro_pj":          0.50,
    "credito_corporativo":      1.00,   # linhas comprometidas: 0% inflow
    "trade_finance":            1.00,
    "interbancario_ativo":      1.00,
    "tvm_hqla":                 0.00,   # não conta (já no HQLA)
}

# Fatores ASF / RSF (NSFR — Circular BCB 3.648)
# ASF: Available Stable Funding (passivos + PL)
ASF_FATORES = {
    "pl_tier1":               1.00,
    "pl_tier2_longo":         1.00,
    "captacao_longo_varejo":  0.90,   # vencimento > 1 ano, varejo estável
    "captacao_longo_wp":      1.00,   # vencimento > 1 ano, wholesale
    "captacao_medio_varejo":  0.70,   # 6m–1a, varejo
    "captacao_curto_estavel": 0.50,   # < 6m, varejo estável
    "captacao_curto_wp":      0.00,   # < 6m, wholesale → 0%
    "deposito_vista_pf":      0.95,
    "deposito_vista_pj":      0.50,
    "poupanca":               0.90,
}

# RSF: Required Stable Funding (ativos)
RSF_FATORES = {
    "hqla_nivel1":            0.00,   # LFT, reservas BCB
    "hqla_nivel2a":           0.15,
    "hqla_nivel2b":           0.50,
    "credito_residual_lt1a":  0.50,   # < 1 ano
    "credito_residual_gt1a":  0.65,   # > 1 ano, PF
    "credito_pj_lt1a":        0.50,
    "credito_pj_gt1a":        0.85,
    "ativos_nao_liquidos":    1.00,
}


# ─────────────────────────────────────────────────────────────
# CLASSE PRINCIPAL
# ─────────────────────────────────────────────────────────────
class LiquidityModule:
    """Módulo completo de gestão de liquidez bancária."""

    def __init__(self):
        self.ativos   = pd.read_csv(DATA_DIR / "ativos.csv")
        self.passivos = pd.read_csv(DATA_DIR / "passivos.csv")
        with open(DATA_DIR / "patrimonio.json") as f:
            self.patrimonio = json.load(f)
        with open(SCEN_DIR / "projecoes_crescimento.json") as f:
            self.crescimento = json.load(f)
        self.cdi_base = 11.75   # % a.a.

    # ─── 1. STRESS TEST LCR ──────────────────────────────────
    def stress_test_lcr(self, cenario_stress: str = "base") -> Dict:
        """
        Calcula LCR no cenário de stress de 30 dias (Basileia III).
        Retorna dicionário com HQLA, outflows, inflows e LCR.
        """
        # ── HQLA ──────────────────────────────────────────────
        hqla_detalhado = {}
        hqla_total = 0.0

        for _, row in self.ativos.iterrows():
            nivel = row.get("hqla", "nao_hqla")
            if nivel == "nao_hqla":
                continue
            haircut = HAIRCUT_HQLA.get(nivel, 1.0)
            valor_ajustado = row["saldo"] * (1 - haircut)

            # Cap nível 2A: máx 40% do HQLA total
            # Cap nível 2B: máx 15% do HQLA total
            # (aplicado após cálculo, simplificado aqui)
            if nivel not in hqla_detalhado:
                hqla_detalhado[nivel] = {"saldo_bruto": 0, "saldo_ajustado": 0, "produtos": []}
            hqla_detalhado[nivel]["saldo_bruto"]    += row["saldo"]
            hqla_detalhado[nivel]["saldo_ajustado"] += valor_ajustado
            hqla_detalhado[nivel]["produtos"].append(row["produto"])
            hqla_total += valor_ajustado

        # Cap nível 2 (40% do HQLA ajustado)
        hqla_nivel2_max = hqla_total * 0.40
        hqla_nivel2_atual = (hqla_detalhado.get("nivel2A", {}).get("saldo_ajustado", 0) +
                             hqla_detalhado.get("nivel2B", {}).get("saldo_ajustado", 0))
        if hqla_nivel2_atual > hqla_nivel2_max:
            excesso = hqla_nivel2_atual - hqla_nivel2_max
            hqla_total -= excesso

        # ── OUTFLOWS ──────────────────────────────────────────
        outflows_detalhado = {}
        outflow_total = 0.0

        # Mapeamento produto → categoria de outflow
        mapa_outflow = {
            "Depósitos à vista PF":          ("deposito_pf_estavel",      0.60),
            "Depósitos à vista PJ":          ("deposito_pj_nao_operac",   1.00),
            "Poupança":                      ("deposito_pf_instavel",     1.00),
            "CDB CDI até 1 ano":             ("cdb_wholesale",            0.70),
            "CDB CDI 1-3 anos":              ("lf_longo_prazo",           1.00),
            "CDB IPCA":                      ("lf_longo_prazo",           1.00),
            "CDB Pré-fixado":                ("cdb_varejo_instavel",      1.00),
            "LCI CDI":                       ("lci_lca_longo",            1.00),
            "LCA IPCA":                      ("lci_lca_longo",            1.00),
            "LF CDI Longa":                  ("lf_longo_prazo",           1.00),
            "LF IPCA Longa":                 ("lf_longo_prazo",           1.00),
            "Eurobonds USD":                 ("funding_externo_longo",    1.00),
            "Empréstimos externos curto prazo": ("funding_externo_curto", 1.00),
            "Depósitos interbancários":      ("interbancario_passivo",    1.00),
            "Outras obrigações":             ("outras_obrigacoes",        1.00),
        }

        for _, row in self.passivos.iterrows():
            produto = row["produto"]
            if produto in mapa_outflow:
                cat, pct_30d = mapa_outflow[produto]
                # Proporção do saldo que vence em 30 dias
                prazo = row.get("prazo_medio", 0.5)
                proporcao_30d = min(1.0, (30/365) / max(prazo, 30/365)) * pct_30d
                taxa = OUTFLOW_STRESS.get(cat, 0.20)
                outflow = row["saldo"] * proporcao_30d * taxa
            else:
                outflow = row["saldo"] * row.get("outflow_lcr", 0.20) * 0.10
            outflows_detalhado[produto] = outflow
            outflow_total += outflow

        # ── INFLOWS ───────────────────────────────────────────
        inflow_total = 0.0
        inflows_detalhado = {}

        mapa_inflow = {
            "Crédito consignado PF":    ("credito_consignado",   0.50),
            "Crédito pessoal PF":       ("credito_pessoal",      0.50),
            "Crédito veículos PF":      ("credito_veiculos",     0.50),
            "Capital de giro PJ PME":   ("capital_giro_pj",      0.50),
            "Operações interbancárias": ("interbancario_ativo",  1.00),
        }

        for _, row in self.ativos.iterrows():
            produto = row["produto"]
            if produto in mapa_inflow:
                cat, pct_30d = mapa_inflow[produto]
                prazo = row.get("prazo_medio", 1.0)
                proporcao_30d = min(1.0, (30/365) / max(prazo, 30/365)) * pct_30d
                taxa = INFLOW_STRESS.get(cat, 0.50)
                inflow = row["saldo"] * proporcao_30d * taxa
                inflows_detalhado[produto] = inflow
                inflow_total += inflow

        # Cap de inflows em 75% dos outflows (Basileia III)
        inflow_total_capped = min(inflow_total, 0.75 * outflow_total)
        nco = outflow_total - inflow_total_capped   # Net Cash Outflow

        lcr = hqla_total / nco if nco > 0 else 999.0

        resultado = {
            "hqla_total":              hqla_total,
            "hqla_nivel1":             hqla_detalhado.get("nivel1", {}).get("saldo_ajustado", 0),
            "hqla_nivel2A":            hqla_detalhado.get("nivel2A", {}).get("saldo_ajustado", 0),
            "hqla_nivel2B":            hqla_detalhado.get("nivel2B", {}).get("saldo_ajustado", 0),
            "outflow_total":           outflow_total,
            "inflow_total":            inflow_total,
            "inflow_capped":           inflow_total_capped,
            "nco":                     nco,
            "lcr":                     lcr,
            "lcr_pct":                 lcr * 100,
            "folga_hqla":              hqla_total - nco,  # > 0 = excesso
            "outflows_detalhado":      outflows_detalhado,
            "inflows_detalhado":       inflows_detalhado,
            "hqla_detalhado":          hqla_detalhado,
        }

        # Salva
        df_out = pd.DataFrame([
            {"tipo": "outflow", "produto": k, "valor": v}
            for k, v in outflows_detalhado.items()
        ] + [
            {"tipo": "inflow", "produto": k, "valor": v}
            for k, v in inflows_detalhado.items()
        ])
        df_out.to_csv(OUTPUT_DIR / "stress_test_lcr.csv", index=False)
        return resultado

    # ─── 2. CUSTO DO BUFFER HQLA ─────────────────────────────
    def custo_buffer_hqla(self) -> pd.DataFrame:
        """
        Custo de oportunidade de manter ativos HQLA.
        Custo = (taxa_crédito_alternativo - taxa_HQLA) × saldo

        O banco "sacrifica" spread ao manter LFT em vez de emprestar.
        """
        # Taxa de referência alternativa: crédito PJ capital de giro
        taxa_ref_alternativa = self.cdi_base + 4.50   # CDI + 4.5% (capital de giro)

        rows = []
        for _, row in self.ativos.iterrows():
            nivel = row.get("hqla", "nao_hqla")
            if nivel == "nao_hqla":
                continue

            taxa_hqla = row.get("taxa_total", self.cdi_base)
            custo_oportunidade_pct = taxa_ref_alternativa - taxa_hqla
            custo_oportunidade_BRL = row["saldo"] * custo_oportunidade_pct / 100
            haircut = HAIRCUT_HQLA.get(nivel, 0)
            valor_hqla_ajustado = row["saldo"] * (1 - haircut)

            rows.append({
                "produto":                   row["produto"],
                "nivel_hqla":               nivel,
                "saldo_R$M":                row["saldo"],
                "valor_hqla_ajustado_R$M":  valor_hqla_ajustado,
                "taxa_hqla_pct":            taxa_hqla,
                "taxa_alternativa_pct":     taxa_ref_alternativa,
                "custo_oportunidade_pct":   custo_oportunidade_pct,
                "custo_oportunidade_BRL_aa": custo_oportunidade_BRL,
            })

        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_DIR / "custo_buffer_hqla.csv", index=False)
        return df

    # ─── 3. PROJEÇÃO DINÂMICA LCR/NSFR 12 MESES ─────────────
    def projecao_dinamica(self, horizonte_meses: int = 12) -> pd.DataFrame:
        """
        Projeta LCR e NSFR mês a mês sob crescimento de carteira.
        Simula 3 cenários de crescimento (base, stress_alta, stress_baixa).
        """
        resultados = []

        # Taxas mensais de crescimento por cenário
        crescimento_mensal = {}
        for cen in ["base", "stress_alta", "stress_baixa"]:
            taxas = self.crescimento[cen]
            media_cred = np.mean(list(taxas.values())) / 100 / 12
            crescimento_mensal[cen] = media_cred

        for cen in ["base", "stress_alta", "stress_baixa"]:
            g = crescimento_mensal[cen]

            # NSFR base
            asf_base = (self.passivos["saldo"] * self.passivos["asf_fator"]).sum()
            asf_base += (self.patrimonio["capital_principal"] +
                        self.patrimonio["capital_complementar"] +
                        self.patrimonio["patrimonio_referencia_t2"])
            rsf_base = (self.ativos["saldo"] * self.ativos["rsf_fator"]).sum()

            # HQLA e outflows base
            hqla_base = sum(
                row["saldo"] * (1 - HAIRCUT_HQLA.get(row["hqla"], 1.0))
                for _, row in self.ativos.iterrows()
                if row["hqla"] != "nao_hqla"
            )
            outflow_base = (self.passivos["saldo"] * self.passivos["outflow_lcr"]).sum()

            for mes in range(1, horizonte_meses + 1):
                fator = (1 + g) ** mes

                # Crédito cresce → RSF sobe, HQLA inalterado (HQLA não cresce com crédito)
                rsf_mes = rsf_base * fator
                # Captação cresce junto (funding segue crédito)
                asf_mes = asf_base * fator
                nsfr_mes = asf_mes / rsf_mes if rsf_mes > 0 else 99.0

                # LCR: outflows crescem com captação
                outflow_mes = outflow_base * fator
                # HQLA cresce mais devagar (gestão ativa)
                hqla_mes = hqla_base * (1 + g * 0.5) ** mes
                nco_mes = outflow_mes * 0.70   # NCO ≈ 70% outflow bruto
                lcr_mes = hqla_mes / nco_mes if nco_mes > 0 else 99.0

                resultados.append({
                    "cenario":   cen,
                    "mes":       mes,
                    "lcr_pct":   lcr_mes * 100,
                    "nsfr_pct":  nsfr_mes * 100,
                    "hqla_R$M":  hqla_mes,
                    "nco_R$M":   nco_mes,
                    "asf_R$M":   asf_mes,
                    "rsf_R$M":   rsf_mes,
                    "folga_lcr_pct":  (lcr_mes - 1.0) * 100,
                    "folga_nsfr_pct": (nsfr_mes - 1.0) * 100,
                })

        df = pd.DataFrame(resultados)
        df.to_csv(OUTPUT_DIR / "projecao_lcr_nsfr.csv", index=False)
        return df

    # ─── NSFR DETALHADO ──────────────────────────────────────
    def calcular_nsfr_detalhado(self) -> Dict:
        """Calcula NSFR com decomposição ASF/RSF por categoria."""
        asf_rows, rsf_rows = [], []

        # ASF — Passivos
        for _, row in self.passivos.iterrows():
            asf_rows.append({
                "item": row["produto"], "tipo": "passivo",
                "saldo": row["saldo"],
                "fator_asf": row["asf_fator"],
                "asf": row["saldo"] * row["asf_fator"],
            })
        # ASF — Patrimônio (100%)
        for k, v in self.patrimonio.items():
            if k in ("capital_principal", "capital_complementar",
                     "patrimonio_referencia_t2", "reservas_lucros_retidos"):
                asf_rows.append({
                    "item": k, "tipo": "patrimonio",
                    "saldo": v, "fator_asf": 1.00, "asf": v,
                })

        # RSF — Ativos
        for _, row in self.ativos.iterrows():
            rsf_rows.append({
                "item": row["produto"], "tipo": row["categoria"],
                "saldo": row["saldo"],
                "fator_rsf": row["rsf_fator"],
                "rsf": row["saldo"] * row["rsf_fator"],
            })

        df_asf = pd.DataFrame(asf_rows)
        df_rsf = pd.DataFrame(rsf_rows)

        asf_total = df_asf["asf"].sum()
        rsf_total = df_rsf["rsf"].sum()
        nsfr = asf_total / rsf_total if rsf_total > 0 else 99.0

        df_asf.to_csv(OUTPUT_DIR / "nsfr_asf_detalhado.csv", index=False)
        df_rsf.to_csv(OUTPUT_DIR / "nsfr_rsf_detalhado.csv", index=False)

        return {
            "asf_total": asf_total, "rsf_total": rsf_total,
            "nsfr": nsfr, "nsfr_pct": nsfr * 100,
            "folga_R$M": asf_total - rsf_total,
            "df_asf": df_asf, "df_rsf": df_rsf,
        }

    def imprimir_lcr(self, resultado: Dict):
        print(f"\n{'='*58}")
        print(f"  STRESS TEST LCR — 30 DIAS (Basileia III)")
        print(f"{'='*58}")
        print(f"  HQLA Total:         R$ {resultado['hqla_total']:>10,.0f}M")
        print(f"    Nível 1:          R$ {resultado['hqla_nivel1']:>10,.0f}M")
        print(f"    Nível 2A (−15%):  R$ {resultado['hqla_nivel2A']:>10,.0f}M")
        print(f"    Nível 2B (−25%):  R$ {resultado['hqla_nivel2B']:>10,.0f}M")
        print(f"  Outflows brutos:    R$ {resultado['outflow_total']:>10,.0f}M")
        print(f"  Inflows (cap 75%):  R$ {resultado['inflow_capped']:>10,.0f}M")
        print(f"  NCO:                R$ {resultado['nco']:>10,.0f}M")
        print(f"  {'─'*40}")
        lcr_str = f"{resultado['lcr_pct']:.1f}%"
        status = "✅ APROVADO" if resultado['lcr'] >= 1.0 else "❌ VIOLADO"
        print(f"  LCR:                   {lcr_str:>8}   {status}")
        print(f"  Folga HQLA:         R$ {resultado['folga_hqla']:>10,.0f}M")
        print(f"{'='*58}")
