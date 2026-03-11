"""
M1 - Balance Sheet Module
=========================
Balanço Patrimonial de Banco Maduro Brasileiro (fictício/calibrado com dados reais)
Referência: Estrutura de grandes bancos brasileiros (BACEN, IF.data)

Valores em R$ milhões | Data-base: Dez/2024
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "tables"
CHARTS_DIR = BASE_DIR / "outputs" / "charts"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# CONSTANTES DO BANCO
# ─────────────────────────────────────────────
BANCO_NOME = "Banco Modelo S.A."
DATA_BASE = "2024-12-31"
MOEDA = "BRL"
UNIDADE = "R$ milhões"

# Indexadores disponíveis
class Indexador:
    PRE   = "PRE"      # Taxa pré-fixada
    CDI   = "CDI"      # CDI / SELIC
    IPCA  = "IPCA"     # IPCA + spread
    TR    = "TR"       # TR (poupança, FGTS, imobiliário)
    IGPM  = "IGPM"     # IGP-M
    USD   = "USD"      # Dólar (hedge cambial)

# FPR - Fatores de Ponderação de Risco (Basileia III Brasil — Res. BCB 4.958/2021)
FPR = {
    "Disponibilidades":             0.00,
    "Compulsório BCB":              0.00,
    "Títulos LFT":                  0.00,
    "Títulos LTN":                  0.00,
    "Títulos NTN-B":                0.00,
    "Títulos NTN-F":                0.00,
    "CDB outros bancos":            0.20,
    "Debêntures AAA":               0.20,
    "Debêntures A":                 0.50,
    "Crédito imobiliário PF":       0.35,
    "Crédito consignado PF":        0.75,
    "Crédito pessoal PF":           0.75,
    "Crédito veículos PF":          0.75,
    "Cartão crédito PF":            1.00,
    "Capital de giro PJ PME":       0.85,
    "Crédito corporativo PJ":       1.00,
    "Trade finance":                0.20,
    "Crédito rural":                0.75,
    "Operações interbancárias":     0.20,
    "Outros ativos":                1.00,
}

# ─────────────────────────────────────────────
# CLASSE PRINCIPAL DO BALANÇO
# ─────────────────────────────────────────────
class BalancoPatrimonial:
    """
    Representa o balanço patrimonial completo de um banco maduro.
    Cada produto tem: saldo, indexador, prazo médio, spread, tipo de taxa (pré/pós),
    FPR (para RWA), classificação HQLA (para LCR) e ASF/RSF (para NSFR).
    """

    def __init__(self):
        self.data_base = DATA_BASE
        self.ativos = self._build_ativos()
        self.passivos = self._build_passivos()
        self.patrimonio = self._build_patrimonio()
        self._validar_balanco()

    # ─── ATIVOS ───────────────────────────────
    def _build_ativos(self) -> pd.DataFrame:
        """
        Constrói o ativo do balanço.
        prazo_medio: em anos
        spread: em % a.a. (acima do indexador)
        taxa_total: taxa efetiva total (%)
        fpr: fator de ponderação de risco
        hqla: nível de ativos de alta liquidez (0=não HQLA, 1=nível1, 2A, 2B)
        asf_fator: fator ASF para NSFR (Available Stable Funding)
        """
        dados = [
            # ── CAIXA E EQUIV ──────────────────────────────────────────────────────────────
            {
                "categoria": "Caixa e Equivalentes",
                "produto": "Disponibilidades",
                "saldo": 8_500,
                "indexador": Indexador.PRE,
                "prazo_medio": 0.0,
                "spread": 0.0,
                "taxa_total": 0.0,
                "tipo_taxa": "pre",
                "fpr": FPR["Disponibilidades"],
                "hqla": "nivel1",
                "rsf_fator": 0.00,   # RSF (Required Stable Funding) para NSFR
            },
            {
                "categoria": "Caixa e Equivalentes",
                "produto": "Compulsório BCB",
                "saldo": 32_000,
                "indexador": Indexador.CDI,
                "prazo_medio": 0.0,
                "spread": 0.0,
                "taxa_total": 11.75,
                "tipo_taxa": "pos",
                "fpr": FPR["Compulsório BCB"],
                "hqla": "nivel1",
                "rsf_fator": 0.00,
            },

            # ── CARTEIRA DE TÍTULOS ─────────────────────────────────────────────────────────
            {
                "categoria": "Títulos e Valores Mobiliários",
                "produto": "Títulos LFT",
                "saldo": 45_000,
                "indexador": Indexador.CDI,
                "prazo_medio": 2.5,
                "spread": 0.02,
                "taxa_total": 11.77,
                "tipo_taxa": "pos",
                "fpr": FPR["Títulos LFT"],
                "hqla": "nivel1",
                "rsf_fator": 0.05,
            },
            {
                "categoria": "Títulos e Valores Mobiliários",
                "produto": "Títulos LTN",
                "saldo": 18_000,
                "indexador": Indexador.PRE,
                "prazo_medio": 1.8,
                "spread": 0.0,
                "taxa_total": 13.20,
                "tipo_taxa": "pre",
                "fpr": FPR["Títulos LTN"],
                "hqla": "nivel1",
                "rsf_fator": 0.05,
            },
            {
                "categoria": "Títulos e Valores Mobiliários",
                "produto": "Títulos NTN-B",
                "saldo": 28_000,
                "indexador": Indexador.IPCA,
                "prazo_medio": 6.2,
                "spread": 0.0,
                "taxa_total": 6.45,   # IPCA + 6.45% a.a.
                "tipo_taxa": "pos",
                "fpr": FPR["Títulos NTN-B"],
                "hqla": "nivel1",
                "rsf_fator": 0.05,
            },
            {
                "categoria": "Títulos e Valores Mobiliários",
                "produto": "Títulos NTN-F",
                "saldo": 9_500,
                "indexador": Indexador.PRE,
                "prazo_medio": 4.1,
                "spread": 0.0,
                "taxa_total": 13.85,
                "tipo_taxa": "pre",
                "fpr": FPR["Títulos NTN-F"],
                "hqla": "nivel1",
                "rsf_fator": 0.05,
            },
            {
                "categoria": "Títulos e Valores Mobiliários",
                "produto": "CDB outros bancos",
                "saldo": 6_200,
                "indexador": Indexador.CDI,
                "prazo_medio": 0.8,
                "spread": 0.15,
                "taxa_total": 11.90,
                "tipo_taxa": "pos",
                "fpr": FPR["CDB outros bancos"],
                "hqla": "nivel2A",
                "rsf_fator": 0.15,
            },
            {
                "categoria": "Títulos e Valores Mobiliários",
                "produto": "Debêntures AAA",
                "saldo": 4_800,
                "indexador": Indexador.CDI,
                "prazo_medio": 3.5,
                "spread": 0.80,
                "taxa_total": 12.55,
                "tipo_taxa": "pos",
                "fpr": FPR["Debêntures AAA"],
                "hqla": "nivel2A",
                "rsf_fator": 0.20,
            },
            {
                "categoria": "Títulos e Valores Mobiliários",
                "produto": "Debêntures A",
                "saldo": 3_100,
                "indexador": Indexador.CDI,
                "prazo_medio": 4.0,
                "spread": 1.40,
                "taxa_total": 13.15,
                "tipo_taxa": "pos",
                "fpr": FPR["Debêntures A"],
                "hqla": "nivel2B",
                "rsf_fator": 0.50,
            },

            # ── CARTEIRA DE CRÉDITO ─────────────────────────────────────────────────────────
            {
                "categoria": "Crédito PF",
                "produto": "Crédito imobiliário PF",
                "saldo": 85_000,
                "indexador": Indexador.TR,
                "prazo_medio": 18.0,
                "spread": 8.20,
                "taxa_total": 10.50,   # TR + 8.20% a.a. ≈ 10.5% efetiva
                "tipo_taxa": "pos",
                "fpr": FPR["Crédito imobiliário PF"],
                "hqla": "nao_hqla",
                "rsf_fator": 0.65,
            },
            {
                "categoria": "Crédito PF",
                "produto": "Crédito consignado PF",
                "saldo": 62_000,
                "indexador": Indexador.PRE,
                "prazo_medio": 3.8,
                "spread": 0.0,
                "taxa_total": 22.50,
                "tipo_taxa": "pre",
                "fpr": FPR["Crédito consignado PF"],
                "hqla": "nao_hqla",
                "rsf_fator": 0.65,
            },
            {
                "categoria": "Crédito PF",
                "produto": "Crédito pessoal PF",
                "saldo": 24_000,
                "indexador": Indexador.PRE,
                "prazo_medio": 2.2,
                "spread": 0.0,
                "taxa_total": 38.40,
                "tipo_taxa": "pre",
                "fpr": FPR["Crédito pessoal PF"],
                "hqla": "nao_hqla",
                "rsf_fator": 0.65,
            },
            {
                "categoria": "Crédito PF",
                "produto": "Crédito veículos PF",
                "saldo": 31_000,
                "indexador": Indexador.PRE,
                "prazo_medio": 4.5,
                "spread": 0.0,
                "taxa_total": 26.80,
                "tipo_taxa": "pre",
                "fpr": FPR["Crédito veículos PF"],
                "hqla": "nao_hqla",
                "rsf_fator": 0.65,
            },
            {
                "categoria": "Crédito PF",
                "produto": "Cartão crédito PF",
                "saldo": 19_500,
                "indexador": Indexador.PRE,
                "prazo_medio": 0.5,
                "spread": 0.0,
                "taxa_total": 189.00,  # rotativo efetivo
                "tipo_taxa": "pre",
                "fpr": FPR["Cartão crédito PF"],
                "hqla": "nao_hqla",
                "rsf_fator": 0.85,
            },
            {
                "categoria": "Crédito PJ",
                "produto": "Capital de giro PJ PME",
                "saldo": 38_000,
                "indexador": Indexador.CDI,
                "prazo_medio": 1.5,
                "spread": 4.50,
                "taxa_total": 16.25,
                "tipo_taxa": "pos",
                "fpr": FPR["Capital de giro PJ PME"],
                "hqla": "nao_hqla",
                "rsf_fator": 0.85,
            },
            {
                "categoria": "Crédito PJ",
                "produto": "Crédito corporativo PJ",
                "saldo": 52_000,
                "indexador": Indexador.CDI,
                "prazo_medio": 3.2,
                "spread": 2.80,
                "taxa_total": 14.55,
                "tipo_taxa": "pos",
                "fpr": FPR["Crédito corporativo PJ"],
                "hqla": "nao_hqla",
                "rsf_fator": 0.85,
            },
            {
                "categoria": "Crédito PJ",
                "produto": "Trade finance",
                "saldo": 14_500,
                "indexador": Indexador.USD,
                "prazo_medio": 0.7,
                "spread": 2.20,
                "taxa_total": 7.40,    # libor/sofr + spread em USD
                "tipo_taxa": "pos",
                "fpr": FPR["Trade finance"],
                "hqla": "nao_hqla",
                "rsf_fator": 0.50,
            },
            {
                "categoria": "Crédito PJ",
                "produto": "Crédito rural",
                "saldo": 16_800,
                "indexador": Indexador.PRE,
                "prazo_medio": 2.8,
                "spread": 0.0,
                "taxa_total": 10.50,   # taxas subsidiadas MCR
                "tipo_taxa": "pre",
                "fpr": FPR["Crédito rural"],
                "hqla": "nao_hqla",
                "rsf_fator": 0.65,
            },

            # ── INTERBANCÁRIO E OUTROS ──────────────────────────────────────────────────────
            {
                "categoria": "Interbancário",
                "produto": "Operações interbancárias",
                "saldo": 11_200,
                "indexador": Indexador.CDI,
                "prazo_medio": 0.1,
                "spread": 0.05,
                "taxa_total": 11.80,
                "tipo_taxa": "pos",
                "fpr": FPR["Operações interbancárias"],
                "hqla": "nivel1",
                "rsf_fator": 0.10,
            },
            {
                "categoria": "Outros Ativos",
                "produto": "Outros ativos",
                "saldo": 22_900,
                "indexador": Indexador.PRE,
                "prazo_medio": 1.0,
                "spread": 0.0,
                "taxa_total": 0.0,
                "tipo_taxa": "pre",
                "fpr": FPR["Outros ativos"],
                "hqla": "nao_hqla",
                "rsf_fator": 1.00,
            },
        ]

        df = pd.DataFrame(dados)
        df["tipo"] = "Ativo"
        df["rwa"] = df["saldo"] * df["fpr"]
        return df

    # ─── PASSIVOS ─────────────────────────────
    def _build_passivos(self) -> pd.DataFrame:
        """
        Constrói o passivo do balanço.
        asf_fator: fator ASF para NSFR (Available Stable Funding)
        outflow_taxa: taxa de outflow para LCR (Basileia III)
        """
        dados = [
            # ── DEPÓSITOS À VISTA E POUPANÇA ───────────────────────────────────────────────
            {
                "categoria": "Depósitos",
                "produto": "Depósitos à vista PF",
                "saldo": 28_000,
                "indexador": Indexador.PRE,
                "prazo_medio": 0.0,
                "custo": 0.0,
                "tipo_taxa": "pre",
                "estabilidade": "estavel",
                "asf_fator": 0.95,   # 95% ASF — depósito estável varejo
                "outflow_lcr": 0.05,  # 5% outflow no cenário de stress
            },
            {
                "categoria": "Depósitos",
                "produto": "Depósitos à vista PJ",
                "saldo": 18_500,
                "indexador": Indexador.PRE,
                "prazo_medio": 0.0,
                "custo": 0.0,
                "tipo_taxa": "pre",
                "estabilidade": "instavel",
                "asf_fator": 0.50,
                "outflow_lcr": 0.40,  # 40% outflow (cliente corporativo)
            },
            {
                "categoria": "Depósitos",
                "produto": "Poupança",
                "saldo": 42_000,
                "indexador": Indexador.TR,
                "prazo_medio": 0.0,
                "custo": 6.17,        # TR + 6.17% a.a. (regra antiga > 50% meta Selic)
                "tipo_taxa": "pos",
                "estabilidade": "estavel",
                "asf_fator": 0.90,
                "outflow_lcr": 0.10,
            },

            # ── CDB ─────────────────────────────────────────────────────────────────────────
            {
                "categoria": "CDB",
                "produto": "CDB CDI até 1 ano",
                "saldo": 55_000,
                "indexador": Indexador.CDI,
                "prazo_medio": 0.6,
                "custo": 98.5,        # % do CDI
                "tipo_taxa": "pos_pct_cdi",
                "estabilidade": "instavel",
                "asf_fator": 0.50,
                "outflow_lcr": 0.20,
            },
            {
                "categoria": "CDB",
                "produto": "CDB CDI 1-3 anos",
                "saldo": 38_000,
                "indexador": Indexador.CDI,
                "prazo_medio": 1.8,
                "custo": 102.0,
                "tipo_taxa": "pos_pct_cdi",
                "estabilidade": "semi_estavel",
                "asf_fator": 0.90,   # vence > 1 ano = 90% ASF
                "outflow_lcr": 0.10,
            },
            {
                "categoria": "CDB",
                "produto": "CDB IPCA",
                "saldo": 12_500,
                "indexador": Indexador.IPCA,
                "prazo_medio": 3.2,
                "custo": 7.20,        # IPCA + 7.20% a.a.
                "tipo_taxa": "pos",
                "estabilidade": "semi_estavel",
                "asf_fator": 0.90,
                "outflow_lcr": 0.05,
            },
            {
                "categoria": "CDB",
                "produto": "CDB Pré-fixado",
                "saldo": 9_800,
                "indexador": Indexador.PRE,
                "prazo_medio": 1.1,
                "custo": 13.50,
                "tipo_taxa": "pre",
                "estabilidade": "semi_estavel",
                "asf_fator": 0.50,
                "outflow_lcr": 0.15,
            },

            # ── LCI / LCA ────────────────────────────────────────────────────────────────────
            {
                "categoria": "LCI/LCA",
                "produto": "LCI CDI",
                "saldo": 24_000,
                "indexador": Indexador.CDI,
                "prazo_medio": 1.4,
                "custo": 92.0,        # isenção IR torna mais barato para o banco
                "tipo_taxa": "pos_pct_cdi",
                "estabilidade": "semi_estavel",
                "asf_fator": 0.90,
                "outflow_lcr": 0.05,
            },
            {
                "categoria": "LCI/LCA",
                "produto": "LCA IPCA",
                "saldo": 16_200,
                "indexador": Indexador.IPCA,
                "prazo_medio": 2.1,
                "custo": 6.50,
                "tipo_taxa": "pos",
                "estabilidade": "semi_estavel",
                "asf_fator": 0.90,
                "outflow_lcr": 0.05,
            },

            # ── LETRAS FINANCEIRAS ────────────────────────────────────────────────────────────
            {
                "categoria": "Letras Financeiras",
                "produto": "LF CDI Longa",
                "saldo": 32_000,
                "indexador": Indexador.CDI,
                "prazo_medio": 4.5,
                "custo": 105.0,       # % do CDI
                "tipo_taxa": "pos_pct_cdi",
                "estabilidade": "estavel",
                "asf_fator": 1.00,   # prazo > 1 ano = 100% ASF
                "outflow_lcr": 0.00,
            },
            {
                "categoria": "Letras Financeiras",
                "produto": "LF IPCA Longa",
                "saldo": 18_000,
                "indexador": Indexador.IPCA,
                "prazo_medio": 6.0,
                "custo": 6.80,
                "tipo_taxa": "pos",
                "estabilidade": "estavel",
                "asf_fator": 1.00,
                "outflow_lcr": 0.00,
            },

            # ── FUNDING EXTERNO ───────────────────────────────────────────────────────────────
            {
                "categoria": "Funding Externo",
                "produto": "Eurobonds USD",
                "saldo": 14_000,
                "indexador": Indexador.USD,
                "prazo_medio": 5.2,
                "custo": 6.80,        # SOFR + spread
                "tipo_taxa": "pos",
                "estabilidade": "estavel",
                "asf_fator": 1.00,
                "outflow_lcr": 0.00,
            },
            {
                "categoria": "Funding Externo",
                "produto": "Empréstimos externos curto prazo",
                "saldo": 6_800,
                "indexador": Indexador.USD,
                "prazo_medio": 0.5,
                "custo": 5.90,
                "tipo_taxa": "pos",
                "estabilidade": "instavel",
                "asf_fator": 0.00,   # < 6 meses = 0% ASF
                "outflow_lcr": 1.00,  # 100% outflow
            },

            # ── INTERBANCÁRIO PASSIVO ─────────────────────────────────────────────────────────
            {
                "categoria": "Interbancário Passivo",
                "produto": "Depósitos interbancários",
                "saldo": 8_500,
                "indexador": Indexador.CDI,
                "prazo_medio": 0.2,
                "custo": 100.0,       # % CDI
                "tipo_taxa": "pos_pct_cdi",
                "estabilidade": "instavel",
                "asf_fator": 0.00,
                "outflow_lcr": 1.00,
            },

            # ── OUTRAS OBRIGAÇÕES ──────────────────────────────────────────────────────────────
            {
                "categoria": "Outras Obrigações",
                "produto": "Outras obrigações",
                "saldo": 16_200,
                "indexador": Indexador.PRE,
                "prazo_medio": 0.5,
                "custo": 0.0,
                "tipo_taxa": "pre",
                "estabilidade": "instavel",
                "asf_fator": 0.00,
                "outflow_lcr": 0.20,
            },
        ]

        df = pd.DataFrame(dados)
        df["tipo"] = "Passivo"
        df["rwa"] = 0.0   # passivos não geram RWA diretamente
        return df

    # ─── PATRIMÔNIO ───────────────────────────
    def _build_patrimonio(self) -> dict:
        # Total Ativo = 532.000 | Total Passivo = 339.500
        # Gap = 192.500 → distribuído em PL para fechar o balanço
        # Capital Principal (CET1) + Reservas + Resultados retidos
        return {
            "capital_principal":          28_500,   # CET1 — capital integralizado
            "capital_complementar":        3_200,   # AT1 — instrumentos híbridos
            "patrimonio_referencia_t2":    6_100,   # Tier 2 — dívida subordinada
            "reservas_lucros_retidos":   153_100,   # reservas + lucros acumulados
            "ajuste_avaliacao_patrimonial": 1_200,  # AAP positivo (MtM títulos)
            "patrimonio_referencia":      37_800,   # PR regulatório (CET1+AT1+T2)
        }

    # ─── VALIDAÇÃO ────────────────────────────
    def _validar_balanco(self):
        total_ativo   = self.ativos["saldo"].sum()
        total_passivo = self.passivos["saldo"].sum()
        total_pl      = (self.patrimonio["capital_principal"]
                       + self.patrimonio["capital_complementar"]
                       + self.patrimonio["patrimonio_referencia_t2"]
                       + self.patrimonio["reservas_lucros_retidos"]
                       + self.patrimonio["ajuste_avaliacao_patrimonial"])
        gap = total_ativo - (total_passivo + total_pl)
        print(f"\n{'='*55}")
        print(f"  {BANCO_NOME} — Balanço em {DATA_BASE}")
        print(f"{'='*55}")
        print(f"  Total Ativo:     R$ {total_ativo:>10,.0f} M")
        print(f"  Total Passivo:   R$ {total_passivo:>10,.0f} M")
        print(f"  Patrimônio Líq:  R$ {total_pl:>10,.0f} M")
        print(f"  Gap (deve=0):    R$ {gap:>10,.0f} M")
        if abs(gap) > 500:
            print(f"  ⚠️  ATENÇÃO: Balanço desbalanceado em R$ {gap:.0f}M")
        else:
            print(f"  ✅ Balanço equilibrado")
        print(f"{'='*55}\n")

    # ─── MÉTRICAS DERIVADAS ───────────────────
    def total_ativo(self) -> float:
        return self.ativos["saldo"].sum()

    def total_passivo(self) -> float:
        return self.passivos["saldo"].sum()

    def total_pl(self) -> float:
        return (self.patrimonio["capital_principal"]
              + self.patrimonio["capital_complementar"]
              + self.patrimonio["patrimonio_referencia_t2"]
              + self.patrimonio["reservas_lucros_retidos"]
              + self.patrimonio["ajuste_avaliacao_patrimonial"])

    def rwa_total(self) -> float:
        return self.ativos["rwa"].sum()

    def indice_basileia(self) -> float:
        """IB = PR / RWA × 100 (mínimo regulatório: 10.5% no Brasil)"""
        return (self.patrimonio["patrimonio_referencia"] / self.rwa_total()) * 100

    def cet1_ratio(self) -> float:
        """CET1 = Capital Principal / RWA (mínimo: 7.0% com buffers)"""
        return (self.patrimonio["capital_principal"] / self.rwa_total()) * 100

    def resumo_kpis(self) -> dict:
        total_ativo = self.total_ativo()
        total_credito = self.ativos[
            self.ativos["categoria"].isin(["Crédito PF", "Crédito PJ"])
        ]["saldo"].sum()
        total_tvm = self.ativos[
            self.ativos["categoria"] == "Títulos e Valores Mobiliários"
        ]["saldo"].sum()

        return {
            "total_ativo":          total_ativo,
            "total_passivo":        self.total_passivo(),
            "patrimonio_liquido":   self.total_pl(),
            "total_credito":        total_credito,
            "total_tvm":            total_tvm,
            "rwa_total":            self.rwa_total(),
            "indice_basileia_pct":  self.indice_basileia(),
            "cet1_ratio_pct":       self.cet1_ratio(),
            "leverage_x":           total_ativo / self.total_pl(),
            "credito_ativo_pct":    total_credito / total_ativo * 100,
        }

    def salvar(self):
        """Salva dados processados em CSV e JSON."""
        self.ativos.to_csv(DATA_DIR / "ativos.csv", index=False)
        self.passivos.to_csv(DATA_DIR / "passivos.csv", index=False)
        with open(DATA_DIR / "patrimonio.json", "w") as f:
            json.dump(self.patrimonio, f, indent=2)
        kpis = self.resumo_kpis()
        with open(DATA_DIR / "kpis_balanco.json", "w") as f:
            json.dump({k: round(float(v), 4) for k, v in kpis.items()}, f, indent=2)
        print(f"✅ Dados salvos em: {DATA_DIR}")


# ─────────────────────────────────────────────
# EXECUÇÃO DIRETA
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted")

    banco = BalancoPatrimonial()
    banco.salvar()

    kpis = banco.resumo_kpis()
    print("\n📊 KPIs do Balanço:")
    for k, v in kpis.items():
        print(f"   {k:<30} {v:>12.2f}")

    # ── GRÁFICO 1: Composição do Ativo ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{BANCO_NOME}\nComposição do Balanço — {DATA_BASE}", fontsize=13, fontweight="bold")

    ativo_cat = banco.ativos.groupby("categoria")["saldo"].sum().sort_values(ascending=False)
    colors_a = sns.color_palette("Blues_d", len(ativo_cat))
    axes[0].barh(ativo_cat.index, ativo_cat.values / 1000, color=colors_a)
    axes[0].set_title("Ativo por Categoria (R$ bi)", fontweight="bold")
    axes[0].set_xlabel("R$ bilhões")
    for i, (cat, val) in enumerate(ativo_cat.items()):
        axes[0].text(val/1000 + 0.3, i, f"R${val/1000:.1f}bi", va="center", fontsize=8)

    passivo_cat = banco.passivos.groupby("categoria")["saldo"].sum().sort_values(ascending=False)
    colors_p = sns.color_palette("Oranges_d", len(passivo_cat))
    axes[1].barh(passivo_cat.index, passivo_cat.values / 1000, color=colors_p)
    axes[1].set_title("Passivo por Categoria (R$ bi)", fontweight="bold")
    axes[1].set_xlabel("R$ bilhões")
    for i, (cat, val) in enumerate(passivo_cat.items()):
        axes[1].text(val/1000 + 0.2, i, f"R${val/1000:.1f}bi", va="center", fontsize=8)

    plt.tight_layout()
    path_g1 = CHARTS_DIR / "01_composicao_balanco.png"
    plt.savefig(path_g1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Gráfico salvo: {path_g1}")

    # ── GRÁFICO 2: RWA e Capital ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    rwa_cat = banco.ativos.groupby("categoria")["rwa"].sum().sort_values(ascending=False)
    rwa_cat = rwa_cat[rwa_cat > 0]
    colors_r = sns.color_palette("Reds_d", len(rwa_cat))
    bars = ax.bar(range(len(rwa_cat)), rwa_cat.values / 1000, color=colors_r)
    ax.set_xticks(range(len(rwa_cat)))
    ax.set_xticklabels(rwa_cat.index, rotation=35, ha="right", fontsize=9)
    ax.set_title(f"RWA por Categoria — IB: {kpis['indice_basileia_pct']:.1f}% | CET1: {kpis['cet1_ratio_pct']:.1f}%",
                 fontweight="bold")
    ax.set_ylabel("R$ bilhões")
    ax.axhline(y=0, color="black", linewidth=0.8)
    for bar, val in zip(bars, rwa_cat.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val/1000:.1f}bi", ha="center", fontsize=8)
    plt.tight_layout()
    path_g2 = CHARTS_DIR / "02_rwa_capital.png"
    plt.savefig(path_g2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Gráfico salvo: {path_g2}")

    # ── GRÁFICO 3: Mix de indexadores ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Mix de Indexadores — Ativo vs Passivo", fontsize=12, fontweight="bold")

    idx_ativo = banco.ativos.groupby("indexador")["saldo"].sum()
    idx_passivo = banco.passivos.groupby("indexador")["saldo"].sum()
    palette = {"CDI": "#2196F3", "PRE": "#4CAF50", "IPCA": "#FF9800", "TR": "#9C27B0", "USD": "#F44336", "IGPM": "#795548"}
    colors_a2 = [palette.get(i, "#888") for i in idx_ativo.index]
    colors_p2 = [palette.get(i, "#888") for i in idx_passivo.index]

    axes[0].pie(idx_ativo.values, labels=idx_ativo.index, autopct="%1.1f%%",
                colors=colors_a2, startangle=90)
    axes[0].set_title("Ativo")
    axes[1].pie(idx_passivo.values, labels=idx_passivo.index, autopct="%1.1f%%",
                colors=colors_p2, startangle=90)
    axes[1].set_title("Passivo")

    plt.tight_layout()
    path_g3 = CHARTS_DIR / "03_mix_indexadores.png"
    plt.savefig(path_g3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Gráfico salvo: {path_g3}")

    print("\n🏁 M1 — Balance Sheet concluído com sucesso!")
