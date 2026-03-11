"""
M4a - Market Environment
=========================
Define o "mercado" disponível para o banco em cada rodada de otimização.

Modela:
  1. Curvas de demanda de crédito por segmento (elasticidade spread-volume)
  2. Curvas de oferta de captação por instrumento (custo sobe com volume)
  3. Mercado de títulos públicos (LFT, LTN, NTN-B) com volumes e preços
  4. Instrumentos de hedge (swaps DI×IPCA, DI×Pré, Futuro DI)

Referência metodológica:
  - Elasticidade de demanda de crédito: Klein (1971), Monti (1972)
  - Modelo Monti-Klein de banco: spread ótimo com poder de mercado
  - Curvas de funding: Drechsler, Savov & Schnabl (2017)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCEN_DIR = BASE_DIR / "data" / "scenarios"


# ─────────────────────────────────────────────────────────────
# ESTRUTURAS DE DADOS
# ─────────────────────────────────────────────────────────────

@dataclass
class SegmentoCredito:
    """
    Segmento de crédito com curva de demanda elástica.

    Modelo: Volume_max(spread) = vol_base × exp(-elasticidade × (spread - spread_min))
    Interpretação: demanda cai exponencialmente conforme spread sobe.

    Atributos:
        nome: identificador do segmento
        spread_min: spread mínimo aceito pelo banco (floor de rentabilidade), % a.a.
        spread_mercado: spread corrente de mercado (referência), % a.a.
        spread_max: spread máximo que o mercado absorve, % a.a.
        vol_base: volume disponível no spread de mercado, R$ milhões
        elasticidade: sensibilidade volume/spread (>0: demanda cai com spread)
        fpr: fator de ponderação de risco (Basileia)
        indexador: CDI, PRE, IPCA, TR
        prazo_medio: anos
        hqla: classificação HQLA para LCR
        rsf_fator: fator RSF para NSFR
        pd_anual: probabilidade de default anual (para RAROC)
        lgd: loss given default
    """
    nome: str
    spread_min: float
    spread_mercado: float
    spread_max: float
    vol_base: float           # R$ milhões disponíveis ao spread de mercado
    elasticidade: float       # β na curva exp(-β × Δspread)
    fpr: float
    indexador: str
    prazo_medio: float
    hqla: str
    rsf_fator: float
    pd_anual: float           # % a.a.
    lgd: float                # %

    def volume_disponivel(self, spread: float) -> float:
        """Volume que o mercado absorve ao spread dado."""
        delta = spread - self.spread_mercado
        vol = self.vol_base * np.exp(-self.elasticidade * delta)
        return max(0.0, min(vol, self.vol_base * 2.5))  # cap em 2.5x o base


@dataclass
class InstrumentoCaptacao:
    """
    Instrumento de captação com curva de custo crescente.

    Modelo: custo(vol) = custo_base + γ × (vol / vol_ref)^δ
    Interpretação: quanto mais o banco quer captar, mais caro fica
                   (investidores exigem prêmio por concentração).

    δ = 0.5: custo sobe com raiz quadrada do volume (côncavo — razoável)
    """
    nome: str
    custo_base: float         # % a.a. ou % CDI
    tipo_custo: str           # "spread_cdi_pct", "spread_ipca", "taxa_pre"
    vol_referencia: float     # volume de referência (R$ milhões)
    gamma: float              # sensibilidade custo-volume
    delta: float              # expoente da curva (0.5 padrão)
    vol_max: float            # teto de captação disponível no mercado
    prazo_medio: float        # anos
    indexador: str
    asf_fator: float          # fator ASF para NSFR
    outflow_lcr: float        # fator de outflow para LCR

    def custo_efetivo(self, vol: float, cdi_cenario: float = 11.75) -> float:
        """Retorna custo efetivo (% a.a.) para o volume solicitado."""
        vol = max(0.0, vol)
        incremento = self.gamma * (vol / self.vol_referencia) ** self.delta
        custo_raw = self.custo_base + incremento

        if self.tipo_custo == "spread_cdi_pct":
            # custo_raw em % do CDI
            return cdi_cenario * (custo_raw / 100)
        elif self.tipo_custo == "spread_ipca":
            # custo_raw = taxa real; custo total ≈ IPCA + taxa_real
            return 4.5 + custo_raw   # IPCA base
        elif self.tipo_custo == "taxa_pre":
            return custo_raw
        return custo_raw


@dataclass
class TituloPublico:
    """Título público disponível no mercado com preço e volume."""
    nome: str
    indexador: str
    taxa_mercado: float       # % a.a. (yield)
    prazo_medio: float        # anos
    vol_disponivel: float     # volume total disponível (R$ milhões)
    vol_min_lote: float       # lote mínimo por operação
    duration_mod: float       # duration modificada
    fpr: float
    hqla: str
    rsf_fator: float


@dataclass
class InstrumentoHedge:
    """
    Instrumento de hedge de taxa (swap ou futuro).
    Posição comprada (long) = recebe fixo / paga flutuante → reduz duration
    Posição vendida (short) = paga fixo / recebe flutuante → aumenta duration
    """
    nome: str
    tipo: str                 # "swap_di_pre", "swap_di_ipca", "futuro_di"
    taxa_fixa: float          # taxa do lado fixo (% a.a.)
    indexador_flutuante: str  # CDI, IPCA
    prazo_medio: float        # anos
    dv01_por_milhao: float    # DV01 unitário (R$ por R$1M nocional por 1bp)
    nocional_max: float       # limite operacional (R$ milhões)
    custo_bid_ask: float      # custo de transação (% a.a., aproximado)


# ─────────────────────────────────────────────────────────────
# MERCADO DISPONÍVEL — CALIBRAÇÃO BASE DEZ/2024
# ─────────────────────────────────────────────────────────────

class MercadoDisponivel:
    """
    Agrega todos os instrumentos disponíveis para o otimizador.
    Calibrado com referências de mercado brasileiro Dez/2024.
    """

    def __init__(self, cenario: str = "base"):
        self.cenario = cenario
        self.segmentos_credito   = self._build_credito()
        self.instrumentos_captacao = self._build_captacao()
        self.titulos_publicos    = self._build_titulos()
        self.instrumentos_hedge  = self._build_hedge()

        # CDI do cenário para precificação
        self.cdi = {"base": 11.75, "stress_alta": 13.75, "stress_baixa": 9.25}[cenario]

    # ── CRÉDITO ──────────────────────────────────────────────────────────────
    def _build_credito(self) -> List[SegmentoCredito]:
        """
        Parâmetros de elasticidade calibrados com literatura empírica brasileira:
        - Coelho, De Mello & Funchal (2012): elasticidade crédito PF ~ 0.15–0.30
        - Bonomo & Martins (2016): crédito PJ ~ 0.20–0.40
        Elasticidade aqui = Δ%volume / Δ%spread (não elasticidade-preço clássica)
        """
        return [
            SegmentoCredito(
                nome="imobiliario_pf",
                spread_min=6.0, spread_mercado=8.2, spread_max=14.0,
                vol_base=4_500,       # R$4.5bi disponível/mês ao spread mercado
                elasticidade=0.25,    # demanda pouco elástica (garantia real)
                fpr=0.35, indexador="TR", prazo_medio=18.0,
                hqla="nao_hqla", rsf_fator=0.65,
                pd_anual=0.80, lgd=0.20,
            ),
            SegmentoCredito(
                nome="consignado_pf",
                spread_min=8.0, spread_mercado=10.75, spread_max=20.0,
                vol_base=3_200,
                elasticidade=0.20,    # muito inelástico (desconto em folha)
                fpr=0.75, indexador="PRE", prazo_medio=3.8,
                hqla="nao_hqla", rsf_fator=0.65,
                pd_anual=2.50, lgd=0.30,
            ),
            SegmentoCredito(
                nome="pessoal_pf",
                spread_min=18.0, spread_mercado=26.65, spread_max=50.0,
                vol_base=1_800,
                elasticidade=0.45,    # mais elástico (sem garantia)
                fpr=0.75, indexador="PRE", prazo_medio=2.2,
                hqla="nao_hqla", rsf_fator=0.65,
                pd_anual=8.50, lgd=0.55,
            ),
            SegmentoCredito(
                nome="veiculos_pf",
                spread_min=10.0, spread_mercado=15.05, spread_max=28.0,
                vol_base=2_100,
                elasticidade=0.35,
                fpr=0.75, indexador="PRE", prazo_medio=4.5,
                hqla="nao_hqla", rsf_fator=0.65,
                pd_anual=3.80, lgd=0.40,
            ),
            SegmentoCredito(
                nome="capital_giro_pj",
                spread_min=2.5, spread_mercado=4.5, spread_max=10.0,
                vol_base=2_800,
                elasticidade=0.55,    # PJ mais sensível a preço
                fpr=0.85, indexador="CDI", prazo_medio=1.5,
                hqla="nao_hqla", rsf_fator=0.85,
                pd_anual=4.20, lgd=0.45,
            ),
            SegmentoCredito(
                nome="corporativo_pj",
                spread_min=1.5, spread_mercado=2.8, spread_max=6.0,
                vol_base=3_500,
                elasticidade=0.65,    # grandes empresas têm alternativas (bonds)
                fpr=1.00, indexador="CDI", prazo_medio=3.2,
                hqla="nao_hqla", rsf_fator=0.85,
                pd_anual=1.80, lgd=0.35,
            ),
            SegmentoCredito(
                nome="rural",
                spread_min=0.2, spread_mercado=0.5, spread_max=2.0,
                vol_base=900,
                elasticidade=0.10,    # quase inelástico (subsidiado/MCR)
                fpr=0.75, indexador="PRE", prazo_medio=2.8,
                hqla="nao_hqla", rsf_fator=0.65,
                pd_anual=2.00, lgd=0.30,
            ),
        ]

    # ── CAPTAÇÃO ─────────────────────────────────────────────────────────────
    def _build_captacao(self) -> List[InstrumentoCaptacao]:
        return [
            InstrumentoCaptacao(
                nome="cdb_cdi_curto",
                custo_base=97.0,      # % CDI base
                tipo_custo="spread_cdi_pct",
                vol_referencia=5_000, gamma=3.0, delta=0.5,
                vol_max=15_000,
                prazo_medio=0.6, indexador="CDI",
                asf_fator=0.50, outflow_lcr=0.20,
            ),
            InstrumentoCaptacao(
                nome="cdb_cdi_longo",
                custo_base=101.0,
                tipo_custo="spread_cdi_pct",
                vol_referencia=3_000, gamma=2.5, delta=0.5,
                vol_max=10_000,
                prazo_medio=2.0, indexador="CDI",
                asf_fator=0.90, outflow_lcr=0.10,
            ),
            InstrumentoCaptacao(
                nome="cdb_ipca",
                custo_base=6.80,      # IPCA + 6.80%
                tipo_custo="spread_ipca",
                vol_referencia=2_000, gamma=0.30, delta=0.5,
                vol_max=8_000,
                prazo_medio=3.0, indexador="IPCA",
                asf_fator=0.90, outflow_lcr=0.05,
            ),
            InstrumentoCaptacao(
                nome="lci_cdi",
                custo_base=91.0,      # % CDI (isento IR → mais barato)
                tipo_custo="spread_cdi_pct",
                vol_referencia=2_500, gamma=2.0, delta=0.5,
                vol_max=7_000,
                prazo_medio=1.5, indexador="CDI",
                asf_fator=0.90, outflow_lcr=0.05,
            ),
            InstrumentoCaptacao(
                nome="lf_cdi_longa",
                custo_base=104.0,     # % CDI (prazo longo exige prêmio)
                tipo_custo="spread_cdi_pct",
                vol_referencia=2_000, gamma=2.0, delta=0.5,
                vol_max=8_000,
                prazo_medio=5.0, indexador="CDI",
                asf_fator=1.00, outflow_lcr=0.00,
            ),
            InstrumentoCaptacao(
                nome="lf_ipca_longa",
                custo_base=6.50,      # IPCA + 6.50%
                tipo_custo="spread_ipca",
                vol_referencia=1_500, gamma=0.25, delta=0.5,
                vol_max=6_000,
                prazo_medio=6.0, indexador="IPCA",
                asf_fator=1.00, outflow_lcr=0.00,
            ),
        ]

    # ── TÍTULOS PÚBLICOS ─────────────────────────────────────────────────────
    def _build_titulos(self) -> List[TituloPublico]:
        return [
            TituloPublico(
                nome="LFT_2027",
                indexador="CDI", taxa_mercado=11.77,
                prazo_medio=1.5, vol_disponivel=20_000, vol_min_lote=100,
                duration_mod=0.02, fpr=0.00,
                hqla="nivel1", rsf_fator=0.05,
            ),
            TituloPublico(
                nome="LFT_2029",
                indexador="CDI", taxa_mercado=11.79,
                prazo_medio=3.0, vol_disponivel=15_000, vol_min_lote=100,
                duration_mod=0.02, fpr=0.00,
                hqla="nivel1", rsf_fator=0.05,
            ),
            TituloPublico(
                nome="LTN_2026",
                indexador="PRE", taxa_mercado=13.10,
                prazo_medio=0.8, vol_disponivel=12_000, vol_min_lote=100,
                duration_mod=0.75, fpr=0.00,
                hqla="nivel1", rsf_fator=0.05,
            ),
            TituloPublico(
                nome="LTN_2027",
                indexador="PRE", taxa_mercado=13.20,
                prazo_medio=1.8, vol_disponivel=10_000, vol_min_lote=100,
                duration_mod=1.55, fpr=0.00,
                hqla="nivel1", rsf_fator=0.05,
            ),
            TituloPublico(
                nome="NTNB_2028",
                indexador="IPCA", taxa_mercado=6.30,
                prazo_medio=3.5, vol_disponivel=8_000, vol_min_lote=100,
                duration_mod=3.10, fpr=0.00,
                hqla="nivel1", rsf_fator=0.05,
            ),
            TituloPublico(
                nome="NTNB_2035",
                indexador="IPCA", taxa_mercado=6.45,
                prazo_medio=10.5, vol_disponivel=6_000, vol_min_lote=100,
                duration_mod=7.80, fpr=0.00,
                hqla="nivel1", rsf_fator=0.05,
            ),
            TituloPublico(
                nome="NTNB_2050",
                indexador="IPCA", taxa_mercado=6.55,
                prazo_medio=25.0, vol_disponivel=4_000, vol_min_lote=100,
                duration_mod=14.20, fpr=0.00,
                hqla="nivel1", rsf_fator=0.05,
            ),
            TituloPublico(
                nome="NTNF_2029",
                indexador="PRE", taxa_mercado=13.85,
                prazo_medio=4.5, vol_disponivel=7_000, vol_min_lote=100,
                duration_mod=3.40, fpr=0.00,
                hqla="nivel1", rsf_fator=0.05,
            ),
        ]

    # ── HEDGE ─────────────────────────────────────────────────────────────────
    def _build_hedge(self) -> List[InstrumentoHedge]:
        """
        Swaps e futuros disponíveis para gestão de duration gap.
        DV01 por R$1M nocional calculado como: DV01 = Duration × 0.0001 × 1M
        """
        return [
            InstrumentoHedge(
                nome="Swap_DI_Pre_1A",
                tipo="swap_di_pre",
                taxa_fixa=13.15, indexador_flutuante="CDI",
                prazo_medio=1.0,
                dv01_por_milhao=95.0,   # R$ por R$1M por 1bp
                nocional_max=30_000,
                custo_bid_ask=0.02,
            ),
            InstrumentoHedge(
                nome="Swap_DI_Pre_3A",
                tipo="swap_di_pre",
                taxa_fixa=13.40, indexador_flutuante="CDI",
                prazo_medio=3.0,
                dv01_por_milhao=270.0,
                nocional_max=20_000,
                custo_bid_ask=0.03,
            ),
            InstrumentoHedge(
                nome="Swap_DI_Pre_5A",
                tipo="swap_di_pre",
                taxa_fixa=13.65, indexador_flutuante="CDI",
                prazo_medio=5.0,
                dv01_por_milhao=430.0,
                nocional_max=15_000,
                custo_bid_ask=0.04,
            ),
            InstrumentoHedge(
                nome="Swap_DI_IPCA_3A",
                tipo="swap_di_ipca",
                taxa_fixa=6.35, indexador_flutuante="IPCA",
                prazo_medio=3.0,
                dv01_por_milhao=265.0,
                nocional_max=20_000,
                custo_bid_ask=0.05,
            ),
            InstrumentoHedge(
                nome="Swap_DI_IPCA_5A",
                tipo="swap_di_ipca",
                taxa_fixa=6.50, indexador_flutuante="IPCA",
                prazo_medio=5.0,
                dv01_por_milhao=425.0,
                nocional_max=15_000,
                custo_bid_ask=0.06,
            ),
            InstrumentoHedge(
                nome="Futuro_DI_Jan27",
                tipo="futuro_di",
                taxa_fixa=13.20, indexador_flutuante="CDI",
                prazo_medio=2.1,
                dv01_por_milhao=197.0,
                nocional_max=50_000,   # futuros têm alta liquidez
                custo_bid_ask=0.01,
            ),
            InstrumentoHedge(
                nome="Futuro_DI_Jan29",
                tipo="futuro_di",
                taxa_fixa=13.55, indexador_flutuante="CDI",
                prazo_medio=4.1,
                dv01_por_milhao=368.0,
                nocional_max=30_000,
                custo_bid_ask=0.01,
            ),
        ]

    # ── UTILITÁRIOS ──────────────────────────────────────────────────────────
    def resumo(self):
        print(f"\n{'='*60}")
        print(f"  MERCADO DISPONÍVEL — Cenário: {self.cenario.upper()}")
        print(f"{'='*60}")
        print(f"\n  Crédito ({len(self.segmentos_credito)} segmentos):")
        for s in self.segmentos_credito:
            print(f"    {s.nome:<22} spread={s.spread_mercado:.1f}%  "
                  f"vol_base=R${s.vol_base:,.0f}M  ε={s.elasticidade}")
        print(f"\n  Captação ({len(self.instrumentos_captacao)} instrumentos):")
        for c in self.instrumentos_captacao:
            print(f"    {c.nome:<20} custo_base={c.custo_base:.1f}  "
                  f"vol_max=R${c.vol_max:,.0f}M")
        print(f"\n  Títulos Públicos ({len(self.titulos_publicos)} séries):")
        for t in self.titulos_publicos:
            print(f"    {t.nome:<14} {t.indexador:<5}  "
                  f"yield={t.taxa_mercado:.2f}%  dur={t.duration_mod:.2f}a  "
                  f"vol=R${t.vol_disponivel:,.0f}M")
        print(f"\n  Hedge ({len(self.instrumentos_hedge)} instrumentos):")
        for h in self.instrumentos_hedge:
            print(f"    {h.nome:<22} taxa={h.taxa_fixa:.2f}%  "
                  f"DV01/M=R${h.dv01_por_milhao:.0f}  "
                  f"max=R${h.nocional_max:,.0f}M")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    mercado = MercadoDisponivel(cenario="base")
    mercado.resumo()

    # Demonstra elasticidade
    print("Curva de demanda — Consignado PF:")
    seg = mercado.segmentos_credito[1]
    for sp in [8.0, 9.0, 10.75, 12.0, 15.0, 20.0]:
        print(f"  spread={sp:.1f}%  →  vol={seg.volume_disponivel(sp):,.0f} M")
