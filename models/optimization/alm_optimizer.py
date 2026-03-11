"""
M4b - ALM Optimizer (Baseline QP)
===================================
Otimizador de alocação entre crédito, títulos e hedge.

Formulação:
  max  NII(x) - λ × DV01_gap(x)²
  s.t. LCR  ≥ 100%
       NSFR ≥ 100%
       IB   ≥ 10.5%
       |DurationGap| ≤ 2.0 anos
       x_cred[s] ≤ DemandaCred[s](spread[s])
       x_cap[f]  ≤ vol_max[f]
       x_tvm[t]  ∈ [-estoque, vol_disponivel]
       Σ usos = Σ fontes  (equilíbrio de balanço)
       x_cred ≥ 0, x_cap ≥ 0

Solver: scipy.optimize.minimize (SLSQP — Sequential Least Squares Programming)
        Compatível com substituição por cvxpy (ver nota ao final do arquivo)

Referências:
  - Kouwenberg & Zenios (2001): Stochastic programming models for asset-liability mgmt
  - Grinold & Kahn (2000): Active Portfolio Management
  - BIS BCBS 368 (2016): Interest Rate Risk in the Banking Book (IRRBB)
"""

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize, LinearConstraint, Bounds
from dataclasses import dataclass
import sys

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

DATA_DIR   = BASE_DIR / "data" / "processed"
SCEN_DIR   = BASE_DIR / "data" / "scenarios"
OUTPUT_DIR = BASE_DIR / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from models.optimization.market_environment import MercadoDisponivel
from models.cash_flow.cash_flow_engine import CashFlowEngine


# ─────────────────────────────────────────────────────────────
# PARÂMETROS REGULATÓRIOS
# ─────────────────────────────────────────────────────────────
LCR_MINIMO    = 1.00    # 100%
NSFR_MINIMO   = 1.00    # 100%
IB_MINIMO     = 0.105   # 10.5% (mínimo com buffer de conservação)
DGAP_MAX      = 2.0     # anos — limite interno IRRBB
SPREAD_CDI_BASE = 11.75 # % a.a. — CDI cenário base


# ─────────────────────────────────────────────────────────────
# RESULTADO DA OTIMIZAÇÃO
# ─────────────────────────────────────────────────────────────
@dataclass
class ResultadoOtimizacao:
    status: str
    lambda_risco: float
    nii_incremental: float       # R$ milhões/ano de NII novo
    nii_total: float             # NII total (base + incremental)
    dv01_gap_pos: float          # DV01 gap pós-otimização
    duration_gap_pos: float      # Duration gap pós-otimização
    lcr_pos: float               # LCR pós-otimização
    nsfr_pos: float              # NSFR pós-otimização
    ib_pos: float                # Índice de Basileia pós
    credito_novo: Dict[str, float]   # volume por segmento
    captacao_nova: Dict[str, float]  # volume por instrumento
    tvm_delta: Dict[str, float]      # variação em títulos
    hedge_nocional: Dict[str, float] # nocional por instrumento
    custo_hedge: float
    receita_credito: float
    custo_captacao: float


# ─────────────────────────────────────────────────────────────
# CLASSE PRINCIPAL
# ─────────────────────────────────────────────────────────────
class ALMOptimizer:
    """
    Otimizador de ALM baseado em QP (SLSQP via scipy).

    Variáveis de decisão (vetor x):
      [0:Nc]         — crédito novo por segmento (R$ M)
      [Nc:Nc+Nf]     — nova captação por instrumento (R$ M)
      [Nc+Nf:Nc+Nf+Nt] — delta em títulos públicos (R$ M, pode ser negativo)
      [Nc+Nf+Nt:]    — nocional de hedge (R$ M, pode ser negativo)
    """

    def __init__(self, cenario: str = "base"):
        self.cenario  = cenario
        self.mercado  = MercadoDisponivel(cenario=cenario)
        self.engine   = CashFlowEngine()

        # Dimensões
        self.Nc = len(self.mercado.segmentos_credito)
        self.Nf = len(self.mercado.instrumentos_captacao)
        self.Nt = len(self.mercado.titulos_publicos)
        self.Nh = len(self.mercado.instrumentos_hedge)
        self.N  = self.Nc + self.Nf + self.Nt + self.Nh

        # CDI do cenário
        self.cdi = self.mercado.cdi

        # Estado base do balanço (das fases anteriores)
        self._carregar_estado_base()

    # ─── ESTADO BASE ─────────────────────────────────────────
    def _carregar_estado_base(self):
        """Carrega métricas do balanço atual como ponto de partida."""
        ativos   = self.engine.ativos
        passivos = self.engine.passivos

        self.total_ativo_base   = ativos["saldo"].sum()
        self.total_passivo_base = passivos["saldo"].sum()
        self.rwa_base           = ativos["rwa"].sum()
        self.dv01_ativo_base    = ativos["dv01"].sum()
        self.dv01_passivo_base  = passivos["dv01"].sum()
        self.dv01_gap_base      = self.dv01_ativo_base - self.dv01_passivo_base

        # Duration gap base
        da = np.average(ativos["duration_mod"], weights=ativos["saldo"])
        dp = np.average(passivos["duration_mod"], weights=passivos["saldo"])
        self.dgap_base = da - (self.total_passivo_base / self.total_ativo_base) * dp

        # LCR base
        hqla_n1  = ativos[ativos["hqla"] == "nivel1"]["saldo"].sum()
        hqla_2a  = ativos[ativos["hqla"] == "nivel2A"]["saldo"].sum() * 0.85
        hqla_2b  = ativos[ativos["hqla"] == "nivel2B"]["saldo"].sum() * 0.75
        self.hqla_base = hqla_n1 + hqla_2a + hqla_2b
        self.outflows_base = (passivos["saldo"] * passivos["outflow_lcr"]).sum()
        self.lcr_base = self.hqla_base / self.outflows_base if self.outflows_base > 0 else 99.0

        # NSFR base
        self.asf_base = (passivos["saldo"] * passivos["asf_fator"]).sum()
        self.rsf_base = (ativos["saldo"] * ativos["rsf_fator"]).sum()
        self.nsfr_base = self.asf_base / self.rsf_base if self.rsf_base > 0 else 99.0

        # Patrimônio de referência
        with open(DATA_DIR / "patrimonio.json") as f:
            pat = json.load(f)
        self.pr_base = pat["patrimonio_referencia"]

        # NII base anual (cenário)
        nii_df = self.engine.projetar_nii(12)
        self.nii_base_anual = nii_df[self.cenario]["nii"].sum()

        print(f"\n  Estado base ({self.cenario}):")
        print(f"    NII anual:      R$ {self.nii_base_anual:,.0f}M")
        print(f"    Duration Gap:   {self.dgap_base:.2f} anos")
        print(f"    DV01 Gap:       R$ {self.dv01_gap_base:.1f}M/bp")
        print(f"    LCR:            {self.lcr_base:.2%}")
        print(f"    NSFR:           {self.nsfr_base:.2%}")
        print(f"    IB:             {self.pr_base/self.rwa_base:.2%}")

    # ─── ÍNDICES DO VETOR X ───────────────────────────────────
    def _idx_cred(self):  return slice(0, self.Nc)
    def _idx_cap(self):   return slice(self.Nc, self.Nc + self.Nf)
    def _idx_tvm(self):   return slice(self.Nc + self.Nf, self.Nc + self.Nf + self.Nt)
    def _idx_hdg(self):   return slice(self.Nc + self.Nf + self.Nt, self.N)

    # ─── CÁLCULO DE NII INCREMENTAL ───────────────────────────
    def _nii_incremental(self, x: np.ndarray) -> float:
        """
        NII gerado pelos novos negócios (crédito + TVM + captação + hedge).
        Simplificação: NII anual = receita_nova - custo_novo
        """
        nii = 0.0
        segs = self.mercado.segmentos_credito
        caps = self.mercado.instrumentos_captacao
        tvms = self.mercado.titulos_publicos
        hdgs = self.mercado.instrumentos_hedge

        # Receita de crédito novo
        for i, seg in enumerate(segs):
            vol  = x[i]
            taxa = self.cdi + seg.spread_mercado if seg.indexador == "CDI" else seg.spread_mercado
            # Receita líquida de PD esperada
            receita_bruta = vol * (taxa / 100)
            perda_esperada = vol * seg.pd_anual / 100 * seg.lgd
            nii += receita_bruta - perda_esperada

        # Receita/custo de TVM (delta)
        for j, tvm in enumerate(tvms):
            delta_vol = x[self.Nc + self.Nf + j]
            if delta_vol > 0:   # compra
                nii += delta_vol * (tvm.taxa_mercado / 100)
            else:               # venda (custo de oportunidade)
                nii += delta_vol * (tvm.taxa_mercado / 100)

        # Custo de nova captação
        for k, cap in enumerate(caps):
            vol  = x[self.Nc + k]
            custo = cap.custo_efetivo(vol, self.cdi)
            nii -= vol * (custo / 100)

        # Custo/ganho de hedge (bid-ask como custo)
        for h, hdg in enumerate(hdgs):
            nocional = abs(x[self.Nc + self.Nf + self.Nt + h])
            nii -= nocional * (hdg.custo_bid_ask / 100)

        return nii

    # ─── DV01 GAP PÓS-OTIMIZAÇÃO ─────────────────────────────
    def _dv01_gap_pos(self, x: np.ndarray) -> float:
        """DV01 gap do balanço após os novos negócios."""
        dv01_delta = 0.0
        segs = self.mercado.segmentos_credito
        caps = self.mercado.instrumentos_captacao
        tvms = self.mercado.titulos_publicos
        hdgs = self.mercado.instrumentos_hedge

        # Crédito novo: DV01 = vol × duration × 0.0001
        for i, seg in enumerate(segs):
            from models.cash_flow.cash_flow_engine import duration_modificada
            dur = duration_modificada(seg.prazo_medio,
                                      seg.spread_mercado, seg.indexador)
            dv01_delta += x[i] * dur * 0.0001

        # TVM delta
        for j, tvm in enumerate(tvms):
            delta_vol = x[self.Nc + self.Nf + j]
            dv01_delta += delta_vol * tvm.duration_mod * 0.0001

        # Captação nova (reduz DV01 — é passivo)
        for k, cap in enumerate(caps):
            from models.cash_flow.cash_flow_engine import duration_modificada
            dur = duration_modificada(cap.prazo_medio,
                                      cap.custo_efetivo(x[self.Nc+k], self.cdi),
                                      cap.indexador)
            dv01_delta -= x[self.Nc + k] * dur * 0.0001

        # Hedge: posição comprada (recebe fixo) reduz duration gap
        for h, hdg in enumerate(hdgs):
            nocional = x[self.Nc + self.Nf + self.Nt + h]
            # DV01/M em R$ por R$1M → converter para R$M por R$M
            dv01_delta -= nocional * (hdg.dv01_por_milhao / 1_000_000)

        return self.dv01_gap_base + dv01_delta

    # ─── RESTRIÇÕES ───────────────────────────────────────────
    def _lcr_pos(self, x: np.ndarray) -> float:
        """LCR pós novos negócios. Retorna LCR (>= LCR_MINIMO)."""
        hqla_delta   = 0.0
        outflow_delta = 0.0
        tvms = self.mercado.titulos_publicos
        caps = self.mercado.instrumentos_captacao

        # TVM: compra de HQLA aumenta numerador
        multiplicadores = {"nivel1": 1.0, "nivel2A": 0.85, "nivel2B": 0.75, "nao_hqla": 0.0}
        for j, tvm in enumerate(tvms):
            delta = x[self.Nc + self.Nf + j]
            hqla_delta += delta * multiplicadores.get(tvm.hqla, 0.0)

        # Captação nova: aumenta outflows
        for k, cap in enumerate(caps):
            outflow_delta += x[self.Nc + k] * cap.outflow_lcr

        hqla_total    = self.hqla_base + hqla_delta
        outflows_total = self.outflows_base + outflow_delta
        return hqla_total / outflows_total if outflows_total > 0 else 99.0

    def _nsfr_pos(self, x: np.ndarray) -> float:
        """NSFR pós novos negócios."""
        asf_delta = 0.0
        rsf_delta = 0.0
        segs = self.mercado.segmentos_credito
        caps = self.mercado.instrumentos_captacao
        tvms = self.mercado.titulos_publicos

        for k, cap in enumerate(caps):
            asf_delta += x[self.Nc + k] * cap.asf_fator
        for i, seg in enumerate(segs):
            rsf_delta += x[i] * seg.rsf_fator
        for j, tvm in enumerate(tvms):
            rsf_delta += max(0, x[self.Nc + self.Nf + j]) * tvm.rsf_fator

        return (self.asf_base + asf_delta) / (self.rsf_base + rsf_delta)

    def _ib_pos(self, x: np.ndarray) -> float:
        """Índice de Basileia pós novos negócios."""
        rwa_delta = 0.0
        segs = self.mercado.segmentos_credito
        tvms = self.mercado.titulos_publicos
        for i, seg in enumerate(segs):
            rwa_delta += x[i] * seg.fpr
        for j, tvm in enumerate(tvms):
            rwa_delta += max(0, x[self.Nc + self.Nf + j]) * tvm.fpr
        return self.pr_base / (self.rwa_base + rwa_delta)

    def _dgap_pos(self, x: np.ndarray) -> float:
        """Duration gap pós otimização (anos)."""
        dv01_gap = self._dv01_gap_pos(x)
        # Aproximação: converte DV01 gap para duration gap
        # Duration gap ≈ DV01_gap / (Total_Ativo × 0.0001)
        total_ativo_pos = self.total_ativo_base + x[self._idx_cred()].sum() + \
                          max(0, x[self._idx_tvm()].sum())
        return dv01_gap / (total_ativo_pos * 0.0001) if total_ativo_pos > 0 else 0.0

    # ─── FUNÇÃO OBJETIVO ─────────────────────────────────────
    def _objetivo(self, x: np.ndarray, lambda_risco: float) -> float:
        """
        Minimiza: -NII + λ × DV01_gap²
        (scipy minimiza → negamos NII para maximizar)
        """
        nii   = self._nii_incremental(x)
        dv01g = self._dv01_gap_pos(x)
        return -nii + lambda_risco * (dv01g ** 2)

    # ─── OTIMIZAÇÃO ÚNICA ─────────────────────────────────────
    def otimizar(self, lambda_risco: float = 0.001,
                 spreads_cred: Optional[Dict] = None) -> ResultadoOtimizacao:
        """
        Resolve o problema de otimização para um dado λ.

        spreads_cred: dict {nome_segmento: spread} — se None usa spread de mercado.
        """
        segs = self.mercado.segmentos_credito
        caps = self.mercado.instrumentos_captacao
        tvms = self.mercado.titulos_publicos
        hdgs = self.mercado.instrumentos_hedge

        # Spreads efetivos
        spreads = {}
        for seg in segs:
            spreads[seg.nome] = (spreads_cred or {}).get(seg.nome, seg.spread_mercado)

        # ── Bounds ────────────────────────────────────────────
        lb = np.zeros(self.N)
        ub = np.zeros(self.N)

        for i, seg in enumerate(segs):
            lb[i] = 0.0
            ub[i] = seg.volume_disponivel(spreads[seg.nome])

        for k, cap in enumerate(caps):
            lb[self.Nc + k] = 0.0
            ub[self.Nc + k] = cap.vol_max

        for j, tvm in enumerate(tvms):
            # Pode vender estoque existente (até saldo atual do balanço)
            estoque = self.engine.ativos[
                self.engine.ativos["produto"].str.contains(
                    tvm.nome.split("_")[0], case=False, na=False)
            ]["saldo"].sum()
            lb[self.Nc + self.Nf + j] = -min(estoque, tvm.vol_disponivel)
            ub[self.Nc + self.Nf + j] = tvm.vol_disponivel

        for h, hdg in enumerate(hdgs):
            lb[self.Nc + self.Nf + self.Nt + h] = -hdg.nocional_max
            ub[self.Nc + self.Nf + self.Nt + h] =  hdg.nocional_max

        bounds = Bounds(lb, ub)

        # ── Restrições ────────────────────────────────────────
        constraints = [
            # LCR >= 100%
            {"type": "ineq", "fun": lambda x: self._lcr_pos(x) - LCR_MINIMO},
            # NSFR >= 100%
            {"type": "ineq", "fun": lambda x: self._nsfr_pos(x) - NSFR_MINIMO},
            # IB >= 10.5%
            {"type": "ineq", "fun": lambda x: self._ib_pos(x) - IB_MINIMO},
            # |Duration Gap| <= DGAP_MAX
            {"type": "ineq", "fun": lambda x:  DGAP_MAX - self._dgap_pos(x)},
            {"type": "ineq", "fun": lambda x:  DGAP_MAX + self._dgap_pos(x)},
            # Equilíbrio de balanço: novos ativos = novas fontes
            {"type": "eq",   "fun": lambda x: (
                x[self._idx_cred()].sum() + max(0, x[self._idx_tvm()].sum()) -
                x[self._idx_cap()].sum()
            )},
        ]

        # ── Ponto inicial: 20% dos limites superiores ─────────
        x0 = 0.2 * ub
        x0[self._idx_tvm()] = 0.0   # TVM começa neutro

        # ── Resolve ───────────────────────────────────────────
        resultado = minimize(
            fun=self._objetivo,
            x0=x0,
            args=(lambda_risco,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-8, "disp": False},
        )

        # ── Monta resultado ───────────────────────────────────
        x_opt = resultado.x
        status = "ótimo" if resultado.success else f"convergência parcial ({resultado.message})"

        cred_novo  = {segs[i].nome:  float(x_opt[i])            for i in range(self.Nc)}
        cap_nova   = {caps[k].nome:  float(x_opt[self.Nc + k])  for k in range(self.Nf)}
        tvm_delta  = {tvms[j].nome:  float(x_opt[self.Nc + self.Nf + j]) for j in range(self.Nt)}
        hdg_noc    = {hdgs[h].nome:  float(x_opt[self.Nc + self.Nf + self.Nt + h]) for h in range(self.Nh)}

        nii_incr = self._nii_incremental(x_opt)
        custo_hdg = sum(abs(v) * hdgs[h].custo_bid_ask / 100
                        for h, v in enumerate(x_opt[self._idx_hdg()]))
        rec_cred  = sum(x_opt[i] * (
            (self.cdi + segs[i].spread_mercado if segs[i].indexador == "CDI"
             else segs[i].spread_mercado) / 100
        ) for i in range(self.Nc))
        cst_cap   = sum(x_opt[self.Nc + k] *
                        caps[k].custo_efetivo(x_opt[self.Nc + k], self.cdi) / 100
                        for k in range(self.Nf))

        return ResultadoOtimizacao(
            status=status,
            lambda_risco=lambda_risco,
            nii_incremental=nii_incr,
            nii_total=self.nii_base_anual + nii_incr,
            dv01_gap_pos=self._dv01_gap_pos(x_opt),
            duration_gap_pos=self._dgap_pos(x_opt),
            lcr_pos=self._lcr_pos(x_opt),
            nsfr_pos=self._nsfr_pos(x_opt),
            ib_pos=self._ib_pos(x_opt),
            credito_novo=cred_novo,
            captacao_nova=cap_nova,
            tvm_delta=tvm_delta,
            hedge_nocional=hdg_noc,
            custo_hedge=custo_hdg,
            receita_credito=rec_cred,
            custo_captacao=cst_cap,
        )

    # ─── FRONTEIRA EFICIENTE ──────────────────────────────────
    def fronteira_eficiente(self, n_pontos: int = 15) -> pd.DataFrame:
        """
        Gera fronteira eficiente NII × Risco(DV01 gap).
        Varia λ de quase 0 (max NII) até alto (min risco).
        """
        lambdas = np.logspace(-4, 0, n_pontos)
        rows = []
        print(f"\n  Calculando fronteira eficiente ({n_pontos} pontos)...")
        for i, lam in enumerate(lambdas):
            try:
                res = self.otimizar(lambda_risco=lam)
                rows.append({
                    "lambda":          lam,
                    "nii_total":       res.nii_total,
                    "nii_incremental": res.nii_incremental,
                    "dv01_gap":        abs(res.dv01_gap_pos),
                    "duration_gap":    abs(res.duration_gap_pos),
                    "lcr":             res.lcr_pos,
                    "nsfr":            res.nsfr_pos,
                    "ib":              res.ib_pos,
                    "status":          res.status,
                    "vol_cred_total":  sum(res.credito_novo.values()),
                    "vol_cap_total":   sum(res.captacao_nova.values()),
                    "vol_hdg_total":   sum(abs(v) for v in res.hedge_nocional.values()),
                })
                print(f"    λ={lam:.4f}  NII={res.nii_total:,.0f}M  "
                      f"DV01gap={abs(res.dv01_gap_pos):.1f}  "
                      f"LCR={res.lcr_pos:.1%}  IB={res.ib_pos:.1%}  [{res.status[:15]}]")
            except Exception as e:
                print(f"    λ={lam:.4f}  ERRO: {e}")

        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_DIR / f"fronteira_eficiente_{self.cenario}.csv", index=False)
        return df

    def imprimir_resultado(self, res: ResultadoOtimizacao):
        print(f"\n{'='*65}")
        print(f"  RESULTADO DA OTIMIZAÇÃO — λ={res.lambda_risco:.4f} | {res.status}")
        print(f"{'='*65}")
        print(f"\n  {'OBJETIVO':}")
        print(f"    NII incremental:    R$ {res.nii_incremental:>10,.1f}M/ano")
        print(f"    NII total:          R$ {res.nii_total:>10,.1f}M/ano")
        print(f"    Receita crédito:    R$ {res.receita_credito:>10,.1f}M/ano")
        print(f"    Custo captação:     R$ {res.custo_captacao:>10,.1f}M/ano")
        print(f"    Custo hedge:        R$ {res.custo_hedge:>10,.1f}M/ano")
        print(f"\n  {'RISCO DE TAXA':}")
        print(f"    Duration Gap pós:   {res.duration_gap_pos:>8.2f} anos  "
              f"(limite: ±{DGAP_MAX:.1f}a)")
        print(f"    DV01 Gap pós:       R$ {res.dv01_gap_pos:>8.1f}M/bp")
        print(f"\n  {'REGULATÓRIO':}")
        print(f"    LCR pós:            {res.lcr_pos:>8.1%}  (mín: {LCR_MINIMO:.0%})")
        print(f"    NSFR pós:           {res.nsfr_pos:>8.1%}  (mín: {NSFR_MINIMO:.0%})")
        print(f"    Índice Basileia:    {res.ib_pos:>8.1%}  (mín: {IB_MINIMO:.1%})")
        print(f"\n  {'CRÉDITO NOVO':}")
        for s, v in res.credito_novo.items():
            if v > 1:
                print(f"    {s:<25} R$ {v:>8,.0f}M")
        print(f"\n  {'CAPTAÇÃO NOVA':}")
        for f, v in res.captacao_nova.items():
            if v > 1:
                print(f"    {f:<25} R$ {v:>8,.0f}M")
        print(f"\n  {'TVM DELTA':}")
        for t, v in res.tvm_delta.items():
            if abs(v) > 1:
                print(f"    {t:<20} R$ {v:>+9,.0f}M")
        print(f"\n  {'HEDGE':}")
        for h, v in res.hedge_nocional.items():
            if abs(v) > 1:
                print(f"    {h:<25} R$ {v:>+9,.0f}M nocional")
        print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────
# NOTA PARA SUBSTITUIÇÃO POR CVXPY
# ─────────────────────────────────────────────────────────────
# Para usar cvxpy (após: pip install cvxpy), o problema pode ser
# reformulado como:
#
#   import cvxpy as cp
#   x = cp.Variable(N)
#   obj = cp.Maximize(nii_expr(x) - lambda_risco * cp.square(dv01_gap_expr(x)))
#   constraints = [lcr_expr(x) >= LCR_MINIMO, ...]
#   prob = cp.Problem(obj, constraints)
#   prob.solve(solver=cp.OSQP)  # ou ECOS, SCS, GUROBI
#
# Vantagens do cvxpy: sintaxe declarativa, certifica convexidade,
# acesso a solvers comerciais (Gurobi, MOSEK) para problemas grandes.
# ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    opt = ALMOptimizer(cenario="base")
    opt.mercado.resumo()

    print("\n  Otimização pontual — λ=0.001 (foco em NII):")
    res = opt.otimizar(lambda_risco=0.001)
    opt.imprimir_resultado(res)
