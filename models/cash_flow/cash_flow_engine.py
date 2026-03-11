"""
M3 - Cash Flow Engine
======================
Motor de projeção de fluxos de caixa para ALM bancário.

Entrega:
  1. Gap de reprecificação por bucket temporal (O/N → 10A+)
  2. Projeção de NII mês a mês (12 meses) nos 3 cenários
  3. Duration modificada e DV01 por produto
  4. Gap cumulativo de reprecificação

Metodologia:
  - Buckets de reprecificação: data em que o produto tem sua taxa revisada
    (não confundir com vencimento — um CDB CDI reprecia diariamente)
  - NII = Σ (saldo_ativo × taxa_ativo) - Σ (saldo_passivo × custo_passivo)
  - Duration modificada: via fórmula analítica por tipo de produto

Referência: BIS BCBS 368 (IRRBB), Circular BCB 3.365/2007
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

DATA_DIR   = BASE_DIR / "data" / "processed"
SCEN_DIR   = BASE_DIR / "data" / "scenarios"
OUTPUT_DIR = BASE_DIR / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# BUCKETS TEMPORAIS (anos) — padrão BCBS / BCB
# ─────────────────────────────────────────────────────────────
BUCKETS = [
    ("O/N–1M",   0,      1/12),
    ("1M–3M",    1/12,   3/12),
    ("3M–6M",    3/12,   6/12),
    ("6M–1A",    6/12,   1.0),
    ("1A–2A",    1.0,    2.0),
    ("2A–3A",    2.0,    3.0),
    ("3A–5A",    3.0,    5.0),
    ("5A–7A",    5.0,    7.0),
    ("7A–10A",   7.0,   10.0),
    ("10A+",    10.0,   30.0),
]
BUCKET_LABELS = [b[0] for b in BUCKETS]
BUCKET_MIDS   = [(b[1] + b[2]) / 2 for b in BUCKETS]   # ponto médio de cada bucket


# ─────────────────────────────────────────────────────────────
# MAPEAMENTO: prazo_medio → bucket de reprecificação
# ─────────────────────────────────────────────────────────────
def prazo_para_bucket(prazo_anos: float, indexador: str) -> int:
    """
    Retorna o índice do bucket de reprecificação.
    Produtos pós-fixados (CDI, TR) reprecia no O/N–1M independente do prazo.
    Produtos pré-fixados reprecia no vencimento.
    IPCA: reprecia no vencimento (componente pré da taxa real).
    """
    if indexador in ("CDI",):
        return 0   # reprecia diariamente → bucket O/N–1M
    # pré-fixado, IPCA, TR, USD → reprecia no vencimento
    for i, (label, lo, hi) in enumerate(BUCKETS):
        if lo <= prazo_anos < hi:
            return i
    return len(BUCKETS) - 1   # 10A+


# ─────────────────────────────────────────────────────────────
# DURATION MODIFICADA
# ─────────────────────────────────────────────────────────────
def duration_modificada(prazo_anos: float, taxa_anual_pct: float,
                        indexador: str, freq_cupom: int = 2) -> float:
    """
    Calcula duration modificada (em anos).
    Para produtos pós-fixados CDI: duration ≈ prazo até próximo reset (≈ 0)
    Para bullet (sem cupom): Duration = prazo / (1 + y)
    Para amortizável com cupom: Duration de Macaulay / (1 + y/freq)

    freq_cupom: pagamentos por ano (2 = semestral para NTN-B/F, 1 = anual)
    """
    if indexador == "CDI":
        return 1/252   # reprecifica diariamente ≈ zero duration

    y = taxa_anual_pct / 100
    n = prazo_anos * freq_cupom   # número total de períodos

    if n <= 0 or y <= 0:
        return prazo_anos

    # Produtos sem cupom periódico (bullet): consignado, pessoal, veículos
    if indexador == "PRE" and freq_cupom == 1:
        mac_duration = prazo_anos
        return mac_duration / (1 + y)

    # Produtos com cupom (NTN-B, NTN-F, LF): Macaulay analítico
    freq = freq_cupom
    c = y / freq   # taxa por período
    try:
        pv_factor = (1 + c) ** n
        mac = ((1 + c) / c) - (n / (pv_factor - 1))
        mac_anos = mac / freq
        return mac_anos / (1 + y / freq)
    except (ZeroDivisionError, OverflowError):
        return prazo_anos / (1 + y)


def dv01(saldo: float, duration_mod: float, taxa_anual_pct: float) -> float:
    """
    DV01 = Dollar Value of 1 basis point
    DV01 = Saldo × Duration_Mod × 0.0001
    Unidade: R$ milhões por 1bp de variação na taxa
    """
    return saldo * duration_mod * 0.0001


# ─────────────────────────────────────────────────────────────
# CLASSE PRINCIPAL
# ─────────────────────────────────────────────────────────────
class CashFlowEngine:
    """
    Motor central de fluxos de caixa do ALM.
    Lê o balanço processado e os cenários, e produz:
      - Tabela de gap de reprecificação
      - Projeção de NII mensal (12 meses)
      - Duration e DV01 por produto
    """

    def __init__(self):
        self.ativos   = pd.read_csv(DATA_DIR / "ativos.csv")
        self.passivos = pd.read_csv(DATA_DIR / "passivos.csv")
        self.curvas_di = pd.read_csv(SCEN_DIR / "curvas_di.csv")
        self.curvas_ipca = pd.read_csv(SCEN_DIR / "curvas_ipca.csv")
        with open(SCEN_DIR / "spreads_credito.json") as f:
            self.spreads = json.load(f)
        with open(SCEN_DIR / "projecoes_crescimento.json") as f:
            self.crescimento = json.load(f)

        self._enriquecer_produtos()

    # ─── ENRIQUECIMENTO ──────────────────────────────────────
    def _enriquecer_produtos(self):
        """Adiciona bucket, duration e DV01 a cada produto."""

        for df in [self.ativos, self.passivos]:
            buckets, durations, dv01s = [], [], []
            for _, row in df.iterrows():
                bkt = prazo_para_bucket(row["prazo_medio"], row["indexador"])
                buckets.append(bkt)

                taxa = row.get("taxa_total", row.get("custo", 0)) or 0
                freq = 2 if row["produto"] in (
                    "Títulos NTN-B", "Títulos NTN-F", "LF CDI Longa", "LF IPCA Longa"
                ) else 1
                dur = duration_modificada(row["prazo_medio"], taxa,
                                          row["indexador"], freq)
                durations.append(dur)
                dv01s.append(dv01(row["saldo"], dur, taxa))

            df["bucket_idx"] = buckets
            df["bucket"]     = [BUCKET_LABELS[b] for b in buckets]
            df["duration_mod"] = durations
            df["dv01"]         = dv01s

    # ─── GAP DE REPRECIFICAÇÃO ────────────────────────────────
    def gap_reprecificacao(self) -> pd.DataFrame:
        """
        Tabela de gap por bucket: Ativo - Passivo por bucket temporal.
        Gap positivo = banco está LONG → se taxas sobem, NII melhora.
        Gap negativo = banco está SHORT → se taxas sobem, NII piora.
        """
        rows = []
        for idx, (label, lo, hi) in enumerate(BUCKETS):
            ativo_bkt   = self.ativos[self.ativos["bucket_idx"]   == idx]["saldo"].sum()
            passivo_bkt = self.passivos[self.passivos["bucket_idx"] == idx]["saldo"].sum()
            gap = ativo_bkt - passivo_bkt
            dv01_ativo   = self.ativos[self.ativos["bucket_idx"]   == idx]["dv01"].sum()
            dv01_passivo = self.passivos[self.passivos["bucket_idx"] == idx]["dv01"].sum()
            rows.append({
                "bucket":         label,
                "ponto_medio_a":  BUCKET_MIDS[idx],
                "ativo":          ativo_bkt,
                "passivo":        passivo_bkt,
                "gap":            gap,
                "gap_cumulativo": 0,   # preenchido abaixo
                "dv01_ativo":     dv01_ativo,
                "dv01_passivo":   dv01_passivo,
                "dv01_gap":       dv01_ativo - dv01_passivo,
            })

        df = pd.DataFrame(rows)
        df["gap_cumulativo"] = df["gap"].cumsum()
        df.to_csv(OUTPUT_DIR / "gap_reprecificacao.csv", index=False)
        return df

    # ─── NII MENSAL ───────────────────────────────────────────
    def _taxa_efetiva_cenario(self, row: pd.Series, cenario: str,
                               mes: int, tipo: str) -> float:
        """
        Retorna taxa efetiva (% a.a.) de um produto em um dado mês e cenário.
        Para produtos pós-fixados CDI: usa a curva curta do cenário.
        Para pré-fixados: taxa contratada (não varia com cenário no estoque).
        Para IPCA: taxa real + IPCA do cenário.
        Para pré-fixado com repricing: novos contratos usam curva do cenário.
        """
        taxa_orig = row.get("taxa_total", row.get("custo", 0)) or 0
        indexador = row["indexador"]
        prazo     = row["prazo_medio"]

        # Interpola taxa CDI do cenário para prazo 1M (curto)
        di_1m = float(np.interp(1/12, self.curvas_di["tenor_anos"],
                                self.curvas_di[f"di_{cenario}"]))
        di_prazo = float(np.interp(max(prazo, 1/12), self.curvas_di["tenor_anos"],
                                   self.curvas_di[f"di_{cenario}"]))
        ipca_prazo = float(np.interp(max(prazo, 1/12), self.curvas_ipca["tenor_anos"],
                                     self.curvas_ipca[f"ipca_{cenario}"]))

        if indexador == "CDI":
            # Pós-fixado: flutua com CDI do cenário
            if tipo == "ativo":
                spread = taxa_orig - 11.75   # spread sobre CDI base
                return di_1m + spread
            else:
                # passivo CDI: custo em % do CDI
                pct_cdi = taxa_orig if taxa_orig > 10 else 100.0
                return di_1m * (pct_cdi / 100)

        elif indexador == "IPCA":
            spread_real = taxa_orig   # taxa real contratada
            return ipca_prazo + spread_real

        elif indexador == "TR":
            # TR ≈ 0.5% a.a. + spread contratado
            return 0.5 + (taxa_orig - 0.5)

        elif indexador == "PRE":
            # Se já venceu (prazo < mes/12), reprecia com curva nova
            if prazo < mes / 12:
                return di_prazo   # novo contrato usa curva do cenário
            return taxa_orig   # ainda na taxa contratada

        elif indexador == "USD":
            # Simplificação: mantém taxa em USD (ignora variação cambial)
            return taxa_orig

        return taxa_orig

    def projetar_nii(self, horizonte_meses: int = 12) -> Dict[str, pd.DataFrame]:
        """
        Projeta NII mês a mês para 3 cenários.
        Retorna dict com DataFrames por cenário.
        """
        cenarios = ["base", "stress_alta", "stress_baixa"]
        resultados = {}

        for cenario in cenarios:
            meses, nii_list, receita_list, custo_list = [], [], [], []

            for mes in range(1, horizonte_meses + 1):
                receita = 0.0
                for _, row in self.ativos.iterrows():
                    taxa = self._taxa_efetiva_cenario(row, cenario, mes, "ativo")
                    # NII mensal = saldo × taxa_anual / 12
                    receita += row["saldo"] * (taxa / 100) / 12

                custo = 0.0
                for _, row in self.passivos.iterrows():
                    taxa = self._taxa_efetiva_cenario(row, cenario, mes, "passivo")
                    custo += row["saldo"] * (taxa / 100) / 12

                nii = receita - custo
                meses.append(mes)
                nii_list.append(nii)
                receita_list.append(receita)
                custo_list.append(custo)

            df = pd.DataFrame({
                "mes":         meses,
                "receita":     receita_list,
                "custo":       custo_list,
                "nii":         nii_list,
                "nim_pct":     [n / (self.ativos["saldo"].sum() / 12) * 100
                                for n in nii_list],
            })
            df["nii_acumulado"] = df["nii"].cumsum()
            resultados[cenario] = df
            df.to_csv(OUTPUT_DIR / f"nii_{cenario}.csv", index=False)

        return resultados

    # ─── DURATION SUMMARY ────────────────────────────────────
    def duration_summary(self) -> pd.DataFrame:
        """
        Resumo de duration e DV01 por produto, separando ativo e passivo.
        Também calcula o gap de duration do balanço.
        """
        rows = []
        for _, row in self.ativos.iterrows():
            rows.append({
                "tipo": "Ativo", "produto": row["produto"],
                "saldo": row["saldo"], "indexador": row["indexador"],
                "prazo_medio": row["prazo_medio"],
                "duration_mod": row["duration_mod"],
                "dv01": row["dv01"],
            })
        for _, row in self.passivos.iterrows():
            rows.append({
                "tipo": "Passivo", "produto": row["produto"],
                "saldo": row["saldo"], "indexador": row["indexador"],
                "prazo_medio": row["prazo_medio"],
                "duration_mod": row["duration_mod"],
                "dv01": row["dv01"],
            })
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_DIR / "duration_dv01.csv", index=False)
        return df

    def duration_gap(self) -> dict:
        """
        Duration Gap = Duration_Ativo - (Passivo/Ativo) × Duration_Passivo
        Mede exposição líquida do balanço a variações de taxa.
        """
        ta = self.ativos["saldo"].sum()
        tp = self.passivos["saldo"].sum()

        # Duration média ponderada por saldo
        da = np.average(self.ativos["duration_mod"],
                        weights=self.ativos["saldo"])
        dp = np.average(self.passivos["duration_mod"],
                        weights=self.passivos["saldo"])

        dgap = da - (tp / ta) * dp
        dv01_total = self.ativos["dv01"].sum() - self.passivos["dv01"].sum()

        result = {
            "duration_ativo_anos":    round(da, 4),
            "duration_passivo_anos":  round(dp, 4),
            "duration_gap_anos":      round(dgap, 4),
            "dv01_ativo_R$M":         round(self.ativos["dv01"].sum(), 2),
            "dv01_passivo_R$M":       round(self.passivos["dv01"].sum(), 2),
            "dv01_gap_R$M":           round(dv01_total, 2),
            "interpretacao": (
                "LONG duration: banco perde capital se taxas sobem"
                if dgap > 0 else
                "SHORT duration: banco perde capital se taxas caem"
            ),
        }
        with open(OUTPUT_DIR / "duration_gap.json", "w") as f:
            json.dump(result, f, indent=2)
        return result

    def imprimir_gap(self, df_gap: pd.DataFrame):
        print("\n" + "="*85)
        print("  GAP DE REPRECIFICAÇÃO POR BUCKET (R$ milhões)")
        print("="*85)
        print(f"  {'Bucket':<12} {'Ativo':>12} {'Passivo':>12} {'Gap':>12} "
              f"{'Gap Cum.':>12} {'DV01 Gap':>10}")
        print("-"*85)
        for _, r in df_gap.iterrows():
            sinal = "▲" if r["gap"] >= 0 else "▼"
            print(f"  {r['bucket']:<12} {r['ativo']:>12,.0f} {r['passivo']:>12,.0f} "
                  f"{r['gap']:>+12,.0f} {r['gap_cumulativo']:>+12,.0f} "
                  f"{r['dv01_gap']:>+10.1f} {sinal}")
        print("="*85)

    def imprimir_nii(self, resultados: dict):
        print("\n" + "="*70)
        print("  PROJEÇÃO NII ANUAL (12 MESES) — R$ MILHÕES")
        print("="*70)
        print(f"  {'Cenário':<18} {'NII Acum.':>12} {'NIM Médio':>10} "
              f"{'Receita Acum.':>14} {'Custo Acum.':>12}")
        print("-"*70)
        for cen, df in resultados.items():
            print(f"  {cen:<18} "
                  f"{df['nii_acumulado'].iloc[-1]:>12,.0f} "
                  f"{df['nim_pct'].mean():>9.2f}% "
                  f"{df['receita'].sum():>14,.0f} "
                  f"{df['custo'].sum():>12,.0f}")
        print("="*70)


# ─────────────────────────────────────────────────────────────
# EXECUÇÃO DIRETA
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = CashFlowEngine()
    df_gap  = engine.gap_reprecificacao()
    nii     = engine.projetar_nii(12)
    dur_df  = engine.duration_summary()
    dgap    = engine.duration_gap()
    engine.imprimir_gap(df_gap)
    engine.imprimir_nii(nii)
    print("\n📐 Duration Gap do Balanço:")
    for k, v in dgap.items():
        print(f"   {k:<30} {v}")
