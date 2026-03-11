"""
ALM Bank Optimization System
=============================
Ponto de entrada principal do sistema.

Uso:
    python main.py              # roda todos os módulos disponíveis
    python main.py --fase 1     # roda apenas fase 1 (balanço + cenários)
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def fase1():
    """M1: Balanço + M2: Cenários de Mercado + todos os gráficos"""
    print("\n" + "█"*60)
    print("  FASE 1 — Balanço Patrimonial e Cenários de Mercado")
    print("█"*60)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
    import numpy as np

    sns.set_theme(style="whitegrid", palette="muted")

    CHARTS_DIR = BASE_DIR / "outputs" / "charts"
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── M1: Balanço ──────────────────────────────────────────────────────────
    from models.balance_sheet.balance_sheet import BalancoPatrimonial
    banco = BalancoPatrimonial()
    banco.salvar()

    kpis = banco.resumo_kpis()
    print("\n📊 KPIs do Balanço:")
    for k, v in kpis.items():
        print(f"   {k:<30} {v:>12.2f}")

    # Gráfico 1: Composição do Balanço
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Banco Modelo S.A.\nComposição do Balanço — 2024-12-31", fontsize=13, fontweight="bold")
    ativo_cat = banco.ativos.groupby("categoria")["saldo"].sum().sort_values(ascending=False)
    axes[0].barh(ativo_cat.index, ativo_cat.values / 1000,
                 color=sns.color_palette("Blues_d", len(ativo_cat)))
    axes[0].set_title("Ativo por Categoria (R$ bi)", fontweight="bold")
    axes[0].set_xlabel("R$ bilhões")
    for i, (cat, val) in enumerate(ativo_cat.items()):
        axes[0].text(val/1000 + 0.3, i, f"R${val/1000:.1f}bi", va="center", fontsize=8)
    passivo_cat = banco.passivos.groupby("categoria")["saldo"].sum().sort_values(ascending=False)
    axes[1].barh(passivo_cat.index, passivo_cat.values / 1000,
                 color=sns.color_palette("Oranges_d", len(passivo_cat)))
    axes[1].set_title("Passivo por Categoria (R$ bi)", fontweight="bold")
    axes[1].set_xlabel("R$ bilhões")
    for i, (cat, val) in enumerate(passivo_cat.items()):
        axes[1].text(val/1000 + 0.2, i, f"R${val/1000:.1f}bi", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "01_composicao_balanco.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gráfico salvo: outputs/charts/01_composicao_balanco.png")

    # Gráfico 2: RWA e Capital
    fig, ax = plt.subplots(figsize=(8, 5))
    rwa_cat = banco.ativos.groupby("categoria")["rwa"].sum().sort_values(ascending=False)
    rwa_cat = rwa_cat[rwa_cat > 0]
    bars = ax.bar(range(len(rwa_cat)), rwa_cat.values / 1000,
                  color=sns.color_palette("Reds_d", len(rwa_cat)))
    ax.set_xticks(range(len(rwa_cat)))
    ax.set_xticklabels(rwa_cat.index, rotation=35, ha="right", fontsize=9)
    ax.set_title(f"RWA por Categoria — IB: {kpis['indice_basileia_pct']:.1f}% | CET1: {kpis['cet1_ratio_pct']:.1f}%",
                 fontweight="bold")
    ax.set_ylabel("R$ bilhões")
    for bar, val in zip(bars, rwa_cat.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val/1000:.1f}bi", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "02_rwa_capital.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gráfico salvo: outputs/charts/02_rwa_capital.png")

    # Gráfico 3: Mix de Indexadores
    palette = {"CDI": "#2196F3", "PRE": "#4CAF50", "IPCA": "#FF9800", "TR": "#9C27B0", "USD": "#F44336"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Mix de Indexadores — Ativo vs Passivo", fontsize=12, fontweight="bold")
    idx_ativo   = banco.ativos.groupby("indexador")["saldo"].sum()
    idx_passivo = banco.passivos.groupby("indexador")["saldo"].sum()
    axes[0].pie(idx_ativo.values, labels=idx_ativo.index, autopct="%1.1f%%",
                colors=[palette.get(i, "#888") for i in idx_ativo.index], startangle=90)
    axes[0].set_title("Ativo")
    axes[1].pie(idx_passivo.values, labels=idx_passivo.index, autopct="%1.1f%%",
                colors=[palette.get(i, "#888") for i in idx_passivo.index], startangle=90)
    axes[1].set_title("Passivo")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "03_mix_indexadores.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gráfico salvo: outputs/charts/03_mix_indexadores.png")

    # ── M2: Cenários ─────────────────────────────────────────────────────────
    from models.balance_sheet.market_scenarios import MarketScenarios
    cenarios = MarketScenarios()
    cenarios.salvar()
    cenarios.imprimir_resumo()

    df_di   = cenarios.to_dataframe_di()
    df_ipca = cenarios.to_dataframe_ipca()

    # Gráfico 4: Curvas de Juros
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Curvas de Juros — Cenários ALM (Dez/2024)", fontsize=13, fontweight="bold")
    ax = axes[0]
    ax.plot(df_di["tenor_anos"], df_di["di_base"],         "b-o",  ms=4, lw=2,   label="DI Base")
    ax.plot(df_di["tenor_anos"], df_di["di_stress_alta"],  "r--s", ms=4, lw=1.5, label="Stress Alta (+200bps)")
    ax.plot(df_di["tenor_anos"], df_di["di_stress_baixa"], "g--^", ms=4, lw=1.5, label="Stress Baixa (-250bps)")
    ax.fill_between(df_di["tenor_anos"], df_di["di_stress_baixa"], df_di["di_stress_alta"],
                    alpha=0.08, color="gray", label="Envelope")
    ax.set_title("Curva DI (Pré-fixada)", fontweight="bold")
    ax.set_xlabel("Prazo (anos)"); ax.set_ylabel("Taxa % a.a.")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4); ax.set_xlim(0, 10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax2 = axes[1]
    ax2.plot(df_ipca["tenor_anos"], df_ipca["ipca_base"],         "b-o",  ms=4, lw=2,   label="IPCA Base")
    ax2.plot(df_ipca["tenor_anos"], df_ipca["ipca_stress_alta"],  "r--s", ms=4, lw=1.5, label="IPCA Stress Alta")
    ax2.plot(df_ipca["tenor_anos"], df_ipca["ipca_stress_baixa"], "g--^", ms=4, lw=1.5, label="IPCA Stress Baixa")
    ax2.plot(df_ipca["tenor_anos"], df_ipca["real_base"],         "k-.",  ms=3, lw=1.5, label="Taxa Real Base")
    ax2.set_title("Inflação Implícita (IPCA) e Taxa Real", fontweight="bold")
    ax2.set_xlabel("Prazo (anos)"); ax2.set_ylabel("Taxa % a.a.")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.4); ax2.set_xlim(0, 10)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "04_curvas_juros_cenarios.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gráfico salvo: outputs/charts/04_curvas_juros_cenarios.png")

    # Gráfico 5: Spreads de Crédito
    spreads  = cenarios.spreads_credito
    produtos = list(spreads["base"].keys())
    x = np.arange(len(produtos)); w = 0.28
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, [spreads["base"][p]        for p in produtos], w, label="Base",        color="#2196F3")
    ax.bar(x,     [spreads["stress_alta"][p]  for p in produtos], w, label="Stress Alta",  color="#F44336", alpha=0.8)
    ax.bar(x + w, [spreads["stress_baixa"][p] for p in produtos], w, label="Stress Baixa", color="#4CAF50", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in produtos], fontsize=8)
    ax.set_title("Spreads de Crédito por Segmento e Cenário (% a.a.)", fontweight="bold")
    ax.set_ylabel("Spread % a.a."); ax.legend(); ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "05_spreads_credito.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gráfico salvo: outputs/charts/05_spreads_credito.png")

    print("\n✅ Fase 1 concluída! Todos os gráficos em: outputs/charts/")
    return banco, cenarios


def main():
    parser = argparse.ArgumentParser(description="ALM Bank Optimization System")
    parser.add_argument("--fase", type=int, default=0,
                        help="Fase a executar (0=todas, 1=balanço+cenários)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("   ALM BANK OPTIMIZATION SYSTEM")
    print("   Doutorado Poli-USP | Banco Modelo S.A.")
    print("="*60)

    if args.fase in (0, 1):
        fase1()

    print("\n" + "="*60)
    print("  EXECUÇÃO CONCLUÍDA")
    print(f"  Gráficos em: outputs/charts/")
    print(f"  Dados em:    data/processed/ e data/scenarios/")
    print("="*60 + "\n")




def fase2():
    """M3: Cash Flow Engine — Gap de Reprecificação, NII e Duration"""
    print("\n" + "█"*60)
    print("  FASE 2 — Motor de Fluxos de Caixa e NII")
    print("█"*60)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    CHARTS_DIR = BASE_DIR / "outputs" / "charts"
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    from models.cash_flow.cash_flow_engine import CashFlowEngine, BUCKET_LABELS

    engine = CashFlowEngine()
    df_gap = engine.gap_reprecificacao()
    nii    = engine.projetar_nii(12)
    dur_df = engine.duration_summary()
    dgap   = engine.duration_gap()

    engine.imprimir_gap(df_gap)
    engine.imprimir_nii(nii)
    print("\n📐 Duration Gap do Balanço:")
    for k, v in dgap.items():
        print(f"   {k:<30} {v}")

    # ── GRÁFICO 6: Gap de Reprecificação ─────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("Gap de Reprecificação por Bucket Temporal", fontsize=13, fontweight="bold")

    x = np.arange(len(df_gap))
    w = 0.35
    ax = axes[0]
    bars_a = ax.bar(x - w/2, df_gap["ativo"]   / 1000, w, label="Ativo",   color="#1565C0", alpha=0.85)
    bars_p = ax.bar(x + w/2, df_gap["passivo"] / 1000, w, label="Passivo", color="#E65100", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(BUCKET_LABELS, fontsize=9)
    ax.set_ylabel("R$ bilhões"); ax.legend()
    ax.set_title("Saldo Ativo vs Passivo por Bucket", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    ax2 = axes[1]
    colors_gap = ["#1B5E20" if g >= 0 else "#B71C1C" for g in df_gap["gap"]]
    ax2.bar(x, df_gap["gap"] / 1000, color=colors_gap, alpha=0.85, label="Gap")
    ax2.plot(x, df_gap["gap_cumulativo"] / 1000, "k--o", ms=5, lw=2, label="Gap Cumulativo")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(BUCKET_LABELS, fontsize=9)
    ax2.set_ylabel("R$ bilhões"); ax2.legend()
    ax2.set_title("Gap (Ativo − Passivo) e Gap Cumulativo por Bucket", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    for i, (g, gc) in enumerate(zip(df_gap["gap"], df_gap["gap_cumulativo"])):
        ax2.text(i, g/1000 + (1 if g >= 0 else -3), f"{g/1000:+.0f}", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "06_gap_reprecificacao.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gráfico salvo: outputs/charts/06_gap_reprecificacao.png")

    # ── GRÁFICO 7: NII Projetado — 3 Cenários ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Projeção de NII — 12 Meses | 3 Cenários", fontsize=13, fontweight="bold")

    cores = {"base": "#1565C0", "stress_alta": "#B71C1C", "stress_baixa": "#2E7D32"}
    rotulos = {"base": "Base", "stress_alta": "Stress Alta (+200bps)", "stress_baixa": "Stress Baixa (−250bps)"}

    ax = axes[0]
    for cen, df in nii.items():
        ax.plot(df["mes"], df["nii"], "-o", ms=4, lw=2,
                color=cores[cen], label=rotulos[cen])
    ax.set_title("NII Mensal (R$ milhões)", fontweight="bold")
    ax.set_xlabel("Mês"); ax.set_ylabel("R$ milhões")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}M"))

    ax2 = axes[1]
    for cen, df in nii.items():
        ax2.plot(df["mes"], df["nii_acumulado"], "-o", ms=4, lw=2,
                 color=cores[cen], label=rotulos[cen])
    ax2.set_title("NII Acumulado (R$ milhões)", fontweight="bold")
    ax2.set_xlabel("Mês"); ax2.set_ylabel("R$ milhões")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.4)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}M"))

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "07_nii_cenarios.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gráfico salvo: outputs/charts/07_nii_cenarios.png")

    # ── GRÁFICO 8: NIM por Cenário ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for cen, df in nii.items():
        ax.plot(df["mes"], df["nim_pct"], "-o", ms=4, lw=2,
                color=cores[cen], label=rotulos[cen])
    ax.set_title("NIM — Net Interest Margin (% a.a. mensal)", fontweight="bold")
    ax.set_xlabel("Mês"); ax.set_ylabel("NIM %")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "08_nim_cenarios.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gráfico salvo: outputs/charts/08_nim_cenarios.png")

    # ── GRÁFICO 9: Duration e DV01 por Produto ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Duration Modificada e DV01 por Produto", fontsize=13, fontweight="bold")

    ativos_dur   = dur_df[dur_df["tipo"] == "Ativo"].nlargest(12, "dv01")
    passivos_dur = dur_df[dur_df["tipo"] == "Passivo"].nlargest(8, "dv01")

    ax = axes[0]
    ax.barh(ativos_dur["produto"],   ativos_dur["dv01"],   color="#1565C0", alpha=0.85, label="DV01 Ativo")
    ax.barh(passivos_dur["produto"], -passivos_dur["dv01"], color="#E65100", alpha=0.85, label="DV01 Passivo (−)")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("DV01 por Produto (R$M / bp)", fontweight="bold")
    ax.set_xlabel("DV01 (R$ milhões por 1bp)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="x")

    ax2 = axes[1]
    dur_pivot = dur_df.groupby("tipo").apply(
        lambda x: np.average(x["duration_mod"], weights=x["saldo"])
    ).reset_index()
    dur_pivot.columns = ["tipo", "duration_media"]
    bars = ax2.bar(dur_pivot["tipo"], dur_pivot["duration_media"],
                   color=["#1565C0", "#E65100"], alpha=0.85, width=0.4)
    for bar, v in zip(bars, dur_pivot["duration_media"]):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                 f"{v:.2f}a", ha="center", fontsize=11, fontweight="bold")
    ax2.set_title(f"Duration Média Ponderada\nDuration Gap: {dgap['duration_gap_anos']:.2f} anos",
                  fontweight="bold")
    ax2.set_ylabel("Duration Modificada (anos)")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0, max(dur_pivot["duration_media"]) * 1.3)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "09_duration_dv01.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gráfico salvo: outputs/charts/09_duration_dv01.png")

    # ── GRÁFICO 10: Sensibilidade NII a Choques de Taxa ──────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    choques = [-300, -200, -100, -50, 0, 50, 100, 200, 300]
    nii_base_anual = nii["base"]["nii"].sum()
    nii_alta_anual = nii["stress_alta"]["nii"].sum()
    nii_baixa_anual = nii["stress_baixa"]["nii"].sum()

    # Interpolação linear simples para os pontos intermediários
    # base = 0bps, stress_alta ≈ +200bps, stress_baixa ≈ -250bps
    nii_choques = []
    for choque in choques:
        if choque >= 0:
            frac = choque / 200
            nii_c = nii_base_anual + frac * (nii_alta_anual - nii_base_anual)
        else:
            frac = abs(choque) / 250
            nii_c = nii_base_anual + frac * (nii_baixa_anual - nii_base_anual)
        nii_choques.append(nii_c)

    delta_nii = [n - nii_base_anual for n in nii_choques]
    colors = ["#B71C1C" if d < 0 else "#1B5E20" for d in delta_nii]
    ax.bar([str(c) for c in choques], delta_nii, color=colors, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Choque paralelo na curva de juros (bps)")
    ax.set_ylabel("Variação no NII anual (R$ milhões)")
    ax.set_title("Sensibilidade do NII a Choques de Taxa (IRRBB)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for i, (d, bar_x) in enumerate(zip(delta_nii, choques)):
        ax.text(i, d + (50 if d >= 0 else -150), f"R${d:+,.0f}M",
                ha="center", fontsize=7.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "10_sensibilidade_nii.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gráfico salvo: outputs/charts/10_sensibilidade_nii.png")

    print("\n✅ Fase 2 concluída! 5 novos gráficos em: outputs/charts/")
    return engine, df_gap, nii, dgap


def main():
    parser = argparse.ArgumentParser(description="ALM Bank Optimization System")
    parser.add_argument("--fase", type=int, default=0,
                        help="Fase a executar (0=todas, 1..4 individual)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("   ALM BANK OPTIMIZATION SYSTEM")
    print("   Doutorado Poli-USP | Banco Modelo S.A.")
    print("="*60)

    if args.fase in (0, 1):
        fase1()

    if args.fase in (0, 2):
        fase2()

    if args.fase in (0, 3):
        fase3()

    if args.fase in (0, 4):
        fase4()

    if args.fase in (0, 5):
        fase5()
    # if args.fase in (0, 4): fase4()   # Hedge + Liquidez
    # if args.fase in (0, 5): fase5()   # Capital/RWA

    print("\n" + "="*60)
    print("  EXECUCAO CONCLUIDA")
    print(f"  Graficos em: outputs/charts/")
    print(f"  Dados em:    data/processed/ e data/scenarios/")
    print("="*60 + "\n")




def fase3():
    """M4: Otimizador ALM — alocação ótima + fronteira eficiente"""
    print("\n" + "█"*60)
    print("  FASE 3 — Otimizador ALM (QP Baseline)")
    print("█"*60)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    CHARTS_DIR = BASE_DIR / "outputs" / "charts"
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    from models.optimization.alm_optimizer import ALMOptimizer

    opt = ALMOptimizer(cenario="base")
    opt.mercado.resumo()

    # ── Otimização pontual (λ baixo = foco em NII) ──────────────────────────
    print("\n  Ponto 1 — λ=0.0001 (max NII, pouco hedge):")
    res_nii = opt.otimizar(lambda_risco=0.0001)
    opt.imprimir_resultado(res_nii)

    print("\n  Ponto 2 — λ=0.01 (equilíbrio NII × risco):")
    res_bal = opt.otimizar(lambda_risco=0.01)
    opt.imprimir_resultado(res_bal)

    print("\n  Ponto 3 — λ=0.5 (foco em redução de risco):")
    res_rsk = opt.otimizar(lambda_risco=0.5)
    opt.imprimir_resultado(res_rsk)

    # ── Fronteira eficiente ─────────────────────────────────────────────────
    df_fe = opt.fronteira_eficiente(n_pontos=12)

    # ── GRÁFICO 11: Fronteira Eficiente NII × Risco ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fronteira Eficiente ALM — NII × Risco de Taxa", fontsize=13, fontweight="bold")

    ax = axes[0]
    sc = ax.scatter(df_fe["dv01_gap"], df_fe["nii_total"] / 1000,
                    c=np.log10(df_fe["lambda"]), cmap="RdYlGn_r", s=80, zorder=5)
    ax.plot(df_fe["dv01_gap"], df_fe["nii_total"] / 1000, "k--", lw=1, alpha=0.4)
    plt.colorbar(sc, ax=ax, label="log₁₀(λ)")
    ax.set_xlabel("DV01 Gap (R$M / bp) — risco de taxa")
    ax.set_ylabel("NII Total Anual (R$ bilhões)")
    ax.set_title("Fronteira Eficiente: NII × DV01 Gap", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Destaca 3 pontos
    for res, label, color in [
        (res_nii, "Max NII", "green"),
        (res_bal, "Equilíbrio", "orange"),
        (res_rsk, "Min Risco", "red"),
    ]:
        ax.scatter(abs(res.dv01_gap_pos), res.nii_total / 1000,
                   color=color, s=150, zorder=10, label=label,
                   edgecolors="black", linewidths=1.2)
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.scatter(df_fe["duration_gap"], df_fe["nii_total"] / 1000,
                c=np.log10(df_fe["lambda"]), cmap="RdYlGn_r", s=80, zorder=5)
    ax2.plot(df_fe["duration_gap"], df_fe["nii_total"] / 1000, "k--", lw=1, alpha=0.4)
    ax2.axvline(2.0, color="red", lw=1.5, ls="--", alpha=0.7, label="Limite IRRBB (2a)")
    ax2.set_xlabel("Duration Gap (anos) — risco de taxa")
    ax2.set_ylabel("NII Total Anual (R$ bilhões)")
    ax2.set_title("Fronteira Eficiente: NII × Duration Gap", fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "11_fronteira_eficiente.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gráfico salvo: outputs/charts/11_fronteira_eficiente.png")

    # ── GRÁFICO 12: Alocação Ótima de Crédito ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Alocação Ótima por Estratégia", fontsize=13, fontweight="bold")

    cenarios_res = [
        (res_nii, "Max NII\n(λ=0.0001)", "#1B5E20"),
        (res_bal, "Equilíbrio\n(λ=0.01)",   "#E65100"),
        (res_rsk, "Min Risco\n(λ=0.5)",     "#B71C1C"),
    ]

    segs_nomes = list(res_nii.credito_novo.keys())
    x_pos = np.arange(len(segs_nomes))

    for ax, (res, titulo, cor) in zip(axes, cenarios_res):
        vals = [res.credito_novo[s] for s in segs_nomes]
        bars = ax.bar(x_pos, vals, color=cor, alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.replace("_", "\n") for s in segs_nomes], fontsize=7)
        ax.set_title(titulo, fontweight="bold")
        ax.set_ylabel("R$ milhões")
        ax.grid(True, alpha=0.3, axis="y")
        total = sum(vals)
        ax.set_xlabel(f"Total: R${total:,.0f}M", fontsize=9)
        for bar, v in zip(bars, vals):
            if v > 50:
                ax.text(bar.get_x() + bar.get_width()/2, v + 20,
                        f"{v:,.0f}", ha="center", fontsize=6.5)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "12_alocacao_credito.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gráfico salvo: outputs/charts/12_alocacao_credito.png")

    # ── GRÁFICO 13: Mix de Captação Ótimo ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Mix Ótimo de Captação por Estratégia", fontsize=13, fontweight="bold")

    caps_nomes = list(res_nii.captacao_nova.keys())
    cores_cap  = ["#1565C0", "#1976D2", "#1E88E5", "#42A5F5", "#64B5F6", "#90CAF9"]

    for ax, (res, titulo, _) in zip(axes, cenarios_res):
        vals = [res.captacao_nova[c] for c in caps_nomes]
        total = sum(vals)
        if total > 0:
            ax.pie(vals, labels=[c.replace("_", "\n") for c in caps_nomes],
                   autopct="%1.0f%%", colors=cores_cap, startangle=90,
                   textprops={"fontsize": 7})
        ax.set_title(f"{titulo}\nTotal: R${total:,.0f}M", fontweight="bold", fontsize=9)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "13_mix_captacao.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gráfico salvo: outputs/charts/13_mix_captacao.png")

    # ── GRÁFICO 14: Restrições Regulatórias ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Atendimento às Restrições Regulatórias", fontsize=13, fontweight="bold")

    metricas = [
        ("LCR (%)", [res_nii.lcr_pos*100, res_bal.lcr_pos*100, res_rsk.lcr_pos*100],
         100, "LCR mín 100%"),
        ("NSFR (%)", [res_nii.nsfr_pos*100, res_bal.nsfr_pos*100, res_rsk.nsfr_pos*100],
         100, "NSFR mín 100%"),
        ("Índice Basileia (%)", [res_nii.ib_pos*100, res_bal.ib_pos*100, res_rsk.ib_pos*100],
         10.5, "IB mín 10.5%"),
    ]
    rotulos = ["Max NII", "Equilíbrio", "Min Risco"]
    cores_bar = ["#1B5E20", "#E65100", "#B71C1C"]

    for ax, (titulo, vals, minimo, label_min) in zip(axes, metricas):
        bars = ax.bar(rotulos, vals, color=cores_bar, alpha=0.85)
        ax.axhline(minimo, color="red", lw=2, ls="--", label=label_min)
        ax.set_title(titulo, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                    f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
        y_min = min(minimo * 0.9, min(vals) * 0.95)
        y_max = max(vals) * 1.08
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "14_restricoes_regulatorias.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gráfico salvo: outputs/charts/14_restricoes_regulatorias.png")

    # ── GRÁFICO 15: Hedge e DV01 Gap ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Estratégia de Hedge e Evolução do DV01 Gap", fontsize=13, fontweight="bold")

    ax = axes[0]
    hdg_nomes = [h for h, v in res_bal.hedge_nocional.items() if abs(v) > 1]
    hdg_vals  = [res_bal.hedge_nocional[h] for h in hdg_nomes]
    cores_hdg = ["#1B5E20" if v > 0 else "#B71C1C" for v in hdg_vals]
    ax.barh(hdg_nomes, hdg_vals, color=cores_hdg, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Posição em Hedge — Equilíbrio (λ=0.01)\n(+ = paga flutuante, − = recebe flutuante)",
                 fontweight="bold", fontsize=9)
    ax.set_xlabel("Nocional (R$ milhões)")
    ax.grid(True, alpha=0.3, axis="x")

    ax2 = axes[1]
    estrategias = ["Base\n(sem otimização)", "Max NII\n(λ=0.0001)",
                   "Equilíbrio\n(λ=0.01)", "Min Risco\n(λ=0.5)"]
    dv01_vals = [
        opt.dv01_gap_base,
        res_nii.dv01_gap_pos,
        res_bal.dv01_gap_pos,
        res_rsk.dv01_gap_pos,
    ]
    cores_dv = ["gray", "#1B5E20", "#E65100", "#B71C1C"]
    bars = ax2.bar(estrategias, dv01_vals, color=cores_dv, alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("DV01 Gap por Estratégia (R$M/bp)\nPositivo = LONG duration",
                  fontweight="bold", fontsize=9)
    ax2.set_ylabel("DV01 Gap (R$M / 1bp)")
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, dv01_vals):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 v + (0.5 if v >= 0 else -1.5),
                 f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "15_hedge_dv01.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gráfico salvo: outputs/charts/15_hedge_dv01.png")

    print("\n✅ Fase 3 concluída! 5 novos gráficos em: outputs/charts/")
    return opt, res_nii, res_bal, res_rsk, df_fe




def fase4():
    """M6+M7: Liquidez (LCR/NSFR/Stress) e Capital (RWA/RAROC/Pricing)"""
    print("\n" + "█"*60)
    print("  FASE 4 — Liquidez e Capital")
    print("█"*60)

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    CHARTS_DIR = BASE_DIR / "outputs" / "charts"
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    from models.liquidity.liquidity_module import LiquidityModule
    from models.capital.capital_module import CapitalModule, RAROC_ALVO

    liq = LiquidityModule()
    cap = CapitalModule()

    # ── Cálculos ─────────────────────────────────────────────────────────────
    lcr_res  = liq.stress_test_lcr()
    liq.imprimir_lcr(lcr_res)

    nsfr_res = liq.calcular_nsfr_detalhado()
    print(f"\n  NSFR Detalhado:")
    print(f"    ASF Total: R$ {nsfr_res['asf_total']:,.0f}M")
    print(f"    RSF Total: R$ {nsfr_res['rsf_total']:,.0f}M")
    print(f"    NSFR:      {nsfr_res['nsfr_pct']:.1f}%")

    df_proj  = liq.projecao_dinamica(12)
    df_hqla  = liq.custo_buffer_hqla()
    df_rwa   = cap.calcular_rwa_detalhado()
    df_ce    = cap.capital_economico_segmento()
    df_raroc = cap.calcular_raroc(df_ce)
    df_prec  = cap.pricing_raroc_minimo(df_ce)
    cap.imprimir_raroc(df_raroc)
    cap.imprimir_pricing(df_prec)

    cores_cen = {"base": "#1565C0", "stress_alta": "#B71C1C", "stress_baixa": "#2E7D32"}
    rotulos_cen = {"base": "Base", "stress_alta": "Stress Alta", "stress_baixa": "Stress Baixa"}

    # ── GRÁFICO 16: Stress Test LCR — Waterfall ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Stress Test de Liquidez — 30 Dias (Basileia III)", fontsize=13, fontweight="bold")

    ax = axes[0]
    categorias = ["HQLA\nNível 1", "HQLA\nNível 2A", "HQLA\nNível 2B",
                  "Outflows\nBrutos", "Inflows\n(cap 75%)", "NCO", "HQLA\nTotal"]
    valores = [
        lcr_res["hqla_nivel1"], lcr_res["hqla_nivel2A"], lcr_res["hqla_nivel2B"],
        -lcr_res["outflow_total"], lcr_res["inflow_capped"],
        -lcr_res["nco"], lcr_res["hqla_total"],
    ]
    cores_wf = ["#1B5E20","#388E3C","#81C784","#B71C1C","#EF9A9A","#FF6F00","#1565C0"]
    bars = ax.bar(categorias, [abs(v)/1000 for v in valores], color=cores_wf, alpha=0.85)
    ax.set_ylabel("R$ bilhões")
    ax.set_title("Componentes do LCR", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, valores):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                f"R${abs(v)/1000:.0f}bi", ha="center", fontsize=8)

    ax2 = axes[1]
    # Decomposição outflows por produto (top 8)
    out_df = pd.DataFrame([
        {"produto": k, "outflow": v}
        for k, v in lcr_res["outflows_detalhado"].items() if v > 100
    ]).sort_values("outflow", ascending=True).tail(8)
    ax2.barh(out_df["produto"], out_df["outflow"]/1000, color="#B71C1C", alpha=0.8)
    ax2.set_xlabel("R$ bilhões")
    ax2.set_title(f"Top Outflows em Stress\nLCR = {lcr_res['lcr_pct']:.1f}%", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "16_stress_test_lcr.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gráfico salvo: outputs/charts/16_stress_test_lcr.png")

    # ── GRÁFICO 17: Projeção Dinâmica LCR/NSFR ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Projeção Dinâmica LCR e NSFR — 12 Meses", fontsize=13, fontweight="bold")

    for cen in ["base", "stress_alta", "stress_baixa"]:
        df_c = df_proj[df_proj["cenario"] == cen]
        axes[0].plot(df_c["mes"], df_c["lcr_pct"],  "-o", ms=4, lw=2,
                     color=cores_cen[cen], label=rotulos_cen[cen])
        axes[1].plot(df_c["mes"], df_c["nsfr_pct"], "-o", ms=4, lw=2,
                     color=cores_cen[cen], label=rotulos_cen[cen])

    for ax, titulo, minimo in [(axes[0], "LCR (%)", 100), (axes[1], "NSFR (%)", 100)]:
        ax.axhline(minimo, color="red", lw=2, ls="--", label=f"Mínimo {minimo}%")
        ax.set_xlabel("Mês"); ax.set_ylabel("% "); ax.set_title(titulo, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "17_projecao_lcr_nsfr.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gráfico salvo: outputs/charts/17_projecao_lcr_nsfr.png")

    # ── GRÁFICO 18: Custo do Buffer HQLA ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Custo de Carregamento do Buffer HQLA", fontsize=13, fontweight="bold")

    df_h = df_hqla[df_hqla["custo_oportunidade_BRL_aa"] > 0].sort_values(
        "custo_oportunidade_BRL_aa", ascending=False)
    cores_nivel = {"nivel1": "#1B5E20", "nivel2A": "#F57F17", "nivel2B": "#B71C1C"}

    ax = axes[0]
    bars = ax.barh(df_h["produto"],
                   df_h["custo_oportunidade_BRL_aa"],
                   color=[cores_nivel.get(n, "#888") for n in df_h["nivel_hqla"]],
                   alpha=0.85)
    ax.set_xlabel("Custo de Oportunidade (R$ milhões/ano)")
    ax.set_title("Custo por Produto HQLA", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    for bar, v in zip(bars, df_h["custo_oportunidade_BRL_aa"]):
        ax.text(v+10, bar.get_y()+bar.get_height()/2,
                f"R${v:,.0f}M", va="center", fontsize=7.5)

    ax2 = axes[1]
    resumo_nivel = df_hqla.groupby("nivel_hqla").agg(
        saldo=("saldo_R$M", "sum"),
        custo=("custo_oportunidade_BRL_aa", "sum")
    ).reset_index()
    ax2_twin = ax2.twinx()
    b1 = ax2.bar(resumo_nivel["nivel_hqla"],
                 resumo_nivel["saldo"]/1000,
                 color=[cores_nivel.get(n,"#888") for n in resumo_nivel["nivel_hqla"]],
                 alpha=0.7, label="Saldo (R$ bi)")
    ax2_twin.plot(resumo_nivel["nivel_hqla"],
                  resumo_nivel["custo"],
                  "ko--", ms=8, lw=2, label="Custo Oportunidade")
    ax2.set_ylabel("Saldo (R$ bilhões)"); ax2.set_xlabel("Nível HQLA")
    ax2_twin.set_ylabel("Custo Oportunidade (R$ M/ano)")
    ax2.set_title(f"Resumo por Nível HQLA\nCusto Total: R${df_hqla['custo_oportunidade_BRL_aa'].sum():,.0f}M/ano",
                  fontweight="bold")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labels1+labels2, fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "18_custo_buffer_hqla.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gráfico salvo: outputs/charts/18_custo_buffer_hqla.png")

    # ── GRÁFICO 19: RAROC por Linha de Negócio ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("RAROC por Linha de Negócio", fontsize=13, fontweight="bold")

    df_r = df_raroc.sort_values("raroc_pct", ascending=True)
    cores_raroc = ["#1B5E20" if v >= RAROC_ALVO*100 else "#B71C1C"
                   for v in df_r["raroc_pct"]]
    ax = axes[0]
    bars = ax.barh(df_r["produto"], df_r["raroc_pct"],
                   color=cores_raroc, alpha=0.85)
    ax.axvline(RAROC_ALVO*100, color="orange", lw=2.5, ls="--",
               label=f"Hurdle Rate {RAROC_ALVO:.0%}")
    ax.set_xlabel("RAROC (% a.a.)"); ax.legend(fontsize=9)
    ax.set_title("RAROC por Segmento", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    for bar, v in zip(bars, df_r["raroc_pct"]):
        ax.text(v + 0.3, bar.get_y()+bar.get_height()/2,
                f"{v:.1f}%", va="center", fontsize=8)

    ax2 = axes[1]
    df_r2 = df_raroc.sort_values("eva_R$M", ascending=True)
    cores_eva = ["#1B5E20" if v >= 0 else "#B71C1C" for v in df_r2["eva_R$M"]]
    bars2 = ax2.barh(df_r2["produto"], df_r2["eva_R$M"],
                     color=cores_eva, alpha=0.85)
    ax2.axvline(0, color="black", lw=1)
    ax2.set_xlabel("EVA (R$ milhões/ano)")
    ax2.set_title("EVA — Lucro Econômico por Segmento", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")
    for bar, v in zip(bars2, df_r2["eva_R$M"]):
        ax2.text(v + (20 if v >= 0 else -80),
                 bar.get_y()+bar.get_height()/2,
                 f"R${v:,.0f}M", va="center", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "19_raroc_linhas_negocio.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gráfico salvo: outputs/charts/19_raroc_linhas_negocio.png")

    # ── GRÁFICO 20: Pricing RAROC + Capital Econômico ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Pricing Mínimo via RAROC e Capital Econômico", fontsize=13, fontweight="bold")

    df_p = df_prec.sort_values("folga_spread_pct", ascending=True)
    cores_p = ["#1B5E20" if v >= 0 else "#B71C1C" for v in df_p["folga_spread_pct"]]
    ax = axes[0]
    bars = ax.barh(df_p["produto"], df_p["folga_spread_pct"],
                   color=cores_p, alpha=0.85)
    ax.axvline(0, color="black", lw=1.5, ls="--")
    ax.set_xlabel("Folga de Spread vs. Mínimo RAROC (p.p.)")
    ax.set_title("Folga Spread Atual vs. Mínimo Econômico", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    for bar, v in zip(bars, df_p["folga_spread_pct"]):
        ax.text(v + (0.1 if v >= 0 else -0.5),
                bar.get_y()+bar.get_height()/2,
                f"{v:+.1f}pp", va="center", fontsize=8)

    ax2 = axes[1]
    df_ce_plot = df_ce.sort_values("ce_pct_ead", ascending=True)
    x_pos = np.arange(len(df_ce_plot))
    bars1 = ax2.bar(x_pos - 0.2, df_ce_plot["pe_R$M"]/df_ce_plot["ead_R$M"]*100,
                    0.35, label="Perda Esperada (PE/EAD %)", color="#FF6F00", alpha=0.8)
    bars2 = ax2.bar(x_pos + 0.2, df_ce_plot["ce_pct_ead"],
                    0.35, label="Capital Econômico (CE/EAD %)", color="#1565C0", alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([p.replace("Crédito ","").replace(" PF","").replace(" PJ","")
                         for p in df_ce_plot["produto"]], rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("% do EAD"); ax2.legend(fontsize=8)
    ax2.set_title("PE e Capital Econômico como % do EAD", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "20_pricing_capital_economico.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Gráfico salvo: outputs/charts/20_pricing_capital_economico.png")

    print("\n✅ Fase 4 concluída! 5 novos gráficos em: outputs/charts/")
    return liq, cap, lcr_res, df_raroc, df_prec




def fase5():
    """M8: Dashboard integrado com dados reais de todas as fases"""
    print("\n" + "█"*60)
    print("  FASE 5 — Dashboard Integrado")
    print("█"*60)
    from models.dashboard.dashboard_generator import load_all_data, generate_dashboard
    print("  Carregando dados de todas as fases...")
    d = load_all_data()
    path = generate_dashboard(d)
    print(f"\n  Dashboard salvo em: {path}")
    print("  Abra o arquivo .html no navegador para visualizar.")
    print("    04 · Otimizador       — Fronteira eficiente QP (SLSQP)")
    print("    05 · Liquidez         — LCR/NSFR Stress + Custo HQLA + Projeção")
    print("    06 · Capital & RAROC  — Capital Econômico + Pricing por segmento")
    return path




def fase5():
    """M8: Dashboard integrado com dados reais de todas as fases"""
    print("\n" + "█"*60)
    print("  FASE 5 — Dashboard Integrado")
    print("█"*60)
    from models.dashboard.dashboard_generator import load_all_data, generate_dashboard
    print("  Carregando dados de todas as fases...")
    d = load_all_data()
    path = generate_dashboard(d)
    print(f"\n  Dashboard salvo em: {path}")
    print("  Abra o arquivo .html no navegador para visualizar.")
    return path

if __name__ == "__main__":
    main()




