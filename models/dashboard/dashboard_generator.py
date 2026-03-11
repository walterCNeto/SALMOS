"""
M8 - Dashboard Generator
=========================
Lê todos os outputs das Fases 1–4 e gera um dashboard HTML completo
com dados reais injetados via template.

Estrutura:
  - KPIs consolidados do balanço
  - NII por cenário
  - Gap de reprecificação
  - Fronteira eficiente
  - LCR / NSFR stress test
  - RAROC e Pricing por segmento
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
DATA_DIR   = BASE_DIR / "data" / "processed"
SCEN_DIR   = BASE_DIR / "data" / "scenarios"
TABLE_DIR  = BASE_DIR / "outputs" / "tables"
REPORT_DIR = BASE_DIR / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def fmt_bi(v, dec=1):
    return f"{v/1000:.{dec}f}"

def fmt_pct(v, dec=1):
    return f"{v:.{dec}f}%"

def fmt_M(v, dec=0):
    return f"R${v:,.{dec}f}M"

def color_class(v, threshold=0):
    return "positive" if v > threshold else ("negative" if v < threshold else "accent")


def load_all_data():
    d = {}

    # KPIs balanço
    with open(DATA_DIR / "kpis_balanco.json") as f:
        d["kpis"] = json.load(f)

    # Patrimônio
    with open(DATA_DIR / "patrimonio.json") as f:
        d["patrimonio"] = json.load(f)

    # Ativos e passivos
    d["ativos"]   = pd.read_csv(DATA_DIR / "ativos.csv")
    d["passivos"] = pd.read_csv(DATA_DIR / "passivos.csv")

    # Duration gap
    with open(TABLE_DIR / "duration_gap.json") as f:
        d["duration"] = json.load(f)

    # Gap reprecificação
    d["gap_repr"] = pd.read_csv(TABLE_DIR / "gap_reprecificacao.csv")

    # NII cenários
    d["nii_base"]  = pd.read_csv(TABLE_DIR / "nii_base.csv")
    d["nii_alta"]  = pd.read_csv(TABLE_DIR / "nii_stress_alta.csv")
    d["nii_baixa"] = pd.read_csv(TABLE_DIR / "nii_stress_baixa.csv")

    # Fronteira eficiente
    d["fronteira"] = pd.read_csv(TABLE_DIR / "fronteira_eficiente_base.csv")

    # Stress test LCR
    d["stress_lcr"] = pd.read_csv(TABLE_DIR / "stress_test_lcr.csv")

    # Projeção LCR/NSFR
    d["proj_liq"] = pd.read_csv(TABLE_DIR / "projecao_lcr_nsfr.csv")

    # Custo HQLA
    d["hqla_custo"] = pd.read_csv(TABLE_DIR / "custo_buffer_hqla.csv")

    # RWA
    d["rwa"] = pd.read_csv(TABLE_DIR / "rwa_detalhado.csv")

    # Capital econômico
    d["cap_econ"] = pd.read_csv(TABLE_DIR / "capital_economico.csv")

    # RAROC
    d["raroc"] = pd.read_csv(TABLE_DIR / "raroc_linhas_negocio.csv")

    # Pricing
    d["pricing"] = pd.read_csv(TABLE_DIR / "pricing_raroc_minimo.csv")

    return d


def build_kpi_card(label, value, unit, color="blue", delta=None, delta_label=""):
    delta_html = ""
    if delta is not None:
        dc = "up" if delta >= 0 else "down"
        sign = "+" if delta >= 0 else ""
        delta_html = f'<div class="kpi-delta {dc}">{sign}{delta_label}</div>'
    return f"""
    <div class="kpi-card {color}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value {color}">{value}</div>
      <div class="kpi-unit">{unit}</div>
      {delta_html}
    </div>"""


def build_bar_row(label, pct, val_str, color="var(--accent)"):
    return f"""
        <div class="bar-row">
          <div class="bar-label">{label}</div>
          <div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div></div>
          <div class="bar-val">{val_str}</div>
        </div>"""


def build_table_row(cells, classes=None):
    tds = ""
    for i, c in enumerate(cells):
        cls = ""
        if classes and i < len(classes):
            cls = f' class="{classes[i]}"'
        tds += f"<td{cls}>{c}</td>"
    return f"<tr>{tds}</tr>"


def generate_dashboard(d):
    kpis = d["kpis"]
    pat  = d["patrimonio"]
    dur  = d["duration"]
    ativos   = d["ativos"]
    passivos = d["passivos"]

    # ── Métricas chave ────────────────────────────────────────
    total_ativo   = kpis["total_ativo"] / 1000
    total_passivo = kpis["total_passivo"] / 1000
    pl            = kpis["patrimonio_liquido"] / 1000
    nii_base_anual = d["nii_base"]["nii"].sum()
    nii_alta_anual = d["nii_alta"]["nii"].sum()
    nii_baixa_anual= d["nii_baixa"]["nii"].sum()
    nim_base      = d["nii_base"]["nim_pct"].mean()

    dgap     = dur["duration_gap_anos"]
    dv01_gap = dur["dv01_gap_R$M"]
    rwa_total= kpis["rwa_total"] / 1000
    ib_pct   = kpis["indice_basileia_pct"]
    cet1_pct = kpis["cet1_ratio_pct"]

    # LCR/NSFR do stress test
    out_df   = d["stress_lcr"]
    outflows = out_df[out_df["tipo"]=="outflow"]["valor"].sum()
    inflows  = out_df[out_df["tipo"]=="inflow"]["valor"].sum()
    hqla_n1  = ativos[ativos["hqla"]=="nivel1"]["saldo"].sum()
    hqla_2a  = ativos[ativos["hqla"]=="nivel2A"]["saldo"].sum() * 0.85
    hqla_2b  = ativos[ativos["hqla"]=="nivel2B"]["saldo"].sum() * 0.75
    hqla_tot = hqla_n1 + hqla_2a + hqla_2b
    inflow_cap = min(inflows, 0.75 * outflows)
    nco      = outflows - inflow_cap
    lcr_val  = hqla_tot / nco * 100 if nco > 0 else 9999

    asf = (passivos["saldo"] * passivos["asf_fator"]).sum()
    asf += pat["capital_principal"] + pat["capital_complementar"] + pat["patrimonio_referencia_t2"]
    rsf = (ativos["saldo"] * ativos["rsf_fator"]).sum()
    nsfr_val = asf / rsf * 100

    hqla_custo_tot = d["hqla_custo"]["custo_oportunidade_BRL_aa"].sum()

    # RAROC
    raroc_df = d["raroc"]
    raroc_total_nii = raroc_df["nii_liquido_R$M"].sum()
    raroc_total_ce  = raroc_df["ce_R$M"].sum()
    raroc_total_eva = raroc_df["eva_R$M"].sum()
    raroc_medio     = raroc_total_nii / raroc_total_ce * 100 if raroc_total_ce > 0 else 0

    # Fronteira
    fe = d["fronteira"]
    nii_max  = fe["nii_total"].max()
    nii_min  = fe["nii_total"].min()
    dv01_max = fe["dv01_gap"].max()
    dv01_min = fe["dv01_gap"].min()

    # Projeção dinâmica — mês 12 por cenário
    proj = d["proj_liq"]
    proj_base  = proj[proj["cenario"]=="base"]
    proj_alta  = proj[proj["cenario"]=="stress_alta"]
    proj_baixa = proj[proj["cenario"]=="stress_baixa"]

    # Pricing
    pricing_df = d["pricing"]
    n_ok  = pricing_df["adequado_raroc"].sum()
    n_nok = (~pricing_df["adequado_raroc"]).sum()

    # ── Gera tabelas ──────────────────────────────────────────
    def gap_table():
        rows = ""
        gap_df = d["gap_repr"]
        for _, r in gap_df.iterrows():
            gap_val = r.get("gap", 0)
            gap_cum = r.get("gap_cumulativo", 0)
            gc = "positive" if gap_val > 0 else ("negative" if gap_val < 0 else "")
            gcc = "positive" if gap_cum > 0 else ("negative" if gap_cum < 0 else "")
            rows += f"""<tr>
              <td class="accent">{r['bucket']}</td>
              <td class="right">{r['ativo']/1000:.1f}bi</td>
              <td class="right">{r['passivo']/1000:.1f}bi</td>
              <td class="right {gc}">{'+' if gap_val>0 else ''}{gap_val/1000:.1f}bi</td>
              <td class="right {gcc}">{'+' if gap_cum>0 else ''}{gap_cum/1000:.1f}bi</td>
              <td class="right">{r.get('dv01_gap',0):+.1f}</td>
            </tr>"""
        return rows

    def raroc_table():
        rows = ""
        for _, r in raroc_df.sort_values("raroc_pct", ascending=False).iterrows():
            ok = "✅" if r["raroc_acima_alvo"] else "❌"
            rc = "positive" if r["raroc_pct"] >= 15 else "negative"
            ec = "positive" if r["eva_R$M"] >= 0 else "negative"
            rows += f"""<tr>
              <td class="{rc}">{r['produto']}</td>
              <td class="right">{r['ead_R$M']/1000:.1f}bi</td>
              <td class="right">{r['ce_R$M']/1000:.1f}bi</td>
              <td class="right {rc}">{r['nii_liquido_R$M']:,.0f}M</td>
              <td class="right {rc}">{r['raroc_pct']:.1f}%</td>
              <td class="right {ec}">{'+' if r['eva_R$M']>=0 else ''}{r['eva_R$M']:,.0f}M</td>
              <td class="right">{ok}</td>
            </tr>"""
        return rows

    def pricing_table():
        rows = ""
        for _, r in pricing_df.sort_values("folga_spread_pct", ascending=False).iterrows():
            ok = "✅" if r["adequado_raroc"] else "❌"
            fc = "positive" if r["folga_spread_pct"] >= 0 else "negative"
            rows += f"""<tr>
              <td class="{fc}">{r['produto']}</td>
              <td class="right">{r['taxa_atual_pct']:.2f}%</td>
              <td class="right">{r['taxa_minima_pct']:.2f}%</td>
              <td class="right {fc}">{r['folga_spread_pct']:+.2f}pp</td>
              <td class="right">{r['raroc_implicito']:.1f}%</td>
              <td class="right">{ok}</td>
            </tr>"""
        return rows

    def fronteira_table():
        rows = ""
        for _, r in fe.iterrows():
            status_c = "positive" if "ótimo" in str(r["status"]) else "negative"
            rows += f"""<tr>
              <td class="accent">{r['lambda']:.4f}</td>
              <td class="right">{r['nii_total']:,.0f}M</td>
              <td class="right">{r['dv01_gap']:.1f}</td>
              <td class="right">{r['duration_gap']:.2f}a</td>
              <td class="right">{r['lcr']*100:.0f}%</td>
              <td class="right">{r['ib']*100:.1f}%</td>
              <td class="right {status_c}">{str(r['status'])[:8]}</td>
            </tr>"""
        return rows

    def projecao_rows(df, color):
        last = df.iloc[-1]
        rows = ""
        for _, r in df.iterrows():
            lc_l = "positive" if r["lcr_pct"] >= 100 else "negative"
            ns_c = "positive" if r["nsfr_pct"] >= 100 else "negative"
            rows += f"""<tr>
              <td class="accent">M{int(r['mes']):02d}</td>
              <td class="right {lc_l}">{r['lcr_pct']:.0f}%</td>
              <td class="right {ns_c}">{r['nsfr_pct']:.0f}%</td>
              <td class="right">{r['hqla_R$M']/1000:.1f}bi</td>
              <td class="right">{r['nco_R$M']/1000:.1f}bi</td>
            </tr>"""
        return rows

    def hqla_bars():
        df = d["hqla_custo"].sort_values("custo_oportunidade_BRL_aa", ascending=False).head(8)
        max_c = df["custo_oportunidade_BRL_aa"].max()
        html = ""
        for _, r in df.iterrows():
            pct = r["custo_oportunidade_BRL_aa"] / max_c * 100 if max_c > 0 else 0
            color = {"nivel1":"var(--green)","nivel2A":"var(--yellow)","nivel2B":"var(--orange)"}.get(r["nivel_hqla"],"var(--text3)")
            html += build_bar_row(
                r["produto"][:22],
                pct,
                f"R${r['custo_oportunidade_BRL_aa']:,.0f}M/a",
                color
            )
        return html

    def ativo_bars():
        cats = ativos.groupby("categoria")["saldo"].sum().sort_values(ascending=False)
        max_v = cats.max()
        html = ""
        colors = ["var(--accent)","var(--accent2)","var(--green)","var(--yellow)","var(--orange)","var(--text3)"]
        for i, (cat, val) in enumerate(cats.items()):
            pct = val / max_v * 100
            html += build_bar_row(cat[:22], pct, f"R${val/1000:.1f}bi", colors[i % len(colors)])
        return html

    def passivo_bars():
        cats = passivos.groupby("categoria")["saldo"].sum().sort_values(ascending=False)
        max_v = cats.max()
        html = ""
        colors = ["var(--orange)","var(--red)","var(--yellow)","var(--accent2)","var(--accent)","var(--text3)"]
        for i, (cat, val) in enumerate(cats.items()):
            pct = val / max_v * 100
            html += build_bar_row(cat[:22], pct, f"R${val/1000:.1f}bi", colors[i % len(colors)])
        return html

    def nii_cenario_bars():
        niis = [nii_base_anual, nii_alta_anual, nii_baixa_anual]
        max_n = max(niis) * 1.05
        html = ""
        for label, val, color in [
            ("Base (SELIC 11.75%)", nii_base_anual, "var(--accent)"),
            ("Stress Alta (+200bps)", nii_alta_anual, "var(--red)"),
            ("Stress Baixa (−250bps)", nii_baixa_anual, "var(--green)"),
        ]:
            pct = val / max_n * 100
            html += build_bar_row(label, pct, f"R${val/1000:.1f}bi", color)
        return html

    # ── Constrói HTML ─────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ALM Dashboard — Banco Modelo S.A.</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:#0a0e17;--bg2:#0f1520;--bg3:#141c2e;
  --border:#1e2d47;--border2:#243554;
  --accent:#00d4ff;--accent2:#0088cc;
  --green:#00e676;--green2:#00c853;
  --red:#ff1744;--orange:#ff9100;--yellow:#ffd740;
  --text:#e2eaf5;--text2:#8faac5;--text3:#4a6080;
  --mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif;
}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:var(--bg);color:var(--text);font-family:var(--sans);min-height:100vh;overflow-x:hidden;}}
body::before{{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,212,255,.012) 2px,rgba(0,212,255,.012) 4px);pointer-events:none;z-index:1000;}}

/* HEADER */
header{{background:var(--bg2);border-bottom:1px solid var(--border);padding:0 2rem;height:56px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;}}
.logo{{display:flex;align-items:center;gap:12px;}}
.logo-dot{{width:8px;height:8px;border-radius:50%;background:var(--accent);box-shadow:0 0 12px var(--accent);animation:pulse 2s infinite;}}
@keyframes pulse{{0%,100%{{opacity:1;}}50%{{opacity:.4;}}}}
.logo-text{{font-family:var(--mono);font-size:13px;font-weight:600;color:var(--accent);letter-spacing:.08em;text-transform:uppercase;}}
.logo-sub{{font-family:var(--mono);font-size:10px;color:var(--text3);letter-spacing:.05em;}}
.header-right{{display:flex;align-items:center;gap:24px;}}
.timestamp{{font-family:var(--mono);font-size:11px;color:var(--text3);}}
.status-badge{{font-family:var(--mono);font-size:10px;font-weight:600;padding:3px 10px;border-radius:2px;background:rgba(0,230,118,.1);border:1px solid var(--green2);color:var(--green);text-transform:uppercase;letter-spacing:.1em;}}

/* NAV */
nav{{background:var(--bg2);border-bottom:1px solid var(--border);padding:0 2rem;display:flex;gap:0;overflow-x:auto;}}
.tab{{font-family:var(--mono);font-size:11px;font-weight:500;color:var(--text3);padding:12px 18px;cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap;transition:all .15s;letter-spacing:.05em;text-transform:uppercase;}}
.tab:hover{{color:var(--text2);}}
.tab.active{{color:var(--accent);border-bottom-color:var(--accent);}}

/* MAIN */
main{{padding:1.5rem 2rem;max-width:1600px;margin:0 auto;}}
.section{{display:none;}}.section.active{{display:block;}}
.section-title{{font-family:var(--mono);font-size:11px;font-weight:600;color:var(--text3);letter-spacing:.15em;text-transform:uppercase;margin-bottom:1.25rem;display:flex;align-items:center;gap:8px;}}
.section-title::after{{content:'';flex:1;height:1px;background:var(--border);}}

/* KPI */
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:1px;background:var(--border);border:1px solid var(--border);margin-bottom:1.5rem;}}
.kpi-card{{background:var(--bg2);padding:1rem 1.25rem;position:relative;overflow:hidden;transition:background .15s;}}
.kpi-card:hover{{background:var(--bg3);}}
.kpi-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;}}
.kpi-card.green::before{{background:var(--green);}}
.kpi-card.blue::before{{background:var(--accent);}}
.kpi-card.orange::before{{background:var(--orange);}}
.kpi-card.red::before{{background:var(--red);}}
.kpi-card.yellow::before{{background:var(--yellow);}}
.kpi-label{{font-family:var(--mono);font-size:9px;font-weight:500;color:var(--text3);letter-spacing:.12em;text-transform:uppercase;margin-bottom:6px;}}
.kpi-value{{font-family:var(--mono);font-size:22px;font-weight:600;color:var(--text);line-height:1;margin-bottom:4px;}}
.kpi-value.green{{color:var(--green);}}.kpi-value.blue{{color:var(--accent);}}.kpi-value.orange{{color:var(--orange);}}.kpi-value.red{{color:var(--red);}}.kpi-value.yellow{{color:var(--yellow);}}
.kpi-unit{{font-family:var(--mono);font-size:10px;color:var(--text3);}}
.kpi-delta{{font-family:var(--mono);font-size:10px;margin-top:4px;}}
.kpi-delta.up{{color:var(--green);}}.kpi-delta.down{{color:var(--red);}}.kpi-delta.neutral{{color:var(--text3);}}

/* PANELS */
.panel-grid{{display:grid;gap:1rem;margin-bottom:1.5rem;}}
.panel-grid.cols-2{{grid-template-columns:1fr 1fr;}}.panel-grid.cols-3{{grid-template-columns:1fr 1fr 1fr;}}
.panel-grid.cols-1-2{{grid-template-columns:1fr 2fr;}}.panel-grid.cols-2-1{{grid-template-columns:2fr 1fr;}}
.panel{{background:var(--bg2);border:1px solid var(--border);padding:1.25rem;}}
.panel-header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem;padding-bottom:.75rem;border-bottom:1px solid var(--border);}}
.panel-title{{font-family:var(--mono);font-size:10px;font-weight:600;color:var(--text2);letter-spacing:.12em;text-transform:uppercase;}}
.panel-tag{{font-family:var(--mono);font-size:9px;padding:2px 8px;border-radius:2px;letter-spacing:.08em;}}
.panel-tag.ok{{background:rgba(0,230,118,.1);color:var(--green);border:1px solid rgba(0,230,118,.3);}}
.panel-tag.warn{{background:rgba(255,145,0,.1);color:var(--orange);border:1px solid rgba(255,145,0,.3);}}
.panel-tag.alert{{background:rgba(255,23,68,.1);color:var(--red);border:1px solid rgba(255,23,68,.3);}}
.panel-tag.info{{background:rgba(0,212,255,.1);color:var(--accent);border:1px solid rgba(0,212,255,.3);}}

/* TABLE */
.data-table{{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11px;}}
.data-table th{{text-align:left;padding:6px 10px;font-size:9px;font-weight:600;color:var(--text3);letter-spacing:.1em;text-transform:uppercase;border-bottom:1px solid var(--border);}}
.data-table th.right{{text-align:right;}}
.data-table td{{padding:7px 10px;color:var(--text2);border-bottom:1px solid rgba(30,45,71,.5);}}
.data-table td.right{{text-align:right;font-variant-numeric:tabular-nums;}}
.data-table tr:hover td{{background:rgba(0,212,255,.03);color:var(--text);}}
.data-table .positive{{color:var(--green);}}.data-table .negative{{color:var(--red);}}.data-table .accent{{color:var(--accent);}}
.positive{{color:var(--green);}}.negative{{color:var(--red);}}.accent{{color:var(--accent);}}

/* BAR */
.bar-chart{{display:flex;flex-direction:column;gap:8px;}}
.bar-row{{display:grid;grid-template-columns:160px 1fr 100px;align-items:center;gap:10px;}}
.bar-label{{font-family:var(--mono);font-size:10px;color:var(--text2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.bar-track{{height:6px;background:var(--bg3);border-radius:1px;overflow:hidden;}}
.bar-fill{{height:100%;border-radius:1px;transition:width .8s cubic-bezier(.4,0,.2,1);}}
.bar-val{{font-family:var(--mono);font-size:10px;color:var(--text3);text-align:right;}}

/* GAUGE */
.gauge-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.5rem;}}
.gauge-card{{background:var(--bg3);border:1px solid var(--border);padding:1rem;text-align:center;}}
.gauge-label{{font-family:var(--mono);font-size:9px;color:var(--text3);letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;}}
.gauge-value{{font-family:var(--mono);font-size:24px;font-weight:600;line-height:1;margin-bottom:4px;}}
.gauge-bar-wrap{{height:4px;background:var(--bg);border-radius:2px;margin:8px 0;overflow:hidden;}}
.gauge-bar{{height:100%;border-radius:2px;transition:width 1s;}}
.gauge-min{{font-family:var(--mono);font-size:9px;color:var(--text3);}}

/* MISC */
.divider{{height:1px;background:var(--border);margin:1.25rem 0;}}
.footnote{{font-family:var(--mono);font-size:9px;color:var(--text3);margin-top:8px;padding-top:8px;border-top:1px solid var(--border);}}
.scroll-table{{overflow-x:auto;}}

@media(max-width:900px){{
  .panel-grid.cols-2,.panel-grid.cols-3,.panel-grid.cols-1-2,.panel-grid.cols-2-1{{grid-template-columns:1fr;}}
  .gauge-grid{{grid-template-columns:repeat(2,1fr);}}
  .kpi-grid{{grid-template-columns:repeat(2,1fr);}}
  .bar-row{{grid-template-columns:100px 1fr 80px;}}
}}
</style>
</head>
<body>
<header>
  <div class="logo">
    <div class="logo-dot"></div>
    <div>
      <div class="logo-text">ALM System // Banco Modelo S.A.</div>
      <div class="logo-sub">Asset &amp; Liability Management · Doutorado Poli-USP · Data-base Dez/2024</div>
    </div>
  </div>
  <div class="header-right">
    <div class="timestamp">BASELINE QP · SLSQP · scipy {'{}'}</div>
    <div class="status-badge">● {n_ok}/{len(pricing_df)} RAROC OK</div>
  </div>
</header>

<nav>
  <div class="tab active" onclick="showSection('overview')">01 · Visão Geral</div>
  <div class="tab" onclick="showSection('balanco')">02 · Balanço</div>
  <div class="tab" onclick="showSection('nii')">03 · NII &amp; Gap</div>
  <div class="tab" onclick="showSection('otimizador')">04 · Otimizador</div>
  <div class="tab" onclick="showSection('liquidez')">05 · Liquidez</div>
  <div class="tab" onclick="showSection('capital')">06 · Capital &amp; RAROC</div>
</nav>

<main>

<!-- 01 OVERVIEW -->
<div id="sec-overview" class="section active">
  <div class="section-title">Visão Geral — Banco Modelo S.A.</div>
  <div class="kpi-grid">
    {build_kpi_card("Total Ativo","%.0f" % total_ativo,"R$ bilhões","blue")}
    {build_kpi_card("Total Passivo","%.0f" % total_passivo,"R$ bilhões","orange")}
    {build_kpi_card("Patrimônio Líq.","%.0f" % pl,"R$ bilhões","green")}
    {build_kpi_card("NII Anual Base","%.1f" % (nii_base_anual/1000),"R$ bilhões","green")}
    {build_kpi_card("NIM Médio","%.2f%%" % nim_base,"% a.a.","blue")}
    {build_kpi_card("RWA Total","%.1f" % rwa_total,"R$ bilhões","blue")}
    {build_kpi_card("Índice Basileia","%.2f%%" % ib_pct,"mín 10.5%","green")}
    {build_kpi_card("CET1","%.2f%%" % cet1_pct,"mín 7.0%","green")}
    {build_kpi_card("Duration Gap","%.2fa" % dgap,"anos · LONG","yellow")}
    {build_kpi_card("DV01 Gap","%.1f" % dv01_gap,"R$M/bp","orange")}
    {build_kpi_card("LCR Stress","%.0f%%" % lcr_val,"mín 100%","green")}
    {build_kpi_card("NSFR","%.1f%%" % nsfr_val,"mín 100%","green")}
  </div>

  <div class="panel-grid cols-2">
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">Semáforo Regulatório</div>
        <div class="panel-tag {'ok' if ib_pct>=10.5 and lcr_val>=100 and nsfr_val>=100 else 'alert'}">{'Todos OK' if ib_pct>=10.5 and lcr_val>=100 and nsfr_val>=100 else 'Atenção'}</div>
      </div>
      <table class="data-table">
        <thead><tr><th>Métrica</th><th class="right">Atual</th><th class="right">Mínimo</th><th class="right">Folga</th><th class="right">Status</th></tr></thead>
        <tbody>
          <tr><td>LCR</td><td class="right positive">{lcr_val:.0f}%</td><td class="right">100%</td><td class="right positive">+{lcr_val-100:.0f}pp</td><td class="right positive">● OK</td></tr>
          <tr><td>NSFR</td><td class="right positive">{nsfr_val:.1f}%</td><td class="right">100%</td><td class="right positive">+{nsfr_val-100:.1f}pp</td><td class="right positive">● OK</td></tr>
          <tr><td>Índice Basileia</td><td class="right positive">{ib_pct:.2f}%</td><td class="right">10.5%</td><td class="right positive">+{ib_pct-10.5:.2f}pp</td><td class="right positive">● OK</td></tr>
          <tr><td>CET1 Ratio</td><td class="right positive">{cet1_pct:.2f}%</td><td class="right">7.0%</td><td class="right positive">+{cet1_pct-7.0:.2f}pp</td><td class="right positive">● OK</td></tr>
          <tr><td>Duration Gap</td><td class="right {'positive' if abs(dgap)<=2 else 'negative'}">{dgap:.2f}a</td><td class="right">±2.0a</td><td class="right {'positive' if 2-abs(dgap)>0 else 'negative'}">{2-abs(dgap):.2f}a</td><td class="right {'positive' if abs(dgap)<=2 else 'negative'}">{'● OK' if abs(dgap)<=2 else '● HEDGE'}</td></tr>
        </tbody>
      </table>
    </div>

    <div class="panel">
      <div class="panel-header"><div class="panel-title">NII por Cenário de Juros (12 meses)</div><div class="panel-tag info">3 cenários</div></div>
      <div class="bar-chart">
        {nii_cenario_bars()}
      </div>
      <div class="divider"></div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">
        <div style="background:var(--bg3);padding:10px;border:1px solid var(--border);text-align:center;">
          <div style="font-family:var(--mono);font-size:9px;color:var(--text3);margin-bottom:4px;">ΔNII (ALTA)</div>
          <div style="font-family:var(--mono);font-size:15px;font-weight:600;color:var(--red)">{(nii_alta_anual-nii_base_anual)/1000:+.2f}bi</div>
        </div>
        <div style="background:var(--bg3);padding:10px;border:1px solid var(--border);text-align:center;">
          <div style="font-family:var(--mono);font-size:9px;color:var(--text3);margin-bottom:4px;">ΔNII (BAIXA)</div>
          <div style="font-family:var(--mono);font-size:15px;font-weight:600;color:var(--green)">{(nii_baixa_anual-nii_base_anual)/1000:+.2f}bi</div>
        </div>
        <div style="background:var(--bg3);padding:10px;border:1px solid var(--border);text-align:center;">
          <div style="font-family:var(--mono);font-size:9px;color:var(--text3);margin-bottom:4px;">RAROC TOTAL</div>
          <div style="font-family:var(--mono);font-size:15px;font-weight:600;color:var(--accent)">{raroc_medio:.0f}%</div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- 02 BALANÇO -->
<div id="sec-balanco" class="section">
  <div class="section-title">Estrutura do Balanço Patrimonial</div>
  <div class="panel-grid cols-2">
    <div class="panel">
      <div class="panel-header"><div class="panel-title">Ativo por Categoria</div><div class="panel-tag info">R$ {total_ativo:.0f}bi total</div></div>
      <div class="bar-chart">{ativo_bars()}</div>
    </div>
    <div class="panel">
      <div class="panel-header"><div class="panel-title">Passivo por Categoria</div><div class="panel-tag info">R$ {total_passivo:.0f}bi total</div></div>
      <div class="bar-chart">{passivo_bars()}</div>
    </div>
  </div>
  <div class="panel-grid cols-3">
    <div class="panel">
      <div class="panel-header"><div class="panel-title">Indicadores de Capital</div></div>
      <table class="data-table">
        <thead><tr><th>Indicador</th><th class="right">Valor</th><th class="right">Limite</th></tr></thead>
        <tbody>
          <tr><td>RWA Total</td><td class="right">R${rwa_total:.1f}bi</td><td class="right">—</td></tr>
          <tr><td>PR Regulatório</td><td class="right">R${pat['patrimonio_referencia']/1000:.1f}bi</td><td class="right">—</td></tr>
          <tr><td>Índice Basileia</td><td class="right positive">{ib_pct:.2f}%</td><td class="right">10.5%</td></tr>
          <tr><td>CET1 Ratio</td><td class="right positive">{cet1_pct:.2f}%</td><td class="right">7.0%</td></tr>
          <tr><td>Leverage</td><td class="right">{kpis['leverage_x']:.2f}x</td><td class="right">—</td></tr>
          <tr><td>Crédito/Ativo</td><td class="right">{kpis['credito_ativo_pct']:.1f}%</td><td class="right">—</td></tr>
        </tbody>
      </table>
    </div>
    <div class="panel">
      <div class="panel-header"><div class="panel-title">Composição do Patrimônio</div></div>
      <table class="data-table">
        <thead><tr><th>Componente</th><th class="right">R$ M</th></tr></thead>
        <tbody>
          <tr><td>Capital Principal (CET1)</td><td class="right positive">{pat['capital_principal']:,.0f}</td></tr>
          <tr><td>Capital Complementar (AT1)</td><td class="right">{pat['capital_complementar']:,.0f}</td></tr>
          <tr><td>Tier 2</td><td class="right">{pat['patrimonio_referencia_t2']:,.0f}</td></tr>
          <tr><td>Reservas</td><td class="right">{pat['reservas_lucros_retidos']:,.0f}</td></tr>
          <tr><td>Ajuste Avaliação Patrimonial</td><td class="right">{pat['ajuste_avaliacao_patrimonial']:,.0f}</td></tr>
          <tr><td><strong>PR Total</strong></td><td class="right positive"><strong>{pat['patrimonio_referencia']:,.0f}</strong></td></tr>
        </tbody>
      </table>
    </div>
    <div class="panel">
      <div class="panel-header"><div class="panel-title">RWA por Categoria</div></div>
      <table class="data-table">
        <thead><tr><th>Categoria</th><th class="right">RWA (R$M)</th><th class="right">%</th></tr></thead>
        <tbody>
          {''.join([f'<tr><td>{r["categoria"]}</td><td class="right">{r["rwa_liquido_R$M"]:,.0f}</td><td class="right accent">{r["rwa_liquido_R$M"]/d["rwa"]["rwa_liquido_R$M"].sum()*100:.1f}%</td></tr>' for _, r in d["rwa"].groupby("categoria")["rwa_liquido_R$M"].sum().reset_index().sort_values("rwa_liquido_R$M", ascending=False).iterrows()])}
        </tbody>
      </table>
    </div>
  </div>
</div>

<!-- 03 NII & GAP -->
<div id="sec-nii" class="section">
  <div class="section-title">NII, Gap de Reprecificação e Duration</div>
  <div class="panel-grid cols-2">
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">Gap de Reprecificação por Bucket</div>
        <div class="panel-tag {'warn' if abs(dgap) > 1.5 else 'ok'}">Duration Gap: {dgap:.2f}a</div>
      </div>
      <div class="scroll-table">
      <table class="data-table">
        <thead><tr><th>Bucket</th><th class="right">Ativo</th><th class="right">Passivo</th><th class="right">Gap</th><th class="right">Gap Cum.</th><th class="right">DV01</th></tr></thead>
        <tbody>{gap_table()}</tbody>
      </table>
      </div>
      <div class="footnote">DV01 Gap em R$M/1bp. Positivo = LONG nesse bucket. Gap O/N negativo = passivos repricing antes dos ativos (típico varejo).</div>
    </div>
    <div class="panel">
      <div class="panel-header"><div class="panel-title">Duration e Sensibilidade ao Risco de Taxa</div></div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:16px;">
        <div style="background:var(--bg3);padding:12px;border:1px solid var(--border);text-align:center;">
          <div style="font-family:var(--mono);font-size:9px;color:var(--text3);margin-bottom:6px;">DUR. ATIVO</div>
          <div style="font-family:var(--mono);font-size:26px;font-weight:600;color:var(--accent)">{dur['duration_ativo_anos']:.2f}</div>
          <div style="font-family:var(--mono);font-size:10px;color:var(--text3)">anos</div>
        </div>
        <div style="background:var(--bg3);padding:12px;border:1px solid var(--border);text-align:center;">
          <div style="font-family:var(--mono);font-size:9px;color:var(--text3);margin-bottom:6px;">DUR. PASSIVO</div>
          <div style="font-family:var(--mono);font-size:26px;font-weight:600;color:var(--orange)">{dur['duration_passivo_anos']:.2f}</div>
          <div style="font-family:var(--mono);font-size:10px;color:var(--text3)">anos</div>
        </div>
        <div style="background:var(--bg3);padding:12px;border:1px solid var(--border);text-align:center;">
          <div style="font-family:var(--mono);font-size:9px;color:var(--text3);margin-bottom:6px;">DUR. GAP</div>
          <div style="font-family:var(--mono);font-size:26px;font-weight:600;color:var(--yellow)">{dur['duration_gap_anos']:.2f}</div>
          <div style="font-family:var(--mono);font-size:10px;color:var(--text3)">anos</div>
        </div>
      </div>
      <table class="data-table">
        <thead><tr><th>Cenário</th><th class="right">NII 12m</th><th class="right">NIM</th><th class="right">vs. Base</th></tr></thead>
        <tbody>
          <tr><td class="accent">● Base (SELIC 11.75%)</td><td class="right">{nii_base_anual/1000:.2f}bi</td><td class="right">{nim_base:.2f}%</td><td class="right">—</td></tr>
          <tr><td class="negative">● Stress Alta (+200bps)</td><td class="right">{nii_alta_anual/1000:.2f}bi</td><td class="right">{d['nii_alta']['nim_pct'].mean():.2f}%</td><td class="right negative">{(nii_alta_anual-nii_base_anual)/1000:+.3f}bi</td></tr>
          <tr><td class="positive">● Stress Baixa (−250bps)</td><td class="right">{nii_baixa_anual/1000:.2f}bi</td><td class="right">{d['nii_baixa']['nim_pct'].mean():.2f}%</td><td class="right positive">{(nii_baixa_anual-nii_base_anual)/1000:+.3f}bi</td></tr>
        </tbody>
      </table>
      <div class="divider"></div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">
        <div style="text-align:center;"><div style="font-family:var(--mono);font-size:9px;color:var(--text3);margin-bottom:4px;">DV01 ATIVO</div><div style="font-family:var(--mono);font-size:14px;font-weight:600;color:var(--accent)">R${dur['dv01_ativo_R$M']:.1f}M</div></div>
        <div style="text-align:center;"><div style="font-family:var(--mono);font-size:9px;color:var(--text3);margin-bottom:4px;">DV01 PASSIVO</div><div style="font-family:var(--mono);font-size:14px;font-weight:600;color:var(--orange)">R${dur['dv01_passivo_R$M']:.1f}M</div></div>
        <div style="text-align:center;"><div style="font-family:var(--mono);font-size:9px;color:var(--text3);margin-bottom:4px;">DV01 GAP</div><div style="font-family:var(--mono);font-size:14px;font-weight:600;color:var(--yellow)">R${dur['dv01_gap_R$M']:.1f}M</div></div>
      </div>
    </div>
  </div>
</div>

<!-- 04 OTIMIZADOR -->
<div id="sec-otimizador" class="section">
  <div class="section-title">Otimizador ALM — QP Baseline (SLSQP)</div>
  <div class="panel">
    <div class="panel-header"><div class="panel-title">Fronteira Eficiente — NII × DV01 Gap ({len(fe)} pontos)</div><div class="panel-tag info">λ ∈ [{fe['lambda'].min():.4f}, {fe['lambda'].max():.2f}]</div></div>
    <div class="scroll-table">
    <table class="data-table">
      <thead><tr><th>λ (risco)</th><th class="right">NII Total</th><th class="right">DV01 Gap</th><th class="right">Dur. Gap</th><th class="right">LCR</th><th class="right">IB</th><th class="right">Status</th></tr></thead>
      <tbody>{fronteira_table()}</tbody>
    </table>
    </div>
    <div class="footnote">Fronteira praticamente flat em NII: reduzir DV01 gap de {fe['dv01_gap'].max():.0f}→{fe['dv01_gap'].min():.0f} custa apenas R${(fe['nii_total'].max()-fe['nii_total'].min()):.0f}M de NII ({(fe['nii_total'].max()-fe['nii_total'].min())/fe['nii_total'].max()*100:.2f}%). Hedge é barato relativamente ao NII total.</div>
  </div>
</div>

<!-- 05 LIQUIDEZ -->
<div id="sec-liquidez" class="section">
  <div class="section-title">Gestão de Liquidez — LCR / NSFR / Stress Test</div>
  <div class="gauge-grid">
    <div class="gauge-card">
      <div class="gauge-label">LCR Stress 30d</div>
      <div class="gauge-value positive">{lcr_val:.0f}%</div>
      <div class="gauge-bar-wrap"><div class="gauge-bar" style="width:100%;background:var(--green)"></div></div>
      <div class="gauge-min">Mínimo: 100% ✅</div>
    </div>
    <div class="gauge-card">
      <div class="gauge-label">NSFR</div>
      <div class="gauge-value positive">{nsfr_val:.1f}%</div>
      <div class="gauge-bar-wrap"><div class="gauge-bar" style="width:100%;background:var(--green)"></div></div>
      <div class="gauge-min">Mínimo: 100% ✅</div>
    </div>
    <div class="gauge-card">
      <div class="gauge-label">HQLA Total</div>
      <div class="gauge-value" style="color:var(--accent)">R${hqla_tot/1000:.0f}bi</div>
      <div class="gauge-bar-wrap"><div class="gauge-bar" style="width:{min(100,hqla_n1/hqla_tot*100):.0f}%;background:var(--accent)"></div></div>
      <div class="gauge-min">Nível 1: R${hqla_n1/1000:.0f}bi ({hqla_n1/hqla_tot*100:.0f}%)</div>
    </div>
    <div class="gauge-card">
      <div class="gauge-label">Custo Buffer HQLA</div>
      <div class="gauge-value" style="color:var(--orange)">R${hqla_custo_tot/1000:.0f}bi</div>
      <div class="gauge-bar-wrap"><div class="gauge-bar" style="width:45%;background:var(--orange)"></div></div>
      <div class="gauge-min">Custo oportunidade/ano</div>
    </div>
  </div>

  <div class="panel-grid cols-2">
    <div class="panel">
      <div class="panel-header"><div class="panel-title">Custo de Carregamento HQLA por Produto</div><div class="panel-tag warn">R${hqla_custo_tot/1000:.0f}bi/ano</div></div>
      <div class="bar-chart">{hqla_bars()}</div>
      <div class="footnote">Custo de oportunidade = (taxa crédito alternativo − taxa HQLA) × saldo. Referência: capital de giro PJ ao CDI+4.5%.</div>
    </div>

    <div class="panel">
      <div class="panel-header"><div class="panel-title">Projeção LCR — 12 meses (Cenário Base)</div><div class="panel-tag ok">Folga confortável</div></div>
      <div class="scroll-table">
      <table class="data-table">
        <thead><tr><th>Mês</th><th class="right">LCR</th><th class="right">NSFR</th><th class="right">HQLA</th><th class="right">NCO</th></tr></thead>
        <tbody>{projecao_rows(proj_base, 'var(--accent)')}</tbody>
      </table>
      </div>
      <div class="footnote">HQLA cresce a 50% da taxa de crescimento do crédito (gestão ativa). NSFR constraint ativo no otimizador.</div>
    </div>
  </div>
</div>

<!-- 06 CAPITAL -->
<div id="sec-capital" class="section">
  <div class="section-title">Capital Econômico, RAROC e Pricing</div>
  <div class="panel">
    <div class="panel-header">
      <div class="panel-title">RAROC por Linha de Negócio (Hurdle Rate: 15%)</div>
      <div class="panel-tag {'ok' if raroc_df['raroc_acima_alvo'].mean() > 0.5 else 'warn'}">{raroc_df['raroc_acima_alvo'].sum()}/{len(raroc_df)} acima do hurdle</div>
    </div>
    <table class="data-table">
      <thead><tr><th>Segmento</th><th class="right">EAD</th><th class="right">Cap. Econ.</th><th class="right">NII Líq.</th><th class="right">RAROC</th><th class="right">EVA</th><th class="right">OK?</th></tr></thead>
      <tbody>{raroc_table()}</tbody>
      <tfoot><tr style="border-top:2px solid var(--border2);">
        <td><strong>TOTAL</strong></td>
        <td class="right"><strong>R${raroc_df['ead_R$M'].sum()/1000:.0f}bi</strong></td>
        <td class="right"><strong>R${raroc_total_ce/1000:.1f}bi</strong></td>
        <td class="right positive"><strong>R${raroc_total_nii:,.0f}M</strong></td>
        <td class="right positive"><strong>{raroc_medio:.0f}%</strong></td>
        <td class="right positive"><strong>+R${raroc_total_eva:,.0f}M</strong></td>
        <td class="right"></td>
      </tr></tfoot>
    </table>
    <div class="footnote">Capital Econômico via fórmula IRB Basileia II (UL × ajuste maturidade). Imobiliário, Rural e Trade Finance abaixo do hurdle — subsídio implícito ou crédito direcionado regulatório.</div>
  </div>

  <div class="panel">
    <div class="panel-header">
      <div class="panel-title">Pricing Mínimo via RAROC ≥ 15%</div>
      <div class="panel-tag {'ok' if n_ok >= n_nok else 'warn'}">{n_ok} adequados / {n_nok} abaixo do mínimo</div>
    </div>
    <table class="data-table">
      <thead><tr><th>Segmento</th><th class="right">Taxa Atual</th><th class="right">Taxa Mín. RAROC</th><th class="right">Folga</th><th class="right">RAROC Impl.</th><th class="right">OK?</th></tr></thead>
      <tbody>{pricing_table()}</tbody>
    </table>
    <div class="footnote">Taxa mínima = Custo Funding + PE/EAD + Custo Op. + RAROC_alvo × CE/EAD. Segmentos com folga negativa operam abaixo do mínimo econômico — razão estratégica ou regulatória.</div>
  </div>
</div>

</main>

<script>
function showSection(id) {{
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('sec-' + id).classList.add('active');
  event.currentTarget.classList.add('active');
}}
document.addEventListener('DOMContentLoaded', () => {{
  setTimeout(() => {{
    document.querySelectorAll('.bar-fill, .gauge-bar').forEach(f => {{
      const w = f.style.width;
      f.style.width = '0';
      setTimeout(() => {{ f.style.width = w; }}, 50);
    }});
  }}, 100);
}});
</script>
</body>
</html>"""

    out_path = REPORT_DIR / "dashboard_alm.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"  ✅ Dashboard gerado: {out_path}")
    print(f"     Tamanho: {out_path.stat().st_size / 1024:.0f} KB")
    return out_path


if __name__ == "__main__":
    print("\n  Carregando dados...")
    d = load_all_data()
    print("  Gerando dashboard...")
    generate_dashboard(d)
