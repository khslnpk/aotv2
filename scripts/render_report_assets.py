"""Render every image asset embedded in the project report.

Generates PNG files under `artifacts/report_assets/` for:
    - Section 8 comparison tables (features, models, ensemble, results,
      scoring, application, Apple Health pipeline)
    - Section 16 leaderboard
    - Section 19 system architecture diagram
    - Section 21 sprint plan (one image per sprint)

Each asset is rendered as styled HTML matching the application's visual
language, then captured by a headless Chromium via Playwright. The resulting
PNGs are at retina resolution and look the same as the live app.

Usage:
    .venv/Scripts/python.exe scripts/render_report_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "report_assets"


# ---------------------------------------------------------------------------
# Shared styling — matches the app's aurora dark theme
# ---------------------------------------------------------------------------

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg-deep: #06070d;
  --bg-canvas: #0a0c1a;
  --bg-card: rgba(255, 255, 255, 0.04);
  --border: rgba(255, 255, 255, 0.10);
  --border-hi: rgba(255, 255, 255, 0.18);
  --text: #e8e9f3;
  --text-muted: #8c8fa6;
  --text-faint: #5a5d75;
  --c-recovery: #00d9c0;
  --c-sleep: #8b5cf6;
  --c-stress: #ec4899;
  --c-fatigue: #f59e0b;
  --c-energy: #3b82f6;
  --c-debt: #fb7185;
}

body {
  background: var(--bg-deep);
  color: var(--text);
  font-family: "Inter", system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.55;
  padding: 48px;
  color-scheme: dark;
  position: relative;
  overflow-x: hidden;
}
body::before {
  content: ""; position: fixed; inset: 0; z-index: -1;
  background:
    radial-gradient(ellipse at 18% 0%, rgba(139,92,246,0.18), transparent 50%),
    radial-gradient(ellipse at 82% 100%, rgba(0,217,192,0.14), transparent 50%),
    radial-gradient(ellipse at top, #0d1130 0%, #06070d 60%);
}

.title {
  font-family: "Fraunces", serif;
  font-weight: 500;
  font-size: 38px;
  letter-spacing: -0.02em;
  background: linear-gradient(180deg, #ffffff 0%, var(--c-recovery) 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 6px;
}
.subtitle {
  font-family: "JetBrains Mono", monospace;
  font-size: 11px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--text-faint);
  margin-bottom: 28px;
}

.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 28px 32px;
  backdrop-filter: blur(30px) saturate(150%);
  box-shadow: 0 1px 0 rgba(255,255,255,0.06) inset, 0 24px 64px -32px rgba(0,0,0,0.7);
}

table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-family: "JetBrains Mono", monospace;
  font-size: 13px;
}
th, td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
  text-align: left;
}
th {
  background: rgba(255,255,255,0.03);
  color: var(--c-recovery);
  font-weight: 500;
  font-size: 10.5px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  border-bottom: 1px solid var(--border-hi);
}
td.right { text-align: right; font-feature-settings: "tnum"; }
td.center { text-align: center; }
td.mono-faint { color: var(--text-faint); }
td.win {
  color: var(--text);
  font-weight: 600;
}
td.win::before { content: "★ "; color: var(--c-recovery); }

.tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-family: "JetBrains Mono", monospace;
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.tag.demo { color: var(--text-faint); background: rgba(255,255,255,0.04); border: 1px solid var(--border); }
.tag.final { color: var(--c-recovery); background: rgba(0,217,192,0.08); border: 1px solid rgba(0,217,192,0.3); }
.tag.up { color: #6ee7b7; background: rgba(110,231,183,0.06); border: 1px solid rgba(110,231,183,0.3); }
.tag.down { color: var(--c-stress); background: rgba(236,72,153,0.08); border: 1px solid rgba(236,72,153,0.3); }
.tag.same { color: var(--text-muted); background: rgba(140,143,166,0.06); border: 1px solid var(--border); }

.check { color: var(--c-recovery); font-weight: 700; }
.cross { color: var(--c-stress); font-weight: 700; }
.dash { color: var(--text-faint); }

.note {
  margin-top: 18px;
  font-family: "Inter", sans-serif;
  font-size: 12px;
  line-height: 1.6;
  color: var(--text-muted);
  font-style: italic;
}
"""


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def _render_html(html_body: str, width: int = 1400, css: str = _CSS) -> str:
    return f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<style>{css}</style>
</head><body style="width: {width - 96}px;">{html_body}</body></html>"""


def render_page(name: str, html_body: str, *, width: int = 1400, css: str = _CSS) -> Path:
    from playwright.sync_api import sync_playwright

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT / f"{name}.png"

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": width, "height": 800},
            device_scale_factor=2,  # retina sharpness
        )
        page = ctx.new_page()
        page.set_content(_render_html(html_body, width=width, css=css))
        # Wait for fonts to load
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(500)
        page.screenshot(path=str(out), full_page=True, omit_background=False)
        browser.close()
    print(f"  wrote {out.relative_to(ROOT)}")
    return out


# ---------------------------------------------------------------------------
# Section 8 — Demo vs Final comparison tables
# ---------------------------------------------------------------------------

def _table(rows: list[list[str]], headers: list[str], note: str | None = None) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(cell for cell in row) + "</tr>"
        for row in rows
    )
    note_html = f"<div class='note'>{note}</div>" if note else ""
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{note_html}"


def render_section_8():
    print("rendering section 8 comparison tables...")

    # 8.1 Feature engineering
    rows = [
        ["<td>Features per epoch</td>",         "<td><span class='tag demo'>318</span></td>", "<td><span class='tag final'>540</span></td>"],
        ["<td>Per-subject HR z-scoring</td>",   "<td><span class='cross'>×</span></td>", "<td><span class='check'>✓ primary accuracy lever</span></td>"],
        ["<td>Per-subject ENMO normalization</td>", "<td><span class='cross'>×</span></td>", "<td><span class='check'>✓</span></td>"],
        ["<td>Motion series</td>",              "<td>7 (mag, enmo, energy, jerk, x, y, z)</td>", "<td>8 (+ subject-normalized ENMO)</td>"],
        ["<td>Motion windows</td>",             "<td>30s · 120s · 600s</td>", "<td>30s · 120s · 300s · 600s</td>"],
        ["<td>HR series</td>",                  "<td>3 (bpm, dev-resting, dev-mean)</td>", "<td>4 (+ subject-z-scored bpm)</td>"],
        ["<td>HR windows</td>",                 "<td>30s · 120s · 600s · 1800s</td>", "<td>30s · 120s · 300s · 600s · 1800s · 3600s</td>"],
        ["<td>HRV-style proxies</td>",          "<td><span class='cross'>×</span></td>", "<td><span class='check'>✓</span> range / succdiff RMS / CV on 3 windows</td>"],
        ["<td>Per-window HR slope</td>",        "<td><span class='cross'>×</span></td>", "<td><span class='check'>✓</span> every HR window</td>"],
        ["<td>Per-subject HR constants</td>",   "<td>2 (resting p10, sleep mean)</td>", "<td>4 (+ sleep median, sleep IQR)</td>"],
    ]
    html = (
        "<div class='title'>Feature engineering</div>"
        "<div class='subtitle'>Demo vs Final · per-epoch feature stack</div>"
        f"<div class='card'>{_table(rows, ['Aspect', 'Demo', 'Final'])}</div>"
    )
    render_page("section_8_1_features", html, width=1400)

    # 8.2 + 8.3 Base learners + ensemble combined
    rows = [
        ["<td colspan='3'><span class='tag final' style='background: rgba(139,92,246,0.10); border-color: rgba(139,92,246,0.3); color: var(--c-sleep);'>Boosting</span></td>"],
        ["<td>ExtraTreesClassifier (750 trees)</td>",        "<td>✓</td>", "<td class='mono-faint'>removed</td>"],
        ["<td>HistGradientBoosting</td>",                    "<td>✓</td>", "<td class='mono-faint'>removed</td>"],
        ["<td>XGBoost CUDA</td>",                            "<td>850 estimators, depth 4</td>", "<td><span class='check'>1500 estimators, depth 6</span></td>"],
        ["<td>LightGBM</td>",                                "<td class='dash'>—</td>", "<td><span class='check'>✓</span> 2000 estimators, 95 leaves</td>"],
        ["<td>CatBoost (default)</td>",                      "<td class='dash'>—</td>", "<td><span class='check'>✓</span> standout single learner</td>"],
        ["<td>CatBoost (Optuna-tuned)</td>",                 "<td class='dash'>—</td>", "<td><span class='check'>✓</span> 25 trials, val macro F1 0.7057</td>"],
        ["<td colspan='3'><span class='tag final' style='background: rgba(0,217,192,0.10); border-color: rgba(0,217,192,0.3);'>Sequence</span></td>"],
        ["<td>Tabular MLP</td>",                             "<td>✓</td>", "<td class='mono-faint'>removed</td>"],
        ["<td>Temporal CNN</td>",                            "<td>✓</td>", "<td class='mono-faint'>removed</td>"],
        ["<td>BiLSTM with attention</td>",                   "<td>2 layers, hidden 96, simple linear attn, CrossEntropyLoss, ReduceLROnPlateau</td>", "<td><span class='check'>1 layer, hidden 64, additive attention pooling, focal loss γ=1.5, OneCycleLR cosine</span></td>"],
        ["<td colspan='3'><span class='tag final' style='background: rgba(245,158,11,0.10); border-color: rgba(245,158,11,0.3); color: var(--c-fatigue);'>Combination</span></td>"],
        ["<td>Probability ensembling</td>",                  "<td class='mono-faint'>none</td>", "<td><span class='check'>✓</span> geometric mean · arithmetic mean · logistic-regression stack</td>"],
        ["<td>Temporal smoothing (HMM/Viterbi)</td>",        "<td class='mono-faint'>none</td>", "<td><span class='check'>✓</span> per subject, Laplace-smoothed transitions</td>"],
        ["<td>Configurations on the leaderboard</td>",       "<td>1</td>", "<td><span class='check'>16</span> (every base × ensemble × ±HMM)</td>"],
    ]
    html = (
        "<div class='title'>Models &amp; combination</div>"
        "<div class='subtitle'>Demo vs Final · base learners, sequence model, ensembling</div>"
        f"<div class='card'>{_table(rows, ['Aspect', 'Demo', 'Final'])}</div>"
    )
    render_page("section_8_2_models", html, width=1500)

    # 8.4 Headline results
    rows = [
        ["<td>Accuracy</td>",          "<td class='right'>0.7646</td>", "<td class='right win'>0.8035</td>", "<td class='right'><span class='tag up'>+3.9 pp</span></td>"],
        ["<td>Balanced accuracy</td>", "<td class='right'>0.6659</td>", "<td class='right win'>0.7159</td>", "<td class='right'><span class='tag up'>+5.0 pp</span></td>"],
        ["<td>Macro F1</td>",          "<td class='right'>0.6678</td>", "<td class='right win'>0.6993</td>", "<td class='right'><span class='tag up'>+3.2 pp</span></td>"],
        ["<td>Weighted F1</td>",       "<td class='right'>0.7651</td>", "<td class='right win'>0.7978</td>", "<td class='right'><span class='tag up'>+3.3 pp</span></td>"],
        ["<td>Cohen's κ</td>",         "<td class='right'>0.5050</td>", "<td class='right win'>0.5695</td>", "<td class='right'><span class='tag up'>+0.065</span></td>"],
    ]
    html = (
        "<div class='title'>Headline results</div>"
        "<div class='subtitle'>Demo's best single model vs Final's best ensemble · 7 held-out test subjects</div>"
        f"<div class='card'>{_table(rows, ['Metric', 'Demo (XGBoost CUDA)', 'Final (best ensemble)', 'Δ'], note='Final picks the best per metric across 16 configurations; geometric ensemble + HMM wins accuracy / weighted F1 / kappa, CatBoost + HMM wins balanced accuracy / macro F1.')}</div>"
    )
    render_page("section_8_4_results", html, width=1400)

    # 8.5 Scoring layer
    rows = [
        ["<td>Number of scores</td>",                       "<td>5</td>", "<td><span class='check'>6</span> (added sleep debt)</td>"],
        ["<td>Score formulas</td>",                         "<td>hand-written, undocumented arithmetic</td>", "<td><span class='check'>weighted sums of named, bounded components</span></td>"],
        ["<td>Per-score component breakdown surfaced in UI</td>", "<td><span class='cross'>×</span></td>", "<td><span class='check'>✓</span> drawn as a flip overlay on every score card</td>"],
        ["<td>Literature references</td>",                  "<td class='dash'>none</td>", "<td><span class='check'>9</span> (Buysse, NSF, Carskadon, Reed &amp; Sacco, Trinder, Plews, Kim, Van Dongen, Walker)</td>"],
        ["<td>Sleep architecture summary fields</td>",      "<td>9</td>", "<td><span class='check'>13</span> (+ onset latency, REM latency, WASO, cycle count)</td>"],
        ["<td>Universal physiology abstraction</td>",       "<td>only Apple side</td>", "<td><span class='check'>✓</span> works for both PSG and Apple paths</td>"],
        ["<td>Apple HRV (SDNN) feeding the scoring</td>",   "<td class='cross'>× parsed but never consumed</td>", "<td><span class='check'>✓</span> recovery and stress prefer real SDNN when available</td>"],
        ["<td>Methodology document</td>",                   "<td class='dash'>none</td>", "<td><span class='check'>✓</span> per-component tables with targets and references</td>"],
    ]
    html = (
        "<div class='title'>Digital Twin scoring</div>"
        "<div class='subtitle'>Demo vs Final · interpretation surface</div>"
        f"<div class='card'>{_table(rows, ['Aspect', 'Demo', 'Final'])}</div>"
    )
    render_page("section_8_5_scoring", html, width=1500)

    # 8.6 Application
    rows = [
        ["<td>Framework</td>",                  "<td>Streamlit</td>", "<td><span class='check'>FastAPI + custom single-page app</span> · no npm, no build step</td>"],
        ["<td>Visual identity</td>",            "<td>Streamlit defaults + matplotlib charts</td>", "<td><span class='check'>custom dark aurora theme</span> · serif display numbers, mono data rows, glass bento grid, per-score colour identity</td>"],
        ["<td>Headline metric</td>",            "<td>row of 5 metric cards with progress bars</td>", "<td><span class='check'>270° segmented dial</span> rendered in SVG</td>"],
        ["<td>Hypnogram</td>",                  "<td>matplotlib step plot</td>", "<td><span class='check'>constellation</span> · REM pulsing stars, NREM dim field, Wake spikes, HR + motion overlays</td>"],
        ["<td>Score breakdown UI</td>",         "<td class='dash'>none</td>", "<td><span class='check'>3D flip overlay</span> on tap, with bar segments + literature notes per component</td>"],
        ["<td>Methodology surface</td>",        "<td class='dash'>nowhere</td>", "<td><span class='check'>slide-out drawer</span> (keyboard M) with score-coloured chips</td>"],
        ["<td>What-if simulator</td>",          "<td>sliders, recompute on submit</td>", "<td><span class='check'>live debounced sliders</span> · all 6 delta tiles refresh within 120 ms</td>"],
        ["<td>Boot time</td>",                  "<td>~10 s (cold start + model load)</td>", "<td><span class='check'>~1 s</span> (reuses cached probabilities, no model deserialization per request)</td>"],
    ]
    html = (
        "<div class='title'>Application</div>"
        "<div class='subtitle'>Demo vs Final · what the user sees</div>"
        f"<div class='card'>{_table(rows, ['Aspect', 'Demo', 'Final'])}</div>"
    )
    render_page("section_8_6_application", html, width=1600)

    # 8.7 Apple Health pipeline — first the record types
    record_rows = [
        ["<td>Sleep stages</td>",  "<td class='center check'>✓</td>", "<td class='center check'>✓</td>"],
        ["<td>Heart rate</td>",    "<td class='center check'>✓</td>", "<td class='center check'>✓</td>"],
        ["<td>HRV SDNN</td>",      "<td class='center check'>✓</td>", "<td class='center check'>✓</td>"],
        ["<td>Resting heart rate</td>",  "<td class='center check'>✓</td>", "<td class='center check'>✓</td>"],
        ["<td>Step count</td>",    "<td class='center check'>✓</td>", "<td class='center check'>✓</td>"],
        ["<td>Active energy</td>", "<td class='center check'>✓</td>", "<td class='center check'>✓</td>"],
        ["<td>Exercise time</td>", "<td class='center check'>✓</td>", "<td class='center check'>✓</td>"],
    ]
    capability_rows = [
        ["<td>Streaming parser with element clearing</td>", "<td class='center check'>✓</td>", "<td class='center check'>✓</td>"],
        ["<td>Zip and raw-XML upload</td>",                  "<td class='center check'>✓</td>", "<td class='center check'>✓</td>"],
        ["<td>Night grouping (configurable gap)</td>",       "<td class='center check'>✓</td>", "<td class='center check'>✓</td>"],
        ["<td>Brief-wake smoothing</td>",                    "<td class='center check'>✓</td>", "<td class='center check'>✓</td>"],
        ["<td>Blended prior-day steps (0.65 × prior + 0.35 × 7-day avg)</td>", "<td class='center check'>✓</td>", "<td class='center check'>✓</td>"],
        ["<td>Resting HR baseline from multi-day window</td>", "<td class='center check'>✓</td>", "<td class='center check'>✓</td>"],
        ["<td>7-day history summary (avg daily steps, energy, exercise, prior sleep stats)</td>", "<td class='center check'>✓</td>", "<td class='center check'>✓</td>"],
        ["<td>Detailed-vs-InBed-only session filter</td>",   "<td class='center check'>✓</td>", "<td class='center check'>✓</td>"],
        ["<td>In-app audit of every parsed value</td>",      "<td>Streamlit dataframes</td>", "<td><span class='check'>custom dark-themed audit card</span></td>"],
        ["<td><b>Real Apple HRV fed into recovery/stress scoring</b></td>", "<td class='cross'>× parsed but never consumed</td>", "<td class='check'>✓ scoring prefers SDNN when present</td>"],
        ["<td>Multi-GB upload safety (30-day cutoff)</td>",  "<td class='cross'>× no cap</td>", "<td class='center check'>✓</td>"],
        ["<td>Manual override sliders for HR delta / steps</td>", "<td class='center check'>✓</td>", "<td class='cross'>× consciously skipped</td>"],
    ]
    html = (
        "<div class='title'>Apple Health integration</div>"
        "<div class='subtitle'>Demo vs Final · personal-validation pipeline</div>"
        "<div class='card' style='margin-bottom: 22px;'>"
        "<h3 style='font-family: \"Fraunces\", serif; font-weight: 500; font-size: 17px; margin-bottom: 12px;'>Record types parsed</h3>"
        + _table(record_rows, ['Apple record type', 'Demo', 'Final'])
        + "</div>"
        "<div class='card'>"
        "<h3 style='font-family: \"Fraunces\", serif; font-weight: 500; font-size: 17px; margin-bottom: 12px;'>Capabilities</h3>"
        + _table(capability_rows, ['Capability', 'Demo', 'Final'], note="The demo was already substantial here. The single genuine improvement is that Final actually consumes the real Apple HRV signal in scoring — the demo parsed it but only displayed it in a table.")
        + "</div>"
    )
    render_page("section_8_7_apple", html, width=1600)


# ---------------------------------------------------------------------------
# Section 16 — Full leaderboard
# ---------------------------------------------------------------------------

def render_section_16():
    print("rendering section 16 leaderboard...")
    # The 16 rows in the order used in the report
    rows_data = [
        (1,  "base_catboost_hmm",          0.7746, 0.7159, 0.6993, 0.7793, 0.5501, "win-macro"),
        (2,  "avg_geo_hmm",                0.8035, 0.6745, 0.6932, 0.7978, 0.5695, "win-acc"),
        (3,  "avg_mean_hmm",               0.8022, 0.6733, 0.6921, 0.7965, 0.5668, ""),
        (4,  "avg_geo",                    0.7772, 0.6973, 0.6901, 0.7795, 0.5388, ""),
        (5,  "avg_mean",                   0.7754, 0.6975, 0.6892, 0.7780, 0.5368, ""),
        (6,  "base_catboost_tuned_hmm",    0.7816, 0.6870, 0.6814, 0.7820, 0.5515, ""),
        (7,  "base_lightgbm",              0.7796, 0.6605, 0.6762, 0.7761, 0.5176, ""),
        (8,  "base_catboost_tuned",        0.7538, 0.7053, 0.6750, 0.7615, 0.5201, ""),
        (9,  "base_lightgbm_hmm",          0.7897, 0.6332, 0.6687, 0.7794, 0.5153, ""),
        (10, "base_xgboost_cuda_hmm",      0.7766, 0.6456, 0.6667, 0.7715, 0.5079, ""),
        (11, "stack_logreg_C0.3",          0.7902, 0.6234, 0.6666, 0.7762, 0.4992, ""),
        (12, "base_xgboost_cuda",          0.7588, 0.6631, 0.6646, 0.7599, 0.4934, ""),
        (13, "base_catboost",              0.7369, 0.7041, 0.6641, 0.7470, 0.4968, ""),
        (14, "stack_logreg_C0.3_hmm",      0.7973, 0.6096, 0.6619, 0.7795, 0.5030, ""),
        (15, "base_bilstm_attention_hmm",  0.7312, 0.6460, 0.6368, 0.7348, 0.4462, ""),
        (16, "base_bilstm_attention",      0.7201, 0.6589, 0.6315, 0.7273, 0.4402, ""),
    ]
    rows = []
    for rank, model, acc, bal, f1, wf1, kp, badge in rows_data:
        row_class = ""
        rank_html = f"<td class='center'>{rank}</td>"
        if badge == "win-acc":
            row_class = " style='background: linear-gradient(90deg, rgba(0,217,192,0.10), rgba(139,92,246,0.04));'"
            rank_html = f"<td class='center'><span style='color: var(--c-recovery); font-weight: 700;'>★ {rank}</span></td>"
        elif badge == "win-macro":
            row_class = " style='background: linear-gradient(90deg, rgba(245,158,11,0.10), rgba(0,217,192,0.04));'"
            rank_html = f"<td class='center'><span style='color: var(--c-fatigue); font-weight: 700;'>★ {rank}</span></td>"
        rows.append(
            f"<tr{row_class}>"
            f"{rank_html}"
            f"<td><span style='font-family: \"JetBrains Mono\", monospace; color: var(--text);'>{model}</span></td>"
            f"<td class='right'>{acc:.4f}</td>"
            f"<td class='right'>{bal:.4f}</td>"
            f"<td class='right'>{f1:.4f}</td>"
            f"<td class='right'>{wf1:.4f}</td>"
            f"<td class='right'>{kp:.4f}</td>"
            f"</tr>"
        )

    body = "".join(rows)
    headers = "".join(f"<th>{h}</th>" for h in [
        "Rank", "Configuration", "Accuracy", "Balanced acc", "Macro F1", "Weighted F1", "Cohen's κ"
    ])
    html = (
        "<div class='title'>Final leaderboard</div>"
        "<div class='subtitle'>16 configurations · sorted by macro F1 · ties broken by accuracy · 7-subject held-out test (6,020 epochs)</div>"
        f"<div class='card'><table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table>"
        "<div class='note'><span style='color: var(--c-recovery)'>★</span> avg_geo_hmm — best accuracy / weighted F1 / kappa &nbsp; · &nbsp; <span style='color: var(--c-fatigue)'>★</span> base_catboost_hmm — best balanced accuracy / macro F1</div>"
        "</div>"
    )
    render_page("section_16_leaderboard", html, width=1500)


# ---------------------------------------------------------------------------
# Section 19 — System architecture (split column: offline ↓ vs live ↓)
# ---------------------------------------------------------------------------

_ARCH_CSS = _CSS + """
.arch-grid {
  display: grid;
  grid-template-columns: 1fr 24px 1fr;
  gap: 16px;
  margin-top: 20px;
}
.arch-col h2 {
  font-family: "Fraunces", serif;
  font-weight: 500;
  font-size: 22px;
  letter-spacing: -0.01em;
  margin-bottom: 18px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
}
.arch-col.training h2 { color: var(--c-sleep); }
.arch-col.training h2::before { content: "↓ "; color: var(--c-sleep); }
.arch-col.live h2 { color: var(--c-recovery); }
.arch-col.live h2::before { content: "↻ "; color: var(--c-recovery); }
.arch-col h2 small {
  display: block;
  margin-top: 4px;
  font-family: "JetBrains Mono", monospace;
  font-size: 10px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--text-faint);
  font-weight: 500;
}

.arch-step {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 18px 20px;
  margin-bottom: 14px;
  position: relative;
}
.arch-step::before {
  content: ""; position: absolute;
  top: 0; left: 0; width: 3px; height: 100%;
  background: var(--accent);
  border-radius: 14px 0 0 14px;
}
.arch-col.training .arch-step { --accent: var(--c-sleep); }
.arch-col.live .arch-step     { --accent: var(--c-recovery); }

.arch-step .step-label {
  font-family: "JetBrains Mono", monospace;
  font-size: 10px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--text-faint);
  margin-bottom: 4px;
}
.arch-step .step-title {
  font-family: "Fraunces", serif;
  font-weight: 500;
  font-size: 18px;
  letter-spacing: -0.01em;
  color: var(--text);
  margin-bottom: 6px;
}
.arch-step .step-detail {
  font-family: "Inter", sans-serif;
  font-size: 12.5px;
  color: var(--text-muted);
  line-height: 1.55;
}
.arch-step .step-detail b {
  font-weight: 600;
  color: var(--text);
}

.arch-divider {
  display: flex; flex-direction: column; align-items: center;
  gap: 6px;
  padding-top: 60px;
}
.arch-divider span {
  font-family: "JetBrains Mono", monospace;
  font-size: 9px;
  letter-spacing: 0.2em;
  color: var(--text-faint);
  text-orientation: mixed;
  writing-mode: vertical-rl;
  text-transform: uppercase;
}

.arch-converge {
  margin-top: 20px;
  position: relative;
}
.arch-converge::before {
  content: "";
  position: absolute;
  top: -16px; left: 25%; right: 25%; height: 16px;
  border-left: 1px solid var(--border-hi);
  border-right: 1px solid var(--border-hi);
  border-top: 1px solid var(--border-hi);
  border-radius: 14px 14px 0 0;
}
.arch-app-card {
  background: linear-gradient(135deg, rgba(139,92,246,0.10), rgba(0,217,192,0.10));
  border: 1px solid var(--border-hi);
  border-radius: 18px;
  padding: 26px 32px;
  text-align: center;
}
.arch-app-card .app-label {
  font-family: "JetBrains Mono", monospace;
  font-size: 10px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--c-recovery);
  margin-bottom: 8px;
}
.arch-app-card .app-title {
  font-family: "Fraunces", serif;
  font-weight: 500;
  font-size: 28px;
  letter-spacing: -0.02em;
  margin-bottom: 8px;
  background: linear-gradient(180deg, #fff 0%, var(--c-recovery) 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}
.arch-app-card .app-detail {
  font-family: "Inter", sans-serif;
  font-size: 13px;
  color: var(--text-muted);
  max-width: 700px;
  margin: 0 auto;
  line-height: 1.55;
}
.arch-app-card .app-features {
  margin-top: 20px;
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 14px;
}
.arch-app-card .app-feature {
  padding: 6px 14px;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border);
  border-radius: 999px;
  font-family: "JetBrains Mono", monospace;
  font-size: 10.5px;
  letter-spacing: 0.1em;
  color: var(--text-muted);
}
.arch-app-card .app-feature b { color: var(--c-recovery); font-weight: 500; }
"""


def render_section_19():
    print("rendering section 19 architecture diagram...")
    body = """
    <div class='title'>System architecture</div>
    <div class='subtitle'>Offline training pipeline (left) converges with the live inference path (right) on a shared cached-probabilities surface that the application consumes.</div>

    <div class='arch-grid'>
      <div class='arch-col training'>
        <h2>Training<small>once, offline, on the research dataset</small></h2>

        <div class='arch-step'>
          <div class='step-label'>01 · Raw sensors</div>
          <div class='step-title'>Walch et al. PSG dataset</div>
          <div class='step-detail'>31 subjects · <b>26,773 labelled 30-second epochs</b>. Wrist motion (~50 Hz), heart rate (per few seconds), step counts, PSG sleep labels mapped to Wake/NREM/REM.</div>
        </div>

        <div class='arch-step'>
          <div class='step-label'>02 · Feature extraction</div>
          <div class='step-title'>540 engineered features per epoch</div>
          <div class='step-detail'>Time context, multi-window motion summaries, multi-window HR summaries, HRV-style proxies, step rollups — plus the key lever, <b>robust per-subject z-scoring</b> of HR and ENMO.</div>
        </div>

        <div class='arch-step'>
          <div class='step-label'>03 · Five base learners</div>
          <div class='step-title'>XGBoost · LightGBM · CatBoost · Tuned CatBoost · BiLSTM</div>
          <div class='step-detail'>Trained on a subject-level 19/5/7 split. Tuned CatBoost from a 25-trial Optuna sweep; BiLSTM with attention pooling and focal cross-entropy.</div>
        </div>

        <div class='arch-step'>
          <div class='step-label'>04 · Frozen probabilities</div>
          <div class='step-title'>Per-class predictions cached to disk</div>
          <div class='step-detail'>One probability tensor per learner, covering every subject in the dataset. <b>Loaded once at app boot</b> — the live path never re-runs the models.</div>
        </div>
      </div>

      <div class='arch-divider'>
        <span>shared surface</span>
      </div>

      <div class='arch-col live'>
        <h2>Live inference<small>per request, ~milliseconds</small></h2>

        <div class='arch-step'>
          <div class='step-label'>01 · Subject pick</div>
          <div class='step-title'>One of 31 PSG subjects or one Apple Health night</div>
          <div class='step-detail'>The same code path serves both sources. Apple Health uploads are stream-parsed in memory and grouped into nights at request time.</div>
        </div>

        <div class='arch-step'>
          <div class='step-label'>02 · Ensemble + smoothing</div>
          <div class='step-title'>Geometric mean → Viterbi-decoded HMM</div>
          <div class='step-detail'>Cached probabilities are sliced by subject, log-averaged across the five learners, then smoothed with a Hidden Markov Model fit from training-label transitions.</div>
        </div>

        <div class='arch-step'>
          <div class='step-label'>03 · Digital Twin scoring</div>
          <div class='step-title'>Six interpretable scores from named components</div>
          <div class='step-detail'>Sleep quality, sleep debt, recovery, stress, fatigue, energy — each a weighted sum of literature-grounded components. Real Apple HRV feeds the autonomic components when available.</div>
        </div>

        <div class='arch-step'>
          <div class='step-label'>04 · JSON</div>
          <div class='step-title'>Hypnogram + scores + raw signals + breakdowns</div>
          <div class='step-detail'>Returned in a single payload the application renders without re-fetching.</div>
        </div>
      </div>
    </div>

    <div class='arch-converge'>
      <div class='arch-app-card'>
        <div class='app-label'>Application</div>
        <div class='app-title'>Sleep Twin</div>
        <div class='app-detail'>FastAPI server with a custom single-page front-end on a dark aurora theme. Boots in roughly one second because it consumes the cached probabilities directly.</div>
        <div class='app-features'>
          <div class='app-feature'><b>270°</b> segmented dial</div>
          <div class='app-feature'><b>Bento grid</b> of score cards</div>
          <div class='app-feature'><b>Constellation</b> hypnogram</div>
          <div class='app-feature'><b>Flip overlay</b> for breakdowns</div>
          <div class='app-feature'><b>Live what-if</b> simulator</div>
          <div class='app-feature'><b>Apple Health</b> import + audit</div>
          <div class='app-feature'><b>Methodology</b> drawer</div>
        </div>
      </div>
    </div>
    """
    render_page("section_19_architecture", body, width=1600, css=_ARCH_CSS)


# ---------------------------------------------------------------------------
# Section 21 — Sprint plan (one image per sprint)
# ---------------------------------------------------------------------------

_SPRINT_CSS = _CSS + """
.sprint-meta {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin-bottom: 22px;
}
.sprint-meta .meta-box {
  padding: 14px 18px;
  background: rgba(255,255,255,0.02);
  border-left: 2px solid var(--c-recovery);
  border-radius: 0 10px 10px 0;
}
.sprint-meta .meta-box .meta-label {
  font: 500 10px "JetBrains Mono", monospace;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--text-faint);
  margin-bottom: 6px;
}
.sprint-meta .meta-box .meta-value {
  font-family: "Inter", sans-serif;
  font-size: 13.5px;
  line-height: 1.55;
  color: var(--text-muted);
}
.sprint-meta .meta-box .meta-value b { color: var(--text); }
.us-list {
  margin: 0; padding: 0; list-style: none;
}
.us-list li {
  padding: 8px 0;
  font-size: 13px;
  color: var(--text-muted);
  border-bottom: 1px dashed var(--border);
}
.us-list li:last-child { border-bottom: none; }
.us-list li b { color: var(--c-recovery); margin-right: 8px; font-family: "JetBrains Mono", monospace; font-size: 11px; }
"""


def _sprint_image(name: str, week: str, title: str, goal: str, user_stories: list[tuple[str, str]],
                  tasks: list[tuple[str, str, str]], deliverables: str):
    us_html = "".join(f"<li><b>{uid}</b> {desc}</li>" for uid, desc in user_stories)
    task_rows = "".join(
        f"<tr><td style='color: var(--c-recovery); font-family: \"JetBrains Mono\", monospace; font-size: 11px;'>{tid}</td>"
        f"<td>{desc}</td>"
        f"<td style='color: var(--text-muted); font-family: \"Inter\", sans-serif; font-size: 12px;'>{dod}</td></tr>"
        for tid, desc, dod in tasks
    )
    body = f"""
    <div style='font-family: "JetBrains Mono", monospace; font-size: 11px; letter-spacing: 0.2em; text-transform: uppercase; color: var(--c-recovery); margin-bottom: 6px;'>{week}</div>
    <div class='title'>{title}</div>
    <div class='subtitle'>Sprint goal · {goal}</div>

    <div class='card' style='margin-bottom: 18px;'>
      <h3 style='font-family: "Fraunces", serif; font-weight: 500; font-size: 17px; margin-bottom: 14px;'>User stories</h3>
      <ul class='us-list'>{us_html}</ul>
    </div>

    <div class='card'>
      <h3 style='font-family: "Fraunces", serif; font-weight: 500; font-size: 17px; margin-bottom: 14px;'>Tasks</h3>
      <table>
        <thead><tr><th style='width: 90px;'>ID</th><th>Task</th><th style='width: 360px;'>Definition of done</th></tr></thead>
        <tbody>{task_rows}</tbody>
      </table>
      <div class='note'><b style='color: var(--text); font-style: normal;'>Sprint review deliverables:</b> {deliverables}</div>
    </div>
    """
    render_page(name, body, width=1600, css=_SPRINT_CSS)


def render_section_21():
    print("rendering section 21 sprint plan...")

    _sprint_image(
        "section_21_sprint_5",
        week="Week 5 · Sprint 5",
        title="Re-baseline ML stack",
        goal="push past the 76 % accuracy plateau by rebuilding features and base learners",
        user_stories=[
            ("US-R01", "As a researcher I want subject-normalized HR and motion features so the classifier is robust to between-subject baseline differences."),
            ("US-R02", "As a researcher I want a richer base-learner stack (XGBoost + LightGBM + CatBoost + Optuna-tuned CatBoost + BiLSTM-with-attention) so I have diverse signal sources for ensembling."),
            ("US-R03", "As a researcher I want an Optuna sweep on CatBoost so the strongest single model is tuned, not default."),
        ],
        tasks=[
            ("T-R01-1", "Robust per-subject HR z-scoring (median / IQR)", "z-score features in the cache"),
            ("T-R01-2", "Subject-normalized ENMO", "normalized ENMO features in the cache"),
            ("T-R01-3", "HRV-style proxies (range, succ-diff RMS, CV) on three windows", "9 new HRV columns per epoch"),
            ("T-R01-4", "Widen HR windows to include 1800s and 3600s", "feature count rises to 540"),
            ("T-R02-1", "Rebuild XGBoost trainer with deeper trees and more estimators", "tuned trainer + balanced sample weights"),
            ("T-R02-2", "Add LightGBM trainer with built-in class weighting", "second booster shipped"),
            ("T-R02-3", "Add CatBoost trainer with auto-balanced class weights", "standout single learner"),
            ("T-R02-4", "BiLSTM with attention pooling + focal CE + OneCycleLR", "checkpoint + probabilities"),
            ("T-R02-5", "Isolate BiLSTM training in a sidecar conda env to bypass Windows c10.dll conflict", "documented in README"),
            ("T-R03-1", "Optuna TPE sweep on CatBoost optimizing validation macro F1", "tuned CatBoost + best parameters"),
            ("T-R03-2", "End-to-end orchestrator chains features → 4 boosters → Optuna → BiLSTM → merge", "one-command rebuild from raw data"),
        ],
        deliverables="540-feature cache, 5 trained base learners, leaderboard CSV.",
    )

    _sprint_image(
        "section_21_sprint_6",
        week="Week 6 · Sprint 6",
        title="Ensemble, smoothing, scoring rewrite",
        goal="combine base learners, add temporal post-processing, rewrite the Digital Twin around named components",
        user_stories=[
            ("US-R04", "As a researcher I want a probability-stacking layer (arithmetic / geometric / logistic-regression) so I can pick the best combiner."),
            ("US-R05", "As a researcher I want HMM/Viterbi smoothing fit from training-label transitions so single-epoch flicker noise is removed."),
            ("US-R06", "As a system designer I want every Digital Twin score to be a weighted sum of named, individually-bounded components so the scoring is interpretable end-to-end."),
            ("US-R07", "As an analyst I want a methodology document with literature references per component so the demo question can be answered precisely."),
        ],
        tasks=[
            ("T-R04-1", "Geometric-mean and arithmetic-mean ensemble heads", "both heads scored on the leaderboard"),
            ("T-R04-2", "Logistic-regression stacking with C ∈ {0.3, 1.0, 4.0} plus two MLPs", "best meta-learner persisted, full ablation reported"),
            ("T-R05-1", "Transition matrix with Laplace smoothing from training labels", "matrix persisted alongside the ensemble"),
            ("T-R05-2", "Per-subject Viterbi decoder", "covers every test subject"),
            ("T-R05-3", "Apply HMM to every candidate so the leaderboard pairs ±HMM rows", "16-row leaderboard"),
            ("T-R06-1", "Expand sleep architecture summary with onset latency, REM latency, WASO, cycle count", "13 fields total"),
            ("T-R06-2", "Universal physiology summary working for both PSG and Apple paths", "single abstraction across data sources"),
            ("T-R06-3", "Refactor scoring into 6 helpers returning (total, components)", "per-score breakdown surfaced"),
            ("T-R06-4", "Module-level target-range constants with named source references", "every threshold traces to a citation"),
            ("T-R06-5", "Add sleep_debt as the sixth score", "six-score state vector"),
            ("T-R07-1", "Author methodology document with per-component tables and 9 literature references", "committed in the repo"),
            ("T-R07-2", "Author full training writeup with ablations and deferred work", "committed in the repo"),
        ],
        deliverables="80 %-accuracy ensemble on the leaderboard, fully decomposed scoring layer, two report documents.",
    )

    _sprint_image(
        "section_21_sprint_7",
        week="Week 7 · Sprint 7",
        title="Custom single-page application",
        goal="replace the Streamlit dashboard with a designed front-end that surfaces the per-score breakdown visually",
        user_stories=[
            ("US-R08", "As a project owner I want a backend service that exposes per-subject analysis as JSON so any frontend can consume it."),
            ("US-R09", "As a user I want a modern single-page UI (dark theme, bento grid, score-card breakdown) so the Digital Twin presents like a real product."),
            ("US-R10", "As a user I want a live what-if simulator so I can see how losing sleep or adding activity moves my scores."),
        ],
        tasks=[
            ("T-R08-1", "Analysis service that loads cached probabilities + HMM at boot", "~1 s app boot time"),
            ("T-R08-2", "FastAPI endpoints for subjects, per-subject analysis, what-if, leaderboard", "JSON contract matches the documented schema"),
            ("T-R08-3", "Uvicorn launcher script", "one-command boot"),
            ("T-R09-1", "Aurora dark theme (CSS custom properties, gradient blobs, glass surfaces)", "design system applied"),
            ("T-R09-2", "270° segmented headline dial in SVG with serif number readout", "dial renders the recovery score on load"),
            ("T-R09-3", "Bento grid of six color-coded score cards", "cards render scores from analysis JSON"),
            ("T-R09-4", "Constellation hypnogram with HR / motion overlays", "renders any test subject"),
            ("T-R09-5", "Flip overlay showing per-score component breakdown with bar segments + literature notes", "tap any card → overlay"),
            ("T-R09-6", "Methodology drawer with score-colored chips and a callout box", "keyboard M opens it"),
            ("T-R10-1", "Debounced live what-if dials wired to the simulator endpoint", "all 6 delta tiles refresh within 120 ms"),
            ("T-R10-2", "Fully responsive layout down to mobile width", "tested at 1440 / 1100 / 640 px breakpoints"),
        ],
        deliverables="working single-page app at the local host, zero npm dependencies, no build step.",
    )

    _sprint_image(
        "section_21_sprint_8",
        week="Week 8 · Sprint 8",
        title="Apple Health re-plumbing + polish",
        goal="match every Apple Health data layer the demo had and additionally feed real Apple HRV into the scoring",
        user_stories=[
            ("US-R11", "As a user I want to drop my Apple Health export into the new app and see Digital Twin scores for my own nights, with the same data richness the demo had."),
            ("US-R12", "As a user I want the upload pipeline to handle multi-GB exports without crashing my browser or the server."),
            ("US-R13", "As a user I want the UI to be honest about what is and isn't available for my own data (no broken toggles, no pretending there's PSG truth)."),
            ("US-R14", "As an analyst I want real Apple HRV to feed the recovery and stress scores when present — not just be shown in a table."),
        ],
        tasks=[
            ("T-R11-1", "Streaming Apple Health parser (iterparse + element clearing)", "all 7 demo record types returned"),
            ("T-R11-2", "Night grouping (2-hour gap = new night) and Wake/NREM/REM stage mapping", "every detailed night recovered"),
            ("T-R11-3", "Brief-wake smoothing with configurable threshold", "smoothed sequence + change count"),
            ("T-R11-4", "Per-session wearable summary (mean sleep HR, HRV, resting HR baseline + delta, blended steps)", "summary returned from analysis"),
            ("T-R11-5", "7-day history summary (avg daily steps, energy, exercise, prior sleep stats)", "summary returned from analysis"),
            ("T-R11-6", "Upload endpoints with in-memory storage keyed by uuid", "uploaded nights survive the process lifetime"),
            ("T-R11-7", "What-if simulator routes upload subject IDs to the Apple-Health helper", "uploaded subjects support the simulator"),
            ("T-R12-1", "Filter to last-30-days during parse to bound memory", "multi-GB exports succeed"),
            ("T-R12-2", "Cap upload size at 500 MB with a 413 response", "over-cap returns a clear error"),
            ("T-R13-1", "Import button + drag-drop modal with three-step instructions", "modal opens, accepts files, shows status"),
            ("T-R13-2", "Subject picker groups uploads first, PSG subjects second", "optgroup layout"),
            ("T-R13-3", "Apple split-pill so source is visible at a glance", "amber pill renders for uploads"),
            ("T-R13-4", "Hide PSG-truth and motion toggles when an Apple analysis is loaded", "toggles disappear automatically"),
            ("T-R13-5", "Fix native dropdown contrast on Windows (dark color-scheme + explicit option / optgroup colors)", "dropdown popup renders dark"),
            ("T-R13-6", "Filter the night picker to detailed-stage nights by default with an InBed-only toggle", "picker updates when the toggle flips"),
            ("T-R13-7", "Dedicated Apple Health audit card showing every parsed value, history context, and record counts", "card renders below the bento grid only for Apple sources"),
            ("T-R14-1", "Optional HRV-SDNN field on physiology summary with NaN default", "PSG path unchanged"),
            ("T-R14-2", "Recovery and stress prefer real SDNN when present", "component labels switch to hrv_sdnn / low_hrv_sdnn"),
            ("T-R14-3", "Wire night analysis to populate the SDNN field from the wearable summary", "smoke-test confirms the label change"),
        ],
        deliverables="end-to-end Apple Health import demo with real personal data, real HRV feeding the recovery and stress scores, polished UI.",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("rendering report assets via headless Chromium...\n")
    render_section_8()
    render_section_16()
    render_section_19()
    render_section_21()
    print("\nall report assets generated.")
    print(f"see {OUTPUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
